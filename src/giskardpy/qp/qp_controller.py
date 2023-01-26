import datetime
import os
from collections import OrderedDict, defaultdict
from copy import deepcopy
from time import time
from typing import List, Dict, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from giskardpy import casadi_wrapper as w, identifier
from giskardpy.configs.data_types import SupportedQPSolver
from giskardpy.exceptions import OutOfJointLimitsException, \
    HardConstraintsViolatedException, QPSolverException, InfeasibleException
from giskardpy.god_map import GodMap
from giskardpy.model.world import WorldTree
from giskardpy.my_types import derivative_joint_map, Derivatives
from giskardpy.qp.constraint import VelocityConstraint, Constraint
from giskardpy.qp.free_variable import FreeVariable
from giskardpy.qp.qp_solver import QPSolver
from giskardpy.utils import logging
from giskardpy.utils.time_collector import TimeCollector
from giskardpy.utils.utils import memoize, create_path, suppress_stdout


def save_pandas(dfs, names, path):
    folder_name = f'{path}/pandas_{datetime.datetime.now().strftime("%Yy-%mm-%dd--%Hh-%Mm-%Ss")}/'
    create_path(folder_name)
    for df, name in zip(dfs, names):
        csv_string = 'name\n'
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            if df.shape[1] > 1:
                for column_name, column in df.T.items():
                    csv_string += column.add_prefix(column_name + '||').to_csv(float_format='%.4f')
            else:
                csv_string += df.to_csv(float_format='%.4f')
        file_name2 = f'{folder_name}{name}.csv'
        with open(file_name2, 'w') as f:
            f.write(csv_string)


class Parent(object):
    time_collector: TimeCollector

    def __init__(self, sample_period, prediction_horizon, order, time_collector=None):
        self.time_collector = time_collector
        self.prediction_horizon = prediction_horizon
        self.sample_period = sample_period
        self.order = order

    def _sorter(self, *args):
        """
        Sorts every arg dict individually and then appends all of them.
        :arg args: a bunch of dicts
        :return: list
        """
        result = []
        result_names = []
        for arg in args:
            result.extend(self.__helper(arg))
            result_names.extend(self.__helper_names(arg))
        return result, result_names

    def __helper(self, param):
        return [x for _, x in sorted(param.items())]

    def __helper_names(self, param):
        return [x for x, _ in sorted(param.items())]

    def blow_up(self, d, num_of_copies, weight_inc_f=None):
        result = {}
        for t in range(num_of_copies):
            for name, value in d.items():
                if weight_inc_f is not None:
                    result['t{:03d}/{}'.format(t, name)] = weight_inc_f(value, t)
                else:
                    result['t{:03d}/{}'.format(t, name)] = value
        return result


class H(Parent):
    def __init__(self, free_variables, constraints, velocity_constraints, sample_period, prediction_horizon, order,
                 default_limits=False):
        super().__init__(sample_period, prediction_horizon, order)
        self.free_variables = free_variables
        self.constraints = constraints  # type: list[Constraint]
        self.velocity_constraints = velocity_constraints  # type: list[velocity_constraints]
        self.height = 0
        self._compute_height()
        self.evaluted = True

    def _compute_height(self):
        self.height = self.number_of_free_variables_with_horizon()
        self.height += self.number_of_constraint_vel_variables()
        self.height += self.number_of_contraint_error_variables()

    @property
    def width(self):
        return self.height

    def number_of_free_variables_with_horizon(self):
        h = 0
        for v in self.free_variables:
            h += (min(v.order, self.order) - 1) * self.prediction_horizon
        return h

    def number_of_constraint_vel_variables(self):
        h = 0
        for c in self.velocity_constraints:
            h += c.control_horizon
        return h

    def number_of_contraint_error_variables(self):
        return len(self.constraints)

    @profile
    def weights(self):
        weights = defaultdict(dict)  # maps order to joints
        for t in range(self.prediction_horizon):
            for v in self.free_variables:  # type: FreeVariable
                for o in range(1, min(v.order, self.order)):
                    o = Derivatives(o)
                    weights[o][f't{t:03}/{v.position_name}/{o}'] = v.normalized_weight(t, o,
                                                                                       self.prediction_horizon,
                                                                                       evaluated=self.evaluted)
        slack_weights = {}
        for t in range(self.prediction_horizon):
            for c in self.velocity_constraints:  # type: VelocityConstraint
                if t < c.control_horizon:
                    slack_weights[f't{t:03}/{c.name}'] = c.normalized_weight(t)

        error_slack_weights = {f'{c.name}/error': c.normalized_weight(self.prediction_horizon) for c in
                               self.constraints}

        params = []
        for o, w in sorted(weights.items()):
            params.append(w)
        params.append(slack_weights)
        params.append(error_slack_weights)
        return self._sorter(*params)[0]


class B(Parent):
    def __init__(self, free_variables, constraints, velocity_constraints, sample_period, prediction_horizon, order,
                 default_limits=False):
        super().__init__(sample_period, prediction_horizon, order)
        self.free_variables = free_variables  # type: list[FreeVariable]
        self.constraints = constraints  # type: list[Constraint]
        self.velocity_constraints = velocity_constraints  # type: list[VelocityConstraint]
        self.no_limits = 1e4
        self.evaluated = True
        self.default_limits = default_limits

    def get_lower_slack_limits(self):
        result = {}
        for t in range(self.prediction_horizon):
            for c in self.velocity_constraints:
                if t < c.control_horizon:
                    result[f't{t:03}/{c.name}'] = c.lower_slack_limit[t]
        return result

    def get_upper_slack_limits(self):
        result = {}
        for t in range(self.prediction_horizon):
            for c in self.velocity_constraints:
                if t < c.control_horizon:
                    result[f't{t:03}/{c.name}'] = c.upper_slack_limit[t]
        return result

    def get_lower_error_slack_limits(self):
        return {f'{c.name}/error': c.lower_slack_limit for c in self.constraints}

    def get_upper_error_slack_limits(self):
        return {f'{c.name}/error': c.upper_slack_limit for c in self.constraints}

    def __call__(self):
        lb = defaultdict(dict)
        ub = defaultdict(dict)
        for t in range(self.prediction_horizon):
            for v in self.free_variables:  # type: FreeVariable
                for o in range(1, min(v.order, self.order)):  # start with velocity
                    o = Derivatives(o)
                    if t == self.prediction_horizon - 1 \
                            and o < min(v.order, self.order) - 1 \
                            and self.prediction_horizon > 2:  # and False:
                        lb[o][f't{t:03}/{v.position_name}/{o}'] = 0
                        ub[o][f't{t:03}/{v.position_name}/{o}'] = 0
                    else:
                        lb[o][f't{t:03}/{v.position_name}/{o}'] = v.get_lower_limit(o, evaluated=self.evaluated)
                        ub[o][f't{t:03}/{v.position_name}/{o}'] = v.get_upper_limit(o, evaluated=self.evaluated)
        lb_params = []
        for o, x in sorted(lb.items()):
            lb_params.append(x)
        lb_params.append(self.get_lower_slack_limits())
        lb_params.append(self.get_lower_error_slack_limits())

        ub_params = []
        for o, x in sorted(ub.items()):
            ub_params.append(x)
        ub_params.append(self.get_upper_slack_limits())
        ub_params.append(self.get_upper_error_slack_limits())

        lb, self.names = self._sorter(*lb_params)
        return lb, self._sorter(*ub_params)[0]


class BA(Parent):
    def __init__(self, free_variables, constraints, velocity_constraints, sample_period, prediction_horizon, order,
                 default_limits=False):
        super().__init__(sample_period, prediction_horizon, order)
        self.free_variables = free_variables
        self.constraints = constraints
        self.velocity_constraints = velocity_constraints
        self.round_to = 5
        self.round_to2 = 10
        self.default_limits = default_limits
        self.evaluated = True

    def get_lower_constraint_velocities(self):
        result = {}
        for t in range(self.prediction_horizon):
            for c in self.velocity_constraints:
                if t < c.control_horizon:
                    result[f't{t:03}/{c.name}'] = w.limit(c.lower_velocity_limit[t] * self.sample_period,
                                                          -c.velocity_limit * self.sample_period,
                                                          c.velocity_limit * self.sample_period)
        return result

    def get_upper_constraint_velocities(self):
        result = {}
        for t in range(self.prediction_horizon):
            for c in self.velocity_constraints:
                if t < c.control_horizon:
                    result[f't{t:03}/{c.name}'] = w.limit(c.upper_velocity_limit[t] * self.sample_period,
                                                          -c.velocity_limit * self.sample_period,
                                                          c.velocity_limit * self.sample_period)
        return result

    @memoize
    def get_lower_constraint_error(self):
        return {f'{c.name}/e': w.limit(c.lower_error,
                                       -c.velocity_limit * self.sample_period * c.control_horizon,
                                       c.velocity_limit * self.sample_period * c.control_horizon)
                for c in self.constraints}

    @memoize
    def get_upper_constraint_error(self):
        return {f'{c.name}/e': w.limit(c.upper_error,
                                       -c.velocity_limit * self.sample_period * c.control_horizon,
                                       c.velocity_limit * self.sample_period * c.control_horizon)
                for c in self.constraints}

    def __call__(self) -> tuple:
        lb = {}
        ub = {}
        # position limits
        for t in range(self.prediction_horizon):
            for v in self.free_variables:  # type: FreeVariable
                if v.has_position_limits():
                    normal_lower_bound = w.round_up(
                        v.get_lower_limit(Derivatives.position,
                                          False, evaluated=self.evaluated) - v.get_symbol(Derivatives.position),
                        self.round_to2)
                    normal_upper_bound = w.round_down(
                        v.get_upper_limit(Derivatives.position,
                                          False, evaluated=self.evaluated) - v.get_symbol(Derivatives.position),
                        self.round_to2)
                    if self.default_limits:
                        if self.order >= 4:
                            lower_vel = w.min(v.get_upper_limit(derivative=Derivatives.velocity,
                                                                default=False,
                                                                evaluated=True) * self.sample_period,
                                              v.get_upper_limit(derivative=Derivatives.jerk,
                                                                default=False,
                                                                evaluated=self.evaluated) * self.sample_period ** 3)
                            upper_vel = w.max(v.get_lower_limit(derivative=Derivatives.velocity,
                                                                default=False,
                                                                evaluated=True) * self.sample_period,
                                              v.get_lower_limit(derivative=Derivatives.jerk,
                                                                default=False,
                                                                evaluated=self.evaluated) * self.sample_period ** 3)
                        else:
                            lower_vel = w.min(v.get_upper_limit(derivative=Derivatives.velocity,
                                                                default=False,
                                                                evaluated=True) * self.sample_period,
                                              v.get_upper_limit(derivative=Derivatives.acceleration,
                                                                default=False,
                                                                evaluated=self.evaluated) * self.sample_period ** 2)
                            upper_vel = w.max(v.get_lower_limit(derivative=Derivatives.velocity,
                                                                default=False,
                                                                evaluated=True) * self.sample_period,
                                              v.get_lower_limit(derivative=Derivatives.acceleration,
                                                                default=False,
                                                                evaluated=self.evaluated) * self.sample_period ** 2)
                        lower_bound = w.if_greater(normal_lower_bound, 0,
                                                   if_result=lower_vel,
                                                   else_result=normal_lower_bound)
                        lb[f't{t:03d}/{v.position_name}/p_limit'] = lower_bound

                        upper_bound = w.if_less(normal_upper_bound, 0,
                                                if_result=upper_vel,
                                                else_result=normal_upper_bound)
                        ub[f't{t:03d}/{v.position_name}/p_limit'] = upper_bound
                    else:
                        lb[f't{t:03d}/{v.position_name}/p_limit'] = normal_lower_bound
                        ub[f't{t:03d}/{v.position_name}/p_limit'] = normal_upper_bound

        l_last_stuff = defaultdict(dict)
        u_last_stuff = defaultdict(dict)
        for v in self.free_variables:
            for o in range(1, min(v.order, self.order) - 1):
                o = Derivatives(o)
                l_last_stuff[o][f'{v.position_name}/last_{o}'] = w.round_down(v.get_symbol(o), self.round_to)
                u_last_stuff[o][f'{v.position_name}/last_{o}'] = w.round_up(v.get_symbol(o), self.round_to)

        derivative_link = defaultdict(dict)
        for t in range(self.prediction_horizon - 1):
            for v in self.free_variables:
                for o in range(1, min(v.order, self.order) - 1):
                    derivative_link[o][f't{t:03}/{o}/{v.position_name}/link'] = 0

        lb_params = [lb]
        ub_params = [ub]
        for o in range(1, self.order - 1):
            lb_params.append(l_last_stuff[o])
            lb_params.append(derivative_link[o])
            ub_params.append(u_last_stuff[o])
            ub_params.append(derivative_link[o])
        lb_params.append(self.get_lower_constraint_velocities())
        lb_params.append(self.get_lower_constraint_error())
        ub_params.append(self.get_upper_constraint_velocities())
        ub_params.append(self.get_upper_constraint_error())

        lbA, self.names = self._sorter(*lb_params)
        return lbA, self._sorter(*ub_params)[0]


class A(Parent):
    def __init__(self, free_variables, constraints, velocity_constraints, sample_period, prediction_horizon, order,
                 time_collector, default_limits=False):
        super().__init__(sample_period, prediction_horizon, order, time_collector)
        self.free_variables = free_variables  # type: list[FreeVariable]
        self.constraints = constraints  # type: list[Constraint]
        self.velocity_constraints = velocity_constraints  # type: list[VelocityConstraint]
        self.joints = {}
        self.height = 0
        self._compute_height()
        self.width = 0
        self._compute_width()
        self.default_limits = default_limits

    def _compute_height(self):
        # rows for position limits of non continuous joints
        self.height = self.prediction_horizon * (self.num_position_limits())
        # rows for linking vel/acc/jerk
        self.height += self.number_of_joints * self.prediction_horizon * (self.order - 2)
        # rows for velocity constraints
        for i, c in enumerate(self.velocity_constraints):
            self.height += c.control_horizon
        # row for constraint error
        self.height += len(self.constraints)

    def _compute_width(self):
        # columns for joint vel/acc/jerk symbols
        self.width = self.number_of_joints * self.prediction_horizon * (self.order - 1)
        # columns for velocity constraints
        for i, c in enumerate(self.velocity_constraints):
            self.width += c.control_horizon
        # slack variable for constraint error
        self.width += len(self.constraints)
        # constraints for getting out of hard limits
        # if self.default_limits:
        #     self.width += self.num_position_limits()

    @property
    def number_of_joints(self):
        return len(self.free_variables)

    @memoize
    def num_position_limits(self):
        return self.number_of_joints - self.num_of_continuous_joints()

    @memoize
    def num_of_continuous_joints(self):
        return len([v for v in self.free_variables if not v.has_position_limits()])

    def get_constraint_expressions(self):
        return self._sorter({c.name: c.expression for c in self.constraints})[0]

    def get_velocity_constraint_expressions(self):
        return self._sorter({c.name: c.expression for c in self.velocity_constraints})[0]

    def get_free_variable_symbols(self, order):
        return self._sorter({v.position_name: v.get_symbol(order) for v in self.free_variables})[0]

    @profile
    def construct_A(self):
        #         |   t1   |   tn   |   t1   |   tn   |   t1   |   tn   |   t1   |   tn   |
        #         |v1 v2 vn|v1 v2 vn|a1 a2 an|a1 a2 an|j1 j2 jn|j1 j2 jn|s1 s2 sn|s1 s2 sn|
        #         |-----------------------------------------------------------------------|
        #         |sp      |        |        |        |        |        |        |        |
        #         |   sp   |        |        |        |        |        |        |        |
        #         |      sp|        |        |        |        |        |        |        |
        #         |-----------------------------------------------------------------------|
        #         |sp      |sp      |        |        |        |        |        |        |
        #         |   sp   |   sp   |        |        |        |        |        |        |
        #         |      sp|      sp|        |        |        |        |        |        |
        #         |=======================================================================|
        #         | 1      |        |-sp     |        |        |        |        |        |
        #         |    1   |        |   -sp  |        |        |        |        |        |
        #         |       1|        |     -sp|        |        |        |        |        |
        #         |-----------------------------------------------------------------------|
        #         |-1      | 1      |        |-sp     |        |        |        |        |
        #         |   -1   |    1   |        |   -sp  |        |        |        |        |
        #         |      -1|       1|        |     -sp|        |        |        |        |
        #         |=======================================================================|
        #         |        |        | 1      |        |-sp     |        |-sp     |        |
        #         |        |        |    1   |        |   -sp  |        |   -sp  |        |
        #         |        |        |       1|        |     -sp|        |     -sp|        |
        #         |-----------------------------------------------------------------------|
        #         |        |        |-1      | 1      |        |-sp     |        |-sp     |
        #         |        |        |   -1   |    1   |        |   -sp  |        |   -sp  |
        #         |        |        |      -1|       1|        |     -sp|        |     -sp|
        #         |=======================================================================|
        #         |  J*sp  |        |        |        |        |        |   sp   |        |
        #         |-----------------------------------------------------------------------|
        #         |        |  J*sp  |        |        |        |        |        |   sp   |
        #         |-----------------------------------------------------------------------|
        #         |  J*sp  |  J*sp  |        |        |        |        | sp*ph  | sp*ph  |
        #         |-----------------------------------------------------------------------|

        #         |   t1   |   t2   |   t3   |   t3   |
        #         |v1 v2 vn|v1 v2 vn|v1 v2 vn|v1 v2 vn|
        #         |-----------------------------------|
        #         |sp      |        |        |        |
        #         |   sp   |        |        |        |
        #         |      sp|        |        |        |
        #         |sp      |sp      |        |        |
        #         |   sp   |   sp   |        |        |
        #         |      sp|      sp|        |        |
        #         |sp      |sp      |sp      |        |
        #         |   sp   |   sp   |   sp   |        |
        #         |      sp|      sp|      sp|        |
        #         |sp      |sp      |sp      |sp      |
        #         |   sp   |   sp   |   sp   |   sp   |
        #         |      sp|      sp|      sp|      sp|
        #         |===================================|
        number_of_joints = self.number_of_joints
        A_soft = w.zeros(
            self.prediction_horizon * number_of_joints +  # joint position constraints
            number_of_joints * self.prediction_horizon * (self.order - 2) +  # links
            len(self.velocity_constraints) * (self.prediction_horizon) +  # velocity constraints
            len(self.constraints),  # constraints
            # (self.num_position_limits() if self.default_limits else 0) +  # weights for position limits
            number_of_joints * self.prediction_horizon * (self.order - 1) +
            len(self.velocity_constraints) * self.prediction_horizon + len(self.constraints)
        )
        t = time()
        J_vel = []
        J_err = []
        for order in range(self.order):
            J_vel.append(w.jacobian(expressions=w.Expression(self.get_velocity_constraint_expressions()),
                                    symbols=self.get_free_variable_symbols(order),
                                    order=1) * self.sample_period)
            J_err.append(w.jacobian(expressions=w.Expression(self.get_constraint_expressions()),
                                    symbols=self.get_free_variable_symbols(order),
                                    order=1) * self.sample_period)
        jac_time = time() - t
        logging.loginfo('computed Jacobian in {:.5f}s'.format(jac_time))
        # Jd = w.jacobian(w.Matrix(soft_expressions), controlled_joints, order=2)
        # logging.loginfo('computed Jacobian dot in {:.5f}s'.format(time() - t))
        self.time_collector.jacobians.append(jac_time)

        # position limits
        vertical_offset = number_of_joints * self.prediction_horizon
        for p in range(1, self.prediction_horizon + 1):
            matrix_size = number_of_joints * p
            I = w.eye(matrix_size) * self.sample_period
            start = vertical_offset - matrix_size
            A_soft[start:vertical_offset, :matrix_size] += I

        # derivative links
        block_size = number_of_joints * (self.order - 2) * self.prediction_horizon
        I = w.eye(block_size)
        A_soft[vertical_offset:vertical_offset + block_size, :block_size] += I
        h_offset = number_of_joints * self.prediction_horizon
        A_soft[vertical_offset:vertical_offset + block_size, h_offset:h_offset + block_size] += -I * self.sample_period

        I_height = number_of_joints * (self.prediction_horizon - 1)
        I = -w.eye(I_height)
        offset_v = vertical_offset
        offset_h = 0
        for o in range(self.order - 2):
            offset_v += number_of_joints
            A_soft[offset_v:offset_v + I_height, offset_h:offset_h + I_height] += I
            offset_v += I_height
            offset_h += self.prediction_horizon * number_of_joints
        vertical_offset = vertical_offset + block_size

        # constraints
        # TODO i don't need vel checks for the last 2 entries because the have to be zero with current B's
        # velocity limits
        next_vertical_offset = vertical_offset
        for order in range(self.order - 1):
            J_vel_tmp = J_vel[order]
            J_vel_limit_block = w.kron(w.eye(self.prediction_horizon), J_vel_tmp)
            horizontal_offset = J_vel_limit_block.shape[1]
            if order == 0:
                next_vertical_offset = vertical_offset + J_vel_limit_block.shape[0]
            A_soft[vertical_offset:next_vertical_offset,
            horizontal_offset * order:horizontal_offset * (order + 1)] = J_vel_limit_block
        # velocity constraint slack
        I = w.eye(J_vel_limit_block.shape[0]) * self.sample_period
        if J_err[0].shape[0] > 0:
            A_soft[vertical_offset:next_vertical_offset, -I.shape[1] - J_err[0].shape[0]:-J_err[0].shape[0]] = I
        else:
            A_soft[vertical_offset:next_vertical_offset, -I.shape[1]:] = I
        # delete rows if control horizon of constraint shorter than prediction horizon
        rows_to_delete = []
        for t in range(self.prediction_horizon):
            for i, c in enumerate(self.velocity_constraints):
                index = vertical_offset + i + (t * len(self.velocity_constraints))
                if t + 1 > c.control_horizon:
                    rows_to_delete.append(index)

        # delete columns where control horizon is shorter than prediction horizon
        columns_to_delete = []
        horizontal_offset = A_soft.shape[1] - I.shape[1] - J_err[0].shape[0]
        for t in range(self.prediction_horizon):
            for i, c in enumerate(self.velocity_constraints):
                index = horizontal_offset + (t * len(self.velocity_constraints)) + i
                if t + 1 > c.control_horizon:
                    columns_to_delete.append(index)

        # J stack for total error
        if len(self.constraints) > 0:
            for order in range(self.order - 1):
                J_hstack = w.hstack([J_err[order] for _ in range(self.prediction_horizon)])
                if order == 0:
                    vertical_offset = next_vertical_offset
                    next_vertical_offset = vertical_offset + J_hstack.shape[0]
                # set jacobian entry to 0 if control horizon shorter than prediction horizon
                for i, c in enumerate(self.constraints):
                    # offset = vertical_offset + i
                    J_hstack[i, c.control_horizon * len(self.free_variables):] = 0
                horizontal_offset = J_hstack.shape[1]
                A_soft[vertical_offset:next_vertical_offset,
                horizontal_offset * (order):horizontal_offset * (order + 1)] = J_hstack

                # sum of vel slack for total error
                # I = w.kron(w.Matrix([[1 for _ in range(self.prediction_horizon)]]),
                #            w.eye(J_hstack.shape[0])) * self.sample_period
                # A_soft[vertical_offset:next_vertical_offset, -I.shape[1]-len(self.constraints):-len(self.constraints)] = I
            # extra slack variable for total error
            I = w.eye(J_hstack.shape[0]) * self.sample_period * self.prediction_horizon
            A_soft[vertical_offset:next_vertical_offset, -I.shape[1]:] = I

        # delete rows with position limits of continuous joints
        continuous_joint_indices = [i for i, v in enumerate(self.free_variables) if not v.has_position_limits()]
        for o in range(self.prediction_horizon):
            for i in continuous_joint_indices:
                rows_to_delete.append(i + len(self.free_variables) * (o))

        A_soft.remove(rows_to_delete, [])
        A_soft.remove([], columns_to_delete)

        # position constraints if limits are violated
        # if self.default_limits:
        #     A_soft[0:self.num_position_limits() * self.prediction_horizon,
        #     self.prediction_horizon:self.num_position_limits] = w.eye(self.num_position_limits())
        # hack = blackboard_god_map().to_symbol(identifier.hack)
        # A_soft = w.ca.substitute(A_soft, hack, 1)
        return A_soft

    def A(self):
        return self.construct_A()


class QPController:
    """
    Wraps around QP Solver. Builds the required matrices from constraints.
    """
    time_collector: TimeCollector
    debug_expressions: Dict[str, w.all_expressions]
    compiled_debug_expressions: Dict[str, w.CompiledFunction]
    evaluated_debug_expressions: Dict[str, np.ndarray]

    def __init__(self,
                 sample_period: float,
                 prediction_horizon: int,
                 solver_name: str,
                 free_variables: List[FreeVariable] = None,
                 constraints: List[Constraint] = None,
                 velocity_constraints: List[VelocityConstraint] = None,
                 debug_expressions: Dict[str, Union[w.Symbol, float]] = None,
                 retries_with_relaxed_constraints: int = 0,
                 retry_added_slack: float = 100,
                 retry_weight_factor: float = 100,
                 time_collector: TimeCollector = None):
        self.time_collector = time_collector
        self.free_variables = []
        self.constraints = []
        self.velocity_constraints = []
        self.debug_expressions = {}
        self.prediction_horizon = prediction_horizon
        self.sample_period = sample_period
        self.retries_with_relaxed_constraints = retries_with_relaxed_constraints
        self.retry_added_slack = retry_added_slack
        self.retry_weight_factor = retry_weight_factor
        self.xdot_full = None
        if free_variables is not None:
            self.add_free_variables(free_variables)
        if constraints is not None:
            self.add_constraints(constraints)
        if velocity_constraints is not None:
            self.add_velocity_constraints(velocity_constraints)
        if debug_expressions is not None:
            self.add_debug_expressions(debug_expressions)

        qp_solver_class: Type[QPSolver]
        if solver_name == SupportedQPSolver.gurobi:
            from giskardpy.qp.qp_solver_gurobi import QPSolverGurobi
            qp_solver_class = QPSolverGurobi
        elif solver_name == SupportedQPSolver.cplex:
            from giskardpy.qp.qp_solver_cplex import QPSolverCplex
            qp_solver_class = QPSolverCplex
        else:
            from giskardpy.qp.qp_solver_qpoases import QPSolverQPOases
            qp_solver_class = QPSolverQPOases
        num_non_slack = len(self.free_variables) * self.prediction_horizon * (self.order - 1)
        self.qp_solver = qp_solver_class(num_non_slack=num_non_slack,
                                         retry_added_slack=self.retry_added_slack,
                                         retry_weight_factor=self.retry_weight_factor,
                                         retries_with_relaxed_constraints=self.retries_with_relaxed_constraints)
        logging.loginfo(f'Using QP Solver \'{solver_name}\'')
        logging.loginfo(f'Prediction horizon: \'{self.prediction_horizon}\'')

    def add_free_variables(self, free_variables):
        """
        :type free_variables: list
        """
        if len(free_variables) == 0:
            raise QPSolverException('Cannot solve qp with no free variables')
        self.free_variables.extend(list(sorted(free_variables, key=lambda x: x.position_name)))
        l = [x.position_name for x in free_variables]
        duplicates = set([x for x in l if l.count(x) > 1])
        self.order = min(self.prediction_horizon + 1, max(v.order for v in self.free_variables))
        assert duplicates == set(), f'there are free variables with the same name: {duplicates}'

    def get_free_variable(self, name):
        """
        :type name: str
        :rtype: FreeVariable
        """
        for v in self.free_variables:
            if v.position_name == name:
                return v
        raise KeyError(f'No free variable with name: {name}')

    def add_constraints(self, constraints):
        """
        :type constraints: list
        """
        self.constraints.extend(list(sorted(constraints, key=lambda x: x.name)))
        l = [x.name for x in constraints]
        duplicates = set([x for x in l if l.count(x) > 1])
        assert duplicates == set(), f'there are multiple constraints with the same name: {duplicates}'
        for c in self.constraints:
            c.control_horizon = min(c.control_horizon, self.prediction_horizon)
            self.check_control_horizon(c)

    def add_velocity_constraints(self, constraints):
        """
        :type constraints: list
        """
        self.velocity_constraints.extend(list(sorted(constraints, key=lambda x: x.name)))
        l = [x.name for x in constraints]
        duplicates = set([x for x in l if l.count(x) > 1])
        assert duplicates == set(), f'there are multiple constraints with the same name: {duplicates}'
        for c in self.velocity_constraints:
            self.check_control_horizon(c)

    def check_control_horizon(self, constraint):
        if constraint.control_horizon is None:
            constraint.control_horizon = self.prediction_horizon
        elif constraint.control_horizon <= 0 or not isinstance(constraint.control_horizon, int):
            raise ValueError(f'Control horizon of {constraint.name} is {constraint.control_horizon}, '
                             f'it has to be an integer 1 <= control horizon <= prediction horizon')
        elif constraint.control_horizon > self.prediction_horizon:
            logging.logwarn(f'Specified control horizon of {constraint.name} is bigger than prediction horizon.'
                            f'Reducing control horizon of {constraint.control_horizon} '
                            f'to prediction horizon of {self.prediction_horizon}')
            constraint.control_horizon = self.prediction_horizon

    def add_debug_expressions(self, debug_expressions):
        """
        :type debug_expressions: dict
        """
        self.debug_expressions.update(debug_expressions)

    @profile
    def compile(self):
        self._construct_big_ass_M(default_limits=False)
        self._compile_big_ass_M()
        self._compile_debug_expressions()

    def get_parameter_names(self):
        return self.compiled_big_ass_M.str_params

    @profile
    def _compile_big_ass_M(self):
        t = time()
        free_symbols = w.free_symbols(self.big_ass_M)
        # free_symbols = set(free_symbols)
        # free_symbols = list(free_symbols)
        self.compiled_big_ass_M = self.big_ass_M.compile(free_symbols)
        compilation_time = time() - t
        logging.loginfo(f'Compiled symbolic controller in {compilation_time:.5f}s')
        self.time_collector.compilations.append(compilation_time)

    def _compile_debug_expressions(self):
        t = time()
        self.compiled_debug_expressions = {}
        free_symbols = set()
        for name, expr in self.debug_expressions.items():
            free_symbols.update(expr.free_symbols())
        free_symbols = list(free_symbols)
        for name, expr in self.debug_expressions.items():
            self.compiled_debug_expressions[name] = expr.compile(free_symbols)
        compilation_time = time() - t
        logging.loginfo(f'Compiled debug expressions in {compilation_time:.5f}s')


    def _are_joint_limits_violated(self, percentage: float = 0.0):
        joint_with_position_limits = [x for x in self.free_variables if x.has_position_limits()]
        num_joint_with_position_limits = len(joint_with_position_limits)
        name_replacements = {}
        for old_name in self.p_lbA_raw.index:
            for free_variable in self.free_variables:
                short_old_name = old_name.split('/')[1]
                if short_old_name == free_variable.position_name:
                    name_replacements[old_name] = str(free_variable.name)
        lbA = self.p_lbA_raw[:num_joint_with_position_limits]
        ubA = self.p_ubA_raw[:num_joint_with_position_limits]
        lbA = lbA.rename(name_replacements)
        ubA = ubA.rename(name_replacements)
        joint_range = ubA - lbA
        joint_range *= percentage
        lbA_danger = lbA[lbA > -joint_range].dropna()
        ubA_danger = ubA[ubA < joint_range].dropna()
        msg = None
        if len(lbA_danger) > 0:
            msg = f'The following joints are below their lower position limits by:\n{(-lbA_danger).to_string()}\n'
        if len(ubA_danger) > 0:
            if msg is None:
                msg = ''
            msg += f'The following joints are above their upper position limits by:\n{(-ubA_danger).to_string()}\n'
        return msg

    def save_all_pandas(self):
        if hasattr(self, 'p_xdot') and self.p_xdot is not None:
            save_pandas(
                [self.p_weights, self.p_A, self.p_Ax, self.p_lbA, self.p_ubA, self.p_lb, self.p_ub, self.p_debug,
                 self.p_xdot],
                ['weights', 'A', 'Ax', 'lbA', 'ubA', 'lb', 'ub', 'debug', 'xdot'],
                self.god_map.get_data(identifier.tmp_folder))
        else:
            save_pandas(
                [self.p_weights, self.p_A, self.p_lbA, self.p_ubA, self.p_lb, self.p_ub, self.p_debug],
                ['weights', 'A', 'lbA', 'ubA', 'lb', 'ub', 'debug'],
                self.god_map.get_data(identifier.tmp_folder))

    def _is_inf_in_data(self):
        logging.logerr(f'The following weight entries contain inf:\n'
                       f'{self.p_weights[self.p_weights == np.inf].dropna()}')
        logging.logerr(f'The following lbA entries contain inf:\n'
                       f'{self.p_lbA[self.p_lbA == np.inf].dropna()}')
        logging.logerr(f'The following ubA entries contain inf:\n'
                       f'{self.p_ubA[self.p_ubA == np.inf].dropna()}')
        logging.logerr(f'The following lb entries contain inf:\n'
                       f'{self.p_lb[self.p_lb == np.inf].dropna()}')
        logging.logerr(f'The following ub entries contain inf:\n'
                       f'{self.p_ub[self.p_ub == np.inf].dropna()}')
        if np.inf in self.np_A:
            rows = self.p_A[self.p_A == np.inf].dropna(how='all').dropna(axis=1)
            logging.logerr(f'A contains inf in:\n'
                           f'{list(rows.index)}')
        if np.any(np.isnan(self.np_A)):
            rows = self.p_A.isna()[self.p_A.isna()].dropna(how='all').dropna(axis=1)
            logging.logerr(f'A constrains nan in: \n'
                           f'{list(rows.index)}')
        return True

    @property
    def god_map(self) -> GodMap:
        return GodMap()

    @property
    def world(self) -> WorldTree:
        return self.god_map.get_data(identifier.world)

    def __print_pandas_array(self, array):
        import pandas as pd
        if len(array) > 0:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(array)

    def _init_big_ass_M(self):
        self.big_ass_M = w.zeros(self.A.height + 3,
                                 self.A.width + 2)
        # self.debug_v = w.zeros(len(self.debug_expressions), 1)

    def _set_A_soft(self, A_soft):
        self.big_ass_M[:self.A.height, :self.A.width] = A_soft

    def _set_weights(self, weights):
        self.big_ass_M[self.A.height, :-2] = weights

    def _set_lb(self, lb):
        self.big_ass_M[self.A.height + 1, :-2] = lb

    def _set_ub(self, ub):
        self.big_ass_M[self.A.height + 2, :-2] = ub

    def _set_lbA(self, lbA):
        self.big_ass_M[:self.A.height, self.A.width] = lbA

    def _set_ubA(self, ubA):
        self.big_ass_M[:self.A.height, self.A.width + 1] = ubA

    @profile
    def _construct_big_ass_M(self, default_limits=False):
        self.b = B(free_variables=self.free_variables,
                   constraints=self.constraints,
                   velocity_constraints=self.velocity_constraints,
                   sample_period=self.sample_period,
                   prediction_horizon=self.prediction_horizon,
                   order=self.order,
                   default_limits=default_limits)
        self.H = H(free_variables=self.free_variables,
                   constraints=self.constraints,
                   velocity_constraints=self.velocity_constraints,
                   sample_period=self.sample_period,
                   prediction_horizon=self.prediction_horizon,
                   order=self.order,
                   default_limits=default_limits)
        self.bA = BA(free_variables=self.free_variables,
                     constraints=self.constraints,
                     velocity_constraints=self.velocity_constraints,
                     sample_period=self.sample_period,
                     prediction_horizon=self.prediction_horizon,
                     order=self.order,
                     default_limits=default_limits)
        self.A = A(free_variables=self.free_variables,
                   constraints=self.constraints,
                   velocity_constraints=self.velocity_constraints,
                   sample_period=self.sample_period,
                   prediction_horizon=self.prediction_horizon,
                   order=self.order,
                   time_collector=self.time_collector,
                   default_limits=default_limits)

        logging.loginfo(f'Constructing new controller with {self.A.height} constraints '
                        f'and {self.A.width} free variables...')
        self.time_collector.constraints.append(self.A.height)
        self.time_collector.variables.append(self.A.width)

        self._init_big_ass_M()

        self._set_weights(w.Expression(self.H.weights()))
        self._set_A_soft(self.A.A())
        lbA, ubA = self.bA()
        self._set_lbA(w.Expression(lbA))
        self._set_ubA(w.Expression(ubA))
        lb, ub = self.b()
        self._set_lb(w.Expression(lb))
        self._set_ub(w.Expression(ub))
        self.np_g = np.zeros(self.H.width)
        # self.debug_names = list(sorted(self.debug_expressions.keys()))
        # self.debug_v = w.Expression([self.debug_expressions[name] for name in self.debug_names])

    @profile
    def _eval_debug_exprs(self):
        self.evaluated_debug_expressions = {}
        for name, f in self.compiled_debug_expressions.items():
            params = self.god_map.get_values(f.str_params)
            self.evaluated_debug_expressions[name] = f.call2(params).copy()
        return self.evaluated_debug_expressions

    @profile
    def make_filters(self):
        b_filter = self.np_weights != 0
        b_filter[:self.H.number_of_free_variables_with_horizon()] = True
        # offset = self.H.number_of_free_variables_with_horizon() + self.H.number_of_constraint_vel_variables()
        # map_ = self.H.make_error_id_to_vel_ids_map()
        # for i in range(self.H.number_of_contraint_error_variables()):
        #     index = i+offset
        #     if not b_filter[index]:
        #         b_filter[map_[index]] = False

        bA_filter = np.ones(self.A.height, dtype=bool)
        ll = self.H.number_of_constraint_vel_variables() + self.H.number_of_contraint_error_variables()
        bA_filter[-ll:] = b_filter[-ll:]
        return np.array(b_filter), np.array(bA_filter)

    @profile
    def filter_zero_weight_stuff(self, b_filter, bA_filter):
        return self.np_weights[b_filter], \
               np.zeros(b_filter.shape[0]), \
               self.np_A[bA_filter, :][:, b_filter], \
               self.np_lb[b_filter], \
               self.np_ub[b_filter], \
               self.np_lbA[bA_filter], \
               self.np_ubA[bA_filter]

    def __swap_compiled_matrices(self):
        if not hasattr(self, 'compiled_big_ass_M_with_default_limits'):
            with suppress_stdout():
                self.compiled_big_ass_M_with_default_limits = self.compiled_big_ass_M
                self._construct_big_ass_M(default_limits=True)
                self._compile_big_ass_M()
        else:
            self.compiled_big_ass_M, \
            self.compiled_big_ass_M_with_default_limits = self.compiled_big_ass_M_with_default_limits, \
                                                          self.compiled_big_ass_M

    @property
    def traj_time_in_sec(self):
        return self.god_map.unsafe_get_data(identifier.time) * self.god_map.unsafe_get_data(identifier.sample_period)

    @profile
    def get_cmd(self, substitutions: list) -> Tuple[derivative_joint_map, dict]:
        """
        Uses substitutions for each symbol to compute the next commands for each joint.
        :param substitutions:
        :return: joint name -> joint command
        """
        filtered_stuff = self.evaluate_and_split(substitutions)
        try:
            # self.__swap_compiled_matrices()
            self.xdot_full = self.qp_solver.solve_and_retry(*filtered_stuff)
            # self.__swap_compiled_matrices()
            self._create_debug_pandas()
            return self.split_xdot(self.xdot_full), self._eval_debug_exprs()
        except InfeasibleException as e_original:
            if isinstance(e_original, HardConstraintsViolatedException):
                raise
            self.xdot_full = None
            self._create_debug_pandas()
            joint_limits_violated_msg = self._are_joint_limits_violated()
            if joint_limits_violated_msg is not None:
                self.__swap_compiled_matrices()
                try:
                    self.xdot_full = self.qp_solver.solve(*self.evaluate_and_split(substitutions))
                    return self.split_xdot(self.xdot_full), self._eval_debug_exprs()
                except Exception as e2:
                    # self._create_debug_pandas()
                    # raise OutOfJointLimitsException(self._are_joint_limits_violated())
                    raise OutOfJointLimitsException(joint_limits_violated_msg)
                finally:
                    self.__swap_compiled_matrices()
            #         self.free_variables[0].god_map.get_data(['world']).state.pretty_print()
            self._are_hard_limits_violated(substitutions, str(e_original), *filtered_stuff)
            self._is_inf_in_data()
            raise

    def evaluate_and_split(self, substitutions):
        self.substitutions = substitutions
        np_big_ass_M = self.compiled_big_ass_M.call2(substitutions)
        self.np_weights = np_big_ass_M[self.A.height, :-2]
        self.np_A = np_big_ass_M[:self.A.height, :self.A.width]
        self.np_lb = np_big_ass_M[self.A.height + 1, :-2]
        self.np_ub = np_big_ass_M[self.A.height + 2, :-2]
        self.np_lbA = np_big_ass_M[:self.A.height, -2]
        self.np_ubA = np_big_ass_M[:self.A.height, -1]

        filters = self.make_filters()
        filtered_stuff = self.filter_zero_weight_stuff(*filters)
        return filtered_stuff

    def _are_hard_limits_violated(self, substitutions, error_message, weights, g, A, lb, ub, lbA, ubA):
        num_non_slack = len(self.free_variables) * self.prediction_horizon * (self.order - 1)
        num_of_slack = len(lb) - num_non_slack
        lb[-num_of_slack:] = -100
        ub[-num_of_slack:] = 100
        try:
            self.xdot_full = self.qp_solver.solve(weights, g, A, lb, ub, lbA, ubA)
        except:
            pass
        else:
            self._create_debug_pandas()
            upper_violations = self.p_xdot[self.p_ub.data < self.p_xdot.data]
            lower_violations = self.p_xdot[self.p_lb.data > self.p_xdot.data]
            if len(upper_violations) > 0 or len(lower_violations) > 0:
                error_message += '\n'
                if len(upper_violations) > 0:
                    error_message += 'upper slack bounds of following constraints might be too low: {}\n'.format(
                        list(upper_violations.index))
                if len(lower_violations) > 0:
                    error_message += 'lower slack bounds of following constraints might be too high: {}'.format(
                        list(lower_violations.index))
                raise HardConstraintsViolatedException(error_message)
        logging.loginfo('No slack limit violation detected.')
        return False

    def split_xdot(self, xdot) -> derivative_joint_map:
        split = {}
        offset = len(self.free_variables)
        for derivative in range(self.order - 1):
            split[Derivatives(derivative + 1)] = OrderedDict((x.position_name,
                                                              xdot[i + offset * self.prediction_horizon * derivative])
                                                             for i, x in enumerate(self.free_variables))
        return split

    def b_names(self):
        return self.b.names

    def bA_names(self):
        return self.bA.names

    def _viz_mpc(self, joint_name):
        def pad(a, desired_length):
            tmp = np.zeros(desired_length)
            tmp[:len(a)] = a
            return tmp

        sample_period = self.state[str(self.sample_period)]
        try:
            start_pos = self.state[joint_name]
        except KeyError:
            logging.loginfo('start position not found in state')
            start_pos = 0
        ts = np.array([(i + 1) * sample_period for i in range(self.prediction_horizon)])
        filtered_x = self.p_xdot.filter(like='{}'.format(joint_name), axis=0)
        velocities = filtered_x[:self.prediction_horizon].values
        if joint_name in self.state:
            accelerations = filtered_x[self.prediction_horizon:self.prediction_horizon * 2].values
            jerks = filtered_x[self.prediction_horizon * 2:self.prediction_horizon * 3].values
        positions = [start_pos]
        for x_ in velocities:
            positions.append(positions[-1] + x_ * sample_period)

        positions = np.array(positions[1:])
        velocities = pad(velocities.T[0], len(ts))
        positions = pad(positions.T[0], len(ts))

        f, axs = plt.subplots(4, sharex=True)
        axs[0].set_title('position')
        axs[0].plot(ts, positions, 'b')
        axs[0].grid()
        axs[1].set_title('velocity')
        axs[1].plot(ts, velocities, 'b')
        axs[1].grid()
        if joint_name in self.state:
            axs[2].set_title('acceleration')
            axs[2].plot(ts, accelerations, 'b')
            axs[2].grid()
            axs[3].set_title('jerk')
            axs[3].plot(ts, jerks, 'b')
            axs[3].grid()
        plt.tight_layout()
        path, dirs, files = next(os.walk('tmp_data/mpc'))
        file_count = len(files)
        plt.savefig('tmp_data/mpc/mpc_{}_{}.png'.format(joint_name, file_count))

    @profile
    def _create_debug_pandas(self):
        substitutions = self.substitutions
        self.np_H = np.diag(self.np_weights)
        self.state = {k: v for k, v in zip(self.compiled_big_ass_M.str_params, substitutions)}
        sample_period = self.sample_period
        b_names = self.b_names()
        bA_names = self.bA_names()
        b_filter, bA_filter = self.make_filters()
        filtered_b_names = np.array(b_names)[b_filter]
        filtered_bA_names = np.array(bA_names)[bA_filter]
        H, g, A, lb, ub, lbA, ubA = self.filter_zero_weight_stuff(b_filter, bA_filter)
        # H, g, A, lb, ub, lbA, ubA = self.np_H, self.np_g, self.np_A, self.np_lb, self.np_ub, self.np_lbA, self.np_ubA
        # num_non_slack = len(self.free_variables) * self.prediction_horizon * 3
        # num_of_slack = len(lb) - num_non_slack
        num_vel_constr = len(self.velocity_constraints) * (self.prediction_horizon - 2)
        num_task_constr = len(self.constraints)
        num_constr = num_vel_constr + num_task_constr
        # num_non_slack = l

        self._eval_debug_exprs()
        p_debug = {}
        for name, value in self.evaluated_debug_expressions.items():
            if isinstance(value, np.ndarray):
                p_debug[name] = value.reshape((value.shape[0] * value.shape[1]))
            else:
                p_debug[name] = np.array(value)
        self.p_debug = pd.DataFrame.from_dict(p_debug, orient='index').sort_index()

        self.p_lb = pd.DataFrame(lb, filtered_b_names, ['data'], dtype=float)
        self.p_ub = pd.DataFrame(ub, filtered_b_names, ['data'], dtype=float)
        # self.p_g = pd.DataFrame(g, filtered_b_names, ['data'], dtype=float)
        self.p_lbA_raw = pd.DataFrame(lbA, filtered_bA_names, ['data'], dtype=float)
        self.p_lbA = deepcopy(self.p_lbA_raw)
        self.p_ubA_raw = pd.DataFrame(ubA, filtered_bA_names, ['data'], dtype=float)
        self.p_ubA = deepcopy(self.p_ubA_raw)
        # remove sample period factor
        self.p_lbA[-num_constr:] /= sample_period
        self.p_ubA[-num_constr:] /= sample_period
        self.p_weights = pd.DataFrame(self.np_H.dot(np.ones(self.np_H.shape[0])), b_names, ['data'],
                                      dtype=float)
        self.p_A = pd.DataFrame(A, filtered_bA_names, filtered_b_names, dtype=float)
        if self.xdot_full is not None:
            self.p_xdot = pd.DataFrame(self.xdot_full, filtered_b_names, ['data'], dtype=float)
            # Ax = np.dot(self.np_A, xdot_full)
            xH = np.dot((self.xdot_full ** 2).T, H)
            self.p_xH = pd.DataFrame(xH, filtered_b_names, ['data'], dtype=float)
            # p_xg = p_g * p_xdot
            # xHx = np.dot(np.dot(xdot_full.T, H), xdot_full)

            self.p_pure_xdot = deepcopy(self.p_xdot)
            self.p_pure_xdot[-num_constr:] = 0
            self.p_Ax = pd.DataFrame(self.p_A.dot(self.p_xdot), filtered_bA_names, ['data'], dtype=float)
            self.p_Ax_without_slack_raw = pd.DataFrame(self.p_A.dot(self.p_pure_xdot), filtered_bA_names, ['data'],
                                                       dtype=float)
            self.p_Ax_without_slack = deepcopy(self.p_Ax_without_slack_raw)
            self.p_Ax_without_slack[-num_constr:] /= sample_period

        else:
            self.p_xdot = None

        # if self.lbAs is None:
        #     self.lbAs = p_lbA
        # else:
        #     self.lbAs = self.lbAs.T.append(p_lbA.T, ignore_index=True).T
        # self.lbAs.T[[c for c in self.lbAs.T.columns if 'dist' in c]].plot()

        # self.save_all(p_weights, p_A, p_lbA, p_ubA, p_lb, p_ub, p_xdot)

        # self._viz_mpc('j2')
        # self._viz_mpc(self.p_xdot, 'world_robot_joint_state_r_shoulder_lift_joint_position', state)
        # self._viz_mpc(self.p_Ax_without_slack, bA_names[-91][:-2], state)
        # p_lbA[p_lbA != 0].abs().sort_values(by='data')
        # get non 0 A entries
        # p_A.iloc[[1133]].T.loc[p_A.values[1133] != 0]
        # self.save_all_pandas()
        # self._viz_mpc(bA_names[-1])
        pass
