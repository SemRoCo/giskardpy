import datetime
import os
from collections import OrderedDict, defaultdict
from copy import deepcopy
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rospy
from geometry_msgs.msg import Quaternion, TransformStamped
from tf.transformations import quaternion_about_axis
from tf2_msgs.msg import TFMessage

import giskardpy.utils.tfwrapper
from giskardpy import casadi_wrapper as w
from giskardpy.exceptions import OutOfJointLimitsException, \
    HardConstraintsViolatedException
from giskardpy.qp.constraint import VelocityConstraint, Constraint
from giskardpy.qp.free_variable import FreeVariable
from giskardpy.utils import logging
from giskardpy.utils.tfwrapper import np_to_kdl, kdl_to_quaternion
from giskardpy.utils.time_collector import TimeCollector
from giskardpy.utils.utils import memoize, create_path


def save_pandas(dfs, names, path):
    folder_name = '{}/pandas_{}/'.format(path, datetime.datetime.now().strftime('%Yy-%mm-%dd--%Hh-%Mm-%Ss'))
    create_path(folder_name)
    for df, name in zip(dfs, names):
        csv_string = 'name\n'
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            if df.shape[1] > 1:
                for column_name, column in df.T.items():
                    csv_string += column.add_prefix(column_name + '||').to_csv(float_format='%.4f')
            else:
                csv_string += df.to_csv(float_format='%.4f')
        file_name2 = '{}{}.csv'.format(folder_name, name)
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
    def __init__(self, free_variables, constraints, velocity_constraints, sample_period, prediction_horizon, order):
        super(H, self).__init__(sample_period, prediction_horizon, order)
        self.free_variables = free_variables
        self.constraints = constraints  # type: list[Constraint]
        self.velocity_constraints = velocity_constraints  # type: list[velocity_constraints]
        self.height = 0
        self._compute_height()

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
                    weights[o]['t{:03d}/{}/{}'.format(t, v.name, o)] = v.normalized_weight(t, o,
                                                                                           self.prediction_horizon)
        slack_weights = {}
        for t in range(self.prediction_horizon):
            for c in self.velocity_constraints:  # type: VelocityConstraint
                if t < c.control_horizon:
                    slack_weights['t{:03d}/{}'.format(t, c.name)] = c.normalized_weight(t)

        error_slack_weights = {'{}/error'.format(c.name): c.normalized_weight(self.prediction_horizon) for c in
                               self.constraints}

        params = []
        for o, w in sorted(weights.items()):
            params.append(w)
        params.append(slack_weights)
        params.append(error_slack_weights)
        return self._sorter(*params)[0]

    @profile
    @memoize
    def make_error_id_to_vel_ids_map(self):
        d = defaultdict(list)
        start_id1 = self.number_of_free_variables_with_horizon()
        start_id = self.number_of_free_variables_with_horizon() + self.number_of_constraint_vel_variables()
        c = 0
        for t in range(self.prediction_horizon):
            for i, constraint in enumerate(self.constraints):
                if t < constraint.control_horizon:
                    d[start_id + i].append(start_id1 + c)
                    c += 1
        return {k: np.array(v) for k, v in d.items()}


class B(Parent):
    def __init__(self, free_variables, constraints, velocity_constraints, sample_period, prediction_horizon, order):
        super(B, self).__init__(sample_period, prediction_horizon, order)
        self.free_variables = free_variables  # type: list[FreeVariable]
        self.constraints = constraints  # type: list[Constraint]
        self.velocity_constraints = velocity_constraints  # type: list[VelocityConstraint]
        self.no_limits = 1e4

    def get_lower_slack_limits(self):
        result = {}
        for t in range(self.prediction_horizon):
            for c in self.velocity_constraints:
                if t < c.control_horizon:
                    result['t{:03d}/{}'.format(t, c.name)] = c.lower_slack_limit
        return result

    def get_upper_slack_limits(self):
        result = {}
        for t in range(self.prediction_horizon):
            for c in self.velocity_constraints:
                if t < c.control_horizon:
                    result['t{:03d}/{}'.format(t, c.name)] = c.upper_slack_limit
        return result

    def get_lower_error_slack_limits(self):
        return {'{}/error'.format(c.name): c.lower_slack_limit for c in self.constraints}

    def get_upper_error_slack_limits(self):
        return {'{}/error'.format(c.name): c.upper_slack_limit for c in self.constraints}

    def __call__(self):
        lb = defaultdict(dict)
        ub = defaultdict(dict)
        for t in range(self.prediction_horizon):
            for v in self.free_variables:  # type: FreeVariable
                for o in range(1, min(v.order, self.order)):  # start with velocity
                    if t == self.prediction_horizon - 1 and o < min(v.order, self.order) - 1 and self.prediction_horizon > 2:  # and False:
                        lb[o]['t{:03d}/{}/{}'.format(t, v.name, o)] = 0
                        ub[o]['t{:03d}/{}/{}'.format(t, v.name, o)] = 0
                    else:
                        lb[o]['t{:03d}/{}/{}'.format(t, v.name, o)] = v.get_lower_limit(o)
                        ub[o]['t{:03d}/{}/{}'.format(t, v.name, o)] = v.get_upper_limit(o)
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
    def __init__(self, free_variables, constraints, velocity_constraints, sample_period, prediction_horizon, order):
        super(BA, self).__init__(sample_period, prediction_horizon, order)
        self.free_variables = free_variables
        self.constraints = constraints
        self.velocity_constraints = velocity_constraints
        self.round_to = 5
        self.round_to2 = 10

    def get_lower_constraint_velocities(self):
        result = {}
        for t in range(self.prediction_horizon):
            for c in self.velocity_constraints:
                if t < c.control_horizon:
                    result['t{:03d}/{}'.format(t, c.name)] = c.lower_velocity_limit * self.sample_period
        return result

    def get_upper_constraint_velocities(self):
        result = {}
        for t in range(self.prediction_horizon):
            for c in self.velocity_constraints:
                if t < c.control_horizon:
                    result['t{:03d}/{}'.format(t, c.name)] = c.upper_velocity_limit * self.sample_period
        return result

    @memoize
    def get_lower_constraint_error(self):
        return {'{}/e'.format(c.name): w.limit(c.lower_error,
                                               -c.velocity_limit * self.sample_period * c.control_horizon,
                                               c.velocity_limit * self.sample_period * c.control_horizon)
                for c in self.constraints}

    @memoize
    def get_upper_constraint_error(self):
        return {'{}/e'.format(c.name): w.limit(c.upper_error,
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
                    lb['t{:03d}/{}/p_limit'.format(t, v.name)] = w.round_up(v.get_lower_limit(0) - v.get_symbol(0),
                                                                            self.round_to2)
                    ub['t{:03d}/{}/p_limit'.format(t, v.name)] = w.round_down(v.get_upper_limit(0) - v.get_symbol(0),
                                                                              self.round_to2)

        l_last_stuff = defaultdict(dict)
        u_last_stuff = defaultdict(dict)
        for v in self.free_variables:
            for o in range(1, min(v.order, self.order) - 1):
                l_last_stuff[o]['{}/last_{}'.format(v.name, o)] = w.round_down(v.get_symbol(o), self.round_to)
                u_last_stuff[o]['{}/last_{}'.format(v.name, o)] = w.round_up(v.get_symbol(o), self.round_to)

        derivative_link = defaultdict(dict)
        for t in range(self.prediction_horizon - 1):
            for v in self.free_variables:
                for o in range(1, min(v.order, self.order) - 1):
                    derivative_link[o]['t{:03d}/{}/{}/link'.format(t, o, v.name)] = 0

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
    def __init__(self, free_variables, constraints, velocity_constraints, sample_period, prediction_horizon, order, time_collector):
        super(A, self).__init__(sample_period, prediction_horizon, order, time_collector)
        self.free_variables = free_variables  # type: list[FreeVariable]
        self.constraints = constraints  # type: list[Constraint]
        self.velocity_constraints = velocity_constraints  # type: list[VelocityConstraint]
        self.joints = {}
        self.height = 0
        self._compute_height()
        self.width = 0
        self._compute_width()

    def _compute_height(self):
        # rows for position limits of non continuous joints
        self.height = self.prediction_horizon * (self.number_of_joints - self.num_of_continuous_joints())
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

    @property
    def number_of_joints(self):
        return len(self.free_variables)

    @memoize
    def num_of_continuous_joints(self):
        return len([v for v in self.free_variables if not v.has_position_limits()])

    def get_constraint_expressions(self):
        return self._sorter({c.name: c.expression for c in self.constraints})[0]

    def get_velocity_constraint_expressions(self):
        return self._sorter({c.name: c.expression for c in self.velocity_constraints})[0]

    def get_free_variable_symbols(self):
        return self._sorter({v.name: v.get_symbol(0) for v in self.free_variables})[0]

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
        # TODO possible speed improvement by creating blocks and stitching them together
        number_of_joints = self.number_of_joints
        A_soft = w.zeros(
            self.prediction_horizon * number_of_joints +  # joint position constraints
            number_of_joints * self.prediction_horizon * (self.order - 2) +  # links
            len(self.velocity_constraints) * (self.prediction_horizon) +  # velocity constraints
            len(self.constraints),  # constraints
            number_of_joints * self.prediction_horizon * (self.order - 1) +
            len(self.velocity_constraints) * self.prediction_horizon + len(self.constraints)
        )
        t = time()
        J_vel = w.jacobian(w.Matrix(self.get_velocity_constraint_expressions()), self.get_free_variable_symbols(),
                           order=1)
        J_err = w.jacobian(w.Matrix(self.get_constraint_expressions()), self.get_free_variable_symbols(), order=1)
        J_vel *= self.sample_period
        J_err *= self.sample_period
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
        J_vel_limit_block = w.kron(w.eye(self.prediction_horizon), J_vel)
        next_vertical_offset = vertical_offset + J_vel_limit_block.shape[0]
        A_soft[vertical_offset:next_vertical_offset, :J_vel_limit_block.shape[1]] = J_vel_limit_block
        I = w.eye(J_vel_limit_block.shape[0]) * self.sample_period
        A_soft[vertical_offset:next_vertical_offset, -I.shape[1] - J_err.shape[0]:-J_err.shape[0]] = I
        # delete rows if control horizon of constraint shorter than prediction horizon
        rows_to_delete = []
        for t in range(self.prediction_horizon):
            for i, c in enumerate(self.velocity_constraints):
                index = vertical_offset + i + (t * len(self.velocity_constraints))
                if t + 1 > c.control_horizon:
                    rows_to_delete.append(index)

        # delete columns where control horizon is shorter than prediction horizon
        columns_to_delete = []
        horizontal_offset = A_soft.shape[1] - I.shape[1] - J_err.shape[0]
        for t in range(self.prediction_horizon):
            for i, c in enumerate(self.velocity_constraints):
                index = horizontal_offset + (t * len(self.velocity_constraints)) + i
                if t + 1 > c.control_horizon:
                    columns_to_delete.append(index)

        # J stack for total error
        J_hstack = w.hstack([J_err for _ in range(self.prediction_horizon)])
        vertical_offset = next_vertical_offset
        next_vertical_offset = vertical_offset + J_hstack.shape[0]
        # set jacobian entry to 0 if control horizon shorter than prediction horizon
        for i, c in enumerate(self.constraints):
            # offset = vertical_offset + i
            J_hstack[i, c.control_horizon * len(self.free_variables):] = 0
        A_soft[vertical_offset:next_vertical_offset, :J_hstack.shape[1]] = J_hstack

        # sum of vel slack for total error
        # I = w.kron(w.Matrix([[1 for _ in range(self.prediction_horizon)]]),
        #            w.eye(J_hstack.shape[0])) * self.sample_period
        # A_soft[vertical_offset:next_vertical_offset, -I.shape[1]-len(self.constraints):-len(self.constraints)] = I
        # TODO multiply with control horizon instead?
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
        return A_soft

    def A(self):
        return self.construct_A()


class QPController(object):
    """
    Wraps around QP Solver. Builds the required matrices from constraints.
    """
    time_collector: TimeCollector

    def __init__(self, sample_period, prediction_horizon, solver_name,
                 free_variables=None, constraints=None, velocity_constraints=None, debug_expressions=None,
                 retries_with_relaxed_constraints=0, retry_added_slack=100, retry_weight_factor=100, time_collector=None,
                 tf_topic=None):
        self.time_collector = time_collector
        self.free_variables = []  # type: list[FreeVariable]
        self.constraints = []  # type: list[Constraint]
        self.velocity_constraints = []  # type: list[VelocityConstraint]
        self.debug_expressions = {}  # type: dict
        self.prediction_horizon = prediction_horizon
        self.sample_period = sample_period
        self.retries_with_relaxed_constraints = retries_with_relaxed_constraints
        self.retry_added_slack = retry_added_slack
        self.retry_weight_factor = retry_weight_factor
        self.tf_topic = tf_topic
        self.xdot_full = None
        if free_variables is not None:
            self.add_free_variables(free_variables)
        if constraints is not None:
            self.add_constraints(constraints)
        if velocity_constraints is not None:
            self.add_velocity_constraints(velocity_constraints)
        if debug_expressions is not None:
            self.add_debug_expressions(debug_expressions)

        if solver_name == 'gurobi':
            from giskardpy.qp.qp_solver_gurobi import QPSolverGurobi
            self.qp_solver = QPSolverGurobi()
        elif solver_name == 'qpoases':
            from giskardpy.qp.qp_solver import QPSolver
            self.qp_solver = QPSolver()
        elif solver_name == 'cplex':
            from giskardpy.qp.qp_solver_cplex import QPSolverCplex
            self.qp_solver = QPSolverCplex()
        else:
            raise KeyError('Solver \'{}\' not supported'.format(solver_name))
        logging.loginfo('Using QP Solver \'{}\''.format(solver_name))

    def add_free_variables(self, free_variables):
        """
        :type free_variables: list
        """
        # TODO check for empty goals
        self.free_variables.extend(list(sorted(free_variables, key=lambda x: x.name)))
        l = [x.name for x in free_variables]
        duplicates = set([x for x in l if l.count(x) > 1])
        self.order = min(self.prediction_horizon + 1, max(v.order for v in self.free_variables))
        assert duplicates == set(), 'there are free variables with the same name: {}'.format(duplicates)

    def get_free_variable(self, name):
        """
        :type name: str
        :rtype: FreeVariable
        """
        for v in self.free_variables:
            if v.name == name:
                return v
        raise KeyError('No free variable with name: {}'.format(name))

    def add_constraints(self, constraints):
        """
        :type constraints: list
        """
        self.constraints.extend(list(sorted(constraints, key=lambda x: x.name)))
        l = [x.name for x in constraints]
        duplicates = set([x for x in l if l.count(x) > 1])
        assert duplicates == set(), 'there are multiple constraints with the same name: {}'.format(duplicates)
        for c in self.constraints:
            self.check_control_horizon(c)

    def add_velocity_constraints(self, constraints):
        """
        :type constraints: list
        """
        self.velocity_constraints.extend(list(sorted(constraints, key=lambda x: x.name)))
        l = [x.name for x in constraints]
        duplicates = set([x for x in l if l.count(x) > 1])
        assert duplicates == set(), 'there are multiple constraints with the same name: {}'.format(duplicates)
        for c in self.velocity_constraints:
            self.check_control_horizon(c)

    def check_control_horizon(self, constraint):
        if constraint.control_horizon is None:
            constraint.control_horizon = self.prediction_horizon
        elif constraint.control_horizon <= 0 or not isinstance(constraint.control_horizon, int):
            raise ValueError('Control horizon of {} is {}, it has to be an integer '
                             '1 <= control horizon <= prediction horizon'.format(constraint.name,
                                                                                 constraint.control_horizon))
        elif constraint.control_horizon > self.prediction_horizon:
            logging.logwarn('Specified control horizon of {} is bigger than prediction horizon.'
                            'Reducing control horizon of {} to prediction horizon of {}'.format(constraint.name,
                                                                                                constraint.control_horizon,
                                                                                                self.prediction_horizon))

    def add_debug_expressions(self, debug_expressions):
        """
        :type debug_expressions: dict
        """
        # TODO check duplicates
        self.debug_expressions.update(debug_expressions)

    def publish_debug_frame(self, substitutions, world):
        self.tf_pub = rospy.Publisher(self.tf_topic, TFMessage, queue_size=10)
        if len(self.debug_expressions.keys()) > 0:
            debug_exprs = self._eval_debug_exprs(substitutions)
            p_debug = pd.DataFrame.from_dict(debug_exprs, orient='index', columns=['data']).sort_index()
            frames = dict()
            for r in p_debug.iterrows():
                full_name = r[0]
                name = full_name[:len(full_name) - full_name[::-1].index('/')]
                data = r[1].data
                if '/t_R_a' in name:
                    if name in frames:
                        frames[name].append(data)
                    else:
                        frames[name] = [data]
            tf_msg = TFMessage()
            # pub basefootprint from world
            fk = world.get_fk('map', 'base_footprint')
            parent_link_name = 'map'  # self.world.get_parent_link_of_link(link_name)
            tf = TransformStamped()
            tf.header.frame_id = parent_link_name
            tf.header.stamp = rospy.get_rostime()
            tf.child_frame_id = 'root_link'
            tf.transform.translation.x = fk[0, -1]
            tf.transform.translation.y = fk[1, -1]
            tf.transform.translation.z = fk[2, -1]
            tf.transform.rotation = kdl_to_quaternion(np_to_kdl(fk)) # TODO lol
            tf_msg.transforms.append(tf)
            # pub joint calc from qp solver
            sol = self.split_xdot(self.xdot_full)[0]
            tf = TransformStamped()
            tf.header.frame_id = 'root_link'
            tf.header.stamp = rospy.get_rostime()
            tf.child_frame_id = 'odom_x_and_odom_y'
            tf.transform.translation.z = 0
            tf.transform.rotation = Quaternion(0,0,0,1)
            #tf_msg.transforms.append(tf)
            for j_n, v in sol.items():
                if 'odom_x' in j_n:
                    tf.transform.translation.x = v
                elif 'odom_y' in j_n:
                    tf.transform.translation.y = v
            length = np.sqrt(tf.transform.translation.x**2 + tf.transform.translation.y**2)
            tf.transform.translation.x /= length
            tf.transform.translation.y /= length
            # pub debug constraints
            for name, pos in frames.items():
                parent_link_name = 'root_link' #self.world.get_parent_link_of_link(link_name)
                tf = TransformStamped()
                tf.header.frame_id = parent_link_name
                tf.header.stamp = rospy.get_rostime()
                tf.child_frame_id = name
                tf.transform.translation.x = pos[0]
                tf.transform.translation.y = pos[1]
                tf.transform.translation.z = pos[2]
                tf.transform.rotation = Quaternion(0, 0, 0, 1)
                tf_msg.transforms.append(tf)
            #fk = world.get_fk('map', 'r_gripper_tool_frame')
            #parent_link_name = 'map'  # self.world.get_parent_link_of_link(link_name)
            #tf = TransformStamped()
            #tf.header.frame_id = 'map'
            #tf.header.stamp = rospy.get_rostime()
            #tf.child_frame_id = 'root_P_current'
            #tf.transform.translation.x = fk[0, -1]
            #tf.transform.translation.y = fk[1, -1]
            #tf.transform.translation.z = fk[2, -1]
            #tf.transform.rotation = kdl_to_quaternion(np_to_kdl(fk))  # TODO lol
            #tf_msg.transforms.append(tf)
            #tf = TransformStamped()
            #tf.header.frame_id = 'map'
            #tf.header.stamp = rospy.get_rostime()
            #tf.child_frame_id = 'root_P_goal'
            #tf.transform.translation.x = 1.2 + 1.5
            #tf.transform.translation.y = 0 + 1
            #tf.transform.translation.z = 1.0
            #tf.transform.rotation = Quaternion(*quaternion_about_axis(np.pi / 3, [0, 0, 1]))
            #tf_msg.transforms.append(tf)
            self.tf_pub.publish(tf_msg)
            pass

    @profile
    def compile(self):
        self._construct_big_ass_M()
        self._compile_big_ass_M()

    def get_parameter_names(self):
        return self.compiled_big_ass_M.str_params

    @profile
    def _compile_big_ass_M(self):
        t = time()
        free_symbols = w.free_symbols(self.big_ass_M)
        debug_free_symbols = w.free_symbols(self.debug_v)
        free_symbols = set(free_symbols)
        free_symbols.update(debug_free_symbols)
        free_symbols = list(free_symbols)
        self.compiled_big_ass_M = w.speed_up(self.big_ass_M,
                                             free_symbols)
        compilation_time = time() - t
        logging.loginfo('Compiled symbolic controller in {:.5f}s'.format(compilation_time))
        self.time_collector.compilations.append(compilation_time)
        # TODO should use separate symbols lists
        self.compiled_debug_v = w.speed_up(self.debug_v, free_symbols)

    def _are_joint_limits_violated(self, error_message):
        violations = (self.p_ub - self.p_lb)[self.p_lb.data > self.p_ub.data]
        if len(violations) > 0:
            error_message += '\nThe following joints are outside of their limits: \n {}'.format(violations)
            raise OutOfJointLimitsException(error_message)
        logging.loginfo('All joints are within limits')
        return False

    def _is_close_to_joint_limits(self):
        joint_with_position_limits = [x for x in self.free_variables if x.has_position_limits()]
        num_joint_with_position_limits = len(joint_with_position_limits)
        lbA = self.p_lbA_raw[:num_joint_with_position_limits]
        ubA = self.p_ubA_raw[:num_joint_with_position_limits]
        joint_range = ubA - lbA
        percentage = 0.01
        joint_range *= percentage
        lbA_danger = lbA[lbA > -joint_range].dropna()
        ubA_danger = ubA[ubA < joint_range].dropna()
        result = False
        if len(lbA_danger) > 0:
            logging.logwarn(
                'The following joints ended up closer than {}% to their lower position limits {}'.format(percentage,
                                                                                                         list(
                                                                                                             lbA_danger.index)))
            result = True
        if len(ubA_danger) > 0:
            logging.logwarn(
                'The following joints ended up closer than {}% to their upper position limits {}'.format(percentage,
                                                                                                         list(
                                                                                                             ubA_danger.index)))
            result = True
        return result

    def save_all_pandas(self):
        if hasattr(self, 'p_xdot') and self.p_xdot is not None:
            save_pandas([self.p_weights, self.p_A, self.p_lbA, self.p_ubA, self.p_lb, self.p_ub, self.p_debug,
                         self.p_xdot],
                        ['weights', 'A', 'lbA', 'ubA', 'lb', 'ub', 'debug', 'xdot'],
                        '../tmp_data')
        else:
            save_pandas([self.p_weights, self.p_A, self.p_lbA, self.p_ubA, self.p_lb, self.p_ub, self.p_debug],
                        ['weights', 'A', 'lbA', 'ubA', 'lb', 'ub', 'debug'],
                        '../tmp_data')

    def __is_nan_in_array(self, name, p_array):
        p_filtered = p_array.apply(lambda x: zip(x.index[x.isnull()].tolist(), x[x.isnull()]), 1)
        p_filtered = p_filtered[p_filtered.apply(lambda x: len(x)) > 0]
        if len(p_filtered) > 0:
            logging.logerr('{} has the following nans:'.format(name))
            self.__print_pandas_array(p_filtered)
            return True
        logging.loginfo('{} has no nans'.format(name))
        return False

    def __print_pandas_array(self, array):
        import pandas as pd
        if len(array) > 0:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(array)

    def _init_big_ass_M(self):
        """
        #
        #         |---------------|
        #         |  A  | lba| uba|
        #         |---------------|
        #         |  w  | 0  | 0  |
        #         |---------------|
        #         |  lb | 0  | 0  |
        #         |---------------|
        #         |  ub | 0  | 0  |
        #         |---------------|
        """
        self.big_ass_M = w.zeros(self.A.height + 3,
                                 self.A.width + 2)
        self.debug_v = w.zeros(len(self.debug_expressions), 1)

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
    def _construct_big_ass_M(self):
        self.b = B(self.free_variables, self.constraints, self.velocity_constraints, self.sample_period,
                   self.prediction_horizon, self.order)
        self.H = H(self.free_variables, self.constraints, self.velocity_constraints, self.sample_period,
                   self.prediction_horizon, self.order)
        self.bA = BA(self.free_variables, self.constraints, self.velocity_constraints, self.sample_period,
                     self.prediction_horizon, self.order)
        self.A = A(self.free_variables, self.constraints, self.velocity_constraints, self.sample_period,
                   self.prediction_horizon, self.order, self.time_collector)

        logging.loginfo('Constructing new controller with {} constraints and {} free variables...'.format(
            self.A.height, self.A.width))
        self.time_collector.constraints.append(self.A.height)
        self.time_collector.variables.append(self.A.width)

        self._init_big_ass_M()

        self._set_weights(w.Matrix(self.H.weights()))
        self._set_A_soft(self.A.A())
        lbA, ubA = self.bA()
        self._set_lbA(w.Matrix(lbA))
        self._set_ubA(w.Matrix(ubA))
        lb, ub = self.b()
        self._set_lb(w.Matrix(lb))
        self._set_ub(w.Matrix(ub))
        self.np_g = np.zeros(self.H.width)
        self.debug_names = list(sorted(self.debug_expressions.keys()))
        self.debug_v = w.Matrix([self.debug_expressions[name] for name in self.debug_names])
        self.H.make_error_id_to_vel_ids_map()

    @profile
    def _eval_debug_exprs(self, subsitutions):
        return {name: value[0] for name, value in zip(self.debug_names, self.compiled_debug_v.call2(subsitutions))}

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

    @profile
    def get_cmd(self, substitutions, world):
        """
        Uses substitutions for each symbol to compute the next commands for each joint.
        :param substitutions:
        :type substitutions: list
        :return: joint name -> joint command
        :rtype: [list, dict]
        """
        np_big_ass_M = self.compiled_big_ass_M.call2(substitutions)
        self.np_weights = np_big_ass_M[self.A.height, :-2]
        self.np_A = np_big_ass_M[:self.A.height, :self.A.width]
        self.np_lb = np_big_ass_M[self.A.height + 1, :-2]
        self.np_ub = np_big_ass_M[self.A.height + 2, :-2]
        self.np_lbA = np_big_ass_M[:self.A.height, -2]
        self.np_ubA = np_big_ass_M[:self.A.height, -1]

        filters = self.make_filters()
        filtered_stuff = self.filter_zero_weight_stuff(*filters)
        try:
            self.xdot_full = self.qp_solver.solve(*filtered_stuff)
            self.publish_debug_frame(substitutions, world)
        except Exception as e_original:
            if self.retries_with_relaxed_constraints:
                try:
                    logging.logwarn('Failed to solve QP, retrying with relaxed hard constraints.')
                    self.get_cmd_relaxed_hard_constraints(*filtered_stuff)
                    self.retries_with_relaxed_constraints -= 1
                    return self.split_xdot(self.xdot_full), self._eval_debug_exprs(substitutions)
                except Exception as e_relaxed:
                    logging.logerr('Relaxing hard constraints failed.')
            else:
                logging.logwarn('Ran out of allowed retries with relaxed hard constraints.')
            self._create_debug_pandas(substitutions)
            self._are_joint_limits_violated(str(e_original))
            self._is_close_to_joint_limits()
            self._are_hard_limits_violated(substitutions, str(e_original), *filtered_stuff)
            # if isinstance(e, QPSolverException):
            # FIXME
            #     arrays = [(p_weights, 'H'),
            #               (p_A, 'A'),
            #               (p_lbA, 'lbA'),
            #               (p_ubA, 'ubA'),
            #               (p_lb, 'lb'),
            #               (p_ub, 'ub')]
            #     any_nan = False
            #     for a, name in arrays:
            #         any_nan |= self.__is_nan_in_array(name, a)
            #     if any_nan:
            #         raise e
            raise
        if self.xdot_full is None:
            return None
        # for debugging to might want to execute this line to create named panda matrices
        # self._create_debug_pandas(substitutions)
        return self.split_xdot(self.xdot_full), self._eval_debug_exprs(substitutions)

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
            self._create_debug_pandas(substitutions)
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

    def get_cmd_relaxed_hard_constraints(self, weights, g, A, lb, ub, lbA, ubA):
        num_non_slack = len(self.free_variables) * self.prediction_horizon * (self.order - 1)
        num_of_slack = len(lb) - num_non_slack
        lb_relaxed = lb.copy()
        ub_relaxed = ub.copy()
        lb_relaxed[-num_of_slack:] = -self.retry_added_slack
        ub_relaxed[-num_of_slack:] = self.retry_added_slack
        self.xdot_full = self.qp_solver.solve(weights, g, A, lb_relaxed, ub_relaxed, lbA, ubA)
        upper_violations = ub < self.xdot_full
        lower_violations = lb > self.xdot_full
        if np.any(upper_violations) or np.any(lower_violations):
            weights[upper_violations | lower_violations] *= self.retry_weight_factor
            self.xdot_full = self.qp_solver.solve(weights, g, A, lb_relaxed, ub_relaxed, lbA, ubA)
            return True
        return False

    def split_xdot(self, xdot):
        split = []
        offset = len(self.free_variables)
        for derivative in range(self.order - 1):
            split.append(OrderedDict((x.name, xdot[i + offset * self.prediction_horizon * derivative])
                                     for i, x in enumerate(self.free_variables)))
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
    def _create_debug_pandas(self, substitutions):
        self.np_H = np.diag(self.np_weights)
        self.state = {k: v for k, v in zip(self.compiled_big_ass_M.str_params, substitutions)}
        sample_period = self.state[str(self.sample_period)]
        b_names = self.b_names()
        bA_names = self.bA_names()
        b_filter, bA_filter = self.make_filters()
        filtered_b_names = np.array(b_names)[b_filter]
        filtered_bA_names = np.array(bA_names)[bA_filter]
        H, g, A, lb, ub, lbA, ubA = self.filter_zero_weight_stuff(b_filter, bA_filter)
        # H, g, A, lb, ub, lbA, ubA = self.np_H, self.np_g, self.np_A, self.np_lb, self.np_ub, self.np_lbA, self.np_ubA
        num_non_slack = len(self.free_variables) * self.prediction_horizon * 3
        num_of_slack = len(lb) - num_non_slack

        debug_exprs = self._eval_debug_exprs(substitutions)
        self.p_debug = pd.DataFrame.from_dict(debug_exprs, orient='index', columns=['data']).sort_index()

        self.p_lb = pd.DataFrame(lb, filtered_b_names, ['data'], dtype=float)
        self.p_ub = pd.DataFrame(ub, filtered_b_names, ['data'], dtype=float)
        # self.p_g = pd.DataFrame(g, filtered_b_names, ['data'], dtype=float)
        self.p_lbA_raw = pd.DataFrame(lbA, filtered_bA_names, ['data'], dtype=float)
        self.p_lbA = deepcopy(self.p_lbA_raw)
        self.p_ubA_raw = pd.DataFrame(ubA, filtered_bA_names, ['data'], dtype=float)
        self.p_ubA = deepcopy(self.p_ubA_raw)
        # remove sample period factor
        self.p_lbA[-num_of_slack:] /= sample_period
        self.p_ubA[-num_of_slack:] /= sample_period
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
            self.p_pure_xdot[num_non_slack:] = 0
            self.p_Ax = pd.DataFrame(self.p_A.dot(self.p_xdot), filtered_bA_names, ['data'], dtype=float)
            self.p_Ax_without_slack_raw = pd.DataFrame(self.p_A.dot(self.p_pure_xdot), filtered_bA_names, ['data'],
                                                       dtype=float)
            self.p_Ax_without_slack = deepcopy(self.p_Ax_without_slack_raw)
            self.p_Ax_without_slack[-num_of_slack:] /= sample_period

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
