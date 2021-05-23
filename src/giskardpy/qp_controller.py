from collections import OrderedDict
from copy import deepcopy
from time import time
import matplotlib.pyplot as plt
import numpy as np

from giskardpy import logging, casadi_wrapper as w
from giskardpy.data_types import FreeVariable, Constraint
from giskardpy.exceptions import QPSolverException, InfeasibleException, OutOfJointLimitsException, \
    HardConstraintsViolatedException
from giskardpy.qp_solver import QPSolver
from giskardpy.qp_solver_gurubi import QPSolverGurubi
from giskardpy.qp_solver_osqp import QPSolverOSPQ
from giskardpy.utils import create_path


def print_pd_dfs(dfs, names):
    import pandas as pd
    import datetime
    folder_name = u'debug_matrices/{}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    for df, name in zip(dfs, names):
        path = u'{}/{}.debug'.format(folder_name, name)
        create_path(path)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            with open(path, 'w') as f:
                f.write(df.to_csv())

class Parent(object):
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


class H(Parent):
    def __init__(self, prediction_horizon, control_horizon, horizon_function=None):
        self.__j_weights_v = {}
        self.__j_weights_a = {}
        self.__j_weights_j = {}
        self._s_weights_v = {}
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        self.height = 0
        if horizon_function is None:
            self.horizon_function = lambda base_weight, t: base_weight + 0.0001*t
        else:
            self.horizon_function = horizon_function

    @property
    def width(self):
        return self.height

    def add_joint_constraint(self, free_variable):
        """
        :type free_variable: FreeVariable
        """
        name = free_variable.name
        self.__j_weights_v[name + '/v'] = free_variable.quadratic_velocity_weight
        self.__j_weights_a[name + '/a'] = free_variable.quadratic_acceleration_weight
        self.__j_weights_j[name + '/j'] = free_variable.quadratic_jerk_weight
        self.height += self.prediction_horizon*3

    def add_soft_constraint(self, constraint):
        """
        :type constraint: Constraint
        """
        name = constraint.name
        self._s_weights_v[name + '/v'] = constraint.quadratic_velocity_weight
        self.height += 1

    def weights(self):
        vel_weights = {}
        for t in range(self.prediction_horizon):
            for name, weight in self.__j_weights_v.items():
                vel_weights['t{:03d}/{}'.format(t, name)] = self.horizon_function(weight, t)
        acc_weights = {}
        for t in range(self.prediction_horizon):
            for name, weight in self.__j_weights_a.items():
                acc_weights['t{:03d}/{}'.format(t, name)] = self.horizon_function(weight, t)
        jerk_weights = {}
        for t in range(self.prediction_horizon):
            for name, weight in self.__j_weights_j.items():
                jerk_weights['t{:03d}/{}'.format(t, name)] = self.horizon_function(weight, t)
        return self._sorter(vel_weights,
                            acc_weights,
                            jerk_weights,
                            self._s_weights_v)[0]


class B(Parent):
    def __init__(self, prediction_horizon):
        self._j_lb_v = {}
        self._j_ub_v = {}
        self._j_lb_a = {}
        self._j_ub_a = {}
        self._j_lb_j = {}
        self._j_ub_j = {}
        self._s_lb_v = {}
        self._s_ub_v = {}
        self.no_limits = 1e4
        self.prediction_horizon = prediction_horizon

    def add_joint_constraint(self, free_variable):
        """
        :type constraint: FreeVariable
        """
        name = free_variable.name
        self._j_lb_v[name + '/v'] = free_variable.lower_velocity_limit
        self._j_lb_a[name + '/a'] = free_variable.lower_acceleration_limit
        self._j_lb_j[name + '/j'] = free_variable.lower_jerk_limit
        self._j_ub_v[name + '/v'] = free_variable.upper_velocity_limit
        self._j_ub_a[name + '/a'] = free_variable.upper_acceleration_limit
        self._j_ub_j[name + '/j'] = free_variable.upper_jerk_limit

    def add_soft_constraint(self, constraint):
        """
        :type constraint: Constraint
        """
        name = constraint.name
        self._s_lb_v[name + '/v'] = constraint.lower_slack_limit
        self._s_ub_v[name + '/v'] = constraint.upper_slack_limit

    def blow_up(self, d, end_with_zero=False):
        result = {}
        for t in range(self.prediction_horizon):
            for name, value in d.items():
                if t == self.prediction_horizon -1 and self.prediction_horizon > 1 and end_with_zero:
                    result['t{:03d}/{}'.format(t, name)] = 0
                else:
                    result['t{:03d}/{}'.format(t, name)] = value
        return result

    def lb(self):
        return self._sorter(self.blow_up(self._j_lb_v, True),
                            self.blow_up(self._j_lb_a, True),
                            self.blow_up(self._j_lb_j, True),
                            self._s_lb_v)[0]

    def ub(self):
        return self._sorter(self.blow_up(self._j_ub_v, True),
                            self.blow_up(self._j_ub_a, True),
                            self.blow_up(self._j_ub_j, True),
                            self._s_ub_v)[0]

    def names(self):
        return self._sorter(self.blow_up(self._j_ub_v, True),
                            self.blow_up(self._j_ub_a, True),
                            self.blow_up(self._j_ub_j, True),
                            self._s_ub_v)[1]


class BA(Parent):
    def __init__(self, order, prediction_horizon):
        self._lbA_v = {}
        self._ubA_v = {}
        self._j_lbA_a_link = {}
        self._j_ubA_a_link = {}
        self._j_lbA_j_link = {}
        self._j_ubA_j_link = {}
        self._pos_limits_lba = {}
        self._pos_limits_uba = {}
        self._pos_limits_lba2 = {}
        self._pos_limits_uba2 = {}
        self._joint_names = []
        self.prediction_horizon = prediction_horizon
        self.order = order

    def add_joint_constraint(self, free_variable):
        """
        :type free_variable: FreeVariable
        """
        name = free_variable.name
        if free_variable.has_position_limits():
            self._pos_limits_lba[name + '/pos_limit'] = w.round_up(free_variable.lower_position_limit - free_variable.position_symbol, 5)
            self._pos_limits_lba2[name + '/pos_limit'] = w.round_up(free_variable.lower_position_limit - free_variable.position_symbol, 5)
            self._pos_limits_uba[name + '/pos_limit'] = w.round_down(free_variable.upper_position_limit - free_variable.position_symbol, 5)
            self._pos_limits_uba2[name + '/pos_limit'] = w.round_down(free_variable.upper_position_limit - free_variable.position_symbol, 5)
        # this mean the last velocity only has to be matched to 5 decimal places
        self._j_lbA_a_link[name + '/last_vel'] = w.round_down(free_variable.velocity_symbol, 5)
        self._j_lbA_j_link[name + '/last_acc'] = w.round_down(free_variable.acceleration_symbol, 5)
        self._j_ubA_a_link[name + '/last_vel'] = w.round_up(free_variable.velocity_symbol, 5)
        self._j_ubA_j_link[name + '/last_acc'] = w.round_up(free_variable.acceleration_symbol, 5)
        self._joint_names.append(name)

    def add_constraint(self, constraint):
        """
        :type constraint: Constraint
        """
        name = constraint.name
        self._lbA_v[name + '/v'] = constraint.lower_velocity_limit
        self._ubA_v[name + '/v'] = constraint.upper_velocity_limit

    def blow_up(self, d):
        result = {}
        for t in range(self.prediction_horizon-1):
            for name, value in d.items():
                result['t{:03d}/{}'.format(t, name)] = value
        return result

    def lbA(self):
        # lba = [
        #       j1 pos limit, j2 pos limit, j3 pos limit,
        #       j1 pos limit, j2 pos limit, j3 pos limit,
        #       joint1 last vel for t1, ..., jointn last vel for t1,
        #       0 for each joint t2,
        #       ...
        #       0 for each joint tn,
        #       joint1 last acc for t1, ..., jointn last acc for t1,
        #       0 for each joint t2,
        #       ...
        #       0 for each joint tn,
        #       soft constraints
        #       ]
        vel_link = {}
        for joint in self._joint_names:
            for t in range(self.prediction_horizon-1):
                vel_link['t{:03d}/{}/vel_link/'.format(t+1, joint)] = 0
        acc_link = {}
        for joint in self._joint_names:
            for t in range(self.prediction_horizon-1):
                acc_link['t{:03d}/{}/acc_link/'.format(t+1, joint)] = 0
        return self._sorter(self.blow_up(self._pos_limits_lba),
                            self._pos_limits_lba2,
                            self._j_lbA_a_link,
                            vel_link,
                            self._j_lbA_j_link,
                            acc_link,
                            self._lbA_v)[0]

    def ubA(self):
        vel_link = {}
        for joint in self._joint_names:
            for t in range(self.prediction_horizon-1):
                vel_link['t{:03d}/{}/vel_link/'.format(t+1, joint)] = 0
        acc_link = {}
        for joint in self._joint_names:
            for t in range(self.prediction_horizon-1):
                acc_link['t{:03d}/{}/acc_link/'.format(t+1, joint)] = 0
        return self._sorter(self.blow_up(self._pos_limits_uba),
                            self._pos_limits_uba2,
                            self._j_ubA_a_link,
                            vel_link,
                            self._j_ubA_j_link,
                            acc_link,
                            self._ubA_v)[0]

    def names(self):
        vel_link = {}
        for joint in self._joint_names:
            for t in range(self.prediction_horizon-1):
                vel_link['t{:03d}/{}/vel_link/'.format(t + 1, joint)] = 0
        acc_link = {}
        for joint in self._joint_names:
            for t in range(self.prediction_horizon-1):
                acc_link['t{:03d}/{}/acc_link/'.format(t+1, joint)] = 0
        return self._sorter(self.blow_up(self._pos_limits_lba),
                            self._pos_limits_lba2,
                            self._j_lbA_a_link,
                            vel_link,
                            self._j_lbA_j_link,
                            acc_link,
                            self._lbA_v)[1]


class A(Parent):
    def __init__(self, sample_period, prediction_horizon, control_horizon, order=3):
        self._A_soft = {}
        self._A_hard = {}
        self._A_joint = {}
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        self.sample_period = sample_period
        self.order = order
        self.num_of_continuous_joints = 0
        self.joints = {}

    @property
    def number_of_joints(self):
        return len(self.free_variables())

    @property
    def height(self):
        return self.prediction_horizon * (self.number_of_joints - self.num_of_continuous_joints) + \
               self.number_of_joints * self.prediction_horizon * (self.order - 1) + \
               len(self._A_soft)

    @property
    def width(self):
        return self.number_of_joints * self.prediction_horizon * self.order + len(self._A_soft)

    def add_joint_constraint(self, free_variable):
        """
        :type free_variable: FreeVariable
        :return:
        """
        name = free_variable.name
        if not free_variable.has_position_limits():
            self.num_of_continuous_joints += 1
        self.joints[name] = free_variable
        self._A_joint[name] = free_variable.position_symbol

    def add_constraint(self, constraint):
        """
        :type constraint: Constraint
        """
        self._A_soft[constraint.name] = constraint.expression

    def free_variables(self):
        return self._sorter(self._A_joint)[0]

    @profile
    def construct_A_soft(self):
        #         |   t1   |   tn   |   t1   |   tn   |   t1   |   tn   |
        #         |v1 v2 vn|v1 v2 vn|a1 a2 an|a1 a2 an|j1 j2 jn|j1 j2 jn|
        #         |-----------------------------------------------------|
        #         |sp      |        |        |        |        |        |
        #         |   sp   |        |        |        |        |        |
        #         |      sp|        |        |        |        |        |
        #         |-----------------------------------------------------|
        #         |sp      |sp      |        |        |        |        |
        #         |   sp   |   sp   |        |        |        |        |
        #         |      sp|      sp|        |        |        |        |
        #         |=====================================================|
        #         | 1      |        |-sp     |        |        |        |
        #         |    1   |        |   -sp  |        |        |        |
        #         |       1|        |     -sp|        |        |        |
        #         |-----------------------------------------------------|
        #         |-1      | 1      |        |-sp     |        |        |
        #         |   -1   |    1   |        |   -sp  |        |        |
        #         |      -1|       1|        |     -sp|        |        |
        #         |=====================================================|
        #         |        |        | 1      |        |-sp     |        |
        #         |        |        |    1   |        |   -sp  |        |
        #         |        |        |       1|        |     -sp|        |
        #         |-----------------------------------------------------|
        #         |        |        |-1      | 1      |        |-sp     |
        #         |        |        |   -1   |    1   |        |   -sp  |
        #         |        |        |      -1|       1|        |     -sp|
        #         |=====================================================|
        #         |  J*sp  |  J*sp  |        |        |        |        |
        #         |-----------------------------------------------------|

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
        A_soft = w.zeros(self.height + self.prediction_horizon * self.num_of_continuous_joints, self.width)
        soft_expressions = self._sorter(self._A_soft)[0]
        controlled_joints = self.free_variables()
        t = time()
        J = w.jacobian(w.Matrix(soft_expressions), controlled_joints, order=1) * self.sample_period
        logging.loginfo(u'computed Jacobian in {:.5f}s'.format(time() - t))
        # Jd = w.jacobian(w.Matrix(soft_expressions), controlled_joints, order=2)
        # logging.loginfo(u'computed Jacobian dot in {:.5f}s'.format(time() - t))

        # position limits
        vertical_offset = number_of_joints * self.prediction_horizon
        for p in range(1, self.prediction_horizon+1):
            matrix_size = number_of_joints*p
            I = w.eye(matrix_size) * self.sample_period
            start = vertical_offset - matrix_size
            A_soft[start:vertical_offset, :matrix_size] += I

        # derivative links
        I = w.eye(number_of_joints * (self.order - 1) * self.prediction_horizon)
        block_size = number_of_joints * (self.order - 1) * self.prediction_horizon
        A_soft[vertical_offset:vertical_offset+block_size, :block_size] += I
        h_offset = number_of_joints * self.prediction_horizon
        A_soft[vertical_offset:vertical_offset+block_size, h_offset:h_offset + block_size] += -I * self.sample_period

        I_height = number_of_joints * (self.prediction_horizon - 1)
        I = -w.eye(I_height)
        offset_v = vertical_offset
        offset_h = 0
        for o in range(self.order - 1):
            offset_v += number_of_joints
            A_soft[offset_v:offset_v + I_height,offset_h:offset_h+I_height] += I
            offset_v += I_height
            offset_h += self.prediction_horizon*number_of_joints
        vertical_offset = vertical_offset + block_size

        # soft constraints
        A_soft[vertical_offset:, :(self.control_horizon)*number_of_joints] = w.hstack([J for _ in range(self.control_horizon)])

        number_of_soft_constraints = len(soft_expressions)
        I = w.eye(number_of_soft_constraints)
        A_soft[-number_of_soft_constraints:, -number_of_soft_constraints:] = I * self.sample_period/self.control_horizon

        continuous_joint_indices = [i for i, x in enumerate(self.free_variables()) if not self.joints[str(x)].has_position_limits()]
        indices_to_delete = []
        for o in range(self.prediction_horizon):
            for i in continuous_joint_indices:
                indices_to_delete.append(i+len(self.free_variables())*(o))
        A_soft.remove(indices_to_delete, [])
        return A_soft

    def A(self):
        return self.construct_A_soft()


class QPController(object):
    """
    Wraps around QPOases. Builds the required matrices from constraints.
    """

    def __init__(self, free_variables, constraints, debug_expressions, sample_period,
                 prediciton_horizon, control_horizon, solver_name):
        self.prediction_horizon = prediciton_horizon
        self.control_horizon = control_horizon
        self.sample_period = sample_period
        self.order = 3
        self.b = B(self.prediction_horizon)
        self.H = H(self.prediction_horizon, self.control_horizon)
        self.bA = BA(self.order, self.prediction_horizon)
        self.A = A(sample_period, self.prediction_horizon, self.control_horizon, self.order)
        self.order = 2

        l = [x.name for x in free_variables]
        duplicates = set([x for x in l if l.count(x) > 1])
        assert duplicates == set(), 'there are free variables with the same name: {}'.format(duplicates)
        l = [x.name for x in constraints]
        duplicates = set([x for x in l if l.count(x) > 1])
        assert duplicates == set(), 'there are constraints with the same name: {}'.format(duplicates)

        self.free_variables = list(sorted(free_variables, key=lambda x: x.name))
        self.constraints = list(sorted(constraints, key=lambda x: x.name))
        self.debug_expressions = debug_expressions

        if solver_name == u'gurobi':
            self.qp_solver = QPSolverGurubi()
        elif solver_name == u'qpoases':
            self.qp_solver = QPSolver()
        else:
            raise KeyError(u'Solver \'{}\' not supported'.format(solver_name))
        logging.loginfo(u'Using QP Solver \'{}\''.format(solver_name))
        # self.qp_solver = QPSolverOSPQ()

    def compile(self):
        self._construct_big_ass_M()
        self._compile_big_ass_M()

    def get_parameter_names(self):
        return self.compiled_big_ass_M.str_params

    @profile
    def _compile_big_ass_M(self):
        t = time()
        free_symbols = w.free_symbols(self.big_ass_M)
        self.compiled_big_ass_M = w.speed_up(self.big_ass_M,
                                             free_symbols)
        logging.loginfo(u'Compiled symbolic controller in {:.5f}s'.format(time() - t))
        self.compiled_debug_v = w.speed_up(self.debug_v, free_symbols)

    def __are_joint_limits_violated(self, p_lb, p_ub):
        violations = (p_ub - p_lb)[p_lb.data > p_ub.data]
        if len(violations) > 0:
            logging.logerr(u'The following joints are outside of their limits: \n {}'.format(violations))
            return True
        logging.loginfo(u'All joints are within limits')
        return False

    def __save_all(self, weights, A, lbA, ubA, lb, ub, xdot=None):
        if xdot is not None:
            print_pd_dfs([weights, A, lbA, ubA, lb, ub, xdot],
                         ['weights', 'A', 'lbA', 'ubA', 'lb', 'ub', 'xdot'])
        else:
            print_pd_dfs([weights, A, lbA, ubA, lb, ub],
                         ['weights', 'A', 'lbA', 'ubA', 'lb', 'ub'])

    def __is_nan_in_array(self, name, p_array):
        p_filtered = p_array.apply(lambda x: zip(x.index[x.isnull()].tolist(), x[x.isnull()]), 1)
        p_filtered = p_filtered[p_filtered.apply(lambda x: len(x)) > 0]
        if len(p_filtered) > 0:
            logging.logerr(u'{} has the following nans:'.format(name))
            self.__print_pandas_array(p_filtered)
            return True
        logging.loginfo(u'{} has no nans'.format(name))
        return False

    def __print_pandas_array(self, array):
        import pandas as pd
        if len(array) > 0:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(array)

    def _init_big_ass_M(self):
        """
        #         |---------------|
        #         |  A  | lba| uba|
        #         |---------------|
        #         |  H  | lb | ub |
        #         |---------------|
        """
        self.big_ass_M = w.zeros(self.A.height+self.H.height,
                                 self.A.width+2)
        self.debug_v = w.zeros(len(self.debug_expressions),1)

    def _set_A_soft(self, A_soft):
        self.big_ass_M[:self.A.height,:self.A.width] = A_soft

    def _set_weights(self, weights):
        self.big_ass_M[self.A.height:,:self.H.width] = w.diag(*weights)

    def _set_lb(self, lb):
        self.big_ass_M[self.A.height:, -2] = lb

    def _set_ub(self, ub):
        self.big_ass_M[self.A.height:, -1] = ub

    # def set_linear_weights(self, linear_weights):
    #     self.big_ass_M[self.H_vertical_start:self.H_vertical_stop, self.H_horizontal_stop + 2] = linear_weights

    def _set_lbA(self, lbA):
        self.big_ass_M[:self.A.height, self.A.width] = lbA

    def _set_ubA(self, ubA):
        self.big_ass_M[:self.A.height, self.A.width+1] = ubA


    @profile
    def _construct_big_ass_M(self):
        for free_variable in self.free_variables:  # type: FreeVariable
            self.H.add_joint_constraint(free_variable)
            self.b.add_joint_constraint(free_variable)
            self.bA.add_joint_constraint(free_variable)
            self.A.add_joint_constraint(free_variable)

        for free_variable in self.constraints:  # type: Constraint
            self.H.add_soft_constraint(free_variable)
            self.b.add_soft_constraint(free_variable)
            self.bA.add_constraint(free_variable)
            self.A.add_constraint(free_variable)

        logging.loginfo(u'constructing new controller with {} constraints and {} free variables...'.format(
            self.A.height, self.A.width))

        self._init_big_ass_M()

        self._set_weights(self.H.weights())
        self._set_A_soft(self.A.A())
        self._set_lbA(w.Matrix(self.bA.lbA()))
        self._set_ubA(w.Matrix(self.bA.ubA()))
        self._set_lb(w.Matrix(self.b.lb()))
        self._set_ub(w.Matrix(self.b.ub()))
        self.np_g = np.zeros(self.H.width)
        self.debug_names = list(sorted(self.debug_expressions.keys()))
        self.debug_v = w.Matrix([self.debug_expressions[name] for name in self.debug_names])

    def _eval_debug_exprs(self, subsitutions):
        return {name: value[0] for name, value in zip(self.debug_names, self.compiled_debug_v.call2(subsitutions))}

    @profile
    def get_cmd(self, substitutions):
        """
        Uses substitutions for each symbol to compute the next commands for each joint.
        :param substitutions:
        :type substitutions: list
        :return: joint name -> joint command
        :rtype: dict
        """
        np_big_ass_M = self.compiled_big_ass_M.call2(substitutions)
        self.np_H = np_big_ass_M[self.A.height:, :-2].copy()
        self.np_A = np_big_ass_M[:self.A.height, :self.A.width].copy()
        self.np_lb = np_big_ass_M[self.A.height:, -2].copy()
        self.np_ub = np_big_ass_M[self.A.height:, -1].copy()
        # np_g = np_big_ass_M[self.A.height:, -1].copy()
        self.np_lbA = np_big_ass_M[:self.A.height, -2].copy()
        self.np_ubA = np_big_ass_M[:self.A.height, -1].copy()

        try:
            self.xdot_full = self.qp_solver.solve(self.np_H, self.np_g, self.np_A, self.np_lb, self.np_ub, self.np_lbA,
                                                  self.np_ubA)
        except Exception as e:
            p_weights, p_A, p_lbA, p_ubA, p_lb, p_ub = self._debug_print(substitutions, actually_print=True)
            if isinstance(e, InfeasibleException):
                if self.__are_joint_limits_violated(p_lb, p_ub):
                    raise OutOfJointLimitsException(e)
                raise HardConstraintsViolatedException(e)
            # if isinstance(e, QPSolverException):
            # FIXME
            #     arrays = [(p_weights, u'H'),
            #               (p_A, u'A'),
            #               (p_lbA, u'lbA'),
            #               (p_ubA, u'ubA'),
            #               (p_lb, u'lb'),
            #               (p_ub, u'ub')]
            #     any_nan = False
            #     for a, name in arrays:
            #         any_nan |= self.__is_nan_in_array(name, a)
            #     if any_nan:
            #         raise e
            raise e
        if self.xdot_full is None:
            return None
        # TODO enable debug print in an elegant way, preferably without slowing anything down
        self._debug_print(substitutions, self.xdot_full)
        return self.split_xdot(self.xdot_full), self._eval_debug_exprs(substitutions)

    def split_xdot(self, xdot):
        split = []
        offset = len(self.free_variables)
        for derivative in range(self.order+1):
            split.append(OrderedDict((x.name, xdot[i+offset*self.prediction_horizon*derivative])
                                     for i, x in enumerate(self.free_variables)))
        return split

    def b_names(self):
        return self.b.names()

    def bA_names(self):
        return self.bA.names()

    def _viz_mpc(self, x, joint_name, state):
        start_pos = state[joint_name]
        ts = np.array([(i+1) * self.sample_period for i in range(self.prediction_horizon)])
        filtered_x = x.filter(like='/{}/'.format(joint_name), axis=0)
        velocities = filtered_x[:self.prediction_horizon].values
        accelerations = filtered_x[self.prediction_horizon:self.prediction_horizon * 2].values
        jerks = filtered_x[self.prediction_horizon * 2:self.prediction_horizon * 3].values
        positions = [start_pos]
        for x_ in velocities:
            positions.append(positions[-1] + x_ * self.sample_period)
        positions = positions[1:]

        f, axs = plt.subplots(4, sharex=True)
        axs[0].set_title('position')
        axs[0].plot(ts, positions, 'b')
        axs[0].grid()
        axs[1].set_title('velocity')
        axs[1].plot(ts, velocities, 'b')
        axs[1].grid()
        axs[2].set_title('acceleration')
        axs[2].plot(ts, accelerations, 'b')
        axs[2].grid()
        axs[3].set_title('jerk')
        axs[3].plot(ts, jerks, 'b')
        plt.grid()
        plt.show()

    @profile
    def _debug_print(self, substitutions, xdot_full=None, actually_print=False):
        import pandas as pd
        # bA_mask, b_mask = make_filter_masks(unfiltered_H, self.num_joint_constraints, self.num_hard_constraints)
        b_names = self.b_names()
        bA_names = self.bA_names()
        filtered_b_names = b_names#[b_mask]
        filtered_bA_names = bA_names#[bA_mask]
        filtered_H = self.H#[b_mask][:, b_mask]

        debug_exprs = self._eval_debug_exprs(substitutions)
        p_debug = pd.DataFrame.from_dict(debug_exprs, orient='index', columns=['data']).sort_index()

        p_lb = pd.DataFrame(self.np_lb, filtered_b_names, [u'data'], dtype=float)
        p_ub = pd.DataFrame(self.np_ub, filtered_b_names, [u'data'], dtype=float)
        p_g = pd.DataFrame(self.np_g, filtered_b_names, [u'data'], dtype=float)
        p_lbA = pd.DataFrame(self.np_lbA, filtered_bA_names, [u'data'], dtype=float)
        p_ubA = pd.DataFrame(self.np_ubA, filtered_bA_names, [u'data'], dtype=float)
        p_weights = pd.DataFrame(self.np_H.dot(np.ones(self.np_H.shape[0])), b_names, [u'data'],
                                 dtype=float).sort_index()
        if xdot_full is not None:
            p_xdot = pd.DataFrame(xdot_full, filtered_b_names, [u'data'], dtype=float)
            Ax = np.dot(self.np_A, xdot_full)
            xH = np.dot((xdot_full ** 2).T, self.np_H)
            p_xH = pd.DataFrame(xH, filtered_b_names, [u'data'], dtype=float)
            # p_xg = p_g * p_xdot
            xHx = np.dot(np.dot(xdot_full.T, self.np_H), xdot_full)
            num_non_slack = len(self.free_variables) * self.prediction_horizon * 3
            p_xdot2 = deepcopy(p_xdot)
            p_xdot2[num_non_slack:] = 0
            p_Ax = pd.DataFrame(Ax, filtered_bA_names, [u'data'], dtype=float)

            # x_soft = xdot_full[len(xdot_full) - num_slack:]
            # p_lbA_minus_x = pd.DataFrame(lbA - x_soft, filtered_bA_names, [u'data'], dtype=float).sort_index()
            # p_ubA_minus_x = pd.DataFrame(ubA - x_soft, filtered_bA_names, [u'data'], dtype=float).sort_index()
        else:
            p_xdot = None

        p_A = pd.DataFrame(self.np_A, filtered_bA_names, filtered_b_names, dtype=float)
        if xdot_full is not None:
            p_Ax2 = pd.DataFrame(p_A.dot(p_xdot2), filtered_bA_names, [u'data'], dtype=float)
        # if self.lbAs is None:
        #     self.lbAs = p_lbA
        # else:
        #     self.lbAs = self.lbAs.T.append(p_lbA.T, ignore_index=True).T
        # self.lbAs.T[[c for c in self.lbAs.T.columns if 'dist' in c]].plot()

        # self.save_all(p_weights, p_A, p_lbA, p_ubA, p_lb, p_ub, p_xdot)
        state = {k: v for k, v in zip(self.compiled_big_ass_M.str_params, substitutions)}
        # self._viz_mpc(p_xdot, 'world_robot_joint_state_head_pan_joint_position', state)
        # p_lbA[p_lbA != 0].abs().sort_values(by='data')
        return p_weights, p_A, p_lbA, p_ubA, p_lb, p_ub
