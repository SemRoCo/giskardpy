from collections import OrderedDict
from time import time
import matplotlib.pyplot as plt
import numpy as np

from giskardpy import logging, casadi_wrapper as w
from giskardpy.data_types import JointConstraint
from giskardpy.data_types import SoftConstraint
from giskardpy.exceptions import QPSolverException, InfeasibleException, OutOfJointLimitsException, \
    HardConstraintsViolatedException
from giskardpy.qp_solver import QPSolver
from giskardpy.qp_solver_gurubi import QPSolverGurubi
from giskardpy.qp_solver_osqp import QPSolverOSPQ
from giskardpy.utils import make_filter_masks, create_path


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
    def __init__(self, prediction_horizon, control_horizon):
        self.__j_weights_v = {}
        self.__j_weights_a = {}
        self.__j_weights_j = {}
        self.__s_weights_v = {}
        self.__s_weights_a = {}
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        self.height = 0

    @property
    def width(self):
        return self.height

    def add_joint_constraint(self, name, constraint):
        """
        :type name: str
        :type constraint: JointConstraint
        :return:
        """
        self.__j_weights_v[name + '/v'] = constraint.weight_v
        self.__j_weights_a[name + '/a'] = constraint.weight_a
        self.__j_weights_j[name + '/j'] = constraint.weight_j
        self.height += self.prediction_horizon*3

    def add_soft_constraint(self, name, constraint):
        """
        :type name: str
        :type constraint: SoftConstraint
        :return:
        """
        # self.__s_weights_v[name + '/v'] = constraint.weight_v * (self.prediction_horizon/self.control_horizon)**2
        self.__s_weights_v[name + '/v'] = constraint.weight_v #* (self.control_horizon*0.05)**2
        self.__s_weights_a[name + '/a'] = constraint.weight_a
        self.height += 1

    def weights(self):
        vel_weights = {}
        f = lambda x: 0.0001*x
        for t in range(self.prediction_horizon):
            for name, weight in self.__j_weights_v.items():
                vel_weights['t{:03d}/{}'.format(t, name)] = weight + f(t)
        acc_weights = {}
        for t in range(self.prediction_horizon):
            for name, weight in self.__j_weights_a.items():
                acc_weights['t{:03d}/{}'.format(t, name)] = weight + f(t)
        jerk_weights = {}
        for t in range(self.prediction_horizon):
            for name, weight in self.__j_weights_j.items():
                jerk_weights['t{:03d}/{}'.format(t, name)] = weight + f(t)
        return self._sorter(vel_weights,
                            acc_weights,
                            jerk_weights,
                            self.__s_weights_v)[0]


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
        self._s_lb_a = {}
        self._s_ub_a = {}
        self.prediction_horizon = prediction_horizon

    def add_joint_constraint(self, name, constraint):
        """
        :type name: str
        :type constraint: JointConstraint
        """
        self._j_lb_v[name + '/v'] = constraint.lower_v
        self._j_lb_a[name + '/a'] = constraint.lower_a
        self._j_lb_j[name + '/j'] = constraint.lower_j
        self._j_ub_v[name + '/v'] = constraint.upper_v
        self._j_ub_a[name + '/a'] = constraint.upper_a
        self._j_ub_j[name + '/j'] = constraint.upper_j

    def add_soft_constraint(self, name, constraint):
        """
        :type name: str
        :type constraint: SoftConstraint
        :return:
        """
        self._s_lb_v[name + '/v'] = constraint.lower_slack_limit_v
        self._s_lb_a[name + '/a'] = constraint.lower_slack_limit_a
        self._s_ub_v[name + '/v'] = constraint.upper_slack_limit_v
        self._s_ub_a[name + '/a'] = constraint.upper_slack_limit_a

    def blow_up(self, d):
        result = {}
        for t in range(self.prediction_horizon):
            for name, value in d.items():
                if t == self.prediction_horizon -1 and self.prediction_horizon >= 5:
                    result['t{:03d}/{}'.format(t, name)] = 0
                else:
                    result['t{:03d}/{}'.format(t, name)] = value
        return result

    def lb(self):
        return self._sorter(self.blow_up(self._j_lb_v),
                            self.blow_up(self._j_lb_a),
                            self.blow_up(self._j_lb_j),
                            self._s_lb_v)[0]

    def ub(self):
        return self._sorter(self.blow_up(self._j_ub_v),
                            self.blow_up(self._j_ub_a),
                            self.blow_up(self._j_ub_j),
                            self._s_ub_v)[0]

    def names(self):
        return self._sorter(self.blow_up(self._j_ub_v),
                            self.blow_up(self._j_ub_a),
                            self.blow_up(self._j_ub_j),
                            self._s_ub_v)[1]


class BA(Parent):
    def __init__(self, order, prediction_horizon):
        self._lbA_v = {}
        self._ubA_v = {}
        self._lbA_a = {}
        self._ubA_a = {}
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

    def add_joint_constraint(self, name, constraint):
        """
        :type name: str
        :type constraint: JointConstraint
        """
        if constraint.lower_p is not None:
            self._pos_limits_lba[name + '/pos_limit'] = constraint.lower_p - constraint.joint_symbol
            self._pos_limits_uba[name + '/pos_limit'] = constraint.upper_p - constraint.joint_symbol
            self._pos_limits_lba2[name + '/pos_limit'] = constraint.lower_p - constraint.joint_symbol
            self._pos_limits_uba2[name + '/pos_limit'] = constraint.upper_p - constraint.joint_symbol
        self._j_lbA_a_link[name + '/last_vel'] = constraint.joint_velocity_symbol
        self._j_ubA_a_link[name + '/last_vel'] = constraint.joint_velocity_symbol
        self._j_lbA_j_link[name + '/last_acc'] = constraint.joint_acceleration_symbol
        self._j_ubA_j_link[name + '/last_acc'] = constraint.joint_acceleration_symbol
        self._joint_names.append(name)

    def add_soft_constraint(self, name, constraint):
        """
        :type name: str
        :type constraint: SoftConstraint
        """
        self._lbA_v[name + '/v'] = constraint.lbA_v
        self._lbA_a[name + '/a'] = constraint.lbA_a
        self._ubA_v[name + '/v'] = constraint.ubA_v
        self._ubA_a[name + '/a'] = constraint.ubA_a

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
        self.number_of_continuous = 0

    @property
    def number_of_joints(self):
        return len(self.controlled_joint_symbols())

    @property
    def height(self):
        return self.prediction_horizon * (self.number_of_joints - self.number_of_continuous) + \
               self.number_of_joints * self.prediction_horizon * (self.order - 1) + \
               len(self._A_soft)

    @property
    def width(self):
        return self.number_of_joints * self.prediction_horizon * self.order + len(self._A_soft)

    def add_joint_constraint(self, name, constraint):
        """
        :type name: str
        :type constraint: JointConstraint
        :return:
        """
        self._A_joint[name] = constraint.joint_symbol
        # if constraint.lower_p is None:
        #     self.number_of_continuous += 1

    def add_hard_constraint(self, name, constraint):
        self._A_hard[name] = constraint.expression

    def add_soft_constraint(self, name, constraint):
        """
        :type name:
        :type constraint: SoftConstraint
        """
        self._A_soft[name] = constraint.expression

    def controlled_joint_symbols(self):
        return self._sorter(self._A_joint)[0]

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
        A_soft = w.zeros(self.height, self.width)
        soft_expressions = self._sorter(self._A_soft)[0]
        controlled_joints = self.controlled_joint_symbols()
        t = time()
        J = w.jacobian(w.Matrix(soft_expressions), controlled_joints, order=1)
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
        for c in range(self.control_horizon):
            A_soft[vertical_offset:, c*number_of_joints:(c+1)*number_of_joints] = J * self.sample_period

        number_of_soft_constraints = len(soft_expressions)
        I = w.eye(number_of_soft_constraints)
        A_soft[-number_of_soft_constraints:, -number_of_soft_constraints:] = I * self.sample_period/self.control_horizon

        return A_soft

    def A(self):
        return self.construct_A_soft()


class QProblemBuilder(object):
    """
    Wraps around QPOases. Builds the required matrices from constraints.
    """

    def __init__(self, joint_constraints_dict, hard_constraints_dict, soft_constraints_dict, sample_period,
                 prediciton_horizon, control_horizon, path_to_functions=''):
        """
        :type joint_constraints_dict: dict
        :type hard_constraints_dict: dict
        :type soft_constraints_dict: dict
        :type controlled_joint_symbols: list
        :param path_to_functions: location where the compiled functions can be safed.
        :type path_to_functions: str
        """
        self.prediction_horizon = prediciton_horizon
        self.control_horizon = control_horizon
        self.sample_period = sample_period
        self.order = 3
        self.b = B(self.prediction_horizon)
        self.H = H(self.prediction_horizon, self.control_horizon)
        self.bA = BA(self.order, self.prediction_horizon)
        self.A = A(sample_period, self.prediction_horizon, self.control_horizon, self.order)
        self.order = 2
        self.path_to_functions = path_to_functions
        self.joint_constraints_dict = joint_constraints_dict
        self.hard_constraints_dict = hard_constraints_dict
        self.soft_constraints_dict = soft_constraints_dict
        self.construct_big_ass_M()
        self.compile_big_ass_M()

        self.num_hard_constraints = len(self.hard_constraints_dict)
        self.num_joint_constraints = len(self.joint_constraints_dict)
        self.num_soft_constraints = len(self.soft_constraints_dict)

        self.qp_solver = QPSolver()
        # self.qp_solver = QPSolverGurubi()
        # self.qp_solver = QPSolverOSPQ()
        self.lbAs = None  # for debugging purposes

    def get_joint_symbols(self):
        joint_symbols = []
        for constraint_name, constraint in self.joint_constraints_dict.items():  # type: (str, JointConstraint)
            joint_symbols.append(constraint.joint_symbol)
        return sorted(joint_symbols)

    def get_expr(self):
        return self.compiled_big_ass_M.str_params

    @profile
    def compile_big_ass_M(self):
        t = time()
        self.free_symbols = w.free_symbols(self.big_ass_M)
        self.compiled_big_ass_M = w.speed_up(self.big_ass_M,
                                             self.free_symbols)
        logging.loginfo(u'compiled symbolic expressions in {:.5f}s'.format(time() - t))

    def are_joint_limits_violated(self, p_lb, p_ub):
        violations = (p_ub - p_lb)[p_lb.data > p_ub.data]
        if len(violations) > 0:
            logging.logerr(u'The following joints are outside of their limits: \n {}'.format(violations))
            return True
        logging.loginfo(u'All joints are within limits')
        return False

    def save_all(self, weights, A, lbA, ubA, lb, ub, xdot=None):
        if xdot is not None:
            print_pd_dfs([weights, A, lbA, ubA, lb, ub, xdot],
                         ['weights', 'A', 'lbA', 'ubA', 'lb', 'ub', 'xdot'])
        else:
            print_pd_dfs([weights, A, lbA, ubA, lb, ub],
                         ['weights', 'A', 'lbA', 'ubA', 'lb', 'ub'])

    def is_nan_in_array(self, name, p_array):
        p_filtered = p_array.apply(lambda x: zip(x.index[x.isnull()].tolist(), x[x.isnull()]), 1)
        p_filtered = p_filtered[p_filtered.apply(lambda x: len(x)) > 0]
        if len(p_filtered) > 0:
            logging.logerr(u'{} has the following nans:'.format(name))
            self.print_pandas_array(p_filtered)
            return True
        logging.loginfo(u'{} has no nans'.format(name))
        return False

    def print_pandas_array(self, array):
        import pandas as pd
        if len(array) > 0:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(array)

    def init_big_ass_M(self):
        """
        #         |   t1   |   t2   |   t3   |   t1   |   t2   |   t3   |   t1   |   t2   |   t3   |        |
        #         |v1 v2 vn|v1 v2 vn|v1 v2 vn|a1 a2 an|a1 a2 an|a1 a2 an|j1 j2 jn|j1 j2 jn|j1 j2 jn|sv|sa|sj|
        #         |--------------------------------------------------------------------------------|--------|-----------|
        #         | 1      |        |        |        |        |        |        |        |        |        |lp-p0|up-p0|
        #         | 1      | 1      | 1      |        |        |        |        |        |        |        |lp-p0|up-p0|
        #         |--------------------------------------------------------------------------------|--------|
        #         |    1   |        |        |        |        |        |        |        |        |        |
        #         |    1   |    1   |    1   |        |        |        |        |        |        |        |
        #         |--------------------------------------------------------------------------------|--------|
        #         |       1|        |        |        |        |        |        |        |        |        |
        #         |       1|       1|       1|        |        |        |        |        |        |        |
        #         |================================================================================|========|===========|
        #         | 1      |        |        |-1      |        |        |        |        |        |        |joint1 v0
        #         |-1      | 1      |        |        |-1      |        |        |        |        |        |
        #         |        |-1      | 1      |        |        |-1      |        |        |        |        |
        #         |--------------------------------------------------------------------------------|        |
        #         |    1   |        |        |   -1   |        |        |        |        |        |        |joint2 v0
        #         |   -1   |    1   |        |        |   -1   |        |        |        |        |        |
        #         |        |   -1   |    1   |        |        |   -1   |        |        |        |        |
        #         |--------------------------------------------------------------------------------|        |
        #         |       1|        |        |      -1|        |        |        |        |        |        |joint3 v0
        #         |      -1|       1|        |        |      -1|        |        |        |        |        |
        #         |        |      -1|       1|        |        |      -1|        |        |        |        |
        #         |================================================================================|        |
        #         |        |        |        | 1      |        |        |-1      |        |        |        |joint1 a0
        #         |        |        |        |-1      | 1      |        |        |-1      |        |        |
        #         |        |        |        |        |-1      | 1      |        |        |-1      |        |
        #         |--------------------------------------------------------------------------------|        |
        #         |        |        |        |    1   |        |        |   -1   |        |        |        |joint2 a0
        #         |        |        |        |   -1   |    1   |        |        |   -1   |        |        |
        #         |        |        |        |        |   -1   |    1   |        |        |   -1   |        |
        #         |--------------------------------------------------------------------------------|        |
        #         |        |        |        |       1|        |        |      -1|        |        |        |joint3 a0
        #         |        |        |        |      -1|       1|        |        |      -1|        |        |
        #         |        |        |        |        |      -1|       1|        |        |      -1|        |
        #         |================================================================================|        |
        #         |   J    |   J    |   J    |        |        |        |        |        |        | I|  |  |v lba|v uba|
        #         |-----------------------------------------------------------------------------------------|
        #         |   Jd   |   Jd   |   Jd   |   J    |   J    |   J    |        |        |        |  | I|  |a lba|a uba|
        #         |-----------------------------------------------------------------------------------------------------|
        #         |   Jdd  |   Jdd  |   Jdd  |   Jd   |   Jd   |   Jd   |   J    |   J    |   J    |  |  | I|j lba|j uba|
        #         |=====================================================================================================|
        #         |wv11    |        |        |        |        |        |        |        |        |        |joint1 vl  |
        #         |  wv21  |        |        |        |        |        |        |        |        |        |joint2 vl  |
        #         |    wv31|        |        |        |        |        |        |        |        |        |joint3 vl  |
        #         |        |wv12    |        |        |        |        |        |        |        |        |joint1 vl  |
        #         |        |  wv22  |        |        |        |        |        |        |        |        |           |
        #         |        |    wv32|        |        |        |        |        |        |        |        |           |
        #         |        |        |wv13    |        |        |        |        |        |        |        |           |
        #         |        |        |  wv23  |        |        |        |        |        |        |        |           |
        #         |        |        |    wv33|        |        |        |        |        |        |        |joint3 vl  |
        #         |-----------------------------------------------------------------------------------------------------|
        #         |        |        |        |wa11    |        |        |        |        |        |        |joint1 al  |
        #         |        |        |        |  wa21  |        |        |        |        |        |        |joint2 al  |
        #         |        |        |        |    wa31|        |        |        |        |        |        |joint3 al  |
        #         |        |        |        |        |wa12    |        |        |        |        |        |           |
        #         |        |        |        |        |  wa22  |        |        |        |        |        |           |
        #         |        |        |        |        |    wa32|        |        |        |        |        |           |
        #         |        |        |        |        |        |wa13    |        |        |        |        |           |
        #         |        |        |        |        |        |  wa23  |        |        |        |        |           |
        #         |        |        |        |        |        |    wa33|        |        |        |        |joint3 al  |
        #         |-----------------------------------------------------------------------------------------------------|
        #         |        |        |        |        |        |        |wj11    |        |        |        |joint1 jl  |
        #         |        |        |        |        |        |        |  wj21  |        |        |        |joint2 jl  |
        #         |        |        |        |        |        |        |    wj31|        |        |        |joint3 jl  |
        #         |        |        |        |        |        |        |        |wj12    |        |        |           |
        #         |        |        |        |        |        |        |        |  wj22  |        |        |           |
        #         |        |        |        |        |        |        |        |    wj32|        |        |           |
        #         |        |        |        |        |        |        |        |        |wj13    |        |           |
        #         |        |        |        |        |        |        |        |        |  wj23  |        |           |
        #         |        |        |        |        |        |        |        |        |    wj33|        |joint3 jl  |
        #         |-----------------------------------------------------------------------------------------------------|
        #         |        |        |        |        |        |        |        |        |        |wsv     |sv lb| ub  |
        #         |        |        |        |        |        |        |        |        |        |  wsa   |sa lb| ub  |
        #         |        |        |        |        |        |        |        |        |        |     wsj|sj lb| ub  |
        #         |-----------------------------------------------------------------------------------------------------|
        #

        #         |---------------|
        #         |  A  | lba| uba|
        #         |---------------|
        #         |  H  | lb | ub |
        #         |---------------|
        """
        self.big_ass_M = w.zeros(self.A.height+self.H.height,
                                 self.A.width+2)

    def set_A_soft(self, A_soft):
        self.big_ass_M[:self.A.height,:self.A.width] = A_soft

    def set_weights(self, weights):
        self.big_ass_M[self.A.height:,:self.H.width] = w.diag(*weights)

    def set_lb(self, lb):
        self.big_ass_M[self.A.height:, -2] = lb

    def set_ub(self, ub):
        self.big_ass_M[self.A.height:, -1] = ub

    # def set_linear_weights(self, linear_weights):
    #     self.big_ass_M[self.H_vertical_start:self.H_vertical_stop, self.H_horizontal_stop + 2] = linear_weights

    def set_lbA(self, lbA):
        self.big_ass_M[:self.A.height, self.A.width] = lbA

    def set_ubA(self, ubA):
        self.big_ass_M[:self.A.height, self.A.width+1] = ubA

    # def set_A_hard(self, hard_expressions):
    #     for i, row in enumerate(hard_expressions):
    #         self.big_ass_M[i, :self.j * 2] = row

    def filter_zero_weight_constraints(self, H, A, lb, ub, lbA, ubA, g):
        # bA_mask, b_mask = make_filter_masks(H, self.num_joint_constraints, self.num_hard_constraints)
        # A = A[bA_mask][:, b_mask].copy()
        # lbA = lbA[bA_mask]
        # ubA = ubA[bA_mask]
        # lb = lb[b_mask]
        # ub = ub[b_mask]
        # g = g[b_mask]
        # H = H[b_mask][:, b_mask]
        return H, A, lb, ub, lbA, ubA, g

    @profile
    def construct_big_ass_M(self):
        for i, ((_, constraint_name), constraint) in enumerate(
                self.joint_constraints_dict.items()):  # type: (str, JointConstraint)
            self.H.add_joint_constraint(constraint_name, constraint)
            self.b.add_joint_constraint(constraint_name, constraint)
            self.bA.add_joint_constraint(constraint_name, constraint)
            self.A.add_joint_constraint(constraint_name, constraint)

        for constraint_name, constraint in self.soft_constraints_dict.items():  # type: (str, SoftConstraint)
            self.H.add_soft_constraint(constraint_name, constraint)
            self.b.add_soft_constraint(constraint_name, constraint)
            self.bA.add_soft_constraint(constraint_name, constraint)
            self.A.add_soft_constraint(constraint_name, constraint)

        logging.loginfo(u'constructing new controller with {} soft constraints...'.format(len(self.bA_names())))
        # assert len(self.hard_constraints_dict) == 0, 'hard constraints are not supported anymore'
        self.h = len(self.hard_constraints_dict)
        self.s = len(self.soft_constraints_dict)
        self.j = len(self.joint_constraints_dict)

        self.init_big_ass_M()

        self.set_weights(self.H.weights())

        # self.set_A_hard(hard_expressions)
        # self.construct_A_soft(soft_expressions)
        self.set_A_soft(self.A.A())

        self.set_lbA(w.Matrix(self.bA.lbA()))
        self.set_ubA(w.Matrix(self.bA.ubA()))
        self.set_lb(w.Matrix(self.b.lb()))
        self.set_ub(w.Matrix(self.b.ub()))
        self.np_g = np.zeros(self.H.width)
        # self.set_linear_weights(w.Matrix([0] * len(self.H.weights())))

    @profile
    def get_cmd(self, substitutions, nWSR=None):
        """
        Uses substitutions for each symbol to compute the next commands for each joint.
        :param substitutions:
        :type substitutions: list
        :return: joint name -> joint command
        :rtype: dict
        """
        np_big_ass_M = self.compiled_big_ass_M.call2(substitutions)
        np_H = np_big_ass_M[self.A.height:, :-2].copy()
        np_A = np_big_ass_M[:self.A.height, :self.A.width].copy()
        np_lb = np_big_ass_M[self.A.height:, -2].copy()
        np_ub = np_big_ass_M[self.A.height:, -1].copy()
        # np_g = np_big_ass_M[self.A.height:, -1].copy()
        np_lbA = np_big_ass_M[:self.A.height, -2].copy()
        np_ubA = np_big_ass_M[:self.A.height, -1].copy()
        H, A, lb, ub, lbA, ubA, g = self.filter_zero_weight_constraints(np_H, np_A, np_lb, np_ub, np_lbA, np_ubA, self.np_g)
        # self.debug_print(np_H, A, lb, ub, lbA, ubA, g)
        try:
            xdot_full = self.qp_solver.solve(H, g, A, lb, ub, lbA, ubA, nWSR)
        except Exception as e:
            p_weights, p_A, p_lbA, p_ubA, p_lb, p_ub = self.debug_print(np_H, A, lb, ub, lbA, ubA, g,
                                                                        substitutions,
                                                                        actually_print=True)
            if isinstance(e, InfeasibleException):
                if self.are_joint_limits_violated(p_lb, p_ub):
                    raise OutOfJointLimitsException(e)
                raise HardConstraintsViolatedException(e)
            if isinstance(e, QPSolverException):
                arrays = [(p_weights, u'H'),
                          (p_A, u'A'),
                          (p_lbA, u'lbA'),
                          (p_ubA, u'ubA'),
                          (p_lb, u'lb'),
                          (p_ub, u'ub')]
                any_nan = False
                for a, name in arrays:
                    any_nan |= self.is_nan_in_array(name, a)
                if any_nan:
                    raise e
            raise e
        if xdot_full is None:
            return None
        # TODO enable debug print in an elegant way, preferably without slowing anything down
        self.debug_print(np_H, A, lb, ub, lbA, ubA, g, substitutions, xdot_full)
        velocity = OrderedDict(
            (observable, xdot_full[i]) for i, observable in enumerate(self.joint_names()))
        acceleration = OrderedDict(
            (observable, xdot_full[i + len(self.joint_names())*self.prediction_horizon]) for i, observable in
            enumerate(self.joint_names()))
        jerk = OrderedDict(
            (observable, xdot_full[i + len(self.joint_names()) * self.prediction_horizon * 2]) for i, observable in
            enumerate(self.joint_names()))
        return velocity, acceleration, jerk, np_H, np_A, np_lb, np_ub, np_lbA, np_ubA, xdot_full

    def joint_names(self):
        return self.A.controlled_joint_symbols()

    def b_names(self):
        return self.b.names()

    def bA_names(self):
        return self.bA.names()

    def viz_mpc(self, x, joint_name, start_pos=0):
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

    def debug_print(self, unfiltered_H, A, lb, ub, lbA, ubA, g, substitutions, xdot_full=None, actually_print=False):
        import pandas as pd
        # bA_mask, b_mask = make_filter_masks(unfiltered_H, self.num_joint_constraints, self.num_hard_constraints)
        state = {k:v for k,v in zip(self.compiled_big_ass_M.str_params, substitutions)}
        b_names = np.array(self.b_names())
        bA_names = np.array(self.bA_names())
        filtered_b_names = b_names#[b_mask]
        filtered_bA_names = bA_names#[bA_mask]
        filtered_H = unfiltered_H#[b_mask][:, b_mask]

        p_lb = pd.DataFrame(lb, filtered_b_names, [u'data'], dtype=float)
        p_ub = pd.DataFrame(ub, filtered_b_names, [u'data'], dtype=float)
        p_g = pd.DataFrame(g, filtered_b_names, [u'data'], dtype=float)
        p_lbA = pd.DataFrame(lbA, filtered_bA_names, [u'data'], dtype=float)
        p_ubA = pd.DataFrame(ubA, filtered_bA_names, [u'data'], dtype=float)
        p_weights = pd.DataFrame(unfiltered_H.dot(np.ones(unfiltered_H.shape[0])), b_names, [u'data'],
                                 dtype=float).sort_index()
        if xdot_full is not None:
            p_xdot = pd.DataFrame(xdot_full, filtered_b_names, [u'data'], dtype=float)
            Ax = np.dot(A, xdot_full)
            p_Ax = pd.DataFrame(Ax, filtered_bA_names, [u'data'], dtype=float)
            xH = np.dot((xdot_full ** 2).T, filtered_H)
            p_xH = pd.DataFrame(xH, filtered_b_names, [u'data'], dtype=float)
            p_xg = p_g * p_xdot
            xHx = np.dot(np.dot(xdot_full.T, filtered_H), xdot_full)
            # x_soft = xdot_full[len(xdot_full) - num_slack:]
            # p_lbA_minus_x = pd.DataFrame(lbA - x_soft, filtered_bA_names, [u'data'], dtype=float).sort_index()
            # p_ubA_minus_x = pd.DataFrame(ubA - x_soft, filtered_bA_names, [u'data'], dtype=float).sort_index()
        else:
            p_xdot = None

        p_A = pd.DataFrame(A, filtered_bA_names, filtered_b_names, dtype=float)
        # if self.lbAs is None:
        #     self.lbAs = p_lbA
        # else:
        #     self.lbAs = self.lbAs.T.append(p_lbA.T, ignore_index=True).T
        # self.lbAs.T[[c for c in self.lbAs.T.columns if 'dist' in c]].plot()

        # self.save_all(p_weights, p_A, p_lbA, p_ubA, p_lb, p_ub, p_xdot)
        # self.viz_mpc(p_xdot, 'j2', start_pos=state['j2'])
        return p_weights, p_A, p_lbA, p_ubA, p_lb, p_ub
