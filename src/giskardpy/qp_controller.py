from collections import OrderedDict, defaultdict
from copy import deepcopy
from time import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from giskardpy import logging, casadi_wrapper as w
from giskardpy.data_types import FreeVariable, Constraint
from giskardpy.exceptions import InfeasibleException, OutOfJointLimitsException, \
    HardConstraintsViolatedException
from giskardpy.qp_solver import QPSolver
from giskardpy.qp_solver_gurubi import QPSolverGurubi
from giskardpy.utils import create_path, memoize


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
    def __init__(self, sample_period, prediction_horizon):
        self.prediction_horizon = prediction_horizon
        self.sample_period = sample_period

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
    def __init__(self, free_variables, constraints, sample_period, prediction_horizon):
        super(H, self).__init__(sample_period, prediction_horizon)
        self.free_variables = free_variables
        self.constraints = constraints  # type: list[Constraint]
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
        return len(self.free_variables) * 3 * self.prediction_horizon

    def number_of_constraint_vel_variables(self):
        h = 0
        for c in self.constraints:
            h += c.control_horizon
        return h

    def number_of_contraint_error_variables(self):
        return len(self.constraints)

    def weights(self):
        vel_weights = {}
        for t in range(self.prediction_horizon):
            for v in self.free_variables:  # type: FreeVariable
                vel_weights['t{}/{}/v'.format(t, v.name)] = v.velocity_horizon_function(v.quadratic_velocity_weight, t)
        acc_weights = {}
        for t in range(self.prediction_horizon):
            for v in self.free_variables:  # type: FreeVariable
                acc_weights['t{}/{}/a'.format(t, v.name)] = v.acceleration_horizon_function(
                    v.quadratic_acceleration_weight, t)
        jerk_weights = {}
        for t in range(self.prediction_horizon):
            for v in self.free_variables:  # type: FreeVariable
                jerk_weights['t{}/{}/j'.format(t, v.name)] = v.jerk_horizon_function(v.quadratic_jerk_weight, t)
        slack_weights = {}
        for t in range(self.prediction_horizon):
            for c in self.constraints:  # type: Constraint
                if t < c.control_horizon:
                    slack_weights['t{}/{}'.format(t, c.name)] = c.horizon_function(c.quadratic_velocity_weight, t)
        error_slack_weights = {'{}/error'.format(c.name): c.quadratic_error_weight for c in self.constraints}
        return self._sorter(vel_weights,
                            acc_weights,
                            jerk_weights,
                            slack_weights,
                            error_slack_weights)[0]
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
                    d[start_id + i].append(start_id1+c)
                    c += 1
        return {k: np.array(v) for k, v in d.items()}



class B(Parent):
    def __init__(self, free_variables, constraints, sample_period, prediction_horizon):
        super(B, self).__init__(sample_period, prediction_horizon)
        self.free_variables = free_variables  # type: list[FreeVariable]
        self.constraints = constraints  # type: list[Constraint]
        self.no_limits = 1e4

    def blow_up(self, d, end_with_zero=False):
        def f(value, t):
            if t == self.prediction_horizon - 1 and self.prediction_horizon > 1 and end_with_zero:
                return 0
            return value

        return super(B, self).blow_up(d, self.prediction_horizon, f)

    def get_lower_joint_velocity_limits(self):
        lb_v = {'{}/v'.format(v.name): v.lower_velocity_limit for v in self.free_variables}
        return self.blow_up(lb_v, True)

    def get_upper_joint_velocity_limits(self):
        lb_v = {'{}/v'.format(v.name): v.upper_velocity_limit for v in self.free_variables}
        return self.blow_up(lb_v, True)

    def get_lower_joint_acceleration_limits(self):
        lb_a = {'{}/a'.format(v.name): v.lower_acceleration_limit for v in self.free_variables}
        return self.blow_up(lb_a, True)

    def get_upper_joint_acceleration_limits(self):
        lb_a = {'{}/a'.format(v.name): v.upper_acceleration_limit for v in self.free_variables}
        return self.blow_up(lb_a, True)

    def get_lower_joint_jerk_limits(self):
        lb_j = {'{}/j'.format(v.name): v.lower_jerk_limit for v in self.free_variables}
        return self.blow_up(lb_j, False)

    def get_upper_joint_jerk_limits(self):
        lb_j = {'{}/j'.format(v.name): v.upper_jerk_limit for v in self.free_variables}
        return self.blow_up(lb_j, False)

    def get_lower_slack_limits(self):
        result = {}
        for t in range(self.prediction_horizon):
            for c in self.constraints:
                if t < c.control_horizon:
                    result['t{:03d}/{}'.format(t, c.name)] = -1e3
                    # result['t{:03d}/{}'.format(t, c.name)] = 0
        return result

    def get_upper_slack_limits(self):
        result = {}
        for t in range(self.prediction_horizon):
            for c in self.constraints:
                if t < c.control_horizon:
                    result['t{:03d}/{}'.format(t, c.name)] = 1e3
                    # result['t{:03d}/{}'.format(t, c.name)] = 0
        return result

    def get_lower_error_slack_limits(self):
        # TODO separate slack limits for speed and error
        return {'{}/error'.format(c.name):c.lower_slack_limit for c in self.constraints}

    def get_upper_error_slack_limits(self):
        # TODO separate slack limits for speed and error
        return {'{}/error'.format(c.name):c.upper_slack_limit for c in self.constraints}


    def lb(self):
        return self._sorter(self.get_lower_joint_velocity_limits(),
                            self.get_lower_joint_acceleration_limits(),
                            self.get_lower_joint_jerk_limits(),
                            self.get_lower_slack_limits(),
                            self.get_lower_error_slack_limits())[0]

    def ub(self):
        return self._sorter(self.get_upper_joint_velocity_limits(),
                            self.get_upper_joint_acceleration_limits(),
                            self.get_upper_joint_jerk_limits(),
                            self.get_upper_slack_limits(),
                            self.get_upper_error_slack_limits())[0]

    def names(self):
        return self._sorter(self.get_upper_joint_velocity_limits(),
                            self.get_upper_joint_acceleration_limits(),
                            self.get_upper_joint_jerk_limits(),
                            self.get_upper_slack_limits(),
                            self.get_upper_error_slack_limits())[1]


class BA(Parent):
    def __init__(self, free_variables, constraints, sample_period, order, prediction_horizon):
        super(BA, self).__init__(sample_period, prediction_horizon)
        self.free_variables = free_variables
        self.constraints = constraints
        self.order = order
        self.round_to = 5
        self.round_to2 = 10

    @memoize
    def get_lower_position_limits(self):
        d = {'{}/p_limit'.format(v.name): w.round_up(v.lower_position_limit - v.position_symbol, self.round_to2)
             for v in self.free_variables if v.has_position_limits()}
        return self.blow_up(d, self.prediction_horizon)

    @memoize
    def get_upper_position_limits(self):
        d = {'{}/p_limit'.format(v.name): w.round_down(v.upper_position_limit - v.position_symbol, self.round_to2)
             for v in self.free_variables if v.has_position_limits()}
        return self.blow_up(d, self.prediction_horizon)

    @memoize
    def get_last_velocities(self, down=True):
        if down:
            return {'{}/last_v'.format(v.name): w.round_down(v.velocity_symbol, self.round_to)
                    for v in self.free_variables}
        else:
            return {'{}/last_v'.format(v.name): w.round_up(v.velocity_symbol, self.round_to)
                    for v in self.free_variables}

    @memoize
    def get_last_accelerations(self, down=True):
        if down:
            return {'{}/last_a'.format(v.name): w.round_down(v.acceleration_symbol, self.round_to)
                    for v in self.free_variables}
        else:
            return {'{}/last_a'.format(v.name): w.round_up(v.acceleration_symbol, self.round_to)
                    for v in self.free_variables}

    @memoize
    def get_derivative_link(self, infix):
        return self.blow_up({'{}/{}/link'.format(infix, v.name): 0 for v in self.free_variables}, self.prediction_horizon - 1)

    def get_lower_constraint_velocities(self):
        result = {}
        for t in range(self.prediction_horizon):
            for c in self.constraints:
                if t < c.control_horizon:
                    # if t == 0:
                    #     result['t{:03d}/{}'.format(t, c.name)] = w.limit(c.lower_position_limit,
                    #                                                      c.lower_velocity_limit * self.sample_period,
                    #                                                      c.upper_velocity_limit * self.sample_period)
                    # else:
                        result['t{:03d}/{}'.format(t, c.name)] = c.lower_velocity_limit * self.sample_period
        return result

    def get_upper_constraint_velocities(self):
        result = {}
        for t in range(self.prediction_horizon):
            for c in self.constraints:
                if t < c.control_horizon:
                    # if t == 0:
                    #     result['t{:03d}/{}'.format(t, c.name)] = w.limit(c.upper_position_limit,
                    #                                                      c.lower_velocity_limit * self.sample_period,
                    #                                                      c.upper_velocity_limit * self.sample_period)
                    # else:
                        result['t{:03d}/{}'.format(t, c.name)] = c.upper_velocity_limit * self.sample_period
        return result

    @memoize
    def get_lower_constraint_error(self):
        return {'{}/e'.format(c.name): w.limit(c.lower_position_limit,
                                               c.lower_velocity_limit * self.sample_period * c.control_horizon,
                                               c.upper_velocity_limit * self.sample_period * c.control_horizon)
                for c in self.constraints}

    @memoize
    def get_upper_constraint_error(self):
        return {'{}/e'.format(c.name): w.limit(c.upper_position_limit,
                                               c.lower_velocity_limit * self.sample_period * c.control_horizon,
                                               c.upper_velocity_limit * self.sample_period * c.control_horizon)
                for c in self.constraints}

    def lbA(self):
        return self._sorter(self.get_lower_position_limits(),
                            self.get_last_velocities(),
                            self.get_derivative_link('vel'),
                            self.get_last_accelerations(),
                            self.get_derivative_link('acc'),
                            self.get_lower_constraint_velocities(),
                            self.get_lower_constraint_error())[0]

    def ubA(self):
        return self._sorter(self.get_upper_position_limits(),
                            self.get_last_velocities(False),
                            self.get_derivative_link('vel'),
                            self.get_last_accelerations(False),
                            self.get_derivative_link('acc'),
                            self.get_upper_constraint_velocities(),
                            self.get_upper_constraint_error())[0]

    def names(self):
        return self._sorter(self.get_upper_position_limits(),
                            self.get_last_velocities(),
                            self.get_derivative_link('vel'),
                            self.get_last_accelerations(),
                            self.get_derivative_link('acc'),
                            self.get_upper_constraint_velocities(),
                            self.get_lower_constraint_error())[1]


class A(Parent):
    def __init__(self, free_variables, constraints, sample_period, prediction_horizon, order=3):
        super(A, self).__init__(sample_period, prediction_horizon)
        self.free_variables = free_variables  # type: list[FreeVariable]
        self.constraints = constraints  # type: list[Constraint]
        self.order = order
        self.joints = {}
        self.height = 0
        self._compute_height()
        self.width = 0
        self._compute_width()

    def _compute_height(self):
        # rows for position limits of non continuous joints
        self.height = self.prediction_horizon * (self.number_of_joints - self.num_of_continuous_joints())
        # rows for linking vel/acc/jerk
        self.height += self.number_of_joints * self.prediction_horizon * (self.order - 1)
        # rows for constraints
        for i, c in enumerate(self.constraints):
            self.height += c.control_horizon
        # row for constraint error
        self.height += len(self.constraints)

    def _compute_width(self):
        # columns for joint vel/acc/jerk symbols
        self.width = self.number_of_joints * self.prediction_horizon * self.order
        # columns for constraints
        for i, c in enumerate(self.constraints):
            self.width += c.control_horizon
        # slack variable for constraint error
        self.width += len(self.constraints)

    @property
    def number_of_joints(self):
        return len(self.free_variables)

    @memoize
    def num_of_continuous_joints(self):
        return len([v for v in self.free_variables if not v.has_position_limits()])

    def get_soft_expressions(self):
        return self._sorter({c.name: c.expression for c in self.constraints})[0]

    def get_free_variable_symbols(self):
        return self._sorter({v.name: v.position_symbol for v in self.free_variables})[0]

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
        #         |  J*sp  |        |        |        |        |        |   I    |        |
        #         |-----------------------------------------------------------------------|
        #         |        |  J*sp  |        |        |        |        |        |   I    |
        #         |-----------------------------------------------------------------------|
        #         |  J*sp  |  J*sp  |        |        |        |        |   I    |   I    |
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
        # TODO possible speed improvement by creating blocks and stiching them together
        number_of_joints = self.number_of_joints
        A_soft = w.zeros(
            self.prediction_horizon * number_of_joints +
            number_of_joints * self.prediction_horizon * (self.order - 1) +
            len(self.constraints) * (self.prediction_horizon + 1),
            self.number_of_joints * self.prediction_horizon * self.order +
            len(self.constraints) * self.prediction_horizon + len(self.constraints)
        )
        t = time()
        J = w.jacobian(w.Matrix(self.get_soft_expressions()), self.get_free_variable_symbols(), order=1)
        J *= self.sample_period
        logging.loginfo(u'computed Jacobian in {:.5f}s'.format(time() - t))
        # Jd = w.jacobian(w.Matrix(soft_expressions), controlled_joints, order=2)
        # logging.loginfo(u'computed Jacobian dot in {:.5f}s'.format(time() - t))

        # position limits
        vertical_offset = number_of_joints * self.prediction_horizon
        for p in range(1, self.prediction_horizon + 1):
            matrix_size = number_of_joints * p
            I = w.eye(matrix_size) * self.sample_period
            start = vertical_offset - matrix_size
            A_soft[start:vertical_offset, :matrix_size] += I

        # derivative links
        I = w.eye(number_of_joints * (self.order - 1) * self.prediction_horizon)
        block_size = number_of_joints * (self.order - 1) * self.prediction_horizon
        A_soft[vertical_offset:vertical_offset + block_size, :block_size] += I
        h_offset = number_of_joints * self.prediction_horizon
        A_soft[vertical_offset:vertical_offset + block_size, h_offset:h_offset + block_size] += -I * self.sample_period

        I_height = number_of_joints * (self.prediction_horizon - 1)
        I = -w.eye(I_height)
        offset_v = vertical_offset
        offset_h = 0
        for o in range(self.order - 1):
            offset_v += number_of_joints
            A_soft[offset_v:offset_v + I_height, offset_h:offset_h + I_height] += I
            offset_v += I_height
            offset_h += self.prediction_horizon * number_of_joints
        vertical_offset = vertical_offset + block_size

        # soft constraints
        # TODO i don't need vel checks for the last 2 entries because the have to be zero with current B's
        # velocity limits
        J_vel_limit_block = w.kron(w.eye(self.prediction_horizon), J)
        next_vertical_offset = vertical_offset + J_vel_limit_block.shape[0]
        A_soft[vertical_offset:next_vertical_offset, :J_vel_limit_block.shape[1]] = J_vel_limit_block
        I = w.eye(J_vel_limit_block.shape[0]) * self.sample_period
        A_soft[vertical_offset:next_vertical_offset, -I.shape[1]-J.shape[0]:-J.shape[0]] = I
        # delete rows if control horizon of constraint shorter than prediction horizon
        rows_to_delete = []
        for t in range(self.prediction_horizon):
            for i, c in enumerate(self.constraints):
                index = vertical_offset + i + (t * len(self.constraints))
                if t + 1 > c.control_horizon:
                    rows_to_delete.append(index)

        # delete columns where control horizon is shorter than prediction horizon
        columns_to_delete = []
        horizontal_offset = A_soft.shape[1] - I.shape[1] - J.shape[0]
        for t in range(self.prediction_horizon):
            for i, c in enumerate(self.constraints):
                index = horizontal_offset + (t * len(self.constraints)) + i
                if t + 1 > c.control_horizon:
                    columns_to_delete.append(index)

        # J stack for total error
        J_hstack = w.hstack([J for _ in range(self.prediction_horizon)])
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
        I = w.eye(J_hstack.shape[0]) * self.sample_period / self.prediction_horizon
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

    def __init__(self, sample_period, prediciton_horizon, solver_name,
                 free_variables=None, constraints=None, debug_expressions=None):
        self.free_variables = []  # type: list[FreeVariable]
        self.constraints = []  # type: list[Constraint]
        self.debug_expressions = {}  # type: dict
        self.prediction_horizon = prediciton_horizon
        self.sample_period = sample_period
        self.order = 3  # TODO implement order
        if free_variables is not None:
            self.add_free_variables(free_variables)
        if constraints is not None:
            self.add_constraints(constraints)
        if debug_expressions is not None:
            self.add_debug_expressions(debug_expressions)

        if solver_name == u'gurobi':
            self.qp_solver = QPSolverGurubi()
        elif solver_name == u'qpoases':
            self.qp_solver = QPSolver()
        else:
            raise KeyError(u'Solver \'{}\' not supported'.format(solver_name))
        logging.loginfo(u'Using QP Solver \'{}\''.format(solver_name))

    def add_free_variables(self, free_variables):
        """
        :type free_variables: list
        """
        self.free_variables.extend(list(sorted(free_variables, key=lambda x: x.name)))
        l = [x.name for x in free_variables]
        duplicates = set([x for x in l if l.count(x) > 1])
        assert duplicates == set(), 'there are free variables with the same name: {}'.format(duplicates)

    def get_free_variable(self, name):
        """
        :type name: str
        :rtype: FreeVariable
        """
        for v in self.free_variables:
            if v.name == name:
                return v
        raise KeyError(u'No free variable with name: {}'.format(name))

    def add_constraints(self, constraints):
        """
        :type constraints: list
        """
        self.constraints.extend(list(sorted(constraints, key=lambda x: x.name)))
        l = [x.name for x in constraints]
        duplicates = set([x for x in l if l.count(x) > 1])
        assert duplicates == set(), 'there are constraints with the same name: {}'.format(duplicates)
        for c in self.constraints:
            if c.control_horizon is None:
                c.control_horizon = self.prediction_horizon
            elif c.control_horizon <= 0 or not isinstance(c.control_horizon, int):
                raise ValueError('Control horizon of {} is {}, it has to be an integer '
                                 '1 <= control horizon <= prediction horizon'.format(c.name, c.control_horizon))
            elif c.control_horizon > self.prediction_horizon:
                logging.logwarn(u'Specified control horizon of {} is bigger than prediction horizon.'
                                u'Reducing control horizon of {} to prediction horizon of {}'.format(c.name,
                                                                                                     c.control_horizon,
                                                                                                     self.prediction_horizon))


    def add_debug_expressions(self, debug_expressions):
        """
        :type debug_expressions: dict
        """
        # TODO check duplicates
        self.debug_expressions.update(debug_expressions)

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
        # TODO should use separate symbols lists
        self.compiled_debug_v = w.speed_up(self.debug_v, free_symbols)

    def __are_joint_limits_violated(self):
        violations = (self.p_ub - self.p_lb)[self.p_lb.data > self.p_ub.data]
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
        self.big_ass_M[self.A.height+1, :-2] = lb

    def _set_ub(self, ub):
        self.big_ass_M[self.A.height+2, :-2] = ub

    def _set_lbA(self, lbA):
        self.big_ass_M[:self.A.height, self.A.width] = lbA

    def _set_ubA(self, ubA):
        self.big_ass_M[:self.A.height, self.A.width + 1] = ubA

    @profile
    def _construct_big_ass_M(self):
        self.b = B(self.free_variables, self.constraints, self.sample_period, self.prediction_horizon)
        self.H = H(self.free_variables, self.constraints, self.sample_period, self.prediction_horizon)
        self.bA = BA(self.free_variables, self.constraints, self.sample_period, self.order, self.prediction_horizon)
        self.A = A(self.free_variables, self.constraints, self.sample_period, self.prediction_horizon, self.order)

        logging.loginfo(u'Constructing new controller with {} constraints and {} free variables...'.format(
            self.A.height, self.A.width))

        self._init_big_ass_M()

        self._set_weights(w.Matrix(self.H.weights()))
        self._set_A_soft(self.A.A())
        self._set_lbA(w.Matrix(self.bA.lbA()))
        self._set_ubA(w.Matrix(self.bA.ubA()))
        self._set_lb(w.Matrix(self.b.lb()))
        self._set_ub(w.Matrix(self.b.ub()))
        self.np_g = np.zeros(self.H.width)
        self.debug_names = list(sorted(self.debug_expressions.keys()))
        self.debug_v = w.Matrix([self.debug_expressions[name] for name in self.debug_names])
        self.H.make_error_id_to_vel_ids_map()

    def _eval_debug_exprs(self, subsitutions):
        return {name: value[0] for name, value in zip(self.debug_names, self.compiled_debug_v.call2(subsitutions))}

    @profile
    def make_filters(self):
        b_filter = self.np_weights != 0
        b_filter[:self.H.number_of_free_variables_with_horizon()] = True
        offset = self.H.number_of_free_variables_with_horizon() + self.H.number_of_constraint_vel_variables()
        map_ = self.H.make_error_id_to_vel_ids_map()
        for i in range(self.H.number_of_contraint_error_variables()):
            index = i+offset
            if not b_filter[index]:
                b_filter[map_[index]] = False

        bA_filter = np.ones(self.A.height, dtype=bool)
        ll = self.H.number_of_constraint_vel_variables()+self.H.number_of_contraint_error_variables()
        bA_filter[-ll:] = b_filter[-ll:]
        return b_filter, bA_filter

    @profile
    def filter_zero_weight_stuff(self, b_filter, bA_filter):
        return np.diag(self.np_weights[b_filter]), \
               np.zeros(b_filter.shape[0]), \
               self.np_A[bA_filter, :][:,b_filter], \
               self.np_lb[b_filter], \
               self.np_ub[b_filter], \
               self.np_lbA[bA_filter], \
               self.np_ubA[bA_filter]


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
        self.np_weights = np_big_ass_M[self.A.height, :-2]
        # self.np_H = np.diag(self.np_weights)
        self.np_A = np_big_ass_M[:self.A.height, :self.A.width]
        self.np_lb = np_big_ass_M[self.A.height+1, :-2]
        self.np_ub = np_big_ass_M[self.A.height+2, :-2]
        self.np_lbA = np_big_ass_M[:self.A.height, -2]
        self.np_ubA = np_big_ass_M[:self.A.height, -1]

        try:
            filters = self.make_filters()
            filtered_stuff = self.filter_zero_weight_stuff(*filters)
            self.xdot_full = self.qp_solver.solve(*filtered_stuff)
            # self.xdot_full = self.qp_solver.solve(self.np_H, self.np_g, self.np_A, self.np_lb, self.np_ub, self.np_lbA,
            #                                       self.np_ubA)
        except Exception as e:
            self._create_debug_pandas(substitutions, actually_print=True)
            if isinstance(e, InfeasibleException):
                if self.__are_joint_limits_violated():
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
        # for debugging to might want to execute this line to create named panda matrices
        # self._create_debug_pandas(substitutions, self.xdot_full)
        return self.split_xdot(self.xdot_full), self._eval_debug_exprs(substitutions)

    def split_xdot(self, xdot):
        split = []
        offset = len(self.free_variables)
        for derivative in range(self.order):
            split.append(OrderedDict((x.name, xdot[i + offset * self.prediction_horizon * derivative])
                                     for i, x in enumerate(self.free_variables)))
        return split

    def b_names(self):
        return self.b.names()

    def bA_names(self):
        return self.bA.names()

    def _viz_mpc(self, x, joint_name, state):
        start_pos = state[joint_name]
        ts = np.array([(i + 1) * self.sample_period for i in range(self.prediction_horizon)])
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
        plt.tight_layout()
        plt.show()

    @profile
    def _create_debug_pandas(self, substitutions, xdot_full=None, actually_print=False):
        b_names = self.b_names()
        bA_names = self.bA_names()
        b_filter, bA_filter = self.make_filters()
        filtered_b_names = np.array(b_names)[b_filter]
        filtered_bA_names = np.array(bA_names)[bA_filter]
        H, g, A, lb, ub, lbA, ubA = self.filter_zero_weight_stuff(b_filter, bA_filter)

        debug_exprs = self._eval_debug_exprs(substitutions)
        self.p_debug = pd.DataFrame.from_dict(debug_exprs, orient='index', columns=['data']).sort_index()

        self.p_lb = pd.DataFrame(lb, filtered_b_names, [u'data'], dtype=float)
        self.p_ub = pd.DataFrame(ub, filtered_b_names, [u'data'], dtype=float)
        self.p_g = pd.DataFrame(g, filtered_b_names, [u'data'], dtype=float)
        self.p_lbA = pd.DataFrame(lbA, filtered_bA_names, [u'data'], dtype=float)
        self.p_ubA = pd.DataFrame(ubA, filtered_bA_names, [u'data'], dtype=float)
        self.p_weights = pd.DataFrame(self.np_H.dot(np.ones(self.np_H.shape[0])), b_names, [u'data'],
                                 dtype=float)
        self.p_A = pd.DataFrame(A, filtered_bA_names, filtered_b_names, dtype=float)
        if xdot_full is not None:
            self.p_xdot = pd.DataFrame(xdot_full, filtered_b_names, [u'data'], dtype=float)
            # Ax = np.dot(self.np_A, xdot_full)
            xH = np.dot((xdot_full ** 2).T, H)
            self.p_xH = pd.DataFrame(xH, filtered_b_names, [u'data'], dtype=float)
            # p_xg = p_g * p_xdot
            xHx = np.dot(np.dot(xdot_full.T, H), xdot_full)
            num_non_slack = len(self.free_variables) * self.prediction_horizon * 3
            self.p_xdot2 = deepcopy(self.p_xdot)
            self.p_xdot2[num_non_slack:] = 0
            self.p_Ax = pd.DataFrame(self.p_A.dot(self.p_xdot), filtered_bA_names, [u'data'], dtype=float)
            self.p_Ax2 = pd.DataFrame(self.p_A.dot(self.p_xdot2), filtered_bA_names, [u'data'], dtype=float)

            # x_soft = xdot_full[len(xdot_full) - num_slack:]
            # p_lbA_minus_x = pd.DataFrame(lbA - x_soft, filtered_bA_names, [u'data'], dtype=float).sort_index()
            # p_ubA_minus_x = pd.DataFrame(ubA - x_soft, filtered_bA_names, [u'data'], dtype=float).sort_index()
        else:
            self.p_xdot = None


        # if self.lbAs is None:
        #     self.lbAs = p_lbA
        # else:
        #     self.lbAs = self.lbAs.T.append(p_lbA.T, ignore_index=True).T
        # self.lbAs.T[[c for c in self.lbAs.T.columns if 'dist' in c]].plot()

        # self.save_all(p_weights, p_A, p_lbA, p_ubA, p_lb, p_ub, p_xdot)
        state = {k: v for k, v in zip(self.compiled_big_ass_M.str_params, substitutions)}
        # self._viz_mpc(p_xdot, 'j', state)
        # self._viz_mpc(p_xdot, 'world_robot_joint_state_r_shoulder_lift_joint_position', state)
        # p_lbA[p_lbA != 0].abs().sort_values(by='data')
        # get non 0 A entries
        # p_A.iloc[[1133]].T.loc[p_A.values[1133] != 0]
