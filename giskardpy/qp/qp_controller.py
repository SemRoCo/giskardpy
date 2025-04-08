import abc
import datetime
import os
from abc import ABC
from collections import defaultdict
from copy import deepcopy
from enum import IntEnum
from typing import List, Dict, Tuple, Type, Union, Optional, DefaultDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import giskardpy.casadi_wrapper as cas
from giskardpy.data_types.exceptions import HardConstraintsViolatedException, QPSolverException, InfeasibleException, \
    VelocityLimitUnreachableException
from giskardpy.god_map import god_map
from giskardpy.data_types.data_types import Derivatives
from giskardpy.qp.constraint import InequalityConstraint, EqualityConstraint, DerivativeInequalityConstraint
from giskardpy.qp.free_variable import FreeVariable
from giskardpy.qp.next_command import NextCommands
from giskardpy.qp.pos_in_vel_limits import b_profile, implicit_vel_profile
from giskardpy.qp.qp_solver import QPSolver
from giskardpy.qp.qp_solver_ids import SupportedQPSolver
from giskardpy.symbol_manager import symbol_manager
from giskardpy.middleware import get_middleware
from giskardpy.utils.utils import create_path, get_all_classes_in_package
from giskardpy.utils.decorators import memoize
import giskardpy.utils.math as giskard_math
from giskardpy.qp.weight_gain import QuadraticWeightGain, LinearWeightGain
from line_profiler import profile
from giskardpy.qp.constraint import DerivativeEqualityConstraint

# used for saving pandas in the same folder every time within a run
date_str = datetime.datetime.now().strftime('%Yy-%mm-%dd--%Hh-%Mm-%Ss')


class QPFormulation(IntEnum):
    no_mpc = -1
    explicit = 0
    implicit = 1
    explicit_no_acc = 2
    explicit_explicit_pos_limits = 10
    implicit_explicit_pos_limits = 11
    explicit_no_acc_explicit_pos_limits = 12
    implicit_variable_dt = 21

    def explicit_pos_limits(self) -> bool:
        return 20 > self > 10

    def is_dt_variable(self) -> bool:
        return self > 20

    def is_no_mpc(self) -> bool:
        return self == self.no_mpc

    def has_acc_variables(self) -> bool:
        return self.is_explicit()

    def has_jerk_variables(self) -> bool:
        return self.is_explicit() or self.is_explicit_no_acc()

    def is_mpc(self) -> bool:
        return not self.is_no_mpc()

    def is_implicit(self) -> bool:
        return self in [self.implicit, self.implicit_explicit_pos_limits]

    def is_explicit(self) -> bool:
        return self in [self.explicit, self.explicit_explicit_pos_limits]

    def is_explicit_no_acc(self) -> bool:
        return self in [self.explicit_no_acc, self.explicit_no_acc_explicit_pos_limits]


def save_pandas(dfs, names, path, time: float, folder_name: Optional[str] = None):
    if folder_name is None:
        folder_name = ''
    folder_name = f'{path}/pandas/{folder_name}_{date_str}/{time}/'
    create_path(folder_name)
    for df, name in zip(dfs, names):
        csv_string = 'name\n'
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            if df.shape[1] > 1:
                for column_name, column in df.T.items():
                    zero_filtered_column = column.replace(0, np.nan).dropna(how='all').replace(np.nan, 0)
                    csv_string += zero_filtered_column.add_prefix(column_name + '||').to_csv(float_format='%.6f')
            else:
                csv_string += df.to_csv(float_format='%.6f')
        file_name2 = f'{folder_name}{name}.csv'
        with open(file_name2, 'w') as f:
            f.write(csv_string)


class ProblemDataPart(ABC):
    """
    min_x 0.5*x^T*diag(w)*x + g^T*x
    s.t.  lb <= x <= ub
        lbA <= A*x <= ubA
               E*x = b
    """
    free_variables: List[FreeVariable]
    equality_constraints: List[EqualityConstraint]
    inequality_constraints: List[InequalityConstraint]
    derivative_constraints: List[DerivativeInequalityConstraint]
    eq_derivative_constraints: List[DerivativeEqualityConstraint]
    dt: float
    prediction_horizon: int
    control_horizon: int
    max_derivative: Derivatives
    qp_formulation: QPFormulation

    def __init__(self,
                 free_variables: List[FreeVariable],
                 equality_constraints: List[EqualityConstraint],
                 inequality_constraints: List[InequalityConstraint],
                 derivative_constraints: List[DerivativeInequalityConstraint],
                 eq_derivative_constraints: List[DerivativeEqualityConstraint],
                 sample_period: float,
                 prediction_horizon: int,
                 max_derivative: Derivatives,
                 alpha: float,
                 qp_formulation: QPFormulation):
        self.alpha = alpha
        self.qp_formulation = qp_formulation
        self.free_variables = free_variables
        self.equality_constraints = equality_constraints
        self.inequality_constraints = inequality_constraints
        self.derivative_constraints = derivative_constraints
        self.eq_derivative_constraints = eq_derivative_constraints
        self.prediction_horizon = prediction_horizon
        self.dt = sample_period
        self.max_derivative = max_derivative
        self.control_horizon = self.prediction_horizon - self.max_derivative + 1

    @property
    def number_of_free_variables(self) -> int:
        return len(self.free_variables)

    @property
    def number_ineq_slack_variables(self):
        return sum(self.control_horizon for c in self.velocity_constraints)

    def replace_hack(self, expression: Union[float, cas.Expression], new_value):
        if not isinstance(expression, cas.Expression):
            return expression
        hack = symbol_manager.hack
        expression.s = cas.ca.substitute(expression.s, hack.s, new_value)
        return expression

    def get_derivative_constraints(self, derivative: Derivatives) -> List[DerivativeInequalityConstraint]:
        return [c for c in self.derivative_constraints if c.derivative == derivative]

    def get_eq_derivative_constraints(self, derivative: Derivatives) -> List[DerivativeEqualityConstraint]:
        return [c for c in self.eq_derivative_constraints if c.derivative == derivative]

    @abc.abstractmethod
    def construct_expression(self) -> Union[cas.Expression, Tuple[cas.Expression, cas.Expression]]:
        pass

    @property
    def velocity_constraints(self) -> List[DerivativeInequalityConstraint]:
        return self.get_derivative_constraints(Derivatives.velocity)

    @property
    def velocity_eq_constraints(self) -> List[DerivativeEqualityConstraint]:
        return self.get_eq_derivative_constraints(Derivatives.velocity)

    @property
    def acceleration_constraints(self) -> List[DerivativeInequalityConstraint]:
        return self.get_derivative_constraints(Derivatives.acceleration)

    @property
    def jerk_constraints(self) -> List[DerivativeInequalityConstraint]:
        return self.get_derivative_constraints(Derivatives.jerk)

    def _sorter(self, *args: dict) -> Tuple[List[cas.symbol_expr], np.ndarray]:
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
        return result, np.array(result_names)

    def __helper(self, param: dict):
        return [x for _, x in sorted(param.items())]

    def __helper_names(self, param: dict):
        return [x for x, _ in sorted(param.items())]

    def _remove_columns_columns_where_variables_are_zero(self, free_variable_model: cas.Expression,
                                                         max_derivative: Derivatives) -> cas.Expression:
        if np.prod(free_variable_model.shape) == 0:
            return free_variable_model
        column_ids = []
        end = 0
        for derivative in Derivatives.range(Derivatives.velocity, max_derivative - 1):
            last_non_zero_variable = self.prediction_horizon - (max_derivative - derivative)
            start = end + self.number_of_free_variables * last_non_zero_variable
            end += self.number_of_free_variables * self.prediction_horizon
            column_ids.extend(range(start, end))
        free_variable_model.remove([], column_ids)
        return free_variable_model

    @profile
    def velocity_limit(self, v: FreeVariable, max_derivative: Derivatives) -> Tuple[cas.Expression, cas.Expression]:
        current_position = v.get_symbol(Derivatives.position)
        lower_velocity_limit = v.get_lower_limit(Derivatives.velocity, evaluated=True)
        upper_velocity_limit = v.get_upper_limit(Derivatives.velocity, evaluated=True)
        lower_acc_limit = v.get_lower_limit(Derivatives.acceleration, evaluated=True)
        upper_acc_limit = v.get_upper_limit(Derivatives.acceleration, evaluated=True)
        current_vel = v.get_symbol(Derivatives.velocity)
        current_acc = v.get_symbol(Derivatives.acceleration)

        lower_jerk_limit = v.get_lower_limit(Derivatives.jerk, evaluated=True)
        upper_jerk_limit = v.get_upper_limit(Derivatives.jerk, evaluated=True)
        if self.prediction_horizon == 1:
            return cas.Expression([lower_velocity_limit]), cas.Expression([upper_velocity_limit])

        if upper_jerk_limit is None:
            upper_jerk_limit = giskard_math.find_best_jerk_limit(self.prediction_horizon, self.dt,
                                                                 upper_velocity_limit)
            lower_jerk_limit = -upper_jerk_limit

        if not v.has_position_limits():
            lower_limit = upper_limit = None
        else:
            lower_limit = v.get_lower_limit(Derivatives.position, evaluated=True)
            upper_limit = v.get_upper_limit(Derivatives.position, evaluated=True)

        try:
            lb, ub = b_profile(current_pos=current_position,
                               current_vel=current_vel,
                               current_acc=current_acc,
                               pos_limits=(lower_limit, upper_limit),
                               vel_limits=(lower_velocity_limit, upper_velocity_limit),
                               acc_limits=(lower_acc_limit, upper_acc_limit),
                               jerk_limits=(lower_jerk_limit, upper_jerk_limit),
                               dt=self.dt,
                               ph=self.prediction_horizon)
        except InfeasibleException as e:
            max_reachable_vel = giskard_math.max_velocity_from_horizon_and_jerk_qp(
                prediction_horizon=self.prediction_horizon,
                vel_limit=100,
                acc_limit=upper_acc_limit,
                jerk_limit=upper_jerk_limit,
                dt=self.dt,
                max_derivative=max_derivative)[0]
            if max_reachable_vel < upper_velocity_limit:
                error_msg = f'Free variable "{v.name}" can\'t reach velocity limit of "{upper_velocity_limit}". ' \
                            f'Maximum reachable with prediction horizon = "{self.prediction_horizon}", ' \
                            f'jerk limit = "{upper_jerk_limit}" and dt = "{self.dt}" is "{max_reachable_vel}".'
                get_middleware().logerr(error_msg)
                raise VelocityLimitUnreachableException(error_msg)
            else:
                raise
        # %% set velocity limits to infinite, that can't be reached due to acc/jerk limits anyway
        # unlimited_vel_profile = implicit_vel_profile(acc_limit=upper_acc_limit,
        #                                              jerk_limit=upper_jerk_limit,
        #                                              dt=self.dt,
        #                                              ph=self.prediction_horizon)
        # for i in range(self.prediction_horizon):
        #     ub[i] = cas.if_less(ub[i], unlimited_vel_profile[i], ub[i], np.inf)
        # unlimited_vel_profile = implicit_vel_profile(acc_limit=-lower_acc_limit,
        #                                              jerk_limit=-lower_jerk_limit,
        #                                              dt=self.dt,
        #                                              ph=self.prediction_horizon)
        # for i in range(self.prediction_horizon):
        #     lb[i] = cas.if_less(-lb[i], unlimited_vel_profile[i], lb[i], -np.inf)
        return lb, ub


class Weights(ProblemDataPart):
    """
    format:
        free_variable_velocity
        free_variable_acceleration
        free_variable_jerk
        equality_constraints
        derivative_constraints_velocity
        derivative_constraints_acceleration
        derivative_constraints_jerk
        inequality_constraints
    """

    def __init__(self, free_variables: List[FreeVariable],
                 equality_constraints: List[EqualityConstraint],
                 inequality_constraints: List[InequalityConstraint],
                 derivative_constraints: List[DerivativeInequalityConstraint],
                 eq_derivative_constraints: List[DerivativeEqualityConstraint],
                 sample_period: float,
                 prediction_horizon: int, max_derivative: Derivatives,
                 alpha: float,
                 qp_formulation: QPFormulation):
        super().__init__(free_variables=free_variables,
                         equality_constraints=equality_constraints,
                         inequality_constraints=inequality_constraints,
                         derivative_constraints=derivative_constraints,
                         eq_derivative_constraints=eq_derivative_constraints,
                         sample_period=sample_period,
                         prediction_horizon=prediction_horizon,
                         max_derivative=max_derivative,
                         alpha=alpha,
                         qp_formulation=qp_formulation)
        self.evaluated = True

    def linear_f(self, current_position, limit, target_value, a=10, exp=2) -> Tuple[cas.Expression, float]:
        f = cas.abs(current_position * a) ** exp
        x_offset = cas.solve_for(f, target_value)
        return (cas.abs(current_position + x_offset - limit) * a) ** exp, x_offset

    @profile
    def construct_expression(self, quadratic_weight_gains: List[QuadraticWeightGain] = None,
                             linear_weight_gains: List[LinearWeightGain] = None) -> Union[
        cas.Expression, Tuple[cas.Expression, cas.Expression]]:
        components = []
        components.extend(self.free_variable_weights_expression(quadratic_weight_gains=quadratic_weight_gains))
        components.append(self.equality_weight_expressions())
        components.extend(self.eq_derivative_weight_expressions())
        components.extend(self.derivative_weight_expressions())
        components.append(self.inequality_weight_expressions())
        weights, _ = self._sorter(*components)
        weights = cas.Expression(weights)
        linear_weights = self.linear_weights_expression(linear_weight_gains=linear_weight_gains)
        if linear_weights is None:
            linear_weights = cas.zeros(*weights.shape)
        else:
            # as of now linear weights are only added for joints, therefore equality-, derivative- and inequality
            # weights are missing. Here the missing weights are filled in with zeroes.
            linear_weights, _ = self._sorter(*linear_weights)
            linear_weights = cas.Expression(linear_weights)
            linear_weights = cas.vstack(
                [linear_weights] + [cas.Expression(0)] * (weights.shape[0] - linear_weights.shape[0]))
        return cas.Expression(weights), linear_weights

    @profile
    def free_variable_weights_expression(self, quadratic_weight_gains: List[QuadraticWeightGain]) -> \
            List[defaultdict]:
        max_derivative = self.max_derivative
        params = []
        weights = defaultdict(dict)  # maps order to joints
        for t in range(self.prediction_horizon):
            for v in self.free_variables:
                for derivative in Derivatives.range(Derivatives.velocity, max_derivative):
                    if t >= self.prediction_horizon - (max_derivative - derivative):
                        continue
                    if self.qp_formulation.is_implicit():
                        if derivative >= Derivatives.acceleration:
                            continue
                    if self.qp_formulation.is_explicit_no_acc():
                        if not (derivative == Derivatives.velocity or derivative == max_derivative):
                            continue
                    normalized_weight = v.normalized_weight(t, derivative, self.prediction_horizon - 3,
                                                            alpha=self.alpha,
                                                            evaluated=self.evaluated)
                    weights[derivative][f't{t:03}/{v.position_name}/{derivative}'] = normalized_weight
                    for q_gain in quadratic_weight_gains:
                        if t < len(q_gain.gains) and v in q_gain.gains[t][derivative].keys():
                            weights[derivative][f't{t:03}/{v.position_name}/{derivative}'] *= \
                                q_gain.gains[t][derivative][v]
        for _, weight in sorted(weights.items()):
            params.append(weight)
        return params

    def derivative_weight_expressions(self) -> List[Dict[str, cas.Expression]]:
        params = []
        for d in Derivatives.range(Derivatives.velocity, self.max_derivative):
            derivative_constr_weights = {}
            for t in range(self.prediction_horizon):
                d = Derivatives(d)
                for c in self.get_derivative_constraints(d):
                    if t < self.control_horizon:
                        derivative_constr_weights[f't{t:03}/{c.name}'] = c.normalized_weight(t)
            params.append(derivative_constr_weights)
        return params

    def eq_derivative_weight_expressions(self) -> List[Dict[str, cas.Expression]]:
        params = []
        for d in Derivatives.range(Derivatives.velocity, self.max_derivative):
            derivative_constr_weights = {}
            for t in range(self.prediction_horizon):
                d = Derivatives(d)
                for c in self.get_eq_derivative_constraints(d):
                    if t < self.control_horizon:
                        derivative_constr_weights[f't{t:03}/{c.name}'] = c.normalized_weight(t)
            params.append(derivative_constr_weights)
        return params

    def eq_derivative_weight_expressions(self) -> List[Dict[str, cas.Expression]]:
        params = []
        for d in Derivatives.range(Derivatives.velocity, self.max_derivative):
            derivative_constr_weights = {}
            for t in range(self.prediction_horizon):
                d = Derivatives(d)
                for c in self.get_eq_derivative_constraints(d):
                    if t < self.control_horizon:
                        derivative_constr_weights[f't{t:03}/{c.name}'] = c.normalized_weight(t)
            params.append(derivative_constr_weights)
        return params

    def equality_weight_expressions(self) -> dict:
        error_slack_weights = {f'{c.name}/error': c.normalized_weight(self.control_horizon)
                               for c in self.equality_constraints}
        return error_slack_weights

    def inequality_weight_expressions(self) -> dict:
        error_slack_weights = {f'{c.name}/error': c.normalized_weight(self.control_horizon)
                               for c in self.inequality_constraints}
        return error_slack_weights

    @profile
    def linear_weights_expression(self, linear_weight_gains: List[LinearWeightGain] = None):
        if len(linear_weight_gains) > 0:
            params = []
            weights = defaultdict(dict)  # maps order to joints
            for t in range(self.prediction_horizon):
                for v in self.free_variables:
                    for derivative in Derivatives.range(Derivatives.velocity, self.max_derivative):
                        if t >= self.prediction_horizon - (self.max_derivative - derivative):
                            continue
                        if derivative == Derivatives.acceleration and not self.qp_formulation.has_acc_variables():
                            continue
                        if derivative == Derivatives.jerk and not self.qp_formulation.has_jerk_variables():
                            continue
                        weights[derivative][f't{t:03}/{v.position_name}/{derivative}'] = 0
                        for l_gain in linear_weight_gains:
                            if t < len(l_gain.gains) and v in l_gain.gains[t][derivative].keys():
                                weights[derivative][f't{t:03}/{v.position_name}/{derivative}'] += \
                                    l_gain.gains[t][derivative][v]
            for _, weight in sorted(weights.items()):
                params.append(weight)
            return params
        return None

    def get_free_variable_symbols(self, order: Derivatives) -> List[cas.Symbol]:
        return self._sorter({v.position_name: v.get_symbol(order) for v in self.free_variables})[0]


class FreeVariableBounds(ProblemDataPart):
    """
    format:
        free_variable_velocity
        free_variable_acceleration
        free_variable_jerk
        slack_equality_constraints
        slack_derivative_constraints_velocity
        slack_derivative_constraints_acceleration
        slack_derivative_constraints_jerk
        slack_inequality_constraints
    """
    names: np.ndarray
    names_without_slack: np.ndarray
    names_slack: np.ndarray
    names_neq_slack: np.ndarray
    names_derivative_slack: np.ndarray
    names_eq_slack: np.ndarray

    def __init__(self, free_variables: List[FreeVariable],
                 equality_constraints: List[EqualityConstraint],
                 inequality_constraints: List[InequalityConstraint],
                 derivative_constraints: List[DerivativeInequalityConstraint],
                 eq_derivative_constraints: List[DerivativeEqualityConstraint],
                 sample_period: float,
                 prediction_horizon: int,
                 max_derivative: Derivatives,
                 alpha: float,
                 qp_formulation: QPFormulation):
        super().__init__(free_variables=free_variables,
                         equality_constraints=equality_constraints,
                         inequality_constraints=inequality_constraints,
                         derivative_constraints=derivative_constraints,
                         eq_derivative_constraints=eq_derivative_constraints,
                         sample_period=sample_period,
                         prediction_horizon=prediction_horizon,
                         max_derivative=max_derivative,
                         alpha=alpha,
                         qp_formulation=qp_formulation)
        self.evaluated = True

    @profile
    def free_variable_bounds(self) \
            -> Tuple[List[Dict[str, cas.symbol_expr_float]], List[Dict[str, cas.symbol_expr_float]]]:
        # if self.qp_formulation in [ControllerMode.explicit, ControllerMode.explicit_no_acc]:
        max_derivative = self.max_derivative
        lb: DefaultDict[Derivatives, Dict[str, cas.symbol_expr_float]] = defaultdict(dict)
        ub: DefaultDict[Derivatives, Dict[str, cas.symbol_expr_float]] = defaultdict(dict)
        for v in self.free_variables:
            if self.qp_formulation.explicit_pos_limits():
                for t in range(self.prediction_horizon):
                    for derivative in Derivatives.range(Derivatives.velocity, max_derivative):
                        if t >= self.prediction_horizon - (max_derivative - derivative):
                            continue
                        if self.qp_formulation.is_explicit_no_acc() and derivative == Derivatives.acceleration:
                            continue
                        if self.qp_formulation.is_implicit() and derivative >= Derivatives.acceleration:
                            continue
                        index = t + self.prediction_horizon * (derivative - 1)
                        lb[derivative][f't{t:03}/{v.name}/{derivative}'] = v.get_lower_limit(derivative)
                        ub[derivative][f't{t:03}/{v.name}/{derivative}'] = v.get_upper_limit(derivative)
            else:
                lb_, ub_ = self.velocity_limit(v=v, max_derivative=max_derivative)
                for t in range(self.prediction_horizon):
                    for derivative in Derivatives.range(Derivatives.velocity, max_derivative):
                        if t >= self.prediction_horizon - (max_derivative - derivative):
                            continue
                        if self.qp_formulation.is_explicit_no_acc() and derivative == Derivatives.acceleration:
                            continue
                        if self.qp_formulation.is_implicit() and derivative >= Derivatives.acceleration:
                            continue
                        index = t + self.prediction_horizon * (derivative - 1)
                        lb[derivative][f't{t:03}/{v.name}/{derivative}'] = lb_[index]
                        ub[derivative][f't{t:03}/{v.name}/{derivative}'] = ub_[index]
        lb_params = []
        ub_params = []
        for derivative, name_to_bound_map in sorted(lb.items()):
            lb_params.append(name_to_bound_map)
        for derivative, name_to_bound_map in sorted(ub.items()):
            ub_params.append(name_to_bound_map)
        return lb_params, ub_params

    def derivative_slack_limits(self, derivative: Derivatives) \
            -> Tuple[Dict[str, cas.Expression], Dict[str, cas.Expression]]:
        lower_slack = {}
        upper_slack = {}
        for t in range(self.prediction_horizon):
            for c in self.get_derivative_constraints(derivative):
                if t < self.control_horizon:
                    lower_slack[f't{t:03}/{c.name}'] = c.lower_slack_limit[t]
                    upper_slack[f't{t:03}/{c.name}'] = c.upper_slack_limit[t]
        return lower_slack, upper_slack

    def eq_derivative_slack_limits(self, derivative: Derivatives) \
            -> Tuple[Dict[str, cas.Expression], Dict[str, cas.Expression]]:
        lower_slack = {}
        upper_slack = {}
        for t in range(self.prediction_horizon):
            for c in self.get_eq_derivative_constraints(derivative):
                if t < self.control_horizon:
                    lower_slack[f't{t:03}/{c.name}'] = c.lower_slack_limit[t]
                    upper_slack[f't{t:03}/{c.name}'] = c.upper_slack_limit[t]
        return lower_slack, upper_slack

    def eq_derivative_slack_limits(self, derivative: Derivatives) \
            -> Tuple[Dict[str, cas.Expression], Dict[str, cas.Expression]]:
        lower_slack = {}
        upper_slack = {}
        for t in range(self.prediction_horizon):
            for c in self.get_eq_derivative_constraints(derivative):
                if t < self.control_horizon:
                    lower_slack[f't{t:03}/{c.name}'] = c.lower_slack_limit[t]
                    upper_slack[f't{t:03}/{c.name}'] = c.upper_slack_limit[t]
        return lower_slack, upper_slack

    def equality_constraint_slack_lower_bound(self):
        return {f'{c.name}/error': c.lower_slack_limit for c in self.equality_constraints}

    def equality_constraint_slack_upper_bound(self):
        return {f'{c.name}/error': c.upper_slack_limit for c in self.equality_constraints}

    def inequality_constraint_slack_lower_bound(self):
        return {f'{c.name}/error': c.lower_slack_limit for c in self.inequality_constraints}

    def inequality_constraint_slack_upper_bound(self):
        return {f'{c.name}/error': c.upper_slack_limit for c in self.inequality_constraints}

    @profile
    def construct_expression(self) -> Union[cas.Expression, Tuple[cas.Expression, cas.Expression]]:
        lb_params, ub_params = self.free_variable_bounds()
        num_free_variables = sum(len(x) for x in lb_params)

        equality_constraint_slack_lower_bounds = self.equality_constraint_slack_lower_bound()
        num_eq_slacks = len(equality_constraint_slack_lower_bounds)
        lb_params.append(equality_constraint_slack_lower_bounds)
        ub_params.append(self.equality_constraint_slack_upper_bound())

        num_eq_derivative_slack = 0
        for derivative in Derivatives.range(Derivatives.velocity, self.max_derivative):
            lower_slack, upper_slack = self.eq_derivative_slack_limits(derivative)
            num_eq_derivative_slack += len(lower_slack)
            lb_params.append(lower_slack)
            ub_params.append(upper_slack)

        num_derivative_slack = 0
        for derivative in Derivatives.range(Derivatives.velocity, self.max_derivative):
            lower_slack, upper_slack = self.derivative_slack_limits(derivative)
            num_derivative_slack += len(lower_slack)
            lb_params.append(lower_slack)
            ub_params.append(upper_slack)

        lb_params.append(self.inequality_constraint_slack_lower_bound())
        ub_params.append(self.inequality_constraint_slack_upper_bound())

        lb, self.names = self._sorter(*lb_params)
        ub, _ = self._sorter(*ub_params)
        self.names_without_slack = self.names[:num_free_variables]
        self.names_slack = self.names[num_free_variables:]

        derivative_slack_start = 0
        derivative_slack_stop = derivative_slack_start + num_derivative_slack + num_eq_derivative_slack
        self.names_derivative_slack = self.names_slack[derivative_slack_start:derivative_slack_stop]

        eq_slack_start = derivative_slack_stop
        eq_slack_stop = eq_slack_start + num_eq_slacks
        self.names_eq_slack = self.names_slack[eq_slack_start:eq_slack_stop]

        neq_slack_start = eq_slack_stop
        self.names_neq_slack = self.names_slack[neq_slack_start:]
        return cas.Expression(lb), cas.Expression(ub)


class EqualityBounds(ProblemDataPart):
    """
    Format:
        last free variable velocity
        0
        last free variable acceleration
        0
        equality_constraint_bounds
    """
    names: np.ndarray
    names_equality_constraints: np.ndarray
    names_derivative_links: np.ndarray

    def __init__(self, free_variables: List[FreeVariable],
                 equality_constraints: List[EqualityConstraint],
                 inequality_constraints: List[InequalityConstraint],
                 derivative_constraints: List[DerivativeInequalityConstraint],
                 eq_derivative_constraints: List[DerivativeEqualityConstraint],
                 sample_period: float,
                 prediction_horizon: int,
                 max_derivative: Derivatives,
                 alpha: float,
                 qp_formulation: QPFormulation):
        super().__init__(free_variables=free_variables,
                         equality_constraints=equality_constraints,
                         inequality_constraints=inequality_constraints,
                         derivative_constraints=derivative_constraints,
                         eq_derivative_constraints=eq_derivative_constraints,
                         sample_period=sample_period,
                         prediction_horizon=prediction_horizon,
                         max_derivative=max_derivative,
                         alpha=alpha,
                         qp_formulation=qp_formulation)
        self.evaluated = True

    def equality_constraint_bounds(self) -> Dict[str, cas.Expression]:
        return {f'{c.name}': c.capped_bound(self.dt, self.control_horizon) for c in self.equality_constraints}

    def last_derivative_values(self, derivative: Derivatives) -> Dict[str, cas.symbol_expr_float]:
        last_values = {}
        for v in self.free_variables:
            last_values[f'{v.name}/last_{derivative}'] = v.get_symbol(derivative)
        return last_values

    def derivative_links(self, derivative: Derivatives) -> Dict[str, cas.symbol_expr_float]:
        derivative_link = {}
        for t in range(self.prediction_horizon - 1):
            if t >= self.prediction_horizon - (self.max_derivative - derivative):
                continue  # this row is all zero in the model, because the system has to stop at 0 vel
            for v in self.free_variables:
                derivative_link[f't{t:03}/{derivative}/{v.name}/link'] = 0
        return derivative_link

    def eq_derivative_constraint_bounds(self, derivative: Derivatives) \
            -> Tuple[Dict[str, cas.Expression], Dict[str, cas.Expression]]:
        bound = {}
        for t in range(self.prediction_horizon):
            for c in self.get_eq_derivative_constraints(derivative):
                if t < self.control_horizon:
                    bound[f't{t:03}/{c.name}'] = c.bound[t] * self.dt
        return bound

    @profile
    def construct_expression(self) -> Union[cas.Expression, Tuple[cas.Expression, cas.Expression]]:
        """
        explicit no acc
        -vc - ac*dt = -v0 + j0*dt**2
                 vc = -v1 + 2*v0 + j1*dt**2
                  0 = -vt + 2*vt-1 - vt-2 + jt*dt**2
                  0 =   0 + 2*vt-1 - vt-2 + jt*dt**2
                  0 =   0 +    0   - vt-2 + jt*dt**2
        :return:
        """
        bounds = []
        if self.qp_formulation.is_explicit_no_acc():
            derivative_link = {}
            for t in range(self.prediction_horizon):
                for v in self.free_variables:
                    name = f't{t:03}/{Derivatives.jerk}/{v.name}/link'
                    if t == 0:
                        derivative_link[name] = - v.get_symbol(Derivatives.velocity) - v.get_symbol(
                            Derivatives.acceleration) * self.dt
                    elif t == 1:
                        derivative_link[name] = v.get_symbol(Derivatives.velocity)
                    else:
                        derivative_link[name] = 0
            bounds.append(derivative_link)
        else:
            if self.qp_formulation.is_implicit():
                max_derivative = Derivatives.velocity
            else:
                max_derivative = self.max_derivative

            for derivative in Derivatives.range(Derivatives.velocity, max_derivative - 1):
                bounds.append(self.last_derivative_values(derivative))
                bounds.append(self.derivative_links(derivative))

        num_derivative_links = sum(len(x) for x in bounds)
        num_derivative_constraints = 0
        for derivative in Derivatives.range(Derivatives.velocity, self.max_derivative):
            bound = self.eq_derivative_constraint_bounds(derivative)
            num_derivative_constraints += len(bound)
            bounds.append(bound)

        bounds.append(self.equality_constraint_bounds())
        bounds, self.names = self._sorter(*bounds)
        self.names_derivative_links = self.names[:num_derivative_links]
        # self.names_equality_constraints = self.names[num_derivative_links:]
        return cas.Expression(bounds)


class InequalityBounds(ProblemDataPart):
    """
    Format:
        derivative position+velocity bounds
        derivative acceleration bounds
        derivative jerk bounds
        inequality bounds
    """
    names: np.ndarray
    names_position_limits: np.ndarray
    names_derivative_links: np.ndarray
    names_neq_constraints: np.ndarray
    names_non_position_limits: np.ndarray

    def __init__(self, free_variables: List[FreeVariable],
                 equality_constraints: List[EqualityConstraint],
                 inequality_constraints: List[InequalityConstraint],
                 derivative_constraints: List[DerivativeInequalityConstraint],
                 eq_derivative_constraints: List[DerivativeEqualityConstraint],
                 sample_period: float,
                 prediction_horizon: int,
                 max_derivative: Derivatives,
                 default_limits: bool,
                 alpha: float,
                 qp_formulation: QPFormulation):
        super().__init__(free_variables=free_variables,
                         equality_constraints=equality_constraints,
                         inequality_constraints=inequality_constraints,
                         derivative_constraints=derivative_constraints,
                         eq_derivative_constraints=eq_derivative_constraints,
                         sample_period=sample_period,
                         prediction_horizon=prediction_horizon,
                         max_derivative=max_derivative,
                         alpha=alpha,
                         qp_formulation=qp_formulation)
        self.default_limits = default_limits
        self.evaluated = True

    def derivative_constraint_bounds(self, derivative: Derivatives) \
            -> Tuple[Dict[str, cas.Expression], Dict[str, cas.Expression]]:
        lower = {}
        upper = {}
        for t in range(self.prediction_horizon):
            for c in self.get_derivative_constraints(derivative):
                if t < self.control_horizon:
                    lower[f't{t:03}/{c.name}'] = c.lower_limit[t] * self.dt
                    upper[f't{t:03}/{c.name}'] = c.upper_limit[t] * self.dt
        return lower, upper

    def lower_inequality_constraint_bound(self):
        bounds = {}
        for constraint in self.inequality_constraints:
            if isinstance(constraint.lower_error, float) and np.isinf(constraint.lower_error):
                bounds[f'{constraint.name}'] = constraint.lower_error
            else:
                bounds[f'{constraint.name}'] = constraint.capped_lower_error(self.dt, self.control_horizon)
        return bounds

    def upper_inequality_constraint_bound(self):
        bounds = {}
        for constraint in self.inequality_constraints:
            if isinstance(constraint.upper_error, float) and np.isinf(constraint.upper_error):
                bounds[f'{constraint.name}'] = constraint.upper_error
            else:
                bounds[f'{constraint.name}'] = constraint.capped_upper_error(self.dt, self.control_horizon)
        return bounds

    def implicit_pos_model_limits(self) -> Tuple[List[Dict[str, cas.Expression]], List[Dict[str, cas.Expression]]]:
        lb_acc, ub_acc = {}, {}
        lb_jerk, ub_jerk = {}, {}
        for v in self.free_variables:
            lb_, ub_ = v.get_lower_limit(Derivatives.position), v.get_upper_limit(Derivatives.position)
            for t in range(self.prediction_horizon - 2):
                ptc = v.get_symbol(Derivatives.position)
                lb_jerk[f't{t:03}/{v.name}/{Derivatives.position}'] = lb_ - ptc
                ub_jerk[f't{t:03}/{v.name}/{Derivatives.position}'] = ub_ - ptc
        return [lb_acc, lb_jerk], [ub_acc, ub_jerk]

    def implicit_model_limits(self) -> Tuple[List[Dict[str, cas.Expression]], List[Dict[str, cas.Expression]]]:
        lb_acc, ub_acc = {}, {}
        lb_jerk, ub_jerk = {}, {}
        for v in self.free_variables:
            if self.qp_formulation.explicit_pos_limits():
                lb_, ub_ = v.get_lower_limit(Derivatives.jerk), v.get_upper_limit(Derivatives.jerk)
            else:
                lb_, ub_ = self.velocity_limit(v=v, max_derivative=Derivatives.jerk)
            for t in range(self.prediction_horizon):
                # if self.max_derivative >= Derivatives.acceleration:
                #     a_min = v.get_lower_limit(Derivatives.acceleration)
                #     a_max = v.get_upper_limit(Derivatives.acceleration)
                #     if not ((np.isinf(a_min) or cas.is_inf(a_min)) and (np.isinf(a_max) or cas.is_inf(a_max))):
                #         vtc = v.get_symbol(Derivatives.velocity)
                #         if t == 0:
                #             # vtc/dt + a_min <= vt0/dt <= vtc/dt + a_max
                #             lb_acc[f't{t:03}/{v.name}/{Derivatives.acceleration}'] = vtc / self.dt + a_min
                #             ub_acc[f't{t:03}/{v.name}/{Derivatives.acceleration}'] = vtc / self.dt + a_max
                #         else:
                #             lb_acc[f't{t:03}/{v.name}/{Derivatives.acceleration}'] = a_min
                #             ub_acc[f't{t:03}/{v.name}/{Derivatives.acceleration}'] = a_max
                if self.max_derivative >= Derivatives.jerk:
                    if self.qp_formulation.explicit_pos_limits():
                        j_min = lb_
                        j_max = ub_
                    else:
                        j_min = lb_[self.prediction_horizon * 2 + t]
                        j_max = ub_[self.prediction_horizon * 2 + t]
                    vtc = v.get_symbol(Derivatives.velocity)
                    atc = v.get_symbol(Derivatives.acceleration)
                    if t == 0:
                        # vtc/dt**2 + atc/dt + j_min <=    vt0/dt**2     <= vtc/dt**2 + atc/dt + j_max
                        lb_jerk[f't{t:03}/{v.name}/{Derivatives.jerk}'] = vtc / self.dt ** 2 + atc / self.dt + j_min
                        ub_jerk[f't{t:03}/{v.name}/{Derivatives.jerk}'] = vtc / self.dt ** 2 + atc / self.dt + j_max
                    elif t == 1:
                        # (- vtc)/dt**2 + j_min <= (vt1 - 2vt0)/dt**2 <= (- vtc)/dt**2 + j_max
                        lb_jerk[f't{t:03}/{v.name}/{Derivatives.jerk}'] = -vtc / self.dt ** 2 + j_min
                        ub_jerk[f't{t:03}/{v.name}/{Derivatives.jerk}'] = -vtc / self.dt ** 2 + j_max
                    else:
                        lb_jerk[f't{t:03}/{v.name}/{Derivatives.jerk}'] = j_min
                        ub_jerk[f't{t:03}/{v.name}/{Derivatives.jerk}'] = j_max
        return [lb_acc, lb_jerk], [ub_acc, ub_jerk]

    @profile
    def construct_expression(self) -> Union[cas.Expression, Tuple[cas.Expression, cas.Expression]]:
        lb_params: List[Dict[str, cas.Expression]] = []
        ub_params: List[Dict[str, cas.Expression]] = []

        if self.qp_formulation.explicit_pos_limits():
            lb, ub = self.implicit_pos_model_limits()
            lb_params.extend(lb)
            ub_params.extend(ub)

        if self.qp_formulation.is_implicit():
            lb, ub = self.implicit_model_limits()
            lb_params.extend(lb)
            ub_params.extend(ub)

        num_derivative_constraints = 0
        for derivative in Derivatives.range(Derivatives.velocity, self.max_derivative):
            lower, upper = self.derivative_constraint_bounds(derivative)
            num_derivative_constraints += len(lower)
            lb_params.append(lower)
            ub_params.append(upper)

        lower_inequality_constraint_bounds = self.lower_inequality_constraint_bound()
        lb_params.append(lower_inequality_constraint_bounds)
        ub_params.append(self.upper_inequality_constraint_bound())
        num_neq_constraints = len(lower_inequality_constraint_bounds)

        lbA, self.names = self._sorter(*lb_params)
        ubA, _ = self._sorter(*ub_params)

        self.names_derivative_links = self.names[:num_derivative_constraints]
        self.names_neq_constraints = self.names[num_derivative_constraints + num_neq_constraints:]

        return cas.Expression(lbA), cas.Expression(ubA)


class EqualityModel(ProblemDataPart):
    """
    Format:
        last free variable velocity
        0
        last free variable acceleration
        0
        equality_constraint_bounds
    """

    def equality_constraint_expressions(self) -> List[cas.Expression]:
        return self._sorter({c.name: c.expression for c in self.equality_constraints})[0]

    def get_free_variable_symbols(self, order: Derivatives) -> List[cas.Symbol]:
        return self._sorter({v.position_name: v.get_symbol(order) for v in self.free_variables})[0]

    def get_eq_derivative_constraint_expressions(self, derivative: Derivatives):
        return \
            self._sorter({c.name: c.expression for c in self.eq_derivative_constraints if c.derivative == derivative})[
                0]

    def velocity_constraint_model(self) -> Tuple[cas.Expression, cas.Expression]:
        """
        model
        |   t1   |   t2   |   t3   |   t1   |   t2   |   t3   |   t1   |   t2   |   t3   | prediction horizon
        |v1 v2 v3|v1 v2 v3|v1 v2 v3|a1 a2 a3|a1 a2 a3|a1 a2 a3|j1 j2 j3|j1 j2 j3|j1 j2 j3| free variables
        |--------------------------------------------------------------------------------|
        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |        |
        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |        |
        |--------------------------------------------------------------------------------|
        |        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |
        |        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |
        |--------------------------------------------------------------------------------|

        slack model
        |   t1 |   t2 | prediction horizon
        |s1 s2 |s1 s2 | slack
        |------|------|
        |sp    |      | vel constr 1
        |   sp |      | vel constr 2
        |-------------|
        |      |sp    | vel constr 1
        |      |   sp | vel constr 2
        |-------------|
        """
        number_of_vel_rows = len(self.velocity_eq_constraints) * (self.prediction_horizon - 2)
        if number_of_vel_rows > 0:
            expressions = cas.Expression(self.get_eq_derivative_constraint_expressions(Derivatives.velocity))
            parts = []
            for derivative in Derivatives.range(Derivatives.position, self.max_derivative - 1):
                if derivative == Derivatives.velocity and not self.qp_formulation.has_acc_variables():
                    continue
                if derivative == Derivatives.acceleration and not self.qp_formulation.has_jerk_variables():
                    continue
                J_vel = cas.jacobian(expressions=expressions,
                                     symbols=self.get_free_variable_symbols(derivative)) * self.dt
                missing_variables = self.max_derivative - derivative - 1
                eye = cas.eye(self.prediction_horizon)[:-2, :self.prediction_horizon - missing_variables]
                J_vel_limit_block = cas.kron(eye, J_vel)
                parts.append(J_vel_limit_block)

            # constraint slack
            model = cas.hstack(parts)
            num_slack_variables = sum(self.control_horizon for c in self.velocity_eq_constraints)
            slack_model = cas.eye(num_slack_variables) * self.dt
            return model, slack_model
        return cas.Expression(), cas.Expression()

    @property
    def number_of_non_slack_columns(self) -> int:
        if self.qp_formulation.is_explicit():
            return self.number_of_free_variables * self.prediction_horizon * self.max_derivative
        elif self.qp_formulation.is_implicit():
            return self.number_of_free_variables * (self.prediction_horizon - 2)
        elif self.qp_formulation.is_explicit_no_acc():
            return self.number_of_free_variables * (
                    self.prediction_horizon - 2) + self.number_of_free_variables * self.prediction_horizon
        return self.number_of_free_variables * self.prediction_horizon * self.max_derivative

    @profile
    def derivative_link_model(self, max_derivative: Derivatives) -> cas.Expression:
        """
        Layout for prediction horizon 5
        Slots are matrices of |controlled variables| x |controlled variables|
        | vt0 | vt1 | vt2 | at0 | at1 | at2 | at3 | jt0 | jt1 | jt2 | jt3 | jt4 |
        |-----------------------------------------------------------------------|
        |  1  |     |     | -dt |     |     |     |     |     |     |     |     | last_v =  vt0 - at0*cdt
        | -1  |  1  |     |     | -dt |     |     |     |     |     |     |     |      0 = -vt0 + vt1 - at1 * mdt
        |     | -1  |  1  |     |     | -dt |     |     |     |     |     |     |      0 = -vt1 + vt2 - at2 * mdt
        |     |     | -1  |     |     |     | -dt |     |     |     |     |     |      0 = -vt2 - at3 * mdt
        |=======================================================================|
        |     |     |     |  1  |     |     |     | -dt |     |     |     |     | last_a =  at0 - jt0*cdt
        |     |     |     | -1  |  1  |     |     |     | -dt |     |     |     |      0 = -at0 + at1 - jt1 * mdt
        |     |     |     |     | -1  |  1  |     |     |     | -dt |     |     |      0 = -at1 + at2 - jt2 * mdt
        |     |     |     |     |     | -1  |  1  |     |     |     | -dt |     |      0 = -at2 + at3 - jt3 * mdt
        |     |     |     |     |     |     | -1  |     |     |     |     | -dt |      0 = -at3 - jt4 * mdt
        |-----------------------------------------------------------------------|
        """
        num_rows = self.number_of_free_variables * self.prediction_horizon * (max_derivative - 1)
        num_columns = self.number_of_free_variables * self.prediction_horizon * max_derivative
        derivative_link_model = cas.zeros(num_rows, num_columns)

        x_n = cas.eye(num_rows)
        derivative_link_model[:, :x_n.shape[0]] += x_n

        xd_n = -cas.eye(num_rows) * self.dt
        h_offset = self.number_of_free_variables * self.prediction_horizon
        derivative_link_model[:, h_offset:] += xd_n

        x_c_height = self.number_of_free_variables * (self.prediction_horizon - 1)
        x_c = -cas.eye(x_c_height)
        offset_v = 0
        offset_h = 0
        for derivative in Derivatives.range(Derivatives.velocity, max_derivative - 1):
            offset_v += self.number_of_free_variables
            derivative_link_model[offset_v:offset_v + x_c_height, offset_h:offset_h + x_c_height] += x_c
            offset_v += x_c_height
            offset_h += self.prediction_horizon * self.number_of_free_variables
        derivative_link_model = self._remove_rows_columns_where_variables_are_zero(derivative_link_model)
        return derivative_link_model

    @profile
    def derivative_link_model_no_acc(self, max_derivative: Derivatives) -> cas.Expression:
        """
        Layout for prediction horizon 5
        Slots are matrices of |controlled variables| x |controlled variables|
        | vt0 | vt1 | vt2 | jt0 | jt1 | jt2 | jt3 | jt4 |
        |-----------------------------------------------|
        | -1  |     |     |dt**2|     |     |     |     |       -vc - ac*dt = -v0 + j0*dt**2
        |  2  | -1  |     |     |dt**2|     |     |     |                vc = -v1 + 2*v0 + j1*dt**2
        | -1  |  2  | -1  |     |     |dt**2|     |     |                 0 = -vt + 2*vt-1 - vt-2 + jt*dt**2
        |     | -1  |  2  |     |     |     |dt**2|     |                 0 =   0 + 2*vt-1 - vt-2 + jt*dt**2
        |     |     | -1  |     |     |     |     |dt**2|                 0 =   0 +    0   - vt-2 + jt*dt**2
        |-----------------------------------------------|
        vt = vt-1 + at * dt     <=>  vt/dt - vt-1/dt = at
        at = at-1 + jt * dt     <=>  at/dt - at-1/dt = jt

        vt = vt-1 + (at-1 + jt * dt) * dt
        vt = vt-1 + at-1*dt + jt * dt**2
        vt = vt-1 + (vt-1/dt - vt-2/dt)*dt + jt * dt**2
        vt = vt-1 + vt-1 - vt-2 + jt*dt**2

        0 = -v1 + 2*v0 - vc + j1*dt**2
        vc = -v1 + 2*v0 + j1*dt**2

        a0/dt - ac/dt = j0
        (v0/dt - vc/dt)/dt - ac/dt = j0
        v0/dt**2 - vc/dt**2 - ac/dt = j0
        -vc/dt**2 - ac/dt = -v0/dt**2 + j0
        -vc - ac*dt = -v0 + j0*dt**2

        v0 = vc + ac*dt + j0*dt**2
        vc = -v1 + 2*v0 + j1*dt**2
        v1 = -vc + 2*v0 + j1*dt**2
        """
        n_vel = self.number_of_free_variables * (self.prediction_horizon - 2)
        n_jerk = self.number_of_free_variables * self.prediction_horizon
        model = cas.zeros(rows=n_jerk,
                          columns=n_jerk + n_vel)
        pre_previous = -cas.eye(n_vel)
        same = pre_previous
        previous = -2 * pre_previous
        j_same = cas.eye(n_jerk) * self.dt ** 2
        model[:-self.number_of_free_variables * 2, :n_vel] += pre_previous
        model[self.number_of_free_variables:-self.number_of_free_variables, :n_vel] += previous
        model[self.number_of_free_variables * 2:, :n_vel] += same
        model[:, n_vel:] = j_same
        return model

    def _remove_rows_columns_where_variables_are_zero(self, derivative_link_model: cas.Expression) -> cas.Expression:
        if np.prod(derivative_link_model.shape) == 0:
            return derivative_link_model
        row_ids = []
        end = 0
        for derivative in Derivatives.range(Derivatives.velocity, self.max_derivative - 1):
            last_non_zero_variable = self.prediction_horizon - (self.max_derivative - derivative - 1)
            start = end + self.number_of_free_variables * last_non_zero_variable
            end += self.number_of_free_variables * self.prediction_horizon
            row_ids.extend(range(start, end))
        derivative_link_model.remove(row_ids, [])
        return derivative_link_model

    @profile
    def equality_constraint_model(self) -> Tuple[cas.Expression, cas.Expression]:
        """
        |   t1   |   t2   |   t1   |   t2   |   t1   |   t2   |   t1   |   t2   | prediction horizon
        |v1 v2 v3|v1 v2 v3|a1 a2 a3|a1 a2 a3|j1 j2 j3|j1 j2 j3|s1 s2 s3|s1 s2 s3| free variables / slack
        |-----------------------------------------------------------------------|
        |  J1*sp |  J1*sp |  J2*sp |  J2*sp |  J3*sp | J3*sp  | sp*ch  | sp*ch  |
        |-----------------------------------------------------------------------|

        explicit no acc
        |   t1   |   t2   |   t1   |   t2   |   t1   |   t2   | prediction horizon
        |v1 v2 v3|v1 v2 v3|j1 j2 j3|j1 j2 j3|s1 s2 s3|s1 s2 s3| free variables / slack
        |-----------------------------------------------------|
        |  J1*sp |  J1*sp |  J3*sp | J3*sp  | sp*ch  | sp*ch  |
        |-----------------------------------------------------|
        """
        if self.qp_formulation.is_explicit_no_acc():
            if len(self.equality_constraints) > 0:
                model = cas.zeros(len(self.equality_constraints), self.number_of_non_slack_columns)
                J_eq = cas.jacobian(expressions=cas.Expression(self.equality_constraint_expressions()),
                                    symbols=self.get_free_variable_symbols(Derivatives.position)) * self.dt
                J_hstack = cas.hstack([J_eq for _ in range(self.prediction_horizon - 2)])
                # set jacobian entry to 0 if control horizon shorter than prediction horizon
                horizontal_offset = J_hstack.shape[1]
                model[:, horizontal_offset * 0:horizontal_offset * 1] = J_hstack

                # slack variable for total error
                slack_model = cas.diag(
                    cas.Expression([self.dt for c in self.equality_constraints]))
                return model, slack_model
        else:
            if self.qp_formulation.is_implicit():
                max_derivative = Derivatives.velocity
            else:
                max_derivative = self.max_derivative
            if len(self.equality_constraints) > 0:
                model = cas.zeros(len(self.equality_constraints), self.number_of_non_slack_columns)
                for derivative in Derivatives.range(Derivatives.position, max_derivative - 1):
                    J_eq = cas.jacobian(expressions=cas.Expression(self.equality_constraint_expressions()),
                                        symbols=self.get_free_variable_symbols(derivative)) * self.dt
                    if self.qp_formulation.is_explicit() or self.qp_formulation.is_no_mpc():
                        J_hstack = cas.hstack([J_eq for _ in range(self.prediction_horizon)])
                        horizontal_offset = J_hstack.shape[1]
                        model[:, horizontal_offset * derivative:horizontal_offset * (derivative + 1)] = J_hstack
                    else:
                        J_hstack = cas.hstack([J_eq for _ in range(self.prediction_horizon - 2)])
                        horizontal_offset = J_hstack.shape[1]
                        model[:, horizontal_offset * 0:horizontal_offset * 1] = J_hstack

                # slack variable for total error
                slack_model = cas.diag(
                    cas.Expression([self.dt * self.control_horizon for c in self.equality_constraints]))
                return model, slack_model
        return cas.Expression(), cas.Expression()

    @profile
    def construct_expression(self) -> Union[cas.Expression, Tuple[cas.Expression, cas.Expression]]:
        max_derivative = Derivatives.velocity
        derivative_link_model = cas.Expression()
        if self.qp_formulation.is_mpc():
            if self.qp_formulation.is_explicit():
                max_derivative = self.max_derivative
                derivative_link_model = self.derivative_link_model(max_derivative)
                derivative_link_model = self._remove_columns_columns_where_variables_are_zero(derivative_link_model,
                                                                                              max_derivative)
            elif self.qp_formulation.is_explicit_no_acc():
                max_derivative = Derivatives.velocity
                derivative_link_model = self.derivative_link_model_no_acc(self.max_derivative)
        equality_constraint_model, equality_constraint_slack_model = self.equality_constraint_model()
        if self.qp_formulation.is_explicit():
            equality_constraint_model = self._remove_columns_columns_where_variables_are_zero(equality_constraint_model,
                                                                                              max_derivative)
        vel_constr_model, vel_constr_slack_model = self.velocity_constraint_model()

        model_parts = []
        slack_model_parts = []
        if len(derivative_link_model) > 0:
            model_parts.append(derivative_link_model)
        if len(equality_constraint_model) > 0:
            model_parts.append(equality_constraint_model)
            slack_model_parts.append(equality_constraint_slack_model)
        if len(vel_constr_model) > 0:
            model_parts.append(vel_constr_model)
            slack_model_parts.append(vel_constr_slack_model)
        model = cas.vstack(model_parts)
        slack_model = cas.vstack(slack_model_parts)

        slack_model = cas.vstack([cas.zeros(derivative_link_model.shape[0],
                                            slack_model.shape[1]),
                                  slack_model])
        # model = self._remove_columns_columns_where_variables_are_zero(model, max_derivative)
        return model, slack_model


class InequalityModel(ProblemDataPart):
    """
    Format:
        implicit model
        velocity constraints
        acceleration constraints
        jerk constraints
        inequality constraints
    """

    @property
    def number_of_free_variables(self):
        return len(self.free_variables)

    @property
    def number_of_non_slack_columns(self) -> int:
        if self.qp_formulation.is_explicit():
            return self.number_of_free_variables * self.prediction_horizon * self.max_derivative
        elif self.qp_formulation.is_implicit():
            return self.number_of_free_variables * (self.prediction_horizon - 2)
        elif self.qp_formulation.is_explicit_no_acc():
            return self.number_of_free_variables * (
                    self.prediction_horizon - 2) + self.number_of_free_variables * self.prediction_horizon
        return self.number_of_free_variables * self.prediction_horizon * self.max_derivative

    @memoize
    def num_position_limits(self):
        return self.number_of_free_variables - self.num_of_continuous_joints()

    @memoize
    def num_of_continuous_joints(self):
        return len([v for v in self.free_variables if not v.has_position_limits()])

    def inequality_constraint_expressions(self) -> List[cas.Expression]:
        return self._sorter({c.name: c.expression for c in self.inequality_constraints})[0]

    def get_derivative_constraint_expressions(self, derivative: Derivatives):
        return self._sorter({c.name: c.expression for c in self.derivative_constraints if c.derivative == derivative})[
            0]

    def get_free_variable_symbols(self, order: Derivatives):
        return self._sorter({v.position_name: v.get_symbol(order) for v in self.free_variables})[0]

    def velocity_constraint_model(self) -> Tuple[cas.Expression, cas.Expression]:
        """
        model
        |   t1   |   t2   |   t3   |   t1   |   t2   |   t3   |   t1   |   t2   |   t3   | prediction horizon
        |v1 v2 v3|v1 v2 v3|v1 v2 v3|a1 a2 a3|a1 a2 a3|a1 a2 a3|j1 j2 j3|j1 j2 j3|j1 j2 j3| free variables
        |--------------------------------------------------------------------------------|
        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |        |
        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |        |
        |--------------------------------------------------------------------------------|
        |        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |
        |        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |
        |--------------------------------------------------------------------------------|

        slack model
        |   t1 |   t2 | prediction horizon
        |s1 s2 |s1 s2 | slack
        |------|------|
        |sp    |      | vel constr 1
        |   sp |      | vel constr 2
        |-------------|
        |      |sp    | vel constr 1
        |      |   sp | vel constr 2
        |-------------|
        """
        number_of_vel_rows = len(self.velocity_constraints) * (self.prediction_horizon - 2)
        if number_of_vel_rows > 0:
            expressions = cas.Expression(self.get_derivative_constraint_expressions(Derivatives.velocity))
            parts = []
            for derivative in Derivatives.range(Derivatives.position, self.max_derivative - 1):
                if derivative == Derivatives.velocity and not self.qp_formulation.has_acc_variables():
                    continue
                if derivative == Derivatives.acceleration and not self.qp_formulation.has_jerk_variables():
                    continue
                J_vel = cas.jacobian(expressions=expressions,
                                     symbols=self.get_free_variable_symbols(derivative)) * self.dt
                missing_variables = self.max_derivative - derivative - 1
                eye = cas.eye(self.prediction_horizon)[:-2, :self.prediction_horizon - missing_variables]
                J_vel_limit_block = cas.kron(eye, J_vel)
                parts.append(J_vel_limit_block)

            # constraint slack
            model = cas.hstack(parts)
            num_slack_variables = sum(self.control_horizon for c in self.velocity_constraints)
            slack_model = cas.eye(num_slack_variables) * self.dt
            return model, slack_model
        return cas.Expression(), cas.Expression()

    def acceleration_constraint_model(self) -> Tuple[cas.Expression, cas.Expression]:
        """
        same structure as vel constraint model
        task acceleration = Jd_q * qd + (J_q + Jd_qd) * qdd + J_qd * qddd
        """
        # FIXME no test case for this so probably buggy
        number_of_acc_rows = len(self.acceleration_constraints) * self.prediction_horizon
        if number_of_acc_rows > 0:
            expressions = cas.Expression(self.get_derivative_constraint_expressions(Derivatives.acceleration))
            assert self.max_derivative >= Derivatives.jerk
            model = cas.zeros(number_of_acc_rows, self.number_of_non_slack_columns)
            J_q = cas.jacobian(expressions=expressions,
                               symbols=self.get_free_variable_symbols(Derivatives.position)) * self.dt
            Jd_q = cas.jacobian_dot(expressions=expressions,
                                    symbols=self.get_free_variable_symbols(Derivatives.position),
                                    symbols_dot=self.get_free_variable_symbols(Derivatives.velocity)) * self.dt
            J_qd = cas.jacobian(expressions=expressions,
                                symbols=self.get_free_variable_symbols(Derivatives.velocity)) * self.dt
            Jd_qd = cas.jacobian_dot(expressions=expressions,
                                     symbols=self.get_free_variable_symbols(Derivatives.velocity),
                                     symbols_dot=self.get_free_variable_symbols(
                                         Derivatives.acceleration)) * self.dt
            J_vel_block = cas.kron(cas.eye(self.prediction_horizon), Jd_q)
            J_acc_block = cas.kron(cas.eye(self.prediction_horizon), J_q + Jd_qd)
            J_jerk_block = cas.kron(cas.eye(self.prediction_horizon), J_qd)
            horizontal_offset = self.number_of_free_variables * self.prediction_horizon
            model[:, :horizontal_offset] = J_vel_block
            model[:, horizontal_offset:horizontal_offset * 2] = J_acc_block
            model[:, horizontal_offset * 2:horizontal_offset * 3] = J_jerk_block

            # delete rows if control horizon of constraint shorter than prediction horizon
            rows_to_delete = []
            for t in range(self.prediction_horizon):
                for i, c in enumerate(self.velocity_constraints):
                    v_index = i + (t * len(self.velocity_constraints))
                    if t + 1 > self.control_horizon:
                        rows_to_delete.append(v_index)
            model.remove(rows_to_delete, [])

            # slack model
            num_slack_variables = sum(self.control_horizon for c in self.acceleration_constraints)
            slack_model = cas.eye(num_slack_variables) * self.dt
            return model, slack_model
        return cas.Expression(), cas.Expression()

    def jerk_constraint_model(self) -> Tuple[cas.Expression, cas.Expression]:
        """
        same structure as vel constraint model
        task acceleration = Jd_q * qd + (J_q + Jd_qd) * qdd + J_qd * qddd
        """
        # FIXME no test case for this so probably buggy
        number_of_jerk_rows = len(self.jerk_constraints) * self.prediction_horizon
        if number_of_jerk_rows > 0:
            expressions = cas.Expression(self.get_derivative_constraint_expressions(Derivatives.jerk))
            assert self.max_derivative >= Derivatives.snap
            model = cas.zeros(number_of_jerk_rows, self.number_of_non_slack_columns)
            J_q = self.dt * cas.jacobian(expressions=expressions,
                                         symbols=self.get_free_variable_symbols(Derivatives.position))
            Jd_q = self.dt * cas.jacobian_dot(expressions=expressions,
                                              symbols=self.get_free_variable_symbols(Derivatives.position),
                                              symbols_dot=self.get_free_variable_symbols(Derivatives.velocity))
            Jdd_q = self.dt * cas.jacobian_ddot(expressions=expressions,
                                                symbols=self.get_free_variable_symbols(Derivatives.position),
                                                symbols_dot=self.get_free_variable_symbols(Derivatives.velocity),
                                                symbols_ddot=self.get_free_variable_symbols(Derivatives.acceleration))
            J_qd = self.dt * cas.jacobian(expressions=expressions,
                                          symbols=self.get_free_variable_symbols(Derivatives.velocity))
            Jd_qd = self.dt * cas.jacobian_dot(expressions=expressions,
                                               symbols=self.get_free_variable_symbols(Derivatives.velocity),
                                               symbols_dot=self.get_free_variable_symbols(Derivatives.acceleration))
            Jdd_qd = self.dt * cas.jacobian_ddot(expressions=expressions,
                                                 symbols=self.get_free_variable_symbols(Derivatives.velocity),
                                                 symbols_dot=self.get_free_variable_symbols(Derivatives.acceleration),
                                                 symbols_ddot=self.get_free_variable_symbols(Derivatives.jerk))
            J_vel_block = cas.kron(cas.eye(self.prediction_horizon), Jdd_q)
            J_acc_block = cas.kron(cas.eye(self.prediction_horizon), 2 * Jd_q + Jdd_qd)
            J_jerk_block = cas.kron(cas.eye(self.prediction_horizon), J_q + 2 * Jd_qd)
            J_snap_block = cas.kron(cas.eye(self.prediction_horizon), J_qd)
            horizontal_offset = self.number_of_free_variables * self.prediction_horizon
            model[:, :horizontal_offset] = J_vel_block
            model[:, horizontal_offset:horizontal_offset * 2] = J_acc_block
            model[:, horizontal_offset * 2:horizontal_offset * 3] = J_jerk_block
            model[:, horizontal_offset * 3:horizontal_offset * 4] = J_snap_block

            # delete rows if control horizon of constraint shorter than prediction horizon
            rows_to_delete = []
            for t in range(self.prediction_horizon):
                for i, c in enumerate(self.velocity_constraints):
                    v_index = i + (t * len(self.velocity_constraints))
                    if t + 1 > self.control_horizon:
                        rows_to_delete.append(v_index)
            model.remove(rows_to_delete, [])

            # slack model
            num_slack_variables = sum(self.control_horizon for c in self.jerk_constraints)
            slack_model = cas.eye(num_slack_variables) * self.dt
            return model, slack_model
        return cas.Expression(), cas.Expression()

    @profile
    def inequality_constraint_model(self, max_derivative: Derivatives) -> Tuple[cas.Expression, cas.Expression]:
        """
        |   t1   |   t2   |   t1   |   t2   |   t1   |   t2   |   t1   |   t2   | prediction horizon
        |v1 v2 v3|v1 v2 v3|a1 a2 a3|a1 a2 a3|j1 j2 j3|j1 j2 j3|s1 s2 s3|s1 s2 s3| free variables / slack
        |-----------------------------------------------------------------------|
        |  J1*sp |  J1*sp |  J2*sp |  J2*sp |  J3*sp | J3*sp  | sp*ch  | sp*ch  |
        |-----------------------------------------------------------------------|
        """
        if self.qp_formulation.is_explicit_no_acc():
            if len(self.inequality_constraints) > 0:
                model = cas.zeros(len(self.inequality_constraints), self.number_of_non_slack_columns)
                J_neq = cas.jacobian(expressions=cas.Expression(self.inequality_constraint_expressions()),
                                     symbols=self.get_free_variable_symbols(Derivatives.position)) * self.dt
                J_hstack = cas.hstack([J_neq for _ in range(self.prediction_horizon - 2)])
                # set jacobian entry to 0 if control horizon shorter than prediction horizon
                horizontal_offset = J_hstack.shape[1]
                model[:, horizontal_offset * 0:horizontal_offset * 1] = J_hstack

                # slack variable for total error
                slack_model = cas.diag(
                    cas.Expression([self.dt for c in self.inequality_constraints]))
                return model, slack_model
        else:
            if self.qp_formulation.is_implicit():
                max_derivative = Derivatives.velocity
            else:
                max_derivative = self.max_derivative
        if len(self.inequality_constraints) > 0:
            model = cas.zeros(len(self.inequality_constraints), self.number_of_non_slack_columns)
            for derivative in Derivatives.range(Derivatives.position, max_derivative - 1):
                J_neq = cas.jacobian(expressions=cas.Expression(self.inequality_constraint_expressions()),
                                     symbols=self.get_free_variable_symbols(derivative)) * self.dt
                if self.qp_formulation.is_explicit() or self.qp_formulation.is_no_mpc():
                    J_hstack = cas.hstack([J_neq for _ in range(self.prediction_horizon)])
                    horizontal_offset = J_hstack.shape[1]
                    model[:, horizontal_offset * derivative:horizontal_offset * (derivative + 1)] = J_hstack
                else:
                    J_hstack = cas.hstack([J_neq for _ in range(self.prediction_horizon - 2)])
                    horizontal_offset = J_hstack.shape[1]
                    model[:, horizontal_offset * 0:horizontal_offset * 1] = J_hstack

            # slack variable for total error
            slack_model = cas.diag(
                cas.Expression([self.dt * self.control_horizon for c in self.inequality_constraints]))
            return model, slack_model
        return cas.Expression(), cas.Expression()

    def implicit_pos_limits(self) -> Tuple[cas.Expression, cas.Expression]:
        """
        pk = pk-1 + vk*dt
            p0 = pc + v0*dt
            pk/dt - pk-1/dt = vk
        vk = vk-1 + ak*dt
        ak = (vk - vk-1)/dt
        jk = (vk - 2vk-1 + vk-2)/dt**2

        Layout for prediction horizon 4
        Slots are matrices of |controlled variables| x |controlled variables|
        |  vt0   |  vt1   |  vt2   |  vt3   |
        |-----------------------------------|
        |  1*dt  |        |        |        |       pt0 - ptc = vt0*dt
        |  1*dt  |  1*dt  |        |        |       pt1 - ptc = vt0*dt + vt1*dt
        |  1*dt  |  1*dt  |  1*dt  |        |       pt2 - ptc = vt0*dt + vt1*dt + vt2*dt
        |  1*dt  |  1*dt  |  1*dt  |  1*dt  |       pt3 - ptc = vt0*dt + vt1*dt + vt2*dt + vt3*dt
        |-----------------------------------|

        :param max_derivative:
        :return:
        """
        n_vel = self.number_of_free_variables * (self.prediction_horizon - 2)
        model = cas.tri(n_vel) * self.dt
        slack_model = cas.zeros(model.shape[0], self.number_ineq_slack_variables)
        return model, slack_model

    def implicit_model(self, max_derivative: Derivatives) -> Tuple[cas.Expression, cas.Expression]:
        """
        ak = (vk - vk-1)/dt
        jk = (vk - 2vk-1 + vk-2)/dt**2

        vt0 = vtc + at0 * cdt
        vt1 = vt0 + at1 * mdt
        vt = vt-1 + at * mdt

        at = vt-1 + jt * mdt

        Layout for prediction horizon 6
        Slots are matrices of |controlled variables| x |controlled variables|
        |  vt0   |  vt1   |  vt2   |  vt3   |
        |-----------------------------------|
        |  1/dt  |        |        |        |               vtc/cdt + at0 = vt0/cdt                   vtc/dt + a_min <= vt0/dt <= vtc/dt + a_max
        | -1/dt  |  1/dt  |        |        |                        at1 = (vt1 - vt0)/dt
        |        | -1/dt  |  1/dt  |        |                        at2 = (vt2 - vt1)/mdt
        |        |        | -1/dt  | 1/dt   |                        at3 = (vt3 - vt2)/mdt
        |===================================|
        | 1/dt**2|        |        |        |   vtc/dt**2 + atc/dt + jt0 = vt0/dt**2                vtc/dt**2 + atc/dt + j_min <=    vt0/dt**2     <= vtc/dt**2 + atc/dt + j_max
        |-2/dt**2| 1/dt**2|        |        |          - vtc/dt**2 + jt1 = (vt1 - 2vt0)/dt**2           (- vtc)/dt**2 + j_min <= (vt1 - 2vt0)/dt**2 <= (- vtc)/dt**2 + j_max
        | 1/dt**2|-2/dt**2| 1/dt**2|        |                        jt2 = (vt2 - 2vt1 + vt0)/dt**2
        |        | 1/dt**2|-2/dt**2| 1/dt**2|                        jt3 = (vt3 - 2vt2 + vt1)/dt**2
        |        |        | 1/dt**2|-2/dt**2|                        jt4 = (- 2vt3 + vt2)/dt**2
        |        |        |        | 1/dt**2|                        jt5 = (vt3)/dt**2
        |-----------------------------------|

        :param max_derivative:
        :return:
        """
        n_vel = self.number_of_free_variables * (self.prediction_horizon - 2)
        n_jerk = self.number_of_free_variables * (self.prediction_horizon)
        if max_derivative >= Derivatives.acceleration:
            # previous = cas.eye(self.number_of_free_variables * (self.prediction_horizon)) / self.dt
            # same = -cas.eye(self.number_of_free_variables * (self.prediction_horizon - 1)) / self.dt
            # A_acc = previous
            # A_acc[self.number_of_free_variables:, :-self.number_of_free_variables] += same
            # rows_to_delete = []
            # for i in range(self.prediction_horizon):
            #     for v_i, v in enumerate(self.free_variables):
            #         idx = i * len(self.free_variables) + v_i
            #         a_min = v.get_lower_limit(Derivatives.acceleration)
            #         a_max = v.get_upper_limit(Derivatives.acceleration)
            #         if (np.isinf(a_min) or cas.is_inf(a_min)) and (np.isinf(a_max) or cas.is_inf(a_max)):
            #             rows_to_delete.append(idx)
            # A_acc.remove(rows_to_delete, [])
            model = cas.zeros(rows=n_jerk, columns=n_vel)
            pre_previous = cas.eye(n_vel) / self.dt ** 2
            previous = -2 * cas.eye(n_vel) / self.dt ** 2
            same = pre_previous
            model[:-self.number_of_free_variables * 2, :] += pre_previous
            model[self.number_of_free_variables:-self.number_of_free_variables, :] += previous
            model[self.number_of_free_variables * 2:, :] += same
        else:
            model = cas.Expression()
        slack_model = cas.zeros(model.shape[0], self.number_ineq_slack_variables)
        return model, slack_model

    @profile
    def construct_expression(self) -> Union[cas.Expression, Tuple[cas.Expression, cas.Expression]]:
        if self.qp_formulation.explicit_pos_limits():
            pos_model, pos_slack_model = self.implicit_pos_limits()
        if self.qp_formulation.is_implicit():
            max_derivative = Derivatives.velocity
            derivative_model, derivative_model_slack = self.implicit_model(self.max_derivative)
        else:
            max_derivative = self.max_derivative
            derivative_model, derivative_model_slack = cas.Expression(), cas.Expression()
        vel_constr_model, vel_constr_slack_model = self.velocity_constraint_model()
        # acc_constr_model, acc_constr_slack_model = self.acceleration_constraint_model()
        # jerk_constr_model, jerk_constr_slack_model = self.jerk_constraint_model()
        inequality_model, inequality_slack_model = self.inequality_constraint_model(max_derivative)
        model_parts = []
        slack_model_parts = []
        if self.qp_formulation.explicit_pos_limits():
            if len(pos_model) > 0:
                model_parts.append(pos_model)
                slack_model_parts.append(pos_slack_model)
        if len(derivative_model) > 0:
            model_parts.append(derivative_model)
            slack_model_parts.append(derivative_model_slack)
        if len(vel_constr_model) > 0:
            model_parts.append(vel_constr_model)
            slack_model_parts.append(vel_constr_slack_model)
        # if len(acc_constr_model) > 0:
        #     model_parts.append(acc_constr_model)
        #     slack_model_parts.append(acc_constr_slack_model)
        # if len(jerk_constr_model) > 0:
        #     model_parts.append(jerk_constr_model)
        #     slack_model_parts.append(jerk_constr_slack_model)
        if len(inequality_model) > 0:
            model_parts.append(inequality_model)
            slack_model_parts.append(inequality_slack_model)

        combined_model = cas.vstack(model_parts)
        combined_slack_model = cas.diag_stack(slack_model_parts)
        # combined_model = self._remove_columns_columns_where_variables_are_zero(combined_model, max_derivative)
        return combined_model, combined_slack_model


available_solvers: Dict[SupportedQPSolver, Type[QPSolver]] = {}


def detect_solvers():
    global available_solvers
    solver_name: str
    qp_solver_class: Type[QPSolver]
    for solver_name, qp_solver_class in get_all_classes_in_package('giskardpy.qp', QPSolver, silent=True).items():
        try:
            available_solvers[qp_solver_class.solver_id] = qp_solver_class
        except Exception:
            pass
    solver_names = [solver_name.name for solver_name in available_solvers.keys()]
    print(f'Found these qp solvers: {solver_names}')


detect_solvers()


class QPController:
    """
    Wraps around QP Solver. Builds the required matrices from constraints.
    """
    inequality_constraints: List[InequalityConstraint]
    equality_constraints: List[EqualityConstraint]
    derivative_constraints: List[DerivativeInequalityConstraint]
    weights: Weights
    free_variable_bounds: FreeVariableBounds
    equality_model: EqualityModel
    equality_bounds: EqualityBounds
    inequality_model: InequalityModel
    inequality_bounds: InequalityBounds
    qp_solver: QPSolver
    prediction_horizon: int = None

    @profile
    def __init__(self,
                 mpc_dt: float,
                 prediction_horizon: int,
                 control_dt: Optional[float] = None,
                 max_derivative: Derivatives = Derivatives.jerk,
                 solver_id: Optional[SupportedQPSolver] = None,
                 retries_with_relaxed_constraints: int = 5,
                 retry_added_slack: float = 100,
                 retry_weight_factor: float = 100,
                 qp_formulation: QPFormulation = QPFormulation.explicit_no_acc,
                 alpha: float = 0.1,
                 verbose: bool = True):
        if control_dt is None:
            control_dt = mpc_dt
        self.control_dt = control_dt
        self.alpha = alpha
        self.qp_formulation = qp_formulation
        self.mpc_dt = mpc_dt
        self.max_derivative = max_derivative
        self.prediction_horizon = prediction_horizon
        self.retries_with_relaxed_constraints = retries_with_relaxed_constraints
        self.retry_added_slack = retry_added_slack
        self.retry_weight_factor = retry_weight_factor
        self.verbose = verbose
        self.set_qp_solver(solver_id)
        if self.qp_formulation.is_no_mpc():
            self.prediction_horizon = 1
            self.max_derivative = Derivatives.velocity

        if self.verbose:
            get_middleware().loginfo(f'Initialized QP Controller:\n'
                                     f'sample period: "{self.mpc_dt}"s\n'
                                     f'max derivative: "{self.max_derivative.name}"\n'
                                     f'prediction horizon: "{self.prediction_horizon}"\n'
                                     f'QP solver: "{self.qp_solver_class.solver_id.name}"')
        self.reset()

    def set_qp_solver(self, solver_id: Optional[SupportedQPSolver] = None) -> None:
        print_later = hasattr(self, 'qp_solver_class')
        if solver_id is not None:
            self.qp_solver_class = available_solvers[solver_id]
        else:
            for solver_id in SupportedQPSolver:
                if solver_id in available_solvers:
                    self.qp_solver_class = available_solvers[solver_id]
                    break
            else:
                raise QPSolverException(f'No qp solver found')
        if print_later:
            get_middleware().loginfo(f'QP Solver set to "{self.qp_solver_class.solver_id.name}"')

    def reset(self):
        self.free_variables = []
        self.equality_constraints = []
        self.inequality_constraints = []
        self.derivative_constraints = []
        self.eq_derivative_constraints = []
        self.quadratic_weight_gains = []
        self.linear_weight_gains = []
        self.xdot_full = None

    def init(self,
             free_variables: List[FreeVariable] = None,
             equality_constraints: List[EqualityConstraint] = None,
             inequality_constraints: List[InequalityConstraint] = None,
             derivative_constraints: List[DerivativeInequalityConstraint] = None,
             eq_derivative_constraints: List[DerivativeEqualityConstraint] = None,
             quadratic_weight_gains: List[QuadraticWeightGain] = None,
             linear_weight_gains: List[LinearWeightGain] = None):
        self.reset()
        if free_variables is not None:
            self.add_free_variables(free_variables)
        if inequality_constraints is not None:
            self.add_inequality_constraints(inequality_constraints)
        if equality_constraints is not None:
            self.add_equality_constraints(equality_constraints)
        if derivative_constraints is not None:
            self.add_derivative_constraints(derivative_constraints)
        if eq_derivative_constraints is not None:
            self.add_eq_derivative_constraints(eq_derivative_constraints)
        if quadratic_weight_gains is not None:
            self.add_quadratic_weight_gains(quadratic_weight_gains)
        if linear_weight_gains is not None:
            self.add_linear_weight_gains(linear_weight_gains)

    def add_free_variables(self, free_variables: list):
        if len(free_variables) == 0:
            raise QPSolverException('Cannot solve qp with no free variables')
        self.free_variables.extend(list(sorted(free_variables, key=lambda x: x.position_name)))
        l = [x.position_name for x in free_variables]
        duplicates = set([x for x in l if l.count(x) > 1])
        self.order = Derivatives(min(self.prediction_horizon, self.max_derivative))
        assert duplicates == set(), f'there are free variables with the same name: {duplicates}'

    def add_inequality_constraints(self, constraints: List[InequalityConstraint]):
        self.inequality_constraints.extend(list(sorted(constraints, key=lambda x: x.name)))
        l = [x.name for x in constraints]
        duplicates = set([x for x in l if l.count(x) > 1])
        assert duplicates == set(), f'there are multiple constraints with the same name: {duplicates}'

    def add_equality_constraints(self, constraints: List[EqualityConstraint]):
        self.equality_constraints.extend(list(sorted(constraints, key=lambda x: x.name)))
        l = [x.name for x in constraints]
        duplicates = set([x for x in l if l.count(x) > 1])
        assert duplicates == set(), f'there are multiple constraints with the same name: {duplicates}'

    def add_quadratic_weight_gains(self, gains: List[QuadraticWeightGain]):
        self.quadratic_weight_gains.extend(list(sorted(gains, key=lambda x: x.name)))
        l = [x.name for x in gains]
        duplicates = set([x for x in l if l.count(x) > 1])
        assert duplicates == set(), f'there are multiple quadratic weight gains with the same name: {duplicates}'

    def add_linear_weight_gains(self, gains: List[LinearWeightGain]):
        self.linear_weight_gains.extend(list(sorted(gains, key=lambda x: x.name)))
        l = [x.name for x in gains]
        duplicates = set([x for x in l if l.count(x) > 1])
        assert duplicates == set(), f'there are multiple linear weight gains with the same name: {duplicates}'

    def add_derivative_constraints(self, constraints: List[DerivativeInequalityConstraint]):
        self.derivative_constraints.extend(list(sorted(constraints, key=lambda x: x.name)))
        l = [x.name for x in constraints]
        duplicates = set([x for x in l if l.count(x) > 1])
        assert duplicates == set(), f'there are multiple constraints with the same name: {duplicates}'

    def add_eq_derivative_constraints(self, constraints: List[DerivativeEqualityConstraint]):
        self.eq_derivative_constraints.extend(list(sorted(constraints, key=lambda x: x.name)))
        l = [x.name for x in constraints]
        duplicates = set([x for x in l if l.count(x) > 1])
        assert duplicates == set(), f'there are multiple constraints with the same name: {duplicates}'

    def check_control_horizon(self, constraint):
        if constraint.control_horizon is None:
            constraint.control_horizon = self.prediction_horizon
        elif constraint.control_horizon <= 0 or not isinstance(constraint.control_horizon, int):
            raise ValueError(f'Control horizon of {constraint.name} is {constraint.control_horizon}, '
                             f'it has to be an integer 1 <= control horizon <= prediction horizon')
        elif constraint.control_horizon > self.prediction_horizon:
            get_middleware().logwarn(
                f'Specified control horizon of {constraint.name} is bigger than prediction horizon.'
                f'Reducing control horizon of {constraint.control_horizon} '
                f'to prediction horizon of {self.prediction_horizon}')
            constraint.control_horizon = self.prediction_horizon

    @profile
    def compile(self, default_limits: bool = False) -> None:
        if self.verbose:
            get_middleware().loginfo('Creating controller')
        kwargs = {'free_variables': self.free_variables,
                  'equality_constraints': self.equality_constraints,
                  'inequality_constraints': self.inequality_constraints,
                  'derivative_constraints': self.derivative_constraints,
                  'eq_derivative_constraints': self.eq_derivative_constraints,
                  'sample_period': self.mpc_dt,
                  'prediction_horizon': self.prediction_horizon,
                  'max_derivative': self.order,
                  'alpha': self.alpha,
                  'qp_formulation': self.qp_formulation}
        self.weights = Weights(**kwargs)
        self.free_variable_bounds = FreeVariableBounds(**kwargs)
        self.equality_model = EqualityModel(**kwargs)
        self.equality_bounds = EqualityBounds(**kwargs)
        self.inequality_model = InequalityModel(**kwargs)
        self.inequality_bounds = InequalityBounds(default_limits=default_limits, **kwargs)

        weights, g = self.weights.construct_expression(self.quadratic_weight_gains, self.linear_weight_gains)
        lb, ub = self.free_variable_bounds.construct_expression()
        A, A_slack = self.inequality_model.construct_expression()
        lbA, ubA = self.inequality_bounds.construct_expression()
        E, E_slack = self.equality_model.construct_expression()
        bE = self.equality_bounds.construct_expression()

        self.qp_solver = self.qp_solver_class(weights=weights, g=g, lb=lb, ub=ub,
                                              E=E, E_slack=E_slack, bE=bE,
                                              A=A, A_slack=A_slack, lbA=lbA, ubA=ubA)
        self.qp_solver.update_settings(retries_with_relaxed_constraints=self.retries_with_relaxed_constraints,
                                       retry_added_slack=self.retry_added_slack,
                                       retry_weight_factor=self.retry_weight_factor)

        self.num_free_variables = weights.shape[0]
        self.num_eq_constraints = bE.shape[0]
        self.num_ineq_constraints = lbA.shape[0] * 2
        if self.verbose:
            get_middleware().loginfo('Done compiling controller:')
            get_middleware().loginfo(f'  #free variables: {self.num_free_variables}')
            get_middleware().loginfo(f'  #equality constraints: {self.num_eq_constraints}')
            get_middleware().loginfo(f'  #inequality constraints: {self.num_ineq_constraints}')

    def get_parameter_names(self):
        return self.qp_solver.free_symbols_str

    def save_all_pandas(self, folder_name: Optional[str] = None):
        self._create_debug_pandas(self.qp_solver)
        save_pandas(
            [self.p_weights, self.p_b,
             self.p_E, self.p_bE,
             self.p_A, self.p_lbA, self.p_ubA,
             god_map.debug_expression_manager.to_pandas(), self.p_xdot],
            ['weights', 'b', 'E', 'bE', 'A', 'lbA', 'ubA', 'debug'],
            god_map.tmp_folder,
            god_map.time,
            folder_name)

    def _print_pandas_array(self, array):
        import pandas as pd
        if len(array) > 0:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(array)

    @profile
    def get_cmd(self, substitutions: np.ndarray) -> NextCommands:
        """
        Uses substitutions for each symbol to compute the next commands for each joint.
        """
        try:
            self.xdot_full = self.qp_solver.solve_and_retry(substitutions=substitutions)
            # self._create_debug_pandas(self.qp_solver)
            if self.qp_formulation.is_implicit():
                return NextCommands.from_xdot_implicit(self.free_variables, self.xdot_full, self.order,
                                                       self.prediction_horizon, god_map.world, self.mpc_dt)
            elif self.qp_formulation.is_explicit() or self.qp_formulation.is_no_mpc():
                return NextCommands.from_xdot(self.free_variables, self.xdot_full, self.order, self.prediction_horizon)
            else:
                return NextCommands.from_xdot_explicit_no_acc(self.free_variables, self.xdot_full, self.order,
                                                              self.prediction_horizon, god_map.world,
                                                              self.mpc_dt)
        except InfeasibleException as e_original:
            self.xdot_full = None
            self._create_debug_pandas(self.qp_solver)
            self._has_nan()
            self._print_iis()
            if isinstance(e_original, HardConstraintsViolatedException):
                raise
            self.xdot_full = None
            self._are_hard_limits_violated(str(e_original))
            raise

    def _has_nan(self):
        nan_entries = self.p_A.isnull().stack()
        row_col_names = nan_entries[nan_entries].index.tolist()
        pass

    def _are_hard_limits_violated(self, error_message):
        self._create_debug_pandas(self.qp_solver)
        try:
            lower_violations = self.p_lb[self.qp_solver.lb_filter]
            upper_violations = self.p_ub[self.qp_solver.ub_filter]
            if len(upper_violations) > 0 or len(lower_violations) > 0:
                error_message += '\n'
                if len(upper_violations) > 0:
                    error_message += 'upper slack bounds of following constraints might be too low: {}\n'.format(
                        list(upper_violations.index))
                if len(lower_violations) > 0:
                    error_message += 'lower slack bounds of following constraints might be too high: {}'.format(
                        list(lower_violations.index))
                raise HardConstraintsViolatedException(error_message)
        except AttributeError:
            pass
        get_middleware().loginfo('No slack limit violation detected.')

    def _viz_mpc(self, joint_name):
        def pad(a, desired_length, pad_value):
            tmp = np.ones(desired_length) * pad_value
            tmp[:len(a)] = a
            return tmp

        free_variable: FreeVariable = [x for x in self.free_variables if x.name == joint_name][0]
        try:
            start_pos = god_map.world.state[joint_name].position
        except KeyError:
            get_middleware().loginfo('start position not found in state')
            start_pos = 0
        ts = np.array([(i + 1) * self.mpc_dt for i in range(self.prediction_horizon)])
        filtered_x = self.p_xdot.filter(like=f'/{joint_name}/', axis=0)
        vel_end = self.prediction_horizon - self.order + 1
        acc_end = vel_end + self.prediction_horizon - self.order + 2
        velocities = filtered_x[:vel_end].values
        positions = [start_pos]
        for x_ in velocities:
            positions.append(positions[-1] + x_ * self.mpc_dt)

        positions = np.array(positions[1:])
        positions = pad(positions.T[0], len(ts), pad_value=positions[-1])
        velocities = pad(velocities.T[0], len(ts), pad_value=0)

        if joint_name in god_map.world.state:
            accelerations = filtered_x[vel_end:acc_end].values
            jerks = filtered_x[acc_end:].values
            accelerations = pad(accelerations.T[0], len(ts), pad_value=0)

        f, axs = plt.subplots(4, sharex=True, figsize=(2 + self.prediction_horizon, 16))
        axs[0].set_title('position')
        axs[0].plot(ts, positions, 'b')
        axs[0].grid()
        axs[1].set_title('velocity')
        axs[1].plot(ts, velocities, 'b')
        axs[1].grid()
        if joint_name in god_map.world.state:
            axs[2].set_title('acceleration')
            axs[2].plot(ts, accelerations, 'b')
            axs[2].grid()
            axs[3].set_title('jerk')
            axs[3].plot(ts, jerks, 'b')
            axs[3].grid()
        for i, ax in enumerate(axs):
            derivative = Derivatives(i)
            if not free_variable.has_position_limits():
                continue
            upper_limit = free_variable.get_upper_limit(derivative, evaluated=True)
            if not np.isinf(upper_limit):
                ax.axhline(y=upper_limit, color='k', linestyle='--')
            lower_limit = free_variable.get_lower_limit(derivative, evaluated=True)
            if not np.isinf(lower_limit):
                ax.axhline(y=lower_limit, color='k', linestyle='--')
        # Example: Set x-ticks for each subplot
        tick_labels = [f'{x}/{x * self.mpc_dt:.3f}' for x in range(self.prediction_horizon)]

        axs[-1].set_xticks(ts)  # Set tick locations
        axs[-1].set_xticklabels(tick_labels)  # Set custom tick labels

        plt.tight_layout()
        path, dirs, files = next(os.walk('tmp_data/mpc'))
        file_count = len(files)
        file_name = f'{god_map.tmp_folder}/mpc/mpc_{joint_name}_{file_count}.png'
        create_path(file_name)
        plt.savefig(file_name)

    @profile
    def _create_debug_pandas(self, qp_solver: QPSolver):
        weights, g, lb, ub, E, bE, A, lbA, ubA, weight_filter, bE_filter, bA_filter = qp_solver.get_problem_data()
        self.free_variable_names = self.free_variable_bounds.names[weight_filter]
        self.equality_constr_names = self.equality_bounds.names[bE_filter]
        self.inequality_constr_names = self.inequality_bounds.names[bA_filter]
        num_vel_constr = len(self.derivative_constraints) * (self.prediction_horizon - 2)
        num_eq_vel_constr = len(self.eq_derivative_constraints) * (self.prediction_horizon - 2)
        num_neq_constr = len(self.inequality_constraints)
        num_eq_constr = len(self.equality_constraints)
        num_constr = num_vel_constr + num_neq_constr + num_eq_constr + num_eq_vel_constr

        self.p_weights = pd.DataFrame(weights, self.free_variable_names, ['data'], dtype=float)
        self.p_g = pd.DataFrame(g, self.free_variable_names, ['data'], dtype=float)
        self.p_lb = pd.DataFrame(lb, self.free_variable_names, ['data'], dtype=float)
        self.p_ub = pd.DataFrame(ub, self.free_variable_names, ['data'], dtype=float)
        self.p_b = pd.DataFrame({'lb': lb, 'ub': ub}, self.free_variable_names, dtype=float)
        if len(bE) > 0:
            self.p_bE_raw = pd.DataFrame(bE, self.equality_constr_names, ['data'], dtype=float)
            self.p_bE = deepcopy(self.p_bE_raw)
            self.p_bE[len(self.equality_bounds.names_derivative_links):] /= self.mpc_dt
        else:
            self.p_bE = pd.DataFrame()
        if len(lbA) > 0:
            self.p_lbA_raw = pd.DataFrame(lbA, self.inequality_constr_names, ['data'], dtype=float)
            self.p_lbA = deepcopy(self.p_lbA_raw)
            self.p_lbA /= self.mpc_dt

            self.p_ubA_raw = pd.DataFrame(ubA, self.inequality_constr_names, ['data'], dtype=float)
            self.p_ubA = deepcopy(self.p_ubA_raw)
            self.p_ubA /= self.mpc_dt

            self.p_bA_raw = pd.DataFrame({'lbA': lbA, 'ubA': ubA}, self.inequality_constr_names, dtype=float)
            self.p_bA = deepcopy(self.p_bA_raw)
            self.p_bA /= self.mpc_dt
        else:
            self.p_lbA = pd.DataFrame()
            self.p_ubA = pd.DataFrame()
        # remove sample period factor
        if len(E) > 0:
            self.p_E = pd.DataFrame(E, self.equality_constr_names, self.free_variable_names, dtype=float)
        else:
            self.p_E = pd.DataFrame()
        if len(A) > 0:
            self.p_A = pd.DataFrame(A, self.inequality_constr_names, self.free_variable_names, dtype=float)
        else:
            self.p_A = pd.DataFrame()
        self.p_xdot = None
        if self.xdot_full is not None:
            self.p_xdot = pd.DataFrame(self.xdot_full, self.free_variable_names, ['data'], dtype=float)
            self.p_b['xdot'] = self.p_xdot
            self.p_b = self.p_b[['lb', 'xdot', 'ub']]
            self.p_pure_xdot = deepcopy(self.p_xdot)
            self.p_pure_xdot[-num_constr:] = 0
            # self.p_Ax = pd.DataFrame(self.p_A.dot(self.p_xdot), self.inequality_constr_names, ['data'], dtype=float)
            if len(self.p_A) > 0:
                self.p_Ax = pd.DataFrame(self.p_A.dot(self.p_pure_xdot), self.inequality_constr_names,
                                         ['data'], dtype=float)
            else:
                self.p_Ax = pd.DataFrame()
            # self.p_Ax_without_slack = deepcopy(self.p_Ax_without_slack_raw)
            # self.p_Ax_without_slack[-num_constr:] /= self.sample_period
            if len(self.p_E) > 0:
                self.p_Ex = pd.DataFrame(self.p_E.dot(self.p_pure_xdot), self.equality_constr_names,
                                         ['data'], dtype=float)
            else:
                self.p_Ex = pd.DataFrame()

        else:
            self.p_xdot = None
        self.p_debug = god_map.debug_expression_manager.to_pandas()

    def _print_iis(self):
        result = self.qp_solver.analyze_infeasibility()
        if result is None:
            get_middleware().loginfo(f'Can only compute possible causes with gurobi, '
                                     f'but current solver is {self.qp_solver_class.solver_id.name}.')
            return
        lb_ids, ub_ids, eq_ids, lbA_ids, ubA_ids = result
        b_ids = lb_ids | ub_ids
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
            get_middleware().loginfo('Irreducible Infeasible Subsystem:')
            get_middleware().loginfo('  Free variable bounds')
            free_variables = self.p_lb[b_ids]
            free_variables['ub'] = self.p_ub[b_ids]
            free_variables = free_variables.rename(columns={'data': 'lb'})
            print(free_variables)
            get_middleware().loginfo('  Equality constraints:')
            self._print_iis_matrix(eq_ids, b_ids, self.p_E, self.p_bE)
            get_middleware().loginfo('  Inequality constraint lower bounds:')
            self._print_iis_matrix(lbA_ids, b_ids, self.p_A, self.p_lbA)
            get_middleware().loginfo('  Inequality constraint upper bounds:')
            self._print_iis_matrix(ubA_ids, b_ids, self.p_A, self.p_ubA)

    def _print_iis_matrix(self, row_filter: np.ndarray, column_filter: np.ndarray, matrix: pd.DataFrame,
                          bounds: pd.DataFrame):
        if len(row_filter) == 0:
            return
        filtered_matrix = matrix.loc[row_filter, column_filter]
        filtered_matrix['bounds'] = bounds.loc[row_filter]
        print(filtered_matrix)
