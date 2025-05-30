from __future__ import annotations

import abc
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Tuple, List, Union, Dict, TYPE_CHECKING, DefaultDict, Optional

import giskardpy.casadi_wrapper as cas
import giskardpy.utils.math as giskard_math
import numpy as np
from giskardpy.data_types.data_types import Derivatives
from giskardpy.data_types.exceptions import InfeasibleException, VelocityLimitUnreachableException
from giskardpy.middleware import get_middleware
from giskardpy.qp.constraint import EqualityConstraint, InequalityConstraint, DerivativeInequalityConstraint, \
    DerivativeEqualityConstraint
from giskardpy.qp.free_variable import FreeVariable
from giskardpy.qp.pos_in_vel_limits import b_profile
from giskardpy.qp.qp_formulation import QPFormulation
from giskardpy.qp.weight_gain import QuadraticWeightGain, LinearWeightGain
from giskardpy.symbol_manager import SymbolManager
from giskardpy.utils.decorators import memoize
from line_profiler import profile

if TYPE_CHECKING:
    import scipy.sparse as sp


class ProblemDataPart(ABC):
    """
    min_x 0.5*x^T*diag(w)*x + g^T*x
    s.t.  lb <= x <= ub
               Ex = b
        lbA <= Ax <= ubA
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
                 horizon_weight_gain_scalar: float,
                 qp_formulation: QPFormulation):
        self.horizon_weight_gain_scalar = horizon_weight_gain_scalar
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
                 horizon_weight_gain_scalar: float,
                 qp_formulation: QPFormulation):
        super().__init__(free_variables=free_variables,
                         equality_constraints=equality_constraints,
                         inequality_constraints=inequality_constraints,
                         derivative_constraints=derivative_constraints,
                         eq_derivative_constraints=eq_derivative_constraints,
                         sample_period=sample_period,
                         prediction_horizon=prediction_horizon,
                         max_derivative=max_derivative,
                         horizon_weight_gain_scalar=horizon_weight_gain_scalar,
                         qp_formulation=qp_formulation)
        self.evaluated = True

    def linear_f(self, current_position, limit, target_value, a=10, exp=2) -> Tuple[cas.Expression, float]:
        f = cas.abs(current_position * a) ** exp
        x_offset = cas.solve_for(f, target_value)
        return (cas.abs(current_position + x_offset - limit) * a) ** exp, x_offset

    @profile
    def construct_expression(self, quadratic_weight_gains: List[QuadraticWeightGain] = None,
                             linear_weight_gains: List[LinearWeightGain] = None) \
            -> Union[cas.Expression, Tuple[cas.Expression, cas.Expression]]:
        quadratic_weight_gains = quadratic_weight_gains or []
        linear_weight_gains = linear_weight_gains or []
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
                                                            alpha=self.horizon_weight_gain_scalar,
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
                 horizon_weight_gain_scalar: float,
                 qp_formulation: QPFormulation):
        super().__init__(free_variables=free_variables,
                         equality_constraints=equality_constraints,
                         inequality_constraints=inequality_constraints,
                         derivative_constraints=derivative_constraints,
                         eq_derivative_constraints=eq_derivative_constraints,
                         sample_period=sample_period,
                         prediction_horizon=prediction_horizon,
                         max_derivative=max_derivative,
                         horizon_weight_gain_scalar=horizon_weight_gain_scalar,
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
                 horizon_weight_gain_scalar: float,
                 qp_formulation: QPFormulation):
        super().__init__(free_variables=free_variables,
                         equality_constraints=equality_constraints,
                         inequality_constraints=inequality_constraints,
                         derivative_constraints=derivative_constraints,
                         eq_derivative_constraints=eq_derivative_constraints,
                         sample_period=sample_period,
                         prediction_horizon=prediction_horizon,
                         max_derivative=max_derivative,
                         horizon_weight_gain_scalar=horizon_weight_gain_scalar,
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
                 horizon_weight_gain_scalar: float,
                 qp_formulation: QPFormulation,
                 default_limits: bool = False):
        super().__init__(free_variables=free_variables,
                         equality_constraints=equality_constraints,
                         inequality_constraints=inequality_constraints,
                         derivative_constraints=derivative_constraints,
                         eq_derivative_constraints=eq_derivative_constraints,
                         sample_period=sample_period,
                         prediction_horizon=prediction_horizon,
                         max_derivative=max_derivative,
                         horizon_weight_gain_scalar=horizon_weight_gain_scalar,
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


@dataclass
class QPData:
    quadratic_weights: np.ndarray = field(default=None)
    linear_weights: np.ndarray = field(default=None)

    box_lower_constraints: np.ndarray = field(default=None)
    box_upper_constraints: np.ndarray = field(default=None)

    eq_matrix: Union[sp.csc_matrix, np.ndarray] = field(default=None)
    eq_bounds: np.ndarray = field(default=None)

    neq_matrix: Union[sp.csc_matrix, np.ndarray] = field(default=None)
    neq_lower_bounds: np.ndarray = field(default=None)
    neq_upper_bounds: np.ndarray = field(default=None)

    @property
    def sparse_hessian(self) -> sp.csc_matrix:
        import scipy.sparse as sp
        return sp.diags(self.quadratic_weights)

    @property
    def dense_hessian(self) -> np.ndarray:
        return np.diag(self.quadratic_weights)

    def pretty_print_problem(self):
        print('QP data')
        if self.quadratic_weights is not None:
            print(f'H (quadratic_weights): \n{np.array2string(self.quadratic_weights, max_line_width=np.inf)}')
        if self.linear_weights is not None:
            print(f'g (linear_weights): \n{np.array2string(self.linear_weights, max_line_width=np.inf)})')

        if self.box_lower_constraints is not None:
            print(f'lb (box_lower_constraints): \n{np.array2string(self.box_lower_constraints, max_line_width=np.inf)}')
        if self.box_upper_constraints is not None:
            print(f'ub (box_upper_constraints): \n{np.array2string(self.box_upper_constraints, max_line_width=np.inf)}')

        if self.eq_matrix is not None:
            try:
                print(f'E (eq_matrix): \n{np.array2string(self.eq_matrix.toarray(), max_line_width=np.inf)}')
            except:
                print(f'E (eq_matrix): \n{np.array2string(self.eq_matrix, max_line_width=np.inf)}')
        if self.eq_bounds is not None:
            print(f'bE (eq_bounds): \n{np.array2string(self.eq_bounds, max_line_width=np.inf)}')

        if self.neq_matrix is not None:
            try:
                print(f'A (neq_matrix): \n{np.array2string(self.neq_matrix.toarray(), max_line_width=np.inf)}')
            except:
                print(f'A (neq_matrix): \n{np.array2string(self.neq_matrix, max_line_width=np.inf)}')
        if self.neq_lower_bounds is not None:
            print(f'lbA (neq_lower_bounds): \n{np.array2string(self.neq_lower_bounds, max_line_width=np.inf)}')
        if self.neq_upper_bounds is not None:
            print(f'ubA (neq_upper_bounds): \n{np.array2string(self.neq_upper_bounds, max_line_width=np.inf)}')


@dataclass
class GiskardToQPAdapter(abc.ABC):
    free_variables: List[FreeVariable]
    equality_constraints: List[EqualityConstraint]
    inequality_constraints: List[InequalityConstraint]
    derivative_constraints: List[DerivativeInequalityConstraint]
    eq_derivative_constraints: List[DerivativeEqualityConstraint]
    mpc_dt: float
    prediction_horizon: int
    max_derivative: Derivatives
    horizon_weight_gain_scalar: float
    qp_formulation: QPFormulation
    sparse: bool

    compute_nI_I: bool = True
    _nAi_Ai_cache: dict = field(default_factory=dict)

    """
    Takes free variables and constraints and converts them to a QP problem in the following format, depending on the
    class attributes:
    min_x 0.5 x^T H x + g^T x
    s.t.  lb <= x <= ub     (box constraints)
          Edof x <= bE_dof          (equality constraints)
          Eslack x <= bE_slack        (equality constraints)
          lbA <= Adof x <= ubA_dof  (lower/upper inequality constraints)
          lbA <= Aslack x <= ubA_slack  (lower/upper inequality constraints)
    """

    def __post_init__(self):
        kwargs = {'free_variables': self.free_variables,
                  'equality_constraints': self.equality_constraints,
                  'inequality_constraints': self.inequality_constraints,
                  'derivative_constraints': self.derivative_constraints,
                  'eq_derivative_constraints': self.eq_derivative_constraints,
                  'sample_period': self.mpc_dt,
                  'prediction_horizon': self.prediction_horizon,
                  'max_derivative': self.max_derivative,
                  'horizon_weight_gain_scalar': self.horizon_weight_gain_scalar,
                  'qp_formulation': self.qp_formulation}
        weights = Weights(**kwargs)
        free_variable_bounds = FreeVariableBounds(**kwargs)
        equality_model = EqualityModel(**kwargs)
        equality_bounds = EqualityBounds(**kwargs)
        inequality_model = InequalityModel(**kwargs)
        inequality_bounds = InequalityBounds(**kwargs)

        quadratic_weights, linear_weights = weights.construct_expression()
        box_lower_constraints, box_upper_constraints = free_variable_bounds.construct_expression()
        eq_matrix_dofs, self.eq_matrix_slack = equality_model.construct_expression()
        eq_bounds = equality_bounds.construct_expression()
        neq_matrix_dofs, self.neq_matrix_slack = inequality_model.construct_expression()
        neq_lower_bounds, neq_upper_bounds = inequality_bounds.construct_expression()
        self.general_qp_to_specific_qp(quadratic_weights=quadratic_weights,
                                       linear_weights=linear_weights,
                                       box_lower_constraints=box_lower_constraints,
                                       box_upper_constraints=box_upper_constraints,
                                       eq_matrix_dofs=eq_matrix_dofs,
                                       eq_matrix_slack=self.eq_matrix_slack,
                                       eq_bounds=eq_bounds,
                                       neq_matrix_dofs=neq_matrix_dofs,
                                       neq_matrix_slack=self.neq_matrix_slack,
                                       neq_lower_bounds=neq_lower_bounds,
                                       neq_upper_bounds=neq_upper_bounds)

    def __hash__(self):
        return hash(id(self))

    @property
    def num_eq_constraints(self) -> int:
        return len(self.equality_constraints)

    @property
    def num_neq_constraints(self) -> int:
        return len(self.inequality_constraints)

    @property
    def num_free_variable_constraints(self) -> int:
        return len(self.free_variables)

    @property
    def num_eq_slack_variables(self) -> int:
        return self.eq_matrix_slack.shape[1]

    @property
    def num_neq_slack_variables(self) -> int:
        return self.neq_matrix_slack.shape[1]

    @property
    def num_slack_variables(self) -> int:
        return self.num_eq_slack_variables + self.num_neq_slack_variables

    @property
    def num_non_slack_variables(self) -> int:
        return self.num_free_variable_constraints - self.num_slack_variables

    @abc.abstractmethod
    def general_qp_to_specific_qp(self,
                                  quadratic_weights: cas.Expression,
                                  linear_weights: cas.Expression,
                                  box_lower_constraints: cas.Expression,
                                  box_upper_constraints: cas.Expression,
                                  eq_matrix_dofs: cas.Expression,
                                  eq_matrix_slack: cas.Expression,
                                  eq_bounds: cas.Expression,
                                  neq_matrix_dofs: cas.Expression,
                                  neq_matrix_slack: cas.Expression,
                                  neq_lower_bounds: cas.Expression,
                                  neq_upper_bounds: cas.Expression):
        ...

    @abc.abstractmethod
    def evaluate(self, symbol_manager: SymbolManager):
        ...

    @profile
    def _direct_limit_model(self, dimensions_after_zero_filter: int,
                            Ai_inf_filter: Optional[np.ndarray] = None, two_sided: bool = False) \
            -> Union[np.ndarray, sp.csc_matrix]:
        """
        These models are often identical, yet the computation is expensive. Caching to the rescue
        """
        if Ai_inf_filter is None:
            key = hash(dimensions_after_zero_filter)
        else:
            key = hash((dimensions_after_zero_filter, Ai_inf_filter.tobytes()))
        if key not in self._nAi_Ai_cache:
            nI_I = self._cached_eyes(dimensions_after_zero_filter, two_sided)
            if Ai_inf_filter is None:
                self._nAi_Ai_cache[key] = nI_I
            else:
                self._nAi_Ai_cache[key] = nI_I[Ai_inf_filter]
        return self._nAi_Ai_cache[key]

    @memoize
    def _cached_eyes(self, dimensions: int, two_sided: bool = False) -> Union[np.ndarray, sp.csc_matrix]:
        if self.sparse:
            from scipy import sparse as sp
            if two_sided:
                data = np.ones(dimensions, dtype=float)
                row_indices = np.arange(dimensions)
                col_indices = np.arange(dimensions + 1)
                return sp.csc_matrix((data, row_indices, col_indices))
            else:
                d2 = dimensions * 2
                data = np.ones(d2, dtype=float)
                data[::2] *= -1
                r1 = np.arange(dimensions)
                r2 = np.arange(dimensions, d2)
                row_indices = np.empty((d2,), dtype=int)
                row_indices[0::2] = r1
                row_indices[1::2] = r2
                col_indices = np.arange(0, d2 + 1, 2)
                return sp.csc_matrix((data, row_indices, col_indices))
        else:
            I = np.eye(dimensions)
            if two_sided:
                return I
            else:
                return np.concatenate([-I, I])


# @dataclass
class GiskardToExplicitQPAdapter(GiskardToQPAdapter):
    """
    Takes free variables and constraints and converts them to a QP problem in the following format, depending on the
    class attributes:

    min_x 0.5 x^T H x + g^T x
    s.t.  lb <= x <= ub     (box constraints)
          Ex <= bE          (equality constraints)
          lbA <= Ax <= ubA  (lower/upper inequality constraints)
    """

    quadratic_weights: cas.Expression
    linear_weights: cas.Expression

    box_lower_constraints: cas.Expression
    box_upper_constraints: cas.Expression

    eq_matrix_dofs: cas.Expression
    eq_matrix_slack: cas.Expression
    eq_bounds: cas.Expression

    neq_matrix_dofs: cas.Expression
    neq_matrix_slack: cas.Expression
    neq_lower_bounds: cas.Expression
    neq_upper_bounds: cas.Expression

    bE_filter: np.ndarray
    bA_filter: np.ndarray

    def general_qp_to_specific_qp(self,
                                  quadratic_weights: cas.Expression,
                                  linear_weights: cas.Expression,
                                  box_lower_constraints: cas.Expression,
                                  box_upper_constraints: cas.Expression,
                                  eq_matrix_dofs: cas.Expression,
                                  eq_matrix_slack: cas.Expression,
                                  eq_bounds: cas.Expression,
                                  neq_matrix_dofs: cas.Expression,
                                  neq_matrix_slack: cas.Expression,
                                  neq_lower_bounds: cas.Expression,
                                  neq_upper_bounds: cas.Expression):
        eq_matrix = cas.hstack([eq_matrix_dofs,
                                eq_matrix_slack,
                                cas.zeros(eq_matrix_slack.shape[0], self.num_neq_slack_variables)])
        neq_matrix = cas.hstack([neq_matrix_dofs,
                                 cas.zeros(neq_matrix_slack.shape[0], self.num_eq_slack_variables),
                                 neq_matrix_slack])

        free_symbols = set(quadratic_weights.free_symbols())
        free_symbols.update(linear_weights.free_symbols())
        free_symbols.update(box_lower_constraints.free_symbols())
        free_symbols.update(box_upper_constraints.free_symbols())
        free_symbols.update(eq_matrix.free_symbols())
        free_symbols.update(eq_bounds.free_symbols())
        free_symbols.update(neq_matrix.free_symbols())
        free_symbols.update(neq_lower_bounds.free_symbols())
        free_symbols.update(neq_upper_bounds.free_symbols())
        self.free_symbols = list(free_symbols)

        self.eq_matrix_compiled = eq_matrix.compile(parameters=self.free_symbols, sparse=self.sparse)
        self.neq_matrix_compiled = neq_matrix.compile(parameters=self.free_symbols, sparse=self.sparse)

        self.combined_vector_f = cas.StackedCompiledFunction(expressions=[quadratic_weights,
                                                                          linear_weights,
                                                                          box_lower_constraints,
                                                                          box_upper_constraints,
                                                                          eq_bounds,
                                                                          neq_lower_bounds,
                                                                          neq_upper_bounds],
                                                             parameters=self.free_symbols)

        self.free_symbols_str = [str(x) for x in self.free_symbols]

        self.bE_filter = np.ones(eq_matrix.shape[0], dtype=bool)
        self.bA_filter = np.ones(neq_matrix.shape[0], dtype=bool)

    def create_filters(self,
                       quadratic_weights_np_raw: np.ndarray,
                       num_slack_variables: int,
                       num_eq_slack_variables: int,
                       num_neq_slack_variables: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        zero_quadratic_weight_filter: np.ndarray = quadratic_weights_np_raw != 0
        # don't filter dofs with 0 weight
        zero_quadratic_weight_filter[:-num_slack_variables] = True
        slack_part = zero_quadratic_weight_filter[-(num_eq_slack_variables + num_neq_slack_variables):]
        bE_part = slack_part[:num_eq_slack_variables]
        bA_part = slack_part[num_eq_slack_variables:]

        self.bE_filter.fill(True)
        if len(bE_part) > 0:
            self.bE_filter[-len(bE_part):] = bE_part

        self.bA_filter.fill(True)
        if len(bA_part) > 0:
            self.bA_filter[-len(bA_part):] = bA_part
        return zero_quadratic_weight_filter, self.bE_filter, self.bA_filter

    @profile
    def apply_filters(self,
                      qp_data_raw: QPData,
                      zero_quadratic_weight_filter: np.ndarray,
                      bE_filter: np.ndarray, bA_filter: np.ndarray) -> QPData:
        qp_data_filtered = QPData()
        qp_data_filtered.quadratic_weights = qp_data_raw.quadratic_weights[zero_quadratic_weight_filter]
        qp_data_filtered.linear_weights = qp_data_raw.linear_weights[zero_quadratic_weight_filter]
        qp_data_filtered.box_lower_constraints = qp_data_raw.box_lower_constraints[zero_quadratic_weight_filter]
        qp_data_filtered.box_upper_constraints = qp_data_raw.box_upper_constraints[zero_quadratic_weight_filter]
        # if self.num_filtered_eq_constraints > 0:
        qp_data_filtered.eq_matrix = qp_data_raw.eq_matrix[bE_filter, :][:, zero_quadratic_weight_filter]
        # else:
        # when no eq constraints were filtered, we can just cut off at the end, because that section is always all 0
        # qp_data_filtered.eq_matrix = self.eq_matrix_np_raw[:, :self.zero_quadratic_weight_filter.sum()]
        qp_data_filtered.eq_bounds = qp_data_raw.eq_bounds[bE_filter]
        if (len(qp_data_raw.neq_matrix.shape) > 1
                and qp_data_raw.neq_matrix.shape[0] * qp_data_raw.neq_matrix.shape[1] > 0):
            qp_data_filtered.neq_matrix = qp_data_raw.neq_matrix[:, zero_quadratic_weight_filter][bA_filter, :]
        else:
            qp_data_filtered.neq_matrix = qp_data_raw.neq_matrix
        qp_data_filtered.neq_lower_bounds = qp_data_raw.neq_lower_bounds[bA_filter]
        qp_data_filtered.neq_upper_bounds = qp_data_raw.neq_upper_bounds[bA_filter]
        return qp_data_filtered

    def evaluate(self, symbol_manager: SymbolManager):
        substitutions = symbol_manager.resolve_symbols(self.free_symbols_str)

        eq_matrix_np_raw = self.eq_matrix_compiled.fast_call(substitutions)
        neq_matrix_np_raw = self.neq_matrix_compiled.fast_call(substitutions)
        quadratic_weights_np_raw, \
            linear_weights_np_raw, \
            box_lower_constraints_np_raw, \
            box_upper_constraints_np_raw, \
            eq_bounds_np_raw, \
            neq_lower_bounds_np_raw, \
            neq_upper_bounds_np_raw = self.combined_vector_f.fast_call(substitutions)
        qp_data_raw = QPData(quadratic_weights=quadratic_weights_np_raw,
                             linear_weights=linear_weights_np_raw,
                             box_lower_constraints=box_lower_constraints_np_raw,
                             box_upper_constraints=box_upper_constraints_np_raw,
                             eq_matrix=eq_matrix_np_raw,
                             eq_bounds=eq_bounds_np_raw,
                             neq_matrix=neq_matrix_np_raw,
                             neq_lower_bounds=neq_lower_bounds_np_raw,
                             neq_upper_bounds=neq_upper_bounds_np_raw)

        zero_quadratic_weight_filter, bE_filter, bA_filter = self.create_filters(
            quadratic_weights_np_raw=quadratic_weights_np_raw,
            num_slack_variables=self.num_slack_variables,
            num_eq_slack_variables=self.num_eq_slack_variables,
            num_neq_slack_variables=self.num_neq_slack_variables)

        qp_data = self.apply_filters(qp_data_raw=qp_data_raw,
                                     zero_quadratic_weight_filter=zero_quadratic_weight_filter,
                                     bE_filter=bE_filter,
                                     bA_filter=bA_filter)

        return qp_data


class GiskardToTwoSidedNeqQPAdapter(GiskardToQPAdapter):
    """
    Takes free variables and constraints and converts them to a QP problem in the following format, depending on the
    class attributes:

    min_x 0.5 x^T H x + g^T x
    s.t.  lbA <= Ax <= ubA
    """

    quadratic_weights: cas.Expression
    linear_weights: cas.Expression

    box_lower_constraints: cas.Expression
    box_upper_constraints: cas.Expression

    eq_matrix_dofs: cas.Expression
    eq_matrix_slack: cas.Expression
    eq_bounds: cas.Expression

    neq_matrix_dofs: cas.Expression
    neq_matrix_slack: cas.Expression
    neq_lower_bounds: cas.Expression
    neq_upper_bounds: cas.Expression

    b_bE_bA_filter: np.ndarray
    b_zero_inf_filter_view: np.ndarray
    bE_filter_view: np.ndarray
    bA_filter_view: np.ndarray
    bE_bA_filter: np.ndarray

    def general_qp_to_specific_qp(self,
                                  quadratic_weights: cas.Expression,
                                  linear_weights: cas.Expression,
                                  box_lower_constraints: cas.Expression,
                                  box_upper_constraints: cas.Expression,
                                  eq_matrix_dofs: cas.Expression,
                                  eq_matrix_slack: cas.Expression,
                                  eq_bounds: cas.Expression,
                                  neq_matrix_dofs: cas.Expression,
                                  neq_matrix_slack: cas.Expression,
                                  neq_lower_bounds: cas.Expression,
                                  neq_upper_bounds: cas.Expression):
        if len(neq_matrix_dofs) == 0:
            constraint_matrix = cas.hstack([eq_matrix_dofs, eq_matrix_slack])
        else:
            eq_matrix = cas.hstack([eq_matrix_dofs,
                                    eq_matrix_slack,
                                    cas.zeros(eq_matrix_dofs.shape[0], neq_matrix_slack.shape[1])])
            neq_matrix = cas.hstack([neq_matrix_dofs,
                                     cas.zeros(neq_matrix_dofs.shape[0], eq_matrix_slack.shape[1]),
                                     neq_matrix_slack])
            constraint_matrix = cas.vstack([eq_matrix, neq_matrix])

        free_symbols = set(quadratic_weights.free_symbols())
        free_symbols.update(constraint_matrix.free_symbols())
        free_symbols.update(box_lower_constraints.free_symbols())
        free_symbols.update(box_upper_constraints.free_symbols())
        free_symbols.update(eq_bounds.free_symbols())
        free_symbols.update(neq_lower_bounds.free_symbols())
        free_symbols.update(neq_upper_bounds.free_symbols())
        free_symbols = list(free_symbols)
        self.free_symbols = free_symbols

        len_lb_be_lba_end = (quadratic_weights.shape[0]
                             + box_lower_constraints.shape[0]
                             + eq_bounds.shape[0]
                             + neq_lower_bounds.shape[0])
        len_ub_be_uba_end = (len_lb_be_lba_end
                             + box_upper_constraints.shape[0]
                             + eq_bounds.shape[0]
                             + neq_upper_bounds.shape[0])

        self.combined_vector_f = cas.StackedCompiledFunction(expressions=[quadratic_weights,
                                                                          box_lower_constraints,
                                                                          eq_bounds,
                                                                          neq_lower_bounds,
                                                                          box_upper_constraints,
                                                                          eq_bounds,
                                                                          neq_upper_bounds,
                                                                          linear_weights],
                                                             parameters=free_symbols,
                                                             additional_views=[
                                                                 slice(quadratic_weights.shape[0], len_lb_be_lba_end),
                                                                 slice(len_lb_be_lba_end, len_ub_be_uba_end)])

        self.neq_matrix_compiled = constraint_matrix.compile(parameters=free_symbols, sparse=self.sparse)

        self.free_symbols_str = [str(x) for x in free_symbols]

        self.b_bE_bA_filter = np.ones(box_lower_constraints.shape[0] + eq_bounds.shape[0] + neq_lower_bounds.shape[0],
                                      dtype=bool)
        self.b_zero_inf_filter_view = self.b_bE_bA_filter[:box_lower_constraints.shape[0]]
        self.bE_filter_view = self.b_bE_bA_filter[
                              box_lower_constraints.shape[0]:box_lower_constraints.shape[0] + eq_bounds.shape[0]]
        self.bA_filter_view = self.b_bE_bA_filter[box_lower_constraints.shape[0] + eq_bounds.shape[0]:]
        self.bE_bA_filter = self.b_bE_bA_filter[box_lower_constraints.shape[0]:]

        if self.compute_nI_I:
            self._nAi_Ai_cache = {}

    def create_filters(self,
                       quadratic_weights_np_raw: np.ndarray,
                       box_lower_constraints_np_raw: np.ndarray,
                       box_upper_constraints_np_raw: np.ndarray,
                       num_slack_variables: int,
                       num_eq_slack_variables: int,
                       num_neq_slack_variables: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.b_bE_bA_filter.fill(True)

        zero_quadratic_weight_filter: np.ndarray = quadratic_weights_np_raw != 0
        zero_quadratic_weight_filter[:-num_slack_variables] = True

        slack_part = zero_quadratic_weight_filter[-(num_eq_slack_variables + num_neq_slack_variables):]
        bE_part = slack_part[:num_eq_slack_variables]
        if len(bE_part) > 0:
            self.bE_filter_view[-len(bE_part):] = bE_part

        bA_part = slack_part[num_eq_slack_variables:]
        if len(bA_part) > 0:
            self.bA_filter_view[-len(bA_part):] = bA_part

        b_finite_filter = np.isfinite(box_lower_constraints_np_raw) | np.isfinite(box_upper_constraints_np_raw)
        self.b_zero_inf_filter_view[::] = zero_quadratic_weight_filter & b_finite_filter
        Ai_inf_filter = b_finite_filter[zero_quadratic_weight_filter]
        return zero_quadratic_weight_filter, Ai_inf_filter, self.bE_bA_filter, self.b_bE_bA_filter




    @profile
    def apply_filters(self,
                      qp_data_raw: QPData,
                      zero_quadratic_weight_filter: np.ndarray,
                      Ai_inf_filter: np.ndarray,
                      bE_bA_filter: np.ndarray,
                      b_bE_bA_filter: np.ndarray) -> QPData:
        from scipy import sparse as sp
        qp_data_filtered = QPData()
        qp_data_filtered.quadratic_weights = qp_data_raw.quadratic_weights[zero_quadratic_weight_filter]
        qp_data_filtered.linear_weights = qp_data_raw.linear_weights[zero_quadratic_weight_filter]
        qp_data_filtered.neq_lower_bounds = qp_data_raw.neq_lower_bounds[b_bE_bA_filter]
        qp_data_filtered.neq_upper_bounds = qp_data_raw.neq_upper_bounds[b_bE_bA_filter]
        qp_data_filtered.neq_matrix = qp_data_raw.neq_matrix[:, zero_quadratic_weight_filter][bE_bA_filter, :]

        box_matrix = self._direct_limit_model(qp_data_filtered.quadratic_weights.shape[0],
                                              Ai_inf_filter,
                                              two_sided=True)
        qp_data_filtered.neq_matrix = sp.vstack((box_matrix, qp_data_filtered.neq_matrix))

        return qp_data_filtered

    def evaluate(self, symbol_manager: SymbolManager):
        substitutions = symbol_manager.resolve_symbols(self.free_symbols_str)

        neq_matrix = self.neq_matrix_compiled.fast_call(substitutions)
        quadratic_weights_np_raw, \
            box_lower_constraints_np_raw, \
            _, \
            _, \
            box_upper_constraints_np_raw, \
            _, \
            _, \
            linear_weights_np_raw, \
            box_eq_neq_lower_bounds_np_raw, \
            box_eq_neq_upper_bounds_np_raw = self.combined_vector_f.fast_call(substitutions)
        qp_data_raw = QPData(quadratic_weights=quadratic_weights_np_raw,
                             linear_weights=linear_weights_np_raw,
                             neq_matrix=neq_matrix,
                             neq_lower_bounds=box_eq_neq_lower_bounds_np_raw,
                             neq_upper_bounds=box_eq_neq_upper_bounds_np_raw)

        zero_quadratic_weight_filter, Ai_inf_filter, bE_bA_filter, b_bE_bA_filter = self.create_filters(
            quadratic_weights_np_raw=quadratic_weights_np_raw,
            box_lower_constraints_np_raw=box_lower_constraints_np_raw,
            box_upper_constraints_np_raw=box_upper_constraints_np_raw,
            num_slack_variables=self.num_slack_variables,
            num_eq_slack_variables=self.num_eq_slack_variables,
            num_neq_slack_variables=self.num_neq_slack_variables)

        qp_data = self.apply_filters(qp_data_raw=qp_data_raw,
                                     zero_quadratic_weight_filter=zero_quadratic_weight_filter,
                                     Ai_inf_filter=Ai_inf_filter,
                                     bE_bA_filter=bE_bA_filter,
                                     b_bE_bA_filter=b_bE_bA_filter)

        return qp_data
