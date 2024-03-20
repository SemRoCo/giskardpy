import abc
import datetime
import os
from abc import ABC
from collections import defaultdict
from copy import deepcopy
from typing import List, Dict, Tuple, Type, Union, Optional, DefaultDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import giskardpy.casadi_wrapper as cas
from giskardpy.configs.qp_controller_config import SupportedQPSolver
from giskardpy.exceptions import HardConstraintsViolatedException, QPSolverException, InfeasibleException, \
    VelocityLimitUnreachableException
from giskardpy.god_map import god_map
from giskardpy.data_types import Derivatives
from giskardpy.qp.constraint import InequalityConstraint, EqualityConstraint, DerivativeInequalityConstraint, \
    ManipulabilityConstraint
from giskardpy.qp.free_variable import FreeVariable
from giskardpy.qp.next_command import NextCommands
from giskardpy.qp.pos_in_vel_limits import b_profile, implicit_vel_profile
from giskardpy.qp.qp_solver import QPSolver
from giskardpy.symbol_manager import symbol_manager
from giskardpy.utils import logging
from giskardpy.utils.utils import create_path, get_all_classes_in_package
from giskardpy.utils.decorators import memoize
import giskardpy.utils.math as giskard_math

# used for saving pandas in the same folder every time within a run
date_str = datetime.datetime.now().strftime('%Yy-%mm-%dd--%Hh-%Mm-%Ss')


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
                    zero_filtered_column = column.replace(0, pd.np.nan).dropna(how='all').replace(pd.np.nan, 0)
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
    sample_period: float
    prediction_horizon: int
    max_derivative: Derivatives

    def __init__(self,
                 free_variables: List[FreeVariable],
                 equality_constraints: List[EqualityConstraint],
                 inequality_constraints: List[InequalityConstraint],
                 derivative_constraints: List[DerivativeInequalityConstraint],
                 sample_period: float,
                 prediction_horizon: int,
                 max_derivative: Derivatives):
        self.free_variables = free_variables
        self.equality_constraints = equality_constraints
        self.inequality_constraints = inequality_constraints
        self.derivative_constraints = derivative_constraints
        self.prediction_horizon = prediction_horizon
        self.dt = sample_period
        self.max_derivative = max_derivative

    @property
    def number_of_free_variables(self) -> int:
        return len(self.free_variables)

    def replace_hack(self, expression: Union[float, cas.Expression], new_value):
        if not isinstance(expression, cas.Expression):
            return expression
        hack = symbol_manager.hack
        expression.s = cas.ca.substitute(expression.s, hack.s, new_value)
        return expression

    def get_derivative_constraints(self, derivative: Derivatives) -> List[DerivativeInequalityConstraint]:
        return [c for c in self.derivative_constraints if c.derivative == derivative]

    @abc.abstractmethod
    def construct_expression(self) -> Union[cas.Expression, Tuple[cas.Expression, cas.Expression]]:
        pass

    @property
    def velocity_constraints(self) -> List[DerivativeInequalityConstraint]:
        return self.get_derivative_constraints(Derivatives.velocity)

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

    def _remove_columns_columns_where_variables_are_zero(self, free_variable_model: cas.Expression) -> cas.Expression:
        if np.prod(free_variable_model.shape) == 0:
            return free_variable_model
        column_ids = []
        end = 0
        for derivative in Derivatives.range(Derivatives.velocity, self.max_derivative - 1):
            last_non_zero_variable = self.prediction_horizon - (self.max_derivative - derivative)
            start = end + self.number_of_free_variables * last_non_zero_variable
            end += self.number_of_free_variables * self.prediction_horizon
            column_ids.extend(range(start, end))
        free_variable_model.remove([], column_ids)
        return free_variable_model


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
                 sample_period: float,
                 prediction_horizon: int, max_derivative: Derivatives):
        super().__init__(free_variables=free_variables,
                         equality_constraints=equality_constraints,
                         inequality_constraints=inequality_constraints,
                         derivative_constraints=derivative_constraints,
                         sample_period=sample_period,
                         prediction_horizon=prediction_horizon,
                         max_derivative=max_derivative)
        self.evaluated = True

    def linear_f(self, current_position, limit, target_value, a=10, exp=2) -> Tuple[cas.Expression, float]:
        f = cas.abs(current_position * a) ** exp
        x_offset = cas.solve_for(f, target_value)
        return (cas.abs(current_position + x_offset - limit) * a) ** exp, x_offset

    @profile
    def construct_expression(self) -> Union[cas.Expression, Tuple[cas.Expression, cas.Expression]]:
        components = []
        components.extend(self.free_variable_weights_expression())
        components.append(self.equality_weight_expressions())
        components.extend(self.derivative_weight_expressions())
        components.append(self.inequality_weight_expressions())
        weights, _ = self._sorter(*components)
        weights = cas.Expression(weights)
        linear_weights = cas.zeros(*weights.shape)
        return cas.Expression(weights), linear_weights

    @profile
    def free_variable_weights_expression(self) -> List[defaultdict]:
        params = []
        weights = defaultdict(dict)  # maps order to joints
        for t in range(self.prediction_horizon):
            for v in self.free_variables:
                for derivative in Derivatives.range(Derivatives.velocity, self.max_derivative):
                    if t >= self.prediction_horizon - (self.max_derivative - derivative):
                        continue
                    normalized_weight = v.normalized_weight(t, derivative, self.prediction_horizon,
                                                            evaluated=self.evaluated)
                    weights[derivative][f't{t:03}/{v.position_name}/{derivative}'] = normalized_weight
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
                    if t < c.control_horizon:
                        derivative_constr_weights[f't{t:03}/{c.name}'] = c.normalized_weight(t)
            params.append(derivative_constr_weights)
        return params

    def equality_weight_expressions(self) -> dict:
        error_slack_weights = {f'{c.name}/error': c.normalized_weight() for c in self.equality_constraints}
        return error_slack_weights

    def inequality_weight_expressions(self) -> dict:
        error_slack_weights = {f'{c.name}/error': c.normalized_weight() for c in self.inequality_constraints}
        return error_slack_weights

    @profile
    def linear_weights_expression(self, manip_expressions: List[ManipulabilityConstraint]):
        if len(manip_expressions) == 0:
            return None, None

        stacked_exp = cas.vstack([x.expression for x in manip_expressions])

        J = cas.jacobian(stacked_exp, self.get_free_variable_symbols(Derivatives.position))

        grad_traces = []
        for symbol in self.get_free_variable_symbols(Derivatives.position):
            JJt = J.dot(J.T)
            J_dq = cas.total_derivative(J, [symbol], [1])
            product = cas.matrix_inverse(JJt).dot(J_dq).dot(J.T)
            trace = cas.trace(product)
            grad_traces.append(trace)

        return J, grad_traces

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
                 sample_period: float,
                 prediction_horizon: int,
                 max_derivative: Derivatives):
        super().__init__(free_variables=free_variables,
                         equality_constraints=equality_constraints,
                         inequality_constraints=inequality_constraints,
                         derivative_constraints=derivative_constraints,
                         sample_period=sample_period,
                         prediction_horizon=prediction_horizon,
                         max_derivative=max_derivative)
        self.evaluated = True

    @profile
    def velocity_limit(self, v: FreeVariable):
        current_position = v.get_symbol(Derivatives.position)
        lower_velocity_limit = v.get_lower_limit(Derivatives.velocity, evaluated=True)
        upper_velocity_limit = v.get_upper_limit(Derivatives.velocity, evaluated=True)
        lower_acc_limit = v.get_lower_limit(Derivatives.acceleration, evaluated=True)
        upper_acc_limit = v.get_upper_limit(Derivatives.acceleration, evaluated=True)
        current_vel = v.get_symbol(Derivatives.velocity)
        current_acc = v.get_symbol(Derivatives.acceleration)

        lower_jerk_limit = v.get_lower_limit(Derivatives.jerk, evaluated=True)
        upper_jerk_limit = v.get_upper_limit(Derivatives.jerk, evaluated=True)

        max_reachable_vel = giskard_math.max_velocity_from_horizon_and_jerk(self.prediction_horizon,
                                                                            upper_jerk_limit, self.dt)
        if max_reachable_vel < upper_velocity_limit:
            error_msg = f'Free variable "{v.name}" can\'t reach velocity limit of "{upper_velocity_limit}". ' \
                        f'Maximum reachable with prediction horizon = "{self.prediction_horizon}", ' \
                        f'jerk limit = "{upper_jerk_limit}" and dt = "{self.dt}" is "{max_reachable_vel}".'
            logging.logerr(error_msg)
            raise VelocityLimitUnreachableException(error_msg)

        if not v.has_position_limits():
            lb = cas.Expression([lower_velocity_limit] * self.prediction_horizon
                                + [lower_acc_limit] * self.prediction_horizon
                                + [lower_jerk_limit] * self.prediction_horizon)
            ub = cas.Expression([upper_velocity_limit] * self.prediction_horizon
                                + [upper_acc_limit] * self.prediction_horizon
                                + [upper_jerk_limit] * self.prediction_horizon)
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
                max_reachable_vel = giskard_math.max_velocity_from_horizon_and_jerk(self.prediction_horizon,
                                                                                    upper_jerk_limit, self.dt)
                if max_reachable_vel < upper_velocity_limit:
                    error_msg = f'Free variable "{v.name}" can\'t reach velocity limit of "{upper_velocity_limit}". ' \
                                f'Maximum reachable with prediction horizon = "{self.prediction_horizon}", ' \
                                f'jerk limit = "{upper_jerk_limit}" and dt = "{self.dt}" is "{max_reachable_vel}".'
                    logging.logerr(error_msg)
                    raise VelocityLimitUnreachableException(error_msg)
                else:
                    raise
        # %% set velocity limits to infinite, that can't be reached due to acc/jerk limits anyway
        unlimited_vel_profile = implicit_vel_profile(acc_limit=upper_acc_limit,
                                                     jerk_limit=upper_jerk_limit,
                                                     dt=self.dt,
                                                     ph=self.prediction_horizon)
        for i in range(self.prediction_horizon):
            ub[i] = cas.if_less(ub[i], unlimited_vel_profile[i], ub[i], np.inf)
        unlimited_vel_profile = implicit_vel_profile(acc_limit=-lower_acc_limit,
                                                     jerk_limit=-lower_jerk_limit,
                                                     dt=self.dt,
                                                     ph=self.prediction_horizon)
        for i in range(self.prediction_horizon):
            lb[i] = cas.if_less(-lb[i], unlimited_vel_profile[i], lb[i], -np.inf)
        return lb, ub

    @profile
    def free_variable_bounds(self) \
            -> Tuple[List[Dict[str, cas.symbol_expr_float]], List[Dict[str, cas.symbol_expr_float]]]:
        lb: DefaultDict[Derivatives, Dict[str, cas.symbol_expr_float]] = defaultdict(dict)
        ub: DefaultDict[Derivatives, Dict[str, cas.symbol_expr_float]] = defaultdict(dict)
        for v in self.free_variables:
            lb_, ub_ = self.velocity_limit(v)
            for t in range(self.prediction_horizon):
                for derivative in Derivatives.range(Derivatives.velocity, self.max_derivative):
                    if t >= self.prediction_horizon - (self.max_derivative - derivative):
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
                if t < c.control_horizon:
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
        derivative_slack_stop = derivative_slack_start + num_derivative_slack
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
                 sample_period: float,
                 prediction_horizon: int,
                 max_derivative: Derivatives):
        super().__init__(free_variables=free_variables,
                         equality_constraints=equality_constraints,
                         inequality_constraints=inequality_constraints,
                         derivative_constraints=derivative_constraints,
                         sample_period=sample_period,
                         prediction_horizon=prediction_horizon,
                         max_derivative=max_derivative)
        self.evaluated = True

    def equality_constraint_bounds(self) -> Dict[str, cas.Expression]:
        return {f'{c.name}': c.capped_bound(self.dt) for c in self.equality_constraints}

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

    @profile
    def construct_expression(self) -> Union[cas.Expression, Tuple[cas.Expression, cas.Expression]]:
        bounds = []
        for derivative in Derivatives.range(Derivatives.velocity, self.max_derivative - 1):
            bounds.append(self.last_derivative_values(derivative))
            bounds.append(self.derivative_links(derivative))
        num_derivative_links = sum(len(x) for x in bounds)

        bounds.append(self.equality_constraint_bounds())

        bounds, self.names = self._sorter(*bounds)
        self.names_derivative_links = self.names[:num_derivative_links]
        self.names_equality_constraints = self.names[num_derivative_links:]
        return cas.Expression(bounds)


class GoalReached(ProblemDataPart):

    def construct_expression(self) -> Union[cas.Expression, Tuple[cas.Expression, cas.Expression]]:
        self.names = []
        checks = []
        for c in self.equality_constraints:
            self.names.append(c.name)
            checks.append(c.goal_reached())
        return cas.Expression(checks)

    def to_panda(self, substitutions: np.ndarray) -> pd.DataFrame:
        data = self.goal_reached_f.fast_call(substitutions)
        return pd.DataFrame({'data': data}, index=self.names)

    def compile(self, free_symbols: List[cas.ca.SX]):
        self.goal_reached_f = self.construct_expression().compile(free_symbols)


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
                 sample_period: float,
                 prediction_horizon: int,
                 max_derivative: Derivatives,
                 default_limits: bool):
        super().__init__(free_variables=free_variables,
                         equality_constraints=equality_constraints,
                         inequality_constraints=inequality_constraints,
                         derivative_constraints=derivative_constraints,
                         sample_period=sample_period,
                         prediction_horizon=prediction_horizon,
                         max_derivative=max_derivative)
        self.default_limits = default_limits
        self.evaluated = True

    def derivative_constraint_bounds(self, derivative: Derivatives) \
            -> Tuple[Dict[str, cas.Expression], Dict[str, cas.Expression]]:
        lower = {}
        upper = {}
        for t in range(self.prediction_horizon):
            for c in self.get_derivative_constraints(derivative):
                if t < c.control_horizon:
                    lower[f't{t:03}/{c.name}'] = cas.limit(c.lower_limit[t] * self.dt,
                                                           -c.normalization_factor * self.dt,
                                                           c.normalization_factor * self.dt)
                    upper[f't{t:03}/{c.name}'] = cas.limit(c.upper_limit[t] * self.dt,
                                                           -c.normalization_factor * self.dt,
                                                           c.normalization_factor * self.dt)
        return lower, upper

    def lower_inequality_constraint_bound(self):
        bounds = {}
        for constraint in self.inequality_constraints:
            limit = constraint.velocity_limit * self.dt * constraint.control_horizon
            if isinstance(constraint.lower_error, float) and np.isinf(constraint.lower_error):
                bounds[f'{constraint.name}'] = constraint.lower_error
            else:
                bounds[f'{constraint.name}'] = cas.limit(constraint.lower_error, -limit, limit)
        return bounds

    def upper_inequality_constraint_bound(self):
        bounds = {}
        for constraint in self.inequality_constraints:
            limit = constraint.velocity_limit * self.dt * constraint.control_horizon
            if isinstance(constraint.upper_error, float) and np.isinf(constraint.upper_error):
                bounds[f'{constraint.name}'] = constraint.upper_error
            else:
                bounds[f'{constraint.name}'] = cas.limit(constraint.upper_error, -limit, limit)
        return bounds

    @profile
    def construct_expression(self) -> Union[cas.Expression, Tuple[cas.Expression, cas.Expression]]:
        lb_params = []
        ub_params = []

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

    @property
    def number_of_non_slack_columns(self):
        return self.number_of_free_variables * self.prediction_horizon * self.max_derivative

    @profile
    def derivative_link_model(self) -> cas.Expression:
        """
        Layout for prediction horizon 5
        Slots are matrices of |controlled variables| x |controlled variables|
        | vt0 | vt1 | vt2 | at0 | at1 | at2 | at3 | jt0 | jt1 | jt2 | jt3 | jt4 |
        |-----------------------------------------------------------------------|
        |  1  |     |     | -dt |     |     |     |     |     |     |     |     | last_v =  vt0 - at0*dt
        | -1  |  1  |     |     | -dt |     |     |     |     |     |     |     |      0 = -vt0 + vt1 - at1 * dt
        |     | -1  |  1  |     |     | -dt |     |     |     |     |     |     |      0 = -vt1 + vt2 - at2 * dt
        |     |     | -1  |     |     |     | -dt |     |     |     |     |     |      0 = -vt2 + vt3 - at3 * dt
        |=======================================================================|
        |     |     |     |  1  |     |     |     | -dt |     |     |     |     | last_a =  at0 - jt0*dt
        |     |     |     | -1  |  1  |     |     |     | -dt |     |     |     |      0 = -at0 + at1 - jt1 * dt
        |     |     |     |     | -1  |  1  |     |     |     | -dt |     |     |      0 = -at1 + at2 - jt2 * dt
        |     |     |     |     |     | -1  |  1  |     |     |     | -dt |     |      0 = -at2 + at3 - jt3 * dt
        |     |     |     |     |     |     | -1  |     |     |     |     | -dt |      0 = -at3 + at4 - jt4 * dt
        |-----------------------------------------------------------------------|
        """
        num_rows = self.number_of_free_variables * self.prediction_horizon * (self.max_derivative - 1)
        num_columns = self.number_of_free_variables * self.prediction_horizon * self.max_derivative
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
        for derivative in Derivatives.range(Derivatives.velocity, self.max_derivative - 1):
            offset_v += self.number_of_free_variables
            derivative_link_model[offset_v:offset_v + x_c_height, offset_h:offset_h + x_c_height] += x_c
            offset_v += x_c_height
            offset_h += self.prediction_horizon * self.number_of_free_variables
        derivative_link_model = self._remove_rows_columns_where_variables_are_zero(derivative_link_model)
        return derivative_link_model

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
        """
        if len(self.equality_constraints) > 0:
            model = cas.zeros(len(self.equality_constraints), self.number_of_non_slack_columns)
            for derivative in Derivatives.range(Derivatives.position, self.max_derivative - 1):
                J_eq = cas.jacobian(expressions=cas.Expression(self.equality_constraint_expressions()),
                                    symbols=self.get_free_variable_symbols(derivative)) * self.dt
                J_hstack = cas.hstack([J_eq for _ in range(self.prediction_horizon)])
                # set jacobian entry to 0 if control horizon shorter than prediction horizon
                for i, c in enumerate(self.equality_constraints):
                    # offset = vertical_offset + i
                    J_hstack[i, c.control_horizon * len(self.free_variables):] = 0
                horizontal_offset = J_hstack.shape[1]
                model[:, horizontal_offset * derivative:horizontal_offset * (derivative + 1)] = J_hstack

            # slack variable for total error
            slack_model = cas.diag(cas.Expression([self.dt * c.control_horizon for c in self.equality_constraints]))
            return model, slack_model
        return cas.Expression(), cas.Expression()

    @profile
    def construct_expression(self) -> Union[cas.Expression, Tuple[cas.Expression, cas.Expression]]:
        derivative_link_model = self.derivative_link_model()
        equality_constraint_model, equality_constraint_slack_model = self.equality_constraint_model()

        model_parts = []
        slack_model_parts = []
        if len(derivative_link_model) > 0:
            model_parts.append(derivative_link_model)
        if len(equality_constraint_model) > 0:
            model_parts.append(equality_constraint_model)
            slack_model_parts.append(equality_constraint_slack_model)
        model = cas.vstack(model_parts)
        slack_model = cas.vstack(slack_model_parts)

        slack_model = cas.vstack([cas.zeros(derivative_link_model.shape[0],
                                            slack_model.shape[1]),
                                  slack_model])
        model = self._remove_columns_columns_where_variables_are_zero(model)
        return model, slack_model


class InequalityModel(ProblemDataPart):
    """
    Format:
        velocity constraints
        acceleration constraints
        jerk constraints
        inequality constraints
    """

    @property
    def number_of_free_variables(self):
        return len(self.free_variables)

    @property
    def number_of_non_slack_columns(self):
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
        number_of_vel_rows = len(self.velocity_constraints) * self.prediction_horizon
        if number_of_vel_rows > 0:
            expressions = cas.Expression(self.get_derivative_constraint_expressions(Derivatives.velocity))
            model = cas.zeros(number_of_vel_rows, self.number_of_non_slack_columns)
            for derivative in Derivatives.range(Derivatives.position, self.max_derivative - 1):
                J_vel = cas.jacobian(expressions=expressions,
                                     symbols=self.get_free_variable_symbols(derivative)) * self.dt
                J_vel_limit_block = cas.kron(cas.eye(self.prediction_horizon), J_vel)
                horizontal_offset = self.number_of_free_variables * self.prediction_horizon
                model[:, horizontal_offset * derivative:horizontal_offset * (derivative + 1)] = J_vel_limit_block

            # delete rows if control horizon of constraint shorter than prediction horizon
            rows_to_delete = []
            for t in range(self.prediction_horizon):
                for i, c in enumerate(self.velocity_constraints):
                    v_index = i + (t * len(self.velocity_constraints))
                    if t + 1 > c.control_horizon:
                        rows_to_delete.append(v_index)
            model.remove(rows_to_delete, [])

            # constraint slack
            num_slack_variables = sum(c.control_horizon for c in self.velocity_constraints)
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
                    if t + 1 > c.control_horizon:
                        rows_to_delete.append(v_index)
            model.remove(rows_to_delete, [])

            # slack model
            num_slack_variables = sum(c.control_horizon for c in self.acceleration_constraints)
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
                    if t + 1 > c.control_horizon:
                        rows_to_delete.append(v_index)
            model.remove(rows_to_delete, [])

            # slack model
            num_slack_variables = sum(c.control_horizon for c in self.jerk_constraints)
            slack_model = cas.eye(num_slack_variables) * self.dt
            return model, slack_model
        return cas.Expression(), cas.Expression()

    @profile
    def inequality_constraint_model(self) -> Tuple[cas.Expression, cas.Expression]:
        """
        |   t1   |   t2   |   t1   |   t2   |   t1   |   t2   |   t1   |   t2   | prediction horizon
        |v1 v2 v3|v1 v2 v3|a1 a2 a3|a1 a2 a3|j1 j2 j3|j1 j2 j3|s1 s2 s3|s1 s2 s3| free variables / slack
        |-----------------------------------------------------------------------|
        |  J1*sp |  J1*sp |  J2*sp |  J2*sp |  J3*sp | J3*sp  | sp*ch  | sp*ch  |
        |-----------------------------------------------------------------------|
        """
        if len(self.inequality_constraints) > 0:
            model = cas.zeros(len(self.inequality_constraints), self.number_of_non_slack_columns)
            for derivative in Derivatives.range(Derivatives.position, self.max_derivative - 1):
                J_neq = cas.jacobian(expressions=cas.Expression(self.inequality_constraint_expressions()),
                                     symbols=self.get_free_variable_symbols(derivative)) * self.dt
                J_hstack = cas.hstack([J_neq for _ in range(self.prediction_horizon)])
                # set jacobian entry to 0 if control horizon shorter than prediction horizon
                for i, c in enumerate(self.inequality_constraints):
                    J_hstack[i, c.control_horizon * len(self.free_variables):] = 0
                horizontal_offset = J_hstack.shape[1]
                model[:, horizontal_offset * derivative:horizontal_offset * (derivative + 1)] = J_hstack

            # slack variable for total error
            slack_model = cas.diag(cas.Expression([self.dt * c.control_horizon for c in self.inequality_constraints]))
            return model, slack_model
        return cas.Expression(), cas.Expression()

    @profile
    def construct_expression(self) -> Union[cas.Expression, Tuple[cas.Expression, cas.Expression]]:
        vel_constr_model, vel_constr_slack_model = self.velocity_constraint_model()
        acc_constr_model, acc_constr_slack_model = self.acceleration_constraint_model()
        jerk_constr_model, jerk_constr_slack_model = self.jerk_constraint_model()
        inequality_model, inequality_slack_model = self.inequality_constraint_model()
        model_parts = []
        slack_model_parts = []
        if len(vel_constr_model) > 0:
            model_parts.append(vel_constr_model)
            slack_model_parts.append(vel_constr_slack_model)
        if len(acc_constr_model) > 0:
            model_parts.append(acc_constr_model)
            slack_model_parts.append(acc_constr_slack_model)
        if len(jerk_constr_model) > 0:
            model_parts.append(jerk_constr_model)
            slack_model_parts.append(jerk_constr_slack_model)
        if len(inequality_model) > 0:
            model_parts.append(inequality_model)
            slack_model_parts.append(inequality_slack_model)

        combined_model = cas.vstack(model_parts)
        combined_slack_model = cas.diag_stack(slack_model_parts)
        combined_model = self._remove_columns_columns_where_variables_are_zero(combined_model)
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
    solver_names = [str(solver_name).split('.')[1] for solver_name in available_solvers.keys()]
    logging.loginfo(f'Found these qp solvers: {solver_names}')


detect_solvers()


class QPProblemBuilder:
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
    manipulability_indexes: np.ndarray = np.array([0.0, 0.0])

    def __init__(self,
                 sample_period: float,
                 prediction_horizon: int,
                 solver_id: Optional[SupportedQPSolver] = None,
                 free_variables: List[FreeVariable] = None,
                 equality_constraints: List[EqualityConstraint] = None,
                 inequality_constraints: List[InequalityConstraint] = None,
                 derivative_constraints: List[DerivativeInequalityConstraint] = None,
                 manipulability_constraints: List[ManipulabilityConstraint] = None,
                 retries_with_relaxed_constraints: int = 0,
                 retry_added_slack: float = 100,
                 retry_weight_factor: float = 100):
        self.free_variables = []
        self.equality_constraints = []
        self.inequality_constraints = []
        self.derivative_constraints = []
        self.manipulability_constraints = []
        self.prediction_horizon = prediction_horizon
        self.retries_with_relaxed_constraints = retries_with_relaxed_constraints
        self.retry_added_slack = retry_added_slack
        self.retry_weight_factor = retry_weight_factor
        self.xdot_full = None
        if free_variables is not None:
            self.add_free_variables(free_variables)
        if inequality_constraints is not None:
            self.add_inequality_constraints(inequality_constraints)
        if equality_constraints is not None:
            self.add_equality_constraints(equality_constraints)
        if derivative_constraints is not None:
            self.add_derivative_constraints(derivative_constraints)
        if manipulability_constraints is not None:
            self.add_manipulability_constraints(manipulability_constraints)

        if solver_id is not None:
            self.qp_solver_class = available_solvers[solver_id]
        else:
            for solver_id in SupportedQPSolver:
                if solver_id in available_solvers:
                    self.qp_solver_class = available_solvers[solver_id]
                    break
            else:
                raise QPSolverException(f'No qp solver found')

        logging.loginfo(f'Using QP Solver \'{solver_id.name}\'')
        logging.loginfo(f'Prediction horizon: \'{self.prediction_horizon}\'')
        self.qp_solver = self.compile(self.qp_solver_class)

    def add_free_variables(self, free_variables: list):
        if len(free_variables) == 0:
            raise QPSolverException('Cannot solve qp with no free variables')
        self.free_variables.extend(list(sorted(free_variables, key=lambda x: x.position_name)))
        l = [x.position_name for x in free_variables]
        duplicates = set([x for x in l if l.count(x) > 1])
        self.order = Derivatives(min(self.prediction_horizon, god_map.qp_controller_config.max_derivative))
        assert duplicates == set(), f'there are free variables with the same name: {duplicates}'

    def add_inequality_constraints(self, constraints: List[InequalityConstraint]):
        self.inequality_constraints.extend(list(sorted(constraints, key=lambda x: x.name)))
        l = [x.name for x in constraints]
        duplicates = set([x for x in l if l.count(x) > 1])
        assert duplicates == set(), f'there are multiple constraints with the same name: {duplicates}'
        for c in self.inequality_constraints:
            c.control_horizon = min(c.control_horizon, self.prediction_horizon)
            self.check_control_horizon(c)

    def add_equality_constraints(self, constraints: List[EqualityConstraint]):
        self.equality_constraints.extend(list(sorted(constraints, key=lambda x: x.name)))
        l = [x.name for x in constraints]
        duplicates = set([x for x in l if l.count(x) > 1])
        assert duplicates == set(), f'there are multiple constraints with the same name: {duplicates}'
        for c in self.equality_constraints:
            c.control_horizon = min(c.control_horizon, self.prediction_horizon)
            self.check_control_horizon(c)

    def add_manipulability_constraints(self, constraints: List[ManipulabilityConstraint]):
        self.manipulability_constraints.extend(list(sorted(constraints, key=lambda x: x.name)))
        l = [x.name for x in constraints]
        duplicates = set([x for x in l if l.count(x) > 1])
        assert duplicates == set(), f'there are multiple manipulability constraints with the same name: {duplicates}'

    def add_derivative_constraints(self, constraints: List[DerivativeInequalityConstraint]):
        self.derivative_constraints.extend(list(sorted(constraints, key=lambda x: x.name)))
        l = [x.name for x in constraints]
        duplicates = set([x for x in l if l.count(x) > 1])
        assert duplicates == set(), f'there are multiple constraints with the same name: {duplicates}'
        for c in self.derivative_constraints:
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

    @profile
    def compile(self, solver_class: Type[QPSolver], default_limits: bool = False) -> QPSolver:
        logging.loginfo('Creating controller')
        kwargs = {'free_variables': self.free_variables,
                  'equality_constraints': self.equality_constraints,
                  'inequality_constraints': self.inequality_constraints,
                  'derivative_constraints': self.derivative_constraints,
                  'sample_period': god_map.qp_controller_config.sample_period,
                  'prediction_horizon': self.prediction_horizon,
                  'max_derivative': self.order}
        self.weights = Weights(**kwargs)
        self.free_variable_bounds = FreeVariableBounds(**kwargs)
        self.equality_model = EqualityModel(**kwargs)
        self.equality_bounds = EqualityBounds(**kwargs)
        # self.goal_reached_checks = GoalReached(**kwargs)
        self.inequality_model = InequalityModel(**kwargs)
        self.inequality_bounds = InequalityBounds(default_limits=default_limits, **kwargs)

        weights, g = self.weights.construct_expression()
        lb, ub = self.free_variable_bounds.construct_expression()
        A, A_slack = self.inequality_model.construct_expression()
        lbA, ubA = self.inequality_bounds.construct_expression()
        E, E_slack = self.equality_model.construct_expression()
        bE = self.equality_bounds.construct_expression()
        J, grad_traces = self.weights.linear_weights_expression(
            manip_expressions=self.manipulability_constraints)

        qp_solver = solver_class(weights=weights, g=g, lb=lb, ub=ub,
                                 E=E, E_slack=E_slack, bE=bE,
                                 A=A, A_slack=A_slack, lbA=lbA, ubA=ubA,
                                 constraint_jacobian=J, grad_traces=grad_traces
                                 )
        # self.goal_reached_checks.compile(qp_solver.free_symbols)
        logging.loginfo('Done compiling controller:')
        logging.loginfo(f'  #free variables: {weights.shape[0]}')
        logging.loginfo(f'  #equality constraints: {bE.shape[0]}')
        logging.loginfo(f'  #inequality constraints: {lbA.shape[0]}')
        return qp_solver

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
            god_map.giskard.tmp_folder,
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
            return NextCommands(self.free_variables, self.xdot_full, self.order, self.prediction_horizon)
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
        logging.loginfo('No slack limit violation detected.')

    def _viz_mpc(self, joint_name):
        def pad(a, desired_length, pad_value):
            tmp = np.ones(desired_length) * pad_value
            tmp[:len(a)] = a
            return tmp

        free_variable: FreeVariable = [x for x in self.free_variables if x.name == joint_name][0]
        sample_period = god_map.qp_controller_config.sample_period
        try:
            start_pos = god_map.world.state[joint_name].position
        except KeyError:
            logging.loginfo('start position not found in state')
            start_pos = 0
        ts = np.array([(i + 1) * sample_period for i in range(self.prediction_horizon)])
        filtered_x = self.p_xdot.filter(like=f'/{joint_name}/', axis=0)
        vel_end = self.prediction_horizon - self.order + 1
        acc_end = vel_end + self.prediction_horizon - self.order + 2
        velocities = filtered_x[:vel_end].values
        positions = [start_pos]
        for x_ in velocities:
            positions.append(positions[-1] + x_ * sample_period)

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
        tick_labels = [f'{x}/{x * sample_period:.3f}' for x in range(self.prediction_horizon)]

        axs[-1].set_xticks(ts)  # Set tick locations
        axs[-1].set_xticklabels(tick_labels)  # Set custom tick labels

        plt.tight_layout()
        path, dirs, files = next(os.walk('tmp_data/mpc'))
        file_count = len(files)
        file_name = f'{god_map.giskard.tmp_folder}/mpc/mpc_{joint_name}_{file_count}.png'
        create_path(file_name)
        plt.savefig(file_name)

    @profile
    def _create_debug_pandas(self, qp_solver: QPSolver):
        weights, g, lb, ub, E, bE, A, lbA, ubA, weight_filter, bE_filter, bA_filter = qp_solver.get_problem_data()
        sample_period = god_map.qp_controller_config.sample_period
        self.free_variable_names = self.free_variable_bounds.names[weight_filter]
        self.equality_constr_names = self.equality_bounds.names[bE_filter]
        self.inequality_constr_names = self.inequality_bounds.names[bA_filter]
        num_vel_constr = len(self.derivative_constraints) * (self.prediction_horizon - 2)
        num_neq_constr = len(self.inequality_constraints)
        num_eq_constr = len(self.equality_constraints)
        num_constr = num_vel_constr + num_neq_constr + num_eq_constr

        self.p_weights = pd.DataFrame(weights, self.free_variable_names, ['data'], dtype=float)
        self.p_g = pd.DataFrame(g, self.free_variable_names, ['data'], dtype=float)
        self.p_lb = pd.DataFrame(lb, self.free_variable_names, ['data'], dtype=float)
        self.p_ub = pd.DataFrame(ub, self.free_variable_names, ['data'], dtype=float)
        self.p_b = pd.DataFrame({'lb': lb, 'ub': ub}, self.free_variable_names, dtype=float)
        if len(bE) > 0:
            self.p_bE_raw = pd.DataFrame(bE, self.equality_constr_names, ['data'], dtype=float)
            self.p_bE = deepcopy(self.p_bE_raw)
            self.p_bE[len(self.equality_bounds.names_derivative_links):] /= sample_period
        else:
            self.p_bE = pd.DataFrame()
        if len(lbA) > 0:
            self.p_lbA_raw = pd.DataFrame(lbA, self.inequality_constr_names, ['data'], dtype=float)
            self.p_lbA = deepcopy(self.p_lbA_raw)
            self.p_lbA /= sample_period

            self.p_ubA_raw = pd.DataFrame(ubA, self.inequality_constr_names, ['data'], dtype=float)
            self.p_ubA = deepcopy(self.p_ubA_raw)
            self.p_ubA /= sample_period

            self.p_bA_raw = pd.DataFrame({'lbA': lbA, 'ubA': ubA}, self.inequality_constr_names, dtype=float)
            self.p_bA = deepcopy(self.p_bA_raw)
            self.p_bA /= sample_period
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
            # self.p_Ax_without_slack[-num_constr:] /= sample_period
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
            logging.loginfo(f'Can only compute possible causes with gurobi, '
                            f'but current solver is {self.qp_solver_class.solver_id.name}.')
            return
        lb_ids, ub_ids, eq_ids, lbA_ids, ubA_ids = result
        b_ids = lb_ids | ub_ids
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
            logging.loginfo('Irreducible Infeasible Subsystem:')
            logging.loginfo('  Free variable bounds')
            free_variables = self.p_lb[b_ids]
            free_variables['ub'] = self.p_ub[b_ids]
            free_variables = free_variables.rename(columns={'data': 'lb'})
            print(free_variables)
            logging.loginfo('  Equality constraints:')
            self._print_iis_matrix(eq_ids, b_ids, self.p_E, self.p_bE)
            logging.loginfo('  Inequality constraint lower bounds:')
            self._print_iis_matrix(lbA_ids, b_ids, self.p_A, self.p_lbA)
            logging.loginfo('  Inequality constraint upper bounds:')
            self._print_iis_matrix(ubA_ids, b_ids, self.p_A, self.p_ubA)

    def _print_iis_matrix(self, row_filter: np.ndarray, column_filter: np.ndarray, matrix: pd.DataFrame,
                          bounds: pd.DataFrame):
        if len(row_filter) == 0:
            return
        filtered_matrix = matrix.loc[row_filter, column_filter]
        filtered_matrix['bounds'] = bounds.loc[row_filter]
        print(filtered_matrix)
