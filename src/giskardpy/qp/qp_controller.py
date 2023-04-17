import abc
import datetime
import os
from abc import ABC
from collections import OrderedDict, defaultdict
from copy import deepcopy
from time import time
from typing import List, Dict, Tuple, Type, Union, Optional, DefaultDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import giskardpy.casadi_wrapper as cas
from giskardpy import identifier
from giskardpy.configs.data_types import SupportedQPSolver
from giskardpy.exceptions import OutOfJointLimitsException, \
    HardConstraintsViolatedException, QPSolverException, InfeasibleException
from giskardpy.god_map import GodMap
from giskardpy.model.world import WorldTree
from giskardpy.my_types import derivative_joint_map, Derivatives
from giskardpy.qp.constraint import InequalityConstraint, EqualityConstraint, DerivativeInequalityConstraint
from giskardpy.qp.free_variable import FreeVariable
from giskardpy.qp.next_command import NextCommands
from giskardpy.qp.qp_solver import QPSolver
from giskardpy.utils import logging
from giskardpy.utils.utils import create_path, suppress_stdout, get_all_classes_in_package
from giskardpy.utils.decorators import memoize


def save_pandas(dfs, names, path):
    folder_name = f'{path}/pandas_{datetime.datetime.now().strftime("%Yy-%mm-%dd--%Hh-%Mm-%Ss")}/'
    create_path(folder_name)
    for df, name in zip(dfs, names):
        csv_string = 'name\n'
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            if df.shape[1] > 1:
                for column_name, column in df.T.items():
                    zero_filtered_column = column.replace(0, pd.np.nan).dropna(how='all').replace(pd.np.nan, 0)
                    csv_string += zero_filtered_column.add_prefix(column_name + '||').to_csv(float_format='%.4f')
            else:
                csv_string += df.to_csv(float_format='%.4f')
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

    def replace_hack(self, expression: Union[float, cas.Expression], new_value):
        if not isinstance(expression, cas.Expression):
            return expression
        hack = GodMap().to_symbol(identifier.hack)
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
        # self.height = 0
        # self._compute_height()
        self.evaluated = True

    @profile
    def construct_expression(self) -> Union[cas.Expression, Tuple[cas.Expression, cas.Expression]]:
        components = []
        components.extend(self.free_variable_weights_expression())
        components.append(self.equality_weight_expressions())
        components.extend(self.derivative_weight_expressions())
        components.append(self.inequality_weight_expressions())
        weights, _ = self._sorter(*components)
        weights = cas.Expression(weights)
        # todo replace hack?
        # for i in range(len(weights)):
        #     weights[i] = self.replace_hack(weights[i], 0)
        return cas.Expression(weights), cas.zeros(*weights.shape)

    @profile
    def free_variable_weights_expression(self) -> List[defaultdict]:
        params = []
        weights = defaultdict(dict)  # maps order to joints
        for t in range(self.prediction_horizon):
            for v in self.free_variables:
                for o in Derivatives.range(Derivatives.velocity, min(v.order, self.max_derivative)):
                    o = Derivatives(o)
                    weights[o][f't{t:03}/{v.position_name}/{o}'] = v.normalized_weight(t, o,
                                                                                       self.prediction_horizon,
                                                                                       evaluated=self.evaluated)
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
    def free_variable_bounds(self) -> Tuple[List[Dict[str, cas.symbol_expr_float]],
                                            List[Dict[str, cas.symbol_expr_float]]]:
        lb: DefaultDict[Derivatives, Dict[str, cas.symbol_expr_float]] = defaultdict(dict)
        ub: DefaultDict[Derivatives, Dict[str, cas.symbol_expr_float]] = defaultdict(dict)
        for t in range(self.prediction_horizon):
            for v in self.free_variables:
                for derivative in Derivatives.range(Derivatives.velocity, min(v.order, self.max_derivative)):
                    if t == self.prediction_horizon - 1 \
                            and derivative < min(v.order, self.max_derivative) \
                            and self.prediction_horizon > 2:  # and False:
                        lb[derivative][f't{t:03}/{v.name}/{derivative}'] = 0
                        ub[derivative][f't{t:03}/{v.name}/{derivative}'] = 0
                    else:
                        lb[derivative][f't{t:03}/{v.name}/{derivative}'] = v.get_lower_limit(derivative,
                                                                                             evaluated=self.evaluated)
                        ub[derivative][f't{t:03}/{v.name}/{derivative}'] = v.get_upper_limit(derivative,
                                                                                             evaluated=self.evaluated)
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
        # todo replace hack?
        # for i in range(len(lb)):
        #     lb[i] = self.replace_hack(lb[i], 0)
        #     ub[i] = self.replace_hack(ub[i], 0)
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

    def equality_bounds(self) -> Dict[str, cas.Expression]:
        return {f'{c.name}': cas.limit(c.bound,
                                       -c.velocity_limit * self.dt * c.control_horizon,
                                       c.velocity_limit * self.dt * c.control_horizon)
                for c in self.equality_constraints}

    def last_derivative_values(self, derivative: Derivatives) -> Dict[str, cas.symbol_expr_float]:
        last_values = {}
        for v in self.free_variables:
            last_values[f'{v.name}/last_{derivative}'] = v.get_symbol(derivative)
        return last_values

    def derivative_links(self, derivative: Derivatives) -> Dict[str, cas.symbol_expr_float]:
        derivative_link = {}
        for t in range(self.prediction_horizon - 1):
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

        bounds.append(self.equality_bounds())

        bounds, self.names = self._sorter(*bounds)
        self.names_derivative_links = self.names[:num_derivative_links]
        self.names_equality_constraints = self.names[num_derivative_links:]
        # TODO replace hack?
        # for i in range(len(lbA)):
        #     lbA[i] = self.replace_hack(lbA[i], 0)
        #     ubA[i] = self.replace_hack(ubA[i], 0)
        return cas.Expression(bounds)


class InequalityBounds(ProblemDataPart):
    """
    Format:
        position limits
        derivative velocity bounds
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

    def position_limits(self) -> Tuple[Dict[str, cas.Expression], Dict[str, cas.Expression]]:
        lb = {}
        ub = {}
        for t in range(self.prediction_horizon):
            for v in self.free_variables:
                if v.has_position_limits():
                    normal_lower_bound = v.get_lower_limit(Derivatives.position, False,
                                                           evaluated=self.evaluated) - v.get_symbol(
                        Derivatives.position)
                    normal_upper_bound = v.get_upper_limit(Derivatives.position, False,
                                                           evaluated=self.evaluated) - v.get_symbol(
                        Derivatives.position)
                    if self.default_limits:
                        if self.max_derivative >= Derivatives.jerk:
                            lower_vel = cas.min(v.get_upper_limit(derivative=Derivatives.velocity,
                                                                  default=False,
                                                                  evaluated=True) * self.dt,
                                                v.get_upper_limit(derivative=Derivatives.jerk,
                                                                  default=False,
                                                                  evaluated=self.evaluated) * self.dt ** 3)
                            upper_vel = cas.max(v.get_lower_limit(derivative=Derivatives.velocity,
                                                                  default=False,
                                                                  evaluated=True) * self.dt,
                                                v.get_lower_limit(derivative=Derivatives.jerk,
                                                                  default=False,
                                                                  evaluated=self.evaluated) * self.dt ** 3)
                        else:
                            lower_vel = cas.min(v.get_upper_limit(derivative=Derivatives.velocity,
                                                                  default=False,
                                                                  evaluated=True) * self.dt,
                                                v.get_upper_limit(derivative=Derivatives.acceleration,
                                                                  default=False,
                                                                  evaluated=self.evaluated) * self.dt ** 2)
                            upper_vel = cas.max(v.get_lower_limit(derivative=Derivatives.velocity,
                                                                  default=False,
                                                                  evaluated=True) * self.dt,
                                                v.get_lower_limit(derivative=Derivatives.acceleration,
                                                                  default=False,
                                                                  evaluated=self.evaluated) * self.dt ** 2)
                        lower_bound = cas.if_greater(normal_lower_bound, 0,
                                                     if_result=lower_vel,
                                                     else_result=normal_lower_bound)
                        lb[f't{t:03d}/{v.name}/p_limit'] = lower_bound

                        upper_bound = cas.if_less(normal_upper_bound, 0,
                                                  if_result=upper_vel,
                                                  else_result=normal_upper_bound)
                        ub[f't{t:03d}/{v.name}/p_limit'] = upper_bound
                    else:
                        lb[f't{t:03d}/{v.name}/p_limit'] = normal_lower_bound
                        ub[f't{t:03d}/{v.name}/p_limit'] = normal_upper_bound
        return lb, ub

    @profile
    def construct_expression(self) -> Union[cas.Expression, Tuple[cas.Expression, cas.Expression]]:
        lower_position_bounds, upper_position_bounds = self.position_limits()
        lb_params = [lower_position_bounds]
        ub_params = [upper_position_bounds]
        num_position_limits = len(lower_position_bounds)

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

        self.names_position_limits = self.names[:num_position_limits]
        self.names_non_position_limits = self.names[num_position_limits:]
        self.names_derivative_links = self.names[num_position_limits:num_position_limits + num_derivative_constraints]
        self.names_neq_constraints = self.names[num_position_limits + num_derivative_constraints + num_neq_constraints:]

        # TODO replace hack?
        # for i in range(len(lbA)):
        #     lbA[i] = self.replace_hack(lbA[i], 0)
        #     ubA[i] = self.replace_hack(ubA[i], 0)
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

    @property
    def number_of_free_variables(self):
        return len(self.free_variables)

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
        |   t1   |   t2   |   t3   |   t1   |   t2   |   t3   |   t1   |   t2   |   t3   | prediction horizon
        |v1 v2 v3|v1 v2 v3|v1 v2 v3|a1 a2 a3|a1 a2 a3|a1 a2 a3|j1 j2 j3|j1 j2 j3|j1 j2 j3| free variables / slack
        |--------------------------------------------------------------------------------|
        | 1      |        |        |-sp     |        |        |        |        |        | # v_n - a_n * dt = last vel
        |    1   |        |        |   -sp  |        |        |        |        |        | = last velocity
        |       1|        |        |     -sp|        |        |        |        |        |
        |--------------------------------------------------------------------------------|
        |-1      | 1      |        |        |-sp     |        |        |        |        | # -v_c + v_n - a_n * dt = 0
        |   -1   |    1   |        |        |   -sp  |        |        |        |        | = 0
        |      -1|       1|        |        |     -sp|        |        |        |        |
        |--------------------------------------------------------------------------------|
        |        |-1      | 1      |        |        |-sp     |        |        |        | # -v_c + v_n - a_n * dt = 0
        |        |   -1   |    1   |        |        |   -sp  |        |        |        | = 0
        |        |      -1|       1|        |        |     -sp|        |        |        |
        |================================================================================|
        |        |        |        | 1      |        |        |-sp     |        |        | # a_n - j_n * dt = last acc
        |        |        |        |    1   |        |        |   -sp  |        |        | = last acceleration
        |        |        |        |       1|        |        |     -sp|        |        |
        |--------------------------------------------------------------------------------|
        |        |        |        |-1      | 1      |        |        |-sp     |        | # -a_c + a_n - j_n * dt = 0
        |        |        |        |   -1   |    1   |        |        |   -sp  |        | = 0
        |        |        |        |      -1|       1|        |        |     -sp|        |
        |--------------------------------------------------------------------------------|
        |        |        |        |        |-1      | 1      |        |        |-sp     | # -a_c + a_n - j_n * dt = 0
        |        |        |        |        |   -1   |    1   |        |        |   -sp  | = 0
        |        |        |        |        |      -1|       1|        |        |     -sp|
        |--------------------------------------------------------------------------------|
        x_n - xd_n * dt = x_c
        - x_c + x_n - xd_n * dt = 0
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
        # todo replace hack?
        return model, slack_model


class InequalityModel(ProblemDataPart):
    """
    Format:
        free variable position constraints
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

    def position_limit_model(self) -> cas.Expression:
        """
        |   t1   |   t2   |   t3   |   t4   |
        |v1 v2 vn|v1 v2 vn|v1 v2 vn|v1 v2 vn|
        |-----------------------------------|
        |sp      |        |        |        |
        |   sp   |        |        |        |
        |      sp|        |        |        |
        |sp      |sp      |        |        |
        |   sp   |   sp   |        |        |
        |      sp|      sp|        |        |
        |sp      |sp      |sp      |        |
        |   sp   |   sp   |   sp   |        |
        |      sp|      sp|      sp|        |
        |sp      |sp      |sp      |sp      |
        |   sp   |   sp   |   sp   |   sp   |
        |      sp|      sp|      sp|      sp|
        |===================================|
        """
        model = cas.zeros(self.number_of_free_variables * self.prediction_horizon,
                          self.number_of_non_slack_columns)
        # assume all joints have position limits
        for p in range(1, self.prediction_horizon + 1):
            matrix_size = self.number_of_free_variables * p
            I = cas.eye(matrix_size) * self.dt
            model[-matrix_size:, :matrix_size] += I

        # delete rows corresponding to joints without limits
        rows_to_delete = []
        for variable_index, free_variable in enumerate(self.free_variables):
            if not free_variable.has_position_limits():
                for p in range(self.prediction_horizon):
                    rows_to_delete.append(variable_index + len(self.free_variables) * p)
        model.remove(rows_to_delete, [])
        return model

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
        position_limit_model = self.position_limit_model()
        vel_constr_model, vel_constr_slack_model = self.velocity_constraint_model()
        acc_constr_model, acc_constr_slack_model = self.acceleration_constraint_model()
        jerk_constr_model, jerk_constr_slack_model = self.jerk_constraint_model()
        inequality_model, inequality_slack_model = self.inequality_constraint_model()
        model_parts = []
        slack_model_parts = []
        if len(position_limit_model) > 0:
            model_parts.append(position_limit_model)
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
        combined_slack_model = cas.vstack([cas.zeros(position_limit_model.shape[0],
                                                     combined_slack_model.shape[1]),
                                           combined_slack_model])
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


# class qpSWIFTBuilder:
#     def __init__(self):


class QPProblemBuilder:
    """
    Wraps around QP Solver. Builds the required matrices from constraints.
    """
    debug_expressions: Dict[str, cas.all_expressions]
    compiled_debug_expressions: Dict[str, cas.CompiledFunction]
    evaluated_debug_expressions: Dict[str, np.ndarray]
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

    def __init__(self,
                 sample_period: float,
                 prediction_horizon: int,
                 solver_id: Optional[SupportedQPSolver] = None,
                 free_variables: List[FreeVariable] = None,
                 equality_constraints: List[EqualityConstraint] = None,
                 inequality_constraints: List[InequalityConstraint] = None,
                 derivative_constraints: List[DerivativeInequalityConstraint] = None,
                 debug_expressions: Dict[str, Union[cas.Symbol, float]] = None,
                 retries_with_relaxed_constraints: int = 0,
                 retry_added_slack: float = 100,
                 retry_weight_factor: float = 100):
        self.free_variables = []
        self.equality_constraints = []
        self.inequality_constraints = []
        self.derivative_constraints = []
        self.debug_expressions = {}
        self.prediction_horizon = prediction_horizon
        self.sample_period = sample_period
        self.retries_with_relaxed_constraints = retries_with_relaxed_constraints
        self.retry_added_slack = retry_added_slack
        self.retry_weight_factor = retry_weight_factor
        self.evaluated_debug_expressions = {}
        self.xdot_full = None
        if free_variables is not None:
            self.add_free_variables(free_variables)
        if inequality_constraints is not None:
            self.add_inequality_constraints(inequality_constraints)
        if equality_constraints is not None:
            self.add_equality_constraints(equality_constraints)
        if derivative_constraints is not None:
            self.add_derivative_constraints(derivative_constraints)
        if debug_expressions is not None:
            self.add_debug_expressions(debug_expressions)

        if solver_id is not None:
            self.qp_solver_class = available_solvers[solver_id]
        else:
            for solver_id in SupportedQPSolver:
                if solver_id in available_solvers:
                    self.qp_solver_class = available_solvers[solver_id]
                    break
            else:
                raise QPSolverException(f'No qp solver found')

        # num_non_slack = len(self.free_variables) * self.prediction_horizon * (self.order)

        logging.loginfo(f'Using QP Solver \'{solver_id.name}\'')
        logging.loginfo(f'Prediction horizon: \'{self.prediction_horizon}\'')
        self.qp_solver = self.compile(self.qp_solver_class)

    def add_free_variables(self, free_variables):
        """
        :type free_variables: list
        """
        if len(free_variables) == 0:
            raise QPSolverException('Cannot solve qp with no free variables')
        self.free_variables.extend(list(sorted(free_variables, key=lambda x: x.position_name)))
        l = [x.position_name for x in free_variables]
        duplicates = set([x for x in l if l.count(x) > 1])
        self.order = Derivatives(min(self.prediction_horizon, max(v.order for v in self.free_variables)))
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

    def add_debug_expressions(self, debug_expressions):
        """
        :type debug_expressions: dict
        """
        self.debug_expressions.update(debug_expressions)

    @profile
    def compile(self, solver_class: Type[QPSolver], default_limits: bool = False) -> QPSolver:
        logging.loginfo('Creating controller')
        kwargs = {'free_variables': self.free_variables,
                  'equality_constraints': self.equality_constraints,
                  'inequality_constraints': self.inequality_constraints,
                  'derivative_constraints': self.derivative_constraints,
                  'sample_period': self.sample_period,
                  'prediction_horizon': self.prediction_horizon,
                  'max_derivative': self.order}
        self.weights = Weights(**kwargs)
        self.free_variable_bounds = FreeVariableBounds(**kwargs)
        self.equality_model = EqualityModel(**kwargs)
        self.equality_bounds = EqualityBounds(**kwargs)
        self.inequality_model = InequalityModel(**kwargs)
        self.inequality_bounds = InequalityBounds(default_limits=default_limits, **kwargs)

        weights, g = self.weights.construct_expression()
        lb, ub = self.free_variable_bounds.construct_expression()
        A, A_slack = self.inequality_model.construct_expression()
        lbA, ubA = self.inequality_bounds.construct_expression()
        E, E_slack = self.equality_model.construct_expression()
        bE = self.equality_bounds.construct_expression()

        qp_solver = solver_class(weights=weights, g=g, lb=lb, ub=ub,
                                 E=E, E_slack=E_slack, bE=bE,
                                 A=A, A_slack=A_slack, lbA=lbA, ubA=ubA)
        logging.loginfo('Done compiling controller:')
        logging.loginfo(f'  #free variables: {weights.shape[0]}')
        logging.loginfo(f'  #equality constraints: {bE.shape[0]}')
        logging.loginfo(f'  #inequality constraints: {lbA.shape[0]}')
        self._compile_debug_expressions()
        return qp_solver

    def get_parameter_names(self):
        return self.qp_solver.free_symbols_str

    def _compile_debug_expressions(self):
        self.compiled_debug_expressions = {}
        free_symbols = set()
        for name, expr in self.debug_expressions.items():
            free_symbols.update(expr.free_symbols())
        free_symbols = list(free_symbols)
        for name, expr in self.debug_expressions.items():
            self.compiled_debug_expressions[name] = expr.compile(free_symbols)
        num_debug_expressions = len(self.compiled_debug_expressions)
        if num_debug_expressions > 0:
            logging.loginfo(f'  #debug expressions: {len(self.compiled_debug_expressions)}')

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
                [self.p_weights, self.p_lb, self.p_ub,
                 self.p_E, self.p_bE,
                 self.p_A, self.p_lbA, self.p_ubA,
                 self.p_debug, self.p_xdot],
                ['weights', 'lb', 'ub', 'E', 'bE', 'A', 'lbA', 'ubA', 'debug', 'xdot'],
                self.god_map.get_data(identifier.tmp_folder))
        else:
            save_pandas(
                [self.p_weights, self.p_lb, self.p_ub,
                 self.p_E, self.p_bE,
                 self.p_A, self.p_lbA, self.p_ubA,
                 self.p_debug],
                ['weights', 'lb', 'ub', 'E', 'bE', 'A', 'lbA', 'ubA', 'debug'],
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

    @profile
    def eval_debug_exprs(self):
        self.evaluated_debug_expressions = {}
        for name, f in self.compiled_debug_expressions.items():
            params = self.god_map.get_values(f.str_params)
            self.evaluated_debug_expressions[name] = f.fast_call(params).copy()
        return self.evaluated_debug_expressions

    def __swap_compiled_matrices(self):
        if not hasattr(self, 'qp_solver_default_limits'):
            with suppress_stdout():
                self.qp_solver_default_limits = self.compile(self.qp_solver_class, default_limits=True)
        self.qp_solver, self.qp_solver_default_limits = self.qp_solver_default_limits, self.qp_solver

    @property
    def traj_time_in_sec(self):
        return self.god_map.unsafe_get_data(identifier.time) * self.god_map.unsafe_get_data(identifier.sample_period)

    @profile
    def get_cmd(self, substitutions: np.ndarray) -> NextCommands:
        """
        Uses substitutions for each symbol to compute the next commands for each joint.
        """
        try:
            self.xdot_full = self.qp_solver.solve_and_retry(substitutions=substitutions)
            # self._create_debug_pandas()
            return NextCommands(free_variables=self.free_variables, xdot=self.xdot_full, max_derivative=self.order,
                                prediction_horizon=self.prediction_horizon)
        except InfeasibleException as e_original:
            if isinstance(e_original, HardConstraintsViolatedException):
                raise
            self.xdot_full = None
            self._create_debug_pandas(self.qp_solver)
            joint_limits_violated_msg = self._are_joint_limits_violated()
            if joint_limits_violated_msg is not None:
                logging.logwarn('Joint limits violated, trying to fix.')
                self.__swap_compiled_matrices()
                try:
                    self.xdot_full = self.qp_solver.solve_and_retry(substitutions=substitutions)
                    return NextCommands(free_variables=self.free_variables, xdot=self.xdot_full,
                                        max_derivative=self.order, prediction_horizon=self.prediction_horizon)
                except Exception as e2:
                    # self._create_debug_pandas()
                    # raise OutOfJointLimitsException(self._are_joint_limits_violated())
                    raise OutOfJointLimitsException(joint_limits_violated_msg)
                finally:
                    self.__swap_compiled_matrices()
                    # self.free_variables[0].god_map.get_data(['world']).state.pretty_print()
            self._are_hard_limits_violated(str(e_original))
            # self._is_inf_in_data()
            raise

    def _are_hard_limits_violated(self, error_message):
        # num_non_slack = len(self.free_variables) * self.prediction_horizon * (self.order)
        # num_of_slack = len(self.np_lb_filtered) - num_non_slack
        # lb = self.np_lb_filtered.copy()
        # lb[-num_of_slack:] = -100
        # ub = self.np_ub_filtered.copy()
        # ub[-num_of_slack:] = 100
        # try:
        #     self.xdot_full = self.qp_solver.solve_and_retry(weights=self.np_weights_filtered,
        #                                           g=self.np_g_filtered,
        #                                           A=self.np_A_filtered,
        #                                           lb=self.np_lb_filtered,
        #                                           ub=self.np_ub_filtered,
        #                                           lbA=self.np_lbA_filtered,
        #                                           ubA=self.np_ubA_filtered)
        # except Exception as e:
        #     logging.loginfo(f'Can\'t determine if hard constraints are violated: {e}.')
        #     return False
        # else:
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
    def _create_debug_pandas(self, qp_solver):
        weights, g, lb, ub, E, bE, A, lbA, ubA, weight_filter, bE_filter, bA_filter = qp_solver.get_problem_data()
        # substitutions = self.substitutions
        # self.state = {k: v for k, v in zip(self.compiled_big_ass_M.str_params, substitutions)}
        sample_period = self.sample_period
        self.free_variable_names = self.free_variable_bounds.names[weight_filter]
        self.equality_constr_names = self.equality_bounds.names[bE_filter]
        self.inequality_constr_names = self.inequality_bounds.names[bA_filter]

        # self._eval_debug_exprs()
        p_debug = {}
        for name, value in self.evaluated_debug_expressions.items():
            if isinstance(value, np.ndarray):
                p_debug[name] = value.reshape((value.shape[0] * value.shape[1]))
            else:
                p_debug[name] = np.array(value)
        self.p_debug = pd.DataFrame.from_dict(p_debug, orient='index').sort_index()

        self.p_weights = pd.DataFrame(weights, self.free_variable_names, ['data'], dtype=float)
        # self.p_g = pd.DataFrame(g, self.free_variable_names, ['data'], dtype=float)
        self.p_lb = pd.DataFrame(lb, self.free_variable_names, ['data'], dtype=float)
        self.p_ub = pd.DataFrame(ub, self.free_variable_names, ['data'], dtype=float)
        self.p_bE_raw = pd.DataFrame(bE, self.equality_constr_names, ['data'], dtype=float)
        self.p_bE = deepcopy(self.p_bE_raw)
        self.p_bE[len(self.equality_bounds.names_derivative_links):] /= sample_period
        self.p_lbA_raw = pd.DataFrame(lbA, self.inequality_constr_names, ['data'], dtype=float)
        self.p_lbA = deepcopy(self.p_lbA_raw)
        self.p_lbA[len(self.inequality_bounds.names_position_limits):] /= sample_period
        self.p_ubA_raw = pd.DataFrame(ubA, self.inequality_constr_names, ['data'], dtype=float)
        self.p_ubA = deepcopy(self.p_ubA_raw)
        self.p_ubA[len(self.inequality_bounds.names_position_limits):] /= sample_period
        # remove sample period factor
        self.p_E = pd.DataFrame(E, self.equality_constr_names, self.free_variable_names, dtype=float)
        self.p_A = pd.DataFrame(A, self.inequality_constr_names, self.free_variable_names, dtype=float)
        self.p_xdot = None
        if self.xdot_full is not None:
            self.p_xdot = pd.DataFrame(self.xdot_full, self.free_variable_names, ['data'], dtype=float)
            # Ax = np.dot(self.np_A, xdot_full)
            # xH = np.dot((self.xdot_full ** 2).T, H)
            # self.p_xH = pd.DataFrame(xH, filtered_b_names, ['data'], dtype=float)
            # p_xg = p_g * p_xdot
            # xHx = np.dot(np.dot(xdot_full.T, H), xdot_full)

            # self.p_pure_xdot = deepcopy(self.p_xdot)
            # self.p_pure_xdot[-num_constr:] = 0
            # self.p_Ax = pd.DataFrame(self.p_A.dot(self.p_xdot), filtered_bA_names, ['data'], dtype=float)
            # self.p_Ax_without_slack_raw = pd.DataFrame(self.p_A.dot(self.p_pure_xdot), filtered_bA_names, ['data'],
            #                                            dtype=float)
            # self.p_Ax_without_slack = deepcopy(self.p_Ax_without_slack_raw)
            # self.p_Ax_without_slack[-num_constr:] /= sample_period

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
