from collections import namedtuple
from typing import List, Union, Optional, Callable

import giskardpy.casadi_wrapper as cas
from giskardpy.god_map import god_map
from giskardpy.data_types import Derivatives, PrefixName

DebugConstraint = namedtuple('debug', ['expr'])


class Constraint:
    _name: str
    _parent_task_name: PrefixName

    def __init__(self, name: str, parent_task_name: PrefixName):
        self._name = name
        self._parent_task_name = parent_task_name

    @property
    def name(self) -> str:
        return str(PrefixName(self._name, self._parent_task_name))


class InequalityConstraint(Constraint):
    lower_error = -1e4
    upper_error = 1e4
    lower_slack_limit = -1e4
    upper_slack_limit = 1e4
    linear_weight = 0

    def __init__(self,
                 name: str,
                 parent_task_name: PrefixName,
                 expression: cas.Expression,
                 lower_error: cas.symbol_expr_float, upper_error: cas.symbol_expr_float,
                 velocity_limit: cas.symbol_expr_float,
                 quadratic_weight: cas.symbol_expr_float,
                 control_horizon: Optional[int] = None,
                 linear_weight: Optional[cas.symbol_expr_float] = None,
                 lower_slack_limit: Optional[cas.symbol_expr_float] = None,
                 upper_slack_limit: Optional[cas.symbol_expr_float] = None):
        super().__init__(name, parent_task_name)
        self.expression = expression
        self.quadratic_weight = quadratic_weight
        if control_horizon is None:
            self.control_horizon = god_map.qp_controller_config.prediction_horizon - (
                    god_map.qp_controller_config.max_derivative - 1)
        else:
            self.control_horizon = control_horizon
        self.control_horizon = max(1, self.control_horizon)
        self.velocity_limit = velocity_limit
        self.lower_error = lower_error
        self.upper_error = upper_error
        if lower_slack_limit is not None:
            self.lower_slack_limit = lower_slack_limit
        if upper_slack_limit is not None:
            self.upper_slack_limit = upper_slack_limit
        if linear_weight is not None:
            self.linear_weight = linear_weight

    def __str__(self):
        return self.name

    def normalized_weight(self):
        weight_normalized = self.quadratic_weight * (1 / (self.velocity_limit)) ** 2
        return weight_normalized * self.control_horizon


class EqualityConstraint(Constraint):
    bound = 0
    lower_slack_limit = -1e4
    upper_slack_limit = 1e4
    linear_weight = 0

    def __init__(self,
                 name: str,
                 parent_task_name: PrefixName,
                 expression: cas.Expression,
                 derivative_goal: cas.symbol_expr_float,
                 velocity_limit: cas.symbol_expr_float,
                 quadratic_weight: cas.symbol_expr_float, control_horizon: int,
                 linear_weight: Optional[cas.symbol_expr_float] = None,
                 lower_slack_limit: Optional[cas.symbol_expr_float] = None,
                 upper_slack_limit: Optional[cas.symbol_expr_float] = None):
        super().__init__(name, parent_task_name)
        self.expression = expression
        self.quadratic_weight = quadratic_weight
        if control_horizon is None:
            self.control_horizon = god_map.qp_controller_config.prediction_horizon - (
                    god_map.qp_controller_config.max_derivative - 1)
        else:
            self.control_horizon = control_horizon
        self.control_horizon = max(1, self.control_horizon)
        self.velocity_limit = velocity_limit
        self.bound = derivative_goal
        if lower_slack_limit is not None:
            self.lower_slack_limit = lower_slack_limit
        if upper_slack_limit is not None:
            self.upper_slack_limit = upper_slack_limit
        if linear_weight is not None:
            self.linear_weight = linear_weight

    def __str__(self):
        return self.name

    def normalized_weight(self):
        weight_normalized = self.quadratic_weight * (1 / self.velocity_limit) ** 2
        return weight_normalized * self.control_horizon

    def capped_bound(self, dt: float):
        return cas.limit(self.bound,
                         -self.velocity_limit * dt * self.control_horizon,
                         self.velocity_limit * dt * self.control_horizon)


class DerivativeInequalityConstraint(Constraint):

    def __init__(self,
                 name: str,
                 parent_task_name: PrefixName,
                 derivative: Derivatives,
                 expression: cas.Expression,
                 lower_limit: Union[cas.symbol_expr_float, List[cas.symbol_expr_float]],
                 upper_limit: Union[cas.symbol_expr_float, List[cas.symbol_expr_float]],
                 quadratic_weight: cas.symbol_expr_float,
                 normalization_factor: Optional[cas.symbol_expr_float],
                 lower_slack_limit: Union[cas.symbol_expr_float, List[cas.symbol_expr_float]],
                 upper_slack_limit: Union[cas.symbol_expr_float, List[cas.symbol_expr_float]],
                 control_horizon: Optional[cas.symbol_expr_float] = None,
                 linear_weight: cas.symbol_expr_float = None,
                 horizon_function: Optional[Callable[[float, int], float]] = None):
        super().__init__(name, parent_task_name)
        self.derivative = derivative
        self.expression = expression
        self.quadratic_weight = quadratic_weight
        self.control_horizon = control_horizon if control_horizon is not None else max(
            god_map.qp_controller_config.prediction_horizon - 2, 1)
        self.normalization_factor = normalization_factor
        if self.is_iterable(lower_limit):
            self.lower_limit = lower_limit
        else:
            self.lower_limit = [lower_limit] * self.control_horizon

        if self.is_iterable(upper_limit):
            self.upper_limit = upper_limit
        else:
            self.upper_limit = [upper_limit] * self.control_horizon

        if self.is_iterable(lower_slack_limit):
            self.lower_slack_limit = lower_slack_limit
        else:
            self.lower_slack_limit = [lower_slack_limit] * self.control_horizon

        if self.is_iterable(upper_slack_limit):
            self.upper_slack_limit = upper_slack_limit
        else:
            self.upper_slack_limit = [upper_slack_limit] * self.control_horizon

        if linear_weight is not None:
            self.linear_weight = linear_weight

        def default_horizon_function(weight, t):
            return weight

        self.horizon_function = default_horizon_function
        if horizon_function is not None:
            self.horizon_function = horizon_function

    def is_iterable(self, thing):
        if isinstance(thing, cas.ca.SX) and sum(thing.shape) == 2:
            return False
        return hasattr(thing, '__iter__')

    def normalized_weight(self, t):
        weight_normalized = self.quadratic_weight * (1 / self.normalization_factor) ** 2
        return self.horizon_function(weight_normalized, t)


class ManipulabilityConstraint(Constraint):
    def __init__(self,
                 name: str,
                 parent_task_name: PrefixName,
                 expression: cas.Expression,
                 gain: float = 1,
                 prediction_horizon: int = 1):
        super().__init__(name, parent_task_name)
        self.expression = expression
        if gain < 0:
            raise Exception('Manipulability gain value has to be positive')
        self.gain = gain
        self.prediction_horizon = prediction_horizon

    def __str__(self):
        return self.name
