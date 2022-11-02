from collections import namedtuple
from typing import List, Union, Optional, Callable

import giskardpy.casadi_wrapper as w
from giskardpy.my_types import expr_symbol

DebugConstraint = namedtuple('debug', ['expr'])


class Constraint:
    lower_error = -1e4
    upper_error = 1e4
    lower_slack_limit = -1e4
    upper_slack_limit = 1e4
    linear_weight = 0

    def __init__(self, name, expression,
                 lower_error, upper_error,
                 velocity_limit,
                 quadratic_weight, control_horizon, linear_weight=None,
                 lower_slack_limit=None, upper_slack_limit=None, ):
        self.name = name
        self.expression = expression
        self.quadratic_weight = quadratic_weight
        self.control_horizon = control_horizon
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

    def normalized_weight(self, prediction_horizon):
        weight_normalized = self.quadratic_weight * (1 / (self.velocity_limit)) ** 2
        return weight_normalized


class VelocityConstraint:
    lower_acceleration_limit = -1e4
    upper_acceleration_limit = 1e4
    lower_slack_limit = -1e3
    upper_slack_limit = 1e3
    linear_weight = 0

    def __init__(self,
                 name,
                 expression,
                 lower_velocity_limit: Union[expr_symbol, List[expr_symbol]],
                 upper_velocity_limit: Union[expr_symbol, List[expr_symbol]],
                 velocity_limit,
                 quadratic_weight,
                 control_horizon,
                 lower_slack_limit: Union[expr_symbol, List[expr_symbol]],
                 upper_slack_limit: Union[expr_symbol, List[expr_symbol]],
                 linear_weight=None,
                 horizon_function: Optional[Callable[[float, int], float]] = None):
        self.name = name
        self.expression = expression
        self.quadratic_weight = quadratic_weight
        self.control_horizon = control_horizon
        self.velocity_limit = velocity_limit
        if self.is_iterable(lower_velocity_limit):
            self.lower_velocity_limit = lower_velocity_limit
        else:
            self.lower_velocity_limit = [lower_velocity_limit] * self.control_horizon

        if self.is_iterable(upper_velocity_limit):
            self.upper_velocity_limit = upper_velocity_limit
        else:
            self.upper_velocity_limit = [upper_velocity_limit] * self.control_horizon

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
        if isinstance(thing, w.ca.SX) and sum(thing.shape) == 2:
            return False
        return hasattr(thing, '__iter__')

    def normalized_weight(self, t):
        weight_normalized = self.quadratic_weight * (1 / self.velocity_limit) ** 2
        return self.horizon_function(weight_normalized, t)
