from collections import defaultdict
from typing import Dict, Optional, Union

import numpy as np

import giskardpy.casadi_wrapper as cas
from giskardpy.data_types.data_types import Derivatives, PrefixName
from giskardpy.god_map import god_map
from giskardpy.symbol_manager import symbol_manager
from giskardpy.utils.decorators import memoize
from line_profiler import profile

def my_cycloid(x_in: float, weight: float, h: int, alpha: float) -> float:
    from scipy.optimize import fsolve
    start_y = weight * alpha
    end_y = weight
    x = (x_in / h) * (np.pi / 2)
    r = 1

    def equation(theta):
        return theta - np.sin(theta) - (2 * x) / r

    # Solve for theta using fsolve with a fixed initial guess
    theta = fsolve(equation, 0.0)[0]
    y = r / 2 * (1 - np.cos(theta))
    y *= (end_y - start_y)
    y += start_y
    return y

def my_cycloid2(x_in: float, weight: float, h: int, alpha: float, q: float) -> float:
    from scipy.optimize import fsolve
    x_in *= q
    start_y = weight * alpha
    end_y = weight
    x = (x_in / h) * (np.pi / 2)
    r = 1

    def equation(theta):
        return theta - np.sin(theta) - (2 * x) / r

    # Solve for theta using fsolve with a fixed initial guess
    theta = fsolve(equation, 0.0)[0]
    y = r / 2 * (1 - np.cos(theta))
    y *= (end_y - start_y)
    y += start_y
    return y


def first_low(x_in: float, weight: float, h: int, alpha: float) -> float:
    if x_in == 0:
        return weight * alpha
    return weight


def linear(x_in: float, weight: float, h: int, alpha: float) -> float:
    start = weight * alpha
    a = (weight - start) / (h)
    return a * x_in + start


def even(x_in: float, weight: float, h: int, alpha: float) -> float:
    return weight


def quadratic(x_in: float, weight: float, h: int, alpha: float, q: float) -> float:
    start = weight * alpha
    a = (weight - start) / (h) ** q
    return a * x_in ** q + start

def sigmoid(x_in: float, weight: float, h: int, alpha: float, q: float) -> float:
    start = weight*alpha
    return (weight-start)*(2 * (1/(1 + np.exp(-q * x_in))) - 1) + start


def parabel(x_in: float, weight: float, h: int, alpha: float, q: float) -> float:
    # Setting up the boundary conditions
    y0 = weight * alpha  # Value at x = 0
    y_half_h = weight  # Value at x = h/2

    # Solving for the coefficients a, b, c in ax^2 + bx + c
    a = 4 * (y0 - y_half_h) / (h ** q)
    b = -4 * (y0 - y_half_h) / h
    c = y0

    # Calculating the output for the given x_in
    return a * x_in ** q + b * x_in + c


class FreeVariable:
    is_base: bool
    threshold: float = 0.005

    def __init__(self,
                 name: PrefixName,
                 lower_limits: Dict[Derivatives, float],
                 upper_limits: Dict[Derivatives, float],
                 quadratic_weights: Dict[Derivatives, float],
                 horizon_functions: Optional[Dict[Derivatives, float]] = None,
                 is_base: bool = False):
        self.is_base = is_base
        self._symbols = {}
        self.name = name
        for derivative in Derivatives:
            symbol_name = f'{self.name}_{derivative.name}'
            self._symbols[derivative] = symbol_manager.register_symbol(symbol_name, lambda n=name, d=derivative: god_map.world.state[n][d])
        self.position_name = str(self._symbols[Derivatives.position])
        self.default_lower_limits = lower_limits
        self.default_upper_limits = upper_limits
        self.lower_limits = {}
        self.upper_limits = {}
        self.quadratic_weights = quadratic_weights
        assert max(self._symbols.keys()) == len(self._symbols) - 1

        self.horizon_functions = defaultdict(lambda: 0.00001)
        if horizon_functions is None:
            horizon_functions = {Derivatives.velocity: 1,
                                 Derivatives.acceleration: 0.1,
                                 Derivatives.jerk: 0.1}
        self.horizon_functions.update(horizon_functions)

    def get_symbol(self, derivative: Derivatives) -> Union[cas.Symbol, float]:
        try:
            return self._symbols[derivative]
        except KeyError:
            raise KeyError(f'Free variable {self} doesn\'t have symbol for derivative of order {derivative}')

    @property
    def position_symbol(self) -> cas.Symbol:
        return self.get_symbol(Derivatives.position)

    @property
    def velocity_symbol(self) -> cas.Symbol:
        return self.get_symbol(Derivatives.velocity)

    @property
    def acceleration_symbol(self) -> cas.Symbol:
        return self.get_symbol(Derivatives.acceleration)

    @property
    def jerk_symbol(self) -> cas.Symbol:
        return self.get_symbol(Derivatives.jerk)

    def reset_cache(self):
        for method_name in dir(self):
            try:
                getattr(self, method_name).memo.clear()
            except:
                pass

    @memoize
    def get_lower_limit(self, derivative: Derivatives, default: bool = False, evaluated: bool = False) \
            -> Union[cas.Expression, float]:
        if not default and derivative in self.default_lower_limits and derivative in self.lower_limits:
            expr = cas.max(self.default_lower_limits[derivative], self.lower_limits[derivative])
        elif derivative in self.default_lower_limits:
            expr = self.default_lower_limits[derivative]
        elif derivative in self.lower_limits:
            expr = self.lower_limits[derivative]
        else:
            raise KeyError(f'Free variable {self} doesn\'t have lower limit for derivative of order {derivative}')
        if evaluated:
            if expr is None:
                return None
            return float(symbol_manager.evaluate_expr(expr))
        return expr

    def set_lower_limit(self, derivative: Derivatives, limit: Union[cas.Expression, float]):
        self.lower_limits[derivative] = limit

    def set_upper_limit(self, derivative: Derivatives, limit: Union[Union[cas.Symbol, float], float]):
        self.upper_limits[derivative] = limit

    @memoize
    def get_upper_limit(self, derivative: Derivatives, default: bool = False, evaluated: bool = False) \
            -> Union[Union[cas.Symbol, float], float]:
        if not default and derivative in self.default_upper_limits and derivative in self.upper_limits:
            expr = cas.min(self.default_upper_limits[derivative], self.upper_limits[derivative])
        elif derivative in self.default_upper_limits:
            expr = self.default_upper_limits[derivative]
        elif derivative in self.upper_limits:
            expr = self.upper_limits[derivative]
        else:
            raise KeyError(f'Free variable {self} doesn\'t have upper limit for derivative of order {derivative}')
        if evaluated:
            if expr is None:
                return None
            return symbol_manager.evaluate_expr(expr)
        return expr

    def get_lower_limits(self, max_derivative: Derivatives) -> Dict[Derivatives, float]:
        lower_limits = {}
        for derivative in Derivatives.range(Derivatives.position, max_derivative):
            lower_limits[derivative] = self.get_lower_limit(derivative, default=False, evaluated=True)
        return lower_limits

    def get_upper_limits(self, max_derivative: Derivatives) -> Dict[Derivatives, float]:
        upper_limits = {}
        for derivative in Derivatives.range(Derivatives.position, max_derivative):
            upper_limits[derivative] = self.get_upper_limit(derivative, default=False, evaluated=True)
        return upper_limits

    @memoize
    def has_position_limits(self) -> bool:
        try:
            lower_limit = self.get_lower_limit(Derivatives.position)
            upper_limit = self.get_upper_limit(Derivatives.position)
            return lower_limit is not None and upper_limit is not None
        except KeyError:
            return False

    @memoize
    @profile
    def normalized_weight(self, t: int, derivative: Derivatives, prediction_horizon: int, alpha: float,
                          evaluated: bool = False) -> Union[Union[cas.Symbol, float], float]:
        limit = self.get_upper_limit(derivative)
        if limit is None:
            return 0.0
        weight = symbol_manager.evaluate_expr(self.quadratic_weights[derivative])
        limit = symbol_manager.evaluate_expr(limit)

        # weight = my_cycloid(t, weight, prediction_horizon, alpha) # -0.6975960014626146
        # weight = my_cycloid2(t, weight, prediction_horizon, alpha, q=2) # -0.7023704675764478
        # weight = first_low(t, weight, prediction_horizon, alpha)
        # weight = even(t, weight, prediction_horizon, alpha)
        weight = linear(t, weight, prediction_horizon, alpha)  # -0.857950480223123
        # weight = parabel(t, weight, prediction_horizon, alpha, 2)  # -0.8332195779250338
        # weight = quadratic(t, weight, prediction_horizon, alpha, 1/1.59)  # -0.6672011565358271
        # weight = sigmoid(t, weight, prediction_horizon, alpha, 1.5)

        return weight * (1 / limit) ** 2

    def __str__(self) -> str:
        return self.position_name

    def __repr__(self):
        return str(self)
