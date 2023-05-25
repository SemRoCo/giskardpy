from collections import defaultdict
from typing import Dict, Optional, List, Union
import numpy as np
import giskardpy.casadi_wrapper as w
from giskardpy import identifier
from giskardpy.god_map import GodMap
from giskardpy.my_types import Derivatives, PrefixName
from giskardpy.utils.decorators import memoize


class FreeVariable:
    state_identifier: List[str] = identifier.joint_states

    def __init__(self,
                 name: PrefixName,
                 lower_limits: Dict[Derivatives, float],
                 upper_limits: Dict[Derivatives, float],
                 quadratic_weights: Optional[Dict[Derivatives, float]] = None,
                 horizon_functions: Optional[Dict[Derivatives, float]] = None):
        self.god_map = GodMap()
        self._symbols = {}
        self.name = name
        for derivative in Derivatives:
            self._symbols[derivative] = self.god_map.to_symbol(self.state_identifier + [name, derivative])
        self.position_name = str(self._symbols[Derivatives.position])
        self.default_lower_limits = lower_limits
        self.default_upper_limits = upper_limits
        self.lower_limits = {}
        self.upper_limits = {}
        if quadratic_weights is None:
            self.quadratic_weights = {}
            for i in range(self.god_map.get_data(identifier.max_derivative)):
                derivative = Derivatives(i + 1)
                quadratic_weight_symbol = self.god_map.to_symbol(identifier.joint_weights + [derivative, self.name])
                self.quadratic_weights[derivative] = quadratic_weight_symbol
        else:
            self.quadratic_weights = quadratic_weights
        assert max(self._symbols.keys()) == len(self._symbols) - 1

        self.horizon_functions = defaultdict(lambda: 0.00001)
        if horizon_functions is None:
            horizon_functions = {Derivatives.velocity: 0.1,
                                 Derivatives.acceleration: 0.1,
                                 Derivatives.jerk: 0.1}
        self.horizon_functions.update(horizon_functions)

    @property
    def order(self) -> Derivatives:
        return self.god_map.get_data(identifier.max_derivative)

    def get_symbol(self, derivative: Derivatives) -> Union[w.Symbol, float]:
        try:
            return self._symbols[derivative]
        except KeyError:
            raise KeyError(f'Free variable {self} doesn\'t have symbol for derivative of order {derivative}')

    def reset_cache(self):
        for method_name in dir(self):
            try:
                getattr(self, method_name).memo.clear()
            except:
                pass

    @memoize
    def get_lower_limit(self, derivative: Derivatives, default: bool = False, evaluated: bool = False) -> Union[
        w.Expression, float]:
        if not default and derivative in self.default_lower_limits and derivative in self.lower_limits:
            expr = w.max(self.default_lower_limits[derivative], self.lower_limits[derivative])
        elif derivative in self.default_lower_limits:
            expr = self.default_lower_limits[derivative]
        elif derivative in self.lower_limits:
            expr = self.lower_limits[derivative]
        else:
            raise KeyError(f'Free variable {self} doesn\'t have lower limit for derivative of order {derivative}')
        if evaluated:
            return float(self.god_map.evaluate_expr(expr))
        return expr

    def set_lower_limit(self, derivative: Derivatives, limit: Union[w.Expression, float]):
        self.lower_limits[derivative] = limit

    def set_upper_limit(self, derivative: Derivatives, limit: Union[Union[w.Symbol, float], float]):
        self.upper_limits[derivative] = limit

    @memoize
    def get_upper_limit(self, derivative: Derivatives, default: bool = False, evaluated: bool = False) \
            -> Union[Union[w.Symbol, float], float]:
        if not default and derivative in self.default_upper_limits and derivative in self.upper_limits:
            expr = w.min(self.default_upper_limits[derivative], self.upper_limits[derivative])
        elif derivative in self.default_upper_limits:
            expr = self.default_upper_limits[derivative]
        elif derivative in self.upper_limits:
            expr = self.upper_limits[derivative]
        else:
            raise KeyError(f'Free variable {self} doesn\'t have upper limit for derivative of order {derivative}')
        if evaluated:
            return self.god_map.evaluate_expr(expr)
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

    def has_position_limits(self) -> bool:
        try:
            lower_limit = self.get_lower_limit(Derivatives.position)
            upper_limit = self.get_upper_limit(Derivatives.position)
            return lower_limit is not None and upper_limit is not None
        except KeyError:
            return False

    @memoize
    @profile
    def normalized_weight(self, t: int, derivative: Derivatives, prediction_horizon: int,
                          evaluated: bool = False) -> Union[Union[w.Symbol, float], float]:
        weight = self.quadratic_weights[derivative]
        start = weight * self.horizon_functions[derivative]
        a = (weight - start) / prediction_horizon
        weight = a * t + start
        expr = weight * (1 / self.get_upper_limit(derivative)) ** 2
        if evaluated:
            return self.god_map.evaluate_expr(expr)

    def __str__(self) -> str:
        return self.position_name

    def __repr__(self):
        return str(self)
