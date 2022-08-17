from collections import defaultdict
from typing import Dict, Optional, List, Union

import giskardpy.casadi_wrapper as w
from giskardpy import identifier
from giskardpy.data_types import derivative_to_name
from giskardpy.god_map import GodMap
from giskardpy.my_types import expr_symbol


class FreeVariable:
    state_identifier: List[str] = identifier.joint_states

    def __init__(self,
                 name: str,
                 god_map: GodMap,
                 lower_limits: Dict[int, float],
                 upper_limits: Dict[int, float],
                 quadratic_weights: Optional[Dict[int, float]] = None,
                 horizon_functions: Optional[Dict[int, float]] = None):
        self.god_map = god_map
        self._symbols = {}
        self.name = name
        for derivative_number, derivative_name in derivative_to_name.items():
            self._symbols[derivative_number] = self.god_map.to_symbol(self.state_identifier + [name, derivative_name])
        self.position_name = str(self._symbols[0])
        self.default_lower_limits = lower_limits
        self.default_upper_limits = upper_limits
        self.lower_limits = {}
        self.upper_limits = {}
        if quadratic_weights is None:
            self.quadratic_weights = {}
        else:
            self.quadratic_weights = quadratic_weights
        assert max(self._symbols.keys()) == len(self._symbols) - 1

        self.horizon_functions = defaultdict(float)
        if horizon_functions is None:
            horizon_functions = {1: 0.1}
        self.horizon_functions.update(horizon_functions)

    @property
    def order(self) -> int:
        return len(self.quadratic_weights) + 1

    def get_symbol(self, order: int) -> expr_symbol:
        try:
            return self._symbols[order]
        except KeyError:
            raise KeyError(f'Free variable {self} doesn\'t have symbol for derivative of order {order}')

    def get_lower_limit(self, order: int, default: bool = False, evaluated: bool = False) -> Union[expr_symbol, float]:
        if not default and order in self.default_lower_limits and order in self.lower_limits:
            expr = w.max(self.default_lower_limits[order], self.lower_limits[order])
        elif order in self.default_lower_limits:
            expr = self.default_lower_limits[order]
        elif order in self.lower_limits:
            expr = self.lower_limits[order]
        else:
            raise KeyError(f'Free variable {self} doesn\'t have lower limit for derivative of order {order}')
        if evaluated:
            return self.god_map.evaluate_expr(expr)
        return expr

    def set_lower_limit(self, order: int, limit: Union[expr_symbol, float]):
        self.lower_limits[order] = limit

    def set_upper_limit(self, order: int, limit: Union[expr_symbol, float]):
        self.upper_limits[order] = limit

    def get_upper_limit(self, order: int, default: bool = False, evaluated: bool = False) -> Union[expr_symbol, float]:
        if not default and order in self.default_upper_limits and order in self.upper_limits:
            expr = w.min(self.default_upper_limits[order], self.upper_limits[order])
        elif order in self.default_upper_limits:
            expr = self.default_upper_limits[order]
        elif order in self.upper_limits:
            expr = self.upper_limits[order]
        else:
            raise KeyError(f'Free variable {self} doesn\'t have upper limit for derivative of order {order}')
        if evaluated:
            return self.god_map.evaluate_expr(expr)
        return expr

    def has_position_limits(self) -> bool:
        try:
            lower_limit = self.get_lower_limit(0)
            upper_limit = self.get_upper_limit(0)
            return lower_limit is not None and abs(lower_limit) < 100 \
                   and upper_limit is not None and abs(upper_limit) < 100
        except Exception:
            return False

    def normalized_weight(self, t: int, order: int, prediction_horizon: int,
                          evaluated: bool = False) -> Union[expr_symbol, float]:
        weight = self.quadratic_weights[order]
        start = weight * self.horizon_functions[order]
        a = (weight - start) / prediction_horizon
        weight = a * t + start
        expr = weight * (1 / self.get_upper_limit(order)) ** 2
        if evaluated:
            return self.god_map.evaluate_expr(expr)

    def __str__(self) -> str:
        return self.position_name
