from collections import defaultdict
from typing import Dict, Optional

import giskardpy.casadi_wrapper as w


class FreeVariable(object):
    def __init__(self,
                 symbols: Dict[int, w.ca.SX],
                 lower_limits: Dict[int, float],
                 upper_limits: Dict[int, float],
                 quadratic_weights: Optional[Dict[int, float]]= None,
                 horizon_functions: Optional[Dict[int, float]] = None):
        self._symbols = symbols
        self.name = str(self._symbols[0])
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
    def order(self):
        return len(self.quadratic_weights) + 1

    def get_symbol(self, order):
        try:
            return self._symbols[order]
        except KeyError:
            raise KeyError('Free variable {} doesn\'t have symbol for derivative of order {}'.format(self, order))

    def get_lower_limit(self, order):
        if order in self.default_lower_limits and order in self.lower_limits:
            return w.max(self.default_lower_limits[order], self.lower_limits[order])
        if order in self.default_lower_limits:
            return self.default_lower_limits[order]
        if order in self.lower_limits:
            return self.lower_limits[order]
        raise KeyError('Free variable {} doesn\'t have lower limit for derivative of order {}'.format(self, order))

    def set_lower_limit(self, order, limit):
        self.lower_limits[order] = limit

    def set_upper_limit(self, order, limit):
        self.upper_limits[order] = limit

    def get_upper_limit(self, order):
        if order in self.default_upper_limits and order in self.upper_limits:
            return w.min(self.default_upper_limits[order], self.upper_limits[order])
        if order in self.default_upper_limits:
            return self.default_upper_limits[order]
        if order in self.upper_limits:
            return self.upper_limits[order]
        raise KeyError('Free variable {} doesn\'t have upper limit for derivative of order {}'.format(self, order))

    def has_position_limits(self):
        try:
            lower_limit = self.get_lower_limit(0)
            upper_limit = self.get_upper_limit(0)
            return lower_limit is not None and abs(lower_limit) < 100 \
                   and upper_limit is not None and abs(upper_limit) < 100
        except Exception:
            return False

    def normalized_weight(self, t, order, prediction_horizon):
        weight = self.quadratic_weights[order]
        start = weight * self.horizon_functions[order]
        a = (weight - start) / prediction_horizon
        weight = a*t + start
        return weight * (1 / self.get_upper_limit(order)) ** 2

    def __str__(self):
        return self.name
