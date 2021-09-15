from collections import defaultdict


class FreeVariable(object):
    def __init__(self, symbols, lower_limits, upper_limits, quadratic_weights, horizon_functions=None):
        """
        :type symbols:  dict
        :type lower_limits: dict
        :type upper_limits: dict
        :type quadratic_weights: dict
        :type horizon_functions: dict
        """
        self._symbols = symbols
        self.name = str(self._symbols[0])
        self.lower_limits = lower_limits
        self.upper_limits = upper_limits
        self.quadratic_weights = quadratic_weights
        assert max(self._symbols.keys()) == len(self._symbols) - 1
        assert len(self._symbols) == len(self.quadratic_weights) + 1
        self.order = len(self._symbols)

        self.horizon_functions = defaultdict(float)
        if horizon_functions is not None:
            self.horizon_functions.update(horizon_functions)

    def get_symbol(self, order):
        try:
            return self._symbols[order]
        except KeyError:
            raise KeyError(u'Free variable {} doesn\'t have symbol for derivative of order {}'.format(self, order))

    def get_lower_limit(self, order):
        try:
            return self.lower_limits[order]
        except KeyError:
            raise KeyError(u'Free variable {} doesn\'t have lower limit for derivative of order {}'.format(self, order))

    def get_upper_limit(self, order):
        try:
            return self.upper_limits[order]
        except KeyError:
            raise KeyError(u'Free variable {} doesn\'t have upper limit for derivative of order {}'.format(self, order))

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
