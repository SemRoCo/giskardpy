from typing import Union

import giskardpy.casadi_wrapper as cas
from giskardpy import identifier
from giskardpy.god_map_user import GodMapWorshipper


class Monitor(GodMapWorshipper):
    id: int
    expression: cas.Expression

    def __init__(self, crucial: bool, stay_one: bool = False):
        self.id = -1
        self.stay_one = stay_one
        self.crucial = crucial
        self.add_self_to_god_map()
        self.substitution_values = []
        self.substitution_keys = []
        self.expression = None

    def set_expression(self, expression: cas.symbol_expr):
        self.expression = expression

    def substitute_with_on_flip_symbols(self, expression: cas.Expression) -> cas.Expression:
        symbol_paths = []
        old_symbols = []
        new_symbols = []
        for i, symbol in enumerate(expression.free_symbols()):
            substitution_key = str(symbol)
            self.substitution_keys.append(substitution_key)
            identifier_ = self.god_map.expr_to_key[substitution_key]
            symbol_paths.append(identifier_)
            old_symbols.append(symbol)
            new_symbols.append(self.get_substitution_key(i))
        new_expression = cas.substitute(expression, old_symbols, new_symbols)
        self.update_substitution_values()
        return new_expression

    def get_expression(self):
        return self.expression

    def update_substitution_values(self):
        self.substitution_values = self.god_map.get_values(self.substitution_keys)

    def add_self_to_god_map(self):
        monitors: list = self.god_map.get_data(identifier.monitors, [])
        monitors.append(self)
        self.id = len(monitors) - 1

    def get_substitution_key(self, substitution_id: int) -> cas.Symbol:
        return self.god_map.to_symbol(identifier.monitors + [self.id, 'substitution_values', substitution_id])

    def get_state_expression(self):
        return self.god_map.to_symbol(identifier.monitor_manager + ['state', self.id])


class AlwaysOne(Monitor):
    def __init__(self, crucial: bool):
        super().__init__(cas.Expression(1), crucial)

    def get_expression(self):
        return self.expression


class AlwaysZero(Monitor):
    def __init__(self, crucial: bool):
        super().__init__(cas.Expression(0), crucial)

    def get_expression(self):
        return self.expression
