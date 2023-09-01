from typing import Union, List

import giskardpy.casadi_wrapper as cas
from giskardpy import identifier
from giskardpy.god_map_user import GodMapWorshipper


class Monitor(GodMapWorshipper):
    id: int
    expression: cas.Expression
    state_flip_times: List[float]
    name: str

    def __init__(self, name: str, *, crucial: bool, stay_one: bool = False):
        self.id = -1
        self.name = name
        self.stay_one = stay_one
        self.crucial = crucial
        self.substitution_values = []
        self.substitution_keys = []
        self.expression = None
        self.state_flip_times = []

    def set_id(self, id_: int):
        self.id = id_

    def set_expression(self, expression: cas.symbol_expr):
        self.expression = expression

    def notify_flipped(self, time: float):
        self.state_flip_times.append(time)

    @profile
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

    @profile
    def update_substitution_values(self):
        self.substitution_values = self.god_map.get_values(self.substitution_keys)

    @profile
    def get_substitution_key(self, substitution_id: int) -> cas.Symbol:
        return self.god_map.to_symbol(identifier.monitors + [self.id, 'substitution_values', substitution_id])

    def get_state_expression(self):
        return self.god_map.to_symbol(identifier.monitor_manager + ['state', self.id])
