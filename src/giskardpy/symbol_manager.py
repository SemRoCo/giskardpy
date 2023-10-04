from typing import Dict, Callable

from giskardpy.god_map_interpreter import god_map
import numpy as np
import giskardpy.casadi_wrapper as cas
from giskardpy.utils.singleton import SingletonMeta


class SymbolManager(metaclass=SingletonMeta):
    symbol_str_to_lambda: Dict[str, Callable[[], float]]
    symbol_str_to_symbol: Dict[str, cas.Symbol]

    def __init__(self):
        self.symbol_str_to_lambda = {}
        self.symbol_str_to_symbol = {}
        self.last_hash = -1

    def get_symbol(self, symbol_reference):
        # TODO check for start with god_map?
        if symbol_reference not in self.symbol_str_to_lambda:
            lambda_expr = eval(f'lambda: {symbol_reference}')
            self.symbol_str_to_lambda[symbol_reference] = lambda_expr
            self.symbol_str_to_symbol[symbol_reference] = cas.Symbol(symbol_reference)
        return self.symbol_str_to_symbol[symbol_reference]

    def resolve_symbols(self, symbols):
        return np.array([self.symbol_str_to_lambda[s]() for s in symbols])

    def compile_resolve_symbols(self, symbols):
        self.c = eval('lambda: np.array([' + ', '.join(symbols) + '])')

    def resolve_symbols2(self):
        return self.c()

    def resolve_symbols3(self, symbols):
        h = hash(tuple(symbols))
        if h != self.last_hash:
            self.compile_resolve_symbols(symbols)
            self.last_hash = h
        return self.resolve_symbols2()

    @property
    def time(self):
        return self.get_symbol('god_map.time')


symbol_manager = SymbolManager()
