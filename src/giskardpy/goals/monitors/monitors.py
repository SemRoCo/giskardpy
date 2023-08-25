from typing import Union

import giskardpy.casadi_wrapper as cas
from giskardpy import identifier
from giskardpy.god_map_user import GodMapWorshipper


class Monitor(GodMapWorshipper):
    id: int

    def __init__(self, expression: cas.symbol_expr, crucial: bool, stay_one: bool = False):
        self.id = -1
        self.stay_one = stay_one
        self.crucial = crucial
        self.add_self_to_god_map()
        self.expression = expression

    def get_expression(self):
        return self.expression

    def add_self_to_god_map(self):
        monitors: list = self.god_map.get_data(identifier.monitors, [])
        monitors.append(self)
        self.id = len(monitors) - 1

    def get_state_expression(self):
        return self.god_map.to_symbol(identifier.monitor_manager + ['state', self.id])

    def _get_identifier(self):
        try:
            return identifier.monitors + [self.id]
        except AttributeError as e:
            raise AttributeError(
                f'You have to ensure that str(self) is possible before calling parents __init__: {e}')

    def get_parameter_as_symbolic_expression(self, name: str) -> Union[Union[cas.Symbol, float], cas.Expression]:
        """
        :param name: name of a class attribute, e.g. self.muh
        :return: a symbol (or matrix of symbols) that refers to self.muh
        """
        if not hasattr(self, name):
            raise AttributeError(f'{self.__class__.__name__} doesn\'t have attribute {name}')
        return self.god_map.to_expr(self._get_identifier() + [name])


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
