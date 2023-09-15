from typing import Union, List

import numpy as np

import giskardpy.casadi_wrapper as cas
from giskardpy import identifier
from giskardpy.god_map_user import GodMapWorshipper
from giskardpy.my_types import Derivatives
from giskardpy.qp.free_variable import FreeVariable


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


class LocalMinimumReached(Monitor):
    def __init__(self, name: str = 'local minimum reached', min_cut_off: float = 0.01, max_cut_off: float = 0.06,
                 joint_convergence_threshold: float = 0.01):
        super().__init__(name=name, crucial=True, stay_one=False)
        condition_list = []
        traj_length_in_sec = self.god_map.to_symbol(identifier.time) * self.sample_period
        condition_list.append(cas.greater(traj_length_in_sec, 1))
        for free_variable_name, free_variable in self.world.free_variables.items():
            velocity_limit = self.god_map.evaluate_expr(free_variable.get_upper_limit(Derivatives.velocity))
            joint_vel_symbol = free_variable.get_symbol(Derivatives.velocity)
            velocity_limit *= joint_convergence_threshold
            velocity_limit = min(max(min_cut_off, velocity_limit), max_cut_off)
            condition_list.append(cas.less(cas.abs(joint_vel_symbol), velocity_limit))
        self.expression = cas.logic_all(cas.Expression(condition_list))
