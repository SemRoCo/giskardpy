from typing import Union, List

import numpy as np

import giskardpy.casadi_wrapper as cas
from giskardpy.exceptions import UnknownGroupException
from giskardpy.god_map import god_map
from giskardpy.my_types import Derivatives, my_string, transformable_message
from giskardpy.qp.free_variable import FreeVariable
from giskardpy.symbol_manager import symbol_manager
import giskardpy.utils.tfwrapper as tf

class Monitor:
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

    def transform_msg(self, target_frame: my_string, msg: transformable_message, tf_timeout: float = 1) \
            -> transformable_message:
        """
        First tries to transform the message using the worlds internal kinematic tree.
        If it fails, it uses tf as a backup.
        :param target_frame:
        :param msg:
        :param tf_timeout: for how long Giskard should wait for tf.
        :return: message relative to target frame
        """
        try:
            try:
                msg.header.frame_id = god_map.world.search_for_link_name(msg.header.frame_id)
            except UnknownGroupException:
                pass
            return god_map.world.transform_msg(target_frame, msg)
        except KeyError:
            return tf.transform_msg(target_frame, msg, timeout=tf_timeout)

    @profile
    def substitute_with_on_flip_symbols(self, expression: cas.Expression) -> cas.Expression:
        old_symbols = []
        new_symbols = []
        for i, symbol in enumerate(expression.free_symbols()):
            substitution_key = str(symbol)
            self.substitution_keys.append(substitution_key)
            old_symbols.append(symbol)
            new_symbols.append(self.get_substitution_key(i))
        new_expression = cas.substitute(expression, old_symbols, new_symbols)
        self.update_substitution_values()
        return new_expression

    def get_expression(self):
        return self.expression

    @profile
    def update_substitution_values(self):
        self.substitution_values = symbol_manager.resolve_symbols(self.substitution_keys)

    @profile
    def get_substitution_key(self, substitution_id: int) -> cas.Symbol:
        return symbol_manager.get_symbol(f'god_map.monitor_manager.monitors[{self.id}].substitution_values[{substitution_id}]')

    def get_state_expression(self):
        return symbol_manager.get_symbol(f'god_map.monitor_manager.state[{self.id}]')


class LocalMinimumReached(Monitor):
    def __init__(self, name: str = 'local minimum reached', min_cut_off: float = 0.01, max_cut_off: float = 0.06,
                 joint_convergence_threshold: float = 0.01):
        super().__init__(name=name, crucial=True, stay_one=False)
        condition_list = []
        traj_length_in_sec = symbol_manager.time * god_map.qp_controller_config.sample_period
        condition_list.append(cas.greater(traj_length_in_sec, 1))
        for free_variable_name, free_variable in god_map.world.free_variables.items():
            velocity_limit = symbol_manager.evaluate_expr(free_variable.get_upper_limit(Derivatives.velocity))
            joint_vel_symbol = free_variable.get_symbol(Derivatives.velocity)
            velocity_limit *= joint_convergence_threshold
            velocity_limit = min(max(min_cut_off, velocity_limit), max_cut_off)
            condition_list.append(cas.less(cas.abs(joint_vel_symbol), velocity_limit))
        self.expression = cas.logic_all(cas.Expression(condition_list))