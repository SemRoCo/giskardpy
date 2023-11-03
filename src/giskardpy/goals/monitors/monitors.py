from typing import Union, List, TypeVar

import numpy as np

import giskardpy.casadi_wrapper as cas
from giskardpy.casadi_wrapper import PreservedCasType
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

    def get_expression(self):
        return self.expression

    def get_state_expression(self):
        return symbol_manager.get_symbol(f'god_map.monitor_manager.state[{self.id}]')

    def compile(self):
        # use this if you need to do stuff, after the qp controller has been initialized
        pass


class LocalMinimumReached(Monitor):
    def __init__(self, name: str = 'local minimum reached', min_cut_off: float = 0.01, max_cut_off: float = 0.06,
                 joint_convergence_threshold: float = 0.01, windows_size: int = 1):
        super().__init__(name=name, crucial=True, stay_one=False)
        self.joint_convergence_threshold = joint_convergence_threshold
        self.min_cut_off = min_cut_off
        self.max_cut_off = max_cut_off
        self.windows_size = windows_size

    def compile(self):
        condition_list = []
        traj_length_in_sec = symbol_manager.time
        condition_list.append(cas.greater(traj_length_in_sec, 1))
        for free_variable in god_map.free_variables:
            free_variable_name = free_variable.name
            velocity_limit = symbol_manager.evaluate_expr(free_variable.get_upper_limit(Derivatives.velocity))
            velocity_limit *= self.joint_convergence_threshold
            velocity_limit = min(max(self.min_cut_off, velocity_limit), self.max_cut_off)
            for t in range(self.windows_size):
                if t == 0:
                    joint_vel_symbol = free_variable.get_symbol(Derivatives.velocity)
                else:
                    expr = f'god_map.trajectory.get_exact({-t})[\'{free_variable_name}\'].velocity'
                    joint_vel_symbol = symbol_manager.get_symbol(expr)
                condition_list.append(cas.less(cas.abs(joint_vel_symbol), velocity_limit))

        self.expression = cas.logic_all(cas.Expression(condition_list))


class TimeAbove(Monitor):
    def __init__(self, *, threshold: float, name: str = 'time above'):
        super().__init__(name=name, crucial=True, stay_one=False)
        traj_length_in_sec = symbol_manager.time
        condition = cas.greater(traj_length_in_sec, threshold)
        god_map.debug_expression_manager.add_debug_expression('time', traj_length_in_sec)
        self.set_expression(condition)
