from __future__ import annotations

from functools import cached_property
from typing import List, Optional

import numpy as np

import giskardpy.casadi_wrapper as cas
from giskardpy.exceptions import GiskardException, MonitorInitalizationException
from giskardpy.god_map import god_map
from giskardpy.my_types import Derivatives
from giskardpy.symbol_manager import symbol_manager
from giskardpy.utils.utils import string_shortener


class Monitor:
    id: int
    name: str
    start_monitors: List[Monitor]
    plot: bool
    stay_true: bool

    def __init__(self, *,
                 name: Optional[str] = None,
                 start_monitors: Optional[List[Monitor]] = None,
                 plot: bool = True,
                 stay_true: bool = False):
        self.name = name or self.__class__.__name__
        self.start_monitors = start_monitors or []
        self.id = -1
        self.plot = plot
        self.stay_true = stay_true

    def set_id(self, id_: int):
        self.id = id_

    @cached_property
    def state_filter(self) -> np.ndarray:
        return god_map.monitor_manager.to_state_filter(self.start_monitors)

    def get_state_expression(self):
        if self.id == -1:
            raise MonitorInitalizationException(f'Id of {self.name} is not set.')
        return symbol_manager.get_symbol(f'god_map.monitor_manager.state[{self.id}]')

    def formatted_name(self, quoted: bool = False) -> str:
        formatted_name = string_shortener(original_str=self.name,
                                          max_lines=4,
                                          max_line_length=25)
        if quoted:
            return '"' + formatted_name + '"'
        return formatted_name

    def __repr__(self) -> str:
        return self.name


class ExpressionMonitor(Monitor):
    _expression: cas.Expression

    def __init__(self,
                 name: Optional[str] = None,
                 stay_true: bool = False,
                 start_monitors: Optional[List[Monitor]] = None,
                 plot: bool = True):
        self.substitution_values = []
        self.substitution_keys = []
        self._expression = None
        super().__init__(name=name, start_monitors=start_monitors, plot=plot, stay_true=stay_true)

    def set_expression(self, expression: cas.symbol_expr):
        self._expression = expression

    def apply_start_monitors_to_expression(self):
        for monitor in self.start_monitors:
            monitor_state = monitor.get_state_expression()
            self._expression = cas.logic_and(self._expression, monitor_state)

    def get_expression(self):
        return self._expression

    def compile(self):
        # use this if you need to do stuff, after the qp controller has been initialized
        self.apply_start_monitors_to_expression()


class LocalMinimumReached(ExpressionMonitor):
    def __init__(self,
                 name: Optional[str] = None,
                 min_cut_off: float = 0.01,
                 max_cut_off: float = 0.06,
                 joint_convergence_threshold: float = 0.01,
                 windows_size: int = 1,
                 start_monitors: Optional[List[Monitor]] = None,
                 stay_true: bool = True):
        super().__init__(name=name, stay_true=stay_true, start_monitors=start_monitors)
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

        self.set_expression(cas.logic_all(cas.Expression(condition_list)))
        self.apply_start_monitors_to_expression()


class TimeAbove(ExpressionMonitor):
    def __init__(self,
                 threshold: Optional[float] = None,
                 name: Optional[str] = None,
                 start_monitors: Optional[List[Monitor]] = None):
        super().__init__(name=name,
                         stay_true=False,
                         start_monitors=start_monitors)
        if threshold is None:
            threshold = god_map.qp_controller_config.max_trajectory_length
        traj_length_in_sec = symbol_manager.time
        condition = cas.greater(traj_length_in_sec, threshold)
        self.set_expression(condition)


class Alternator(ExpressionMonitor):

    def __init__(self,
                 name: Optional[str] = None,
                 stay_true: bool = False,
                 start_monitors: Optional[List[Monitor]] = None,
                 mod: int = 2,
                 plot: bool = True):
        super().__init__(name, stay_true=stay_true, start_monitors=start_monitors, plot=plot)
        time = symbol_manager.time
        expr = cas.equal(cas.fmod(cas.floor(time), mod), 0)
        self.set_expression(expr)
