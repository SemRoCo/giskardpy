from __future__ import annotations

import abc
from abc import ABC
from functools import cached_property
from typing import Optional
from line_profiler import profile

import numpy as np

import giskardpy.casadi_wrapper as cas
from giskardpy.data_types.data_types import Derivatives, PrefixName, ObservationState
from giskardpy.data_types.exceptions import GiskardException, MonitorInitalizationException
from giskardpy.god_map import god_map
from giskardpy.motion_graph.graph_node import MotionGraphNode
from giskardpy.symbol_manager import symbol_manager


class Monitor(MotionGraphNode):

    def __init__(self, *,
                 name: Optional[str] = None,
                 plot: bool = True):
        super().__init__(name=name,
                         plot=plot)

    def get_observation_state_expression(self) -> cas.Symbol:
        return symbol_manager.get_symbol(f'god_map'
                                         f'.motion_graph_manager'
                                         f'.monitor_state'
                                         f'.get_observation_state(\'{self.name}\')')

    def get_life_cycle_state_expression(self) -> cas.Symbol:
        return symbol_manager.get_symbol(f'god_map'
                                         f'.motion_graph_manager'
                                         f'.monitor_state'
                                         f'.get_life_cycle_state(\'{self.name}\')')


# class ConditionWrapper(Monitor):
#     def __init__(self,)

class PayloadMonitor(Monitor, ABC):
    state: ObservationState
    run_call_in_thread: bool

    def __init__(self, *,
                 run_call_in_thread: bool,
                 name: Optional[str] = None):
        """
        A monitor which executes its __call__ function when start_condition becomes True.
        Subclass this and implement __init__.py and __call__. The __call__ method should change self.state to True when
        it's done.
        :param run_call_in_thread: if True, calls __call__ in a separate thread. Use for expensive operations
        """
        self.state = ObservationState.unknown
        self.run_call_in_thread = run_call_in_thread
        super().__init__(name=name)

    def get_state(self) -> ObservationState:
        return self.state

    @abc.abstractmethod
    def __call__(self):
        pass


class EndMotion(PayloadMonitor):
    def __init__(self,
                 name: Optional[str] = None):
        super().__init__(name=name,
                         run_call_in_thread=False)

    def __call__(self):
        self.state = ObservationState.true

    def get_state(self) -> ObservationState:
        return self.state


class CancelMotion(PayloadMonitor):
    def __init__(self,
                 exception: Exception = GiskardException,
                 name: Optional[str] = None):
        super().__init__(name=name,
                         run_call_in_thread=False)
        self.exception = exception

    @profile
    def __call__(self):
        self.state = ObservationState.true
        raise self.exception

    def get_state(self) -> ObservationState:
        return self.state


class LocalMinimumReached(Monitor):
    def __init__(self,
                 name: Optional[str] = None,
                 min_cut_off: float = 0.01,
                 max_cut_off: float = 0.06,
                 joint_convergence_threshold: float = 0.01,
                 windows_size: int = 1):
        super().__init__(name=name)
        self.joint_convergence_threshold = joint_convergence_threshold
        self.min_cut_off = min_cut_off
        self.max_cut_off = max_cut_off
        self.windows_size = windows_size

    def pre_compile(self):
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
    def __init__(self,
                 threshold: float,
                 name: Optional[str] = None):
        super().__init__(name=name)
        if threshold is None:
            threshold = god_map.qp_controller_config.max_trajectory_length
        traj_length_in_sec = symbol_manager.time
        condition = cas.greater(traj_length_in_sec, threshold)
        self.expression = condition


class Alternator(Monitor):

    def __init__(self,
                 name: Optional[str] = None,
                 mod: int = 2,
                 plot: bool = True):
        super().__init__(name=name,
                         plot=plot)
        time = symbol_manager.time
        expr = cas.equal(cas.fmod(cas.floor(time), mod), 0)
        self.expression = expr


class TrueMonitor(Monitor):
    def __init__(self,
                 name: Optional[str] = None,
                 plot: bool = True):
        super().__init__(name=name,
                         plot=plot)
        self.expression = cas.BinaryTrue


class FalseMonitor(Monitor):
    def __init__(self,
                 name: Optional[str] = None,
                 plot: bool = True):
        super().__init__(name=name,
                         plot=plot)
        self.expression = cas.BinaryFalse
