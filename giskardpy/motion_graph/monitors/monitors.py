from __future__ import annotations

import abc
from abc import ABC
from functools import cached_property
from typing import Optional
from line_profiler import profile

import numpy as np

import giskardpy.casadi_wrapper as cas
from giskardpy.data_types.data_types import Derivatives, PrefixName
from giskardpy.data_types.exceptions import GiskardException, MonitorInitalizationException
from giskardpy.god_map import god_map
from giskardpy.motion_graph.graph_node import MotionGraphNode
from giskardpy.symbol_manager import symbol_manager


class Monitor(MotionGraphNode):

    def __init__(self, *,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol,
                 plot: bool = True):
        """
        Every class inheriting from this can be called via the ROS interface.
        :param name: name of the monitor
        :param start_condition: A logical casadi expression using monitor variables, "not", "and" and "or". This monitor
                                 will only get executed once this condition becomes True.
        :param plot: If true, this monitor will not be plotted in the gantt chart and task graph.
        """
        self.name = PrefixName(name or self.__class__.__name__)
        super().__init__(name=name,
                         start_condition=start_condition, hold_condition=hold_condition, end_condition=end_condition,
                         plot=plot)
        self.start_condition = start_condition

    @cached_property
    def state_filter(self) -> np.ndarray:
        return god_map.monitor_manager.to_state_filter(self.start_condition)

    def get_state_expression(self):
        return symbol_manager.get_symbol(f'god_map.monitor_manager.state[{self.id}]')

    def get_life_cycle_state_expression(self):
        return symbol_manager.get_symbol(f'god_map.monitor_manager.life_cycle_state[{self.id}]')

    def __repr__(self) -> str:
        return str(self.name)


class PayloadMonitor(Monitor, ABC):
    state: bool
    run_call_in_thread: bool

    def __init__(self, *,
                 run_call_in_thread: bool,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
        """
        A monitor which executes its __call__ function when start_condition becomes True.
        Subclass this and implement __init__.py and __call__. The __call__ method should change self.state to True when
        it's done.
        :param run_call_in_thread: if True, calls __call__ in a separate thread. Use for expensive operations
        """
        self.state = False
        self.run_call_in_thread = run_call_in_thread
        super().__init__(name=name,
                         start_condition=start_condition,
                         hold_condition=hold_condition,
                         end_condition=end_condition)

    def get_state(self) -> bool:
        return self.state

    @abc.abstractmethod
    def __call__(self):
        pass


class EndMotion(PayloadMonitor):
    def __init__(self,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
        super().__init__(name=name,
                         start_condition=start_condition,
                         hold_condition=hold_condition,
                         end_condition=end_condition,
                         run_call_in_thread=False)

    def __call__(self):
        self.state = True

    def get_state(self) -> bool:
        return self.state


class CancelMotion(PayloadMonitor):
    def __init__(self,
                 exception: Exception = GiskardException,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
        super().__init__(name=name,
                         start_condition=start_condition,
                         hold_condition=hold_condition,
                         end_condition=end_condition,
                         run_call_in_thread=False)
        self.exception = exception

    @profile
    def __call__(self):
        self.state = True
        raise self.exception

    def get_state(self) -> bool:
        return self.state


class ExpressionMonitor(Monitor):
    _expression: cas.Expression

    def __init__(self, *,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol,
                 plot: bool = True):
        """
        A Monitor whose state is determined by its expression.
        Override this method, create an expression and assign its expression at the end.
        """
        self.substitution_values = []
        self.substitution_keys = []
        self._expression = None
        super().__init__(name=name,
                         start_condition=start_condition,
                         hold_condition=hold_condition,
                         end_condition=end_condition,
                         plot=plot)

    @property
    def expression(self) -> cas.Expression:
        return self._expression

    @expression.setter
    def expression(self, expression: cas.Expression) -> None:
        self._expression = expression

    def compile(self) -> None:
        """
        Use this if you need to do stuff, after the qp controller has been initialized.
        I only needed this once, so you probably don't either.
        """
        pass


class LocalMinimumReached(ExpressionMonitor):
    def __init__(self,
                 name: Optional[str] = None,
                 min_cut_off: float = 0.01,
                 max_cut_off: float = 0.06,
                 joint_convergence_threshold: float = 0.01,
                 windows_size: int = 1,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
        super().__init__(name=name,
                         start_condition=start_condition,
                         hold_condition=hold_condition,
                         end_condition=end_condition)
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


class TimeAbove(ExpressionMonitor):
    def __init__(self,
                 threshold: float,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
        super().__init__(name=name,
                         start_condition=start_condition,
                         hold_condition=hold_condition,
                         end_condition=end_condition)
        if threshold is None:
            threshold = god_map.qp_controller_config.max_trajectory_length
        traj_length_in_sec = symbol_manager.time
        condition = cas.greater(traj_length_in_sec, threshold)
        self.expression = condition


class Alternator(ExpressionMonitor):

    def __init__(self,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol,
                 mod: int = 2,
                 plot: bool = True):
        super().__init__(name=name,
                         start_condition=start_condition,
                         hold_condition=hold_condition,
                         end_condition=end_condition,
                         plot=plot)
        time = symbol_manager.time
        expr = cas.equal(cas.fmod(cas.floor(time), mod), 0)
        self.expression = expr
