from __future__ import annotations
from typing import Optional

import giskardpy.casadi_wrapper as cas
from giskardpy.data_types.data_types import LifeCycleState
from giskardpy.data_types.exceptions import GiskardException
from giskardpy.god_map import god_map
from giskardpy.utils.utils import string_shortener


class MotionGraphNode:
    _start_condition: cas.Expression
    _reset_condition: cas.Expression
    _pause_condition: cas.Expression
    _end_condition: cas.Expression
    _expression: cas.Expression
    _name: str
    _id: int
    plot: bool

    logic3_start_condition: cas.Expression
    logic3_pause_condition: cas.Expression
    logic3_end_condition: cas.Expression
    logic3_reset_condition: cas.Expression

    def __init__(self, *,
                 name: Optional[str] = None,
                 plot: bool = True):
        self._name = name or self.__class__.__name__
        self._expression = cas.BinaryFalse
        self.plot = plot
        self._id = -1
        self._name = name
        self._start_condition = cas.BinaryTrue
        self._pause_condition = cas.BinaryFalse
        self._end_condition = cas.BinaryFalse
        self._reset_condition = cas.BinaryFalse

    def set_conditions(self,
                       start_condition: cas.Expression,
                       pause_condition: cas.Expression,
                       end_condition: cas.Expression,
                       reset_condition: cas.Expression):
        self._start_condition = start_condition
        self._pause_condition = pause_condition
        self._end_condition = end_condition
        self._reset_condition = reset_condition

    @property
    def name(self) -> str:
        return self._name

    def __str__(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: MotionGraphNode) -> bool:
        return self.name == other.name

    def formatted_name(self, quoted: bool = False) -> str:
        formatted_name = string_shortener(original_str=str(self.name),
                                          max_lines=4,
                                          max_line_length=25)
        result = (f'{formatted_name}\n'
                  f'----start_condition----\n'
                  f'{god_map.motion_graph_manager.format_condition(self.start_condition)}\n'
                  f'----pause_condition----\n'
                  f'{god_map.motion_graph_manager.format_condition(self.pause_condition)}\n'
                  f'----end_condition----\n'
                  f'{god_map.motion_graph_manager.format_condition(self.end_condition)}')
        if quoted:
            return '"' + result + '"'
        return result

    def update_expression_on_starting(self, expression: cas.PreservedCasType) -> cas.PreservedCasType:
        # TODO this creates quite some unnecessary overhead, because there is no need to keep updating expressions,
        #  when its goal is not started
        if len(expression.free_symbols()) == 0:
            return expression
        condition = cas.equal(self.get_life_cycle_state_expression(), LifeCycleState.not_started)
        return god_map.motion_graph_manager.register_expression_updater(expression, condition)

    @property
    def expression(self) -> cas.Expression:
        return self._expression

    @expression.setter
    def expression(self, expression: cas.Expression) -> None:
        self._expression = expression

    def get_observation_state_expression(self) -> cas.Symbol:
        raise NotImplementedError('get_state_expression is not implemented')

    def get_life_cycle_state_expression(self) -> cas.Symbol:
        raise NotImplementedError('get_life_cycle_state_expression is not implemented')

    @property
    def start_condition(self) -> cas.Expression:
        return self._start_condition

    @start_condition.setter
    def start_condition(self, value: cas.Expression) -> None:
        self._start_condition = value

    @property
    def pause_condition(self) -> cas.Expression:
        return self._pause_condition

    @pause_condition.setter
    def pause_condition(self, value: cas.Expression) -> None:
        self._pause_condition = value

    @property
    def end_condition(self) -> cas.Expression:
        return self._end_condition

    @end_condition.setter
    def end_condition(self, value: cas.Expression) -> None:
        self._end_condition = value

    @property
    def reset_condition(self) -> cas.Expression:
        return self._reset_condition

    @reset_condition.setter
    def reset_condition(self, value: cas.Expression) -> None:
        self._reset_condition = value

    def pre_compile(self) -> None:
        """
        Use this if you need to do stuff, after the qp controller has been initialized.
        I only needed this once, so you probably don't either.
        """
        pass
