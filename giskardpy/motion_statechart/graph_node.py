from __future__ import annotations
from typing import Optional

import giskardpy.casadi_wrapper as cas
from giskardpy.data_types.data_types import LifeCycleState
from giskardpy.data_types.exceptions import GiskardException
from giskardpy.god_map import god_map
from giskardpy.utils.utils import string_shortener


class MotionStatechartNode:
    _unparsed_start_condition: Optional[str]
    _unparsed_pause_condition: Optional[str]
    _unparsed_end_condition: Optional[str]
    _unparsed_reset_condition: Optional[str]

    _expression: cas.Expression
    _name: str
    _id: int
    plot: bool
    _parsed: bool

    logic3_start_condition: cas.Expression
    logic3_pause_condition: cas.Expression
    logic3_end_condition: cas.Expression
    logic3_reset_condition: cas.Expression

    def __init__(self, *,
                 name: str,
                 plot: bool = True):
        self._expression = cas.TrinaryUnknown
        self.plot = plot
        self._id = -1
        self._parsed = False
        self._name = name
        self._unparsed_start_condition = None
        self._unparsed_pause_condition = None
        self._unparsed_end_condition = None
        self._unparsed_reset_condition = None

    def set_unparsed_conditions(self,
                                start_condition: Optional[str] = None,
                                pause_condition: Optional[str] = None,
                                end_condition: Optional[str] = None,
                                reset_condition: Optional[str] = None):
        if start_condition is not None:
            self._unparsed_start_condition = start_condition
        if pause_condition is not None:
            self._unparsed_pause_condition = pause_condition
        if end_condition is not None:
            self._unparsed_end_condition = end_condition
        if reset_condition is not None:
            self._unparsed_reset_condition = reset_condition

    def set_conditions(self,
                       start_condition: cas.Expression,
                       pause_condition: cas.Expression,
                       end_condition: cas.Expression,
                       reset_condition: cas.Expression):
        self.logic3_start_condition = start_condition
        self.logic3_pause_condition = pause_condition
        self.logic3_end_condition = end_condition
        self.logic3_reset_condition = reset_condition

    @property
    def name(self) -> str:
        return self._name

    def __str__(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: MotionStatechartNode) -> bool:
        return self.name == other.name

    def formatted_name(self, quoted: bool = False) -> str:
        formatted_name = string_shortener(original_str=str(self.name),
                                          max_lines=4,
                                          max_line_length=25)
        result = (f'{formatted_name}\n'
                  f'----start_condition----\n'
                  f'{god_map.motion_statechart_manager.format_condition(self.start_condition)}\n'
                  f'----pause_condition----\n'
                  f'{god_map.motion_statechart_manager.format_condition(self.pause_condition)}\n'
                  f'----end_condition----\n'
                  f'{god_map.motion_statechart_manager.format_condition(self.end_condition)}')
        if quoted:
            return '"' + result + '"'
        return result

    def update_expression_on_starting(self, expression: cas.PreservedCasType) -> cas.PreservedCasType:
        if len(expression.free_symbols()) == 0:
            return expression
        return god_map.motion_statechart_manager.register_expression_updater(expression, self)

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
    def start_condition(self) -> str:
        return self._unparsed_start_condition

    @start_condition.setter
    def start_condition(self, value: str) -> None:
        self._unparsed_start_condition = value

    @property
    def pause_condition(self) -> str:
        return self._unparsed_pause_condition

    @pause_condition.setter
    def pause_condition(self, value: str) -> None:
        self._unparsed_pause_condition = value

    @property
    def end_condition(self) -> str:
        return self._unparsed_end_condition

    @end_condition.setter
    def end_condition(self, value: str) -> None:
        self._unparsed_end_condition = value

    @property
    def reset_condition(self) -> str:
        return self._unparsed_reset_condition

    @reset_condition.setter
    def reset_condition(self, value: str) -> None:
        self._unparsed_reset_condition = value

    def pre_compile(self) -> None:
        """
        Use this if you need to do stuff, after the qp controller has been initialized.
        I only needed this once, so you probably don't either.
        """
        pass
