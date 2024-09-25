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

    def __init__(self, *,
                 name: Optional[str] = None,
                 plot: bool = True):
        self._name = name or self.__class__.__name__
        self._expression = cas.FalseSymbol
        self.plot = plot
        self._id = -1
        self._name = name
        self._start_condition = cas.TrueSymbol
        self._reset_condition = cas.FalseSymbol
        self._pause_condition = cas.FalseSymbol
        self._end_condition = cas.FalseSymbol

    def set_conditions(self,
                       start_condition: cas.Expression,
                       reset_condition: cas.Expression,
                       pause_condition: cas.Expression,
                       end_condition: cas.Expression):
        self._start_condition = start_condition
        self._reset_condition = reset_condition
        self._pause_condition = pause_condition
        self._end_condition = end_condition

    @property
    def name(self) -> str:
        return self._name

    def __str__(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return str(self)

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

    def update_expression_on_enter_running(self, expression: cas.PreservedCasType) -> cas.PreservedCasType:
        condition = cas.equal(self.get_life_cycle_state_expression(), LifeCycleState.running)
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
        # for monitor_state_expr in value.free_symbols():
        #     if not god_map.motion_graph_manager.is_node_registered(monitor_state_expr):
        #         raise GiskardException(f'No monitor found for this state expr: "{monitor_state_expr}".')
        self._start_condition = value

    @property
    def reset_condition(self) -> cas.Expression:
        return self._reset_condition

    @reset_condition.setter
    def reset_condition(self, value: cas.Expression) -> None:
        # for node_state_expr in value.free_symbols():
        #     if not god_map.motion_graph_manager.is_node_registered(node_state_expr):
        #         raise GiskardException(f'No monitor found for this state expr: "{node_state_expr}".')
        self._reset_condition = value

    @property
    def pause_condition(self) -> cas.Expression:
        return self._pause_condition

    @pause_condition.setter
    def pause_condition(self, value: cas.Expression) -> None:
        # for monitor_state_expr in value.free_symbols():
        #     if not god_map.motion_graph_manager.is_node_registered(monitor_state_expr):
        #         raise GiskardException(f'No monitor found for this state expr: "{monitor_state_expr}".')
        self._pause_condition = value

    @property
    def end_condition(self) -> cas.Expression:
        return self._end_condition

    @end_condition.setter
    def end_condition(self, value: cas.Expression) -> None:
        # for monitor_state_expr in value.free_symbols():
        #     if not god_map.motion_graph_manager.is_node_registered(monitor_state_expr):
        #         raise GiskardException(f'No monitor found for this state expr: "{monitor_state_expr}".')
        self._end_condition = value

    def pre_compile(self) -> None:
        """
        Use this if you need to do stuff, after the qp controller has been initialized.
        I only needed this once, so you probably don't either.
        """
        pass
