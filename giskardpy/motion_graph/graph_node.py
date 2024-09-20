import giskardpy.casadi_wrapper as cas
from giskardpy.data_types.exceptions import GiskardException
from giskardpy.god_map import god_map
from giskardpy.utils.utils import string_shortener


class MotionGraphNode:
    _start_condition: cas.Expression
    _pause_condition: cas.Expression
    _end_condition: cas.Expression
    _name: str
    _id: int
    plot: bool

    def __init__(self, name: str,
                 start_condition: cas.Expression,
                 pause_condition: cas.Expression,
                 end_condition: cas.Expression,
                 plot: bool = True):
        self._start_condition = start_condition
        self._pause_condition = pause_condition
        self._end_condition = end_condition
        self.plot = plot
        self._id = -1
        self._name = name

    def formatted_name(self, quoted: bool = False) -> str:
        formatted_name = string_shortener(original_str=str(self.name),
                                          max_lines=4,
                                          max_line_length=25)
        result = (f'{formatted_name}\n'
                  f'----start_condition----\n'
                  f'{god_map.monitor_manager.format_condition(self.start_condition)}\n'
                  f'----pause_condition----\n'
                  f'{god_map.monitor_manager.format_condition(self.pause_condition)}\n'
                  f'----end_condition----\n'
                  f'{god_map.monitor_manager.format_condition(self.end_condition)}')
        if quoted:
            return '"' + result + '"'
        return result

    @property
    def id(self) -> int:
        assert self._id >= 0, f'id of {self._name} is not set.'
        return self._id

    @id.setter
    def id(self, new_id: int) -> None:
        self._id = new_id

    def get_life_cycle_state_expression(self) -> cas.Symbol:
        raise NotImplementedError('get_state_expression is not implemented')

    @property
    def start_condition(self) -> cas.Expression:
        return self._start_condition

    @start_condition.setter
    def start_condition(self, value: cas.Expression) -> None:
        for monitor_state_expr in value.free_symbols():
            if not god_map.monitor_manager.is_monitor_registered(monitor_state_expr):
                raise GiskardException(f'No monitor found for this state expr: "{monitor_state_expr}".')
        self._start_condition = value

    @property
    def pause_condition(self) -> cas.Expression:
        return self._pause_condition

    @pause_condition.setter
    def pause_condition(self, value: cas.Expression) -> None:
        for monitor_state_expr in value.free_symbols():
            if not god_map.monitor_manager.is_monitor_registered(monitor_state_expr):
                raise GiskardException(f'No monitor found for this state expr: "{monitor_state_expr}".')
        self._pause_condition = value

    @property
    def end_condition(self) -> cas.Expression:
        return self._end_condition

    @end_condition.setter
    def end_condition(self, value: cas.Expression) -> None:
        for monitor_state_expr in value.free_symbols():
            if not god_map.monitor_manager.is_monitor_registered(monitor_state_expr):
                raise GiskardException(f'No monitor found for this state expr: "{monitor_state_expr}".')
        self._end_condition = value
