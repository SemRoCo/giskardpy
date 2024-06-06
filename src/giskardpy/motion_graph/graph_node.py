import giskardpy.casadi_wrapper as cas
from giskardpy.data_types import PrefixName


class MotionGraphNode:
    _start_condition: cas.Expression
    _hold_condition: cas.Expression
    _end_condition: cas.Expression
    _name: str
    _id: int
    plot: bool

    def __init__(self, name: str, start_condition: cas.Expression, hold_condition: cas.Expression, end_condition: cas.Expression,):
        self._start_condition = start_condition
        self._hold_condition = hold_condition
        self._end_condition = end_condition
        self.plot = True
        self._id = -1
        self._name = name

    @property
    def id(self) -> int:
        assert self._id >= 0, f'id of {self._name} is not set.'
        return self._id

    def set_id(self, new_id: int) -> None:
        self._id = new_id
