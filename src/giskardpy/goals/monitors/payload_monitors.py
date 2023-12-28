import abc
from abc import ABC
from typing import Union, List, TypeVar, Optional

import numpy as np

import giskardpy.casadi_wrapper as cas
from giskardpy.casadi_wrapper import PreservedCasType
from giskardpy.exceptions import UnknownGroupException
from giskardpy.goals.monitors.monitors import ExpressionMonitor, Monitor
from giskardpy.god_map import god_map
from giskardpy.my_types import Derivatives, my_string, transformable_message
from giskardpy.qp.free_variable import FreeVariable
from giskardpy.symbol_manager import symbol_manager
import giskardpy.utils.tfwrapper as tf


class PayloadMonitor(Monitor, ABC):
    id: int
    name: str

    def __init__(self, name: str, start_monitors: List[Monitor]):
        self.id = -1
        self.name = name
        self.state_flip_times = []
        super().__init__(name=name, start_monitors=start_monitors)

    @abc.abstractmethod
    def get_state(self) -> bool:
        pass

    def set_id(self, id_: int):
        self.id = id_

    def notify_flipped(self, time: float):
        self.state_flip_times.append(time)

    @abc.abstractmethod
    def __call__(self):
        pass


class EndMotion(PayloadMonitor):
    state: bool

    def __init__(self, name: str, start_monitors: List[Monitor]):
        super().__init__(name, start_monitors)
        self.state = False

    def __call__(self):
        self.state = True

    def get_state(self) -> bool:
        return self.state

