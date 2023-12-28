import abc
from abc import ABC
from typing import Union, List, TypeVar, Optional, Dict, Tuple

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
from giskardpy.utils import logging


class PayloadMonitor(Monitor, ABC):
    id: int
    name: str
    state: bool

    def __init__(self, name: str, start_monitors: List[Monitor]):
        self.id = -1
        self.name = name
        self.state_flip_times = []
        self.state = False
        super().__init__(name=name, start_monitors=start_monitors)

    def get_state(self) -> bool:
        return self.state

    def set_id(self, id_: int):
        self.id = id_

    @abc.abstractmethod
    def __call__(self):
        pass


class EndMotion(PayloadMonitor):
    def __init__(self, name: str, start_monitors: List[Monitor]):
        super().__init__(name, start_monitors)

    def __call__(self):
        self.state = True

    def get_state(self) -> bool:
        return self.state


class Print(PayloadMonitor):
    def __init__(self, name: str, start_monitors: List[Monitor], message: str):
        self.message = message
        super().__init__(name, start_monitors)

    def __call__(self):
        logging.loginfo(self.message)
        self.state = True


class CollisionMatrixUpdater(PayloadMonitor):
    collision_matrix: Dict[Tuple[str, str], float]

    def __init__(self, name: str, start_monitors: List[Monitor], new_collision_matrix: Dict[Tuple[str, str], float]):
        super().__init__(name, start_monitors)
        self.collision_matrix = new_collision_matrix

    def __call__(self):
        god_map.collision_scene.set_collision_matrix(self.collision_matrix)
        god_map.collision_scene.reset_cache()
        self.state = True
