import abc
from threading import Lock
from typing import Optional, Dict, Tuple

import numpy as np

import giskardpy.casadi_wrapper as cas
from giskardpy.data_types.exceptions import MaxTrajectoryLengthException
from giskardpy.data_types.exceptions import MonitorInitalizationException
from giskardpy.god_map import god_map
from giskardpy.middleware import get_middleware
from giskardpy.motion_graph.monitors.monitors import PayloadMonitor, CancelMotion
from line_profiler import profile


class SetMaxTrajectoryLength(CancelMotion):
    length: float

    def __init__(self,
                 length: float,
                 name: Optional[str] = None):
        # if not (start_condition == cas.TrueSymbol).to_np():
        #     raise MonitorInitalizationException(f'Cannot set start_condition for {SetMaxTrajectoryLength.__name__}')
        self.length = length
        error_message = f'Trajectory longer than {self.length}'
        super().__init__(name=name, exception=MaxTrajectoryLengthException(error_message))

    @profile
    def __call__(self):
        if god_map.time > self.length:
            return super().__call__()


class Print(PayloadMonitor):
    def __init__(self,
                 message: str,
                 name: Optional[str] = None):
        self.message = message
        super().__init__(name=name, run_call_in_thread=False)

    def __call__(self):
        get_middleware().loginfo(self.message)
        self.state = True


class Sleep(PayloadMonitor):
    start_time: float

    def __init__(self,
                 seconds: float,
                 name: Optional[str] = None):
        self.seconds = seconds
        super().__init__(name=name,
                         run_call_in_thread=False)
        self.start_time = None

    def __call__(self):
        if self.start_time is None:
            self.start_time = god_map.time
        self.state = god_map.time - self.start_time >= self.seconds


class CollisionMatrixUpdater(PayloadMonitor):
    collision_matrix: Dict[Tuple[str, str], float]

    def __init__(self,
                 new_collision_matrix: Dict[Tuple[str, str], float],
                 name: Optional[str] = None):
        super().__init__(name=name,
                         run_call_in_thread=False)
        self.collision_matrix = new_collision_matrix

    @profile
    def __call__(self):
        god_map.collision_scene.set_collision_matrix(self.collision_matrix)
        god_map.collision_scene.reset_cache()
        self.state = True


class PayloadAlternator(PayloadMonitor):

    def __init__(self,
                 mod: int = 2,
                 name: Optional[str] = None):
        super().__init__(name=name,
                         run_call_in_thread=False)
        self.mod = mod

    def __call__(self):
        self.state = np.floor(god_map.time) % self.mod == 0


class Counter(PayloadMonitor):
    def __init__(self, name: str, number: int):
        super().__init__(name=name, run_call_in_thread=False)
        self.counter = 0
        self.number = number

    def __call__(self):
        self.state = self.counter >= self.number
        self.counter += 1


class Pulse(PayloadMonitor):
    def __init__(self, name: str, after_ticks: int):
        super().__init__(name=name, run_call_in_thread=False)
        self.counter = 0
        self.after_ticks = after_ticks

    def __call__(self):
        self.state = self.counter == self.after_ticks
        self.counter += 1
