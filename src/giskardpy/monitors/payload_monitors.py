import abc
from threading import Lock
from time import sleep
from typing import Optional, Dict, Tuple

import numpy as np

from giskardpy.data_types.data_types import PrefixName
from giskardpy.data_types.exceptions import MonitorInitalizationException, MaxTrajectoryLengthException
from giskardpy.monitors.monitors import PayloadMonitor, CancelMotion
from giskardpy.god_map import god_map
from giskardpy.middleware import logging
import giskardpy.casadi_wrapper as cas


class WorldUpdatePayloadMonitor(PayloadMonitor):
    world_lock = Lock()

    def __init__(self, *,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol):
        super().__init__(name=name, start_condition=start_condition, run_call_in_thread=True)

    @abc.abstractmethod
    def apply_world_update(self):
        pass

    def __call__(self):
        with WorldUpdatePayloadMonitor.world_lock:
            self.apply_world_update()
        self.state = True


class SetMaxTrajectoryLength(CancelMotion):
    length: float

    def __init__(self,
                 length: float,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol, ):
        if not (start_condition == cas.TrueSymbol).to_np():
            raise MonitorInitalizationException(f'Cannot set start_condition for {SetMaxTrajectoryLength.__name__}')
        self.length = length
        error_message = f'Trajectory longer than {self.length}'
        super().__init__(name=name,
                         start_condition=start_condition,
                         error_message=error_message,
                         error_type=MaxTrajectoryLengthException)

    @profile
    def __call__(self):
        if god_map.time > self.length:
            return super().__call__()


class Print(PayloadMonitor):
    def __init__(self,
                 message: str,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol):
        self.message = message
        super().__init__(name=name, start_condition=start_condition, run_call_in_thread=False)

    def __call__(self):
        logging.loginfo(self.message)
        self.state = True


class Sleep(PayloadMonitor):
    def __init__(self,
                 seconds: float,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol):
        self.seconds = seconds
        super().__init__(name=name, start_condition=start_condition, run_call_in_thread=True)

    def __call__(self):
        sleep(self.seconds)
        self.state = True


class CollisionMatrixUpdater(PayloadMonitor):
    collision_matrix: Dict[Tuple[str, str], float]

    def __init__(self,
                 new_collision_matrix: Dict[Tuple[str, str], float],
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol):
        super().__init__(name=name, start_condition=start_condition, run_call_in_thread=False)
        self.collision_matrix = new_collision_matrix

    @profile
    def __call__(self):
        god_map.collision_scene.set_collision_matrix(self.collision_matrix)
        god_map.collision_scene.reset_cache()
        self.state = True


class PayloadAlternator(PayloadMonitor):

    def __init__(self,
                 mod: int = 2,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol):
        super().__init__(name=name, stay_true=False, start_condition=start_condition, run_call_in_thread=False)
        self.mod = mod

    def __call__(self):
        self.state = np.floor(god_map.time) % self.mod == 0
