import abc
from abc import ABC
from threading import Lock
from typing import Union, List, TypeVar, Optional, Dict, Tuple

import numpy as np
import rospy

import giskardpy.casadi_wrapper as cas
from giskard_msgs.msg import MoveResult
from giskardpy.casadi_wrapper import PreservedCasType
from giskardpy.exceptions import UnknownGroupException, PlanningException, GiskardException
from giskardpy.goals.monitors.monitors import ExpressionMonitor, Monitor
from giskardpy.god_map import god_map
from giskardpy.my_types import Derivatives, my_string, transformable_message
from giskardpy.qp.free_variable import FreeVariable
from giskardpy.symbol_manager import symbol_manager
import giskardpy.utils.tfwrapper as tf
from giskardpy.utils import logging
from giskardpy.utils.decorators import catch_and_raise_to_blackboard


class PayloadMonitor(Monitor, ABC):
    state: bool
    run_call_in_thread: bool

    def __init__(self, name: str, start_monitors: List[Monitor], run_call_in_thread: bool, stay_one: bool = True):
        self.state = False
        self.run_call_in_thread = run_call_in_thread
        super().__init__(name=name, start_monitors=start_monitors, stay_one=stay_one)

    def get_state(self) -> bool:
        return self.state

    def set_id(self, id_: int):
        self.id = id_

    @abc.abstractmethod
    def __call__(self):
        pass


class WorldUpdatePayloadMonitor(PayloadMonitor):
    world_lock = Lock()

    def __init__(self, name: str, start_monitors: List[Monitor]):
        super().__init__(name=name, start_monitors=start_monitors, run_call_in_thread=True)

    @abc.abstractmethod
    def apply_world_update(self):
        pass

    def __call__(self):
        with WorldUpdatePayloadMonitor.world_lock:
            self.apply_world_update()
        self.state = True


class EndMotion(PayloadMonitor):
    def __init__(self, name: str, start_monitors: List[Monitor]):
        super().__init__(name, start_monitors, run_call_in_thread=False)

    def __call__(self):
        self.state = True

    def get_state(self) -> bool:
        return self.state


class CancelMotion(PayloadMonitor):
    def __init__(self,
                 name: str,
                 start_monitors: List[Monitor],
                 error_message: str,
                 error_code: int = MoveResult.ERROR):
        super().__init__(name, start_monitors, run_call_in_thread=False)
        self.error_message = error_message
        self.error_code = error_code

    @profile
    def __call__(self):
        self.state = True
        raise GiskardException.from_error_code(error_code=self.error_code, error_message=self.error_message)

    def get_state(self) -> bool:
        return self.state


class SetMaxTrajectoryLength(CancelMotion):
    new_length: float

    def __init__(self, name: str, new_length: Optional[float] = None, start_monitors: Optional[List[Monitor]] = None):
        if start_monitors:
            raise GiskardException(f'Cannot set start_monitors for {SetMaxTrajectoryLength.__name__}')
        if new_length is None:
            self.new_length = god_map.qp_controller_config.max_trajectory_length
        else:
            self.new_length = new_length
        error_message = f'Trajectory longer than {self.new_length}'
        super().__init__(name, start_monitors=[], error_message=error_message, error_code=MoveResult.CONTROL_ERROR)

    @profile
    def __call__(self):
        if god_map.time > self.new_length:
            return super().__call__()


class Print(PayloadMonitor):
    def __init__(self, name: str, start_monitors: List[Monitor], message: str):
        self.message = message
        super().__init__(name, start_monitors, run_call_in_thread=False)

    def __call__(self):
        logging.loginfo(self.message)
        self.state = True


class Sleep(PayloadMonitor):
    def __init__(self, name: str, start_monitors: List[Monitor], seconds: float):
        self.seconds = seconds
        super().__init__(name, start_monitors, run_call_in_thread=True)

    def __call__(self):
        rospy.sleep(self.seconds)
        self.state = True


class UpdateParentLinkOfGroup(WorldUpdatePayloadMonitor):
    def __init__(self,
                 name: str,
                 start_monitors: List[Monitor],
                 group_name: str,
                 parent_link: str,
                 parent_link_group: Optional[str] = ''):
        self.group_name = group_name
        self.new_parent_link = god_map.world.search_for_link_name(parent_link, parent_link_group)
        super().__init__(name, start_monitors)

    def apply_world_update(self):
        god_map.world.move_group(group_name=self.group_name,
                                 new_parent_link_name=self.new_parent_link)
        rospy.sleep(2)


class CollisionMatrixUpdater(PayloadMonitor):
    collision_matrix: Dict[Tuple[str, str], float]

    def __init__(self, name: str, start_monitors: List[Monitor], new_collision_matrix: Dict[Tuple[str, str], float]):
        super().__init__(name, start_monitors, run_call_in_thread=False)
        self.collision_matrix = new_collision_matrix

    @profile
    def __call__(self):
        god_map.collision_scene.set_collision_matrix(self.collision_matrix)
        god_map.collision_scene.reset_cache()
        self.state = True


class PayloadAlternator(PayloadMonitor):

    def __init__(self, name: str, start_monitors: Optional[List[Monitor]] = None, mod: int = 2):
        super().__init__(name, stay_one=False, start_monitors=start_monitors, run_call_in_thread=False)
        self.mod = mod

    def __call__(self):
        self.state = np.floor(god_map.time) % self.mod == 0


