import abc
from abc import ABC
from threading import Lock
from typing import List, Optional, Dict, Tuple

import numpy as np
import rospy

from giskard_msgs.msg import MoveResult, GiskardError
from giskardpy.exceptions import GiskardException, MonitorInitalizationException
from giskardpy.monitors.monitors import Monitor
from giskardpy.god_map import god_map
from giskardpy.utils import logging
import giskardpy.casadi_wrapper as cas


class PayloadMonitor(Monitor, ABC):
    state: bool
    run_call_in_thread: bool

    def __init__(self, *,
                 run_call_in_thread: bool,
                 name: Optional[str] = None,
                 stay_true: bool = True,
                 start_condition: cas.Expression = cas.TrueSymbol):
        """
        A monitor which executes its __call__ function when start_condition becomes True.
        Subclass this and implement __init__ and __call__. The __call__ method should change self.state to True when
        it's done.
        :param run_call_in_thread: if True, calls __call__ in a separate thread. Use for expensive operations
        """
        self.state = False
        self.run_call_in_thread = run_call_in_thread
        super().__init__(name=name, start_condition=start_condition, stay_true=stay_true)

    def get_state(self) -> bool:
        return self.state

    @abc.abstractmethod
    def __call__(self):
        pass


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


class EndMotion(PayloadMonitor):
    def __init__(self,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,):
        super().__init__(name=name, start_condition=start_condition, run_call_in_thread=False)

    def __call__(self):
        self.state = True

    def get_state(self) -> bool:
        return self.state


class CancelMotion(PayloadMonitor):
    def __init__(self,
                 error_message: str,
                 error_code: int = GiskardError.ERROR,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,):
        super().__init__(name=name, start_condition=start_condition, run_call_in_thread=False)
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

    def __init__(self,
                 new_length: Optional[float] = None,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,):
        if not (start_condition == cas.TrueSymbol).evaluate():
            raise MonitorInitalizationException(f'Cannot set start_condition for {SetMaxTrajectoryLength.__name__}')
        if new_length is None:
            self.new_length = god_map.qp_controller_config.max_trajectory_length
        else:
            self.new_length = new_length
        error_message = f'Trajectory longer than {self.new_length}'
        super().__init__(name=name,
                         start_condition=start_condition,
                         error_message=error_message,
                         error_code=GiskardError.MAX_TRAJECTORY_LENGTH)

    @profile
    def __call__(self):
        if god_map.time > self.new_length:
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
        rospy.sleep(self.seconds)
        self.state = True


class UpdateParentLinkOfGroup(WorldUpdatePayloadMonitor):
    def __init__(self,
                 group_name: str,
                 parent_link: str,
                 parent_link_group: Optional[str] = '',
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol):
        if not god_map.is_standalone():
            raise MonitorInitalizationException(f'This monitor can only be used in standalone mode.')
        self.group_name = group_name
        self.new_parent_link = god_map.world.search_for_link_name(parent_link, parent_link_group)
        super().__init__(name=name, start_condition=start_condition)

    def apply_world_update(self):
        god_map.world.move_group(group_name=self.group_name,
                                 new_parent_link_name=self.new_parent_link)
        rospy.sleep(2)


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


