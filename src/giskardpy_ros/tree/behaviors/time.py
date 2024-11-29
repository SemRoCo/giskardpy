from typing import Optional

import rospy
from line_profiler import profile
from py_trees import Status

from giskardpy.god_map import god_map
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior


class TimePlugin(GiskardBehavior):
    def __init__(self, name: Optional[str] = 'time'):
        super().__init__(name)

    @profile
    def update(self):
        god_map.time += god_map.qp_controller.mpc_dt
        return Status.SUCCESS


class ControlCycleCounter(GiskardBehavior):

    @profile
    def __init__(self, name: Optional[str] = 'control cycle counter'):
        super().__init__(name)

    @profile
    def update(self):
        god_map.control_cycle_counter += 1
        return Status.SUCCESS


class RosTime(GiskardBehavior):
    def __init__(self, name: Optional[str] = 'ros time'):
        super().__init__(name)

    @property
    def start_time(self) -> float:
        return god_map.motion_start_time

    @profile
    def update(self):
        god_map.time = rospy.get_rostime().to_sec() - self.start_time
        return Status.SUCCESS