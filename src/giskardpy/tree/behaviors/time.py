from typing import Optional

import rospy
from py_trees import Status

from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior


class TimePlugin(GiskardBehavior):
    def __init__(self, name: Optional[str] = 'time'):
        super().__init__(name)

    @profile
    def update(self):
        god_map.time += god_map.qp_controller_config.sample_period
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
    def start_time(self):
        return god_map.tracking_start_time

    @profile
    def update(self):
        god_map.time = (rospy.get_rostime() - self.start_time).to_sec()
        return Status.SUCCESS