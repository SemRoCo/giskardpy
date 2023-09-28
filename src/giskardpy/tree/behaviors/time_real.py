import rospy
from py_trees import Status

from giskardpy import identifier
from giskardpy.god_map_interpreter import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior


class RosTime(GiskardBehavior):
    @property
    def start_time(self):
        return god_map.tracking_start_time

    @profile
    def update(self):
        with god_map:
            god_map.unsafe_set_data(identifier.time, (rospy.get_rostime() - self.start_time).to_sec())
        return Status.SUCCESS
