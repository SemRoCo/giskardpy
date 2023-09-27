import rospy
from py_trees import Status

from giskardpy import identifier
from giskardpy.tree.behaviors.plugin import GiskardBehavior


class RosTime(GiskardBehavior):
    @property
    def start_time(self):
        return GodMap.god_map.get_data(identifier.tracking_start_time)

    @profile
    def update(self):
        with GodMap.god_map:
            GodMap.god_map.unsafe_set_data(identifier.time, (rospy.get_rostime() - self.start_time).to_sec())
        return Status.SUCCESS
