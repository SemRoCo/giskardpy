import rospy
from py_trees import Status

from giskardpy import identifier
from giskardpy.tree.behaviors.plugin import GiskardBehavior


class RosTime(GiskardBehavior):
    @property
    def start_time(self):
        return self.god_map.get_data(identifier.tracking_start_time)

    @profile
    def update(self):
        with self.god_map:
            self.god_map.unsafe_set_data(identifier.time, (rospy.get_rostime() - self.start_time).to_sec())
        return Status.SUCCESS
