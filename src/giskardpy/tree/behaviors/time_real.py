import rospy
from py_trees import Status

from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior


class RosTime(GiskardBehavior):
    @property
    def start_time(self):
        return god_map.tracking_start_time

    @profile
    def update(self):
        god_map.time = (rospy.get_rostime() - self.start_time).to_sec()
        return Status.SUCCESS
