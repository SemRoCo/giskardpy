import rospy
from py_trees import Status

from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.middleware import logging


class SetTrackingStartTime(GiskardBehavior):
    def __init__(self, name, offset: float = 0.5):
        super().__init__(name)
        self.offset = rospy.Duration(offset)

    @profile
    def initialise(self):
        super().initialise()
        god_map.motion_start_time = rospy.get_rostime().to_sec() + self.offset.to_sec()

    @profile
    def update(self):
        return Status.SUCCESS
