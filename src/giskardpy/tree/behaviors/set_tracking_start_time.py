import rospy
from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.tree.behaviors.plugin import GiskardBehavior


class SetTrackingStartTime(GiskardBehavior):
    def __init__(self, name, offset: float = 0.5):
        super().__init__(name)
        self.offset = rospy.Duration(offset)

    @profile
    def initialise(self):
        super().initialise()
        self.god_map.set_data(identifier.tracking_start_time, rospy.get_rostime() + self.offset)

    @profile
    def update(self):
        return Status.SUCCESS

