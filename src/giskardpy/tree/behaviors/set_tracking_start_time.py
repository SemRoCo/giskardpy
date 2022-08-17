import rospy
from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.tree.behaviors.plugin import GiskardBehavior


class SetTrackingStartTime(GiskardBehavior):
    @profile
    def initialise(self):
        super().initialise()
        self.god_map.set_data(identifier.tracking_start_time, rospy.get_rostime() + rospy.Duration(0.5))

    @profile
    def update(self):
        return Status.SUCCESS

