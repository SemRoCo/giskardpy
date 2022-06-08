import rospy
from py_trees import Status

from giskardpy import identifier
from giskardpy.tree.behaviors.plugin import GiskardBehavior


class RosTime(GiskardBehavior):
    def initialise(self):
        self.last_call = rospy.get_rostime()

    @profile
    def update(self):
        with self.god_map:
            self.god_map.unsafe_set_data(identifier.time, (rospy.get_rostime() - self.last_call).to_sec())
        # print(f'time {self.god_map.get_data(identifier.time):.4}')
        return Status.RUNNING