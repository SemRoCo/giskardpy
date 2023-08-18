from geometry_msgs.msg import PoseStamped

import rospy
from py_trees import Status

from giskardpy import identifier
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.model.utils import make_world_body_box


class SyncBoxPose(GiskardBehavior):

    @profile
    def __init__(self, group_name: str, state_topic='/boxUpdate'):
        """
        :type js_identifier: str
        """
        super().__init__(str(self))
        self.state_topic = state_topic
        if not self.state_topic.startswith('/'):
            self.ft_topic = '/' + self.state_topic
        super().__init__(str(self))
        self.pose_data = PoseStamped()
        self.new_data = False

    @profile
    def setup(self, timeout=0.0):
        self.sub = rospy.Subscriber(self.state_topic, PoseStamped, self.cb, queue_size=1)
        return super().setup(timeout)

    def cb(self, data):
        self.pose_data = data
        self.new_data = True

    @profile
    def update(self):
        if self.new_data:
            joint = self.world.joints['connection/box']
            joint.update_transform(self.pose_data.pose)
            self.new_data = False
        return Status.RUNNING
