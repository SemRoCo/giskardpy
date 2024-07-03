import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from py_trees import Status

from giskardpy.data_types import PrefixName
from giskardpy.god_map import god_map
from giskardpy.model.joints import OmniDrive
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import catch_and_raise_to_blackboard, record_time
from giskardpy.utils.utils import wait_for_topic_to_appear


class SyncOdometry(GiskardBehavior):

    @profile
    def __init__(self, odometry_topic: str, joint_name: PrefixName, name_suffix: str = ''):
        self.data = None
        self.odometry_topic = odometry_topic
        if not self.odometry_topic.startswith('/'):
            self.odometry_topic = '/' + self.odometry_topic
        super().__init__(str(self) + name_suffix)
        self.joint_name = joint_name

    def __str__(self):
        return f'{super().__str__()} ({self.odometry_topic})'

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def setup(self, timeout=0.0):
        actual_type = wait_for_topic_to_appear(topic_name=self.odometry_topic,
                                               supported_types=[Odometry, PoseWithCovarianceStamped])
        self.joint: OmniDrive = god_map.world.joints[self.joint_name]
        self.odometry_sub = rospy.Subscriber(self.odometry_topic, actual_type, self.cb, queue_size=1)

        return super().setup(timeout)

    def cb(self, data: Odometry):
        self.data = data

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        if self.data:
            self.joint.update_transform(self.data.pose.pose)
            self.data = None
            return Status.SUCCESS
        else:
            return Status.RUNNING


class SyncOdometryNoLock(SyncOdometry):

    @profile
    def __init__(self, odometry_topic: str, joint_name: PrefixName, name_suffix: str = ''):
        self.odometry_topic = odometry_topic
        GiskardBehavior.__init__(self, str(self) + name_suffix)
        self.joint_name = joint_name
        self.last_msg = None

    def cb(self, data: Odometry):
        self.odom = data

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        self.joint.update_transform(self.odom.pose.pose)
        return Status.SUCCESS
