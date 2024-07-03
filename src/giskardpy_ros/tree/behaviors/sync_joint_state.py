from typing import Optional

import rospy
from py_trees import Status
from sensor_msgs.msg import JointState

from giskardpy.data_types import JointStates
from giskardpy.data_types import PrefixName, Derivatives
from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time
from giskardpy.utils.utils import wait_for_topic_to_appear


class SyncJointState(GiskardBehavior):

    @record_time
    @profile
    def __init__(self, group_name: str, joint_state_topic: str = 'joint_states'):
        self.data = None
        self.group_name = group_name
        self.joint_state_topic = joint_state_topic
        if not self.joint_state_topic.startswith('/'):
            self.joint_state_topic = '/' + self.joint_state_topic
        super().__init__(str(self))

    @record_time
    @profile
    def setup(self, timeout=0.0):
        wait_for_topic_to_appear(topic_name=self.joint_state_topic, supported_types=[JointState])
        self.joint_state_sub = rospy.Subscriber(self.joint_state_topic, JointState, self.cb, queue_size=1)
        return super().setup(timeout)

    def cb(self, data):
        self.data = data

    @record_time
    @profile
    def update(self):
        if self.data:
            mjs = JointStates.from_msg(self.data, self.group_name)
            god_map.world.state.update(mjs)
            self.data = None
            return Status.SUCCESS
        return Status.RUNNING

    def __str__(self):
        return f'{super().__str__()} ({self.joint_state_topic})'


class SyncJointStatePosition(GiskardBehavior):
    """
    Listens to a joint state topic, transforms it into a dict and writes it to the got map.
    Gets replace with a kinematic sim plugin during a parallel universe.
    """

    msg: JointState

    @record_time
    @profile
    def __init__(self, group_name: str, joint_state_topic='joint_states'):
        super().__init__(str(self))
        self.joint_state_topic = joint_state_topic
        if not self.joint_state_topic.startswith('/'):
            self.joint_state_topic = '/' + self.joint_state_topic
        wait_for_topic_to_appear(topic_name=self.joint_state_topic, supported_types=[JointState])
        super().__init__(str(self))
        self.mjs: Optional[JointStates] = None
        self.group_name = group_name

    @record_time
    @profile
    def setup(self, timeout=0.0):
        self.joint_state_sub = rospy.Subscriber(self.joint_state_topic, JointState, self.cb, queue_size=1)
        return super().setup(timeout)

    def cb(self, data):
        self.msg = data

    @profile
    def initialise(self):
        self.last_time = rospy.get_rostime()
        super().initialise()

    @record_time
    @profile
    def update(self):
        for joint_name, position in zip(self.msg.name, self.msg.position):
            joint_name = PrefixName(joint_name, self.group_name)
            god_map.world.state[joint_name][Derivatives.position] = position

        return Status.SUCCESS
