from copy import deepcopy

import rospy
from py_trees import Status
from std_msgs.msg import Float64

from giskardpy.data_types import KeyDefaultDict
from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import catch_and_raise_to_blackboard, record_time
from giskardpy.utils.utils import wait_for_topic_to_appear


class JointPosController(GiskardBehavior):
    last_time: float

    @profile
    def __init__(self, namespaces=None):
        super().__init__('joint position publisher')
        self.namespaces = namespaces
        self.publishers = []
        self.cmd_topics = []
        self.joint_names = []
        for namespace in self.namespaces:
            cmd_topic = f'/{namespace}/command'
            self.cmd_topics.append(cmd_topic)
            wait_for_topic_to_appear(topic_name=cmd_topic, supported_types=[Float64])
            self.publishers.append(rospy.Publisher(cmd_topic, Float64, queue_size=10))
            self.joint_names.append(rospy.get_param(f'{namespace}/joint'))
        for i in range(len(self.joint_names)):
            self.joint_names[i] = god_map.world.search_for_joint_name(self.joint_names[i])
        god_map.world.register_controlled_joints(self.joint_names)

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        msg = Float64()
        for i, joint_name in enumerate(self.joint_names):
            msg.data = god_map.world.state[joint_name].position
            self.publishers[i].publish(msg)
        return Status.RUNNING
