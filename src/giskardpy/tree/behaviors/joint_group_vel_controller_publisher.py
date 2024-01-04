from copy import deepcopy

import rospy
from py_trees import Status
from py_trees.behaviours import Running
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

from giskardpy.data_types import KeyDefaultDict, JointStates
from giskardpy.god_map import god_map
from giskardpy.data_types import Derivatives
from giskardpy.tree.behaviors.cmd_publisher import CommandPublisher
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import catch_and_raise_to_blackboard, record_time
from giskardpy.utils.utils import wait_for_topic_to_appear


class JointGroupVelController(GiskardBehavior):
    @profile
    def __init__(self, namespace, group_name: str = None, hz=100):
        super().__init__(namespace)
        self.namespace = namespace
        self.cmd_topic = f'{self.namespace}/command'
        wait_for_topic_to_appear(topic_name=self.cmd_topic, supported_types=[Float64MultiArray])
        self.cmd_pub = rospy.Publisher(self.cmd_topic, Float64MultiArray, queue_size=10)
        self.joint_names = rospy.get_param(f'{self.namespace}/joints')
        for i in range(len(self.joint_names)):
            self.joint_names[i] = god_map.world.search_for_joint_name(self.joint_names[i])
        god_map.world.register_controlled_joints(self.joint_names)
        self.msg = None

    @profile
    def initialise(self):
        def f(joint_symbol):
            return god_map.expr_to_key[joint_symbol][-2]

        self.symbol_to_joint_map = KeyDefaultDict(f)
        super().initialise()

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        msg = Float64MultiArray()
        for i, joint_name in enumerate(self.joint_names):
            msg.data.append(god_map.world.state[joint_name].velocity)
        self.cmd_pub.publish(msg)
        return Status.RUNNING

    def terminate(self, new_status):
        msg = Float64MultiArray()
        for joint_name in self.joint_names:
            msg.data.append(0.0)
        self.cmd_pub.publish(msg)
        super().terminate(new_status)
