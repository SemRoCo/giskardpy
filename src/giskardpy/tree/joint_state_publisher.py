from threading import Thread

import rospy
from geometry_msgs.msg import Twist
from py_trees import Status
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectoryPoint

import giskardpy.identifier as identifier
from giskardpy.data_types import KeyDefaultDict
from giskardpy.tree.plugin import GiskardBehavior
from giskardpy.utils import logging


class JointStatePublisher(GiskardBehavior):
    def __init__(self, name, namespace):
        super().__init__(name)
        self.namespace = namespace
        self.cmd_topic = '{}/command'.format(self.namespace)
        self.cmd_pub = rospy.Publisher(self.cmd_topic, JointState, queue_size=10)
        self.joint_names = rospy.get_param('{}/controlled_joints'.format(self.namespace))
        self.world.register_controlled_joints(self.joint_names)

    def initialise(self):
        self.sample_period = self.god_map.get_data(identifier.sample_period)

        def f(joint_symbol):
            return self.god_map.expr_to_key[joint_symbol][-2]

        self.symbol_to_joint_map = KeyDefaultDict(f)
        super().initialise()

    def update(self):
        next_cmds = self.god_map.get_data(identifier.qp_solver_solution)
        for joint_symbol in next_cmds[0]:
            joint_name = self.symbol_to_joint_map[joint_symbol]
            self.world.joints[joint_name].update_state(next_cmds, self.sample_period)
        self.world.notify_state_change()
        msg = JointState()
        msg.header.stamp = rospy.get_rostime()
        for joint_name in self.joint_names:
            msg.name.append(joint_name)
            msg.position.append(self.world.state[joint_name].position)
            msg.velocity.append(self.world.state[joint_name].velocity)
        # print(self.world.state['odom_z_joint'].velocity)
        self.cmd_pub.publish(msg)
        return Status.RUNNING
