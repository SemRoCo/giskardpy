from collections import OrderedDict

import rospy
from py_trees import Status
from sensor_msgs.msg import JointState

from giskardpy.data_types import SingleJointState
import giskardpy.identifier as identifier
from giskardpy.plugin import GiskardBehavior


class SendJointStatePlugin(GiskardBehavior):
    def __init__(self, name):
        """
        :type js_identifier: str
        :type next_cmd_identifier: str
        :type time_identifier: str
        :param sample_period: the time difference in s between each step.
        :type sample_period: float
        """
        super(SendJointStatePlugin, self).__init__(name)

    def initialise(self):
        self.pub = rospy.Publisher('body_commands', JointState, queue_size=10)
        self.sample_period = self.get_god_map().get_data(identifier.sample_period)
        super(SendJointStatePlugin, self).initialise()

    def js_to_msg(self, js_dict):
        js = JointState()
        js.header.stamp = rospy.get_rostime()
        for joint_name, sjs in js_dict.items(): # type: SingleJointState
            js.name.append(sjs.name)
            js.position.append(sjs.position)
            js.velocity.append(sjs.velocity)
        js.effort = [0]*len(js.velocity)
        return js

    @profile
    def update(self):
        motor_commands = self.get_god_map().get_data(identifier.cmd)
        current_js = self.get_god_map().get_data(identifier.joint_states)
        next_js = None
        if motor_commands:
            next_js = OrderedDict()
            for joint_name, sjs in current_js.items():
                if joint_name in motor_commands:
                    cmd = motor_commands[joint_name]
                else:
                    cmd = 0.0
                next_js[joint_name] = SingleJointState(sjs.name, sjs.position + cmd,
                                                            velocity=cmd/self.sample_period)
        # if next_js is not None:
        #     self.get_god_map().set_data(identifier.joint_states, next_js)
        # else:
        #     self.get_god_map().set_data(identifier.joint_states, current_js)
        self.pub.publish(self.js_to_msg(next_js))
        self.get_god_map().set_data(identifier.last_joint_states, current_js)
        return Status.RUNNING
