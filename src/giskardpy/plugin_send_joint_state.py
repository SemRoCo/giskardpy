from collections import OrderedDict

import rospy
from py_trees import Status
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist

from giskardpy.data_types import SingleJointState
import giskardpy.identifier as identifier
from giskardpy.plugin import GiskardBehavior

#Todo: List of controllers, topics, msg_type
#Todo: read corresponding position/velocity from calculated js (loop z.50)
#Todo: publish corresponding msgs

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

        self.min_pos_diff = 0.02
        self.min_vel = 0.01

        self.position_publisher = {
            u'head_tilt_joint': rospy.Publisher('/hsrb/head_tilt_joint_position_controller/command', Float64, queue_size=10),
            u'head_pan_joint': rospy.Publisher('/hsrb/head_pan_joint_position_controller/command', Float64, queue_size=10),
            u'arm_lift_joint': rospy.Publisher('/hsrb/arm_lift_joint_position_controller/command', Float64, queue_size=10),
            u'arm_flex_joint': rospy.Publisher('/hsrb/arm_flex_joint_position_controller/command', Float64, queue_size=10),
            u'arm_roll_joint': rospy.Publisher('/hsrb/arm_roll_joint_position_controller/command', Float64, queue_size=10),
            u'wrist_flex_joint': rospy.Publisher('/hsrb/wrist_flex_joint_position_controller/command', Float64, queue_size=10),
            u'wrist_roll_joint': rospy.Publisher('/hsrb/wrist_roll_joint_position_controller/command', Float64, queue_size=10)}

        self.base_publisher = rospy.Publisher('/hsrb/command_velocity', Twist, queue_size=10)

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
        for joint, pub in self.position_publisher.iteritems():
            pub.publish(Float64(data=next_js[joint].position))

        vel = Twist()
        vel.linear.x = next_js[u'odom_x'].velocity
        vel.linear.y = next_js[u'odom_y'].velocity
        vel.angular.z = next_js[u'odom_t'].velocity
        self.base_publisher.publish(vel)

        #self.pub.publish(self.js_to_msg(next_js))
        self.get_god_map().set_data(identifier.last_joint_states, current_js)
        return Status.RUNNING
