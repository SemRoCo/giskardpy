#!/usr/bin/env python

import rospy
import sys
from giskardpy import urdf_object
from sensor_msgs.msg import JointState

# Visualizes the given joint states w.r.t. their limits
# rosrun giskardpy joint_limit_observer <joint_state_topic>

warning_threshold = 0.1
critical_threshold = 0.02
scale_steps = 20


class OutColor:
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    NORM = '\033[0m'


class JSObserver(rospy.Subscriber):
    def __init__(self, robot, topic, message_type):
        self.robot = robot
        self.joint_limits = robot.get_all_joint_limits()
        super(JSObserver, self).__init__(topic, message_type, self.js_cb)

    def js_cb(self, msg):
        output = ""
        for joint in msg.name:
            if not self.joint_limits[joint] == (None, None):
                current = msg.position[msg.name.index(joint)]
                lower, upper = self.joint_limits[joint]
                js_range = upper - lower
                relative = int(round(scale_steps / js_range * (current - lower)))

                color = OutColor.NORM
                limit_distance = min(abs(current - lower), abs(current - upper))
                if limit_distance < js_range * critical_threshold:
                    color = OutColor.FAIL
                elif limit_distance < js_range * warning_threshold:
                    color = OutColor.WARNING
                s = "+" * (scale_steps + 1)
                s_form = s[:relative] + "0" + s[relative + 1:]
                output += ("{}[" + s_form + "] {}\n").format(color, joint)
            else:
                output += "limitless: %s\n" % joint
        print(output + OutColor.NORM + "===")


if __name__ == '__main__':
    rospy.init_node('joint_limit_observer')
    robot_desc = rospy.get_param('robot_description')
    js_topic = '/joint_states' if len(sys.argv) == 1 else sys.argv[1]
    if robot_desc:
        robot_object = urdf_object.URDFObject(robot_desc)
        JSObserver(robot_object, js_topic, JointState)
        rospy.loginfo("About to spin")
        rospy.spin()
    else:
        rospy.loginfo("Couldn't get robot description from parameter server")
