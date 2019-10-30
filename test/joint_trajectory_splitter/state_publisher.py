#! /usr/bin/env python

import rospy
import control_msgs.msg


if __name__ == '__main__':

    rospy.init_node("state_publisher")
    name = rospy.get_name()

    joint_names = rospy.get_param(name + "/joint_names")

    pub = rospy.Publisher(name, control_msgs.msg.JointTrajectoryControllerState, queue_size=10)
    rate = rospy.Rate(10)  # 10hz
    while not rospy.is_shutdown():
        msg = control_msgs.msg.JointTrajectoryControllerState()
        msg.joint_names = joint_names
        pub.publish(msg)
        rate.sleep()
    rospy.spin()