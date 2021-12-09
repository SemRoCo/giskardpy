#! /usr/bin/env python

import rospy
import actionlib
import control_msgs.msg





if __name__ == '__main__':
    rospy.init_node('TimeooutActionServer')
    name = rospy.get_name()
    server = TimeoutActionServer(name)
    rospy.spin()