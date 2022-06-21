#!/usr/bin/env python
import rospy
from sensor_msgs.msg._JointState import JointState

from giskardpy.utils.utils import print_joint_state


def cb(data):
    print_joint_state(data)
    rospy.signal_shutdown('time is up')

if __name__ == '__main__':
    rospy.init_node('muh', anonymous=True)

    rospy.Subscriber('joint_states', JointState, cb)

    rospy.spin()