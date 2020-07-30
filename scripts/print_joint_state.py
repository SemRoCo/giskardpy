#!/usr/bin/env python
import rospy
from sensor_msgs.msg._JointState import JointState


def cb(data):
    for i, joint_name in enumerate(data.name):
        print("{}: {}".format(joint_name, data.position[i]))
    rospy.signal_shutdown('time is up')

if __name__ == '__main__':
    rospy.init_node('muh', anonymous=True)

    rospy.Subscriber('joint_states', JointState, cb)

    rospy.spin()