#!/usr/bin/env python
import rospy
import sys
from geometry_msgs.msg import PoseStamped

from giskardpy.tfwrapper import lookup_pose, init


def cb(dt):
    try:
        pose_pub.publish(lookup_pose(parent, child))
    except:
        pass

rospy.init_node('pose_publisher')
init(2)
parent = sys.argv[1]
child = sys.argv[2]
pose_pub = rospy.Publisher('pose_{}_{}'.format(parent, child), PoseStamped, queue_size=10)
timer = rospy.Timer(rospy.Duration(0.1), cb)
rospy.spin()