#!/usr/bin/env python
import rospy
from sensor_msgs.msg import JointState
from tf.transformations import rotation_from_matrix, quaternion_matrix

from giskardpy.utils.tfwrapper import lookup_pose, init


def cb(dt):
    pose = lookup_pose(odom, base_footprint)
    js = JointState()
    js.header.stamp = pose.header.stamp
    js.name = [x, y, z]
    js.position = [pose.pose.position.x,
                   pose.pose.position.y,
                   rotation_from_matrix(quaternion_matrix([pose.pose.orientation.x,
                                                           pose.pose.orientation.y,
                                                           pose.pose.orientation.z,
                                                           pose.pose.orientation.w]))[0]
                   ]
    js.velocity = [0,0,0]
    js.effort = [0,0,0]
    pose_pub.publish(js)

rospy.init_node('base_joint_state_publisher')
init(2)
x = rospy.get_param('~odom_x_joint')
y = rospy.get_param('~odom_y_joint')
z = rospy.get_param('~odom_z_joint')
odom = rospy.get_param('~odom')
base_footprint = rospy.get_param('~base_footprint')
pose_pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
timer = rospy.Timer(rospy.Duration(0.1), cb)
rospy.loginfo('{} started'.format(rospy.get_name()))
rospy.spin()