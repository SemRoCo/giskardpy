#!/usr/bin/env python
import rospy
from sensor_msgs.msg import JointState
from tf.transformations import quaternion_matrix, rotation_from_matrix

from giskardpy.tfwrapper import lookup_pose

rospy.init_node('base_joint_state_publisher')
js_pub = rospy.Publisher('base/joint_states', JointState, queue_size=10)
hz = 10
r = rospy.Rate(hz)
js = JointState()
js.name = ['odom_x_joint', 'odom_y_joint', 'odom_z_joint']
while not rospy.is_shutdown():
    map_T_base_footprint = lookup_pose('map', 'base_footprint')
    js.position = [map_T_base_footprint.pose.position.x,
                   map_T_base_footprint.pose.position.y,
                   rotation_from_matrix(quaternion_matrix([map_T_base_footprint.pose.orientation.x,
                                                           map_T_base_footprint.pose.orientation.y,
                                                           map_T_base_footprint.pose.orientation.z,
                                                           map_T_base_footprint.pose.orientation.w]))[0]]
    js_pub.publish(js)
    r.sleep()
