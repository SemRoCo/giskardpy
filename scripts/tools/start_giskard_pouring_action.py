#!/usr/bin/env python
from giskardpy.python_interface import GiskardWrapper
import rospy
from geometry_msgs.msg import Quaternion, PointStamped, QuaternionStamped
from tf.transformations import quaternion_from_matrix

rospy.init_node('pouring')

giskard = GiskardWrapper()
orientation = QuaternionStamped()
orientation.header.frame_id = 'map'
orientation.quaternion = Quaternion(*quaternion_from_matrix([[0, 0, 1, 0],
                                                             [0, -1, 0, 0],
                                                             [1, 0, 0, 0],
                                                             [0, 0, 0, 1]]))

down_orientation = QuaternionStamped()
down_orientation.header.frame_id = 'map'
down_orientation.quaternion = Quaternion(*quaternion_from_matrix([[0, 0, 1, 0],
                                                                  [0, 1, 0, 0],
                                                                  [-1, 0, 0, 0],
                                                                  [0, 0, 0, 1]]))

container_plane = PointStamped()
container_plane.header.frame_id = 'map'
container_plane.point.x = 2
container_plane.point.z = 1

giskard.set_json_goal('PouringAction',
                      tip_link='hand_palm_link',
                      root_link='map',
                      upright_orientation=orientation,
                      down_orientation=down_orientation,
                      container_plane=container_plane,
                      tilt_joint='wrist_roll_joint')
giskard.set_avoid_joint_limits_goal()
giskard.plan_and_execute()
