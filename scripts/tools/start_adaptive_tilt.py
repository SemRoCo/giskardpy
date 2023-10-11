#!/usr/bin/env python
from giskardpy.python_interface import GiskardWrapper
import rospy
from geometry_msgs.msg import Quaternion, PointStamped, PoseStamped
from tf.transformations import quaternion_from_matrix
from giskardpy.goals.goal import WEIGHT_ABOVE_CA

rospy.init_node('adaptiveTilt')

giskard = GiskardWrapper()

goal_pose = PoseStamped()
goal_pose.header.frame_id = 'map'
goal_pose.pose.position.x = 1.9
goal_pose.pose.position.y = - 0.2
goal_pose.pose.position.z = 0.65
goal_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 0, 1, 0],
                                                                 [0, -1, 0, 0],
                                                                 [1, 0, 0, 0],
                                                                 [0, 0, 0, 1]]))
giskard.set_cart_goal(goal_pose, 'hand_palm_link', 'map')
giskard.set_joint_goal({'arm_flex_joint': -0.8}, weight=WEIGHT_ABOVE_CA)
giskard.allow_all_collisions()
giskard.plan_and_execute()

giskard.set_json_goal('PouringAdaptiveTilt',
                      tip='hand_palm_link',
                      root='map',
                      tilt_angle=-1.5)
goal_point = PointStamped()
goal_point.header.frame_id = 'map'
goal_point.point = goal_pose.pose.position
giskard.set_json_goal('CartesianPosition',
                      root_link='map',
                      tip_link='hand_palm_link',
                      goal_point=goal_point)
giskard.allow_all_collisions()
giskard.plan_and_execute()
