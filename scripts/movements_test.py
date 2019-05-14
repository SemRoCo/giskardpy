#!/usr/bin/env python
from time import time

import rospy

from actionlib.simple_action_client import SimpleActionClient
from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg._Point import Point
from geometry_msgs.msg._PoseStamped import PoseStamped
from geometry_msgs.msg._Quaternion import Quaternion
from giskard_msgs.msg import CollisionEntry
from giskard_msgs.msg._Controller import Controller
from giskard_msgs.msg._MoveAction import MoveAction
from giskard_msgs.msg._MoveCmd import MoveCmd
from giskard_msgs.msg._MoveGoal import MoveGoal
from sensor_msgs.msg._JointState import JointState
import numpy as np
from giskardpy import logging

from giskardpy.python_interface import GiskardWrapper

if __name__ == '__main__':
    rospy.init_node('donbot_test_movements')
    t = time()
    g = GiskardWrapper()

    js = {
        'gripper_joint': 0.0065,
        'odom_x_joint': 0.0,
        'odom_y_joint': 0.0,
        'odom_z_joint': 0.0,
        'ur5_elbow_joint': 1.05104740416,
        'ur5_shoulder_lift_joint': -0.723444608389,
        'ur5_shoulder_pan_joint': -0.0106331042379,
        'ur5_wrist_1_joint': 3.41359659947,
        'ur5_wrist_2_joint': -1.52307799476,
        'ur5_wrist_3_joint': 0.052335263781,
    }
    g.set_joint_goal(js)
    g.avoid_collision(0.01, [], 'iai_donbot', [])
    g.plan_and_execute()
    g.avoid_collision(0.01, [], 'iai_donbot', [])
    # g.allow_all_collisions()

    #
    # goal_pose = PoseStamped()
    # goal_pose.header.frame_id = u'base_link'
    # goal_pose.pose.position.x = 0.212
    # goal_pose.pose.position.y = -0.314
    # goal_pose.pose.position.z = 0.873
    # goal_pose.pose.orientation.x = 0.004
    # goal_pose.pose.orientation.y = 0.02
    # goal_pose.pose.orientation.z = 0.435
    # goal_pose.pose.orientation.w = .9
    #
    # g.set_cart_goal('base_link', 'gripper_tool_frame', goal_pose)
    # g.allow_all_collisions()
    # g.plan_and_execute()
    tip = 'gripper_tool_frame'
    p = PoseStamped()
    p.header.frame_id = tip
    p.pose.position = Point(0.0, 0, -1)
    p.pose.orientation = Quaternion(0, 0, 0, 1)
    g.set_cart_goal('base_footprint', tip, p)

    g.plan_and_execute()
    logging.loginfo('shit took {}'.format(time()-t))

#nothing 30.92
#coll    31.22