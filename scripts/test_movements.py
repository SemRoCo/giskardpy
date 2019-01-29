#!/usr/bin/env python
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

from giskardpy.python_interface import GiskardWrapper


if __name__ == '__main__':
    rospy.init_node('donbot_test_movements')

    g = GiskardWrapper()
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
    g.add_box(position=[1.2,0,0.5])
    r_tip = 'r_gripper_tool_frame'
    p = PoseStamped()
    p.header.frame_id = r_tip
    p.pose.position = Point(0.1, 0, 0)
    p.pose.orientation = Quaternion(0, 0, 0, 1)
    g.set_cart_goal('base_link', r_tip, p)

    collision_entry = CollisionEntry()
    collision_entry.type = CollisionEntry.AVOID_ALL_COLLISIONS
    collision_entry.min_dist = 0.05
    g.set_collision_entries([collision_entry])
    g.plan_and_execute()
