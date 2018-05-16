#!/usr/bin/env python
import rospy
from actionlib import SimpleActionClient
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from giskard_msgs.msg import MoveAction, MoveGoal, MoveCmd, Controller, CollisionEntry
from giskard_msgs.srv import UpdateWorld

from giskardpy.python_interface import GiskardWrapper
from test_update_world import add_table, clear_world

def reset(giskard):
    """
    :param giskard:
    :type giskard: GiskardWrapper
    """
    p = PoseStamped()
    p.header.frame_id = 'base_link'
    p.pose.position = Point(0.628, -0.316, 0.636)
    p.pose.orientation = Quaternion(0,0,0,1)
    giskard.set_cart_goal('r_gripper_tool_frame', p)

    p = PoseStamped()
    p.header.frame_id = 'base_link'
    p.pose.position = Point(0.628, 0.316, 0.636)
    p.pose.orientation = Quaternion(0,0,0,1)
    giskard.set_cart_goal('l_gripper_tool_frame', p)

    collision_entry = CollisionEntry()
    collision_entry.type = CollisionEntry.ALLOW_ALL_COLLISIONS
    giskard.set_collision_entries([collision_entry])

    giskard.send_goals()

def test_1(giskard):
    """
    :param giskard:
    :type giskard: GiskardWrapper
    """
    p = PoseStamped()
    p.header.frame_id = 'r_gripper_tool_frame'
    p.pose.position = Point(0.1,0,0)
    p.pose.orientation = Quaternion(0,0,0,1)
    giskard.set_cart_goal('r_gripper_tool_frame', p)

    collision_entry = CollisionEntry()
    collision_entry.type = CollisionEntry.ALLOW_COLLISION
    collision_entry.body_b = 'table'
    giskard.set_collision_entries([collision_entry])

    giskard.send_goals()

def test_2(giskard):
    """
    :param giskard:
    :type giskard: GiskardWrapper
    """
    p = PoseStamped()
    p.header.frame_id = 'r_gripper_tool_frame'
    p.pose.position = Point(0.1,0,0)
    p.pose.orientation = Quaternion(0,0,0,1)
    giskard.set_cart_goal('r_gripper_tool_frame', p)

    collision_entry = CollisionEntry()
    collision_entry.type = CollisionEntry.AVOID_COLLISION
    collision_entry.min_dist = 0.2
    collision_entry.body_b = 'table'
    giskard.set_collision_entries([collision_entry])

    giskard.send_goals()

if __name__ == '__main__':
    rospy.init_node('muhkuh')
    giskard = GiskardWrapper([('base_link', 'l_gripper_tool_frame'),
                              ('base_link', 'r_gripper_tool_frame')])

    rospy.wait_for_service('muh/update_world')
    update_world = rospy.ServiceProxy('muh/update_world', UpdateWorld)
    clear_world(update_world)
    add_table(update_world, position=(1.5, 0, 0), orientation=(0,0,0,1))


    reset(giskard)
    test_1(giskard)
    reset(giskard)
    test_2(giskard)
    reset(giskard)
