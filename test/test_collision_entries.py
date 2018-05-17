#!/usr/bin/env python
import rospy
from actionlib import SimpleActionClient
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from giskard_msgs.msg import MoveAction, MoveGoal, MoveCmd, Controller, CollisionEntry, MoveResult
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

    result = giskard.plan_and_execute()
    assert (result.error_code == MoveResult.SUCCESS)
    print('test 0 successful')

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
    collision_entry.body_b = 'box'
    giskard.set_collision_entries([collision_entry])

    result = giskard.plan_and_execute()
    assert (result.error_code == MoveResult.SUCCESS)
    print('test 1 successful')

def test_2(giskard):
    """
    :param giskard:
    :type giskard: GiskardWrapper
    """
    p = PoseStamped()
    p.header.frame_id = 'r_gripper_tool_frame'
    p.pose.position = Point(0,0,0)
    p.pose.orientation = Quaternion(0,0,0,1)
    giskard.set_cart_goal('r_gripper_tool_frame', p)

    collision_entry = CollisionEntry()
    collision_entry.type = CollisionEntry.AVOID_COLLISION
    collision_entry.min_dist = 0.2
    collision_entry.body_b = 'box'
    giskard.set_collision_entries([collision_entry])

    result = giskard.plan_and_execute()
    assert(result.error_code == MoveResult.SUCCESS)
    print('test 2 successful')

def test_3(giskard):
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
    collision_entry.body_b = 'boxy'
    giskard.set_collision_entries([collision_entry])

    result = giskard.plan_and_execute()
    assert(result.error_code == MoveResult.UNKNOWN_OBJECT)
    print('test 3 successful')

def test_4(giskard):
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
    collision_entry.type = CollisionEntry.AVOID_ALL_COLLISIONS
    collision_entry.min_dist = 0.2
    giskard.set_collision_entries([collision_entry])

    result = giskard.plan_and_execute()
    assert(result.error_code == MoveResult.SUCCESS)
    print('test 4 successful')

def test_5(giskard):
    """
    :param giskard:
    :type giskard: GiskardWrapper
    :return:
    """
    #test1
    p = PoseStamped()
    p.header.frame_id = 'r_gripper_tool_frame'
    p.pose.position = Point(0.1,0,0)
    p.pose.orientation = Quaternion(0,0,0,1)
    giskard.set_cart_goal('r_gripper_tool_frame', p)

    collision_entry = CollisionEntry()
    collision_entry.type = CollisionEntry.ALLOW_COLLISION
    collision_entry.body_b = 'box'
    giskard.set_collision_entries([collision_entry])

    giskard.add_cmd()

    p = PoseStamped()
    p.header.frame_id = 'r_gripper_tool_frame'
    p.pose.position = Point(-0.1,0,0)
    p.pose.orientation = Quaternion(0,0,0,1)
    giskard.set_cart_goal('r_gripper_tool_frame', p)

    collision_entry = CollisionEntry()
    collision_entry.type = CollisionEntry.ALLOW_COLLISION
    collision_entry.body_b = 'box'
    giskard.set_collision_entries([collision_entry])

    giskard.add_cmd()

    #test2
    p = PoseStamped()
    p.header.frame_id = 'r_gripper_tool_frame'
    p.pose.position = Point(0,0,0)
    p.pose.orientation = Quaternion(0,0,0,1)
    giskard.set_cart_goal('r_gripper_tool_frame', p)

    collision_entry = CollisionEntry()
    collision_entry.type = CollisionEntry.AVOID_COLLISION
    collision_entry.min_dist = 0.2
    collision_entry.body_b = 'box'
    giskard.set_collision_entries([collision_entry])

    giskard.add_cmd()

    #test4
    p = PoseStamped()
    p.header.frame_id = 'r_gripper_tool_frame'
    p.pose.position = Point(0.1,0,0)
    p.pose.orientation = Quaternion(0,0,0,1)
    giskard.set_cart_goal('r_gripper_tool_frame', p)

    collision_entry = CollisionEntry()
    collision_entry.type = CollisionEntry.ALLOW_ALL_COLLISIONS
    giskard.set_collision_entries([collision_entry])

    result = giskard.plan_and_execute()
    assert(result.error_code == MoveResult.SUCCESS)
    print('test 5 successful')




if __name__ == '__main__':
    rospy.init_node('muhkuh')
    giskard = GiskardWrapper([('base_link', 'l_gripper_tool_frame'),
                              ('base_link', 'r_gripper_tool_frame')])

    giskard.clear_world()
    giskard.add_box(name='box', position=(1.2,0,0.5))


    # reset(giskard)
    # test_1(giskard)
    # reset(giskard)
    # test_2(giskard)
    # reset(giskard)
    # test_3(giskard)
    # reset(giskard)
    # test_4(giskard)
    reset(giskard)
    test_5(giskard)
