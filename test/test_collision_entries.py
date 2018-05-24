#!/usr/bin/env python
import unittest

import rospy
from actionlib import SimpleActionClient
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from giskard_msgs.msg import MoveAction, MoveGoal, MoveCmd, Controller, CollisionEntry, MoveResult
from giskard_msgs.srv import UpdateWorld

from giskardpy.python_interface import GiskardWrapper
from test_update_world import add_table, clear_world

PKG = 'giskardpy'

class TestCollisionAvoidance(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        rospy.init_node('test_collision_avoidance')
        self.giskard = GiskardWrapper([('base_link', 'l_gripper_tool_frame'),
                              ('base_link', 'r_gripper_tool_frame')])

        super(TestCollisionAvoidance, self).__init__(methodName)

    def setUp(self):
        self.giskard.clear_world()
        p = PoseStamped()
        p.header.frame_id = 'base_link'
        p.pose.position = Point(0.628, -0.316, 0.636)
        p.pose.orientation = Quaternion(0,0,0,1)
        self.giskard.set_cart_goal('r_gripper_tool_frame', p)

        p = PoseStamped()
        p.header.frame_id = 'base_link'
        p.pose.position = Point(0.628, 0.316, 0.636)
        p.pose.orientation = Quaternion(0,0,0,1)
        self.giskard.set_cart_goal('l_gripper_tool_frame', p)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_ALL_COLLISIONS
        self.giskard.set_collision_entries([collision_entry])

        result = self.giskard.plan_and_execute()
        self.assertEqual(result.error_code, MoveResult.SUCCESS)
        super(TestCollisionAvoidance, self).setUpClass()


    def add_box(self):
        self.giskard.add_box(name='box', position=(1.2, 0, 0.5))

    def test_AllowCollision1(self):
        self.add_box()
        p = PoseStamped()
        p.header.frame_id = 'r_gripper_tool_frame'
        p.pose.position = Point(0.1,0,0)
        p.pose.orientation = Quaternion(0,0,0,1)
        self.giskard.set_cart_goal('r_gripper_tool_frame', p)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_COLLISION
        collision_entry.body_b = 'box'
        self.giskard.set_collision_entries([collision_entry])

        result = self.giskard.plan_and_execute()
        self.assertEqual(result.error_code, MoveResult.SUCCESS)

    def test_avoid_collision1(self):
        self.add_box()
        p = PoseStamped()
        p.header.frame_id = 'r_gripper_tool_frame'
        p.pose.position = Point(0.1,0,0)
        p.pose.orientation = Quaternion(0,0,0,1)
        self.giskard.set_cart_goal('r_gripper_tool_frame', p)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_COLLISION
        collision_entry.min_dist = 0.01
        collision_entry.body_b = 'box'
        self.giskard.set_collision_entries([collision_entry])

        result = self.giskard.plan_and_execute()
        self.assertEqual(result.error_code, MoveResult.SUCCESS)

    def test_unknown_object1(self):
        self.add_box()
        p = PoseStamped()
        p.header.frame_id = 'r_gripper_tool_frame'
        p.pose.position = Point(0.1,0,0)
        p.pose.orientation = Quaternion(0,0,0,1)
        self.giskard.set_cart_goal('r_gripper_tool_frame', p)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_COLLISION
        collision_entry.min_dist = 0.2
        collision_entry.body_b = 'boxy'
        self.giskard.set_collision_entries([collision_entry])

        result = self.giskard.plan_and_execute()
        self.assertEqual(result.error_code, MoveResult.UNKNOWN_OBJECT)

    def test_avoid_all_collision1(self):
        self.add_box()
        p = PoseStamped()
        p.header.frame_id = 'r_gripper_tool_frame'
        p.pose.position = Point(0.1,0,0)
        p.pose.orientation = Quaternion(0,0,0,1)
        self.giskard.set_cart_goal('r_gripper_tool_frame', p)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_ALL_COLLISIONS
        collision_entry.min_dist = 0.2
        self.giskard.set_collision_entries([collision_entry])

        result = self.giskard.plan_and_execute()
        self.assertEqual(result.error_code, MoveResult.SUCCESS)

    def test_start_state_collision1(self):
        self.add_box()
        p = PoseStamped()
        p.header.frame_id = 'r_gripper_tool_frame'
        p.pose.position = Point(0.15,0,0)
        p.pose.orientation = Quaternion(0,0,0,1)
        self.giskard.set_cart_goal('r_gripper_tool_frame', p)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_ALL_COLLISIONS
        collision_entry.min_dist = 0.05
        self.giskard.set_collision_entries([collision_entry])

        result = self.giskard.plan_and_execute()
        self.assertEqual(result.error_code, MoveResult.SUCCESS)

        p = PoseStamped()
        p.header.frame_id = 'r_gripper_tool_frame'
        p.pose.position = Point(0.1,0,0)
        p.pose.orientation = Quaternion(0,0,0,1)
        self.giskard.set_cart_goal('r_gripper_tool_frame', p)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_ALL_COLLISIONS
        collision_entry.min_dist = 0.05
        self.giskard.set_collision_entries([collision_entry])

        result = self.giskard.plan_and_execute()
        # self.assertEqual(result.error_code, MoveResult.START_STATE_COLLISION)
        self.assertEqual(result.error_code, MoveResult.SUCCESS)

    def test_end_state_collision(self):
        self.add_box()
        p = PoseStamped()
        p.header.frame_id = 'r_gripper_tool_frame'
        p.pose.position = Point(0.1,0,0)
        p.pose.orientation = Quaternion(0,0,0,1)
        self.giskard.set_cart_goal('r_gripper_tool_frame', p)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_ALL_COLLISIONS
        collision_entry.min_dist = 5
        self.giskard.set_collision_entries([collision_entry])

        result = self.giskard.plan_and_execute()
        self.assertEqual(result.error_code, MoveResult.END_STATE_COLLISION)



if __name__ == '__main__':
    import rosunit

    rosunit.unitrun(package=PKG,
                    test_name='TestCollisionAvoidance',
                    test=TestCollisionAvoidance)


