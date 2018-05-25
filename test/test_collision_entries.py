#!/usr/bin/env python
import unittest
from math import pi
import rospy
from actionlib import SimpleActionClient
from angles import normalize_angle
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from giskard_msgs.msg import MoveAction, MoveGoal, MoveCmd, Controller, CollisionEntry, MoveResult
from giskard_msgs.srv import UpdateWorld
from sensor_msgs.msg import JointState

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
        # TODO set joint goal instead of cart
        self.giskard.clear_world()
        js = {'r_elbow_flex_joint': -1.29610152504,
              'r_forearm_roll_joint': -0.0301682323805,
              'r_gripper_joint': 2.22044604925e-16,
              'r_gripper_l_finger_joint': 2.22044604925e-16,
              'r_gripper_l_finger_tip_joint': 2.22044604925e-16,
              'r_gripper_motor_screw_joint': 0.0,
              'r_gripper_motor_slider_joint': 0.0,
              'r_gripper_r_finger_joint': 2.22044604925e-16,
              'r_gripper_r_finger_tip_joint': 2.22044604925e-16,
              'r_shoulder_lift_joint': 1.20324921318,
              'r_shoulder_pan_joint': -0.643456435706,
              'r_upper_arm_roll_joint': -0.680790051778,
              'r_wrist_flex_joint': -2.22044604925e-16,
              'r_wrist_roll_joint': 0.258268529825,

              'l_elbow_flex_joint': -1.29610152504,
              'l_forearm_roll_joint': 0.0301682323805,
              'l_gripper_joint': 2.22044604925e-16,
              'l_gripper_l_finger_joint': 2.22044604925e-16,
              'l_gripper_l_finger_tip_joint': 2.22044604925e-16,
              'l_gripper_motor_screw_joint': 0.0,
              'l_gripper_motor_slider_joint': 0.0,
              'l_gripper_r_finger_joint': 2.22044604925e-16,
              'l_gripper_r_finger_tip_joint': 2.22044604925e-16,
              'l_shoulder_lift_joint': 1.20324921318,
              'l_shoulder_pan_joint': 0.643456435706,
              'l_upper_arm_roll_joint': 0.680790051778,
              'l_wrist_flex_joint': -2.22044604925e-16,
              'l_wrist_roll_joint': -0.258268529825,

              'torso_lift_joint': 0.3,
              'head_pan_joint': 0,
              'head_tilt_joint': 0,
              }
        self.giskard.set_joint_goal(js)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_ALL_COLLISIONS
        self.giskard.set_collision_entries([collision_entry])

        self.test_js_goal(js)
        super(TestCollisionAvoidance, self).setUpClass()

    def test_js_goal(self, goal_js):
        self.giskard.set_joint_goal(goal_js)
        self.giskard.plan_and_execute()
        result = self.giskard.get_result()
        self.assertEqual(result.error_code, MoveResult.SUCCESS)
        current_joint_state = rospy.wait_for_message('joint_states', JointState)  # type: JointState
        for i, joint_name in enumerate(current_joint_state.name):
            if joint_name in goal_js:
                goal = normalize_angle(goal_js[joint_name])
                current = normalize_angle(current_joint_state.position[i])
                self.assertAlmostEqual(goal, current, 2,
                                       msg='{} is {} instead of {}'.format(joint_name,
                                                                           current,
                                                                           goal))

    def add_box(self):
        self.giskard.add_box(name='box', position=(1.2, 0, 0.5))

    def test_AllowCollision1(self):
        self.add_box()
        p = PoseStamped()
        p.header.frame_id = 'r_gripper_tool_frame'
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
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
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
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
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
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
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
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
        p.pose.position = Point(0.15, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        self.giskard.set_cart_goal('r_gripper_tool_frame', p)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_ALL_COLLISIONS
        collision_entry.min_dist = 0.05
        self.giskard.set_collision_entries([collision_entry])

        result = self.giskard.plan_and_execute()
        self.assertEqual(result.error_code, MoveResult.SUCCESS)

        p = PoseStamped()
        p.header.frame_id = 'r_gripper_tool_frame'
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        self.giskard.set_cart_goal('r_gripper_tool_frame', p)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_ALL_COLLISIONS
        collision_entry.min_dist = 0.05
        self.giskard.set_collision_entries([collision_entry])

        result = self.giskard.plan_and_execute()
        # self.assertEqual(result.error_code, MoveResult.START_STATE_COLLISION)
        self.assertEqual(result.error_code, MoveResult.SUCCESS)

    def test_interrupt1(self):
        self.add_box()
        p = PoseStamped()
        p.header.frame_id = 'r_gripper_tool_frame'
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        self.giskard.set_cart_goal('r_gripper_tool_frame', p)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_ALL_COLLISIONS
        collision_entry.min_dist = 0.5
        self.giskard.set_collision_entries([collision_entry])

        self.giskard.plan_and_execute(wait=False)
        self.giskard.interrupt()
        result = self.giskard.get_result()
        self.assertEqual(result.error_code, MoveResult.INTERRUPTED)

    def test_waypoints(self):
        p = PoseStamped()
        p.header.frame_id = 'r_gripper_tool_frame'
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        self.giskard.set_cart_goal('r_gripper_tool_frame', p)

        self.giskard.add_cmd()
        p = PoseStamped()
        p.header.frame_id = 'r_gripper_tool_frame'
        p.pose.position = Point(0, 0, 0.1)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        self.giskard.set_cart_goal('r_gripper_tool_frame', p)

        self.giskard.add_cmd()
        p = PoseStamped()
        p.header.frame_id = 'r_gripper_tool_frame'
        p.pose.position = Point(-0.1, 0, -0.1)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        self.giskard.set_cart_goal('r_gripper_tool_frame', p)

        self.giskard.plan_and_execute()
        result = self.giskard.get_result()
        self.assertEqual(result.error_code, MoveResult.SUCCESS)

    def test_joint_state_goal1(self):
        goal_js = {'torso_lift_joint': 0.2}
        self.giskard.set_joint_goal(goal_js)
        self.giskard.plan_and_execute()

        # goal = MoveGoal()
        # goal.type = MoveGoal.PLAN_AND_EXECUTE
        #
        # # translation
        # controller = Controller()
        # controller.type = Controller.JOINT
        # controller.tip_link = 'gripper_tool_frame'
        # controller.root_link = 'base_footprint'
        #
        # for i, joint_name in enumerate(self.joint_names):
        #     controller.goal_state.name.append(joint_name)
        #     # controller.goal_state.position.append(0)
        #     controller.goal_state.position.append(np.random.random()-0.5)
        #
        # controller.p_gain = 3
        # controller.max_trajectory_length = 0.05
        # controller.weight = 1
        # goal.cmd_seq.append(MoveCmd())
        # goal.cmd_seq[-1].controllers.append(controller)
        #
        # self.client.send_goal(goal)
        # result = self.client.wait_for_result()
        # final_js = rospy.wait_for_message('/whole_body_controller/state', JointTrajectoryControllerState) # type: JointTrajectoryControllerState
        # asdf = {}
        # for i, joint_name in enumerate(final_js.joint_names):
        #     asdf[joint_name] = final_js.actual.positions[i]
        # for i, joint_name in enumerate(controller.goal_state.name):
        #     print('{} real:{} | exp:{}'.format(joint_name, asdf[joint_name], controller.goal_state.position[i]))
        # print('finished in 10s?: {}'.format(result))

    def test_continuous_joint1(self):
        goal_js = {'r_wrist_roll_joint': 0,
                   'l_wrist_roll_joint': 3.5 * pi, }
        self.giskard.set_joint_goal(goal_js)
        self.giskard.plan_and_execute()
        current_joint_state = rospy.wait_for_message('joint_states', JointState)  # type: JointState
        for i, joint_name in enumerate(current_joint_state.name):
            if joint_name in goal_js:
                goal = normalize_angle(goal_js[joint_name])
                current = normalize_angle(current_joint_state.position[i])
                self.assertAlmostEqual(goal, current, 2,
                                       msg='{} is {} instead of {}'.format(joint_name,
                                                                           current,
                                                                           goal))


if __name__ == '__main__':
    import rosunit

    rosunit.unitrun(package=PKG,
                    test_name='TestCollisionAvoidance',
                    test=TestCollisionAvoidance)
