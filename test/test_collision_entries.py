#!/usr/bin/env python
import unittest
from math import pi
from rospkg import RosPack

import rospy
from actionlib import SimpleActionClient
from angles import normalize_angle
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from giskard_msgs.msg import MoveAction, MoveGoal, MoveCmd, Controller, CollisionEntry, MoveResult
from giskard_msgs.srv import UpdateWorld, UpdateWorldResponse
from sensor_msgs.msg import JointState

from giskardpy.python_interface import GiskardWrapper
from giskardpy.tfwrapper import lookup_transform, transform_pose
from test_update_world import add_table, clear_world
from numpy import pi

PKG = 'giskardpy'

default_joint_state = {'r_elbow_flex_joint': -1.29610152504,
                       'r_forearm_roll_joint': -0.0301682323805,
                       'r_gripper_joint': 2.22044604925e-16,
                       'r_shoulder_lift_joint': 1.20324921318,
                       'r_shoulder_pan_joint': -0.73456435706,
                       'r_upper_arm_roll_joint': -0.70790051778,
                       'r_wrist_flex_joint': -0.10001,
                       'r_wrist_roll_joint': 0.258268529825,

                       'l_elbow_flex_joint': -1.29610152504,
                       'l_forearm_roll_joint': 0.0301682323805,
                       'l_gripper_joint': 2.22044604925e-16,
                       'l_shoulder_lift_joint': 1.20324921318,
                       'l_shoulder_pan_joint': 0.73456435706,
                       'l_upper_arm_roll_joint': 0.70790051778,
                       'l_wrist_flex_joint': -0.1001,
                       'l_wrist_roll_joint': -0.258268529825,

                       'torso_lift_joint': 0.2,
                       'head_pan_joint': 0,
                       'head_tilt_joint': 0}

gaya_pose = {'r_shoulder_pan_joint': -1.7125,
             'r_shoulder_lift_joint': -0.25672,
             'r_upper_arm_roll_joint': -1.46335,
             'r_elbow_flex_joint': -2.12216,
             'r_forearm_roll_joint': 1.76632,
             'r_wrist_flex_joint': -0.10001,
             'r_wrist_roll_joint': 0.05106,

             'l_shoulder_pan_joint': 1.9652,
             'l_shoulder_lift_joint': - 0.26499,
             'l_upper_arm_roll_joint': 1.3837,
             'l_elbow_flex_joint': - 2.1224,
             'l_forearm_roll_joint': 16.99,
             'l_wrist_flex_joint': - 0.10001,
             'l_wrist_roll_joint': 0}


class testPythonInterface(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        rospy.init_node('test_collision_avoidance')
        self.giskard = GiskardWrapper()
        self.default_root = 'base_link'
        self.r_tip = 'r_gripper_tool_frame'
        self.l_tip = 'l_gripper_tool_frame'
        self.map = 'map'
        self.simple_base_pose_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        rospy.sleep(0.5)
        super(testPythonInterface, self).__init__(methodName)

    def setUp(self):
        # TODO set joint goal instead of cart
        self.giskard.clear_world()
        self.giskard.set_joint_goal(default_joint_state)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_ALL_COLLISIONS
        self.giskard.set_collision_entries([collision_entry])

        self.set_and_check_js_goal(default_joint_state)
        self.reset_pr2_base()
        super(testPythonInterface, self).setUpClass()

    def move_pr2_base(self, goal_pose):
        self.simple_base_pose_pub.publish(goal_pose)

    def reset_pr2_base(self):
        p = PoseStamped()
        p.header.frame_id = self.map
        p.pose.orientation.w = 1
        self.move_pr2_base(p)

    def get_allow_l_gripper(self, body_b='box'):
        ces = []
        ce = CollisionEntry()
        ce.type = CollisionEntry.ALLOW_COLLISION
        ce.robot_link = 'l_gripper_l_finger_tip_link'
        ce.body_b = body_b
        ces.append(ce)
        ce = CollisionEntry()
        ce.type = CollisionEntry.ALLOW_COLLISION
        ce.robot_link = 'l_gripper_r_finger_tip_link'
        ce.body_b = body_b
        ces.append(ce)
        ce = CollisionEntry()
        ce.type = CollisionEntry.ALLOW_COLLISION
        ce.robot_link = 'l_gripper_l_finger_link'
        ce.body_b = body_b
        ces.append(ce)
        ce = CollisionEntry()
        ce.type = CollisionEntry.ALLOW_COLLISION
        ce.robot_link = 'l_gripper_r_finger_link'
        ce.body_b = body_b
        ces.append(ce)
        ce = CollisionEntry()
        ce.type = CollisionEntry.ALLOW_COLLISION
        ce.robot_link = 'l_gripper_palm_link'
        ce.body_b = body_b
        ces.append(ce)
        return ces

    def set_and_check_js_goal(self, goal_js):
        self.giskard.set_joint_goal(goal_js)
        self.giskard.plan_and_execute()
        result = self.giskard.get_result()
        self.assertEqual(MoveResult.SUCCESS, result.error_code)
        current_joint_state = rospy.wait_for_message('joint_states', JointState)  # type: JointState
        for i, joint_name in enumerate(current_joint_state.name):
            if joint_name in goal_js:
                goal = normalize_angle(goal_js[joint_name])
                current = normalize_angle(current_joint_state.position[i])
                self.assertAlmostEqual(goal, current, 2,
                                       msg='{} is {} instead of {}'.format(joint_name,
                                                                           current,
                                                                           goal))

    def set_and_check_cart_goal(self, root, tip, goal_pose):
        goal_in_base = transform_pose('base_footprint', goal_pose)
        self.giskard.set_cart_goal(root, tip, goal_pose)
        self.giskard.plan_and_execute()
        current_pose = lookup_transform('base_footprint', tip)
        self.assertAlmostEqual(goal_in_base.pose.position.x, current_pose.pose.position.x, 2)
        self.assertAlmostEqual(goal_in_base.pose.position.y, current_pose.pose.position.y, 2)
        self.assertAlmostEqual(goal_in_base.pose.position.z, current_pose.pose.position.z, 2)

        self.assertAlmostEqual(goal_in_base.pose.orientation.x, current_pose.pose.orientation.x, 1)
        self.assertAlmostEqual(goal_in_base.pose.orientation.y, current_pose.pose.orientation.y, 1)
        self.assertAlmostEqual(goal_in_base.pose.orientation.z, current_pose.pose.orientation.z, 1)
        self.assertAlmostEqual(goal_in_base.pose.orientation.w, current_pose.pose.orientation.w, 1)

    def add_box(self, name='box', position=(1.2, 0, 0.5)):
        r = self.giskard.add_box(name=name, position=position)
        self.assertEqual(r.error_codes, UpdateWorldResponse.SUCCESS)

    def add_kitchen(self):
        # p = lookup_transform('world', 'map')
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.orientation.w = 1
        self.giskard.add_urdf('kitchen', rospy.get_param('kitchen_description'), 'kitchen_joint_states', p)
        rospy.sleep(.5)

    def test_AllowCollision1(self):
        self.add_box()
        p = PoseStamped()
        p.header.frame_id = self.r_tip
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        self.giskard.set_cart_goal(self.default_root, self.r_tip, p)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_COLLISION
        collision_entry.body_b = 'box'
        self.giskard.set_collision_entries([collision_entry])

        result = self.giskard.plan_and_execute()
        self.assertEqual(result.error_code, MoveResult.SUCCESS)

    def test_avoid_collision1(self):
        self.add_box()
        p = PoseStamped()
        p.header.frame_id = self.r_tip
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        self.giskard.set_cart_goal(self.default_root, self.r_tip, p)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_COLLISION
        collision_entry.min_dist = 0.05
        collision_entry.body_b = 'box'
        self.giskard.set_collision_entries([collision_entry])

        result = self.giskard.plan_and_execute()
        self.assertEqual(result.error_code, MoveResult.SUCCESS)

    def test_unknown_object1(self):
        self.add_box()
        p = PoseStamped()
        p.header.frame_id = self.r_tip
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        self.giskard.set_cart_goal(self.default_root, self.r_tip, p)

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
        p.header.frame_id = self.r_tip
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        self.giskard.set_cart_goal(self.default_root, self.r_tip, p)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_ALL_COLLISIONS
        collision_entry.min_dist = 0.05
        self.giskard.set_collision_entries([collision_entry])

        result = self.giskard.plan_and_execute()
        self.assertEqual(result.error_code, MoveResult.SUCCESS)

    def test_get_out_of_collision(self):
        self.add_box()
        p = PoseStamped()
        p.header.frame_id = self.r_tip
        p.pose.position = Point(0.15, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        self.giskard.set_cart_goal(self.default_root, self.r_tip, p)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_ALL_COLLISIONS
        collision_entry.min_dist = 0.05
        self.giskard.set_collision_entries([collision_entry])

        result = self.giskard.plan_and_execute()
        self.assertEqual(result.error_code, MoveResult.SUCCESS)

        p = PoseStamped()
        p.header.frame_id = self.r_tip
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        self.giskard.set_cart_goal(self.default_root, self.r_tip, p)

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
        p.header.frame_id = self.r_tip
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        self.giskard.set_cart_goal(self.default_root, self.r_tip, p)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_ALL_COLLISIONS
        collision_entry.min_dist = 0.5
        self.giskard.set_collision_entries([collision_entry])

        self.giskard.plan_and_execute(wait=False)
        rospy.sleep(.5)
        self.giskard.interrupt()
        result = self.giskard.get_result()
        self.assertEqual(result.error_code, MoveResult.INTERRUPTED)

    def test_waypoints(self):
        p = PoseStamped()
        p.header.frame_id = self.r_tip
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        self.giskard.set_cart_goal(self.default_root, self.r_tip, p)

        self.giskard.add_cmd()
        p = PoseStamped()
        p.header.frame_id = self.r_tip
        p.pose.position = Point(0, 0, 0.1)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        self.giskard.set_cart_goal(self.default_root, self.r_tip, p)

        self.giskard.add_cmd()
        p = PoseStamped()
        p.header.frame_id = self.r_tip
        p.pose.position = Point(-0.1, 0, -0.1)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        self.giskard.set_cart_goal(self.default_root, self.r_tip, p)

        self.giskard.plan_and_execute()
        result = self.giskard.get_result()
        self.assertEqual(result.error_code, MoveResult.SUCCESS)

    def test_joint_state_goal1(self):
        goal_js = {'torso_lift_joint': 0.3}
        self.giskard.set_joint_goal(goal_js)
        self.giskard.plan_and_execute()
        current_joint_state = rospy.wait_for_message('joint_states', JointState)  # type: JointState
        for i, joint_name in enumerate(current_joint_state.name):
            if joint_name in goal_js:
                goal = goal_js[joint_name]
                current = current_joint_state.position[i]
                self.assertAlmostEqual(goal, current, 2,
                                       msg='{} is {} instead of {}'.format(joint_name,
                                                                           current,
                                                                           goal))

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

    def test_place_spoon1(self):
        base_pose = PoseStamped()
        base_pose.header.frame_id = 'map'
        base_pose.pose.position = Point(-1.010, -0.152, 0.000)
        base_pose.pose.orientation = Quaternion(0.000, 0.000, -0.707, 0.707)
        self.move_pr2_base(base_pose)

        goal_js = {
            'r_elbow_flex_joint': -1.43322543123,
            'r_forearm_roll_joint': pi / 2,
            'r_gripper_joint': 2.22044604925e-16,
            'r_shoulder_lift_joint': 0,
            'r_shoulder_pan_joint': -1.39478600655,
            'r_upper_arm_roll_joint': -pi / 2,
            'r_wrist_flex_joint': -0.1001,
            'r_wrist_roll_joint': 0,

            'l_elbow_flex_joint': -1.53432386765,
            'l_forearm_roll_joint': -0.335634766956,
            'l_gripper_joint': 2.22044604925e-16,
            'l_shoulder_lift_joint': 0.199493756207,
            'l_shoulder_pan_joint': 0.854317292495,
            'l_upper_arm_roll_joint': 1.90837777308,
            'l_wrist_flex_joint': -0.623267982468,
            'l_wrist_roll_joint': -0.910310693429,

            'torso_lift_joint': 0.206303584043,
        }
        self.set_and_check_js_goal(goal_js)
        self.add_kitchen()
        self.giskard.avoid_collision(0.05, body_b='kitchen')
        p = PoseStamped()
        p.header.frame_id = 'base_footprint'
        p.pose.position = Point(0.69, -0.374, 0.82)
        p.pose.orientation = Quaternion(-0.010, 0.719, 0.006, 0.695)
        self.set_and_check_cart_goal(self.default_root, self.r_tip, p)

    def test_weird_wiggling(self):
        goal_js = {
            'torso_lift_joint': 0.137863459754,

            'r_upper_arm_roll_joint': -3.24008158875,
            'r_shoulder_pan_joint': 0.206949462075,
            'r_shoulder_lift_joint': -0.249453873652,
            'r_forearm_roll_joint': 1.83979114862,
            'r_elbow_flex_joint': -1.36820120012,
            'r_wrist_flex_joint': -1.52492789587,
            'r_wrist_roll_joint': -13.6248743778,

            'l_upper_arm_roll_joint': 1.63487737202,
            'l_shoulder_pan_joint': 1.36222920328,
            'l_shoulder_lift_joint': 0.229120778526,
            'l_forearm_roll_joint': 13.7578920265,
            'l_elbow_flex_joint': -1.48141189643,
            'l_wrist_flex_joint': -1.22662876066,
            'l_wrist_roll_joint': -53.6150824007,
        }
        self.giskard.allow_all_collisions()
        self.set_and_check_js_goal(goal_js)

        p = PoseStamped()
        p.header.frame_id = self.l_tip
        p.pose.position.x = -0.1
        p.pose.orientation.w = 1
        self.giskard.allow_all_collisions()
        self.set_and_check_cart_goal(self.default_root, self.l_tip, p)

        p = PoseStamped()
        p.header.frame_id = self.l_tip
        p.pose.position.x = 0.2
        p.pose.orientation.w = 1
        # self.giskard.allow_all_collisions()
        self.set_and_check_cart_goal(self.default_root, self.l_tip, p)

    def test_root_link_not_equal_chain_root(self):
        p = PoseStamped()
        p.header.frame_id = 'base_footprint'
        p.pose.position.x = 0.8
        p.pose.position.y = -0.5
        p.pose.position.z = 1
        p.pose.orientation.w = 1
        self.giskard.allow_all_collisions()
        self.giskard.set_tranlation_goal('torso_lift_link', self.r_tip, p)
        self.giskard.plan_and_execute()

    def test_did_not_reach_joint_state_with_launch_file(self):
        self.giskard.allow_all_collisions()
        self.set_and_check_js_goal(gaya_pose)

        goal_js = {
            'r_shoulder_pan_joint': 0,
            'r_shoulder_lift_joint': 0,
            'r_upper_arm_roll_joint': 0,
            'r_elbow_flex_joint': -0.1508,
            'r_forearm_roll_joint': 0,
            'r_wrist_flex_joint': -0.10001,
            'r_wrist_roll_joint': 0,

            'l_shoulder_pan_joint': 0,
            'l_shoulder_lift_joint': 0,
            'l_upper_arm_roll_joint': 0,
            'l_elbow_flex_joint': -0.1508,
            'l_forearm_roll_joint': 0,
            'l_wrist_flex_joint': -0.10001,
            'l_wrist_roll_joint': 0,
        }
        self.set_and_check_js_goal(goal_js)

    def test_pick_up_spoon(self):
        base_pose = PoseStamped()
        base_pose.header.frame_id = 'map'
        base_pose.pose.position = Point(0.365, 0.368, 0.000)
        base_pose.pose.orientation = Quaternion(0.000, 0.000, -0.007, 1.000)
        self.move_pr2_base(base_pose)

        self.giskard.allow_all_collisions()
        self.set_and_check_js_goal(gaya_pose)

        self.add_kitchen()
        kitchen_js = {'sink_area_left_upper_drawer_main_joint': 0.45}
        self.giskard.set_object_joint_state('kitchen', kitchen_js)
        rospy.sleep(.5)

        # put gripper above drawer
        pick_spoon_pose = PoseStamped()
        pick_spoon_pose.header.frame_id = 'base_footprint'
        pick_spoon_pose.pose.position = Point(0.567, 0.498, 0.89)
        pick_spoon_pose.pose.orientation = Quaternion(0.018, 0.702, 0.004, 0.712)
        self.set_and_check_cart_goal(self.default_root, self.l_tip, pick_spoon_pose)

        #put gripper in drawer
        self.giskard.set_collision_entries(self.get_allow_l_gripper('kitchen'))
        p = PoseStamped()
        p.header.frame_id = self.l_tip
        p.pose.position.x = 0.1
        p.pose.orientation.w = 1
        self.set_and_check_cart_goal(self.default_root, self.l_tip, p)

        #attach spoon
        r = self.giskard.attach_box('pocky', [0.02, 0.02, 0.1], self.l_tip, [0, 0, 0])
        self.assertEqual(r.error_codes, UpdateWorldResponse.SUCCESS)

        # allow grippe and spoon
        ces = self.get_allow_l_gripper('kitchen')
        ce = CollisionEntry()
        ce.type = CollisionEntry.ALLOW_COLLISION
        ce.robot_link = 'pocky'
        ce.body_b = 'kitchen'
        ces.append(ce)
        self.giskard.set_collision_entries(ces)

        # pick up
        p = PoseStamped()
        p.header.frame_id = self.l_tip
        p.pose.position.x = -0.1
        p.pose.orientation.w = 1
        self.set_and_check_cart_goal(self.default_root, self.l_tip, p)


    def test_base_link_in_collision(self):
        self.add_box(position=[0.5,0.5,-0.2])
        self.set_and_check_js_goal(gaya_pose)

    def test_allow_collision2(self):
        self.add_box()
        ces = self.get_allow_l_gripper('box')
        self.giskard.set_collision_entries(ces)
        p = PoseStamped()
        p.header.frame_id = self.l_tip
        p.pose.position.x = 0.11
        p.pose.orientation.w = 1
        self.set_and_check_cart_goal(self.default_root, self.l_tip, p)


    def test_attached_collision1(self):
        self.add_box()
        r = self.giskard.attach_box('pocky', [0.1, 0.02, 0.02], self.r_tip, [0.05,0,0])
        self.assertEqual(r.error_codes, UpdateWorldResponse.SUCCESS)
        r = self.giskard.attach_box('pocky', [0.1, 0.02, 0.02], self.r_tip, [0.05,0,0])
        self.assertEqual(r.error_codes, UpdateWorldResponse.DUPLICATE_BODY_ERROR)
        # self.giskard.attach_box('pocky2', [0.1, 0.02, 0.02], self.l_tip, [0.05,0,0])
        p = PoseStamped()
        p.header.frame_id = self.r_tip
        p.pose.position.x = -0.11
        p.pose.orientation.w = 1
        self.set_and_check_cart_goal(self.default_root, self.r_tip, p)
        r = self.giskard.remove_object('pocky')
        self.assertEqual(r.error_codes, UpdateWorldResponse.SUCCESS)

    def test_attached_collision2(self):
        pocky = 'http://muh#pocky'
        self.add_box()
        self.add_box(pocky, position=[1.2,0,1.6])
        r = self.giskard.attach_box(pocky, [0.1, 0.02, 0.02], self.r_tip, [0.05,0,0])
        self.assertEqual(r.error_codes, UpdateWorldResponse.DUPLICATE_BODY_ERROR)
        r = self.giskard.remove_object(pocky)
        self.assertEqual(r.error_codes, UpdateWorldResponse.SUCCESS)
        r = self.giskard.attach_box(pocky, [0.1, 0.02, 0.02], self.r_tip, [0.05,0,0])
        self.assertEqual(r.error_codes, UpdateWorldResponse.SUCCESS)
        p = PoseStamped()
        p.header.frame_id = self.r_tip
        p.pose.position.x = -0.11
        p.pose.orientation.w = 1
        self.set_and_check_cart_goal(self.default_root, self.r_tip, p)
        r = self.giskard.remove_object(pocky)
        self.assertEqual(r.error_codes, UpdateWorldResponse.SUCCESS)

    def test_attached_collision3(self):
        pocky = 'http://muh#pocky'
        self.add_box()
        r = self.giskard.attach_box(pocky, [0.1, 0.02, 0.02], self.r_tip, [0.05,0,0])
        self.assertEqual(r.error_codes, UpdateWorldResponse.SUCCESS)

        ces = []
        ce = CollisionEntry()
        ce.type = CollisionEntry.ALLOW_COLLISION
        ce.robot_link = 'pocky'
        ce.body_b = 'box'
        ces.append(ce)
        self.giskard.set_collision_entries(ces)

        p = PoseStamped()
        p.header.frame_id = self.r_tip
        p.pose.position.x = -0.11
        p.pose.orientation.w = 1
        self.set_and_check_cart_goal(self.default_root, self.r_tip, p)
        r = self.giskard.remove_object(pocky)
        self.assertEqual(r.error_codes, UpdateWorldResponse.SUCCESS)



if __name__ == '__main__':
    import rosunit

    rosunit.unitrun(package=PKG,
                    test_name='TestCollisionAvoidance',
                    test=testPythonInterface)
