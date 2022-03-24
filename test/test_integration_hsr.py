from copy import deepcopy

import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped, Point, Quaternion, PointStamped, Vector3Stamped
from numpy import pi
from tf.transformations import quaternion_from_matrix, quaternion_about_axis

import giskardpy.utils.tfwrapper as tf
from utils_for_tests import PR2, HSR, compare_poses


@pytest.fixture(scope='module')
def giskard(request, ros):
    c = HSR()
    request.addfinalizer(c.tear_down)
    return c


@pytest.fixture()
def box_setup(zero_pose):
    """
    :type pocky_pose_setup: PR2
    :rtype: PR2
    """
    p = PoseStamped()
    p.header.frame_id = 'map'
    p.pose.position.x = 1.2
    p.pose.position.y = 0
    p.pose.position.z = 0.1
    p.pose.orientation.w = 1
    zero_pose.add_box(name='box', size=[1, 1, 1], pose=p)
    return zero_pose


class TestJointGoals(object):
    def test_move_base(self, zero_pose):
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position.y = -0.3
        p.pose.orientation = Quaternion(0, 0, 0.47942554, 0.87758256)
        zero_pose.move_base(p)

    def test_mimic_joints(self, zero_pose):
        """
        :type zero_pose: HSR
        """
        zero_pose.open_gripper()
        hand_T_finger_current = zero_pose.world.compute_fk_pose('hand_palm_link', 'hand_l_distal_link')
        hand_T_finger_expected = tf.lookup_pose('hand_palm_link', 'hand_l_distal_link')
        compare_poses(hand_T_finger_current.pose, hand_T_finger_expected.pose)

        js = {'torso_lift_joint': 0.1}
        zero_pose.set_joint_goal(js, check=False)
        zero_pose.plan_and_execute()
        np.testing.assert_almost_equal(zero_pose.world.state['arm_lift_joint'].position, 0.2, decimal=2)
        base_T_torso = tf.lookup_pose('base_footprint', 'torso_lift_link')
        base_T_torso2 = zero_pose.world.compute_fk_pose('base_footprint', 'torso_lift_link')
        compare_poses(base_T_torso2.pose, base_T_torso.pose)

    def test_mimic_joints2(self, zero_pose):
        """
        :type zero_pose: HSR
        """
        zero_pose.open_gripper()
        hand_T_finger_current = zero_pose.world.compute_fk_pose('hand_palm_link', 'hand_l_distal_link')
        hand_T_finger_expected = tf.lookup_pose('hand_palm_link', 'hand_l_distal_link')
        compare_poses(hand_T_finger_current.pose, hand_T_finger_expected.pose)

        tip = 'hand_gripper_tool_frame'
        p = PoseStamped()
        p.header.frame_id = tip
        p.pose.position.z = 0.2
        p.pose.orientation.w = 1
        zero_pose.set_cart_goal(goal_pose=p, tip_link=tip,
                                root_link='base_footprint')
        zero_pose.plan_and_execute()
        np.testing.assert_almost_equal(zero_pose.world.state['arm_lift_joint'].position, 0.2, decimal=2)
        base_T_torso = tf.lookup_pose('base_footprint', 'torso_lift_link')
        base_T_torso2 = zero_pose.world.compute_fk_pose('base_footprint', 'torso_lift_link')
        compare_poses(base_T_torso2.pose, base_T_torso.pose)

    def test_mimic_joints3(self, zero_pose):
        """
        :type zero_pose: HSR
        """
        zero_pose.open_gripper()
        hand_T_finger_current = zero_pose.world.compute_fk_pose('hand_palm_link', 'hand_l_distal_link')
        hand_T_finger_expected = tf.lookup_pose('hand_palm_link', 'hand_l_distal_link')
        compare_poses(hand_T_finger_current.pose, hand_T_finger_expected.pose)

        tip = 'head_pan_link'
        p = PoseStamped()
        p.header.frame_id = tip
        p.pose.position.z = 0.15
        p.pose.orientation.w = 1
        zero_pose.set_cart_goal(goal_pose=p, tip_link=tip,
                                root_link='base_footprint')
        zero_pose.plan_and_execute()
        np.testing.assert_almost_equal(zero_pose.world.state['arm_lift_joint'].position, 0.3, decimal=2)
        base_T_torso = tf.lookup_pose('base_footprint', 'torso_lift_link')
        base_T_torso2 = zero_pose.world.compute_fk_pose('base_footprint', 'torso_lift_link')
        compare_poses(base_T_torso2.pose, base_T_torso.pose)


class TestCartGoals(object):

    def test_rotate_gripper(self, zero_pose):
        """
        :type zero_pose: HSR
        """
        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.tip
        r_goal.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        zero_pose.set_cart_goal(r_goal, zero_pose.tip)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()


class TestConstraints:

    def test_open_fridge(self, kitchen_setup):
        """
        :type kitchen_setup: HSR
        """
        handle_frame_id = 'iai_kitchen/iai_fridge_door_handle'
        handle_name = 'iai_fridge_door_handle'
        kitchen_setup.open_gripper()
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position = Point(0.3, -0.5, 0)
        base_goal.pose.orientation.w = 1
        kitchen_setup.move_base(base_goal)

        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = handle_frame_id
        bar_axis.vector.z = 1

        bar_center = PointStamped()
        bar_center.header.frame_id = handle_frame_id

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = kitchen_setup.tip
        tip_grasp_axis.vector.x = 1

        kitchen_setup.set_json_goal('GraspBar',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=kitchen_setup.tip,
                                    tip_grasp_axis=tip_grasp_axis,
                                    bar_center=bar_center,
                                    bar_axis=bar_axis,
                                    bar_length=.4)
        x_gripper = Vector3Stamped()
        x_gripper.header.frame_id = kitchen_setup.tip
        x_gripper.vector.z = 1

        x_goal = Vector3Stamped()
        x_goal.header.frame_id = handle_frame_id
        x_goal.vector.x = -1
        kitchen_setup.set_align_planes_goal(kitchen_setup.tip, x_gripper, root_normal=x_goal)
        kitchen_setup.allow_all_collisions()
        # kitchen_setup.add_json_goal('AvoidJointLimits', percentage=10)
        kitchen_setup.plan_and_execute()

        kitchen_setup.set_json_goal('Open',
                                    tip_link=kitchen_setup.tip,
                                    environment_link=handle_name,
                                    goal_joint_state=1.5)
        # kitchen_setup.set_json_goal('AvoidJointLimits', percentage=40)
        kitchen_setup.allow_all_collisions()
        # kitchen_setup.add_json_goal('AvoidJointLimits')
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_kitchen_js({'iai_fridge_door_joint': 1.5})

        kitchen_setup.set_json_goal('Open',
                                    tip_link=kitchen_setup.tip,
                                    environment_link=handle_name,
                                    goal_joint_state=0)
        kitchen_setup.allow_all_collisions()
        # kitchen_setup.set_json_goal('AvoidJointLimits', percentage=40)
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_kitchen_js({'iai_fridge_door_joint': 0})

        kitchen_setup.plan_and_execute()

        kitchen_setup.set_joint_goal(kitchen_setup.better_pose)
        kitchen_setup.plan_and_execute()


class TestCollisionAvoidanceGoals(object):

    def test_self_collision_avoidance(self, zero_pose):
        """
        :type zero_pose: HSR
        """
        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.tip
        r_goal.pose.position.z = 0.5
        r_goal.pose.orientation.w = 1
        zero_pose.set_cart_goal(r_goal, zero_pose.tip)
        zero_pose.plan_and_execute()

    def test_self_collision_avoidance2(self, zero_pose):
        """
        :type zero_pose: HSR
        """
        js = {
            'arm_flex_joint': 0.0,
            'arm_lift_joint': 0.0,
            'arm_roll_joint': -1.52,
            'head_pan_joint': -0.09,
            'head_tilt_joint': -0.62,
            'wrist_flex_joint': -1.55,
            'wrist_roll_joint': 0.11,
        }
        zero_pose.set_joint_goal(js)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'hand_palm_link'
        goal_pose.pose.position.x = 0.5
        goal_pose.pose.orientation.w = 1
        zero_pose.set_cart_goal(goal_pose, zero_pose.tip)
        zero_pose.plan_and_execute()

    def test_attached_collision1(self, box_setup):
        """
        :type box_setup: HSR
        """
        box_name = 'asdf'
        box_pose = PoseStamped()
        box_pose.header.frame_id = 'map'
        box_pose.pose.position = Point(0.85, 0.3, .66)
        box_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        box_setup.add_box(box_name, [0.07, 0.04, 0.1], box_pose)
        box_setup.open_gripper()

        grasp_pose = deepcopy(box_pose)
        # grasp_pose.pose.position.x -= 0.05
        grasp_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 0, 1, 0],
                                                                          [0, -1, 0, 0],
                                                                          [1, 0, 0, 0],
                                                                          [0, 0, 0, 1]]))
        box_setup.set_cart_goal(grasp_pose, box_setup.tip)
        box_setup.plan_and_execute()
        box_setup.reattach_object(box_name, box_setup.tip)

        base_goal = PoseStamped()
        base_goal.header.frame_id = box_setup.default_root
        base_goal.pose.position.x -= 0.5
        base_goal.pose.orientation.w = 1
        box_setup.move_base(base_goal)

    def test_collision_avoidance(self, zero_pose):
        """
        :type zero_pose: HSR
        """
        js = {'arm_flex_joint': -np.pi / 2}
        zero_pose.set_joint_goal(js)
        zero_pose.plan_and_execute()

        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position.x = 0.9
        p.pose.position.y = 0
        p.pose.position.z = 0.5
        p.pose.orientation.w = 1
        zero_pose.add_box(name='box', size=[1, 1, 0.01], pose=p)

        js = {'arm_flex_joint': 0}
        zero_pose.set_joint_goal(js, check=False)
        zero_pose.plan_and_execute()
