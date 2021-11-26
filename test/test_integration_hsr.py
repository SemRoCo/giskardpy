from copy import deepcopy

import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from numpy import pi
from tf.transformations import quaternion_from_matrix, quaternion_about_axis

import giskardpy.utils.tfwrapper as tf
from utils_for_tests import PR2, HSR, compare_poses


@pytest.fixture(scope=u'module')
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
    p.header.frame_id = u'map'
    p.pose.position.x = 1.2
    p.pose.position.y = 0
    p.pose.position.z = 0.1
    p.pose.orientation.w = 1
    zero_pose.add_box(name='box', size=[1, 1, 1], pose=p)
    return zero_pose


class TestJointGoals(object):
    def test_move_base(self, zero_pose):
        p = PoseStamped()
        p.header.frame_id = u'map'
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
            u'arm_flex_joint': 0.0,
            u'arm_lift_joint': 0.0,
            u'arm_roll_joint': -1.52,
            u'head_pan_joint': -0.09,
            u'head_tilt_joint': -0.62,
            u'wrist_flex_joint': -1.55,
            u'wrist_roll_joint': 0.11,
        }
        zero_pose.set_joint_goal(js)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = u'hand_palm_link'
        goal_pose.pose.position.x = 0.5
        goal_pose.pose.orientation.w = 1
        zero_pose.set_cart_goal(goal_pose, zero_pose.tip)
        zero_pose.plan_and_execute()

    def test_attached_collision1(self, box_setup):
        """
        :type box_setup: HSR
        """
        box_name = u'asdf'
        box_pose = PoseStamped()
        box_pose.header.frame_id = u'map'
        box_pose.pose.position = Point(0.85, 0.3, .66)
        box_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        box_setup.add_box(box_name, [0.07, 0.04, 0.1], box_pose)
        box_setup.open_gripper()

        grasp_pose = deepcopy(box_pose)
        grasp_pose.pose.position.x -= 0.05
        grasp_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 0, 1, 0],
                                                                          [0, -1, 0, 0],
                                                                          [1, 0, 0, 0],
                                                                          [0, 0, 0, 1]]))
        box_setup.set_cart_goal(grasp_pose, box_setup.tip)
        box_setup.plan_and_execute()
        box_setup.attach_object(box_name, box_setup.tip)

        base_goal = PoseStamped()
        base_goal.header.frame_id = box_setup.default_root
        base_goal.pose.position.x -= 0.5
        base_goal.pose.orientation.w = 1
        box_setup.move_base(base_goal)

    def test_collision_avoidance(self, zero_pose):
        """
        :type zero_pose: HSR
        """
        js = {u'arm_flex_joint': -np.pi / 2}
        zero_pose.set_joint_goal(js)
        zero_pose.plan_and_execute()

        p = PoseStamped()
        p.header.frame_id = u'map'
        p.pose.position.x = 0.9
        p.pose.position.y = 0
        p.pose.position.z = 0.5
        p.pose.orientation.w = 1
        zero_pose.add_box(name='box', size=[1, 1, 0.01], pose=p)

        js = {u'arm_flex_joint': 0}
        zero_pose.set_joint_goal(js, check=False)
        zero_pose.plan_and_execute()
