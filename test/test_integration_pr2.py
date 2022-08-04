from __future__ import division

import itertools
import re
from copy import deepcopy

import numpy as np
import pytest
import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Vector3Stamped, PointStamped
from numpy import pi
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from tf.transformations import quaternion_from_matrix, quaternion_about_axis, quaternion_from_euler

import giskardpy.utils.tfwrapper as tf
from giskard_msgs.msg import CollisionEntry, MoveResult, WorldBody, MoveGoal
from giskard_msgs.srv import UpdateWorldResponse, UpdateWorldRequest
from giskardpy import identifier
from giskardpy.goals.goal import WEIGHT_ABOVE_CA, WEIGHT_BELOW_CA, WEIGHT_COLLISION_AVOIDANCE
from giskardpy.identifier import fk_pose
from utils_for_tests import PR2, compare_poses, compare_points, compare_orientations, publish_marker_vector, \
    JointGoalChecker

# TODO roslaunch iai_pr2_sim ros_control_sim_with_base.launch
# TODO roslaunch iai_kitchen upload_kitchen_obj.launch

# scopes = ['module', 'class', 'function']
pocky_pose = {'r_elbow_flex_joint': -1.29610152504,
              'r_forearm_roll_joint': -0.0301682323805,
              'r_shoulder_lift_joint': 1.20324921318,
              'r_shoulder_pan_joint': -0.73456435706,
              'r_upper_arm_roll_joint': -0.70790051778,
              'r_wrist_flex_joint': -0.10001,
              'r_wrist_roll_joint': 0.258268529825,

              'l_elbow_flex_joint': -1.29610152504,
              'l_forearm_roll_joint': 0.0301682323805,
              'l_shoulder_lift_joint': 1.20324921318,
              'l_shoulder_pan_joint': 0.73456435706,
              'l_upper_arm_roll_joint': 0.70790051778,
              'l_wrist_flex_joint': -0.1001,
              'l_wrist_roll_joint': -0.258268529825,

              'torso_lift_joint': 0.2,
              'head_pan_joint': 0,
              'head_tilt_joint': 0,
              }

pick_up_pose = {
    'head_pan_joint': -2.46056758502e-16,
    'head_tilt_joint': -1.97371778181e-16,
    'l_elbow_flex_joint': -0.962150355946,
    'l_forearm_roll_joint': 1.44894622393,
    'l_shoulder_lift_joint': -0.273579583084,
    'l_shoulder_pan_joint': 0.0695426768038,
    'l_upper_arm_roll_joint': 1.3591238067,
    'l_wrist_flex_joint': -1.9004529902,
    'l_wrist_roll_joint': 2.23732576003,
    'r_elbow_flex_joint': -2.1207193579,
    'r_forearm_roll_joint': 1.76628402882,
    'r_shoulder_lift_joint': -0.256729037039,
    'r_shoulder_pan_joint': -1.71258744959,
    'r_upper_arm_roll_joint': -1.46335011257,
    'r_wrist_flex_joint': -0.100010762609,
    'r_wrist_roll_joint': 0.0509923457388,
    'torso_lift_joint': 0.261791330751,
}

oven_area_cereal = {
    'odom_x_joint': 0.5940842695605993,
    'odom_y_joint': 0.5646523972590731,
    'odom_z_joint': 1.2424133925817196,
    'fl_caster_rotation_joint': 0.0,
    'fr_caster_rotation_joint': 0.0,
    'bl_caster_rotation_joint': 0.0,
    'br_caster_rotation_joint': 0.0,
    'torso_lift_joint': 0.22617875616830663,
    'torso_lift_motor_screw_joint': 0.0,
    'fl_caster_l_wheel_joint': 0.0,
    'fl_caster_r_wheel_joint': 0.0,
    'fr_caster_l_wheel_joint': 0.0,
    'fr_caster_r_wheel_joint': 0.0,
    'bl_caster_l_wheel_joint': 0.0,
    'bl_caster_r_wheel_joint': 0.0,
    'br_caster_l_wheel_joint': 0.0,
    'br_caster_r_wheel_joint': 0.0,
    'head_pan_joint': 0.0,
    'laser_tilt_mount_joint': 0.0,
    'r_shoulder_pan_joint': -0.24293873201958266,
    'l_shoulder_pan_joint': 1.561020003858116,
    'head_tilt_joint': 0.0,
    'r_shoulder_lift_joint': -0.013819636563032563,
    'r_upper_arm_roll_joint': -1.4108187392665519,
    'r_elbow_flex_joint': -0.5344932623724951,
    'r_forearm_roll_joint': -0.29683611261924375,
    'r_wrist_flex_joint': -0.4680856109600189,
    'r_wrist_roll_joint': 1.7792377315663064,
    'r_gripper_motor_slider_joint': 0.0,
    'r_gripper_l_finger_joint': 0.54,
    'r_gripper_r_finger_joint': 0.54,
    'r_gripper_motor_screw_joint': 0.0,
    'r_gripper_l_finger_tip_joint': 0.54,
    'r_gripper_r_finger_tip_joint': 0.54,
    'r_gripper_joint': 2.220446049250313e-16,
    'l_shoulder_lift_joint': 0.015148469507495575,
    'l_upper_arm_roll_joint': 1.3837000000005018,
    'l_elbow_flex_joint': -1.681037408201828,
    'l_forearm_roll_joint': -1.8595559215384385,
    'l_wrist_flex_joint': -0.5217665869722147,
    'l_wrist_roll_joint': 0.0,
    'l_gripper_motor_slider_joint': 0.0,
    'l_gripper_l_finger_joint': 0.54,
    'l_gripper_r_finger_joint': 0.54,
    'l_gripper_motor_screw_joint': 0.0,
    'l_gripper_l_finger_tip_joint': 0.54,
    'l_gripper_r_finger_tip_joint': 0.54,
    'l_gripper_joint': 2.220446049250313e-16
}

folder_name = 'tmp_data/'


@pytest.fixture(scope='module')
def giskard(request, ros):
    c = PR2()
    request.addfinalizer(c.tear_down)
    return c


@pytest.fixture()
def pocky_pose_setup(resetted_giskard: PR2) -> PR2:
    resetted_giskard.set_joint_goal(pocky_pose)
    resetted_giskard.allow_all_collisions()
    resetted_giskard.plan_and_execute()
    return resetted_giskard


@pytest.fixture()
def box_setup(pocky_pose_setup: PR2) -> PR2:
    p = PoseStamped()
    p.header.frame_id = 'map'
    p.pose.position.x = 1.2
    p.pose.position.y = 0
    p.pose.position.z = 0.5
    p.pose.orientation.w = 1
    pocky_pose_setup.add_box(name='box', size=(1, 1, 1), pose=p)
    return pocky_pose_setup


@pytest.fixture()
def fake_table_setup(pocky_pose_setup: PR2) -> PR2:
    p = PoseStamped()
    p.header.frame_id = 'map'
    p.pose.position.x = 1.2
    p.pose.position.y = 0
    p.pose.position.z = 0.3
    p.pose.orientation.w = 1
    pocky_pose_setup.add_box(name='box', size=(1, 1, 1), pose=p)
    return pocky_pose_setup


@pytest.fixture()
def kitchen_setup(kitchen_setup):
    """
    :type resetted_giskard: GiskardTestWrapper
    :return:
    """
    resetted_giskard.avoid_all_collisions(distance=0.0)
    resetted_giskard.set_joint_goal(gaya_pose)
    resetted_giskard.plan_and_execute()
    object_name = u'kitchen'
    resetted_giskard.add_urdf(object_name, rospy.get_param(u'kitchen_description'),
                              tf.lookup_pose(u'map', u'iai_kitchen/world'), u'/kitchen/joint_states',
                              set_js_topic=u'/kitchen/cram_joint_states')
    js = {str(k): 0.0 for k in resetted_giskard.world.groups[object_name].movable_joints}
    resetted_giskard.set_kitchen_js(js)
    return resetted_giskard


@pytest.fixture()
def refills_lab(resetted_giskard):
    """
    :type resetted_giskard: GiskardTestWrapper
    :return:
    """
    resetted_giskard.avoid_all_collisions(distance=0.0)
    resetted_giskard.set_joint_goal(gaya_pose)
    resetted_giskard.plan_and_execute()
    object_name = u'kitchen'
    resetted_giskard.add_urdf(object_name, rospy.get_param(u'kitchen_description'),
                              tf.lookup_pose(u'map', u'iai_kitchen/room_link'), u'/kitchen/joint_states',
                              set_js_topic=u'/kitchen/cram_joint_states')
    return resetted_giskard


class TestFk(object):
    def test_fk(self, zero_pose: PR2):
        for root, tip in itertools.product(zero_pose.robot.link_names, repeat=2):
            fk1 = zero_pose.god_map.get_data(fk_pose + [(root, tip)])
            fk2 = tf.lookup_pose(str(root), str(tip))
            compare_poses(fk1.pose, fk2.pose)

    def test_fk_attached(self, zero_pose: PR2):
        pocky = 'box'
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.position.x = 0.05
        p.pose.orientation.x = 1
        zero_pose.add_box(pocky, size=(0.1, 0.02, 0.02), parent_link=zero_pose.r_tip, pose=p)
        for root, tip in itertools.product(zero_pose.robot.link_names, [pocky]):
            fk1 = zero_pose.god_map.get_data(fk_pose + [(root, tip)])
            fk2 = tf.lookup_pose(str(root), str(tip))
            compare_poses(fk1.pose, fk2.pose)


class TestJointGoals(object):

    # @pytest.mark.repeat(3)
    def test_joint_movement1(self, zero_pose: PR2):
        zero_pose.allow_all_collisions()
        zero_pose.set_joint_goal(pocky_pose)
        zero_pose.plan_and_execute()

    def test_partial_joint_state_goal1(self, zero_pose: PR2):
        zero_pose.allow_self_collision()
        js = dict(list(pocky_pose.items())[:3])
        zero_pose.set_joint_goal(js)
        zero_pose.plan_and_execute()

    def test_continuous_joint1(self, zero_pose: PR2):
        zero_pose.allow_self_collision()
        # zero_pose.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
        js = {'r_wrist_roll_joint': -pi,
              'l_wrist_roll_joint': -2.1 * pi, }
        zero_pose.set_joint_goal(js)
        zero_pose.plan_and_execute()

    def test_prismatic_joint1(self, zero_pose: PR2):
        zero_pose.allow_self_collision()
        js = {'torso_lift_joint': 0.1}
        zero_pose.set_joint_goal(js)
        zero_pose.plan_and_execute()

    def test_hard_joint_limits(self, zero_pose: PR2):
        zero_pose.allow_self_collision()
        r_elbow_flex_joint_limits = zero_pose.robot.get_joint_position_limits('r_elbow_flex_joint')
        torso_lift_joint_limits = zero_pose.robot.get_joint_position_limits('torso_lift_joint')
        head_pan_joint_limits = zero_pose.robot.get_joint_position_limits('head_pan_joint')

        goal_js = {'r_elbow_flex_joint': r_elbow_flex_joint_limits[0] - 0.2,
                   'torso_lift_joint': torso_lift_joint_limits[0] - 0.2,
                   'head_pan_joint': head_pan_joint_limits[0] - 0.2}
        zero_pose.set_joint_goal(goal_js, check=False)
        zero_pose.plan_and_execute()
        js = {u'torso_lift_joint': 0.32}
        zero_pose.send_and_check_joint_goal(js)

        goal_js = {'r_elbow_flex_joint': r_elbow_flex_joint_limits[1] + 0.2,
                   'torso_lift_joint': torso_lift_joint_limits[1] + 0.2,
                   'head_pan_joint': head_pan_joint_limits[1] + 0.2}

        zero_pose.set_joint_goal(goal_js, check=False)
        zero_pose.plan_and_execute()

    # TODO test goal for unknown joint


class TestConstraints(object):
    # TODO write buggy constraints that test sanity checks

    def test_SetPredictionHorizon(self, zero_pose: PR2):
        zero_pose.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
        zero_pose.set_joint_goal(zero_pose.better_pose)
        zero_pose.plan_and_execute()
        zero_pose.set_joint_goal(zero_pose.default_pose)
        zero_pose.plan_and_execute()

    def test_JointPositionRange(self, zero_pose: PR2):
        # FIXME needs to be implemented like other position limits, or override limits
        joint_name = 'head_pan_joint'
        lower_limit, upper_limit = zero_pose.robot.joints[joint_name].position_limits
        lower_limit *= 0.5
        upper_limit *= 0.5
        zero_pose.set_joint_goal({
            joint_name: 2
        }, check=False)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()
        zero_pose.set_json_goal('JointPositionRange',
                                joint_name=joint_name,
                                upper_limit=upper_limit,
                                lower_limit=lower_limit)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()
        assert zero_pose.robot.state[joint_name].position <= upper_limit + 3e-3
        assert zero_pose.robot.state[joint_name].position >= lower_limit - 3e-3

        zero_pose.set_json_goal('JointPositionRange',
                                joint_name=joint_name,
                                upper_limit=upper_limit,
                                lower_limit=lower_limit)
        zero_pose.set_joint_goal({
            joint_name: -0.5
        }, check=False)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()
        assert zero_pose.robot.state[joint_name].position <= upper_limit
        assert zero_pose.robot.state[joint_name].position >= lower_limit

        # zero_pose.set_json_goal('JointPositionRange',
        #                         joint_name=joint_name,
        #                         upper_limit=10,
        #                         lower_limit=9,
        #                         hard=True)
        # zero_pose.set_joint_goal({
        #     joint_name: 0
        # }, check=False)
        # zero_pose.allow_all_collisions()
        # zero_pose.plan_and_execute(expected_error_codes=[MoveResult.CONSTRAINT_INITIALIZATION_ERROR])

    def test_CollisionAvoidanceHint(self, kitchen_setup: PR2):
        # FIXME bouncy
        tip = 'base_footprint'
        base_pose = PoseStamped()
        base_pose.header.frame_id = 'map'
        base_pose.pose.position.x = 0
        base_pose.pose.position.y = 1.5
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0, 0, 1]))
        kitchen_setup.teleport_base(base_pose)
        base_pose = PoseStamped()
        base_pose.header.frame_id = tip
        base_pose.pose.position.x = 2.3
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))

        avoidance_hint = Vector3Stamped()
        avoidance_hint.header.frame_id = 'map'
        avoidance_hint.vector.y = -1
        kitchen_setup.avoid_all_collisions(0.1)
        kitchen_setup.set_json_goal('CollisionAvoidanceHint',
                                    tip_link='base_link',
                                    max_threshold=0.4,
                                    spring_threshold=0.5,
                                    # max_linear_velocity=1,
                                    object_link_name='kitchen_island',
                                    weight=WEIGHT_COLLISION_AVOIDANCE,
                                    avoidance_hint=avoidance_hint)
        kitchen_setup.set_joint_goal(kitchen_setup.better_pose)

        kitchen_setup.set_cart_goal(base_pose, tip, weight=WEIGHT_BELOW_CA, linear_velocity=0.5)
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()

    def test_CartesianPosition(self, zero_pose: PR2):
        tip = zero_pose.r_tip
        p = PointStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = tip
        p.point = Point(-0.4, -0.2, -0.3)

        expected = tf.transform_point('map', p)

        zero_pose.allow_all_collisions()
        zero_pose.set_json_goal('CartesianPosition',
                                root_link=zero_pose.default_root,
                                tip_link=tip,
                                goal_point=p)
        zero_pose.plan_and_execute()
        new_pose = tf.lookup_pose('map', tip)
        compare_points(expected.point, new_pose.pose.position)

    def test_CartesianPose(self, zero_pose: PR2):
        tip = zero_pose.r_tip
        p = PoseStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = tip
        p.pose.position = Point(-0.4, -0.2, -0.3)
        p.pose.orientation = Quaternion(0, 0, 1, 0)

        expected = tf.transform_pose('map', p)

        zero_pose.allow_all_collisions()
        zero_pose.set_json_goal('CartesianPose',
                                root_link=zero_pose.default_root,
                                tip_link=tip,
                                goal_pose=p)
        zero_pose.plan_and_execute()
        new_pose = tf.lookup_pose('map', tip)
        compare_points(expected.pose.position, new_pose.pose.position)

    def test_JointPositionRevolute(self, zero_pose: PR2):
        joint = 'r_shoulder_lift_joint'
        joint_goal = 1
        zero_pose.allow_all_collisions()
        zero_pose.set_json_goal('JointPositionRevolute',
                                joint_name=joint,
                                goal=joint_goal,
                                max_velocity=0.5)
        zero_pose.plan_and_execute()
        np.testing.assert_almost_equal(zero_pose.robot.state[joint].position, joint_goal, decimal=3)

    def test_JointPositionContinuous(self, zero_pose: PR2):
        joint = 'odom_z_joint'
        joint_goal = 4
        zero_pose.allow_all_collisions()
        zero_pose.set_json_goal('JointPositionContinuous',
                                joint_name=joint,
                                goal=joint_goal,
                                max_velocity=1)
        zero_pose.plan_and_execute()
        np.testing.assert_almost_equal(zero_pose.robot.state[joint].position, -2.283, decimal=2)

    def test_JointPosition_kitchen(self, kitchen_setup: PR2):
        joint_name1 = 'iai_fridge_door_joint'
        joint_name2 = 'sink_area_left_upper_drawer_main_joint'
        joint_goal = 0.4
        kitchen_setup.allow_all_collisions()
        kitchen_setup.set_json_goal('JointPosition',
                                    joint_name=joint_name1,
                                    goal=joint_goal,
                                    max_velocity=1)
        kitchen_setup.set_json_goal('JointPosition',
                                    joint_name=joint_name2,
                                    goal=joint_goal,
                                    max_velocity=1)
        kitchen_setup.plan_and_execute()
        np.testing.assert_almost_equal(
            kitchen_setup.god_map.get_data(identifier.trajectory).get_last()[joint_name1].position,
            joint_goal, decimal=2)
        np.testing.assert_almost_equal(
            kitchen_setup.god_map.get_data(identifier.trajectory).get_last()[joint_name2].position,
            joint_goal, decimal=2)

    def test_CartesianOrientation(self, zero_pose: PR2):
        tip = 'base_footprint'
        root = 'odom_combined'
        p = PoseStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = tip
        p.pose.orientation = Quaternion(*quaternion_about_axis(4, [0, 0, 1]))

        expected = tf.transform_pose('map', p)

        zero_pose.allow_all_collisions()
        zero_pose.set_json_goal('CartesianOrientation',
                                root_link=root,
                                tip_link=tip,
                                goal_orientation=p,
                                max_velocity=0.15
                                )
        zero_pose.plan_and_execute()
        new_pose = tf.lookup_pose('map', tip)
        compare_orientations(expected.pose.orientation, new_pose.pose.orientation)

    def test_CartesianPoseStraight(self, zero_pose: PR2):
        zero_pose.close_l_gripper()
        goal_position = PoseStamped()
        goal_position.header.frame_id = 'base_link'
        goal_position.pose.position.x = 0.3
        goal_position.pose.position.y = 0.5
        goal_position.pose.position.z = 1
        goal_position.pose.orientation.w = 1

        start_pose = tf.lookup_pose('map', zero_pose.l_tip)
        map_T_goal_position = tf.transform_pose('map', goal_position)

        object_pose = PoseStamped()
        object_pose.header.frame_id = 'map'
        object_pose.pose.position.x = (start_pose.pose.position.x + map_T_goal_position.pose.position.x) / 2.
        object_pose.pose.position.y = (start_pose.pose.position.y + map_T_goal_position.pose.position.y) / 2.
        object_pose.pose.position.z = (start_pose.pose.position.z + map_T_goal_position.pose.position.z) / 2.
        object_pose.pose.position.z += 0.08
        object_pose.pose.orientation.w = 1

        zero_pose.add_sphere('sphere', 0.05, pose=object_pose)

        publish_marker_vector(start_pose.pose.position, map_T_goal_position.pose.position)
        zero_pose.allow_self_collision()
        zero_pose.set_straight_cart_goal(goal_position, zero_pose.l_tip)
        zero_pose.plan_and_execute()

    def test_CartesianVelocityLimit(self, zero_pose: PR2):
        base_linear_velocity = 0.1
        base_angular_velocity = 0.2
        zero_pose.set_limit_cartesian_velocity_goal(
            root_link=zero_pose.default_root,
            tip_link='base_footprint',
            max_linear_velocity=base_linear_velocity,
            max_angular_velocity=base_angular_velocity,
            hard=True,
        )
        eef_linear_velocity = 1
        eef_angular_velocity = 1
        goal_position = PoseStamped()
        goal_position.header.frame_id = 'r_gripper_tool_frame'
        goal_position.pose.position.x = 1
        goal_position.pose.position.y = 0
        goal_position.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 4, [0, 0, 1]))
        zero_pose.allow_all_collisions()
        zero_pose.set_cart_goal(goal_pose=goal_position,
                                tip_link='r_gripper_tool_frame',
                                linear_velocity=eef_linear_velocity,
                                angular_velocity=eef_angular_velocity,
                                weight=WEIGHT_BELOW_CA)
        zero_pose.plan_and_execute()

        for time, state in zero_pose.god_map.get_data(identifier.debug_trajectory).items():
            key = '{}/{}/{}/{}/trans_error'.format('CartesianVelocityLimit',
                                                   'TranslationVelocityLimit',
                                                   zero_pose.default_root,
                                                   'base_footprint')
            assert key in state
            assert state[key].position <= base_linear_velocity + 2e3
            assert state[key].position >= -base_linear_velocity - 2e3

    def test_AvoidJointLimits1(self, zero_pose: PR2):
        percentage = 10
        zero_pose.allow_all_collisions()
        zero_pose.avoid_joint_limits(percentage=percentage)
        zero_pose.plan_and_execute()

        joint_non_continuous = [j for j in zero_pose.robot.controlled_joints if
                                not zero_pose.robot.is_joint_continuous(j)]

        current_joint_state = zero_pose.robot.state.to_position_dict()
        percentage *= 0.99  # if will not reach the exact percentager, because the weight is so low
        for joint in joint_non_continuous:
            position = current_joint_state[joint]
            lower_limit, upper_limit = zero_pose.robot.get_joint_position_limits(joint)
            joint_range = upper_limit - lower_limit
            center = (upper_limit + lower_limit) / 2.
            upper_limit2 = center + joint_range / 2. * (1 - percentage / 100.)
            lower_limit2 = center - joint_range / 2. * (1 - percentage / 100.)
            assert upper_limit2 >= position >= lower_limit2

    def test_AvoidJointLimits2(self, zero_pose: PR2):
        percentage = 10
        joints = [j for j in zero_pose.robot.controlled_joints if
                  not zero_pose.robot.is_joint_continuous(j)]
        goal_state = {j: zero_pose.robot.get_joint_position_limits(j)[1] for j in joints}
        del goal_state['odom_x_joint']
        del goal_state['odom_y_joint']
        zero_pose.set_json_goal('AvoidJointLimits',
                                percentage=percentage)
        zero_pose.set_joint_goal(goal_state, check=False)
        zero_pose.allow_self_collision()
        zero_pose.plan_and_execute()

        zero_pose.set_json_goal('AvoidJointLimits',
                                percentage=percentage)
        zero_pose.allow_self_collision()
        zero_pose.plan_and_execute()

        joint_non_continuous = [j for j in zero_pose.robot.controlled_joints if
                                not zero_pose.robot.is_joint_continuous(j)]

        current_joint_state = zero_pose.robot.state.to_position_dict()
        percentage *= 0.9  # if will not reach the exact percentage, because the weight is so low
        for joint in joint_non_continuous:
            position = current_joint_state[joint]
            lower_limit, upper_limit = zero_pose.robot.get_joint_position_limits(joint)
            joint_range = upper_limit - lower_limit
            center = (upper_limit + lower_limit) / 2.
            upper_limit2 = center + joint_range / 2. * (1 - percentage / 100.)
            lower_limit2 = center - joint_range / 2. * (1 - percentage / 100.)
            assert upper_limit2 >= position >= lower_limit2

    def test_OverwriteWeights1(self, pocky_pose_setup: PR2):
        # joint_velocity_weight = identifier.joint_weights + ['velocity', 'override']
        # old_torso_value = pocky_pose_setup.world.joints['torso_lift_joint'].free_variable.quadratic_weights
        # old_odom_x_value = pocky_pose_setup.world.joints['odom_x_joint'].free_variable.quadratic_weights

        r_goal = PoseStamped()
        r_goal.header.frame_id = pocky_pose_setup.r_tip
        r_goal.pose.orientation.w = 1
        r_goal.pose.position.x += 0.1
        updates = {
            1: {
                'odom_x_joint': 1000000,
                'odom_y_joint': 1000000,
                'odom_z_joint': 1000000
            },
        }

        old_pose = tf.lookup_pose('map', 'base_footprint')

        pocky_pose_setup.set_overwrite_joint_weights_goal(updates)
        pocky_pose_setup.set_cart_goal(r_goal, pocky_pose_setup.r_tip, check=False)
        pocky_pose_setup.plan_and_execute()

        new_pose = tf.lookup_pose('map', 'base_footprint')
        compare_poses(new_pose.pose, old_pose.pose)

        assert pocky_pose_setup.world.joints['odom_x_joint'].free_variable.quadratic_weights[1] == 1000000
        assert not isinstance(pocky_pose_setup.world.joints['torso_lift_joint'].free_variable.quadratic_weights[1], int)

        updates = {
            1: {
                'odom_x_joint': 0.0001,
                'odom_y_joint': 0.0001,
                'odom_z_joint': 0.0001,
            },
        }
        # old_pose = tf.lookup_pose('map', 'base_footprint')
        # old_pose.pose.position.x += 0.1
        pocky_pose_setup.set_overwrite_joint_weights_goal(updates)
        pocky_pose_setup.set_cart_goal(r_goal, pocky_pose_setup.r_tip)
        pocky_pose_setup.plan_and_execute()

        new_pose = tf.lookup_pose('map', 'base_footprint')

        # compare_poses(old_pose.pose, new_pose.pose)
        assert new_pose.pose.position.x >= 0.03
        assert pocky_pose_setup.world.joints['odom_x_joint'].free_variable.quadratic_weights[1] == 0.0001
        assert not isinstance(pocky_pose_setup.world.joints['torso_lift_joint'].free_variable.quadratic_weights[1],
                              float)
        pocky_pose_setup.plan_and_execute()
        assert not isinstance(pocky_pose_setup.world.joints['odom_x_joint'].free_variable.quadratic_weights[1],
                              float)
        assert not isinstance(pocky_pose_setup.world.joints['torso_lift_joint'].free_variable.quadratic_weights[1],
                              float)

    def test_pointing(self, kitchen_setup: PR2):
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'base_footprint'
        base_goal.pose.position.y = -1
        base_goal.pose.orientation.w = 1
        kitchen_setup.teleport_base(base_goal)

        tip = 'head_mount_kinect_rgb_link'
        goal_point = tf.lookup_point('map', 'iai_kitchen/iai_fridge_door_handle')
        goal_point.header.stamp = rospy.Time()
        pointing_axis = Vector3Stamped()
        pointing_axis.header.frame_id = tip
        pointing_axis.vector.x = 1
        kitchen_setup.set_pointing_goal(tip, goal_point, pointing_axis=pointing_axis)
        kitchen_setup.plan_and_execute()

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'base_footprint'
        base_goal.pose.position.y = 2
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(1, [0, 0, 1]))
        kitchen_setup.set_pointing_goal(tip, goal_point, pointing_axis=pointing_axis)
        gaya_pose2 = deepcopy(kitchen_setup.better_pose)
        del gaya_pose2['head_pan_joint']
        del gaya_pose2['head_tilt_joint']
        kitchen_setup.set_joint_goal(gaya_pose2)
        kitchen_setup.move_base(base_goal)

        current_x = Vector3Stamped()
        current_x.header.frame_id = tip
        current_x.vector.x = 1

        expected_x = tf.transform_point(tip, goal_point)
        np.testing.assert_almost_equal(expected_x.point.y, 0, 1)
        np.testing.assert_almost_equal(expected_x.point.z, 0, 1)

        rospy.loginfo("Starting looking")
        tip = 'head_mount_kinect_rgb_link'
        goal_point = tf.lookup_point('map', kitchen_setup.r_tip)
        goal_point.header.stamp = rospy.Time()
        pointing_axis = Vector3Stamped()
        pointing_axis.header.frame_id = tip
        pointing_axis.vector.x = 1
        kitchen_setup.set_pointing_goal(tip, goal_point, pointing_axis=pointing_axis, root_link=kitchen_setup.r_tip)
        dgaya_pose = deepcopy(gaya_pose)
        del dgaya_pose[u'head_pan_joint']
        del dgaya_pose[u'head_tilt_joint']
        kitchen_setup.set_joint_goal(dgaya_pose)
        kitchen_setup.plan_and_execute()

        rospy.loginfo("Starting pointing")
        r_goal = PoseStamped()
        r_goal.header.frame_id = kitchen_setup.r_tip
        r_goal.pose.position.x -= 0.3
        r_goal.pose.position.z += 0.6
        r_goal.pose.orientation.w = 1
        r_goal = tf.transform_pose(kitchen_setup.default_root, r_goal)
        r_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 0, -1, 0],
                                                                      [0, 1, 0, 0],
                                                                      [1, 0, 0, 0],
                                                                      [0, 0, 0, 1]]))

        kitchen_setup.set_cart_goal(r_goal, kitchen_setup.r_tip, 'base_footprint', weight=WEIGHT_BELOW_CA)
        kitchen_setup.plan_and_execute()

    def test_open_fridge(self, kitchen_setup: PR2):
        handle_frame_id = 'iai_kitchen/iai_fridge_door_handle'
        handle_name = 'iai_fridge_door_handle'

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position = Point(0.3, -0.5, 0)
        base_goal.pose.orientation.w = 1
        kitchen_setup.teleport_base(base_goal)

        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = handle_frame_id
        bar_axis.vector.z = 1

        bar_center = PointStamped()
        bar_center.header.frame_id = handle_frame_id

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = kitchen_setup.r_tip
        tip_grasp_axis.vector.z = 1

        kitchen_setup.set_json_goal('GraspBar',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=kitchen_setup.r_tip,
                                    tip_grasp_axis=tip_grasp_axis,
                                    bar_center=bar_center,
                                    bar_axis=bar_axis,
                                    bar_length=.4)
        x_gripper = Vector3Stamped()
        x_gripper.header.frame_id = kitchen_setup.r_tip
        x_gripper.vector.x = 1

        x_goal = Vector3Stamped()
        x_goal.header.frame_id = handle_frame_id
        x_goal.vector.x = -1
        kitchen_setup.set_align_planes_goal(kitchen_setup.r_tip, x_gripper, root_normal=x_goal)
        kitchen_setup.allow_all_collisions()
        # kitchen_setup.add_json_goal('AvoidJointLimits', percentage=10)
        kitchen_setup.plan_and_execute()

        kitchen_setup.set_json_goal('Open',
                                    tip_link=kitchen_setup.r_tip,
                                    environment_link=handle_name,
                                    goal_joint_state=1.5)
        kitchen_setup.set_json_goal('AvoidJointLimits', percentage=40)
        kitchen_setup.allow_all_collisions()
        # kitchen_setup.add_json_goal('AvoidJointLimits')
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_kitchen_js({'iai_fridge_door_joint': 1.5})

        kitchen_setup.set_json_goal('Open',
                                    tip_link=kitchen_setup.r_tip,
                                    environment_link=handle_name,
                                    goal_joint_state=0)
        kitchen_setup.allow_all_collisions()
        kitchen_setup.set_json_goal('AvoidJointLimits', percentage=40)
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_kitchen_js({'iai_fridge_door_joint': 0})

        kitchen_setup.plan_and_execute()

        kitchen_setup.set_joint_goal(kitchen_setup.better_pose)
        kitchen_setup.plan_and_execute()

    def test_open_drawer(self, kitchen_setup: PR2):
        handle_frame_id = 'iai_kitchen/sink_area_left_middle_drawer_handle'
        handle_name = 'sink_area_left_middle_drawer_handle'
        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = handle_frame_id
        bar_axis.vector.y = 1

        bar_center = PointStamped()
        bar_center.header.frame_id = handle_frame_id

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = kitchen_setup.l_tip
        tip_grasp_axis.vector.z = 1

        kitchen_setup.set_json_goal('GraspBar',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=kitchen_setup.l_tip,
                                    tip_grasp_axis=tip_grasp_axis,
                                    bar_center=bar_center,
                                    bar_axis=bar_axis,
                                    bar_length=0.4)  # TODO: check for real length
        x_gripper = Vector3Stamped()
        x_gripper.header.frame_id = kitchen_setup.l_tip
        x_gripper.vector.x = 1

        x_goal = Vector3Stamped()
        x_goal.header.frame_id = handle_frame_id
        x_goal.vector.x = -1

        kitchen_setup.set_align_planes_goal(kitchen_setup.l_tip,
                                            x_gripper,
                                            root_normal=x_goal)
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()

        kitchen_setup.set_json_goal('Open',
                                    tip_link=kitchen_setup.l_tip,
                                    environment_link=handle_name)
        kitchen_setup.allow_all_collisions()  # makes execution faster
        kitchen_setup.plan_and_execute()  # send goal to Giskard
        # Update kitchen object
        kitchen_setup.set_kitchen_js({'sink_area_left_middle_drawer_main_joint': 0.48})

        # Close drawer partially
        kitchen_setup.set_json_goal('Open',
                                    tip_link=kitchen_setup.l_tip,
                                    environment_link=handle_name,
                                    goal_joint_state=0.2)
        kitchen_setup.allow_all_collisions()  # makes execution faster
        kitchen_setup.plan_and_execute()  # send goal to Giskard
        # Update kitchen object
        kitchen_setup.set_kitchen_js({'sink_area_left_middle_drawer_main_joint': 0.2})

        kitchen_setup.set_json_goal('Close',
                                    tip_link=kitchen_setup.l_tip,
                                    environment_link=handle_name)
        kitchen_setup.allow_all_collisions()  # makes execution faster
        kitchen_setup.plan_and_execute()  # send goal to Giskard
        # Update kitchen object
        kitchen_setup.set_kitchen_js({'sink_area_left_middle_drawer_main_joint': 0.0})

        # TODO: calculate real and desired value and compare
        pass

    def test_open_close_dishwasher(self, kitchen_setup: PR2):
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.orientation.w = 1
        p.pose.position.x = 0.5
        p.pose.position.y = 0.2
        kitchen_setup.teleport_base(p)

        hand = kitchen_setup.r_tip

        goal_angle = np.pi / 4
        handle_frame_id = 'iai_kitchen/sink_area_dish_washer_door_handle'
        handle_name = 'sink_area_dish_washer_door_handle'
        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = handle_frame_id
        bar_axis.vector.y = 1

        bar_center = PointStamped()
        bar_center.header.frame_id = handle_frame_id

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = hand
        tip_grasp_axis.vector.z = 1

        kitchen_setup.set_json_goal('GraspBar',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=hand,
                                    tip_grasp_axis=tip_grasp_axis,
                                    bar_center=bar_center,
                                    bar_axis=bar_axis,
                                    bar_length=.3)
        # kitchen_setup.allow_collision([], 'kitchen', [handle_name])
        # kitchen_setup.allow_all_collisions()

        x_gripper = Vector3Stamped()
        x_gripper.header.frame_id = hand
        x_gripper.vector.x = 1

        x_goal = Vector3Stamped()
        x_goal.header.frame_id = handle_frame_id
        x_goal.vector.x = -1
        kitchen_setup.set_align_planes_goal(hand, x_gripper, root_normal=x_goal)
        # kitchen_setup.allow_all_collisions()

        kitchen_setup.plan_and_execute()

        kitchen_setup.set_json_goal('Open',
                                    tip_link=hand,
                                    environment_link=handle_name,
                                    goal_joint_state=goal_angle,
                                    )
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_kitchen_js({'sink_area_dish_washer_door_joint': goal_angle})

        kitchen_setup.set_json_goal('Open',
                                    tip_link=hand,
                                    environment_link=handle_name,
                                    goal_joint_state=0)
        kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_kitchen_js({'sink_area_dish_washer_door_joint': 0})

    # def test_open_close_dishwasher_palm(self, kitchen_setup: PR2):
    #     # FIXME
    #     handle_frame_id = 'iai_kitchen/sink_area_dish_washer_door_handle'
    #     handle_name = 'sink_area_dish_washer_door_handle'
    #     hand = kitchen_setup.r_tip
    #     goal_angle = np.pi / 3.5
    #
    #     p = PoseStamped()
    #     p.header.frame_id = 'map'
    #     p.pose.orientation.w = 1
    #     p.pose.position.x = 0.5
    #     p.pose.position.y = 0.2
    #     kitchen_setup.teleport_base(p)
    #
    #     kitchen_setup.set_kitchen_js({'sink_area_dish_washer_door_joint': 0.})
    #
    #     hand_goal = PoseStamped()
    #     hand_goal.header.frame_id = handle_frame_id
    #     hand_goal.pose.position.x -= 0.03
    #     hand_goal.pose.position.z = 0.03
    #     hand_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 0, 1, 0],
    #                                                                      [0, -1, 0, 0],
    #                                                                      [1, 0, 0, 0],
    #                                                                      [0, 0, 0, 1]]))
    #     # kitchen_setup.allow_all_collisions()
    #     kitchen_setup.set_json_goal('CartesianVelocityLimit',
    #                                 root_link='odom_combined',
    #                                 tip_link='base_footprint',
    #                                 max_linear_velocity=0.05,
    #                                 max_angular_velocity=0.08,
    #                                 )
    #     kitchen_setup.set_cart_goal(hand_goal, hand)
    #     kitchen_setup.send_goal(goal_type=MoveGoal.PLAN_AND_EXECUTE_AND_CUT_OFF_SHAKING)
    #
    #     kitchen_setup.set_json_goal('Open',
    #                                 tip_link=hand,
    #                                 object_name='kitchen',
    #                                 handle_link=handle_name,
    #                                 goal_joint_state=goal_angle,
    #                                 )
    #
    #     kitchen_setup.set_json_goal('CartesianVelocityLimit',
    #                                 root_link='odom_combined',
    #                                 tip_link='base_footprint',
    #                                 max_linear_velocity=0.05,
    #                                 max_angular_velocity=0.1,
    #                                 )
    #
    #     kitchen_setup.set_json_goal('CartesianVelocityLimit',
    #                                 root_link='odom_combined',
    #                                 tip_link='r_forearm_link',
    #                                 max_linear_velocity=0.1,
    #                                 max_angular_velocity=0.5,
    #                                 )
    #
    #     # kitchen_setup.allow_all_collisions()
    #     kitchen_setup.plan_and_execute()
    #     kitchen_setup.set_kitchen_js({'sink_area_dish_washer_door_joint': goal_angle})
    #
    #     kitchen_setup.set_json_goal('Open',
    #                                 tip_link=hand,
    #                                 object_name='kitchen',
    #                                 handle_link=handle_name,
    #                                 goal_joint_state=0,
    #                                 )
    #
    #     kitchen_setup.set_json_goal('CartesianVelocityLimit',
    #                                 root_link='odom_combined',
    #                                 tip_link='base_footprint',
    #                                 max_linear_velocity=0.05,
    #                                 max_angular_velocity=0.1,
    #                                 )
    #
    #     kitchen_setup.set_json_goal('CartesianVelocityLimit',
    #                                 root_link='odom_combined',
    #                                 tip_link='r_forearm_link',
    #                                 max_linear_velocity=0.05,
    #                                 max_angular_velocity=0.1,
    #                                 )
    #
    #     # kitchen_setup.allow_all_collisions()
    #     kitchen_setup.plan_and_execute()
    #     kitchen_setup.set_kitchen_js({'sink_area_dish_washer_door_joint': 0})
    #
    #     kitchen_setup.set_joint_goal(kitchen_setup.better_pose)
    #     kitchen_setup.plan_and_execute()

    def test_align_planes1(self, zero_pose: PR2):
        x_gripper = Vector3Stamped()
        x_gripper.header.frame_id = zero_pose.r_tip
        x_gripper.vector.x = 1
        y_gripper = Vector3Stamped()
        y_gripper.header.frame_id = zero_pose.r_tip
        y_gripper.vector.y = 1

        x_goal = Vector3Stamped()
        x_goal.header.frame_id = 'map'
        x_goal.vector.x = 1
        y_goal = Vector3Stamped()
        y_goal.header.frame_id = 'map'
        y_goal.vector.z = 1
        zero_pose.set_align_planes_goal(zero_pose.r_tip, x_gripper, root_normal=x_goal)
        zero_pose.set_align_planes_goal(zero_pose.r_tip, y_gripper, root_normal=y_goal)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

    def test_wrong_constraint_type(self, zero_pose: PR2):
        goal_state = JointState()
        goal_state.name = ['r_elbow_flex_joint']
        goal_state.position = [-1.0]
        kwargs = {'goal_state': goal_state}
        zero_pose.set_json_goal('jointpos', **kwargs)
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.UNKNOWN_CONSTRAINT])

    def test_python_code_in_constraint_type(self, zero_pose: PR2):
        goal_state = JointState()
        goal_state.name = ['r_elbow_flex_joint']
        goal_state.position = [-1.0]
        kwargs = {'goal_state': goal_state}
        zero_pose.set_json_goal('print("asd")', **kwargs)
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.UNKNOWN_CONSTRAINT])

    def test_wrong_params1(self, zero_pose: PR2):
        goal_state = JointState()
        goal_state.name = 'r_elbow_flex_joint'
        goal_state.position = [-1.0]
        kwargs = {'goal_state': goal_state}
        zero_pose.set_json_goal('JointPositionList', **kwargs)
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.CONSTRAINT_INITIALIZATION_ERROR])

    def test_wrong_params2(self, zero_pose: PR2):
        goal_state = JointState()
        goal_state.name = [5432]
        goal_state.position = 'test'
        kwargs = {'goal_state': goal_state}
        zero_pose.set_json_goal('JointPositionList', **kwargs)
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.CONSTRAINT_INITIALIZATION_ERROR])

    def test_align_planes2(self, zero_pose: PR2):
        x_gripper = Vector3Stamped()
        x_gripper.header.frame_id = zero_pose.r_tip
        x_gripper.vector.y = 1

        x_goal = Vector3Stamped()
        x_goal.header.frame_id = 'map'
        x_goal.vector.y = -1
        x_goal.vector = tf.normalize(x_goal.vector)
        zero_pose.set_align_planes_goal(zero_pose.r_tip, x_gripper, root_normal=x_goal)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

    def test_align_planes3(self, zero_pose: PR2):
        eef_vector = Vector3Stamped()
        eef_vector.header.frame_id = 'base_footprint'
        eef_vector.vector.y = 1

        goal_vector = Vector3Stamped()
        goal_vector.header.frame_id = 'map'
        goal_vector.vector.y = -1
        goal_vector.vector = tf.normalize(goal_vector.vector)
        zero_pose.set_align_planes_goal('base_footprint', eef_vector, root_normal=goal_vector)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

    def test_align_planes4(self, kitchen_setup: PR2):
        elbow = 'r_elbow_flex_link'
        handle_frame_id = 'iai_kitchen/iai_fridge_door_handle'

        tip_axis = Vector3Stamped()
        tip_axis.header.frame_id = elbow
        tip_axis.vector.x = 1

        env_axis = Vector3Stamped()
        env_axis.header.frame_id = handle_frame_id
        env_axis.vector.z = 1
        kitchen_setup.set_align_planes_goal(elbow, tip_axis, root_normal=env_axis, weight=WEIGHT_ABOVE_CA)
        kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()

    def test_grasp_fridge_handle(self, kitchen_setup: PR2):
        handle_name = 'iai_kitchen/iai_fridge_door_handle'
        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = handle_name
        bar_axis.vector.z = 1

        bar_center = PointStamped()
        bar_center.header.frame_id = handle_name

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = kitchen_setup.r_tip
        tip_grasp_axis.vector.z = 1

        kitchen_setup.set_json_goal('GraspBar',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=kitchen_setup.r_tip,
                                    tip_grasp_axis=tip_grasp_axis,
                                    bar_center=bar_center,
                                    bar_axis=bar_axis,
                                    bar_length=.4)
        #
        x_gripper = Vector3Stamped()
        x_gripper.header.frame_id = kitchen_setup.r_tip
        x_gripper.vector.x = 1

        x_goal = Vector3Stamped()
        x_goal.header.frame_id = 'iai_kitchen/iai_fridge_door_handle'
        x_goal.vector.x = -1
        kitchen_setup.set_align_planes_goal(kitchen_setup.r_tip, x_gripper, root_normal=x_goal)
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()

    def test_close_fridge_with_elbow(self, kitchen_setup: PR2):
        base_pose = PoseStamped()
        base_pose.header.frame_id = 'map'
        base_pose.pose.position.y = -1.5
        base_pose.pose.orientation.w = 1
        kitchen_setup.teleport_base(base_pose)

        handle_frame_id = 'iai_kitchen/iai_fridge_door_handle'
        handle_name = 'iai_fridge_door_handle'

        kitchen_setup.set_kitchen_js({'iai_fridge_door_joint': np.pi / 2})

        elbow = 'r_elbow_flex_link'

        tip_axis = Vector3Stamped()
        tip_axis.header.frame_id = elbow
        tip_axis.vector.x = 1

        env_axis = Vector3Stamped()
        env_axis.header.frame_id = handle_frame_id
        env_axis.vector.z = 1
        kitchen_setup.set_align_planes_goal(elbow, tip_axis, root_normal=env_axis, weight=WEIGHT_ABOVE_CA)
        kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()
        elbow_point = PointStamped()
        elbow_point.header.frame_id = handle_frame_id
        elbow_point.point.x += 0.1
        kitchen_setup.set_translation_goal(elbow_point, elbow)
        kitchen_setup.set_align_planes_goal(elbow, tip_axis, root_normal=env_axis, weight=WEIGHT_ABOVE_CA)
        kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()

        kitchen_setup.set_json_goal('Close',
                                    tip_link=elbow,
                                    environment_link=handle_name)
        kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_kitchen_js({'iai_fridge_door_joint': 0})

    def test_open_close_oven(self, kitchen_setup: PR2):
        goal_angle = 0.5
        handle_frame_id = 'iai_kitchen/oven_area_oven_door_handle'
        handle_name = 'oven_area_oven_door_handle'
        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = handle_frame_id
        bar_axis.vector.y = 1

        bar_center = PointStamped()
        bar_center.header.frame_id = handle_frame_id

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = kitchen_setup.l_tip
        tip_grasp_axis.vector.z = 1

        kitchen_setup.set_json_goal('GraspBar',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=kitchen_setup.l_tip,
                                    tip_grasp_axis=tip_grasp_axis,
                                    bar_center=bar_center,
                                    bar_axis=bar_axis,
                                    bar_length=.3)
        # kitchen_setup.allow_collision([], 'kitchen', [handle_name])
        kitchen_setup.allow_all_collisions()

        x_gripper = Vector3Stamped()
        x_gripper.header.frame_id = kitchen_setup.l_tip
        x_gripper.vector.x = 1

        x_goal = Vector3Stamped()
        x_goal.header.frame_id = handle_frame_id
        x_goal.vector.x = -1
        kitchen_setup.set_align_planes_goal(kitchen_setup.l_tip, x_gripper, root_normal=x_goal)
        # kitchen_setup.allow_all_collisions()

        kitchen_setup.plan_and_execute()

        kitchen_setup.set_json_goal('Open',
                                    tip_link=kitchen_setup.l_tip,
                                    environment_link=handle_name,
                                    goal_joint_state=goal_angle)
        kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_kitchen_js({'oven_area_oven_door_joint': goal_angle})

        kitchen_setup.set_json_goal('Close',
                                    tip_link=kitchen_setup.l_tip,
                                    environment_link=handle_name)
        kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_kitchen_js({'oven_area_oven_door_joint': 0})

    def test_grasp_dishwasher_handle(self, kitchen_setup: PR2):
        handle_name = 'iai_kitchen/sink_area_dish_washer_door_handle'
        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = handle_name
        bar_axis.vector.y = 1

        bar_center = PointStamped()
        bar_center.header.frame_id = handle_name

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = kitchen_setup.r_tip
        tip_grasp_axis.vector.z = 1

        kitchen_setup.set_grasp_bar_goal(root_link=kitchen_setup.default_root,
                                         tip_link=kitchen_setup.r_tip,
                                         tip_grasp_axis=tip_grasp_axis,
                                         bar_center=bar_center,
                                         bar_axis=bar_axis,
                                         bar_length=.3)
        kitchen_setup.register_group('handle', 'kitchen', 'sink_area_dish_washer_door_handle')
        kitchen_setup.allow_collision(kitchen_setup.get_robot_name(), 'handle')
        kitchen_setup.plan_and_execute()

    def test_open_all_drawers(self, kitchen_setup):
        """"
        :type kitchen_setup: PR2
        """
        handle_name = [
            # u'oven_area_area_middle_upper_drawer_handle',
            # u'oven_area_area_middle_lower_drawer_handle',
            # u'sink_area_left_upper_drawer_handle',
            # u'sink_area_left_middle_drawer_handle',
            # u'sink_area_left_bottom_drawer_handle',
            # u'sink_area_trash_drawer_handle',
            # u'fridge_area_lower_drawer_handle',
            # u'kitchen_island_left_upper_drawer_handle',
            # u'kitchen_island_left_lower_drawer_handle',
            # u'kitchen_island_middle_upper_drawer_handle',
            # u'kitchen_island_middle_lower_drawer_handle',
            # u'kitchen_island_right_upper_drawer_handle',
            # u'kitchen_island_right_lower_drawer_handle',
            u'oven_area_area_right_drawer_handle',
            # u'oven_area_area_right_drawer_handle'
        ]

        handle_frame_id = [u'iai_kitchen/' + item for item in handle_name]
        joint_name = [item.replace(u'handle', u'main_joint') for item in handle_name]

        for i_handle_id, i_handle_name, i_joint_name in zip(handle_frame_id, handle_name, joint_name):
            logging.loginfo('=== Opening drawer: {} ==='.format(i_handle_name.replace(u'_handle', u'')))
            bar_axis = Vector3Stamped()
            bar_axis.header.frame_id = i_handle_id
            bar_axis.vector.y = 1

            bar_center = PointStamped()
            bar_center.header.frame_id = i_handle_id

            tip_grasp_axis = Vector3Stamped()
            tip_grasp_axis.header.frame_id = kitchen_setup.l_tip
            tip_grasp_axis.vector.z = 1

            kitchen_setup.set_json_goal(u'GraspBar',
                                        root_link=kitchen_setup.default_root,
                                        tip_link=kitchen_setup.l_tip,
                                        tip_grasp_axis=tip_grasp_axis,
                                        bar_center=bar_center,
                                        bar_axis=bar_axis,
                                        bar_length=0.4)  # TODO: check for real length
            x_gripper = Vector3Stamped()
            x_gripper.header.frame_id = kitchen_setup.l_tip
            x_gripper.vector.x = 1

            x_goal = Vector3Stamped()
            x_goal.header.frame_id = i_handle_id
            x_goal.vector.x = -1

            kitchen_setup.set_align_planes_goal(kitchen_setup.l_tip,
                                                x_gripper,
                                                root_normal=x_goal)
            kitchen_setup.allow_all_collisions()
            kitchen_setup.avoid_self_collision()
            kitchen_setup.plan_and_execute()

            kitchen_setup.set_json_goal(u'Open',
                                        tip_link=kitchen_setup.l_tip,
                                        environment_link=i_handle_name)
            kitchen_setup.allow_all_collisions()  # makes execution faster
            kitchen_setup.avoid_self_collision()
            kitchen_setup.plan_and_execute()  # send goal to Giskard
            # Update kitchen object
            kitchen_setup.set_kitchen_js({i_joint_name: 0.48})  # TODO: get real value from URDF

            # Close drawer partially
            kitchen_setup.set_json_goal(u'Open',
                                        tip_link=kitchen_setup.l_tip,
                                        environment_link=i_handle_name,
                                        goal_joint_state=0.2)
            kitchen_setup.allow_all_collisions()  # makes execution faster
            kitchen_setup.avoid_self_collision()
            kitchen_setup.plan_and_execute()  # send goal to Giskard
            # Update kitchen object
            kitchen_setup.set_kitchen_js({i_joint_name: 0.2})

            kitchen_setup.set_json_goal(u'Close',
                                        tip_link=kitchen_setup.l_tip,
                                        object_name=u'kitchen',
                                        object_link_name=i_handle_name)
            kitchen_setup.allow_all_collisions()  # makes execution faster
            kitchen_setup.avoid_self_collision()
            kitchen_setup.plan_and_execute()  # send goal to Giskard
            # Update kitchen object
            kitchen_setup.set_kitchen_js({i_joint_name: 0.0})


class TestCartGoals(object):
    def test_move_base_1(self, zero_pose: PR2):
        map_T_odom = PoseStamped()
        map_T_odom.pose.position.x = 1
        map_T_odom.pose.position.y = 1
        map_T_odom.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 3, [0, 0, 1]))
        zero_pose.set_localization(map_T_odom)

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 1
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        zero_pose.set_cart_goal(base_goal, 'base_footprint')
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

    def test_move_base_2(self, kitchen_setup: PR2):
        map_T_odom = PoseStamped()
        map_T_odom.header.frame_id = 'map'
        map_T_odom.pose.position.x = 0
        map_T_odom.pose.position.y = 1
        map_T_odom.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 3, [0, 0, 1]))
        kitchen_setup.move_base(map_T_odom)

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = -2
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        kitchen_setup.set_cart_goal(base_goal, 'base_footprint')
        kitchen_setup.plan_and_execute()

    def test_rotate_gripper(self, zero_pose: PR2):
        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.r_tip
        r_goal.pose.orientation = Quaternion(*quaternion_about_axis(pi, [1, 0, 0]))
        zero_pose.set_cart_goal(r_goal, zero_pose.r_tip)
        zero_pose.plan_and_execute()

    def test_shaky_grippers(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        f = 0.5
        amp = 1.0

        zero_pose.set_json_goal(u'ShakyCartesianPosition',
                                root_link=zero_pose.default_root,
                                tip_link=zero_pose.r_tip,
                                frequency=f,
                                noise_amplitude=amp)
        zero_pose.set_json_goal(u'ShakyCartesianPosition',
                                root_link=zero_pose.default_root,
                                tip_link=zero_pose.l_tip,
                                frequency=f,
                                noise_amplitude=amp)

        r_gripper_vec = Vector3Stamped()
        r_gripper_vec.header.frame_id = zero_pose.r_tip
        r_gripper_vec.vector.z = 1
        l_gripper_vec = Vector3Stamped()
        l_gripper_vec.header.frame_id = zero_pose.l_tip
        l_gripper_vec.vector.z = 1

        gripper_goal_vec = Vector3Stamped()
        gripper_goal_vec.header.frame_id = u'map'
        gripper_goal_vec.vector.z = 1
        # zero_pose.set_align_planes_goal(zero_pose.r_tip, r_gripper_vec, root_normal=gripper_goal_vec)
        # zero_pose.set_align_planes_goal(zero_pose.l_tip, l_gripper_vec, root_normal=gripper_goal_vec)
        zero_pose.plan_and_execute()

    def test_keep_position1(self, zero_pose: PR2):
        zero_pose.allow_self_collision()
        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.r_tip
        r_goal.pose.position.x = -.1
        r_goal.pose.orientation.w = 1
        zero_pose.set_cart_goal(r_goal, zero_pose.r_tip, 'torso_lift_link')
        zero_pose.plan_and_execute()

        js = {'torso_lift_joint': 0.1}
        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.r_tip
        r_goal.pose.orientation.w = 1
        zero_pose.set_cart_goal(r_goal, zero_pose.r_tip, 'torso_lift_link')
        zero_pose.set_joint_goal(js)
        zero_pose.allow_self_collision()
        zero_pose.plan_and_execute()

    def test_keep_position2(self, zero_pose: PR2):
        zero_pose.allow_self_collision()
        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.r_tip
        r_goal.pose.position.x = -.1
        r_goal.pose.orientation.w = 1
        zero_pose.set_cart_goal(r_goal, zero_pose.r_tip, 'torso_lift_link')
        zero_pose.plan_and_execute()

        zero_pose.allow_self_collision()
        js = {'torso_lift_joint': 0.1}
        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.r_tip
        r_goal.pose.orientation.w = 1
        expected_pose = tf.lookup_pose(zero_pose.default_root, zero_pose.r_tip)
        expected_pose.header.stamp = rospy.Time()
        zero_pose.set_cart_goal(r_goal, zero_pose.r_tip, zero_pose.default_root)
        zero_pose.set_joint_goal(js)
        zero_pose.plan_and_execute()

    def test_keep_position3(self, zero_pose: PR2):
        js = {
            'r_elbow_flex_joint': -1.58118094489,
            'r_forearm_roll_joint': -0.904933033043,
            'r_shoulder_lift_joint': 0.822412440711,
            'r_shoulder_pan_joint': -1.07866800992,
            'r_upper_arm_roll_joint': -1.34905471854,
            'r_wrist_flex_joint': -1.20182042644,
            'r_wrist_roll_joint': 0.190433188769,
        }
        zero_pose.set_joint_goal(js)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.r_tip
        r_goal.pose.position.x = 0.3
        r_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[-1, 0, 0, 0],
                                                                      [0, 1, 0, 0],
                                                                      [0, 0, -1, 0],
                                                                      [0, 0, 0, 1]]))
        zero_pose.set_cart_goal(r_goal, zero_pose.l_tip, 'torso_lift_link')
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.r_tip
        r_goal.pose.orientation.w = 1
        zero_pose.set_cart_goal(r_goal, zero_pose.r_tip, zero_pose.l_tip)

        l_goal = PoseStamped()
        l_goal.header.frame_id = zero_pose.r_tip
        l_goal.pose.position.y = -.1
        l_goal.pose.orientation.w = 1
        zero_pose.set_cart_goal(l_goal, zero_pose.r_tip, zero_pose.default_root)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

    def test_cart_goal_1eef(self, zero_pose: PR2):
        p = PoseStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = zero_pose.r_tip
        p.pose.position = Point(-0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.allow_self_collision()
        zero_pose.set_cart_goal(p, zero_pose.r_tip, 'base_footprint')
        zero_pose.plan_and_execute()

    def test_cart_goal_1eef2(self, zero_pose: PR2):
        # zero_pose.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
        p = PoseStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = 'base_footprint'
        p.pose.position = Point(0.599, -0.009, 0.983)
        p.pose.orientation = Quaternion(0.524, -0.495, 0.487, -0.494)
        zero_pose.allow_all_collisions()
        zero_pose.set_cart_goal(p, zero_pose.l_tip, 'torso_lift_link')
        zero_pose.plan_and_execute()

    def test_cart_goal_1eef3(self, zero_pose: PR2):
        self.test_cart_goal_1eef(zero_pose)
        self.test_cart_goal_1eef2(zero_pose)

    def test_cart_goal_1eef4(self, zero_pose: PR2):
        p = PoseStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = 'map'
        p.pose.position = Point(2., 0, 1.)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.allow_all_collisions()
        zero_pose.set_cart_goal(p, zero_pose.r_tip, zero_pose.default_root)
        zero_pose.plan_and_execute()

    def test_cart_goal_orientation_singularity(self, zero_pose: PR2):
        root = 'base_link'
        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.r_tip
        r_goal.header.stamp = rospy.get_rostime()
        r_goal.pose.position = Point(-0.1, 0, 0)
        r_goal.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_cart_goal(r_goal, zero_pose.r_tip, root)
        l_goal = PoseStamped()
        l_goal.header.frame_id = zero_pose.l_tip
        l_goal.header.stamp = rospy.get_rostime()
        l_goal.pose.position = Point(-0.05, 0, 0)
        l_goal.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_cart_goal(l_goal, zero_pose.l_tip, root)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

    def test_CartesianPoseChanging(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        root = u'base_link'
        r_goal_a = PoseStamped()
        r_goal_a.header.frame_id = zero_pose.r_tip
        r_goal_a.header.stamp = rospy.get_rostime()
        r_goal_a.pose.position = Point(-0.1, 0, 0.6)
        r_goal_a.pose.orientation = Quaternion(0, 0, 0, 1)
        r_goal_b = PoseStamped()
        r_goal_b.header.frame_id = zero_pose.r_tip
        r_goal_b.header.stamp = rospy.get_rostime()
        r_goal_b.pose.position = Point(0, 0, 0.1)
        r_goal_b.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_json_goal(u'CartesianPoseChanging',
                                root_link=root,
                                tip_link=zero_pose.r_tip,
                                goal_a=r_goal_a,
                                goal_b=r_goal_b
                                )
        # l_goal = PoseStamped()
        # l_goal.header.frame_id = zero_pose.l_tip
        # l_goal.header.stamp = rospy.get_rostime()
        # l_goal.pose.position = Point(0, 0, -0.1)
        # l_goal.pose.orientation = Quaternion(0, 0, 0, 1)
        # zero_pose.set_json_goal(u'CartesianPoseChanging',
        #                        root_link=root,
        #                        tip_link=zero_pose.l_tip,
        #                        goal=l_goal
        #                        )
        zero_pose.allow_self_collision()
        zero_pose.send_goal()
        # zero_pose.check_cart_goal(zero_pose.r_tip, r_goal)
        # zero_pose.check_cart_goal(zero_pose.l_tip, l_goal)

    def test_pathAroundKitchenIsland(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        tip_link = u'base_footprint'

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = 0
        base_pose.pose.position.y = 2.0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        kitchen_setup.teleport_base(base_pose)

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = 0
        base_pose.pose.position.y = -2.0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0, 0, 1]))
        goal_a = base_pose

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = -1.75
        base_pose.pose.position.y = -2.25
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0, 0, 1]))
        goal_b = base_pose

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = -1.75
        base_pose.pose.position.y = 0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        goal_c = base_pose

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = -2.5
        base_pose.pose.position.y = 0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0, 0, 1]))
        goal_d = base_pose

        kitchen_setup.set_json_goal(u'CartesianPath',
                                    tip_link=tip_link,
                                    root_link=kitchen_setup.default_root,
                                    goals=[goal_a, goal_b, goal_c, goal_d]
                                    )

        kitchen_setup.plan_and_execute()
        # kitchen_setup.send_goal()
        # kitchen_setup.check_cart_goal(tip_link, goal_a)
        # zero_pose.check_cart_goal(zero_pose.l_tip, l_goal)

    def test_cart_goal_2eef2(self, zero_pose: PR2):
        root = 'odom_combined'

        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.r_tip
        r_goal.header.stamp = rospy.get_rostime()
        r_goal.pose.position = Point(0, -0.1, 0)
        r_goal.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_cart_goal(r_goal, zero_pose.r_tip, root)
        l_goal = PoseStamped()
        l_goal.header.frame_id = zero_pose.l_tip
        l_goal.header.stamp = rospy.get_rostime()
        l_goal.pose.position = Point(-0.05, 0, 0)
        l_goal.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_cart_goal(l_goal, zero_pose.l_tip, root)
        zero_pose.allow_self_collision()
        zero_pose.plan_and_execute()

    def test_cart_goal_left_right_chain(self, zero_pose: PR2):
        r_goal = tf.lookup_pose(zero_pose.l_tip, zero_pose.r_tip)
        r_goal.pose.position.x -= 0.1
        zero_pose.allow_all_collisions()
        zero_pose.set_cart_goal(r_goal, zero_pose.r_tip, zero_pose.l_tip)
        zero_pose.plan_and_execute()

    def test_wiggle1(self, kitchen_setup: PR2):
        tray_pose = PoseStamped()
        tray_pose.header.frame_id = 'iai_kitchen/sink_area_surface'
        tray_pose.pose.position = Point(0.1, -0.4, 0.07)
        tray_pose.pose.orientation.w = 1

        l_goal = deepcopy(tray_pose)
        l_goal.pose.position.y -= 0.18
        l_goal.pose.position.z += 0.05
        l_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[0, -1, 0, 0],
                                                                      [1, 0, 0, 0],
                                                                      [0, 0, 1, 0],
                                                                      [0, 0, 0, 1]]))

        r_goal = deepcopy(tray_pose)
        r_goal.pose.position.y += 0.18
        r_goal.pose.position.z += 0.05
        r_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 1, 0, 0],
                                                                      [-1, 0, 0, 0],
                                                                      [0, 0, 1, 0],
                                                                      [0, 0, 0, 1]]))

        kitchen_setup.set_cart_goal(l_goal, kitchen_setup.l_tip, weight=WEIGHT_BELOW_CA)
        kitchen_setup.set_cart_goal(r_goal, kitchen_setup.r_tip, weight=WEIGHT_BELOW_CA)
        # kitchen_setup.allow_collision([], tray_name, [])
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.set_json_goal('CartesianVelocityLimit',
                                    root_link=kitchen_setup.default_root,
                                    tip_link='base_footprint',
                                    max_linear_velocity=0.1,
                                    max_angular_velocity=0.2
                                    )
        kitchen_setup.plan_and_execute()

    def test_wiggle2(self, zero_pose: PR2):
        goal_js = {
            'l_upper_arm_roll_joint': 1.63487737202,
            'l_shoulder_pan_joint': 1.36222920328,
            'l_shoulder_lift_joint': 0.229120778526,
            'l_forearm_roll_joint': 13.7578920265,
            'l_elbow_flex_joint': -1.48141189643,
            'l_wrist_flex_joint': -1.22662876066,
            'l_wrist_roll_joint': -53.6150824007,
        }
        zero_pose.allow_all_collisions()
        zero_pose.set_joint_goal(goal_js)
        zero_pose.plan_and_execute()

        p = PoseStamped()
        p.header.frame_id = zero_pose.l_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = -0.1
        p.pose.orientation.w = 1
        # zero_pose.allow_all_collisions()
        zero_pose.set_cart_goal(p, zero_pose.l_tip, zero_pose.default_root)
        zero_pose.plan_and_execute()

        p = PoseStamped()
        p.header.frame_id = zero_pose.l_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.2
        p.pose.orientation.w = 1
        # zero_pose.allow_all_collisions()
        zero_pose.set_cart_goal(p, zero_pose.l_tip, zero_pose.default_root)
        zero_pose.plan_and_execute()

    def test_wiggle3(self, zero_pose: PR2):
        goal_js = {
            'r_upper_arm_roll_joint': -0.0812729778068,
            'r_shoulder_pan_joint': -1.20939684714,
            'r_shoulder_lift_joint': 0.135095147908,
            'r_forearm_roll_joint': -1.50201448056,
            'r_elbow_flex_joint': -0.404527363115,
            'r_wrist_flex_joint': -1.11738043795,
            'r_wrist_roll_joint': 8.0946050982,
        }
        zero_pose.allow_all_collisions()
        zero_pose.set_joint_goal(goal_js)
        zero_pose.plan_and_execute()

        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.5
        p.pose.orientation.w = 1
        zero_pose.set_cart_goal(p, zero_pose.r_tip, zero_pose.default_root)
        zero_pose.plan_and_execute()

    def test_root_link_not_equal_chain_root(self, zero_pose: PR2):
        p = PoseStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = 'base_footprint'
        p.pose.position.x = 0.8
        p.pose.position.y = -0.5
        p.pose.position.z = 1
        p.pose.orientation.w = 1
        zero_pose.allow_self_collision()
        zero_pose.set_cart_goal(p, zero_pose.r_tip, 'torso_lift_link')
        zero_pose.plan_and_execute()


class TestActionServerEvents(object):
    def test_interrupt1(self, zero_pose: PR2):
        p = PoseStamped()
        p.header.frame_id = 'base_footprint'
        p.pose.position = Point(2, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_cart_goal(p, 'base_footprint')
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.PREEMPTED], stop_after=1)

    def test_interrupt2(self, zero_pose: PR2):
        p = PoseStamped()
        p.header.frame_id = 'base_footprint'
        p.pose.position = Point(2, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_cart_goal(p, 'base_footprint')
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.PREEMPTED], stop_after=6)

    def test_undefined_type(self, zero_pose: PR2):
        zero_pose.allow_all_collisions()
        zero_pose.send_goal(goal_type=MoveGoal.UNDEFINED,
                            expected_error_codes=[MoveResult.INVALID_GOAL])

    def test_empty_goal(self, zero_pose: PR2):
        zero_pose.cmd_seq = []
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.INVALID_GOAL])

    def test_plan_only(self, zero_pose: PR2):
        zero_pose.allow_self_collision()
        zero_pose.set_joint_goal(pocky_pose, check=False)
        zero_pose.add_goal_check(JointGoalChecker(zero_pose.god_map, zero_pose.default_pose))
        zero_pose.send_goal(goal_type=MoveGoal.PLAN_ONLY)


class TestWayPoints(object):
    def test_interrupt_way_points1(self, zero_pose: PR2):
        p = PoseStamped()
        p.header.frame_id = 'base_footprint'
        p.pose.position = Point(0, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_cart_goal(deepcopy(p), 'base_footprint')
        zero_pose.add_cmd()
        p.pose.position.x += 1
        zero_pose.set_cart_goal(deepcopy(p), 'base_footprint')
        zero_pose.add_cmd()
        p.pose.position.x += 1
        zero_pose.set_cart_goal(p, 'base_footprint')
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.SUCCESS,
                                                         MoveResult.PREEMPTED,
                                                         MoveResult.PREEMPTED],
                                   stop_after=5)

        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position = Point(-0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_cart_goal(p, zero_pose.r_tip, zero_pose.default_root)

        zero_pose.add_cmd()
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position = Point(0.0, -0.1, -0.1)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_cart_goal(p, zero_pose.r_tip, zero_pose.default_root)

        zero_pose.add_cmd()
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position = Point(0.1, 0.1, 0.1)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_cart_goal(p, zero_pose.r_tip, zero_pose.default_root)

        zero_pose.plan_and_execute()

    def test_waypoints2(self, zero_pose: PR2):
        zero_pose.set_joint_goal(pocky_pose, check=False)
        zero_pose.allow_all_collisions()
        zero_pose.add_cmd()
        zero_pose.set_joint_goal(pick_up_pose, check=False)
        zero_pose.allow_all_collisions()
        zero_pose.add_cmd()
        zero_pose.set_joint_goal(zero_pose.better_pose, check=False)
        zero_pose.allow_all_collisions()

        traj = zero_pose.plan_and_execute()
        for i, p in enumerate(traj.points):
            js = {joint_name: position for joint_name, position in zip(traj.joint_names, p.positions)}
            try:
                zero_pose.compare_joint_state(js, pocky_pose)
                break
            except AssertionError:
                pass
        else:  # if no break
            assert False, 'pocky pose not in trajectory'

        traj.points = traj.points[i:]
        for i, p in enumerate(traj.points):
            js = {joint_name: position for joint_name, position in zip(traj.joint_names, p.positions)}
            try:
                zero_pose.compare_joint_state(js, pick_up_pose)
                break
            except AssertionError:
                pass
        else:  # if no break
            assert False, 'pick_up_pose not in trajectory'

        traj.points = traj.points[i:]
        for i, p in enumerate(traj.points):
            js = {joint_name: position for joint_name, position in zip(traj.joint_names, p.positions)}
            try:
                zero_pose.compare_joint_state(js, zero_pose.better_pose)
                break
            except AssertionError:
                pass
        else:  # if no break
            assert False, 'gaya_pose not in trajectory'

        pass

    def test_waypoints_with_fail(self, zero_pose: PR2):
        zero_pose.set_joint_goal(pocky_pose)
        zero_pose.add_cmd()
        zero_pose.set_json_goal('muh')
        zero_pose.add_cmd()
        zero_pose.set_joint_goal(zero_pose.better_pose)

        traj = zero_pose.send_goal(expected_error_codes=[MoveResult.SUCCESS,
                                                         MoveResult.UNKNOWN_CONSTRAINT,
                                                         MoveResult.SUCCESS],
                                   goal_type=MoveGoal.PLAN_AND_EXECUTE_AND_SKIP_FAILURES)

        for i, p in enumerate(traj.points):
            js = {joint_name: position for joint_name, position in zip(traj.joint_names, p.positions)}
            try:
                zero_pose.compare_joint_state(js, pocky_pose)
                break
            except AssertionError:
                pass
        else:  # if no break
            assert False, 'pocky pose not in trajectory'

        traj.points = traj.points[i:]
        for i, p in enumerate(traj.points):
            js = {joint_name: position for joint_name, position in zip(traj.joint_names, p.positions)}
            try:
                zero_pose.compare_joint_state(js, zero_pose.better_pose)
                break
            except AssertionError:
                pass
        else:  # if no break
            assert False, 'gaya_pose not in trajectory'

    def test_waypoints_with_fail1(self, zero_pose: PR2):
        zero_pose.set_json_goal('muh')
        zero_pose.add_cmd()
        zero_pose.set_joint_goal(pocky_pose)
        zero_pose.add_cmd()
        zero_pose.set_joint_goal(zero_pose.better_pose)

        traj = zero_pose.send_goal(expected_error_codes=[MoveResult.UNKNOWN_CONSTRAINT,
                                                         MoveResult.SUCCESS,
                                                         MoveResult.SUCCESS],
                                   goal_type=MoveGoal.PLAN_AND_EXECUTE_AND_SKIP_FAILURES)

        for i, p in enumerate(traj.points):
            js = {joint_name: position for joint_name, position in zip(traj.joint_names, p.positions)}
            try:
                zero_pose.compare_joint_state(js, pocky_pose)
                break
            except AssertionError:
                pass
        else:  # if no break
            assert False, 'pocky pose not in trajectory'

        traj.points = traj.points[i:]
        for i, p in enumerate(traj.points):
            js = {joint_name: position for joint_name, position in zip(traj.joint_names, p.positions)}
            try:
                zero_pose.compare_joint_state(js, zero_pose.better_pose)
                break
            except AssertionError:
                pass
        else:  # if no break
            assert False, 'gaya_pose not in trajectory'

    def test_waypoints_with_fail2(self, zero_pose: PR2):
        zero_pose.set_joint_goal(pocky_pose)
        zero_pose.add_cmd()
        zero_pose.set_joint_goal(zero_pose.better_pose)
        zero_pose.add_cmd()
        zero_pose.set_json_goal('muh')

        traj = zero_pose.send_goal(expected_error_codes=[MoveResult.SUCCESS,
                                                         MoveResult.SUCCESS,
                                                         MoveResult.UNKNOWN_CONSTRAINT, ],
                                   goal_type=MoveGoal.PLAN_AND_EXECUTE_AND_SKIP_FAILURES)

        for i, p in enumerate(traj.points):
            js = {joint_name: position for joint_name, position in zip(traj.joint_names, p.positions)}
            try:
                zero_pose.compare_joint_state(js, pocky_pose)
                break
            except AssertionError:
                pass
        else:  # if no break
            assert False, 'pocky pose not in trajectory'

        traj.points = traj.points[i:]
        for i, p in enumerate(traj.points):
            js = {joint_name: position for joint_name, position in zip(traj.joint_names, p.positions)}
            try:
                zero_pose.compare_joint_state(js, zero_pose.better_pose)
                break
            except AssertionError:
                pass
        else:  # if no break
            assert False, 'gaya_pose not in trajectory'

    def test_waypoints_with_fail3(self, zero_pose: PR2):
        zero_pose.set_joint_goal(pocky_pose)
        zero_pose.add_cmd()
        zero_pose.set_json_goal('muh')
        zero_pose.add_cmd()
        zero_pose.set_joint_goal(zero_pose.better_pose)

        traj = zero_pose.send_goal(expected_error_codes=[MoveResult.SUCCESS,
                                                         MoveResult.UNKNOWN_CONSTRAINT,
                                                         MoveResult.ERROR],
                                   goal_type=MoveGoal.PLAN_AND_EXECUTE)

        for i, p in enumerate(traj.points):
            js = {joint_name: position for joint_name, position in zip(traj.joint_names, p.positions)}
            try:
                zero_pose.compare_joint_state(js, zero_pose.default_pose)
                break
            except AssertionError:
                pass
        else:  # if no break
            assert False, 'pocky pose not in trajectory'

    def test_skip_failures1(self, zero_pose: PR2):
        zero_pose.set_json_goal('muh')
        zero_pose.send_goal(expected_error_codes=[MoveResult.UNKNOWN_CONSTRAINT, ],
                            goal_type=MoveGoal.PLAN_AND_EXECUTE_AND_SKIP_FAILURES)

    def test_skip_failures2(self, zero_pose: PR2):
        zero_pose.set_joint_goal(pocky_pose)
        traj = zero_pose.send_goal(expected_error_codes=[MoveResult.SUCCESS, ],
                                   goal_type=MoveGoal.PLAN_AND_EXECUTE_AND_SKIP_FAILURES)

        for i, p in enumerate(traj.points):
            js = {joint_name: position for joint_name, position in zip(traj.joint_names, p.positions)}
            try:
                zero_pose.compare_joint_state(js, pocky_pose)
                break
            except AssertionError:
                pass
        else:  # if no break
            assert False, 'pocky pose not in trajectory'

    # TODO test translation and orientation goal in different frame


class TestCartesianPath(object):

    def test_pathAroundKitchenIsland_without_global_planner(self, kitchen_setup):
        # kernprof -lv py.test -s test/test_integration_pr2.py::TestCartGoals::test_pathAroundKitchenIsland_with_global_planner
        """
        :type kitchen_setup: PR2
        """
        tip_link = u'base_footprint'

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = 0
        base_pose.pose.position.y = 2.0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        kitchen_setup.teleport_base(base_pose)

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = -1.2
        base_pose.pose.position.y = -2.0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        goal_c = base_pose

        kitchen_setup.set_json_goal(u'CartesianPose',
                                    tip_link=tip_link,
                                    root_link=kitchen_setup.default_root,
                                    goal=goal_c
                                    )

        kitchen_setup.plan_and_execute()
        # kitchen_setup.send_goal()
        # kitchen_setup.check_cart_goal(tip_link, goal_c)
        # zero_pose.check_cart_goal(zero_pose.l_tip, l_goal)

    def test_navi_1_native(self, kitchen_setup):
        # kernprof -lv py.test -s test/test_integration_pr2.py::TestCartesianPath::test_pathAroundKitchenIsland_with_global_planner
        """
        :type kitchen_setup: PR2
        """
        tj_1 = [[-0.0, 2.0, 1.83068e-06], [0.0648163, 1.82825, 0.0328465], [0.129633, 1.65649, 0.0656911],
                [0.194449, 1.48474, 0.0985357], [0.259265, 1.31298, 0.13138], [0.324082, 1.14123, 0.164225],
                [0.284965, 1.01414, 0.0301605], [0.245848, 0.88706, -0.103904], [0.206731, 0.759976, -0.237968],
                [0.167614, 0.632892, -0.372033], [0.128497, 0.505808, -0.506097], [0.00333374, 0.387847, -0.440602],
                [-0.121829, 0.269885, -0.375107], [-0.246993, 0.151924, -0.309612], [-0.372156, 0.0339625, -0.244117],
                [-0.418265, 0.0158486, 0.0568035], [-0.464375, -0.00226537, 0.357724],
                [-0.510484, -0.0203793, 0.658645], [-0.556593, -0.0384933, 0.959565], [-0.602702, -0.0566073, 1.26049],
                [-0.620613, -0.0636434, 1.37738], [-0.740769, -0.056253, 1.21814], [-0.860924, -0.0488625, 1.05891],
                [-0.98108, -0.041472, 0.899671], [-1.10124, -0.0340816, 0.740436], [-1.22139, -0.0266911, 0.581201],
                [-1.40784, -0.0167777, 0.607779], [-1.59429, -0.00686438, 0.634357], [-1.78073, 0.00304898, 0.660935],
                [-1.96718, 0.0129623, 0.687512], [-2.15363, 0.0228757, 0.71409], [-2.13332, 0.0857915, 0.446318],
                [-2.113, 0.148707, 0.178547], [-2.09269, 0.211623, -0.0892252], [-2.07237, 0.274539, -0.356997],
                [-2.05206, 0.337455, -0.624769], [-2.0752, 0.508894, -0.570758], [-2.09835, 0.680333, -0.516747],
                [-2.12149, 0.851772, -0.462736], [-2.14464, 1.02321, -0.408725], [-2.16779, 1.19465, -0.354713],
                [-2.11186, 1.4631, -0.236475], [-2.05593, 1.73155, -0.118237], [-2.0, 2.0, 1.83068e-06]]
        tj_2 = [[-0.0, 2.0, 1.72343e-06], [0.0894005, 1.89605, 0.125795], [0.178801, 1.79211, 0.251588],
                [0.268202, 1.68816, 0.377382], [0.357602, 1.58421, 0.503175], [0.447003, 1.48027, 0.628968],
                [0.374934, 1.39671, 0.808284], [0.302866, 1.31316, 0.987599], [0.230798, 1.2296, 1.16691],
                [0.15873, 1.14604, 1.34623], [0.0866619, 1.06249, 1.52555], [0.0708665, 0.970561, 1.739],
                [0.0550711, 0.878635, 1.95245], [0.0392757, 0.786708, 2.1659], [0.0234803, 0.694782, 2.37936],
                [0.00768495, 0.602855, 2.59281], [-0.11887, 0.475883, 2.63427], [-0.245424, 0.348912, 2.67573],
                [-0.371979, 0.22194, 2.71719], [-0.498534, 0.0949686, 2.75865], [-0.625088, -0.032003, 2.8001],
                [-0.618509, -0.0929569, 3.07749], [-0.61193, -0.153911, -2.92831], [-0.605351, -0.214865, -2.65093],
                [-0.598771, -0.275819, -2.37354], [-0.592192, -0.336772, -2.09616], [-0.668661, -0.39985, -1.91778],
                [-0.74513, -0.462929, -1.73941], [-0.821599, -0.526007, -1.56103], [-0.898067, -0.589085, -1.38266],
                [-0.974536, -0.652163, -1.20428], [-1.06107, -0.596989, -1.00953], [-1.1476, -0.541814, -0.81478],
                [-1.23413, -0.48664, -0.62003], [-1.32066, -0.431465, -0.42528], [-1.40719, -0.376291, -0.230531],
                [-1.5004, -0.380811, -0.0171662], [-1.59361, -0.38533, 0.196198], [-1.68682, -0.38985, 0.409562],
                [-1.78003, -0.394369, 0.622927], [-1.87324, -0.398889, 0.836291], [-1.89521, -0.20199, 0.832534],
                [-1.91718, -0.00509106, 0.828776], [-1.93916, 0.191808, 0.825019], [-1.96113, 0.388707, 0.821262],
                [-1.98311, 0.585606, 0.817504], [-2.05678, 0.705064, 0.698206], [-2.13046, 0.824522, 0.578908],
                [-2.20414, 0.94398, 0.45961], [-2.27781, 1.06344, 0.340311], [-2.35149, 1.1829, 0.221013],
                [-2.23432, 1.45526, 0.147343], [-2.11716, 1.72763, 0.0736722], [-2.0, 2.0, 1.72343e-06]]
        tj_3 = [[-0.0, 2.0, 1.68329e-06], [0.0762069, 1.92424, -0.185084], [0.152414, 1.84848, -0.37017],
                [0.228621, 1.77272, -0.555255], [0.304828, 1.69696, -0.740341], [0.381034, 1.6212, -0.925427],
                [0.34605, 1.47261, -1.02012], [0.311066, 1.32402, -1.11482], [0.276082, 1.17544, -1.20952],
                [0.241098, 1.02685, -1.30422], [0.206113, 0.878258, -1.39891], [0.249974, 0.711035, -1.45315],
                [0.293835, 0.543812, -1.50739], [0.337695, 0.376589, -1.56164], [0.381556, 0.209365, -1.61588],
                [0.425416, 0.0421423, -1.67012], [0.371521, -0.0910613, -1.55751], [0.317625, -0.224265, -1.44489],
                [0.26373, -0.357469, -1.33228], [0.209834, -0.490672, -1.21967], [0.155938, -0.623876, -1.10706],
                [0.0541559, -0.578844, -0.977564], [-0.0476265, -0.533813, -0.848072],
                [-0.163259, -0.482653, -0.700959], [-0.278891, -0.431494, -0.553847], [-0.394523, -0.380335, -0.406735],
                [-0.510155, -0.329176, -0.259622], [-0.625787, -0.278017, -0.11251], [-0.741419, -0.226858, 0.0346028],
                [-0.857051, -0.175699, 0.181715], [-0.972683, -0.12454, 0.328828], [-1.08832, -0.0733805, 0.47594],
                [-1.20395, -0.0222214, 0.623053], [-1.36899, 0.0339959, 0.571759], [-1.53403, 0.0902131, 0.520466],
                [-1.69907, 0.14643, 0.469173], [-1.86411, 0.202648, 0.41788], [-2.02916, 0.258865, 0.366587],
                [-2.04012, 0.414823, 0.279272], [-2.05108, 0.570781, 0.191957], [-2.06204, 0.726738, 0.104642],
                [-2.073, 0.882696, 0.0173269], [-2.08396, 1.03865, -0.069988], [-2.05597, 1.3591, -0.0466581],
                [-2.02799, 1.67955, -0.0233282], [-2.0, 2.0, 1.68329e-06]]
        tj_4 = [[-0.0, 2.0, 1.82763e-06], [0.0384361, 1.83586, 0.0628411], [0.0768721, 1.67172, 0.12568],
                [0.115308, 1.50758, 0.18852], [0.153744, 1.34344, 0.251359], [0.19218, 1.1793, 0.314198],
                [0.138522, 1.01517, 0.368843], [0.0848637, 0.851041, 0.423488], [0.0312054, 0.686912, 0.478133],
                [-0.022453, 0.522783, 0.532779], [-0.0761113, 0.358655, 0.587424], [-0.155905, 0.176882, 0.584455],
                [-0.235698, -0.00489145, 0.581486], [-0.315492, -0.186664, 0.578517], [-0.395285, -0.368437, 0.575548],
                [-0.475079, -0.55021, 0.572579], [-0.465833, -0.641697, 0.788674], [-0.456588, -0.733184, 1.00477],
                [-0.447343, -0.82467, 1.22086], [-0.438097, -0.916157, 1.43696], [-0.428852, -1.00764, 1.65305],
                [-0.392021, -0.917352, 1.83767], [-0.35519, -0.82706, 2.02228], [-0.318359, -0.736769, 2.20689],
                [-0.281528, -0.646477, 2.3915], [-0.244697, -0.556186, 2.57612], [-0.262884, -0.494584, 2.84765],
                [-0.281072, -0.432982, 3.11919], [-0.299259, -0.37138, -2.89245], [-0.317446, -0.309778, -2.62092],
                [-0.335634, -0.248176, -2.34938], [-0.341106, -0.229641, -2.26768], [-0.527515, -0.188981, -2.28609],
                [-0.713925, -0.148322, -2.30451], [-0.900334, -0.107662, -2.32292], [-1.08674, -0.0670021, -2.34134],
                [-1.27315, -0.0263424, -2.35975], [-1.46416, 0.0129849, -2.34979], [-1.65517, 0.0523121, -2.33982],
                [-1.84618, 0.0916393, -2.32985], [-2.03719, 0.130966, -2.31988], [-2.2282, 0.170294, -2.30992],
                [-2.22257, 0.260827, -2.09133], [-2.21694, 0.35136, -1.87275], [-2.21131, 0.441893, -1.65416],
                [-2.20568, 0.532426, -1.43558], [-2.20005, 0.622959, -1.21699], [-2.1813, 0.754609, -1.08295],
                [-2.16254, 0.88626, -0.948915], [-2.14378, 1.01791, -0.814875], [-2.12502, 1.14956, -0.680835],
                [-2.10626, 1.28121, -0.546795], [-2.07084, 1.52081, -0.364529], [-2.03542, 1.7604, -0.182264],
                [-2.0, 2.0, 1.82763e-06]]
        tj_5 = [[-0.0, 2.0, 1.98696e-06], [-0.013276, 1.8596, -0.117947], [-0.0265521, 1.7192, -0.235896],
                [-0.0398281, 1.5788, -0.353845], [-0.0531042, 1.4384, -0.471794], [-0.0663802, 1.298, -0.589743],
                [0.0193096, 1.13135, -0.564529], [0.104999, 0.964697, -0.539315], [0.190689, 0.798043, -0.5141],
                [0.276379, 0.63139, -0.488886], [0.362069, 0.464737, -0.463672], [0.330287, 0.337669, -0.325636],
                [0.298506, 0.210601, -0.187601], [0.266725, 0.0835328, -0.0495647], [0.234943, -0.0435351, 0.0884711],
                [0.203162, -0.170603, 0.226507], [0.146312, -0.263542, 0.408612], [0.0894627, -0.356481, 0.590718],
                [0.0326131, -0.449419, 0.772824], [-0.0242365, -0.542358, 0.954929], [-0.0810861, -0.635297, 1.13703],
                [-0.227061, -0.577945, 1.036], [-0.367858, -0.522627, 0.93855], [-0.508656, -0.467309, 0.8411],
                [-0.649453, -0.41199, 0.743649], [-0.790251, -0.356672, 0.646198], [-0.931048, -0.301354, 0.548748],
                [-1.07185, -0.246036, 0.451297], [-1.21264, -0.190718, 0.353847], [-1.35344, -0.1354, 0.256396],
                [-1.49424, -0.0800815, 0.158946], [-1.63504, -0.0247633, 0.061495], [-1.77583, 0.0305548, -0.0359556],
                [-1.91663, 0.085873, -0.133406], [-2.05743, 0.141191, -0.230857], [-2.19823, 0.196509, -0.328307],
                [-2.33902, 0.251827, -0.425758], [-2.29316, 0.409367, -0.353919], [-2.24729, 0.566907, -0.282079],
                [-2.20143, 0.724447, -0.21024], [-2.15556, 0.881986, -0.138401], [-2.1097, 1.03953, -0.0665619],
                [-2.08227, 1.27964, -0.0499209], [-2.05485, 1.51976, -0.03328], [-2.02742, 1.75988, -0.016639],
                [-2.0, 2.0, 1.98696e-06]]
        tj_6 = [[-0.0, 2.0, 1.95959e-06], [0.0532439, 1.87019, -0.119381], [0.106488, 1.74037, -0.238765],
                [0.159732, 1.61056, -0.358148], [0.212975, 1.48075, -0.477531], [0.266219, 1.35093, -0.596914],
                [0.258381, 1.3045, -0.902729], [0.250543, 1.25806, -1.20854], [0.242705, 1.21163, -1.51436],
                [0.234867, 1.16519, -1.82017], [0.227029, 1.11875, -2.12599], [0.194209, 0.94011, -2.23186],
                [0.161389, 0.761465, -2.33772], [0.133405, 0.609147, -2.42799], [0.105422, 0.456828, -2.51825],
                [0.0774382, 0.30451, -2.60852], [0.0494547, 0.152192, -2.69878], [0.0214712, -0.000126025, -2.78905],
                [-0.00651223, -0.152444, -2.87932], [-0.0344957, -0.304762, -2.96958], [-0.0624792, -0.45708, -3.05985],
                [-0.0904626, -0.609399, 3.13307], [-0.118446, -0.761717, 3.04281], [-0.104655, -0.811138, 2.74543],
                [-0.0908644, -0.86056, 2.44805], [-0.0770735, -0.909981, 2.15067], [-0.0632827, -0.959403, 1.85329],
                [-0.0494918, -1.00882, 1.5559], [-0.208003, -0.988363, 1.47556], [-0.366513, -0.967903, 1.39521],
                [-0.525024, -0.947442, 1.31486], [-0.683535, -0.926982, 1.23451], [-0.842046, -0.906521, 1.15416],
                [-0.936719, -0.811758, 1.02207], [-1.03139, -0.716994, 0.88997], [-1.12606, -0.622231, 0.757874],
                [-1.22074, -0.527467, 0.625777], [-1.31541, -0.432704, 0.493681], [-1.47187, -0.388931, 0.418614],
                [-1.62833, -0.345158, 0.343548], [-1.78479, -0.301385, 0.268481], [-1.94125, -0.257612, 0.193415],
                [-2.09771, -0.213839, 0.118349], [-2.12288, -0.0978266, -0.0442268], [-2.14805, 0.0181861, -0.206802],
                [-2.17322, 0.134199, -0.369377], [-2.1984, 0.250212, -0.531953], [-2.22357, 0.366224, -0.694528],
                [-2.20746, 0.564132, -0.691652], [-2.19136, 0.76204, -0.688776], [-2.17525, 0.959948, -0.685901],
                [-2.15914, 1.15786, -0.683025], [-2.14304, 1.35576, -0.680149], [-2.10728, 1.51682, -0.510111],
                [-2.07152, 1.67788, -0.340074], [-2.03576, 1.83894, -0.170036], [-2.0, 2.0, 1.95959e-06]]
        tj_7 = [[-0.0, 2.0, 1.98841e-06], [0.0732607, 1.82583, -0.0220933], [0.146521, 1.65166, -0.0441886],
                [0.219782, 1.47748, -0.0662839], [0.293043, 1.30331, -0.0883791], [0.366303, 1.12914, -0.110474],
                [0.297584, 1.01879, 0.0295327], [0.228864, 0.908445, 0.16954], [0.160144, 0.798097, 0.309547],
                [0.0914242, 0.687749, 0.449554], [0.0227044, 0.577401, 0.589561], [0.0663939, 0.428619, 0.679434],
                [0.110083, 0.279837, 0.769306], [0.153773, 0.131055, 0.859178], [0.197463, -0.0177264, 0.94905],
                [0.241152, -0.166508, 1.03892], [0.0860431, -0.20496, 1.11833], [-0.0690658, -0.243412, 1.19774],
                [-0.224175, -0.281865, 1.27716], [-0.379284, -0.320317, 1.35657], [-0.534393, -0.358769, 1.43598],
                [-0.63993, -0.331649, 1.25391], [-0.745468, -0.304528, 1.07184], [-0.851006, -0.277408, 0.889776],
                [-0.956543, -0.250287, 0.707709], [-1.06208, -0.223167, 0.525642], [-1.16762, -0.196047, 0.343575],
                [-1.27316, -0.168926, 0.161508], [-1.37869, -0.141806, -0.0205591], [-1.48423, -0.114686, -0.202626],
                [-1.58977, -0.0875652, -0.384693], [-1.6796, -0.0644813, -0.539662], [-1.76943, -0.0413974, -0.694631],
                [-1.79372, 0.0691618, -0.521025], [-1.81802, 0.179721, -0.34742], [-1.84232, 0.29028, -0.173814],
                [-1.86661, 0.400839, -0.000208564], [-1.89091, 0.511398, 0.173397], [-1.96871, 0.664873, 0.22926],
                [-2.04651, 0.818347, 0.285122], [-2.12432, 0.971821, 0.340985], [-2.20212, 1.1253, 0.396848],
                [-2.27992, 1.27877, 0.45271], [-2.18661, 1.51918, 0.301808], [-2.09331, 1.75959, 0.150905],
                [-2.0, 2.0, 1.98841e-06]]
        tj_8 = [[-0.0, 2.0, 1.84256e-06], [0.105681, 1.93182, -0.148466], [0.211361, 1.86364, -0.296934],
                [0.317042, 1.79546, -0.445402], [0.422722, 1.72727, -0.59387], [0.528403, 1.65909, -0.742338],
                [0.456694, 1.49898, -0.791471], [0.384985, 1.33887, -0.840604], [0.313277, 1.17877, -0.889737],
                [0.241568, 1.01866, -0.938869], [0.16986, 0.858549, -0.988002], [0.0931581, 0.714105, -0.915092],
                [0.0164564, 0.569662, -0.842182], [-0.0602453, 0.425219, -0.769273], [-0.136947, 0.280776, -0.696363],
                [-0.213649, 0.136333, -0.623453], [-0.37389, 0.0345454, -0.603125], [-0.534131, -0.0672419, -0.582798],
                [-0.694372, -0.169029, -0.56247], [-0.854612, -0.270816, -0.542143], [-1.01485, -0.372604, -0.521815],
                [-1.25958, -0.316495, -0.512949], [-1.50431, -0.260387, -0.504083], [-1.62143, -0.131432, -0.555683],
                [-1.73854, -0.00247763, -0.607282], [-1.85566, 0.126477, -0.658882], [-1.97278, 0.255432, -0.710482],
                [-2.08989, 0.384386, -0.762081], [-2.08656, 0.538362, -0.670104], [-2.08323, 0.692337, -0.578127],
                [-2.07989, 0.846312, -0.48615], [-2.07656, 1.00029, -0.394172], [-2.07323, 1.15426, -0.302195],
                [-2.05492, 1.3657, -0.226646], [-2.03661, 1.57713, -0.151097], [-2.01831, 1.78857, -0.0755473],
                [-2.0, 2.0, 1.84256e-06]]
        tj_9 = [[-0.0, 2.0, 1.79292e-06], [0.0775313, 1.83191, -0.0297871], [0.155063, 1.66383, -0.0595759],
                [0.232594, 1.49574, -0.0893648], [0.310125, 1.32766, -0.119154], [0.387657, 1.15957, -0.148943],
                [0.305637, 1.03652, -0.253191], [0.223618, 0.91348, -0.35744], [0.141598, 0.790435, -0.461689],
                [0.059579, 0.66739, -0.565937], [-0.0224405, 0.544345, -0.670186], [0.0715215, 0.421021, -0.760105],
                [0.165483, 0.297698, -0.850023], [0.259445, 0.174374, -0.939942], [0.353407, 0.05105, -1.02986],
                [0.447369, -0.0722738, -1.11978], [0.465004, -0.206815, -0.991163], [0.482638, -0.341356, -0.862546],
                [0.500273, -0.475896, -0.733929], [0.517907, -0.610437, -0.605312], [0.535541, -0.744978, -0.476696],
                [0.4646, -0.704386, -0.240162], [0.393659, -0.663794, -0.0036291], [0.322718, -0.623202, 0.232904],
                [0.251777, -0.58261, 0.469437], [0.180836, -0.542018, 0.705971], [0.109895, -0.501426, 0.942504],
                [0.0389539, -0.460834, 1.17904], [-0.0319872, -0.420242, 1.41557], [-0.102928, -0.37965, 1.6521],
                [-0.173869, -0.339058, 1.88864], [-0.249442, -0.295816, 2.14061], [-0.318568, -0.267953, 1.88967],
                [-0.387693, -0.240091, 1.63873], [-0.456818, -0.212228, 1.38779], [-0.525943, -0.184365, 1.13685],
                [-0.595068, -0.156503, 0.885906], [-0.780526, -0.187033, 0.861814], [-0.965984, -0.217563, 0.837723],
                [-1.15144, -0.248093, 0.813631], [-1.3369, -0.278624, 0.789539], [-1.52236, -0.309154, 0.765447],
                [-1.68718, -0.289669, 0.697391], [-1.85201, -0.270183, 0.629335], [-2.01683, -0.250698, 0.561279],
                [-2.18165, -0.231213, 0.493223], [-2.34648, -0.211728, 0.425167], [-2.27315, -0.0795684, 0.327448],
                [-2.19982, 0.0525912, 0.22973], [-2.12649, 0.184751, 0.132011], [-2.05316, 0.31691, 0.034293],
                [-1.97983, 0.44907, -0.0634254], [-1.96992, 0.607499, -0.145948], [-1.96001, 0.765928, -0.228471],
                [-1.95011, 0.924357, -0.310994], [-1.9402, 1.08279, -0.393517], [-1.93029, 1.24122, -0.476039],
                [-1.94772, 1.43091, -0.357029], [-1.96515, 1.62061, -0.238019], [-1.98257, 1.8103, -0.119009],
                [-2.0, 2.0, 1.79292e-06]]
        tj_10 = [[-0.0, 2.0, 1.8623e-06], [0.0395045, 1.84553, 0.0811279], [0.0790089, 1.69107, 0.162254],
                 [0.118513, 1.5366, 0.24338], [0.158018, 1.38214, 0.324506], [0.197522, 1.22767, 0.405632],
                 [0.204777, 1.29557, 0.669064], [0.212032, 1.36347, 0.932496], [0.219287, 1.43137, 1.19593],
                 [0.226542, 1.49926, 1.45936], [0.233797, 1.56716, 1.72279], [0.24692, 1.39014, 1.73208],
                 [0.260044, 1.21312, 1.74137], [0.273167, 1.0361, 1.75066], [0.28629, 0.859085, 1.75995],
                 [0.299414, 0.682066, 1.76924], [0.228787, 0.532735, 1.83886], [0.15816, 0.383405, 1.90848],
                 [0.087533, 0.234074, 1.9781], [0.016906, 0.0847431, 2.04772], [-0.0537209, -0.0645877, 2.11733],
                 [-0.18287, -0.0984067, 2.25033], [-0.31202, -0.132226, 2.38332], [-0.441169, -0.166045, 2.51631],
                 [-0.570319, -0.199864, 2.6493], [-0.699468, -0.233683, 2.78229], [-0.824884, -0.22807, 2.93121],
                 [-0.950299, -0.222458, 3.08013], [-1.07571, -0.216845, -3.05414], [-1.20113, -0.211233, -2.90522],
                 [-1.32655, -0.205621, -2.7563], [-1.37982, -0.251884, -3.10808], [-1.43309, -0.298148, 2.82333],
                 [-1.56266, -0.205995, 2.74132], [-1.69223, -0.113843, 2.65932], [-1.8218, -0.0216905, 2.57731],
                 [-1.95137, 0.0704618, 2.49531], [-2.08094, 0.162614, 2.41331], [-2.14108, 0.319648, 2.47699],
                 [-2.20122, 0.476682, 2.54068], [-2.26137, 0.633716, 2.60436], [-2.32151, 0.790751, 2.66805],
                 [-2.38165, 0.947785, 2.73174], [-2.32691, 1.11425, 2.78127], [-2.27217, 1.28071, 2.83081],
                 [-2.21743, 1.44717, 2.88034], [-2.16269, 1.61363, 2.92988], [-2.10795, 1.7801, 2.97942],
                 [-2.12161, 1.78063, -2.93111], [-2.13527, 1.78116, -2.55845], [-2.14893, 1.78169, -2.1858],
                 [-2.16259, 1.78222, -1.81314], [-2.17625, 1.78275, -1.44048], [-2.13219, 1.83706, -1.08036],
                 [-2.08813, 1.89137, -0.720241], [-2.04406, 1.94569, -0.360119], [-2.0, 2.0, 1.8623e-06]]
        tj_11 = [[-0.0, 2.0, 2.00349e-06], [0.0489151, 1.84527, -0.0754369], [0.0978302, 1.69053, -0.150876],
                 [0.146745, 1.5358, -0.226315], [0.19566, 1.38107, -0.301754], [0.244575, 1.22634, -0.377193],
                 [0.253587, 1.03734, -0.355613], [0.262599, 0.848344, -0.334034], [0.271611, 0.659348, -0.312455],
                 [0.280623, 0.470352, -0.290876], [0.289635, 0.281357, -0.269297], [0.155668, 0.165215, -0.223901],
                 [0.0217016, 0.0490723, -0.178506], [-0.112265, -0.06707, -0.133111], [-0.246232, -0.183212, -0.087716],
                 [-0.380199, -0.299355, -0.0423209], [-0.51013, -0.328502, -0.176], [-0.640061, -0.357649, -0.30968],
                 [-0.769992, -0.386796, -0.44336], [-0.899923, -0.415943, -0.577039], [-1.02985, -0.44509, -0.710719],
                 [-1.15398, -0.400246, -0.785503], [-1.27811, -0.355402, -0.860288], [-1.42468, -0.302448, -0.948596],
                 [-1.57126, -0.249495, -1.0369], [-1.71783, -0.196541, -1.12521], [-1.86441, -0.143588, -1.21352],
                 [-2.01098, -0.0906347, -1.30183], [-2.00859, 0.0533926, -1.18992], [-2.00621, 0.19742, -1.07802],
                 [-2.00382, 0.341447, -0.96611], [-2.00143, 0.485474, -0.854204], [-1.99905, 0.629502, -0.742298],
                 [-2.0557, 0.849816, -0.787333], [-2.11236, 1.07013, -0.832368], [-2.16901, 1.29044, -0.877403],
                 [-2.22567, 1.51076, -0.922438], [-2.16925, 1.63307, -0.691828], [-2.11284, 1.75538, -0.461218],
                 [-2.05642, 1.87769, -0.230608], [-2.0, 2.0, 2.00349e-06]]
        tj_12 = [[-0.0, 2.0, 1.68144e-06], [0.0563253, 1.82846, 0.0389025], [0.112651, 1.65692, 0.0778033],
                 [0.168976, 1.48538, 0.116704], [0.225301, 1.31384, 0.155605], [0.281627, 1.14231, 0.194506],
                 [0.361877, 0.964462, 0.184728], [0.442128, 0.786619, 0.17495], [0.522378, 0.608776, 0.165172],
                 [0.602629, 0.430933, 0.155394], [0.68288, 0.25309, 0.145616], [0.515513, 0.197226, 0.192728],
                 [0.348146, 0.141361, 0.239841], [0.18078, 0.0854971, 0.286953], [0.0134129, 0.0296328, 0.334065],
                 [-0.153954, -0.0262315, 0.381178], [-0.0938381, -0.186801, 0.211295], [-0.277252, -0.203682, 0.231603],
                 [-0.466005, -0.221054, 0.252502], [-0.654758, -0.238426, 0.273401], [-0.84351, -0.255798, 0.2943],
                 [-1.03226, -0.27317, 0.315199], [-1.22102, -0.290543, 0.336098], [-1.35965, -0.148379, 0.338964],
                 [-1.49828, -0.00621548, 0.341831], [-1.63691, 0.135948, 0.344697], [-1.77554, 0.278112, 0.347564],
                 [-1.91417, 0.420275, 0.35043], [-2.01407, 0.588993, 0.342582], [-2.11397, 0.75771, 0.334734],
                 [-2.21387, 0.926427, 0.326885], [-2.31377, 1.09514, 0.319037], [-2.41367, 1.26386, 0.311189],
                 [-2.31025, 1.4479, 0.233392], [-2.20684, 1.63193, 0.155595], [-2.10342, 1.81597, 0.0777984],
                 [-2.0, 2.0, 1.68144e-06]]
        tj_13 = [[-0.0, 2.0, 1.85912e-06], [0.0731189, 1.90907, -0.166635], [0.146238, 1.81814, -0.333271],
                 [0.219357, 1.72721, -0.499908], [0.292476, 1.63628, -0.666545], [0.365595, 1.54535, -0.833181],
                 [0.371497, 1.57161, -1.17936], [0.3774, 1.59787, -1.52553], [0.383303, 1.62412, -1.87171],
                 [0.389206, 1.65038, -2.21788], [0.395108, 1.67664, -2.56405], [0.326351, 1.55661, -2.68741],
                 [0.257594, 1.43659, -2.81076], [0.188836, 1.31657, -2.93412], [0.120079, 1.19654, -3.05747],
                 [0.0513211, 1.07652, 3.10236], [0.00780073, 0.94734, 2.97499], [-0.0357196, 0.818159, 2.84762],
                 [-0.07924, 0.688979, 2.72024], [-0.12276, 0.559799, 2.59287], [-0.166281, 0.430619, 2.4655],
                 [-0.209801, 0.301439, 2.33813], [-0.253321, 0.172259, 2.21076], [-0.296842, 0.0430786, 2.08339],
                 [-0.340362, -0.0861015, 1.95601], [-0.383883, -0.215282, 1.82864], [-0.427403, -0.344462, 1.70127],
                 [-0.470923, -0.473642, 1.5739], [-0.514444, -0.602822, 1.44653], [-0.557964, -0.732002, 1.31915],
                 [-0.601484, -0.861182, 1.19178], [-0.632837, -0.954245, 1.10002], [-0.751483, -0.820694, 1.14274],
                 [-0.87013, -0.687142, 1.18545], [-0.988776, -0.553591, 1.22817], [-1.10742, -0.420039, 1.27089],
                 [-1.22607, -0.286487, 1.3136], [-1.36359, -0.23257, 1.20902], [-1.5011, -0.178653, 1.10444],
                 [-1.63862, -0.124736, 0.999859], [-1.77614, -0.0708187, 0.895278], [-1.91366, -0.0169015, 0.790697],
                 [-1.95009, 0.107587, 0.650121], [-1.98653, 0.232076, 0.509544], [-2.02297, 0.356565, 0.368968],
                 [-2.0594, 0.481054, 0.228391], [-2.09584, 0.605543, 0.0878149], [-2.13998, 0.789491, 0.209473],
                 [-2.18413, 0.973439, 0.331132], [-2.22827, 1.15739, 0.45279], [-2.27242, 1.34134, 0.574448],
                 [-2.20431, 1.506, 0.430837], [-2.13621, 1.67067, 0.287225], [-2.0681, 1.83533, 0.143613],
                 [-2.0, 2.0, 1.85912e-06]]
        tj_14 = [[-0.0, 2.0, 1.62384e-06], [-0.0235331, 1.82641, -0.0496454], [-0.0470662, 1.65282, -0.0992925],
                 [-0.0705993, 1.47923, -0.14894], [-0.0941324, 1.30565, -0.198587], [-0.117666, 1.13206, -0.248234],
                 [-0.115834, 0.980809, -0.345716], [-0.114002, 0.829561, -0.443198], [-0.11217, 0.678314, -0.54068],
                 [-0.110338, 0.527066, -0.638162], [-0.108507, 0.375818, -0.735644], [-0.178448, 0.253721, -0.854222],
                 [-0.248389, 0.131623, -0.972801], [-0.318331, 0.00952613, -1.09138], [-0.388272, -0.112571, -1.20996],
                 [-0.458214, -0.234668, -1.32854], [-0.582297, -0.279934, -1.1927], [-0.706381, -0.3252, -1.05687],
                 [-0.830465, -0.370465, -0.92103], [-0.954549, -0.415731, -0.785195], [-1.07863, -0.460997, -0.64936],
                 [-1.24219, -0.358485, -0.635415], [-1.40575, -0.255973, -0.621469], [-1.5693, -0.153461, -0.607524],
                 [-1.73286, -0.0509491, -0.593579], [-1.89642, 0.0515628, -0.579633], [-1.90756, 0.261105, -0.517297],
                 [-1.9168, 0.434994, -0.465567], [-1.92605, 0.608884, -0.413837], [-1.93529, 0.782773, -0.362108],
                 [-1.94453, 0.956663, -0.310378], [-1.95378, 1.13055, -0.258648], [-1.96533, 1.34791, -0.193985],
                 [-1.97689, 1.56528, -0.129323], [-1.98844, 1.78264, -0.0646607], [-2.0, 2.0, 1.62384e-06]]
        tj_15 = [[-0.0, 2.0, 1.72539e-06], [0.0773114, 1.84554, 0.0545518], [0.154623, 1.69109, 0.109102],
                 [0.231934, 1.53663, 0.163652], [0.309245, 1.38217, 0.218202], [0.386557, 1.22772, 0.272752],
                 [0.316899, 1.09462, 0.372313], [0.247242, 0.961531, 0.471874], [0.177585, 0.828438, 0.571436],
                 [0.107927, 0.695345, 0.670997], [0.0382698, 0.562252, 0.770558], [0.0506016, 0.365477, 0.776236],
                 [0.0629334, 0.168703, 0.781915], [0.0752652, -0.0280722, 0.787593], [0.087597, -0.224847, 0.793271],
                 [0.0999288, -0.421622, 0.79895], [-0.0253063, -0.409908, 0.667968], [-0.150541, -0.398194, 0.536986],
                 [-0.275777, -0.38648, 0.406004], [-0.401012, -0.374766, 0.275022], [-0.531961, -0.362517, 0.138063],
                 [-0.66291, -0.350269, 0.0011048], [-0.793859, -0.33802, -0.135854], [-0.924809, -0.325772, -0.272812],
                 [-1.05576, -0.313523, -0.40977], [-1.18671, -0.301275, -0.546729], [-1.31766, -0.289027, -0.683687],
                 [-1.44861, -0.276778, -0.820646], [-1.57955, -0.26453, -0.957604], [-1.7105, -0.252281, -1.09456],
                 [-1.75829, -0.123283, -0.969694], [-1.80608, 0.00571582, -0.844825], [-1.85387, 0.134714, -0.719956],
                 [-1.90166, 0.263713, -0.595087], [-1.94944, 0.392711, -0.470219], [-2.02614, 0.531812, -0.387907],
                 [-2.10284, 0.670912, -0.305596], [-2.17954, 0.810013, -0.223284], [-2.25623, 0.949114, -0.140973],
                 [-2.33293, 1.08821, -0.0586614], [-2.22195, 1.39214, -0.039107], [-2.11098, 1.69607, -0.0195526],
                 [-2.0, 2.0, 1.72539e-06]]
        tj_16 = [[-0.0, 2.0, 1.94659e-06], [0.0332138, 1.83864, -0.0705206], [0.0664276, 1.67729, -0.141043],
                 [0.0996414, 1.51593, -0.211566], [0.132855, 1.35458, -0.282088], [0.166069, 1.19322, -0.352611],
                 [0.259565, 1.06848, -0.264395], [0.353062, 0.943735, -0.176179], [0.446558, 0.818992, -0.0879635],
                 [0.540054, 0.694249, 0.000252192], [0.633551, 0.569506, 0.0884679], [0.575196, 0.395167, 0.0561611],
                 [0.516842, 0.220827, 0.0238543], [0.458487, 0.0464871, -0.00845256], [0.400133, -0.127853, -0.0407594],
                 [0.341778, -0.302192, -0.0730662], [0.199642, -0.287703, -0.0180613],
                 [0.0575049, -0.273213, 0.0369436], [-0.0846318, -0.258723, 0.0919485],
                 [-0.251483, -0.241714, 0.156517], [-0.418333, -0.224706, 0.221086], [-0.585184, -0.207697, 0.285655],
                 [-0.752035, -0.190688, 0.350224], [-0.918886, -0.173679, 0.414793], [-1.08574, -0.15667, 0.479362],
                 [-1.25259, -0.139661, 0.543931], [-1.41944, -0.122652, 0.6085], [-1.58629, -0.105643, 0.673069],
                 [-1.75314, -0.0886337, 0.737638], [-1.82804, 0.0339812, 0.625004], [-1.90295, 0.156596, 0.512371],
                 [-1.97785, 0.279211, 0.399738], [-2.05275, 0.401826, 0.287105], [-2.12766, 0.524441, 0.174472],
                 [-2.15319, 0.701268, 0.0317939], [-2.17872, 0.878095, -0.110884], [-2.20426, 1.05492, -0.253561],
                 [-2.22979, 1.23175, -0.396239], [-2.17234, 1.42381, -0.297179], [-2.11489, 1.61588, -0.198118],
                 [-2.05745, 1.80794, -0.0990583], [-2.0, 2.0, 1.94659e-06]]
        tj_17 = [[-0.0, 2.0, 1.99502e-06], [0.0791482, 1.89863, 0.142787], [0.158296, 1.79726, 0.285572],
                 [0.237445, 1.6959, 0.428357], [0.316593, 1.59453, 0.571142], [0.395741, 1.49316, 0.713927],
                 [0.426633, 1.33838, 0.629586], [0.457525, 1.18361, 0.545245], [0.488416, 1.02883, 0.460904],
                 [0.519308, 0.874053, 0.376563], [0.5502, 0.719277, 0.292222], [0.452605, 0.583393, 0.226821],
                 [0.35501, 0.447508, 0.161421], [0.257416, 0.311624, 0.0960203], [0.159821, 0.17574, 0.0306198],
                 [0.0622261, 0.0398562, -0.0347806], [-0.0811002, -0.0859955, -0.0162572],
                 [-0.224427, -0.211847, 0.0022663], [-0.367753, -0.337699, 0.0207898],
                 [-0.511079, -0.463551, 0.0393132], [-0.654406, -0.589402, 0.0578367], [-0.751637, -0.53142, -0.115749],
                 [-0.848868, -0.473438, -0.289335], [-0.946099, -0.415455, -0.462921], [-1.04333, -0.357473, -0.636507],
                 [-1.14056, -0.299491, -0.810092], [-1.28705, -0.296867, -0.917061], [-1.43355, -0.294243, -1.02403],
                 [-1.58004, -0.291618, -1.131], [-1.72653, -0.288994, -1.23797], [-1.87302, -0.28637, -1.34494],
                 [-1.87697, -0.268826, -1.33349], [-1.91026, -0.120814, -1.23691], [-1.94355, 0.0271968, -1.14032],
                 [-1.97684, 0.175208, -1.04374], [-2.01014, 0.323219, -0.947162], [-2.04343, 0.471231, -0.85058],
                 [-2.07672, 0.619242, -0.753999], [-2.11001, 0.767253, -0.657417], [-2.1433, 0.915265, -0.560836],
                 [-2.1766, 1.06328, -0.464255], [-2.20989, 1.21129, -0.367673], [-2.13993, 1.47419, -0.245115],
                 [-2.06996, 1.7371, -0.122556], [-2.0, 2.0, 1.99502e-06]]
        tj_18 = [[-0.0, 2.0, 1.70943e-06], [-0.000245407, 1.86174, -0.123483], [-0.000490815, 1.72349, -0.246968],
                 [-0.000736222, 1.58523, -0.370453], [-0.00098163, 1.44697, -0.493937],
                 [-0.00122704, 1.30871, -0.617422], [-0.0186926, 1.17797, -0.481224], [-0.0361582, 1.04723, -0.345025],
                 [-0.0536239, 0.916495, -0.208826], [-0.0710895, 0.785756, -0.0726277],
                 [-0.0885551, 0.655017, 0.063571], [-0.137519, 0.531063, -0.06988], [-0.186483, 0.407109, -0.203331],
                 [-0.235447, 0.283154, -0.336782], [-0.28441, 0.1592, -0.470233], [-0.333374, 0.035246, -0.603684],
                 [-0.449892, -0.0360737, -0.501398], [-0.566409, -0.107394, -0.399113],
                 [-0.690526, -0.183365, -0.290156], [-0.814643, -0.259336, -0.1812], [-0.938759, -0.335307, -0.0722434],
                 [-1.06288, -0.411278, 0.0367131], [-1.18699, -0.48725, 0.14567], [-1.34215, -0.361718, 0.146505],
                 [-1.49731, -0.236186, 0.14734], [-1.65248, -0.110654, 0.148175], [-1.80764, 0.0148776, 0.14901],
                 [-1.9628, 0.140409, 0.149846], [-1.95403, 0.38251, 0.165327], [-1.94525, 0.62461, 0.180808],
                 [-1.93648, 0.866711, 0.19629], [-1.92771, 1.10881, 0.211771], [-1.94578, 1.33161, 0.158829],
                 [-1.96386, 1.55441, 0.105886], [-1.98193, 1.7772, 0.052944], [-2.0, 2.0, 1.70943e-06]]
        tj_19 = [[-0.0, 2.0, 1.70224e-06], [0.0265945, 1.8354, -0.0665374], [0.0531891, 1.67081, -0.133076],
                 [0.0797836, 1.50621, -0.199616], [0.106378, 1.34162, -0.266155], [0.132973, 1.17702, -0.332694],
                 [0.10739, 1.04442, -0.202792], [0.0818075, 0.911813, -0.0728911], [0.056225, 0.779209, 0.0570102],
                 [0.0306424, 0.646604, 0.186911], [0.00505988, 0.514, 0.316813], [-0.0676949, 0.401286, 0.185124],
                 [-0.14045, 0.288572, 0.0534361], [-0.213205, 0.175857, -0.0782522], [-0.285959, 0.0631429, -0.209941],
                 [-0.358714, -0.0495715, -0.341629], [-0.511207, -0.118989, -0.40653], [-0.6637, -0.188407, -0.47143],
                 [-0.816193, -0.257824, -0.536331], [-0.968686, -0.327242, -0.601232], [-1.12118, -0.39666, -0.666133],
                 [-1.25523, -0.303666, -0.592429], [-1.38928, -0.210673, -0.518725], [-1.52333, -0.117679, -0.445022],
                 [-1.65738, -0.0246854, -0.371318], [-1.79143, 0.0683081, -0.297615], [-1.87041, 0.188789, -0.409497],
                 [-1.94938, 0.30927, -0.521379], [-2.02836, 0.42975, -0.633262], [-2.10734, 0.550231, -0.745144],
                 [-2.18632, 0.670712, -0.857026], [-2.21564, 0.841968, -0.909527], [-2.24497, 1.01323, -0.962027],
                 [-2.2743, 1.18448, -1.01453], [-2.30363, 1.35574, -1.06703], [-2.33296, 1.527, -1.11953],
                 [-2.32986, 1.44243, -0.882663], [-2.32677, 1.35786, -0.645798], [-2.32368, 1.27329, -0.408933],
                 [-2.24276, 1.45497, -0.306699], [-2.16184, 1.63665, -0.204466], [-2.08092, 1.81832, -0.102232],
                 [-2.0, 2.0, 1.70224e-06]]
        tj_20 = [[-0.0, 2.0, 1.70765e-06], [0.0711684, 1.85267, 0.072761], [0.142337, 1.70534, 0.14552],
                 [0.213505, 1.558, 0.21828], [0.284674, 1.41067, 0.291039], [0.355842, 1.26334, 0.363798],
                 [0.264191, 1.08563, 0.363886], [0.172539, 0.907911, 0.363973], [0.0808877, 0.730196, 0.36406],
                 [-0.0107637, 0.552481, 0.364148], [-0.102415, 0.374767, 0.364235], [-0.194067, 0.197052, 0.364322],
                 [-0.285718, 0.0193371, 0.36441], [-0.377369, -0.158378, 0.364497], [-0.469021, -0.336093, 0.364584],
                 [-0.560672, -0.513807, 0.364672], [-0.666607, -0.719217, 0.364772], [-0.69552, -0.608105, 0.194398],
                 [-0.724433, -0.496993, 0.0240225], [-0.753346, -0.385881, -0.146352], [-0.78226, -0.274768, -0.316727],
                 [-0.811173, -0.163656, -0.487102], [-0.964913, -0.172319, -0.39507], [-1.11865, -0.180983, -0.303038],
                 [-1.27239, -0.189646, -0.211006], [-1.42613, -0.19831, -0.118974], [-1.57987, -0.206973, -0.0269425],
                 [-1.65699, -0.0878367, -0.143106], [-1.73411, 0.0312996, -0.25927], [-1.81123, 0.150436, -0.375434],
                 [-1.88835, 0.269572, -0.491598], [-1.96547, 0.388709, -0.607762], [-2.02179, 0.550253, -0.549927],
                 [-2.07812, 0.711797, -0.492092], [-2.13445, 0.873341, -0.434257], [-2.19077, 1.03488, -0.376422],
                 [-2.2471, 1.19643, -0.318587], [-2.18533, 1.39732, -0.23894], [-2.12355, 1.59821, -0.159292],
                 [-2.06178, 1.79911, -0.0796454], [-2.0, 2.0, 1.70765e-06]]
        tj_21 = [[-0.0, 2.0, 1.89766e-06], [0.0911176, 1.84464, -0.0397769], [0.182235, 1.68928, -0.0795556],
                 [0.273353, 1.53391, -0.119334], [0.364471, 1.37855, -0.159113], [0.455588, 1.22319, -0.198892],
                 [0.437428, 1.06689, -0.284193], [0.419268, 0.910593, -0.369495], [0.401107, 0.754295, -0.454796],
                 [0.382947, 0.597998, -0.540097], [0.364787, 0.4417, -0.625399], [0.372676, 0.326894, -0.455552],
                 [0.380565, 0.212088, -0.285705], [0.388454, 0.0972821, -0.115858], [0.396344, -0.0175239, 0.0539883],
                 [0.404233, -0.13233, 0.223835], [0.204763, -0.145127, 0.224074], [0.00529238, -0.157925, 0.224314],
                 [-0.194178, -0.170723, 0.224553], [-0.393648, -0.18352, 0.224792], [-0.593118, -0.196318, 0.225032],
                 [-0.750726, -0.212788, 0.141963], [-0.908333, -0.229258, 0.0588944], [-1.06594, -0.245727, -0.0241742],
                 [-1.22355, -0.262197, -0.107243], [-1.38116, -0.278667, -0.190311], [-1.53876, -0.295137, -0.27338],
                 [-1.69637, -0.311606, -0.356448], [-1.85398, -0.328076, -0.439517], [-2.01159, -0.344546, -0.522586],
                 [-2.16919, -0.361016, -0.605654], [-2.34794, -0.379694, -0.699865], [-2.31631, -0.206884, -0.748502],
                 [-2.28468, -0.0340726, -0.797139], [-2.25306, 0.138738, -0.845776], [-2.22143, 0.311549, -0.894413],
                 [-2.1898, 0.48436, -0.94305], [-2.17431, 0.652179, -0.880115], [-2.15883, 0.819999, -0.817179],
                 [-2.14335, 0.987818, -0.754243], [-2.12786, 1.15564, -0.691308], [-2.11238, 1.32346, -0.628372],
                 [-2.08428, 1.49259, -0.471279], [-2.05619, 1.66173, -0.314185], [-2.02809, 1.83086, -0.157092],
                 [-2.0, 2.0, 1.89766e-06]]
        tj_22 = [[-0.0, 2.0, 1.74722e-06], [0.00480128, 1.82686, 0.0535959], [0.00960257, 1.65373, 0.10719],
                 [0.0144038, 1.48059, 0.160784], [0.0192051, 1.30745, 0.214378], [0.0240064, 1.13432, 0.267973],
                 [0.0613656, 1.17767, 0.553511], [0.0987249, 1.22103, 0.83905], [0.136084, 1.26438, 1.12459],
                 [0.173443, 1.30774, 1.41013], [0.210803, 1.35109, 1.69567], [0.159242, 1.18631, 1.75034],
                 [0.107681, 1.02152, 1.80501], [0.0561209, 0.856733, 1.85968], [0.00456038, 0.691946, 1.91435],
                 [-0.0470002, 0.52716, 1.96902], [0.0451675, 0.374317, 2.01206], [0.137335, 0.221475, 2.05509],
                 [0.229503, 0.068632, 2.09813], [0.32167, -0.0842107, 2.14117], [0.413838, -0.237053, 2.1842],
                 [0.276411, -0.216496, 2.08334], [0.138983, -0.195939, 1.98248], [0.00155573, -0.175382, 1.88162],
                 [-0.143573, -0.153674, 1.7751], [-0.288701, -0.131965, 1.66859], [-0.433829, -0.110256, 1.56207],
                 [-0.578958, -0.0885469, 1.45556], [-0.724086, -0.0668381, 1.34905], [-0.869214, -0.0451292, 1.24253],
                 [-1.01434, -0.0234203, 1.13602], [-1.15947, -0.00171144, 1.0295], [-1.3046, 0.0199974, 0.92299],
                 [-1.44973, 0.0417063, 0.816476], [-1.59486, 0.0634152, 0.709962], [-1.73998, 0.0851241, 0.603448],
                 [-1.88511, 0.106833, 0.496934], [-2.03024, 0.128542, 0.39042], [-2.17537, 0.150251, 0.283907],
                 [-2.15288, 0.382467, 0.250511], [-2.1304, 0.614682, 0.217115], [-2.10791, 0.846898, 0.183719],
                 [-2.08542, 1.07911, 0.150323], [-2.06407, 1.30934, 0.112743], [-2.04271, 1.53956, 0.0751624],
                 [-2.02136, 1.76978, 0.0375821], [-2.0, 2.0, 1.74722e-06]]
        tj_23 = [[-0.0, 2.0, 1.86265e-06], [0.0738296, 1.85065, -0.0667951], [0.147659, 1.7013, -0.133592],
                 [0.221489, 1.55195, -0.200389], [0.295318, 1.4026, -0.267186], [0.369148, 1.25325, -0.333983],
                 [0.284605, 1.09369, -0.295137], [0.200062, 0.934125, -0.256291], [0.115519, 0.774561, -0.217446],
                 [0.0309767, 0.614998, -0.1786], [-0.0535662, 0.455434, -0.139754], [-0.124343, 0.33919, -0.267562],
                 [-0.19512, 0.222946, -0.395371], [-0.265897, 0.106702, -0.523179], [-0.336674, -0.00954225, -0.650988],
                 [-0.407451, -0.125786, -0.778796], [-0.581519, -0.157186, -0.83122], [-0.755587, -0.188586, -0.883644],
                 [-0.929655, -0.219985, -0.936068], [-1.10107, -0.250907, -0.987694], [-1.27249, -0.281829, -1.03932],
                 [-1.44392, -0.312751, -1.09095], [-1.61534, -0.343673, -1.14257], [-1.78676, -0.374595, -1.1942],
                 [-1.80666, -0.239141, -1.06802], [-1.82656, -0.103688, -0.941833], [-1.84647, 0.0317663, -0.81565],
                 [-1.86637, 0.16722, -0.689467], [-1.88628, 0.302674, -0.563284], [-1.92314, 0.492901, -0.550818],
                 [-1.96001, 0.683129, -0.538351], [-1.99688, 0.873356, -0.525885], [-2.03374, 1.06358, -0.513418],
                 [-2.07061, 1.25381, -0.500952], [-2.05296, 1.44036, -0.375714], [-2.0353, 1.62691, -0.250475],
                 [-2.01765, 1.81345, -0.125237], [-2.0, 2.0, 1.86265e-06]]
        tj_24 = [[-0.0, 2.0, 1.88114e-06], [-0.00853238, 1.81433, -0.0282711], [-0.0170648, 1.62866, -0.0565441],
                 [-0.0255971, 1.443, -0.0848172], [-0.0341295, 1.25733, -0.11309], [-0.0426619, 1.07166, -0.141363],
                 [-0.0817898, 0.912777, -0.214098], [-0.120918, 0.753891, -0.286833], [-0.160046, 0.595006, -0.359568],
                 [-0.199173, 0.43612, -0.432303], [-0.238301, 0.277235, -0.505038], [-0.329239, 0.190898, -0.65425],
                 [-0.420177, 0.104562, -0.803462], [-0.511115, 0.0182253, -0.952673], [-0.602053, -0.0681112, -1.10189],
                 [-0.692991, -0.154448, -1.2511], [-0.628923, -0.295604, -1.34107], [-0.564855, -0.43676, -1.43103],
                 [-0.500787, -0.577917, -1.521], [-0.436719, -0.719073, -1.61097], [-0.372651, -0.86023, -1.70094],
                 [-0.509012, -0.774711, -1.56494], [-0.645372, -0.689192, -1.42894], [-0.781733, -0.603673, -1.29294],
                 [-0.918094, -0.518154, -1.15694], [-1.03721, -0.443452, -1.03814], [-1.15632, -0.36875, -0.919345],
                 [-1.27544, -0.294047, -0.800546], [-1.39455, -0.219345, -0.681747], [-1.51366, -0.144642, -0.562949],
                 [-1.63278, -0.06994, -0.44415], [-1.75189, 0.00476245, -0.325352], [-1.871, 0.0794649, -0.206553],
                 [-1.99012, 0.154167, -0.0877545], [-2.10923, 0.22887, 0.0310441], [-2.09245, 0.407177, 0.0728529],
                 [-2.07567, 0.585485, 0.114662], [-2.05888, 0.763792, 0.15647], [-2.0421, 0.9421, 0.198279],
                 [-2.02532, 1.12041, 0.240088], [-2.01899, 1.34031, 0.180066], [-2.01266, 1.5602, 0.120045],
                 [-2.00633, 1.7801, 0.0600234], [-2.0, 2.0, 1.88114e-06]]
        tj_25 = [[-0.0, 2.0, 1.68948e-06], [0.0139238, 1.82806, -0.0549987], [0.0278476, 1.65613, -0.109999],
                 [0.0417714, 1.48419, -0.165], [0.0556952, 1.31225, -0.22], [0.069619, 1.14032, -0.275],
                 [0.107971, 1.02287, -0.427895], [0.146322, 0.905416, -0.580789], [0.184674, 0.787966, -0.733684],
                 [0.223025, 0.670517, -0.886578], [0.261377, 0.553067, -1.03947], [0.228066, 0.387334, -1.10138],
                 [0.194754, 0.221601, -1.16328], [0.161443, 0.0558682, -1.22519], [0.128132, -0.109865, -1.28709],
                 [0.0948209, -0.275597, -1.349], [-0.0964871, -0.332327, -1.34808], [-0.287795, -0.389056, -1.34717],
                 [-0.479103, -0.445785, -1.34625], [-0.670411, -0.502514, -1.34533], [-0.861719, -0.559244, -1.34442],
                 [-0.952347, -0.490956, -1.17137], [-1.04298, -0.422668, -0.998319], [-1.1336, -0.35438, -0.825271],
                 [-1.22423, -0.286092, -0.652222], [-1.31486, -0.217804, -0.479173], [-1.43897, -0.208613, -0.328078],
                 [-1.56309, -0.199422, -0.176983], [-1.6872, -0.190231, -0.0258879], [-1.81131, -0.181039, 0.125207],
                 [-1.93542, -0.171848, 0.276302], [-1.99722, -0.135308, 0.53271], [-2.05903, -0.0987675, 0.789118],
                 [-2.12083, -0.0622273, 1.04553], [-2.18263, -0.0256871, 1.30193], [-2.24443, 0.0108531, 1.55834],
                 [-2.24716, 0.0912347, 1.79749], [-2.24988, 0.171616, 2.03663], [-2.2526, 0.251998, 2.27578],
                 [-2.25532, 0.33238, 2.51492], [-2.25805, 0.412761, 2.75407], [-2.29321, 0.302267, 2.52022],
                 [-2.32838, 0.191773, 2.28638], [-2.36355, 0.081279, 2.05253], [-2.34974, 0.198708, 1.88901],
                 [-2.33594, 0.316136, 1.72548], [-2.32214, 0.433565, 1.56195], [-2.30834, 0.550994, 1.39843],
                 [-2.29454, 0.668423, 1.2349], [-2.2928, 0.779306, 1.0567], [-2.29106, 0.890189, 0.87849],
                 [-2.28933, 1.00107, 0.700284], [-2.28759, 1.11196, 0.522078], [-2.28586, 1.22284, 0.343871],
                 [-2.21439, 1.41713, 0.257904], [-2.14293, 1.61142, 0.171937], [-2.07146, 1.80571, 0.0859691],
                 [-2.0, 2.0, 1.68948e-06]]
        for tj in [tj_1, tj_2, tj_3, tj_4, tj_5, tj_6, tj_7, tj_8, tj_9, tj_10, tj_11, tj_12,
                   tj_13, tj_14, tj_15, tj_16, tj_17, tj_18, tj_19, tj_20, tj_21, tj_22, tj_23,
                   tj_24, tj_25]:
            poses = []
            for i, point in enumerate(tj_22):
                base_pose = PoseStamped()
                base_pose.header.frame_id = 'map'
                base_pose.pose.position.x = point[0]
                base_pose.pose.position.y = point[1]
                base_pose.pose.position.z = point[2] if len(point) > 3 else 0
                if len(point) > 3:
                    base_pose.pose.orientation = Quaternion(point[3], point[4], point[5], point[6])
                else:
                    arr = quaternion_from_euler(0, 0, point[2])
                    base_pose.pose.orientation = Quaternion(arr[0], arr[1], arr[2], arr[3])
                if i == 0:
                    # important assumption for constraint:
                    # we do not to reach the first pose, since it is the start pose
                    continue
                else:
                    poses.append(base_pose)

            tip_link = u'base_footprint'

            base_pose = PoseStamped()
            base_pose.header.frame_id = 'map'
            base_pose.pose.position.x = 0.0
            base_pose.pose.position.y = 2.0
            base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
            kitchen_setup.teleport_base(base_pose)

            base_pose = PoseStamped()
            base_pose.header.frame_id = 'map'
            base_pose.pose.position.x = -2.0
            base_pose.pose.position.y = 2.0
            base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
            goal_c = base_pose

            # kitchen_setup.set_json_goal(u'SetPredictionHorizon', prediction_horizon=1)
            kitchen_setup.allow_all_collisions()
            kitchen_setup.set_json_goal(u'CartesianPathCarrot',
                                        root_link=kitchen_setup.default_root,
                                        tip_link=tip_link,
                                        goals=poses,
                                        goal=goal_c,
                                        predict_f=10.0)
            try:
                kitchen_setup.plan()
            except Exception:
                pass
        # kitchen_setup.send_goal()
        # kitchen_setup.check_cart_goal(tip_link, goal_c)
        # zero_pose.check_cart_goal(zero_pose.l_tip, l_goal)


    @pytest.mark.repeat(5)
    def test_navi_1(self, kitchen_setup):
        # kernprof -lv py.test -s test/test_integration_pr2.py::TestCartesianPath::test_pathAroundKitchenIsland_with_global_planner
        """
        :type kitchen_setup: PR2
        """
        tip_link = u'base_footprint'

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = 0
        base_pose.pose.position.y = 2.0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        kitchen_setup.teleport_base(base_pose)

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = -2
        base_pose.pose.position.y = 0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        goal_c = base_pose

        # kitchen_setup.set_json_goal(u'SetPredictionHorizon', prediction_horizon=1)
        kitchen_setup.set_json_goal(u'CartesianPathCarrot',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=tip_link,
                                    goal=goal_c,
                                    predict_f=10.0)
        kitchen_setup.plan_and_execute()

        # kitchen_setup.send_goal()
        # kitchen_setup.check_cart_goal(tip_link, goal_c)
        # zero_pose.check_cart_goal(zero_pose.l_tip, l_goal)

    @pytest.mark.repeat(5)
    def test_navi_refills_lab(self, refills_lab):
        # kernprof -lv py.test -s test/test_integration_pr2.py::TestCartesianPath::test_pathAroundKitchenIsland_with_global_planner
        """
        :type kitchen_setup: PR2
        """
        tip_link = u'base_footprint'

        base_pose = PoseStamped()
        base_pose.header.frame_id = 'map'
        base_pose.pose.position.x = 2.0
        base_pose.pose.position.y = 3.0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        refills_lab.teleport_base(base_pose)

        base_pose = PoseStamped()
        base_pose.header.frame_id = 'map'
        base_pose.pose.position.x = 0
        base_pose.pose.position.y = -3.1
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 2, [0, 0, 1]))
        goal_c = base_pose

        # kitchen_setup.set_json_goal(u'SetPredictionHorizon', prediction_horizon=1)
        refills_lab.allow_all_collisions()
        refills_lab.set_json_goal(u'CartesianPathCarrot',
                                  root_link=refills_lab.default_root,
                                  tip_link=tip_link,
                                  goal=goal_c,
                                  predict_f=10.0)

        refills_lab.plan_and_execute()
        # kitchen_setup.send_goal()
        # kitchen_setup.check_cart_goal(tip_link, goal_c)
        # zero_pose.check_cart_goal(zero_pose.l_tip, l_goal)

    @pytest.mark.repeat(5)
    def test_navi_2(self, kitchen_setup):
        # kernprof -lv py.test -s test/test_integration_pr2.py::TestCartGoals::test_pathAroundKitchenIsland_with_global_planner
        """
        :type kitchen_setup: PR2
        """
        tip_link = u'base_footprint'

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = 0
        base_pose.pose.position.y = 2.0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        kitchen_setup.teleport_base(base_pose)

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = 2.1
        base_pose.pose.position.y = 0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        goal_c = base_pose

        kitchen_setup.set_json_goal(u'CartesianPathCarrot',
                                    tip_link=tip_link,
                                    root_link=kitchen_setup.default_root,
                                    goal=goal_c,
                                    predict_f=10.0)

        kitchen_setup.plan_and_execute()
        # kitchen_setup.send_goal()
        # kitchen_setup.check_cart_goal(tip_link, goal_c)
        # zero_pose.check_cart_goal(zero_pose.l_tip, l_goal)

    def test_pathAroundKitchenIslandChangedOrientation_with_global_planner2(self, kitchen_setup):
        # kernprof -lv py.test -s test/test_integration_pr2.py::TestCartGoals::test_pathAroundKitchenIsland_with_global_planner
        """
        :type kitchen_setup: PR2
        """
        table_navigation_link = 'iai_kitchen/dining_area_footprint'
        # spawn milk
        table_navigation_goal = PoseStamped()
        table_navigation_goal.header.frame_id = table_navigation_link
        table_navigation_goal.pose.position = Point(-0.24, -0.80, 0.0)
        table_navigation_goal.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, np.pi / 2.))

        tip_link = u'base_footprint'

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = 0
        base_pose.pose.position.y = 0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        kitchen_setup.teleport_base(base_pose)

        kitchen_setup.set_json_goal(u'CartesianPose',
                                    tip_link=tip_link,
                                    root_link=kitchen_setup.default_root,
                                    goal=table_navigation_goal
                                    )

        kitchen_setup.plan_and_execute()
        # kitchen_setup.send_goal()
        # kitchen_setup.check_cart_goal(tip_link, goal_c)
        # zero_pose.check_cart_goal(zero_pose.l_tip, l_goal)

    def test_pathAroundKitchenIslandChangedOrientation_with_global_planner_align_planes1(self, kitchen_setup):
        """
        :type zero_pose: PR2
        """
        tip_link = u'base_footprint'

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = 0
        base_pose.pose.position.y = 2.0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        kitchen_setup.teleport_base(base_pose)

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = 2.1
        base_pose.pose.position.y = 0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        goal_c = base_pose

        kitchen_setup.set_json_goal(u'CartesianPose',
                                    tip_link=tip_link,
                                    root_link=kitchen_setup.default_root,
                                    goal=goal_c
                                    )

        r_gripper_vec = Vector3Stamped()
        r_gripper_vec.header.frame_id = kitchen_setup.r_tip
        r_gripper_vec.vector.z = 1
        l_gripper_vec = Vector3Stamped()
        l_gripper_vec.header.frame_id = kitchen_setup.l_tip
        l_gripper_vec.vector.z = 1

        gripper_goal_vec = Vector3Stamped()
        gripper_goal_vec.header.frame_id = u'map'
        gripper_goal_vec.vector.z = 1
        kitchen_setup.set_align_planes_goal(kitchen_setup.r_tip, r_gripper_vec, root_normal=gripper_goal_vec)
        kitchen_setup.set_align_planes_goal(kitchen_setup.l_tip, l_gripper_vec, root_normal=gripper_goal_vec)
        kitchen_setup.plan_and_execute()

    def test_pathAroundKitchenIslandChangedOrientation_with_global_planner_align_planes2(self, kitchen_setup):
        """
        :type zero_pose: PR2
        """
        tip_link = u'base_footprint'

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = 0
        base_pose.pose.position.y = 2.0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        kitchen_setup.teleport_base(base_pose)

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = 2.1
        base_pose.pose.position.y = 0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        goal_c = base_pose

        kitchen_setup.set_json_goal(u'CartesianPose',
                                    tip_link=tip_link,
                                    root_link=kitchen_setup.default_root,
                                    goal=goal_c
                                    )

        r_gripper_vec = Vector3Stamped()
        r_gripper_vec.header.frame_id = kitchen_setup.r_tip
        r_gripper_vec.vector.z = -1
        l_gripper_vec = Vector3Stamped()
        l_gripper_vec.header.frame_id = kitchen_setup.l_tip
        l_gripper_vec.vector.z = 1

        gripper_goal_vec = Vector3Stamped()
        gripper_goal_vec.header.frame_id = u'map'
        gripper_goal_vec.vector.z = 1
        kitchen_setup.set_align_planes_goal(kitchen_setup.r_tip, r_gripper_vec, root_normal=gripper_goal_vec)
        kitchen_setup.set_align_planes_goal(kitchen_setup.l_tip, l_gripper_vec, root_normal=gripper_goal_vec)
        kitchen_setup.plan_and_execute()

    def test_pathAroundKitchenIslandChangedOrientation_with_global_planner_shaky_grippers(self,
                                                                                          kitchen_setup):
        """
        :type zero_pose: PR2
        """
        tip_link = u'base_footprint'

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = 0
        base_pose.pose.position.y = 2.0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        kitchen_setup.teleport_base(base_pose)

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = 2.1
        base_pose.pose.position.y = 0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        goal_c = base_pose

        kitchen_setup.set_json_goal(u'CartesianPose',
                                    tip_link=tip_link,
                                    root_link=kitchen_setup.default_root,
                                    goal=goal_c)

        f = 0.5
        amp = 1.0
        axis = 'z'

        kitchen_setup.set_json_goal(u'ShakyCartesianPosition',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=kitchen_setup.r_tip,
                                    frequency=f,
                                    noise_amplitude=amp,
                                    shaking_axis=axis)
        kitchen_setup.set_json_goal(u'ShakyCartesianPosition',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=kitchen_setup.l_tip,
                                    frequency=f,
                                    noise_amplitude=amp,
                                    shaking_axis=axis)

        kitchen_setup.plan_and_execute()

    @pytest.mark.repeat(15)
    def test_navi_3_stuck(self, kitchen_setup):
        # kernprof -lv py.test -s test/test_integration_pr2.py::TestCartGoals::test_pathAroundKitchenIsland_with_global_planner
        """
        :type kitchen_setup: PR2
        """
        tj_a = [[-0.00000e+00, 2.00000e+00, 1.92195e-06],
                [-1.14698e-01, 1.87903e+00, 6.65996e-02],
                [-5.02469e-02, 1.71219e+00, 2.43104e-02],
                [-4.67477e-02, 1.54885e+00, 9.75557e-02],
                [-1.14831e-01, 1.42465e+00, 2.14280e-01],
                [-4.36070e-02, 1.32318e+00, 6.22169e-02],
                [2.82440e-02, 1.15371e+00, 9.40615e-02],
                [-3.22373e-02, 1.00175e+00, 2.11726e-02],
                [-8.99319e-02, 8.17358e-01, 3.47652e-02],
                [-1.10718e-02, 6.44228e-01, 1.52542e-02],
                [1.03955e-01, 5.21862e-01, 7.93707e-02],
                [1.02730e-01, 3.83932e-01, -4.47585e-02],
                [2.32972e-01, 2.38108e-01, -5.37212e-02],
                [3.39305e-01, 2.52551e-01, -2.39103e-01],
                [2.10944e-01, 1.39223e-01, -2.96644e-01],
                [2.20303e-01, 4.25983e-02, -5.02490e-01],
                [2.34666e-01, 4.40739e-02, -8.73614e-01],
                [3.13372e-01, 1.36479e-02, -1.10485e+00],
                [1.93062e-01, -1.00863e-02, -1.25959e+00],
                [2.46916e-01, -1.77602e-01, -1.30767e+00],
                [2.24101e-01, -3.50326e-01, -1.35923e+00],
                [1.06686e-01, -3.99275e-01, -1.50481e+00],
                [1.16489e-01, -5.56927e-01, -1.58889e+00],
                [3.73072e-02, -6.76895e-01, -1.70141e+00],
                [4.95714e-02, -7.13035e-01, -2.02508e+00],
                [6.61559e-02, -7.72211e-01, -2.30217e+00],
                [-8.14138e-02, -8.04503e-01, -2.40005e+00],
                [-2.34866e-01, -8.81529e-01, -2.34344e+00],
                [-4.24074e-01, -9.02666e-01, -2.32421e+00],
                [-6.09024e-01, -8.88678e-01, -2.29517e+00],
                [-7.66748e-01, -7.73141e-01, -2.30414e+00],
                [-9.61292e-01, -7.92777e-01, -2.31308e+00],
                [-1.02139e+00, -8.36017e-01, -2.56501e+00],
                [-1.02169e+00, -7.83873e-01, -2.34957e+00],
                [-1.02208e+00, -7.18637e-01, -2.08004e+00],
                [-1.02246e+00, -6.53401e-01, -1.81052e+00],
                [-1.02284e+00, -5.88166e-01, -1.54099e+00],
                [-1.02322e+00, -5.22930e-01, -1.27146e+00],
                [-1.02360e+00, -4.57694e-01, -1.00194e+00],
                [-1.09022e+00, -3.88439e-01, -7.94128e-01],
                [-1.17315e+00, -3.24424e-01, -6.03651e-01],
                [-1.25608e+00, -2.60409e-01, -4.13173e-01],
                [-1.42628e+00, -2.95172e-01, -3.60604e-01],
                [-1.59648e+00, -3.29936e-01, -3.08034e-01],
                [-1.76668e+00, -3.64699e-01, -2.55465e-01],
                [-1.72377e+00, -2.63582e-01, -7.51521e-02],
                [-1.83889e+00, -1.62515e-01, 1.84722e-02],
                [-1.95034e+00, -3.27492e-02, 7.63583e-02],
                [-1.93177e+00, 1.04183e-01, 1.99985e-01],
                [-1.85549e+00, 2.05511e-01, 3.46335e-01],
                [-1.95529e+00, 3.68291e-01, 3.64465e-01],
                [-1.89617e+00, 4.94312e-01, 4.86068e-01],
                [-1.94512e+00, 6.49940e-01, 4.12359e-01],
                [-2.01484e+00, 7.22352e-01, 2.13394e-01],
                [-2.02206e+00, 8.59219e-01, 8.75088e-02],
                [-1.93045e+00, 1.03341e+00, 9.38936e-02],
                [-2.07202e+00, 1.13460e+00, 1.45849e-01],
                [-2.06596e+00, 1.25141e+00, 3.11908e-01],
                [-2.02112e+00, 1.39071e+00, 2.04576e-01],
                [-1.99934e+00, 1.53781e+00, 1.01990e-01],
                [-2.02851e+00, 1.66977e+00, -2.77325e-02],
                [-2.07125e+00, 1.83245e+00, 3.58616e-02],
                [-2.00000e+00, 2.00000e+00, 1.92195e-06]]
        poses = []
        for i, point in enumerate(tj_a):
            base_pose = PoseStamped()
            base_pose.header.frame_id = 'map'
            base_pose.pose.position.x = point[0]
            base_pose.pose.position.y = point[1]
            base_pose.pose.position.z = point[2] if len(point) > 3 else 0
            if len(point) > 3:
                base_pose.pose.orientation = Quaternion(point[3], point[4], point[5], point[6])
            else:
                arr = quaternion_from_euler(0, 0, point[2])
                base_pose.pose.orientation = Quaternion(arr[0], arr[1], arr[2], arr[3])
            if i == 0:
                # important assumption for constraint:
                # we do not to reach the first pose, since it is the start pose
                continue
            else:
                poses.append(base_pose)
        tip_link = u'base_footprint'

        box_pose = PoseStamped()
        box_pose.header.frame_id = tip_link
        box_pose.pose.position.x = -0.5
        box_pose.pose.position.y = 0
        box_pose.pose.position.z = 0
        box_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        kitchen_setup.add_box('box', [0.5, 0.5, 1], pose=box_pose)

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = 0
        base_pose.pose.position.y = 2.0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        kitchen_setup.teleport_base(base_pose)

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = -2
        base_pose.pose.position.y = 0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        goal_c = base_pose

        try:
            kitchen_setup.set_json_goal(u'CartesianPathCarrot',
                                        tip_link=tip_link,
                                        root_link=kitchen_setup.default_root,
                                        goal=goal_c,
                                        goals=poses,
                                        predict_f=10.0)
            kitchen_setup.plan_and_execute()
        except Exception:
            pass
        # kitchen_setup.send_goal()
        # kitchen_setup.check_cart_goal(tip_link, goal_c)
        # zero_pose.check_cart_goal(zero_pose.l_tip, l_goal)

    @pytest.mark.repeat(15)
    def test_navi_3(self, kitchen_setup):
        # kernprof -lv py.test -s test/test_integration_pr2.py::TestCartGoals::test_pathAroundKitchenIsland_with_global_planner
        """
        :type kitchen_setup: PR2
        """
        tip_link = u'base_footprint'

        box_pose = PoseStamped()
        box_pose.header.frame_id = tip_link
        box_pose.pose.position.x = -0.5
        box_pose.pose.position.y = 0
        box_pose.pose.position.z = 0
        box_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        kitchen_setup.add_box('box', [0.5, 0.5, 1], pose=box_pose)

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = 0
        base_pose.pose.position.y = 2.0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        kitchen_setup.teleport_base(base_pose)

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = -2
        base_pose.pose.position.y = 0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        goal_c = base_pose

        try:
            kitchen_setup.set_json_goal(u'CartesianPathCarrot',
                                        tip_link=tip_link,
                                        root_link=kitchen_setup.default_root,
                                        goal=goal_c,
                                        predict_f=10.0)
            kitchen_setup.plan_and_execute()
        except Exception:
            pass
        # kitchen_setup.send_goal()
        # kitchen_setup.check_cart_goal(tip_link, goal_c)
        # zero_pose.check_cart_goal(zero_pose.l_tip, l_goal)

    @pytest.mark.repeat(5)
    def test_navi_4(self, kitchen_setup):
        # kernprof -lv py.test -s test/test_integration_pr2.py::TestCartGoals::test_pathAroundKitchenIsland_with_global_planner
        """
        :type kitchen_setup: PR2
        """
        tip_link = u'base_footprint'
        kitchen_setup.set_kitchen_js({'kitchen_island_left_upper_drawer_main_joint': 0.48})
        kitchen_setup.set_kitchen_js({u'oven_area_area_right_drawer_main_joint': 0.28})

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = 0.3
        base_pose.pose.position.y = 2.1
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 2, [0, 0, 1]))
        kitchen_setup.teleport_base(base_pose)

        base_pose = PoseStamped()
        base_pose.header.frame_id = 'map'
        base_pose.pose.position.x = -2.5
        base_pose.pose.position.y = 1
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0, 0, 1]))
        goal_c = base_pose

        try:
            kitchen_setup.set_json_goal(u'CartesianPathCarrot',
                                        root_link=kitchen_setup.default_root,
                                        tip_link=tip_link,
                                        goal=goal_c,
                                        predict_f=10.0)
            kitchen_setup.plan_and_execute()
        except Exception:
            pass
        # kitchen_setup.send_goal()
        # kitchen_setup.check_cart_goal(tip_link, goal_c)
        # zero_pose.check_cart_goal(zero_pose.l_tip, l_goal)

    @pytest.mark.repeat(5)
    def test_navi_5(self, kitchen_setup):
        # kernprof -lv py.test -s test/test_integration_pr2.py::TestCartGoals::test_pathAroundKitchenIsland_with_global_planner
        """
        :type kitchen_setup: PR2
        """
        tip_link = u'base_footprint'
        kitchen_setup.set_kitchen_js({'kitchen_island_left_upper_drawer_main_joint': 0.48})
        kitchen_setup.set_kitchen_js({u'oven_area_area_right_drawer_main_joint': 0.28})

        box_pose = PoseStamped()
        box_pose.header.frame_id = tip_link
        box_pose.pose.position.x = -2.5
        box_pose.pose.position.y = 0
        box_pose.pose.position.z = 0
        box_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        kitchen_setup.add_box('box1', [0.5, 0.5, 1], pose=box_pose)
        box_pose.pose.position.x = -0.5
        kitchen_setup.add_box('box2', [0.5, 0.5, 1], pose=box_pose)

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = 0
        base_pose.pose.position.y = 2.1
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 8, [0, 0, 1]))
        kitchen_setup.teleport_base(base_pose)

        base_pose = PoseStamped()
        base_pose.header.frame_id = 'map'
        base_pose.pose.position.x = -2.5
        base_pose.pose.position.y = 1
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0, 0, 1]))
        goal_c = base_pose
        # kitchen_setup.allow_all_collisions()
        try:
            kitchen_setup.allow_all_collisions()
            kitchen_setup.set_json_goal('CartesianPose',
                                        tip_link=tip_link,
                                        root_link=kitchen_setup.default_root,
                                        goal=goal_c)
                                        #predict_f=10.0)
            kitchen_setup.plan_and_execute()
        except Exception:
            pass
        # kitchen_setup.send_goal()
        # kitchen_setup.check_cart_goal(tip_link, goal_c)
        # zero_pose.check_cart_goal(zero_pose.l_tip, l_goal)

    def test_ease_fridge_with_cart_goals_lifting_and_global_planner(self, kitchen_setup):
        rospy.sleep(10.0)  # 0.5

        tip_link = kitchen_setup.r_tip
        milk_name = u'milk'

        # take milk out of fridge
        kitchen_setup.set_kitchen_js({u'iai_fridge_door_joint': 1.56})

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 0.4
        base_goal.pose.position.y = -1.05
        base_goal.pose.orientation.x = 0
        base_goal.pose.orientation.y = 0
        base_goal.pose.orientation.z = 0
        base_goal.pose.orientation.w = 1
        kitchen_setup.teleport_base(base_goal)

        # spawn milk
        milk_pose = PoseStamped()
        milk_pose.header.frame_id = u'iai_kitchen/iai_fridge_door_shelf1_bottom'
        milk_pose.pose.position = Point(0, 0, 0.12)
        milk_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        milk_lift_pre_pose = PoseStamped()
        milk_lift_pre_pose.header.frame_id = u'iai_kitchen/iai_fridge_door_shelf1_bottom'
        milk_lift_pre_pose.pose.position = Point(0, 0, 0.22)
        milk_lift_pre_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        # milk_grasp_pre_pose = PoseStamped()
        # milk_grasp_pre_pose.header.frame_id = u'iai_kitchen/iai_fridge_door_shelf1_bottom'
        # milk_grasp_pre_pose.pose.position = Point(-0.2, 0, 0.12)
        # milk_grasp_pre_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup.add_box(milk_name, [0.05, 0.05, 0.2], milk_pose)

        # move arm towards milk
        if tip_link == kitchen_setup.l_tip:
            kitchen_setup.open_l_gripper()
        elif tip_link == kitchen_setup.r_tip:
            kitchen_setup.open_r_gripper()
        else:
            raise Exception('Wrong tip link')

        # pregrasping
        kitchen_setup.allow_collision(body_b=milk_name)
        kitchen_setup.set_json_goal(u'CartesianPose',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=tip_link,
                                    goal=milk_pose)

        kitchen_setup.plan_and_execute()

        # grasping
        kitchen_setup.set_json_goal(u'CartesianPose',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=tip_link,
                                    goal=milk_pose)

        kitchen_setup.plan_and_execute()

        # grasp milk
        kitchen_setup.attach_object(milk_name, tip_link)

        if tip_link == kitchen_setup.l_tip:
            kitchen_setup.close_l_gripper()
        elif tip_link == kitchen_setup.r_tip:
            kitchen_setup.close_r_gripper()
        else:
            raise Exception('Wrong tip link')

        kitchen_setup.set_cart_goal(milk_lift_pre_pose, milk_name, kitchen_setup.default_root)
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.plan_and_execute()

        # dont allow robot to move
        js = kitchen_setup.god_map.get_data(identifier.joint_states)
        odom_joints = ['odom_x_joint', 'odom_y_joint', 'odom_z_joint']
        kitchen_setup.set_joint_goal({j_n: js[j_n].position for j_n in odom_joints})

        # place milk back
        kitchen_setup.set_cart_goal(milk_lift_pre_pose, milk_name, kitchen_setup.default_root)
        kitchen_setup.plan_and_execute()

        kitchen_setup.set_cart_goal(milk_pose, milk_name, kitchen_setup.default_root)
        kitchen_setup.plan_and_execute()

        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        if tip_link == kitchen_setup.l_tip:
            kitchen_setup.open_l_gripper()
        elif tip_link == kitchen_setup.r_tip:
            kitchen_setup.open_r_gripper()
        else:
            raise Exception('Wrong tip link')

        kitchen_setup.detach_object(milk_name)

        # kitchen_setup.set_json_goal(u'CartesianPose',
        #                                             root_link=kitchen_setup.default_root,
        #                                             tip_link=tip_link,
        #                                             goal=milk_grasp_pre_pose)
        # kitchen_setup.send_and_check_goal()

        kitchen_setup.set_joint_goal(gaya_pose)

    @pytest.mark.repeat(5)
    def test_milk_1(self, kitchen_setup):
        rospy.sleep(10.0)  # 0.5

        tip_link = kitchen_setup.l_tip
        milk_name = u'milk'

        # take milk out of fridge
        kitchen_setup.set_kitchen_js({u'iai_fridge_door_joint': 1.56})

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 0.0
        base_goal.pose.position.y = -1.05
        base_goal.pose.orientation.x = 0
        base_goal.pose.orientation.y = 0
        base_goal.pose.orientation.z = 0
        base_goal.pose.orientation.w = 1
        kitchen_setup.teleport_base(base_goal)

        # spawn milk
        milk_pose = PoseStamped()
        milk_pose.header.frame_id = u'iai_kitchen/iai_fridge_door_shelf1_bottom'
        milk_pose.pose.position = Point(0, 0, 0.12)
        milk_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        milk_pre_pose = PoseStamped()
        milk_pre_pose.header.frame_id = u'iai_kitchen/iai_fridge_door_shelf1_bottom'
        milk_pre_pose.pose.position = Point(-0.3, -0.1, 0.22)
        milk_pre_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup.add_box(milk_name, [0.05, 0.05, 0.2], milk_pose)

        # move arm towards milk
        if tip_link == kitchen_setup.l_tip:
            kitchen_setup.open_l_gripper()
        elif tip_link == kitchen_setup.r_tip:
            kitchen_setup.open_r_gripper()
        else:
            raise Exception('Wrong tip link')

        # Grasp
        kitchen_setup.set_json_goal(u'CartesianPose',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=tip_link,
                                    goal=milk_pose)
        kitchen_setup.plan_and_execute()

        # attach milk
        kitchen_setup.attach_object(milk_name, tip_link)

        # pick up
        kitchen_setup.set_json_goal(u'CartesianPathCarrot',
                                    tip_link=tip_link,
                                    predict_f=2.0,
                                    narrow=True,
                                    narrow_padding=1.0,
                                    root_link=kitchen_setup.default_root,
                                    goal=milk_pre_pose)
        kitchen_setup.plan_and_execute()

        if tip_link == kitchen_setup.l_tip:
            kitchen_setup.close_l_gripper()
        elif tip_link == kitchen_setup.r_tip:
            kitchen_setup.close_r_gripper()
        else:
            raise Exception('Wrong tip link')

        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.plan_and_execute()

    @pytest.mark.repeat(5)
    def test_milk_2(self, kitchen_setup):
        rospy.sleep(10.0)  # 0.5

        tip_link = kitchen_setup.l_tip
        milk_name = u'milk'

        # take milk out of fridge
        kitchen_setup.set_kitchen_js({u'iai_fridge_door_joint': 1.56})

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 0.0
        base_goal.pose.position.y = -1.05
        base_goal.pose.orientation.x = 0
        base_goal.pose.orientation.y = 0
        base_goal.pose.orientation.z = 0
        base_goal.pose.orientation.w = 1
        kitchen_setup.teleport_base(base_goal)

        # spawn milk
        milk_pose = PoseStamped()
        milk_pose.header.frame_id = u'iai_kitchen/iai_fridge_door_shelf1_bottom'
        milk_pose.pose.position = Point(0, -0.1, 0.125)
        milk_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        milk_lift_pre_pose = PoseStamped()
        milk_lift_pre_pose.header.frame_id = u'iai_kitchen/iai_fridge_door_shelf1_bottom'
        milk_lift_pre_pose.pose.position = Point(0, 0, 0.22)
        milk_lift_pre_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        # milk_grasp_pre_pose = PoseStamped()
        # milk_grasp_pre_pose.header.frame_id = u'iai_kitchen/iai_fridge_door_shelf1_bottom'
        # milk_grasp_pre_pose.pose.position = Point(-0.2, 0, 0.12)
        # milk_grasp_pre_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup.add_box(milk_name, [0.05, 0.05, 0.2], milk_pose)

        # move arm towards milk
        if tip_link == kitchen_setup.l_tip:
            kitchen_setup.open_l_gripper()
        elif tip_link == kitchen_setup.r_tip:
            kitchen_setup.open_r_gripper()
        else:
            raise Exception('Wrong tip link')

        # Pregrasp
        kitchen_setup.allow_collision(body_b=milk_name)
        kitchen_setup.set_json_goal(u'CartesianPose',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=tip_link,
                                    goal=milk_pose)

        kitchen_setup.plan_and_execute()

        # Grasp
        kitchen_setup.set_json_goal(u'CartesianPose',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=tip_link,
                                    goal=milk_pose)

        kitchen_setup.plan_and_execute()

        # attach milk
        kitchen_setup.attach_object(milk_name, tip_link)

        if tip_link == kitchen_setup.l_tip:
            kitchen_setup.close_l_gripper()
        elif tip_link == kitchen_setup.r_tip:
            kitchen_setup.close_r_gripper()
        else:
            raise Exception('Wrong tip link')

        kitchen_setup.set_cart_goal(milk_lift_pre_pose, milk_name, kitchen_setup.default_root)
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.plan_and_execute()

        # dont allow robot to move
        js = kitchen_setup.god_map.get_data(identifier.joint_states)
        odom_joints = ['odom_x_joint', 'odom_y_joint', 'odom_z_joint']
        kitchen_setup.set_joint_goal({j_n: js[j_n].position for j_n in odom_joints})

        # place milk back
        # pregrasping
        kitchen_setup.allow_collision(body_b=milk_name)
        kitchen_setup.set_json_goal(u'CartesianPathCarrot',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=tip_link,
                                    predict_f=2.0,
                                    narrow_padding=1.0,
                                    narrow=True,
                                    goal=milk_pose)
        kitchen_setup.plan_and_execute()

    def test_ease_fridge_placing_presampling(self, kitchen_setup):
        rospy.sleep(10.0)  # 0.5

        tip_link = kitchen_setup.l_tip
        milk_name = u'milk'

        # take milk out of fridge
        kitchen_setup.set_kitchen_js({u'iai_fridge_door_joint': 1.56})

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 0.0
        base_goal.pose.position.y = -1.05
        base_goal.pose.orientation.x = 0
        base_goal.pose.orientation.y = 0
        base_goal.pose.orientation.z = 0
        base_goal.pose.orientation.w = 1
        kitchen_setup.teleport_base(base_goal)

        # spawn milk
        milk_pose = PoseStamped()
        milk_pose.header.frame_id = u'iai_kitchen/iai_fridge_door_shelf1_bottom'
        milk_pose.pose.position = Point(0, 0, 0.12)
        milk_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        milk_lift_pre_pose = PoseStamped()
        milk_lift_pre_pose.header.frame_id = u'iai_kitchen/iai_fridge_door_shelf1_bottom'
        milk_lift_pre_pose.pose.position = Point(0, 0, 0.22)
        milk_lift_pre_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        # milk_grasp_pre_pose = PoseStamped()
        # milk_grasp_pre_pose.header.frame_id = u'iai_kitchen/iai_fridge_door_shelf1_bottom'
        # milk_grasp_pre_pose.pose.position = Point(-0.2, 0, 0.12)
        # milk_grasp_pre_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup.add_box(milk_name, [0.05, 0.05, 0.2], milk_pose)

        # move arm towards milk
        if tip_link == kitchen_setup.l_tip:
            kitchen_setup.open_l_gripper()
        elif tip_link == kitchen_setup.r_tip:
            kitchen_setup.open_r_gripper()
        else:
            raise Exception('Wrong tip link')

        # Pregrasp
        kitchen_setup.allow_collision(body_b=milk_name)
        kitchen_setup.set_json_goal(u'CartesianPose',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=tip_link,
                                    goal=milk_pose)

        kitchen_setup.plan_and_execute()

        # Grasp
        kitchen_setup.set_json_goal(u'CartesianPose',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=tip_link,
                                    goal=milk_pose)

        kitchen_setup.plan_and_execute()

        # attach milk
        kitchen_setup.attach_object(milk_name, tip_link)

        if tip_link == kitchen_setup.l_tip:
            kitchen_setup.close_l_gripper()
        elif tip_link == kitchen_setup.r_tip:
            kitchen_setup.close_r_gripper()
        else:
            raise Exception('Wrong tip link')

        kitchen_setup.set_cart_goal(milk_lift_pre_pose, milk_name, kitchen_setup.default_root)
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.plan_and_execute()

        # dont allow robot to move
        js = kitchen_setup.god_map.get_data(identifier.joint_states)
        odom_joints = ['odom_x_joint', 'odom_y_joint', 'odom_z_joint']
        kitchen_setup.set_joint_goal({j_n: js[j_n].position for j_n in odom_joints})

        # place milk back
        # pregrasping
        kitchen_setup.allow_collision(body_b=milk_name)
        kitchen_setup.set_json_goal(u'CartesianPreGrasp',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=tip_link,
                                    grasping_object=milk_name,
                                    grasping_orientation=milk_pose.pose.orientation,
                                    grasping_goal=milk_pose)
        kitchen_setup.plan_and_execute()

        kitchen_setup.set_cart_goal(milk_pose, milk_name, kitchen_setup.default_root)
        kitchen_setup.plan_and_execute()

        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        if tip_link == kitchen_setup.l_tip:
            kitchen_setup.open_l_gripper()
        elif tip_link == kitchen_setup.r_tip:
            kitchen_setup.open_r_gripper()
        else:
            raise Exception('Wrong tip link')

        kitchen_setup.detach_object(milk_name)

        # kitchen_setup.set_json_goal(u'CartesianPose',
        #                                             root_link=kitchen_setup.default_root,
        #                                             tip_link=tip_link,
        #                                             goal=milk_grasp_pre_pose)
        # kitchen_setup.send_and_check_goal()

        kitchen_setup.set_joint_goal(gaya_pose)

    # ease_fridge_pregrasp_1
    @pytest.mark.repeat(5)
    def test_milk_3(self, kitchen_setup):
        rospy.sleep(10.0)

        tip_link = kitchen_setup.r_tip
        milk_name = u'milk'

        # take milk out of fridge
        kitchen_setup.set_kitchen_js({u'iai_fridge_door_joint': 1.56})

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 0.4
        base_goal.pose.position.y = -1.05
        base_goal.pose.orientation.x = 0
        base_goal.pose.orientation.y = 0
        base_goal.pose.orientation.z = 0
        base_goal.pose.orientation.w = 1
        kitchen_setup.teleport_base(base_goal)

        # spawn milk
        milk_pose = PoseStamped()
        milk_pose.header.frame_id = u'iai_kitchen/iai_fridge_door_shelf1_bottom'
        milk_pose.pose.position = Point(0, 0, 0.13)
        milk_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        milk_pre_pose = PoseStamped()
        milk_pre_pose.header.frame_id = u'iai_kitchen/iai_fridge_door_shelf1_bottom'
        milk_pre_pose.pose.position = Point(-0.2, -0.1, 0.22)
        milk_pre_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup.add_box(milk_name, [0.05, 0.05, 0.2], milk_pose)

        # move arm towards milk
        if tip_link == kitchen_setup.l_tip:
            kitchen_setup.open_l_gripper()
        elif tip_link == kitchen_setup.r_tip:
            kitchen_setup.open_r_gripper()
        else:
            raise Exception('Wrong tip link')

        kitchen_setup.set_json_goal(u'CartesianPathCarrot',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=tip_link,
                                    predict_f=5.0,
                                    goal=milk_pre_pose)
        kitchen_setup.plan_and_execute()

    def test_ease_fridge_with_cart_goals_and_global_planner(self, kitchen_setup):
        rospy.sleep(10.0)  # 0.5

        tip_link = kitchen_setup.r_tip
        milk_name = u'milk'

        # take milk out of fridge
        kitchen_setup.set_kitchen_js({u'iai_fridge_door_joint': 1.56})

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 0.4
        base_goal.pose.position.y = -1.05
        base_goal.pose.orientation.x = 0
        base_goal.pose.orientation.y = 0
        base_goal.pose.orientation.z = 0
        base_goal.pose.orientation.w = 1
        kitchen_setup.teleport_base(base_goal)

        # spawn milk
        milk_pose = PoseStamped()
        milk_pose.header.frame_id = u'iai_kitchen/iai_fridge_door_shelf1_bottom'
        milk_pose.pose.position = Point(0, 0, 0.13)
        milk_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup.add_box(milk_name, [0.05, 0.05, 0.2], milk_pose)

        # move arm towards milk
        if tip_link == kitchen_setup.l_tip:
            kitchen_setup.open_l_gripper()
        elif tip_link == kitchen_setup.r_tip:
            kitchen_setup.open_r_gripper()
        else:
            raise Exception('Wrong tip link')

        kitchen_setup.set_json_goal('CartesianPreGrasp',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=tip_link,
                                    grasping_goal=milk_pose)
        rospy.logerr('Pregrasping')
        kitchen_setup.plan_and_execute()

        kitchen_setup.set_json_goal(u'CartesianPose',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=tip_link,
                                    goal=milk_pose)
        kitchen_setup.plan_and_execute()

        # grasp milk
        kitchen_setup.attach_object(milk_name, tip_link)

        if tip_link == kitchen_setup.l_tip:
            kitchen_setup.close_l_gripper()
        elif tip_link == kitchen_setup.r_tip:
            kitchen_setup.close_r_gripper()
        else:
            raise Exception('Wrong tip link')

        kitchen_setup.set_json_goal('CartesianPreGrasp',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=tip_link,
                                    grasping_goal=milk_pose)
        rospy.logerr('Pregrasping')
        kitchen_setup.plan_and_execute()

        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.plan_and_execute()

        # place milk back
        kitchen_setup.set_json_goal('CartesianPreGrasp',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=tip_link,
                                    grasping_goal=milk_pose)
        rospy.logerr('Pregrasping')
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup.set_cart_goal(milk_pose, milk_name, kitchen_setup.default_root)
        kitchen_setup.plan_and_execute()

        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        if tip_link == kitchen_setup.l_tip:
            kitchen_setup.open_l_gripper()
        elif tip_link == kitchen_setup.r_tip:
            kitchen_setup.open_r_gripper()
        else:
            raise Exception('Wrong tip link')

        kitchen_setup.detach_object(milk_name)

        # kitchen_setup.set_json_goal(u'CartesianPose',
        #                                             root_link=kitchen_setup.default_root,
        #                                             tip_link=tip_link,
        #                                             goal=milk_grasp_pre_pose)
        # kitchen_setup.send_and_check_goal()

        kitchen_setup.set_joint_goal(gaya_pose)

    def test_ease_fridge_with_cart_goals_and_global_planner_aligned_planes(self, kitchen_setup):
        rospy.sleep(10.0)  # 0.5

        tip_link = kitchen_setup.r_tip
        milk_name = u'milk'

        # take milk out of fridge
        kitchen_setup.set_kitchen_js({u'iai_fridge_door_joint': 1.56})

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 0.4
        base_goal.pose.position.y = -1.05
        base_goal.pose.orientation.x = 0
        base_goal.pose.orientation.y = 0
        base_goal.pose.orientation.z = 0
        base_goal.pose.orientation.w = 1
        kitchen_setup.teleport_base(base_goal)

        # spawn milk
        milk_pose = PoseStamped()
        milk_pose.header.frame_id = u'iai_kitchen/iai_fridge_door_shelf1_bottom'
        milk_pose.pose.position = Point(0, 0, 0.12)
        milk_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        milk_lift_pre_pose = PoseStamped()
        milk_lift_pre_pose.header.frame_id = u'iai_kitchen/iai_fridge_door_shelf1_bottom'
        milk_lift_pre_pose.pose.position = Point(0, 0, 0.22)
        milk_lift_pre_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        # milk_grasp_pre_pose = PoseStamped()
        # milk_grasp_pre_pose.header.frame_id = u'iai_kitchen/iai_fridge_door_shelf1_bottom'
        # milk_grasp_pre_pose.pose.position = Point(-0.2, 0, 0.12)
        # milk_grasp_pre_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup.add_box(milk_name, [0.05, 0.05, 0.2], milk_pose)

        # move arm towards milk
        if tip_link == kitchen_setup.l_tip:
            kitchen_setup.open_l_gripper()
        elif tip_link == kitchen_setup.r_tip:
            kitchen_setup.open_r_gripper()
        else:
            raise Exception('Wrong tip link')

        kitchen_setup.set_json_goal(u'CartesianPose',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=tip_link,
                                    goal=milk_pose)

        kitchen_setup.plan_and_execute()

        kitchen_setup.set_json_goal(u'CartesianPose',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=tip_link,
                                    goal=milk_pose)

        kitchen_setup.plan_and_execute()

        # grasp milk
        kitchen_setup.attach_object(milk_name, tip_link)

        if tip_link == kitchen_setup.l_tip:
            kitchen_setup.close_l_gripper()
        elif tip_link == kitchen_setup.r_tip:
            kitchen_setup.close_r_gripper()
        else:
            raise Exception('Wrong tip link')

        gripper_vec = Vector3Stamped()
        gripper_vec.header.frame_id = tip_link
        gripper_vec.vector.x = 1

        gripper_goal_vec = Vector3Stamped()
        gripper_goal_vec.header.frame_id = u'milk'
        gripper_goal_vec.vector.z = 1
        kitchen_setup.set_align_planes_goal(tip_link, gripper_vec, root_normal=gripper_goal_vec)

        kitchen_setup.set_cart_goal(milk_lift_pre_pose, milk_name, kitchen_setup.default_root)
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.plan_and_execute()

        # place milk back
        kitchen_setup.set_cart_goal(milk_lift_pre_pose, milk_name, kitchen_setup.default_root)
        kitchen_setup.plan_and_execute()

        kitchen_setup.set_cart_goal(milk_pose, milk_name, kitchen_setup.default_root)
        kitchen_setup.plan_and_execute()

        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        if tip_link == kitchen_setup.l_tip:
            kitchen_setup.open_l_gripper()
        elif tip_link == kitchen_setup.r_tip:
            kitchen_setup.open_r_gripper()
        else:
            raise Exception('Wrong tip link')

        kitchen_setup.detach_object(milk_name)

        # kitchen_setup.set_json_goal(u'CartesianPose',
        #                                             root_link=kitchen_setup.default_root,
        #                                             tip_link=tip_link,
        #                                             goal=milk_grasp_pre_pose)
        kitchen_setup.send_and_check_goal()

        kitchen_setup.send_and_check_joint_goal(gaya_pose)

    def test_faster_ease_cereal_with_planner(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        # FIXME collision avoidance needs soft_threshholds at 0
        cereal_name = u'cereal'
        drawer_frame_id = u'iai_kitchen/oven_area_area_right_drawer_board_2_link'

        # take milk out of fridge
        kitchen_setup.set_kitchen_js({u'oven_area_area_right_drawer_main_joint': 0.48})

        # spawn milk
        cereal_pose = PoseStamped()
        cereal_pose.header.frame_id = drawer_frame_id
        cereal_pose.pose.position = Point(0.123, 0.0, 0.15)
        cereal_pose.pose.orientation = Quaternion(0.0087786, 0.005395, -0.838767, -0.544393)
        kitchen_setup.add_box(cereal_name, [0.1028, 0.0634, 0.20894], cereal_pose)

        cereal_pose_in_map = tf.msg_to_kdl(tf.transform_pose(u'map', cereal_pose))

        drawer_T_box = tf.msg_to_kdl(cereal_pose)

        # grasp milk
        kitchen_setup.open_l_gripper()
        grasp_pose = PoseStamped()
        grasp_pose.header.frame_id = cereal_name
        grasp_pose.pose.position = Point(0.13, 0, 0.05)
        grasp_pose.pose.orientation = Quaternion(0, 0, 1, 0)
        box_T_r_goal = tf.msg_to_kdl(grasp_pose)

        grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal, drawer_frame_id)

        box_T_r_goal_post = deepcopy(box_T_r_goal)

        box_T_r_goal_post.p[0] += 0.3
        box_T_r_goal_post.p[1] += -0.2
        post_grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal_post, drawer_frame_id)

        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup.set_joint_goal(oven_area_cereal, check=False)
        kitchen_setup.plan_and_execute()

        kitchen_setup.attach_object(cereal_name, kitchen_setup.r_tip)
        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        kitchen_setup.close_l_gripper()

        # x = Vector3Stamped()
        # x.header.frame_id = 'milk'
        # x.vector.x = 1
        # x_map = Vector3Stamped()
        # x_map.header.frame_id = 'iai_kitchen/iai_fridge_door'
        # x_map.vector.x = 1
        # z = Vector3Stamped()
        # z.header.frame_id = 'milk'
        # z.vector.z = 1
        # z_map = Vector3Stamped()
        # z_map.header.frame_id = 'map'
        # z_map.vector.z = 1
        # kitchen_setup.align_planes('milk', x, root_normal=x_map)
        # kitchen_setup.align_planes('milk', z, root_normal=z_map)
        # kitchen_setup.keep_orientation(u'milk')
        # kitchen_setup.set_cart_goal(grasp_pose, cereal_name, kitchen_setup.default_root)
        kitchen_setup.set_json_goal(u'CartesianPathCarrot',
                                    tip_link=kitchen_setup.r_tip,
                                    root_link=kitchen_setup.default_root,
                                    goal=post_grasp_pose,
                                    goal_sampling_axis=[True, False, False])
        kitchen_setup.plan_and_execute()
        # kitchen_setup.set_joint_goal(gaya_pose)

        # place milk back

        # kitchen_setup.add_json_goal(u'BasePointingForward')
        # milk_goal = PoseStamped()
        # milk_goal.header.frame_id = u'iai_kitchen/kitchen_island_surface'
        # milk_goal.pose.position = Point(.1, -.2, .13)
        # milk_goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0,0,1]))

        kitchen_setup.set_json_goal(u'CartesianPathCarrot',
                                    tip_link=kitchen_setup.r_tip,
                                    root_link=kitchen_setup.default_root,
                                    goal=grasp_pose)
        kitchen_setup.plan_and_execute()

        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        kitchen_setup.open_l_gripper()
        kitchen_setup.detach_object(cereal_name)

        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.plan_and_execute()

    def publish_frame(self, name, pose_stamped):
        from giskardpy.utils.tfwrapper import normalize_quaternion_msg
        from tf2_msgs.msg import TFMessage
        from geometry_msgs.msg import TransformStamped

        def make_transform(parent_frame, child_frame, pose):
            tf = TransformStamped()
            tf.header.frame_id = parent_frame
            tf.header.stamp = rospy.get_rostime()
            tf.child_frame_id = child_frame
            tf.transform.translation.x = pose.position.x
            tf.transform.translation.y = pose.position.y
            tf.transform.translation.z = pose.position.z
            tf.transform.rotation = normalize_quaternion_msg(pose.orientation)
            return tf

        pub = rospy.Publisher('/tf', TFMessage)
        msg = TFMessage()
        msg.transforms.append(make_transform(pose_stamped.header.frame_id, name, pose_stamped.pose))
        pub.publish(msg)

    @pytest.mark.repeat(5)
    def test_cereal_1_smaller(self, kitchen_setup):
        # FIXME collision avoidance needs soft_threshholds at 0
        cereal_name = u'cereal'
        drawer_frame_id = u'iai_kitchen/oven_area_area_right_drawer_board_2_link'

        # take milk out of fridge
        kitchen_setup.set_kitchen_js({u'oven_area_area_right_drawer_main_joint': 0.48})

        cereal_pose = PoseStamped()
        cereal_pose.header.frame_id = drawer_frame_id
        cereal_pose.pose.position = Point(0.123, 0.25, 0.15)
        cereal_pose.pose.orientation = Quaternion(0.0087786, 0.005395, -0.838767, -0.544393)
        kitchen_setup.add_box(cereal_name, [0.1028, 0.0634, 0.20894], cereal_pose)

        drawer_T_pick_box = tf.msg_to_kdl(cereal_pose)
        cereal_pose.pose.position = Point(0.123, 0.0, 0.15)
        drawer_T_place_box = tf.msg_to_kdl(cereal_pose)

        # grasp milk
        kitchen_setup.open_l_gripper()
        grasp_pose = PoseStamped()
        grasp_pose.header.frame_id = cereal_name
        grasp_pose.pose.position = Point(0.03, 0, 0.05)
        grasp_pose.pose.orientation = Quaternion(0, 0, 1, 0)
        box_T_r_goal = tf.msg_to_kdl(grasp_pose)

        grasp_pose = tf.kdl_to_pose_stamped(drawer_T_pick_box * box_T_r_goal, drawer_frame_id)

        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup.set_cart_goal(grasp_pose, tip_link=kitchen_setup.r_tip)
        kitchen_setup.plan_and_execute()

        kitchen_setup.attach_object(cereal_name, kitchen_setup.r_tip)
        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        kitchen_setup.close_l_gripper()

        grasp_pose = tf.kdl_to_pose_stamped(drawer_T_place_box * box_T_r_goal, drawer_frame_id)
        kitchen_setup.set_json_goal(u'CartesianPathCarrot',
                                    tip_link=kitchen_setup.r_tip,
                                    root_link=kitchen_setup.default_root,
                                    goal=grasp_pose)
        kitchen_setup.plan_and_execute()

        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        kitchen_setup.open_l_gripper()
        kitchen_setup.detach_object(cereal_name)

        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.plan_and_execute()

    def test_cereal_1_native(self, kitchen_setup):
        # FIXME collision avoidance needs soft_threshholds at 0
        tj_1 = [[0.926, 1.451, 1.09, 0.00874306, 0.00536876, 0.83864, 0.54459],
                [0.930764, 1.44635, 1.09172, -0.00132936, -0.00586545, 0.844098, 0.536156],
                [0.935528, 1.44171, 1.09344, -0.0114013, -0.0170977, 0.849278, 0.527545],
                [0.940292, 1.43706, 1.09516, -0.0214696, -0.0283244, 0.85418, 0.518761],
                [0.945056, 1.43242, 1.09688, -0.0315307, -0.0395418, 0.858801, 0.509807],
                [0.950562, 1.43231, 1.0961, -0.0382918, -0.0457099, 0.849497, 0.524213],
                [0.956068, 1.43219, 1.09532, -0.0450384, -0.0518608, 0.839873, 0.53842],
                [0.961574, 1.43208, 1.09455, -0.0517679, -0.0579921, 0.829931, 0.552424],
                [0.967079, 1.43197, 1.09377, -0.0584779, -0.0641014, 0.819675, 0.566219],
                [0.965901, 1.42975, 1.09144, -0.0743229, -0.0780225, 0.815678, 0.568381],
                [0.964722, 1.42753, 1.08912, -0.0901333, -0.0919073, 0.811302, 0.570279],
                [0.963544, 1.4253, 1.08679, -0.105902, -0.105749, 0.806548, 0.571911],
                [0.962365, 1.42308, 1.08447, -0.121621, -0.119542, 0.801418, 0.573277],
                [0.96375, 1.42021, 1.08743, -0.12741, -0.137095, 0.793881, 0.578554],
                [0.965135, 1.41734, 1.0904, -0.133144, -0.154589, 0.786006, 0.583584],
                [0.96652, 1.41446, 1.09336, -0.138822, -0.172017, 0.777795, 0.588365],
                [0.967905, 1.41159, 1.09632, -0.14444, -0.189372, 0.769253, 0.592895],
                [0.966194, 1.41405, 1.09334, -0.142953, -0.207973, 0.761268, 0.597313],
                [0.964483, 1.41652, 1.09036, -0.141404, -0.226484, 0.752955, 0.601473],
                [0.962772, 1.41898, 1.08738, -0.139793, -0.244897, 0.744316, 0.605374],
                [0.961061, 1.42144, 1.0844, -0.138123, -0.263204, 0.735356, 0.609013],
                [0.95889, 1.41544, 1.08352, -0.138992, -0.280804, 0.729686, 0.607774],
                [0.956719, 1.40943, 1.08264, -0.139813, -0.298306, 0.723764, 0.606326],
                [0.954548, 1.40342, 1.08176, -0.140586, -0.315706, 0.717593, 0.604669],
                [0.952377, 1.39741, 1.08088, -0.141311, -0.332997, 0.711175, 0.602805],
                [0.956162, 1.39064, 1.08251, -0.151296, -0.344764, 0.709492, 0.595708],
                [0.959946, 1.38386, 1.08414, -0.161238, -0.356431, 0.707602, 0.588437],
                [0.963731, 1.37709, 1.08578, -0.171132, -0.367994, 0.705506, 0.580995],
                [0.967515, 1.37032, 1.08741, -0.180977, -0.379449, 0.703205, 0.573384],
                [0.971905, 1.36595, 1.08743, -0.175806, -0.394885, 0.694067, 0.575699],
                [0.976294, 1.36159, 1.08745, -0.170572, -0.410182, 0.684683, 0.577811],
                [0.980683, 1.35723, 1.08747, -0.165278, -0.425333, 0.675057, 0.579718],
                [0.985073, 1.35287, 1.0875, -0.159925, -0.440334, 0.665192, 0.58142],
                [0.988819, 1.35228, 1.09008, -0.16736, -0.456665, 0.657702, 0.575219],
                [0.992565, 1.3517, 1.09267, -0.174725, -0.472804, 0.649937, 0.568778],
                [0.996311, 1.35111, 1.09526, -0.182017, -0.488747, 0.641902, 0.5621],
                [1.00006, 1.35053, 1.09785, -0.189234, -0.504487, 0.6336, 0.555189],
                [1.00509, 1.35033, 1.09425, -0.189658, -0.520379, 0.624794, 0.550335],
                [1.01012, 1.35014, 1.09065, -0.190015, -0.536086, 0.615766, 0.545287],
                [1.01515, 1.34995, 1.08705, -0.190305, -0.551604, 0.606521, 0.540046],
                [1.02018, 1.34975, 1.08345, -0.190527, -0.566927, 0.597061, 0.534613],
                [1.02269, 1.3443, 1.08718, -0.189089, -0.577723, 0.583397, 0.538637],
                [1.0252, 1.33885, 1.09091, -0.18759, -0.588333, 0.569545, 0.542488],
                [1.02771, 1.33339, 1.09464, -0.186031, -0.598754, 0.55551, 0.546164],
                [1.03021, 1.32794, 1.09837, -0.184412, -0.608983, 0.541296, 0.549664],
                [1.02521, 1.32544, 1.094, -0.191963, -0.62502, 0.522063, 0.547676],
                [1.02021, 1.32294, 1.08962, -0.199383, -0.640628, 0.502471, 0.545312],
                [1.0152, 1.32044, 1.08525, -0.206665, -0.655795, 0.482534, 0.542572],
                [1.0221615825604722, 1.2981679599114058, 1.0853390936267142, -0.18983362077376312, -0.6047770524999773,
                 0.5347349629621968, 0.5588080462727618],
                [1.029123165120944, 1.2758959198228117, 1.0854281872534286, -0.17188712824450564, -0.5502065458361619,
                 0.5837947596676017, 0.5717615330253896],
                [1.0360847476814161, 1.2536238797342174, 1.0855172808801428, -0.1529308549986148, -0.4924037554881342,
                 0.629424952907031, 0.5813561075868697],
                [1.0430463302418882, 1.2313518396456233, 1.0856063745068572, -0.13307616294770333, -0.43170825396290624,
                 0.671357480204254, 0.5875354049665518],
                [1.0500079128023603, 1.209079799557029, 1.0856954681335713, -0.11243969192031725, -0.36847660750050104,
                 0.7093460016622876, 0.5902631238113321],
                [1.0569694953628321, 1.1868077594684348, 1.0857845617602855, -0.09114267444105353, -0.30308028136379694,
                 0.7431673471298028, 0.5895232396642435],
                [1.0639310779233042, 1.1645357193798407, 1.0858736553869999, -0.06931022352908946, -0.23590345760147843,
                 0.7726228272528479, 0.5853200991029022],
                [1.0708926604837763, 1.1422636792912464, 1.085962749013714, -0.04707059770007358, -0.1673407781040275,
                 0.7975394007098807, 0.5776783942047867],
                [1.0778542430442484, 1.1199916392026523, 1.0860518426404282, -0.02455444748925329, -0.09779502621155987,
                 0.8177706907729954, 0.5666430174893503],
                [1.088090925162347, 1.0872416441364399, 1.0861828506587201, 0.0087786000250608, 0.0053950000154014315,
                 0.8387670057979981, 0.5443930015541114]]
        tj_2 = [[0.925, 1.452, 1.09, 0.00874038, 0.00537245, 0.838893, 0.544199],
                [0.928428, 1.45123, 1.08769, 0.00872036, 0.00118408, 0.827647, 0.56118],
                [0.931855, 1.45046, 1.08537, 0.00869658, -0.0030048, 0.816043, 0.577917],
                [0.935283, 1.44969, 1.08306, 0.00866904, -0.00719238, 0.804087, 0.594405],
                [0.938711, 1.44893, 1.08074, 0.00863774, -0.0113769, 0.791783, 0.610636],
                [0.93768, 1.45024, 1.08145, -0.000405319, -0.0297763, 0.784857, 0.61896],
                [0.93665, 1.45155, 1.08216, -0.00944816, -0.0481598, 0.77751, 0.626952],
                [0.935619, 1.45287, 1.08286, -0.0184859, -0.0665173, 0.769745, 0.634607],
                [0.934589, 1.45418, 1.08357, -0.0275138, -0.0848391, 0.761566, 0.64192],
                [0.937217, 1.44403, 1.08425, -0.021549, -0.0866394, 0.753061, 0.651866],
                [0.939845, 1.43389, 1.08493, -0.0155797, -0.0884214, 0.744396, 0.661674],
                [0.942473, 1.42374, 1.08561, -0.00960708, -0.0901848, 0.735576, 0.671344],
                [0.945101, 1.41359, 1.08629, -0.00363248, -0.0919293, 0.726601, 0.680872],
                [0.949455, 1.41081, 1.08493, -0.0155013, -0.106245, 0.721237, 0.684317],
                [0.95381, 1.40803, 1.08357, -0.0273642, -0.12052, 0.715594, 0.687496],
                [0.958164, 1.40526, 1.08221, -0.0392164, -0.134748, 0.709675, 0.690411],
                [0.962518, 1.40248, 1.08085, -0.0510536, -0.148924, 0.703481, 0.693058],
                [0.9637, 1.40408, 1.07927, -0.0568168, -0.170222, 0.699514, 0.69172],
                [0.964882, 1.40569, 1.07769, -0.0625514, -0.191435, 0.695193, 0.690034],
                [0.966063, 1.4073, 1.07612, -0.0682545, -0.212551, 0.690522, 0.687999],
                [0.967245, 1.4089, 1.07454, -0.0739231, -0.23356, 0.685503, 0.685617],
                [0.965894, 1.40669, 1.07724, -0.0918644, -0.244179, 0.682499, 0.682739],
                [0.964542, 1.40449, 1.07994, -0.109764, -0.254687, 0.679187, 0.679552],
                [0.963191, 1.40228, 1.08264, -0.127614, -0.265081, 0.675568, 0.676058],
                [0.961839, 1.40007, 1.08534, -0.145407, -0.275355, 0.671644, 0.672258],
                [0.959327, 1.39734, 1.08176, -0.141234, -0.288082, 0.659241, 0.680046],
                [0.956815, 1.39461, 1.07819, -0.137006, -0.300696, 0.646579, 0.687566],
                [0.954303, 1.39188, 1.07461, -0.132723, -0.313192, 0.633662, 0.694815],
                [0.951791, 1.38915, 1.07103, -0.128389, -0.325564, 0.620495, 0.701791],
                [0.949421, 1.39431, 1.06795, -0.137081, -0.323018, 0.608065, 0.712127],
                [0.947052, 1.39948, 1.06486, -0.145727, -0.320362, 0.595426, 0.722219],
                [0.944682, 1.40465, 1.06177, -0.154322, -0.317596, 0.582583, 0.732062],
                [0.942312, 1.40982, 1.05868, -0.162865, -0.31472, 0.569539, 0.741654],
                [0.946576, 1.41138, 1.05761, -0.151868, -0.329312, 0.560978, 0.744173],
                [0.950839, 1.41294, 1.05653, -0.140808, -0.343767, 0.552184, 0.746385],
                [0.955102, 1.41449, 1.05546, -0.12969, -0.35808, 0.543162, 0.748288],
                [0.959366, 1.41605, 1.05438, -0.118519, -0.372245, 0.533915, 0.749881],
                [0.962108, 1.41501, 1.05866, -0.129953, -0.386526, 0.530819, 0.742927],
                [0.96485, 1.41397, 1.06293, -0.141336, -0.400656, 0.527515, 0.735681],
                [0.967592, 1.41293, 1.06721, -0.152663, -0.414629, 0.524004, 0.728146],
                [0.970335, 1.41189, 1.07148, -0.163931, -0.428438, 0.520287, 0.720326],
                [0.973848, 1.41422, 1.07298, -0.176828, -0.43493, 0.506048, 0.723521],
                [0.977362, 1.41654, 1.07447, -0.189651, -0.441239, 0.491596, 0.726412],
                [0.980876, 1.41887, 1.07596, -0.202394, -0.447362, 0.476936, 0.728996],
                [0.984389, 1.42119, 1.07746, -0.215052, -0.453297, 0.462076, 0.731273],
                [0.979102, 1.42363, 1.0767, -0.213129, -0.468551, 0.450889, 0.729202],
                [0.973815, 1.42608, 1.07594, -0.211129, -0.483635, 0.439536, 0.726863],
                [0.968528, 1.42852, 1.07518, -0.209051, -0.498541, 0.428023, 0.724259],
                [0.963241, 1.43096, 1.07442, -0.206897, -0.513264, 0.416353, 0.72139],
                [0.967336, 1.42923, 1.07415, -0.221637, -0.522478, 0.406978, 0.715725],
                [0.971432, 1.4275, 1.07387, -0.236283, -0.531471, 0.397432, 0.709758],
                [0.975527, 1.42577, 1.0736, -0.250829, -0.54024, 0.387717, 0.703492],
                [0.979623, 1.42404, 1.07332, -0.265269, -0.548781, 0.377839, 0.696928],
                [0.985835, 1.41386, 1.07645, -0.268117, -0.557605, 0.378665, 0.688334],
                [0.992048, 1.40368, 1.07957, -0.270921, -0.566339, 0.37943, 0.679628],
                [0.998261, 1.3935, 1.0827, -0.273682, -0.574982, 0.380134, 0.670814],
                [1.00447, 1.38332, 1.08583, -0.276399, -0.583533, 0.380778, 0.661892],
                [1.00308, 1.37404, 1.08709, -0.289032, -0.57661, 0.364459, 0.671677],
                [1.00168, 1.36476, 1.08836, -0.301501, -0.569359, 0.347933, 0.68108],
                [1.00029, 1.35549, 1.08963, -0.313798, -0.561784, 0.331209, 0.690095],
                [1.007934823644028, 1.332133624545122, 1.0893298568210987, -0.29193631141015736, -0.5236663700100012,
                 0.39200468453764337, 0.6977671891837012],
                [1.0155796472880563, 1.3087772490902438, 1.0890297136421971, -0.2684148635278293, -0.48257151436059986,
                 0.45057164314742093, 0.7014723009189227],
                [1.0232244709320844, 1.2854208736353658, 1.0887295704632958, -0.2433672457284187, -0.43873282359574256,
                 0.5065767134987613, 0.7011889378384863],
                [1.0308692945761126, 1.2620644981804876, 1.0884294272843944, -0.21693587531429737, -0.39239955846861635,
                 0.5597014584811004, 0.6969187111058102],
                [1.0385141182201407, 1.2387081227256096, 1.088129284105493, -0.18927103741447446, -0.343835163538393,
                 0.6096438181169129, 0.68868590064151],
                [1.046158941864169, 1.2153517472707316, 1.0878291409265914, -0.16053003048406803, -0.2933157692607772,
                 0.6561198270331093, 0.6765373170711172],
                [1.053803765508197, 1.1919953718158534, 1.08752899774769, -0.13087627192697063, -0.24112862194829546,
                 0.6988652290490092, 0.6605420355663141],
                [1.0614485891522254, 1.1686389963609753, 1.0872288545687887, -0.10047836892701467, -0.187570450527356,
                 0.7376369797005581, 0.6407910030930287],
                [1.0690934127962535, 1.1452826209060971, 1.0869287113898873, -0.06950915977076631, -0.13294577937847335,
                 0.7722146281576515, 0.6173965212995228],
                [1.0767382364402818, 1.121926245451219, 1.0866285682109857, -0.03814473111284949, -0.07756519685261311,
                 0.8024015706771698, 0.5904916079846982],
                [1.088090925162347, 1.0872416441364399, 1.0861828506587201, 0.0087786000250608, 0.0053950000154014315,
                 0.8387670057979981, 0.5443930015541114]]
        tj_3 = [[0.927, 1.451, 1.09, 0.00874379, 0.00536766, 0.83857, 0.544697],
                [0.930341, 1.44952, 1.08769, 0.0196722, -0.0121788, 0.838152, 0.544945],
                [0.933683, 1.44804, 1.08539, 0.0305923, -0.02972, 0.837376, 0.54496],
                [0.937024, 1.44656, 1.08308, 0.0414993, -0.0472485, 0.836242, 0.544742],
                [0.940366, 1.44507, 1.08077, 0.0523885, -0.0647568, 0.834751, 0.544292],
                [0.944221, 1.44143, 1.0821, 0.0407663, -0.0757553, 0.84062, 0.53475],
                [0.948077, 1.4378, 1.08344, 0.0291285, -0.0867249, 0.846168, 0.525005],
                [0.951932, 1.43416, 1.08477, 0.0174796, -0.0976615, 0.851393, 0.515059],
                [0.955788, 1.43052, 1.0861, 0.005824, -0.108561, 0.856294, 0.504917],
                [0.956952, 1.42776, 1.08669, -0.0145608, -0.10543, 0.860302, 0.498551],
                [0.958117, 1.42501, 1.08727, -0.0349385, -0.102248, 0.863895, 0.491944],
                [0.959282, 1.42226, 1.08786, -0.0552994, -0.0990163, 0.867073, 0.485101],
                [0.960446, 1.4195, 1.08844, -0.0756336, -0.0957373, 0.869832, 0.478023],
                [0.961617, 1.41803, 1.088, -0.0856453, -0.10824, 0.875791, 0.462536],
                [0.962788, 1.41655, 1.08755, -0.0956114, -0.120686, 0.881284, 0.446802],
                [0.963959, 1.41508, 1.08711, -0.105527, -0.133067, 0.886308, 0.43083],
                [0.96513, 1.4136, 1.08666, -0.115386, -0.145378, 0.890861, 0.41463],
                [0.969736, 1.41266, 1.08821, -0.132475, -0.155051, 0.886812, 0.414697],
                [0.974341, 1.41171, 1.08976, -0.149511, -0.164663, 0.882406, 0.414598],
                [0.978946, 1.41077, 1.09131, -0.166487, -0.174208, 0.877646, 0.414332],
                [0.983551, 1.40983, 1.09285, -0.183397, -0.183683, 0.872533, 0.4139],
                [0.984725, 1.4096, 1.09184, -0.17724, -0.200012, 0.877158, 0.398968],
                [0.985899, 1.40937, 1.09082, -0.170986, -0.216232, 0.881301, 0.383817],
                [0.987072, 1.40914, 1.0898, -0.164638, -0.232333, 0.884961, 0.368456],
                [0.988246, 1.40891, 1.08878, -0.158199, -0.248306, 0.888135, 0.352892],
                [0.985129, 1.40469, 1.08799, -0.171724, -0.261227, 0.883528, 0.348783],
                [0.982012, 1.40048, 1.08719, -0.185183, -0.274047, 0.878578, 0.344538],
                [0.978895, 1.39626, 1.08639, -0.198569, -0.28676, 0.873287, 0.34016],
                [0.975778, 1.39205, 1.0856, -0.211879, -0.299362, 0.867657, 0.33565],
                [0.975236, 1.385, 1.08538, -0.224457, -0.305562, 0.866559, 0.32454],
                [0.974694, 1.37795, 1.08517, -0.236963, -0.311664, 0.865183, 0.313326],
                [0.974151, 1.3709, 1.08495, -0.249393, -0.317665, 0.863528, 0.302012],
                [0.973609, 1.36384, 1.08473, -0.261743, -0.323565, 0.861596, 0.2906],
                [0.975331, 1.36663, 1.0871, -0.266717, -0.341766, 0.85256, 0.291891],
                [0.977054, 1.36942, 1.08946, -0.271574, -0.359816, 0.843148, 0.293054],
                [0.978776, 1.3722, 1.09183, -0.276311, -0.377709, 0.833367, 0.294088],
                [0.980498, 1.37499, 1.09419, -0.280928, -0.395435, 0.823219, 0.294993],
                [0.986033, 1.38034, 1.09085, -0.281714, -0.409836, 0.815199, 0.296853],
                [0.991568, 1.38569, 1.08751, -0.282423, -0.424124, 0.806955, 0.29863],
                [0.997102, 1.39105, 1.08417, -0.283053, -0.438295, 0.798488, 0.300326],
                [1.00264, 1.3964, 1.08083, -0.283606, -0.452346, 0.7898, 0.301938],
                [1.00019, 1.38908, 1.08205, -0.292926, -0.461307, 0.779414, 0.306437],
                [0.997746, 1.38175, 1.08327, -0.302159, -0.470132, 0.768798, 0.310845],
                [0.995301, 1.37443, 1.08449, -0.311303, -0.478818, 0.757956, 0.315161],
                [0.992855, 1.36711, 1.08571, -0.320356, -0.487363, 0.746889, 0.319385],
                [0.987863, 1.36238, 1.08925, -0.316884, -0.49269, 0.738778, 0.333238],
                [0.982871, 1.35765, 1.09278, -0.313318, -0.49787, 0.730448, 0.346992],
                [0.977879, 1.35292, 1.09632, -0.309659, -0.502901, 0.721899, 0.360642],
                [0.972887, 1.34819, 1.09986, -0.305907, -0.507783, 0.713136, 0.374185],
                [0.970784, 1.34414, 1.09827, -0.315236, -0.504693, 0.70336, 0.388839],
                [0.968681, 1.3401, 1.09668, -0.324437, -0.501398, 0.693297, 0.403335],
                [0.966578, 1.33605, 1.09509, -0.333506, -0.497899, 0.682953, 0.417667],
                [0.964475, 1.33201, 1.0935, -0.34244, -0.494197, 0.672331, 0.431829],
                [0.966698, 1.31892, 1.09875, -0.366678, -0.471565, 0.667686, 0.444262],
                [0.96892, 1.30583, 1.104, -0.390449, -0.448332, 0.662189, 0.456129],
                [0.9822112724651558, 1.281450585377389, 1.102012832525008, -0.35112126529029614, -0.4037343924702254,
                 0.6975930954897203, 0.47652520449889063],
                [0.9955025449303115, 1.257071170754778, 1.100025665050016, -0.3099659072243361, -0.35703530470612443,
                 0.7293661484643225, 0.4944410470686259],
                [1.0087938173954674, 1.2326917561321669, 1.0980384975750241, -0.2671971541059359, -0.30847782225817716,
                 0.7573427976799211, 0.5097832880890578],
                [1.022085089860623, 1.2083123415095558, 1.096051330100032, -0.22303762040033878, -0.2583146903480825,
                 0.7813774226460803, 0.5224720700817389],
                [1.0353763623257788, 1.1839329268869447, 1.09406416262504, -0.17771715968672244, -0.2068070117195693,
                 0.8013449213947971, 0.5324413470151171],
                [1.0486676347909347, 1.1595535122643337, 1.092076995150048, -0.13147166825363182, -0.1542228875796575,
                 0.817141361645327, 0.5396392280789271],
                [1.0619589072560904, 1.1351740976417226, 1.090089827675056, -0.08454185724166918, -0.10083602211240998,
                 0.8286845217792526, 0.5440282477796227],
                [1.088090925162347, 1.0872416441364399, 1.0861828506587201, 0.0087786000250608, 0.0053950000154014315,
                 0.8387670057979981, 0.5443930015541114]]
        tj_4 = [[0.926, 1.452, 1.09, 0.00873964, 0.00537412, 0.838971, 0.544079],
                [0.92925, 1.45239, 1.08754, 0.0117646, 0.0127651, 0.828222, 0.560132],
                [0.9325, 1.45279, 1.08508, 0.0147845, 0.0201506, 0.81711, 0.57594],
                [0.93575, 1.45318, 1.08262, 0.0177979, 0.0275272, 0.805641, 0.591496],
                [0.939, 1.45358, 1.08016, 0.0208035, 0.0348918, 0.793821, 0.606794],
                [0.94065, 1.45304, 1.08433, 0.026587, 0.0389009, 0.781766, 0.62179],
                [0.942301, 1.45251, 1.08849, 0.0323594, 0.0428937, 0.769383, 0.636525],
                [0.943951, 1.45197, 1.09266, 0.0381182, 0.0468685, 0.756676, 0.650992],
                [0.945601, 1.45144, 1.09683, 0.043861, 0.0508236, 0.743653, 0.665187],
                [0.944006, 1.4517, 1.09495, 0.0573587, 0.0450611, 0.731855, 0.677545],
                [0.942412, 1.45195, 1.09307, 0.0708273, 0.0392758, 0.719686, 0.68956],
                [0.940817, 1.45221, 1.0912, 0.08426, 0.0334706, 0.707152, 0.701225],
                [0.939222, 1.45247, 1.08932, 0.0976499, 0.0276484, 0.694259, 0.712534],
                [0.942664, 1.451, 1.0877, 0.108851, 0.0446974, 0.689928, 0.71425],
                [0.946107, 1.44954, 1.08609, 0.120004, 0.0617269, 0.685295, 0.715653],
                [0.949549, 1.44807, 1.08447, 0.131104, 0.0787293, 0.680362, 0.716743],
                [0.952991, 1.44661, 1.08286, 0.142147, 0.0956972, 0.675132, 0.717519],
                [0.956286, 1.43942, 1.08075, 0.150083, 0.102318, 0.664249, 0.725107],
                [0.95958, 1.43223, 1.07865, 0.157977, 0.108911, 0.653178, 0.732489],
                [0.962875, 1.42504, 1.07654, 0.165825, 0.115472, 0.641922, 0.739665],
                [0.96617, 1.41785, 1.07444, 0.173627, 0.122001, 0.630485, 0.746631],
                [0.972041, 1.41457, 1.07612, 0.18822, 0.128992, 0.622661, 0.748483],
                [0.977912, 1.4113, 1.0778, 0.202752, 0.135941, 0.614635, 0.750091],
                [0.983784, 1.40802, 1.07948, 0.217217, 0.142845, 0.606407, 0.751453],
                [0.989655, 1.40474, 1.08116, 0.231611, 0.149703, 0.597982, 0.752571],
                [0.993559, 1.40054, 1.08243, 0.249451, 0.151165, 0.591282, 0.751871],
                [0.997463, 1.39634, 1.0837, 0.2672, 0.152571, 0.584365, 0.750895],
                [1.00137, 1.39213, 1.08497, 0.284851, 0.153921, 0.577235, 0.749645],
                [1.00527, 1.38793, 1.08624, 0.302398, 0.155215, 0.569894, 0.748121],
                [1.0076, 1.38728, 1.08733, 0.31406, 0.172825, 0.567575, 0.741186],
                [1.00992, 1.38664, 1.08842, 0.325565, 0.190348, 0.564973, 0.73388],
                [1.01225, 1.38599, 1.0895, 0.336907, 0.207777, 0.562089, 0.726209],
                [1.01457, 1.38534, 1.09059, 0.348081, 0.225101, 0.558924, 0.718174],
                [1.01768, 1.37923, 1.09157, 0.362646, 0.217675, 0.56176, 0.711008],
                [1.02078, 1.37311, 1.09254, 0.377093, 0.210178, 0.564412, 0.703609],
                [1.02388, 1.367, 1.09351, 0.391417, 0.202612, 0.56688, 0.69598],
                [1.02699, 1.36088, 1.09448, 0.405613, 0.194979, 0.569163, 0.688124],
                [1.02968, 1.36349, 1.09136, 0.41852, 0.20655, 0.559451, 0.684977],
                [1.03238, 1.3661, 1.08823, 0.431258, 0.218037, 0.549513, 0.681552],
                [1.03508, 1.36871, 1.08511, 0.443821, 0.229436, 0.539352, 0.677851],
                [1.03777, 1.37132, 1.08199, 0.456204, 0.240742, 0.528973, 0.673876],
                [1.03555, 1.36968, 1.0843, 0.459888, 0.259139, 0.518799, 0.672457],
                [1.03332, 1.36803, 1.08662, 0.463361, 0.277417, 0.508388, 0.67073],
                [1.03109, 1.36639, 1.08894, 0.466622, 0.295568, 0.497744, 0.668696],
                [1.02886, 1.36475, 1.09125, 0.469669, 0.313585, 0.486872, 0.666357],
                [1.02971, 1.35657, 1.09558, 0.480785, 0.324346, 0.492567, 0.648863],
                [1.03056, 1.34839, 1.09991, 0.491624, 0.334919, 0.497978, 0.630994],
                [1.0314, 1.34021, 1.10424, 0.502178, 0.345299, 0.503101, 0.612761],
                [1.0372776279891829, 1.313982629666154, 1.1023678625094013, 0.4587175507368504, 0.31534322238701035,
                 0.5530840563832735, 0.6198668303072731],
                [1.0431552559783657, 1.2877552593323083, 1.1004957250188026, 0.41281006505836787, 0.2837052411332115,
                 0.6001166355781291, 0.6236659442701994],
                [1.0490328839675485, 1.2615278889984627, 1.0986235875282038, 0.36470025589938093, 0.25055370493738,
                 0.6439476190042666, 0.624137827941725],
                [1.0549105119567312, 1.2353005186646169, 1.096751450037605, 0.3146447869687515, 0.21606547578220175,
                 0.6843431702992918, 0.6212799638432859],
                [1.060788139945914, 1.209073148330771, 1.0948793125470062, 0.26291070198436434, 0.18042454684831474,
                 0.7210877809796301, 0.6151075985537345],
                [1.0666657679350966, 1.1828457779969253, 1.0930071750564072, 0.2097740000066775, 0.14382106091880986,
                 0.7539854201687772, 0.6056536613695149],
                [1.0725433959242794, 1.1566184076630794, 1.0911350375658084, 0.1555181629959125, 0.10645029597587335,
                 0.782860580413374, 0.5929685886283192],
                [1.0784210239134622, 1.1303910373292338, 1.0892629000752097, 0.10043264344830727, 0.06851162340137,
                 0.8075592140077256, 0.5771200546334521],
                [1.088090925162347, 1.0872416441364399, 1.0861828506587201, 0.0087786000250608, 0.0053950000154014315,
                 0.8387670057979981, 0.5443930015541114]]
        tj_5 = [[0.925, 1.453, 1.09, 0.00873808, 0.00537674, 0.839145, 0.54381],
                [0.929715, 1.44797, 1.08747, 0.000417929, -0.0075088, 0.843886, 0.536469],
                [0.934431, 1.44293, 1.08493, -0.00790235, -0.020392, 0.848364, 0.528961],
                [0.939146, 1.4379, 1.0824, -0.0162202, -0.0332688, 0.852577, 0.521289],
                [0.943861, 1.43287, 1.07987, -0.0245329, -0.0461353, 0.856525, 0.513453],
                [0.947524, 1.43045, 1.08275, -0.0322038, -0.0642897, 0.855346, 0.513043],
                [0.951186, 1.42804, 1.08564, -0.0398621, -0.0824191, 0.853833, 0.512432],
                [0.954848, 1.42562, 1.08853, -0.0475048, -0.100516, 0.851988, 0.511622],
                [0.958511, 1.4232, 1.09141, -0.055129, -0.118574, 0.84981, 0.510612],
                [0.956748, 1.42227, 1.09033, -0.0748876, -0.126354, 0.851855, 0.502761],
                [0.954986, 1.42134, 1.08925, -0.0946074, -0.134068, 0.853461, 0.494651],
                [0.953224, 1.42041, 1.08816, -0.114278, -0.141713, 0.854626, 0.486285],
                [0.951461, 1.41948, 1.08708, -0.13389, -0.149285, 0.855349, 0.477667],
                [0.956249, 1.41915, 1.08764, -0.1482, -0.159314, 0.847557, 0.484048],
                [0.961036, 1.41881, 1.08819, -0.162448, -0.169279, 0.839421, 0.490231],
                [0.965824, 1.41848, 1.08875, -0.176631, -0.179176, 0.830944, 0.496216],
                [0.970611, 1.41815, 1.08931, -0.190742, -0.188999, 0.822128, 0.501998],
                [0.973101, 1.41622, 1.09057, -0.206645, -0.201351, 0.814434, 0.503441],
                [0.975591, 1.41429, 1.09184, -0.222452, -0.213609, 0.806359, 0.50465],
                [0.978082, 1.41236, 1.09311, -0.238155, -0.225768, 0.797908, 0.505623],
                [0.980572, 1.41044, 1.09438, -0.253746, -0.237821, 0.789084, 0.506359],
                [0.985665, 1.40724, 1.0927, -0.249259, -0.255489, 0.787678, 0.502154],
                [0.990758, 1.40405, 1.09103, -0.244684, -0.273067, 0.785995, 0.497772],
                [0.995852, 1.40086, 1.08936, -0.240024, -0.290549, 0.784034, 0.493214],
                [1.00095, 1.39767, 1.08769, -0.235278, -0.307929, 0.781798, 0.488483],
                [0.996663, 1.39678, 1.08514, -0.234047, -0.325641, 0.780554, 0.479496],
                [0.992381, 1.39589, 1.08259, -0.232722, -0.343222, 0.778999, 0.470318],
                [0.988099, 1.395, 1.08004, -0.231305, -0.360668, 0.777135, 0.460953],
                [0.983817, 1.39411, 1.07749, -0.229796, -0.37797, 0.774962, 0.451406],
                [0.981236, 1.38745, 1.07591, -0.239782, -0.387815, 0.773186, 0.440781],
                [0.978655, 1.38078, 1.07432, -0.249692, -0.397538, 0.771169, 0.430019],
                [0.976073, 1.37412, 1.07274, -0.259525, -0.407137, 0.768911, 0.419122],
                [0.973492, 1.36746, 1.07115, -0.269276, -0.416608, 0.766412, 0.408094],
                [0.973453, 1.3605, 1.07052, -0.282818, -0.420451, 0.756523, 0.413411],
                [0.973413, 1.35353, 1.06989, -0.296269, -0.424157, 0.746388, 0.418593],
                [0.973374, 1.34657, 1.06925, -0.309624, -0.427726, 0.736012, 0.423639],
                [0.973335, 1.3396, 1.06862, -0.322878, -0.431156, 0.725397, 0.428549],
                [0.974795, 1.33436, 1.07057, -0.309894, -0.444915, 0.724706, 0.425226],
                [0.976254, 1.32912, 1.07252, -0.296796, -0.458509, 0.723748, 0.421747],
                [0.977714, 1.32388, 1.07447, -0.283589, -0.471933, 0.722523, 0.418112],
                [0.979174, 1.31863, 1.07642, -0.270276, -0.485184, 0.72103, 0.414322],
                [0.981569, 1.31803, 1.07706, -0.258625, -0.498707, 0.710793, 0.423294],
                [0.983963, 1.31743, 1.07769, -0.246842, -0.51198, 0.700197, 0.432053],
                [0.986358, 1.31682, 1.07833, -0.234936, -0.524994, 0.689249, 0.440594],
                [0.988752, 1.31622, 1.07897, -0.222911, -0.537743, 0.677953, 0.448913],
                [0.984476, 1.31034, 1.07782, -0.20611, -0.542093, 0.680768, 0.447447],
                [0.9802, 1.30447, 1.07668, -0.189245, -0.546273, 0.683371, 0.445843],
                [0.975923, 1.29859, 1.07553, -0.172321, -0.550283, 0.685762, 0.4441],
                [0.971647, 1.29272, 1.07439, -0.155343, -0.554123, 0.687939, 0.442218],
                [0.972328, 1.29293, 1.07234, -0.147126, -0.570882, 0.683975, 0.429681],
                [0.97301, 1.29315, 1.07029, -0.138833, -0.587344, 0.679654, 0.416921],
                [0.973691, 1.29337, 1.06825, -0.130467, -0.603499, 0.674979, 0.403943],
                [0.974372, 1.29358, 1.0662, -0.122034, -0.61934, 0.669952, 0.390754],
                [0.979043, 1.28563, 1.06464, -0.12293, -0.630371, 0.658928, 0.39158],
                [0.983714, 1.27767, 1.06309, -0.123797, -0.641247, 0.647741, 0.39231],
                [0.988384, 1.26971, 1.06153, -0.124634, -0.651967, 0.636397, 0.392944],
                [0.993055, 1.26176, 1.05997, -0.12544, -0.662527, 0.624896, 0.393482],
                [0.991392, 1.25566, 1.05746, -0.129584, -0.675517, 0.604017, 0.402552],
                [0.989729, 1.24957, 1.05495, -0.133636, -0.688031, 0.582713, 0.411339],
                [0.988066, 1.24347, 1.05244, -0.137595, -0.700061, 0.560999, 0.419836],
                [0.9981476227019953, 1.22772357143119, 1.0558409791927248, 0.12542290914904716, 0.6425449539450735,
                 -0.6096656179124028, -0.4468925042510797],
                [1.0082292454039907, 1.2119771428623798, 1.0592419583854495, -0.11242844704321613, -0.580815877884512,
                 0.6543347920789632, 0.4710188320732777],
                [1.018310868105986, 1.1962307142935698, 1.0626429375781743, -0.09869681975190546, -0.5152785374957447,
                 0.6947136567993454, 0.4920568073135334],
                [1.0283924908079813, 1.1804842857247597, 1.066043916770899, -0.08431806211812298, -0.44636264471515547,
                 0.7305374579647465, 0.5098684892341775],
                [1.0384741135099766, 1.1647378571559497, 1.0694448959636236, -0.06938645205654968, -0.37452006380148045,
                 0.7615713078779217, 0.5243370910952715],
                [1.048555736211972, 1.1489914285871394, 1.0728458751563483, -0.0539998923968522, -0.3002218485749709,
                 0.7876117253530248, 0.5353677458960218],
                [1.0586373589139673, 1.1332450000183294, 1.076246854349073, -0.038259268959334285, -0.22395515383540066,
                 0.8084879698908048, 0.5428881283939669],
                [1.0687189816159626, 1.1174985714495194, 1.0796478335417978, -0.022267789071865763,
                 -0.14622004120998386, 0.8240631611813275, 0.5468489293234823],
                [1.088090925162347, 1.0872416441364399, 1.0861828506587201, 0.0087786000250608, 0.0053950000154014315,
                 0.8387670057979981, 0.5443930015541114]]
        tj_6 = [[0.926, 1.451, 1.09, 0.0087444, 0.00536657, 0.838518, 0.544777],
                [0.930386, 1.44991, 1.09169, -0.00057711, 0.00523067, 0.848164, 0.529707],
                [0.934772, 1.44883, 1.09338, -0.00989838, 0.00509265, 0.857465, 0.514422],
                [0.939158, 1.44774, 1.09507, -0.0192156, 0.00495256, 0.866417, 0.498927],
                [0.943544, 1.44665, 1.09676, -0.0285251, 0.00481045, 0.875016, 0.48323],
                [0.943768, 1.44749, 1.09508, -0.0464045, -0.00960946, 0.875569, 0.480763],
                [0.943992, 1.44833, 1.0934, -0.0642591, -0.0240242, 0.875655, 0.47804],
                [0.944217, 1.44917, 1.09172, -0.0820794, -0.0384262, 0.875273, 0.475061],
                [0.944441, 1.45001, 1.09004, -0.0998559, -0.0528076, 0.874424, 0.471828],
                [0.94679, 1.45089, 1.09115, -0.0933782, -0.0609316, 0.883745, 0.454492],
                [0.94914, 1.45176, 1.09226, -0.0868543, -0.0690254, 0.892627, 0.436931],
                [0.95149, 1.45264, 1.09337, -0.0802874, -0.077085, 0.901067, 0.419153],
                [0.953839, 1.45352, 1.09447, -0.0736807, -0.0851065, 0.909061, 0.401168],
                [0.956837, 1.4536, 1.09192, -0.0579869, -0.081165, 0.915649, 0.389406],
                [0.959834, 1.45369, 1.08937, -0.0422674, -0.0771875, 0.92183, 0.377472],
                [0.962832, 1.45377, 1.08681, -0.0265291, -0.0731757, 0.927602, 0.365371],
                [0.965829, 1.45385, 1.08426, -0.0107791, -0.0691315, 0.932963, 0.353107],
                [0.968163, 1.452, 1.08632, -0.0125025, -0.0579714, 0.939877, 0.336325],
                [0.970497, 1.45015, 1.08838, -0.0142202, -0.0467847, 0.946361, 0.319389],
                [0.972831, 1.4483, 1.09044, -0.0159314, -0.0355766, 0.952413, 0.302307],
                [0.975165, 1.44644, 1.0925, -0.0176354, -0.0243523, 0.95803, 0.285087],
                [0.978007, 1.44246, 1.09225, -0.016725, -0.0420431, 0.960142, 0.275825],
                [0.980849, 1.43848, 1.092, -0.0158079, -0.0597169, 0.961867, 0.266451],
                [0.983692, 1.4345, 1.09175, -0.0148844, -0.0773666, 0.963203, 0.25697],
                [0.986534, 1.43052, 1.09151, -0.0139549, -0.0949851, 0.964149, 0.247385],
                [0.987351, 1.42885, 1.09169, -0.0174088, -0.10573, 0.967856, 0.227536],
                [0.988168, 1.42718, 1.09187, -0.0208534, -0.116417, 0.971045, 0.207566],
                [0.988985, 1.42551, 1.09206, -0.0242868, -0.127043, 0.973715, 0.187484],
                [0.989802, 1.42384, 1.09224, -0.0277073, -0.137601, 0.975863, 0.167302],
                [0.992412, 1.41677, 1.09146, -0.0242525, -0.150605, 0.97587, 0.156227],
                [0.995023, 1.40971, 1.09069, -0.0207904, -0.163563, 0.975582, 0.145105],
                [0.997634, 1.40264, 1.08991, -0.017322, -0.176472, 0.974996, 0.133938],
                [1.00024, 1.39558, 1.08913, -0.0138484, -0.189328, 0.974115, 0.122731],
                [1.0054, 1.38999, 1.08584, -0.0197464, -0.202174, 0.972497, 0.113951],
                [1.01055, 1.3844, 1.08255, -0.0256388, -0.214965, 0.970607, 0.105139],
                [1.0157, 1.37882, 1.07926, -0.0315241, -0.227695, 0.968446, 0.0962978],
                [1.02085, 1.37323, 1.07597, -0.0374006, -0.240361, 0.966014, 0.0874295],
                [1.02064, 1.37002, 1.07699, -0.0426734, -0.258877, 0.961766, 0.0785413],
                [1.02044, 1.36681, 1.07801, -0.0479263, -0.277272, 0.957067, 0.0696163],
                [1.02023, 1.3636, 1.07903, -0.0531568, -0.295537, 0.951921, 0.0606588],
                [1.02003, 1.3604, 1.08005, -0.0583625, -0.313663, 0.946329, 0.0516729],
                [1.02008, 1.35786, 1.07934, -0.0395077, -0.325258, 0.943467, 0.0501657],
                [1.02014, 1.35533, 1.07863, -0.0206332, -0.336689, 0.940133, 0.0486334],
                [1.02019, 1.3528, 1.07791, -0.00174841, -0.347953, 0.936328, 0.0470768],
                [1.02025, 1.35027, 1.0772, 0.0171373, -0.359042, 0.932054, 0.0454966],
                [1.01756, 1.34878, 1.07769, 0.0389887, -0.35967, 0.93117, 0.0451711],
                [1.01486, 1.34728, 1.07819, 0.0608214, -0.360126, 0.929839, 0.044824],
                [1.01217, 1.34579, 1.07868, 0.082625, -0.36041, 0.928063, 0.0444554],
                [1.00948, 1.34429, 1.07917, 0.104389, -0.360521, 0.925843, 0.0440656],
                [1.01633, 1.34765, 1.07528, 0.118826, -0.361691, 0.923973, 0.0365209],
                [1.02318, 1.351, 1.0714, 0.133231, -0.362764, 0.921853, 0.0289664],
                [1.03003, 1.35435, 1.06751, 0.1476, -0.363739, 0.919484, 0.021404],
                [1.03689, 1.3577, 1.06362, 0.161929, -0.364615, 0.916866, 0.0138358],
                [1.03598, 1.35595, 1.06718, 0.179316, -0.367883, 0.912412, 0.00357557],
                [1.03508, 1.35419, 1.07074, 0.196624, -0.370989, 0.907558, -0.00668625],
                [1.03417, 1.35244, 1.0743, 0.213846, -0.373933, 0.902306, -0.0169451],
                [1.03327, 1.35068, 1.07785, 0.230975, -0.376713, 0.896659, -0.0271966],
                [1.03053, 1.34647, 1.07666, 0.246616, -0.384011, 0.889154, -0.0334712],
                [1.02779, 1.34226, 1.07546, 0.26216, -0.391158, 0.8813, -0.0397327],
                [1.02505, 1.33806, 1.07426, 0.2776, -0.398151, 0.873098, -0.0459785],
                [1.02231, 1.33385, 1.07306, 0.292932, -0.404987, 0.864553, -0.0522062],
                [1.0284, 1.33601, 1.0773, 0.300019, -0.414106, 0.858387, -0.0409422],
                [1.03449, 1.33818, 1.08154, 0.307016, -0.423101, 0.851966, -0.0296661],
                [1.04059, 1.34034, 1.08578, 0.313922, -0.431971, 0.845291, -0.0183812],
                [1.04668, 1.3425, 1.09002, 0.320734, -0.440711, 0.838363, -0.00709072],
                [1.04406, 1.34196, 1.0936, 0.333855, -0.451568, 0.827358, -0.0102986],
                [1.04144, 1.34141, 1.09718, 0.346836, -0.462234, 0.816004, -0.0135021],
                [1.03882, 1.34086, 1.10076, 0.35967, -0.472706, 0.804306, -0.0167],
                [1.0362, 1.34032, 1.10434, 0.372352, -0.482978, 0.792269, -0.0198908],
                [1.03278, 1.33175, 1.10912, 0.3831, -0.50361, 0.774317, 0.00666839],
                [1.02937, 1.32318, 1.1139, 0.393246, -0.523453, 0.75515, 0.0332171],
                [1.034683400668861, 1.301830966169394, 1.1113919959206753, 0.36664653304730244, -0.4862939143559399,
                 0.788393199954682, 0.08674509167663713],
                [1.039996801337722, 1.2804819323387882, 1.1088839918413507, 0.33782574083255673, -0.446188621026705,
                 0.8168599094520682, 0.1397475281612539],
                [1.0453102020065832, 1.2591328985081822, 1.1063759877620263, 0.30695816905845696, -0.4033800116420295,
                 0.8403775225265413, 0.19190327847149624],
                [1.0506236026754443, 1.237783864677576, 1.1038679836827017, 0.27423083460180575, -0.35812745005748226,
                 0.858803553381705, 0.24289634737451213],
                [1.0559370033443054, 1.2164348308469701, 1.101359979603377, 0.23984202205463268, -0.31070510726379796,
                 0.8720263645111295, 0.29241778395720003],
                [1.0612504040131665, 1.1950857970163642, 1.0988519755240524, 0.20400008238042108, -0.26140030027136063,
                 0.8799658430747742, 0.34016755346229843],
                [1.0665638046820276, 1.1737367631857583, 1.0963439714447278, 0.16692217058136877, -0.21051175134710395,
                 0.8825738862774054, 0.38585635510430194],
                [1.0718772053508887, 1.1523877293551523, 1.0938359673654034, 0.12883293002475324, -0.15834777815056042,
                 0.8798346928082524, 0.4292073748516281],
                [1.0771906060197498, 1.1310386955245462, 1.0913279632860788, 0.08996313139965996, -0.10522442573445548,
                 0.871764858576165, 0.4699579625554828],
                [1.082504006688611, 1.1096896616939402, 1.0888199592067542, 0.050548274550197704, -0.051463551727463835,
                 0.858413276160237, 0.507861223264231],
                [1.088090925162347, 1.0872416441364399, 1.0861828506587201, 0.0087786000250608, 0.0053950000154014315,
                 0.8387670057979981, 0.5443930015541114]]
        tj_7 = [[0.926, 1.452, 1.09, 0.00873846, 0.00537554, 0.839078, 0.543914],
                [0.930105, 1.4458, 1.08762, 0.0138544, 0.00230896, 0.830158, 0.557351],
                [0.934211, 1.43961, 1.08523, 0.0189662, -0.000758293, 0.820993, 0.570622],
                [0.938316, 1.43341, 1.08285, 0.0240724, -0.00382533, 0.811586, 0.583725],
                [0.942421, 1.42722, 1.08047, 0.0291715, -0.00689123, 0.801938, 0.596655],
                [0.941259, 1.42216, 1.08182, 0.026853, -0.0263757, 0.8013, 0.597077],
                [0.940097, 1.4171, 1.08317, 0.0245241, -0.04585, 0.800353, 0.59727],
                [0.938935, 1.41204, 1.08451, 0.0221858, -0.0653066, 0.799098, 0.597232],
                [0.937773, 1.40698, 1.08586, 0.0198389, -0.0847381, 0.797534, 0.596964],
                [0.942106, 1.40573, 1.08544, 0.00425807, -0.0974247, 0.79902, 0.593344],
                [0.946439, 1.40448, 1.08502, -0.0113245, -0.11007, 0.800172, 0.589476],
                [0.950771, 1.40323, 1.0846, -0.0269024, -0.12267, 0.800988, 0.58536],
                [0.955104, 1.40198, 1.08418, -0.042469, -0.135218, 0.801469, 0.581],
                [0.958366, 1.40451, 1.08387, -0.0511573, -0.146528, 0.808581, 0.567546],
                [0.961628, 1.40705, 1.08355, -0.0598234, -0.157774, 0.815343, 0.553845],
                [0.96489, 1.40958, 1.08324, -0.0684635, -0.168951, 0.821749, 0.539904],
                [0.968151, 1.41212, 1.08293, -0.0770738, -0.180055, 0.827798, 0.525728],
                [0.970049, 1.41344, 1.08315, -0.0742327, -0.186933, 0.838163, 0.50698],
                [0.971947, 1.41476, 1.08338, -0.0713534, -0.193715, 0.848096, 0.487972],
                [0.973845, 1.41608, 1.0836, -0.0684374, -0.200398, 0.857593, 0.468712],
                [0.975743, 1.41739, 1.08382, -0.0654862, -0.206977, 0.866649, 0.449212],
                [0.974886, 1.41092, 1.08423, -0.0652248, -0.206613, 0.875067, 0.432798],
                [0.974028, 1.40444, 1.08465, -0.0649413, -0.20618, 0.883187, 0.416237],
                [0.97317, 1.39796, 1.08506, -0.0646356, -0.205675, 0.891006, 0.399534],
                [0.972313, 1.39148, 1.08547, -0.064308, -0.205101, 0.898522, 0.382695],
                [0.969743, 1.39072, 1.08271, -0.0749143, -0.2222, 0.896208, 0.376599],
                [0.967174, 1.38995, 1.07995, -0.0854871, -0.239198, 0.893492, 0.370335],
                [0.964604, 1.38919, 1.07719, -0.0960217, -0.25609, 0.890377, 0.363905],
                [0.962034, 1.38842, 1.07443, -0.106513, -0.272868, 0.886863, 0.357312],
                [0.965634, 1.38205, 1.07698, -0.119751, -0.282896, 0.883188, 0.354414],
                [0.969234, 1.37569, 1.07952, -0.132953, -0.29284, 0.879249, 0.351411],
                [0.972834, 1.36932, 1.08207, -0.146115, -0.302696, 0.875048, 0.348304],
                [0.976433, 1.36295, 1.08462, -0.159234, -0.312463, 0.870588, 0.345092],
                [0.977018, 1.36074, 1.08773, -0.152927, -0.326835, 0.861894, 0.356273],
                [0.977603, 1.35853, 1.09083, -0.146552, -0.34106, 0.852816, 0.367295],
                [0.978188, 1.35632, 1.09394, -0.14011, -0.355133, 0.843356, 0.378153],
                [0.978773, 1.35411, 1.09705, -0.133607, -0.369048, 0.83352, 0.388841],
                [0.982354, 1.3537, 1.09826, -0.12853, -0.385441, 0.832595, 0.376431],
                [0.985935, 1.3533, 1.09947, -0.123395, -0.401661, 0.831297, 0.363851],
                [0.989517, 1.35289, 1.10067, -0.118204, -0.417701, 0.829625, 0.351108],
                [0.993098, 1.35249, 1.10188, -0.112961, -0.433553, 0.827579, 0.338207],
                [0.997728, 1.35219, 1.09926, -0.103263, -0.44609, 0.818919, 0.345994],
                [1.00236, 1.35188, 1.09663, -0.0935255, -0.458455, 0.809941, 0.353648],
                [1.00699, 1.35158, 1.09401, -0.0837516, -0.470643, 0.80065, 0.361165],
                [1.01162, 1.35127, 1.09139, -0.0739452, -0.482648, 0.79105, 0.368542],
                [1.01459, 1.3499, 1.09504, -0.0635618, -0.497941, 0.783536, 0.366176],
                [1.01756, 1.34852, 1.09869, -0.0531527, -0.513034, 0.775706, 0.363663],
                [1.02053, 1.34715, 1.10235, -0.0427221, -0.527919, 0.767563, 0.361003],
                [1.02351, 1.34578, 1.106, -0.0322743, -0.542591, 0.75911, 0.358197],
                [1.02078, 1.3521, 1.10409, -0.0271926, -0.552343, 0.748049, 0.366881],
                [1.01806, 1.35843, 1.10218, -0.0221022, -0.561918, 0.736748, 0.375448],
                [1.01534, 1.36476, 1.10028, -0.0170049, -0.571315, 0.725213, 0.383896],
                [1.01261, 1.37109, 1.09837, -0.0119021, -0.580529, 0.713447, 0.392221],
                [1.01117, 1.36083, 1.1007, -0.0136474, -0.585217, 0.704111, 0.401948],
                [1.00973, 1.35056, 1.10302, -0.0153899, -0.589784, 0.69463, 0.411592],
                [1.00829, 1.3403, 1.10535, -0.0171292, -0.594228, 0.685005, 0.421151],
                [1.00684, 1.33004, 1.10767, -0.018865, -0.59855, 0.675238, 0.430623],
                [1.00334, 1.32791, 1.10253, -0.0301801, -0.605419, 0.664317, 0.43731],
                [0.999837, 1.32578, 1.09738, -0.041485, -0.612083, 0.653171, 0.443849],
                [0.996333, 1.32364, 1.09223, -0.0527759, -0.618538, 0.641803, 0.450238],
                [0.99283, 1.32151, 1.08708, -0.0640488, -0.624784, 0.630217, 0.456474],
                [0.988382, 1.316, 1.08387, -0.0715184, -0.623096, 0.621619, 0.469283],
                [0.983933, 1.31049, 1.08065, -0.0789669, -0.621223, 0.612835, 0.481954],
                [0.979484, 1.30499, 1.07744, -0.0863919, -0.619165, 0.60387, 0.494481],
                [0.975035, 1.29948, 1.07422, -0.0937913, -0.616924, 0.594726, 0.506862],
                [0.970915, 1.29964, 1.07235, -0.101334, -0.620068, 0.580004, 0.518501],
                [0.966795, 1.2998, 1.07047, -0.108834, -0.622951, 0.565038, 0.529923],
                [0.962675, 1.29997, 1.06859, -0.116289, -0.625574, 0.549836, 0.541124],
                [0.958555, 1.30013, 1.06672, -0.123695, -0.627934, 0.534403, 0.552097],
                [0.963647, 1.28882, 1.06483, -0.111275, -0.630258, 0.508514, 0.576027],
                [0.968739, 1.27751, 1.06294, -0.0986978, -0.631698, 0.481911, 0.599148],
                [0.9808688232154712, 1.258172888202398, 1.0653021878669353, -0.08939918648544831, -0.5778699386498156,
                 0.5352080475521306, 0.6096117332335678],
                [0.9929986464309423, 1.238835776404796, 1.0676643757338706, -0.07957008692375961, -0.520612852992469,
                 0.5853292142466726, 0.6164580841511936],
                [1.0051284696464133, 1.219498664607194, 1.070026563600806, -0.06926881934727519, -0.46026645642975517,
                 0.6319770438647125, 0.619646379622967],
                [1.0172582928618845, 1.2001615528095917, 1.0723887514677415, -0.058556511363736984, -0.3971888438107145,
                 0.6748747287017879, 0.6191577003388594],
                [1.0293881160773557, 1.1808244410119897, 1.0747509393346768, -0.0474967296901114, -0.3317543169902253,
                 0.7137677143710434, 0.6149949461163322],
                [1.0415179392928269, 1.1614873292143877, 1.0771131272016121, -0.03615510294839435, -0.26435116372594125,
                 0.7484252103261463, 0.6071828186928508],
                [1.053647762508298, 1.1421502174167857, 1.0794753150685474, -0.024598932226095317, -0.19537935358303646,
                 0.7786415593698308, 0.59576767514605],
                [1.065777585723769, 1.1228131056191837, 1.0818375029354828, -0.012896791712329105, -0.1252481645192343,
                 0.8042374580214305, 0.5808172528113462],
                [1.088090925162347, 1.0872416441364399, 1.0861828506587201, 0.0087786000250608, 0.0053950000154014315,
                 0.8387670057979981, 0.5443930015541114]]
        tj_8 = [[0.926, 1.452, 1.09, 0.00873846, 0.00537554, 0.839078, 0.543914],
                [0.928537, 1.44792, 1.09004, 0.0198601, -0.0103624, 0.835611, 0.548865],
                [0.931073, 1.44384, 1.09007, 0.0309737, -0.0260962, 0.831802, 0.553592],
                [0.93361, 1.43977, 1.09011, 0.0420746, -0.0418193, 0.827655, 0.558093],
                [0.936147, 1.43569, 1.09015, 0.0531583, -0.0575254, 0.82317, 0.562367],
                [0.939966, 1.4371, 1.08762, 0.0544913, -0.0657842, 0.81218, 0.577119],
                [0.943786, 1.43852, 1.0851, 0.0558021, -0.0740161, 0.800859, 0.591636],
                [0.947605, 1.43993, 1.08257, 0.0570901, -0.0822178, 0.789211, 0.60591],
                [0.951424, 1.44134, 1.08005, 0.0583548, -0.090386, 0.77724, 0.619938],
                [0.954654, 1.44292, 1.0816, 0.0406164, -0.101754, 0.777588, 0.619156],
                [0.957883, 1.4445, 1.08315, 0.02286, -0.113077, 0.777589, 0.618099],
                [0.961112, 1.44608, 1.08469, 0.00509345, -0.124349, 0.777245, 0.616767],
                [0.964341, 1.44766, 1.08624, -0.0126754, -0.135567, 0.776556, 0.61516],
                [0.967694, 1.44119, 1.08446, -0.017037, -0.148798, 0.781418, 0.605768],
                [0.971046, 1.43471, 1.08268, -0.0213933, -0.161984, 0.786041, 0.596191],
                [0.974399, 1.42823, 1.08089, -0.0257431, -0.17512, 0.790423, 0.586431],
                [0.977752, 1.42175, 1.07911, -0.0300851, -0.188203, 0.794564, 0.576492],
                [0.976522, 1.4213, 1.07836, -0.0476606, -0.20314, 0.790135, 0.576324],
                [0.975292, 1.42084, 1.07761, -0.0652098, -0.217964, 0.78527, 0.575839],
                [0.974062, 1.42038, 1.07686, -0.082723, -0.232668, 0.779972, 0.575036],
                [0.972832, 1.41992, 1.0761, -0.100191, -0.247244, 0.774243, 0.573916],
                [0.97478, 1.41565, 1.07544, -0.105366, -0.261648, 0.778114, 0.561229],
                [0.976728, 1.41137, 1.07478, -0.110499, -0.275945, 0.781665, 0.548311],
                [0.978676, 1.40709, 1.07412, -0.115586, -0.290128, 0.784895, 0.535169],
                [0.980624, 1.40282, 1.07346, -0.120626, -0.304192, 0.787803, 0.521807],
                [0.986611, 1.39744, 1.07463, -0.127548, -0.318466, 0.782523, 0.519585],
                [0.992599, 1.39205, 1.07579, -0.134433, -0.33265, 0.777021, 0.517215],
                [0.998586, 1.38667, 1.07696, -0.14128, -0.346739, 0.771297, 0.514697],
                [1.00457, 1.38129, 1.07813, -0.148086, -0.360729, 0.765354, 0.512034],
                [1.00673, 1.37499, 1.07567, -0.13576, -0.372842, 0.764929, 0.507388],
                [1.00889, 1.36869, 1.07322, -0.123389, -0.384835, 0.764259, 0.502579],
                [1.01105, 1.36239, 1.07077, -0.110979, -0.396706, 0.763343, 0.497609],
                [1.01321, 1.35609, 1.06831, -0.0985341, -0.408449, 0.762184, 0.49248],
                [1.01161, 1.35328, 1.06788, -0.110706, -0.417196, 0.765551, 0.477099],
                [1.01001, 1.35047, 1.06745, -0.122826, -0.425746, 0.768556, 0.461493],
                [1.00841, 1.34766, 1.06702, -0.134888, -0.434095, 0.771198, 0.445669],
                [1.00681, 1.34486, 1.06659, -0.146886, -0.442239, 0.773475, 0.429634],
                [1.01012, 1.3406, 1.06396, -0.135978, -0.454952, 0.773027, 0.420664],
                [1.01342, 1.33635, 1.06133, -0.125022, -0.467501, 0.772299, 0.411543],
                [1.01673, 1.3321, 1.0587, -0.11402, -0.479881, 0.771292, 0.402272],
                [1.02003, 1.32785, 1.05607, -0.102977, -0.492087, 0.770006, 0.392857],
                [1.0233, 1.32459, 1.05877, -0.0878551, -0.503193, 0.764211, 0.393777],
                [1.02657, 1.32134, 1.06146, -0.0726991, -0.514105, 0.75812, 0.394544],
                [1.02984, 1.31808, 1.06416, -0.057515, -0.524818, 0.751736, 0.395159],
                [1.0331, 1.31483, 1.06685, -0.0423087, -0.535328, 0.745062, 0.395622],
                [1.03531, 1.31044, 1.07022, -0.0233845, -0.53445, 0.746874, 0.394964],
                [1.03752, 1.30605, 1.07359, -0.0044518, -0.533378, 0.748414, 0.394163],
                [1.03973, 1.30166, 1.07696, 0.0144825, -0.532113, 0.749683, 0.393219],
                [1.04194, 1.29727, 1.08033, 0.0334115, -0.530655, 0.75068, 0.392132],
                [1.04185, 1.29293, 1.07781, 0.0316053, -0.540591, 0.752428, 0.374987],
                [1.04176, 1.28858, 1.07529, 0.0297865, -0.550311, 0.753875, 0.357692],
                [1.04168, 1.28423, 1.07277, 0.0279558, -0.559811, 0.755021, 0.340254],
                [1.04159, 1.27989, 1.07025, 0.0261139, -0.569088, 0.755866, 0.322681],
                [1.04396, 1.27492, 1.07688, 0.0327655, -0.587303, 0.740668, 0.324672],
                [1.04633, 1.26995, 1.0835, 0.039397, -0.605159, 0.725018, 0.326464],
                [1.04871, 1.26497, 1.09013, 0.0460045, -0.622646, 0.708925, 0.328057],
                [1.0532177339221533, 1.2446263403385782, 1.0896781899191514, 0.04255512434016141, -0.5605593269977553,
                 0.7429996171884123, 0.36319949224371156],
                [1.0577254678443067, 1.2242826806771565, 1.0892263798383028, 0.03883924924621352, -0.49496218219183635,
                 0.7724212108858058, 0.3960674486644811],
                [1.06223320176646, 1.2039390210157348, 1.0887745697574542, 0.03488013458126433, -0.42626522492314245,
                 0.7970053422837312, 0.42645494322848254],
                [1.0667409356886133, 1.183595361354313, 1.0883227596766056, 0.03070257519514574, -0.3548986854031076,
                 0.8165980477065063, 0.45417166738740417],
                [1.0712486696107666, 1.1632517016928912, 1.087870949595757, 0.026332733997096087, -0.2813095127049916,
                 0.8310766234110293, 0.4790440388958417],
                [1.07575640353292, 1.1429080420314695, 1.0874191395149084, 0.021797978104631553, -0.20595857564435693,
                 0.8403503940466835, 0.5009162889093162],
                [1.0802641374550732, 1.1225643823700477, 1.0869673294340598, 0.017126707450791007, -0.12931777648401468,
                 0.8443612805302019, 0.5196514375218481],
                [1.088090925162347, 1.0872416441364399, 1.0861828506587201, 0.0087786000250608, 0.005395000015401431,
                 0.8387670057979981, 0.5443930015541114]]
        tj_9 = [[0.927, 1.452, 1.09, 0.00874129, 0.00537187, 0.83883, 0.544296],
                [0.932462, 1.45399, 1.09011, 0.0170483, 0.0176952, 0.831956, 0.554297],
                [0.937923, 1.45597, 1.09022, 0.0253491, 0.030012, 0.824775, 0.564094],
                [0.943385, 1.45796, 1.09033, 0.0336405, 0.0423178, 0.817291, 0.573684],
                [0.948847, 1.45994, 1.09043, 0.0419196, 0.0546079, 0.809506, 0.583062],
                [0.950783, 1.45371, 1.09117, 0.0468445, 0.06707, 0.801279, 0.592671],
                [0.952719, 1.44747, 1.0919, 0.0517536, 0.0795094, 0.792781, 0.602079],
                [0.954655, 1.44123, 1.09263, 0.0566451, 0.0919217, 0.784013, 0.611282],
                [0.956591, 1.43499, 1.09336, 0.0615174, 0.104303, 0.774978, 0.620278],
                [0.965031, 1.44004, 1.0907, 0.0756706, 0.10591, 0.776502, 0.616524],
                [0.97347, 1.44509, 1.08803, 0.0898072, 0.107493, 0.777855, 0.612634],
                [0.98191, 1.45014, 1.08536, 0.103924, 0.109054, 0.779038, 0.60861],
                [0.99035, 1.4552, 1.0827, 0.118018, 0.11059, 0.78005, 0.604453],
                [0.988431, 1.45942, 1.08514, 0.117865, 0.130194, 0.777826, 0.603444],
                [0.986512, 1.46365, 1.08757, 0.117665, 0.149748, 0.775298, 0.602199],
                [0.984592, 1.46787, 1.09001, 0.11742, 0.169243, 0.772467, 0.60072],
                [0.982673, 1.4721, 1.09245, 0.117128, 0.188673, 0.769335, 0.599005],
                [0.985658, 1.46466, 1.09337, 0.126153, 0.192291, 0.75938, 0.608647],
                [0.988642, 1.45722, 1.09428, 0.135142, 0.195853, 0.749207, 0.618115],
                [0.991627, 1.44977, 1.0952, 0.144093, 0.19936, 0.73882, 0.627405],
                [0.994612, 1.44233, 1.09612, 0.153002, 0.202809, 0.728221, 0.636516],
                [0.993211, 1.44334, 1.0943, 0.167203, 0.207244, 0.71476, 0.646692],
                [0.99181, 1.44434, 1.09249, 0.18132, 0.211574, 0.700938, 0.656541],
                [0.990409, 1.44535, 1.09067, 0.195345, 0.215797, 0.68676, 0.666057],
                [0.989008, 1.44635, 1.08886, 0.209271, 0.219911, 0.672235, 0.675237],
                [0.990306, 1.44461, 1.08955, 0.208451, 0.213602, 0.657774, 0.69156],
                [0.991604, 1.44286, 1.09023, 0.207523, 0.207182, 0.642974, 0.707527],
                [0.992902, 1.44111, 1.09092, 0.206489, 0.200656, 0.627842, 0.723128],
                [0.9942, 1.43936, 1.09161, 0.205347, 0.194026, 0.612386, 0.738356],
                [0.994795, 1.4318, 1.08846, 0.220565, 0.189357, 0.607185, 0.739473],
                [0.99539, 1.42423, 1.08531, 0.235721, 0.184635, 0.601814, 0.740382],
                [0.995985, 1.41667, 1.08216, 0.25081, 0.17986, 0.596273, 0.741083],
                [0.99658, 1.4091, 1.07901, 0.265828, 0.175036, 0.590564, 0.741574],
                [1.00019, 1.41113, 1.08, 0.275625, 0.190392, 0.58066, 0.742035],
                [1.00381, 1.41316, 1.081, 0.285302, 0.205666, 0.570508, 0.742176],
                [1.00742, 1.41519, 1.08199, 0.294857, 0.220852, 0.560109, 0.741998],
                [1.01103, 1.41721, 1.08299, 0.304286, 0.235943, 0.54947, 0.741501],
                [1.00966, 1.41467, 1.08114, 0.299504, 0.234631, 0.53365, 0.755291],
                [1.00829, 1.41212, 1.07928, 0.294583, 0.23321, 0.517581, 0.768729],
                [1.00691, 1.40958, 1.07743, 0.289526, 0.23168, 0.501272, 0.781809],
                [1.00554, 1.40703, 1.07557, 0.284333, 0.230043, 0.484729, 0.794526],
                [1.00737, 1.41042, 1.07461, 0.298764, 0.233906, 0.470111, 0.796884],
                [1.0092, 1.4138, 1.07364, 0.313063, 0.237665, 0.455284, 0.798889],
                [1.01103, 1.41718, 1.07268, 0.327223, 0.24132, 0.440256, 0.80054],
                [1.01286, 1.42056, 1.07171, 0.341238, 0.244868, 0.425033, 0.801837],
                [1.01035, 1.41346, 1.07073, 0.350147, 0.253544, 0.41286, 0.801661],
                [1.00784, 1.40636, 1.06975, 0.358949, 0.262144, 0.400561, 0.801241],
                [1.00533, 1.39925, 1.06877, 0.367643, 0.270664, 0.388142, 0.800578],
                [1.00283, 1.39215, 1.06778, 0.376225, 0.279102, 0.375605, 0.799673],
                [1.00944, 1.39327, 1.07012, 0.381444, 0.293433, 0.36697, 0.796072],
                [1.01606, 1.39439, 1.07246, 0.386541, 0.307669, 0.358217, 0.792216],
                [1.02268, 1.39551, 1.07479, 0.391514, 0.321807, 0.34935, 0.788106],
                [1.0293, 1.39664, 1.07713, 0.396362, 0.335843, 0.340371, 0.783744],
                [1.02992, 1.38989, 1.07955, 0.397615, 0.351264, 0.342743, 0.77527],
                [1.03055, 1.38314, 1.08198, 0.398742, 0.366575, 0.345007, 0.766549],
                [1.03117, 1.37639, 1.08441, 0.399742, 0.381769, 0.347162, 0.757586],
                [1.0318, 1.36964, 1.08683, 0.400616, 0.396842, 0.349206, 0.748383],
                [1.03283, 1.36437, 1.08464, 0.412698, 0.40305, 0.35453, 0.735893],
                [1.03386, 1.3591, 1.08246, 0.424628, 0.409109, 0.359724, 0.723132],
                [1.03488, 1.35384, 1.08027, 0.436401, 0.415017, 0.364784, 0.710104],
                [1.03591, 1.34857, 1.07808, 0.448014, 0.420771, 0.36971, 0.696814],
                [1.03769, 1.34275, 1.07532, 0.452164, 0.435132, 0.367323, 0.6865],
                [1.03947, 1.33694, 1.07256, 0.456162, 0.449347, 0.364812, 0.675955],
                [1.04125, 1.33112, 1.06979, 0.460007, 0.463411, 0.362179, 0.665184],
                [1.04303, 1.3253, 1.06703, 0.463698, 0.47732, 0.359424, 0.65419],
                [1.04641, 1.3271, 1.06804, 0.485449, 0.475445, 0.340228, 0.650028],
                [1.0498, 1.32891, 1.06904, 0.506781, 0.47316, 0.320738, 0.645306],
                [1.05318, 1.33071, 1.07005, 0.527676, 0.470467, 0.300972, 0.640027],
                [1.056180015717243, 1.3097879278737143, 1.0714363512729903, 0.49508766425844575, 0.44117060762241045,
                 0.3650741365632882, 0.6534352106228737],
                [1.059180031434486, 1.2888658557474288, 1.0728227025459809, 0.45942522284228887, 0.40913489214128795,
                 0.42690945534576213, 0.6627861054557758],
                [1.0621800471517289, 1.2679437836211433, 1.0742090538189712, 0.4209101446679749, 0.37455879974061723,
                 0.4860940324097256, 0.6680216667954008],
                [1.0651800628689718, 1.2470217114948576, 1.0755954050919618, 0.37978157535111357, 0.33765701831133565,
                 0.5422603822718949, 0.6691093862943961],
                [1.0681800785862148, 1.226099639368572, 1.0769817563649522, 0.3362948880467047, 0.29865867627938275,
                 0.5950597600523154, 0.6660425101471993],
                [1.0711800943034577, 1.2051775672422864, 1.0783681076379426, 0.2907200978007825, 0.25780591991461116,
                 0.6441643268810782, 0.658840081025382],
                [1.0741801100207007, 1.1842554951160007, 1.0797544589109331, 0.24334018498846993, 0.21535240981004902,
                 0.6892691854966392, 0.647546819838984],
                [1.0771801257379436, 1.1633334229897152, 1.0811408101839235, 0.19444933824845462, 0.17156174586708584,
                 0.7300942733964164, 0.6322328480580034],
                [1.0801801414551866, 1.1424113508634295, 1.0825271614569139, 0.14435112782375673, 0.1267058305660747,
                 0.7663861017848287, 0.612993252318184],
                [1.0831801571724295, 1.121489278737144, 1.0839135127299044, 0.09335662065077863, 0.08106318068503916,
                 0.7979193295213944, 0.5899474940145264],
                [1.088090925162347, 1.0872416441364399, 1.0861828506587201, 0.0087786000250608, 0.0053950000154014315,
                 0.8387670057979981, 0.5443930015541114]]
        tj_10 = [[0.926, 1.452, 1.09, 0.00874246, 0.00536936, 0.838699, 0.544499],
                 [0.930777, 1.44889, 1.09045, 0.00856279, -0.0102008, 0.832425, 0.553978],
                 [0.935554, 1.44577, 1.09091, 0.00837995, -0.0257671, 0.825841, 0.563251],
                 [0.940331, 1.44266, 1.09136, 0.00819398, -0.0413239, 0.818951, 0.572315],
                 [0.945107, 1.43954, 1.09182, 0.00800498, -0.0568653, 0.811756, 0.581166],
                 [0.944414, 1.43943, 1.09293, -0.00682249, -0.0608764, 0.800947, 0.595593],
                 [0.943721, 1.43931, 1.09404, -0.0216461, -0.0648533, 0.789689, 0.609686],
                 [0.943028, 1.4392, 1.09515, -0.0364576, -0.0687939, 0.777987, 0.623437],
                 [0.942335, 1.43908, 1.09627, -0.0512487, -0.0726959, 0.76585, 0.636838],
                 [0.950478, 1.44207, 1.09389, -0.0608416, -0.0840539, 0.768413, 0.631486],
                 [0.958621, 1.44505, 1.09152, -0.0704189, -0.0953902, 0.770779, 0.625973],
                 [0.966764, 1.44804, 1.08914, -0.0799781, -0.106702, 0.772947, 0.620299],
                 [0.974907, 1.45103, 1.08677, -0.0895169, -0.117987, 0.774918, 0.614466],
                 [0.9741, 1.44458, 1.08739, -0.102934, -0.124996, 0.766748, 0.62119],
                 [0.973294, 1.43812, 1.088, -0.116315, -0.131963, 0.758318, 0.627703],
                 [0.972487, 1.43167, 1.08862, -0.129657, -0.138884, 0.749628, 0.634001],
                 [0.971681, 1.42522, 1.08924, -0.142955, -0.145759, 0.740683, 0.640083],
                 [0.970787, 1.42728, 1.08541, -0.161496, -0.153423, 0.736346, 0.638885],
                 [0.969893, 1.42933, 1.08158, -0.179969, -0.161021, 0.731698, 0.637417],
                 [0.968999, 1.43138, 1.07775, -0.198367, -0.168552, 0.726741, 0.63568],
                 [0.968106, 1.43344, 1.07392, -0.21668, -0.176012, 0.721476, 0.633673],
                 [0.965748, 1.43271, 1.07144, -0.221418, -0.195771, 0.721086, 0.626644],
                 [0.963389, 1.43198, 1.06897, -0.226054, -0.21544, 0.720362, 0.619325],
                 [0.961031, 1.43124, 1.06649, -0.230586, -0.235009, 0.719305, 0.61172],
                 [0.958673, 1.43051, 1.06402, -0.235011, -0.254469, 0.717915, 0.603832],
                 [0.957493, 1.42533, 1.06701, -0.242164, -0.265306, 0.705896, 0.610476],
                 [0.956314, 1.42014, 1.06999, -0.249231, -0.276049, 0.693624, 0.616901],
                 [0.955134, 1.41496, 1.07298, -0.256209, -0.286693, 0.681105, 0.623106],
                 [0.953954, 1.40977, 1.07597, -0.263095, -0.297234, 0.668342, 0.629088],
                 [0.959037, 1.40982, 1.07458, -0.27392, -0.309306, 0.668986, 0.617863],
                 [0.964119, 1.40988, 1.07319, -0.284639, -0.321257, 0.669371, 0.606397],
                 [0.969201, 1.40993, 1.07181, -0.295247, -0.333083, 0.669494, 0.594695],
                 [0.974283, 1.40999, 1.07042, -0.305739, -0.34478, 0.669357, 0.582762],
                 [0.978155, 1.4111, 1.07458, -0.311589, -0.353461, 0.675053, 0.567698],
                 [0.982027, 1.41222, 1.07874, -0.317325, -0.362013, 0.680499, 0.552425],
                 [0.985899, 1.41333, 1.0829, -0.322943, -0.37043, 0.685694, 0.536948],
                 [0.989771, 1.41445, 1.08706, -0.328442, -0.378711, 0.690636, 0.521273],
                 [0.987511, 1.42002, 1.08894, -0.325302, -0.395898, 0.686256, 0.516232],
                 [0.985252, 1.42558, 1.09082, -0.322048, -0.412947, 0.681636, 0.511011],
                 [0.982992, 1.43115, 1.0927, -0.318682, -0.429851, 0.676777, 0.505612],
                 [0.980732, 1.43672, 1.09458, -0.315204, -0.446605, 0.671681, 0.500035],
                 [0.98034, 1.42551, 1.09539, -0.304685, -0.455415, 0.671319, 0.499094],
                 [0.979947, 1.41431, 1.0962, -0.294108, -0.46414, 0.67083, 0.498058],
                 [0.979554, 1.4031, 1.09701, -0.283476, -0.472776, 0.670214, 0.496928],
                 [0.979162, 1.39189, 1.09782, -0.272791, -0.481323, 0.669471, 0.495703],
                 [0.983608, 1.38823, 1.09843, -0.268463, -0.496822, 0.667429, 0.485422],
                 [0.988053, 1.38456, 1.09904, -0.264036, -0.512137, 0.665141, 0.474961],
                 [0.992499, 1.38089, 1.09965, -0.259512, -0.527263, 0.662608, 0.464325],
                 [0.996945, 1.37723, 1.10027, -0.254892, -0.542195, 0.65983, 0.453518],
                 [0.997263, 1.37443, 1.09714, -0.254468, -0.557954, 0.646282, 0.45415],
                 [0.997581, 1.37164, 1.09402, -0.253933, -0.573472, 0.632456, 0.454585],
                 [0.997899, 1.36884, 1.09089, -0.253289, -0.588742, 0.618355, 0.454824],
                 [0.998217, 1.36605, 1.08777, -0.252535, -0.603757, 0.603988, 0.454866],
                 [0.999079, 1.35989, 1.08574, -0.260472, -0.614573, 0.591599, 0.452179],
                 [0.999941, 1.35373, 1.08371, -0.26832, -0.62518, 0.579008, 0.449338],
                 [1.0008, 1.34757, 1.08168, -0.276076, -0.635573, 0.56622, 0.446345],
                 [1.00166, 1.34141, 1.07966, -0.283739, -0.645751, 0.553239, 0.443199],
                 [0.990185, 1.3272, 1.08779, -0.281058, -0.656914, 0.56187, 0.416861],
                 [0.978705, 1.31298, 1.09593, -0.278124, -0.667485, 0.569995, 0.390149],
                 [0.9889472379083829, 1.291843222646147, 1.0950173357579254, 0.25643708005675775, 0.6170802340308545,
                  -0.6153540739508861, -0.4180805812429433],
                 [0.9991894758167658, 1.2707064452922938, 1.094104671515851, -0.233250328694377, -0.5630663322046205,
                  0.6571141209502241, 0.44356693040556505],
                 [1.0094317137251485, 1.2495696679384405, 1.0931920072737762, -0.2086993935302181, -0.5057592904543413,
                  0.6950309817695568, 0.46645904176003117],
                 [1.0196739516335314, 1.2284328905845874, 1.0922793430317015, -0.1829278627352259, -0.4454942737442902,
                  0.7288828965184112, 0.486623028903381],
                 [1.0299161895419142, 1.2072961132307343, 1.0913666787896268, -0.15608646323464992,
                  -0.38262374699946816, 0.7584718794789626, 0.5039409610923041],
                 [1.040158427450297, 1.1861593358768812, 1.0904540145475523, -0.12833217916917983, -0.3175154136853947,
                  0.7836248770418464, 0.5183115529708302],
                 [1.05040066535868, 1.165022558523028, 1.0895413503054776, -0.09982733376024058, -0.25055006526403706,
                  0.8041947798244803, 0.5296507569460882],
                 [1.0606429032670628, 1.1438857811691747, 1.088628686063403, -0.07073863994923181, -0.18211935410351052,
                  0.8200612830524299, 0.5378922547475987],
                 [1.0708851411754456, 1.1227490038153216, 1.0877160218213284, -0.04123622536312822,
                  -0.11262350286687363, 0.8311315901718149, 0.5429878452951475],
                 [1.088090925162347, 1.0872416441364399, 1.0861828506587201, 0.0087786000250608, 0.0053950000154014315,
                  0.8387670057979981, 0.5443930015541114]]
        tj_11 = [[0.926, 1.452, 1.09, 0.00873997, 0.00537314, 0.838919, 0.54416],
                 [0.925251, 1.44713, 1.08784, 0.0157771, -0.012904, 0.838179, 0.545015],
                 [0.924501, 1.44225, 1.08569, 0.0228081, -0.0311761, 0.837116, 0.54566],
                 [0.923752, 1.43738, 1.08353, 0.0298303, -0.0494362, 0.835731, 0.546095],
                 [0.923003, 1.43251, 1.08137, 0.0368411, -0.0676773, 0.834024, 0.54632],
                 [0.92335, 1.42488, 1.08233, 0.0283693, -0.0804164, 0.829022, 0.552676],
                 [0.923697, 1.41725, 1.08328, 0.019889, -0.0931315, 0.823772, 0.558866],
                 [0.924045, 1.40962, 1.08424, 0.0114027, -0.105819, 0.818275, 0.564888],
                 [0.924392, 1.402, 1.0852, 0.00291306, -0.118474, 0.812533, 0.570742],
                 [0.926105, 1.40344, 1.08463, -0.00302383, -0.139193, 0.814084, 0.563812],
                 [0.927819, 1.40487, 1.08407, -0.00895916, -0.15984, 0.815216, 0.556591],
                 [0.929533, 1.40631, 1.0835, -0.0148899, -0.180405, 0.815928, 0.549084],
                 [0.931246, 1.40775, 1.08294, -0.0208129, -0.200877, 0.81622, 0.541294],
                 [0.93468, 1.40807, 1.08089, -0.0255572, -0.220784, 0.811734, 0.540082],
                 [0.938113, 1.40839, 1.07883, -0.0302902, -0.240594, 0.806891, 0.538631],
                 [0.941547, 1.40872, 1.07678, -0.0350099, -0.260298, 0.801692, 0.536944],
                 [0.94498, 1.40904, 1.07472, -0.0397142, -0.279887, 0.796141, 0.53502],
                 [0.944635, 1.40365, 1.07326, -0.0416551, -0.297922, 0.789248, 0.535345],
                 [0.94429, 1.39826, 1.07181, -0.0435803, -0.315845, 0.782059, 0.535469],
                 [0.943944, 1.39287, 1.07035, -0.0454892, -0.333649, 0.774574, 0.535391],
                 [0.943599, 1.38749, 1.06889, -0.0473808, -0.351328, 0.766798, 0.535111],
                 [0.949426, 1.38069, 1.06953, -0.0340578, -0.356284, 0.761544, 0.540326],
                 [0.955254, 1.3739, 1.07016, -0.0207261, -0.36115, 0.756094, 0.545402],
                 [0.961081, 1.36711, 1.0708, -0.00738904, -0.365922, 0.750449, 0.550338],
                 [0.966909, 1.36032, 1.07144, 0.00594992, -0.370601, 0.744612, 0.555133],
                 [0.966004, 1.3631, 1.06973, 0.025054, -0.378486, 0.738659, 0.557229],
                 [0.965099, 1.36589, 1.06803, 0.0441465, -0.386194, 0.73236, 0.559065],
                 [0.964194, 1.36868, 1.06633, 0.0632183, -0.393723, 0.72572, 0.56064],
                 [0.963289, 1.37147, 1.06462, 0.0822606, -0.401067, 0.71874, 0.561953],
                 [0.964286, 1.36945, 1.06292, 0.0876002, -0.398049, 0.706068, 0.579096],
                 [0.965283, 1.36744, 1.06122, 0.0928967, -0.394834, 0.693048, 0.595954],
                 [0.966279, 1.36543, 1.05953, 0.0981474, -0.391426, 0.679687, 0.612518],
                 [0.967276, 1.36341, 1.05783, 0.10335, -0.387825, 0.665991, 0.628782],
                 [0.965701, 1.35727, 1.05493, 0.102427, -0.403561, 0.657262, 0.628215],
                 [0.964127, 1.35112, 1.05204, 0.101472, -0.419166, 0.648319, 0.627444],
                 [0.962552, 1.34498, 1.04914, 0.100483, -0.434634, 0.639166, 0.626469],
                 [0.960977, 1.33884, 1.04625, 0.0994615, -0.449961, 0.629805, 0.62529],
                 [0.958303, 1.34073, 1.04404, 0.118534, -0.45654, 0.624466, 0.622546],
                 [0.955629, 1.34263, 1.04184, 0.137554, -0.462917, 0.61885, 0.619525],
                 [0.952954, 1.34453, 1.03964, 0.156513, -0.469088, 0.612961, 0.61623],
                 [0.95028, 1.34642, 1.03743, 0.175402, -0.475051, 0.606799, 0.612662],
                 [0.945685, 1.33975, 1.03659, 0.175264, -0.489653, 0.602433, 0.605473],
                 [0.94109, 1.33307, 1.03575, 0.175076, -0.504115, 0.597895, 0.598113],
                 [0.936495, 1.32639, 1.03491, 0.174838, -0.518433, 0.593187, 0.590582],
                 [0.9319, 1.31972, 1.03407, 0.174551, -0.532605, 0.588311, 0.582884],
                 [0.929868, 1.31333, 1.03401, 0.164448, -0.545149, 0.579625, 0.582928],
                 [0.927836, 1.30695, 1.03395, 0.15429, -0.55751, 0.570745, 0.582776],
                 [0.925804, 1.30056, 1.03389, 0.144081, -0.569685, 0.561673, 0.582429],
                 [0.923772, 1.29418, 1.03382, 0.133823, -0.581668, 0.552413, 0.581887],
                 [0.922758, 1.29644, 1.03641, 0.131429, -0.594896, 0.555836, 0.565572],
                 [0.921744, 1.29871, 1.03899, 0.128976, -0.60785, 0.559004, 0.548997],
                 [0.920729, 1.30098, 1.04158, 0.126463, -0.620526, 0.561916, 0.532171],
                 [0.919715, 1.30325, 1.04416, 0.123892, -0.632917, 0.56457, 0.5151],
                 [0.918294, 1.29689, 1.04507, 0.124698, -0.646444, 0.559226, 0.503813],
                 [0.916873, 1.29054, 1.04597, 0.125461, -0.659751, 0.553692, 0.492354],
                 [0.915453, 1.28419, 1.04688, 0.126182, -0.672835, 0.54797, 0.480729],
                 [0.914032, 1.27784, 1.04778, 0.12686, -0.68569, 0.542062, 0.46894],
                 [0.920441, 1.28086, 1.04517, 0.135355, -0.691881, 0.543906, 0.455134],
                 [0.92685, 1.28388, 1.04257, 0.143809, -0.697861, 0.545584, 0.441189],
                 [0.933259, 1.2869, 1.03996, 0.152219, -0.703628, 0.547096, 0.42711],
                 [0.939669, 1.28992, 1.03735, 0.160583, -0.709181, 0.548442, 0.412901],
                 [0.936363, 1.29217, 1.04003, 0.148972, -0.717362, 0.551452, 0.398874],
                 [0.933057, 1.29441, 1.04272, 0.137301, -0.725251, 0.554237, 0.384684],
                 [0.929751, 1.29665, 1.0454, 0.125574, -0.732844, 0.556795, 0.370338],
                 [0.926444, 1.2989, 1.04808, 0.113795, -0.740138, 0.559128, 0.35584],
                 [0.925054, 1.29158, 1.04621, 0.124746, -0.747779, 0.551721, 0.347664],
                 [0.923664, 1.28426, 1.04434, 0.13566, -0.755194, 0.544148, 0.339382],
                 [0.922274, 1.27695, 1.04247, 0.146532, -0.762384, 0.536413, 0.330999],
                 [0.920884, 1.26963, 1.0406, 0.157361, -0.769344, 0.528516, 0.322517],
                 [0.924757, 1.26665, 1.03802, 0.166734, -0.77568, 0.525662, 0.306921],
                 [0.928631, 1.26368, 1.03545, 0.176045, -0.781721, 0.522607, 0.291209],
                 [0.932505, 1.2607, 1.03287, 0.185288, -0.787465, 0.519355, 0.275386],
                 [0.936378, 1.25773, 1.03029, 0.194461, -0.792911, 0.515905, 0.259458],
                 [0.93651, 1.24868, 1.02856, 0.187046, -0.800722, 0.516972, 0.237902],
                 [0.936642, 1.23964, 1.02682, 0.179521, -0.808068, 0.517739, 0.216206],
                 [0.936774, 1.23059, 1.02509, 0.171892, -0.814943, 0.518204, 0.194385],
                 [0.949172669483353, 1.2188442624797098, 1.030095851541727, -0.16316801908236916, 0.7690042706890777,
                  -0.5710041839184197, -0.23656468704850034],
                 [0.9615713389667059, 1.2070985249594195, 1.035101703083454, -0.15334207793043228, 0.7178720470677772,
                  -0.6199480790739802, -0.27714673086143066],
                 [0.9739700084500589, 1.195352787439129, 1.0401075546251808, 0.14248054710223512, -0.6618917028858793,
                  0.664705179325159, 0.3158570751567661],
                 [0.9863686779334119, 1.1836070499188387, 1.0451134061669078, 0.1306567795179015, -0.601441299190232,
                  0.7049732193604055, 0.35243429115084984],
                 [0.9987673474167648, 1.1718613123985484, 1.0501192577086347, 0.11795062652554206, -0.5369290853983333,
                  0.740480250559311, 0.386631356057888],
                 [1.0111660169001178, 1.1601155748782581, 1.0551251092503615, 0.10444789862832392, -0.46879074220198874,
                  0.7709864775868931, 0.41821732134859174],
                 [1.0235646863834706, 1.1483698373579676, 1.0601309607920886, 0.0902397859666208, -0.3974864392149395,
                  0.7962858778419339, 0.4469788724519623],
                 [1.0359633558668238, 1.1366240998376773, 1.0651368123338154, 0.07542224246899693, -0.3234977272355246,
                  0.8162075928224922, 0.47272176936683447],
                 [1.0483620253501766, 1.124878362317387, 1.0701426638755425, 0.06009533783113377, -0.24732428611235335,
                  0.8306170820120596, 0.4952721584540646],
                 [1.0607606948335295, 1.1131326247970967, 1.0751485154172693, 0.044362581699086974,
                  -0.16948055017615243, 0.8394170314936328, 0.5144777465502257],
                 [1.0731593643168824, 1.1013868872768064, 1.0801543669589961, 0.028330224620973208,
                  -0.09049223402779988, 0.8425480111554124, 0.530208829473484],
                 [1.088090925162347, 1.0872416441364399, 1.0861828506587201, 0.0087786000250608, 0.0053950000154014315,
                  0.8387670057979981, 0.5443930015541114]]
        tj_12 = [[0.925, 1.452, 1.09, 0.00874212, 0.00537019, 0.838737, 0.54444],
                 [0.926239, 1.44872, 1.09017, -0.00904459, -0.00327972, 0.843283, 0.537384],
                 [0.927479, 1.44543, 1.09035, -0.0268271, -0.0119281, 0.847439, 0.53008],
                 [0.928718, 1.44215, 1.09052, -0.0445973, -0.020571, 0.851204, 0.522532],
                 [0.929958, 1.43887, 1.0907, -0.0623468, -0.0292044, 0.854576, 0.514743],
                 [0.931187, 1.43333, 1.09011, -0.0621657, -0.0481351, 0.855845, 0.511221],
                 [0.932416, 1.42779, 1.08953, -0.0619614, -0.0670478, 0.856796, 0.50751],
                 [0.933645, 1.42226, 1.08894, -0.0617341, -0.0859356, 0.857428, 0.503609],
                 [0.934874, 1.41672, 1.08836, -0.0614838, -0.104791, 0.85774, 0.499521],
                 [0.938424, 1.42071, 1.08753, -0.0750804, -0.115944, 0.860035, 0.491183],
                 [0.941973, 1.42469, 1.08669, -0.0886482, -0.127052, 0.861999, 0.482656],
                 [0.945522, 1.42868, 1.08586, -0.102182, -0.138111, 0.863632, 0.473944],
                 [0.949072, 1.43267, 1.08503, -0.115676, -0.149117, 0.864934, 0.465051],
                 [0.955291, 1.42591, 1.08327, -0.121869, -0.159771, 0.867175, 0.455664],
                 [0.96151, 1.41916, 1.08151, -0.128032, -0.170386, 0.869203, 0.446165],
                 [0.96773, 1.41241, 1.07976, -0.134163, -0.180958, 0.871018, 0.436557],
                 [0.973949, 1.40565, 1.078, -0.140261, -0.191487, 0.87262, 0.426843],
                 [0.976933, 1.41029, 1.08139, -0.155526, -0.200975, 0.868149, 0.426307],
                 [0.979917, 1.41492, 1.08478, -0.170738, -0.210395, 0.86338, 0.425625],
                 [0.9829, 1.41956, 1.08816, -0.185891, -0.219742, 0.858315, 0.424798],
                 [0.985884, 1.4242, 1.09155, -0.20098, -0.229014, 0.852955, 0.423824],
                 [0.986197, 1.4271, 1.09273, -0.22003, -0.233669, 0.844579, 0.428569],
                 [0.98651, 1.43001, 1.09392, -0.238975, -0.238213, 0.8358, 0.433109],
                 [0.986824, 1.43291, 1.0951, -0.257807, -0.242643, 0.826622, 0.437442],
                 [0.987137, 1.43582, 1.09628, -0.276515, -0.246957, 0.81705, 0.441567],
                 [0.98774, 1.43158, 1.09636, -0.272065, -0.257282, 0.807359, 0.456024],
                 [0.988344, 1.42735, 1.09644, -0.267498, -0.267496, 0.797322, 0.470286],
                 [0.988947, 1.42311, 1.09651, -0.262816, -0.277595, 0.786943, 0.484345],
                 [0.989551, 1.41888, 1.09659, -0.258021, -0.287575, 0.776226, 0.498197],
                 [0.988265, 1.41659, 1.09393, -0.250703, -0.298497, 0.765891, 0.51133],
                 [0.98698, 1.41431, 1.09126, -0.243272, -0.309284, 0.755211, 0.524231],
                 [0.985695, 1.41203, 1.08859, -0.235731, -0.319931, 0.744189, 0.536896],
                 [0.98441, 1.40975, 1.08593, -0.228083, -0.330434, 0.732831, 0.549318],
                 [0.979541, 1.41154, 1.08469, -0.227187, -0.348022, 0.731384, 0.540689],
                 [0.974671, 1.41333, 1.08344, -0.226203, -0.365476, 0.729654, 0.531851],
                 [0.969802, 1.41512, 1.0822, -0.225131, -0.382788, 0.727641, 0.522807],
                 [0.964933, 1.41691, 1.08096, -0.223972, -0.399953, 0.725348, 0.513561],
                 [0.965038, 1.42134, 1.08483, -0.24001, -0.406847, 0.71788, 0.511389],
                 [0.965144, 1.42577, 1.08871, -0.255959, -0.413593, 0.710151, 0.50903],
                 [0.965249, 1.4302, 1.09259, -0.271815, -0.420188, 0.702162, 0.506485],
                 [0.965355, 1.43463, 1.09647, -0.287572, -0.42663, 0.693916, 0.503755],
                 [0.97579, 1.43783, 1.09078, -0.292312, -0.435797, 0.691178, 0.496898],
                 [0.986226, 1.44102, 1.08508, -0.297006, -0.444894, 0.688328, 0.48996],
                 [0.996661, 1.44422, 1.07939, -0.301651, -0.45392, 0.685368, 0.482943],
                 [1.0071, 1.44742, 1.0737, -0.306249, -0.462872, 0.682297, 0.475849],
                 [1.00646, 1.44699, 1.06883, -0.311107, -0.47048, 0.667551, 0.486043],
                 [1.00582, 1.44657, 1.06396, -0.315839, -0.477898, 0.652536, 0.496041],
                 [1.00518, 1.44615, 1.05909, -0.320445, -0.485123, 0.637257, 0.505839],
                 [1.00454, 1.44572, 1.05422, -0.324921, -0.492153, 0.621722, 0.515434],
                 [1.00379, 1.44984, 1.05455, -0.323115, -0.501474, 0.606373, 0.525768],
                 [1.00303, 1.45396, 1.05488, -0.32117, -0.510578, 0.590762, 0.535874],
                 [1.00227, 1.45808, 1.0552, -0.319085, -0.51946, 0.574895, 0.545748],
                 [1.00151, 1.4622, 1.05553, -0.316863, -0.528118, 0.55878, 0.555386],
                 [1.00438, 1.4611, 1.05931, -0.326927, -0.532706, 0.542895, 0.560899],
                 [1.00725, 1.46, 1.06309, -0.336858, -0.537078, 0.526791, 0.566185],
                 [1.01012, 1.45889, 1.06688, -0.346653, -0.541232, 0.510473, 0.571242],
                 [1.01298, 1.45779, 1.07066, -0.356308, -0.545167, 0.493948, 0.576067],
                 [1.01601, 1.45102, 1.0734, -0.360037, -0.557733, 0.486575, 0.567936],
                 [1.01903, 1.44425, 1.07614, -0.36366, -0.570137, 0.47906, 0.55964],
                 [1.02205, 1.43748, 1.07888, -0.367178, -0.582373, 0.471405, 0.55118],
                 [1.02508, 1.43071, 1.08162, -0.370588, -0.59444, 0.463612, 0.542559],
                 [1.02921, 1.42845, 1.08391, -0.38178, -0.599053, 0.448002, 0.542838],
                 [1.03335, 1.42618, 1.0862, -0.392824, -0.603433, 0.432216, 0.542906],
                 [1.03749, 1.42391, 1.08849, -0.403714, -0.607577, 0.416263, 0.542762],
                 [1.04162, 1.42164, 1.09077, -0.414446, -0.611484, 0.400146, 0.542406],
                 [1.04564, 1.4188, 1.08648, -0.412496, -0.624739, 0.388555, 0.53719],
                 [1.04966, 1.41596, 1.08219, -0.410406, -0.637781, 0.376831, 0.531791],
                 [1.05368, 1.41312, 1.07789, -0.408175, -0.650605, 0.364978, 0.526211],
                 [1.0577, 1.41028, 1.0736, -0.405805, -0.663207, 0.353001, 0.520451],
                 [1.05033, 1.41041, 1.07016, -0.412292, -0.668472, 0.329492, 0.524018],
                 [1.04297, 1.41055, 1.06673, -0.418518, -0.673312, 0.305774, 0.527253],
                 [1.0356, 1.41068, 1.06329, -0.424478, -0.677724, 0.281862, 0.530153],
                 [1.03783, 1.40429, 1.06467, -0.43853, -0.665614, 0.266963, 0.541645],
                 [1.04006, 1.3979, 1.06605, -0.452276, -0.65304, 0.251877, 0.55276],
                 [1.0423, 1.39151, 1.06743, -0.465706, -0.64001, 0.236615, 0.563488],
                 [1.0456715985020926, 1.369106636686709, 1.0688107775878462, -0.44289759826518477, -0.6092586878270672,
                  0.300103312964076, 0.5853063901121852],
                 [1.0490431970041854, 1.3467032733734183, 1.0701915551756924, -0.41744397494040064, -0.5748685564833363,
                  0.3617992378866543, 0.6036290102516276],
                 [1.052414795506278, 1.3242999100601274, 1.0715723327635385, -0.38949709499238955, -0.5370449228686928,
                  0.4213342540399349, 0.618346351316203],
                 [1.0557863940083705, 1.3018965467468366, 1.0729531103513845, -0.3592238760141983, -0.49601369520240646,
                  0.4783527782025776, 0.629370511430958],
                 [1.0591579925104633, 1.2794931834335457, 1.0743338879392308, -0.3268051300626654, -0.4520199396161735,
                  0.5325142573386089, 0.6366356468854262],
                 [1.0625295910125558, 1.257089820120255, 1.0757146655270768, -0.292434483727016, -0.4053264164540889,
                  0.5834952026095582, 0.6400983653966246],
                 [1.0659011895146486, 1.234686456806964, 1.077095443114923, -0.2563172216604379, -0.3562120108908694,
                  0.6309911214677175, 0.6397379852769955],
                 [1.0692727880167412, 1.212283093493673, 1.078476220702769, -0.21866906048184537, -0.30497006724172254,
                  0.6747183362907512, 0.635556658959378],
                 [1.0726443865188338, 1.1898797301803823, 1.0798569982906152, -0.17971486037091716,
                  -0.25190663691248927, 0.7144156786955529, 0.6275793601412284],
                 [1.0760159850209265, 1.1674763668670913, 1.0812377758784613, -0.13968728205162914,
                  -0.19733865045448043, 0.7498460494117604, 0.6158537346248775],
                 [1.0793875835230191, 1.1450730035538004, 1.0826185534663073, -0.09882539718568309,
                  -0.14159202464173062, 0.7807978343983183, 0.6004498157446999],
                 [1.0827591820251117, 1.1226696402405096, 1.0839993310541536, -0.05737326047549516,
                  -0.08499971587647343, 0.8070861687450797, 0.5814596060808684],
                 [1.088090925162347, 1.0872416441364399, 1.0861828506587201, 0.0087786000250608, 0.0053950000154014315,
                  0.8387670057979981, 0.5443930015541114]]
        tj_13 = [[0.926, 1.452, 1.09, 0.00874135, 0.00537103, 0.838792, 0.544355],
                 [0.930255, 1.45268, 1.08972, -0.00480762, 0.00108763, 0.846912, 0.53171],
                 [0.93451, 1.45335, 1.08944, -0.0183545, -0.00319623, 0.85467, 0.518838],
                 [0.938764, 1.45403, 1.08915, -0.0318936, -0.00747873, 0.862062, 0.505743],
                 [0.943019, 1.4547, 1.08887, -0.045419, -0.011758, 0.869085, 0.492432],
                 [0.947058, 1.44844, 1.09033, -0.0384188, -0.0166441, 0.876657, 0.479291],
                 [0.951097, 1.44217, 1.09178, -0.0314071, -0.0215251, 0.883962, 0.466005],
                 [0.955135, 1.4359, 1.09323, -0.0243858, -0.0263996, 0.891, 0.452578],
                 [0.959174, 1.42964, 1.09469, -0.0173571, -0.0312661, 0.897769, 0.439013],
                 [0.963029, 1.42868, 1.09166, -0.00639164, -0.036666, 0.904567, 0.424703],
                 [0.966884, 1.42773, 1.08864, 0.00457639, -0.0420513, 0.911004, 0.410223],
                 [0.970738, 1.42677, 1.08562, 0.0155426, -0.0474197, 0.917075, 0.395579],
                 [0.974593, 1.42582, 1.0826, 0.0265026, -0.0527691, 0.92278, 0.380777],
                 [0.968837, 1.42396, 1.08253, 0.0386512, -0.0625159, 0.925937, 0.370457],
                 [0.96308, 1.42209, 1.08246, 0.050786, -0.0722402, 0.928762, 0.360005],
                 [0.957324, 1.42023, 1.08239, 0.0629025, -0.0819386, 0.931253, 0.349424],
                 [0.951567, 1.41836, 1.08232, 0.0749965, -0.0916075, 0.93341, 0.338717],
                 [0.949708, 1.41628, 1.08339, 0.0627563, -0.105407, 0.936946, 0.327235],
                 [0.947849, 1.4142, 1.08445, 0.0504857, -0.119156, 0.940028, 0.315596],
                 [0.94599, 1.41212, 1.08552, 0.0381907, -0.132848, 0.942654, 0.303803],
                 [0.944131, 1.41004, 1.08658, 0.0258771, -0.146474, 0.944824, 0.291863],
                 [0.948098, 1.40685, 1.08417, 0.0324391, -0.161553, 0.945266, 0.28164],
                 [0.952065, 1.40367, 1.08176, 0.0389889, -0.176572, 0.945353, 0.271311],
                 [0.956032, 1.40048, 1.07935, 0.0455241, -0.191524, 0.945086, 0.260881],
                 [0.959999, 1.39729, 1.07694, 0.0520423, -0.206405, 0.944464, 0.250352],
                 [0.95734, 1.40065, 1.08041, 0.0369838, -0.213104, 0.946339, 0.240129],
                 [0.954681, 1.40402, 1.08388, 0.0219113, -0.219723, 0.947854, 0.229816],
                 [0.952022, 1.40738, 1.08735, 0.00683052, -0.226258, 0.949009, 0.219415],
                 [0.949363, 1.41074, 1.09082, -0.00825288, -0.232707, 0.949804, 0.20893],
                 [0.95245, 1.40543, 1.09191, -0.0263694, -0.236578, 0.949152, 0.206024],
                 [0.955538, 1.40013, 1.093, -0.0444766, -0.240366, 0.948166, 0.203044],
                 [0.958626, 1.39482, 1.09408, -0.0625681, -0.244069, 0.946846, 0.199994],
                 [0.961714, 1.38951, 1.09517, -0.0806377, -0.247686, 0.945193, 0.196873],
                 [0.964702, 1.39115, 1.09554, -0.0900022, -0.265815, 0.938802, 0.199731],
                 [0.967689, 1.39279, 1.0959, -0.0993249, -0.28382, 0.931974, 0.202496],
                 [0.970677, 1.39443, 1.09627, -0.108601, -0.301693, 0.924713, 0.205167],
                 [0.973664, 1.39607, 1.09663, -0.117827, -0.319426, 0.917021, 0.207743],
                 [0.977441, 1.38889, 1.09482, -0.134282, -0.319385, 0.915257, 0.205587],
                 [0.981217, 1.38171, 1.093, -0.150699, -0.319255, 0.913239, 0.203373],
                 [0.984993, 1.37454, 1.09118, -0.167074, -0.319037, 0.910966, 0.201103],
                 [0.988769, 1.36736, 1.08936, -0.183403, -0.318729, 0.90844, 0.198777],
                 [0.989266, 1.36607, 1.08981, -0.202407, -0.321088, 0.906435, 0.185229],
                 [0.989762, 1.36477, 1.09026, -0.221298, -0.323268, 0.903928, 0.171579],
                 [0.990258, 1.36348, 1.09071, -0.240068, -0.325269, 0.900919, 0.157834],
                 [0.990755, 1.36218, 1.09116, -0.258704, -0.32709, 0.897412, 0.144002],
                 [0.993965, 1.36687, 1.08983, -0.266191, -0.341936, 0.888938, 0.148364],
                 [0.997176, 1.37156, 1.0885, -0.273581, -0.356656, 0.880137, 0.152672],
                 [1.00039, 1.37624, 1.08717, -0.280871, -0.371245, 0.871013, 0.156924],
                 [1.0036, 1.38093, 1.08584, -0.288057, -0.385697, 0.861569, 0.161118],
                 [1.00884, 1.37301, 1.08629, -0.29732, -0.394808, 0.85369, 0.164137],
                 [1.01409, 1.3651, 1.08675, -0.306511, -0.403824, 0.845606, 0.167116],
                 [1.01934, 1.35718, 1.0872, -0.315628, -0.412743, 0.837319, 0.170055],
                 [1.02458, 1.34926, 1.08766, -0.32467, -0.421562, 0.828831, 0.172954],
                 [1.02823, 1.34684, 1.08479, -0.341985, -0.425321, 0.820353, 0.170791],
                 [1.03188, 1.34441, 1.08192, -0.359167, -0.428914, 0.811553, 0.168561],
                 [1.03553, 1.34199, 1.07905, -0.376208, -0.43234, 0.802437, 0.166265],
                 [1.03918, 1.33956, 1.07618, -0.393102, -0.435597, 0.793008, 0.163905],
                 [1.04093, 1.33773, 1.07408, -0.409932, -0.428623, 0.786203, 0.173556],
                 [1.04268, 1.33591, 1.07197, -0.42657, -0.421448, 0.779028, 0.183126],
                 [1.04442, 1.33408, 1.06987, -0.443006, -0.414074, 0.771485, 0.19261],
                 [1.04617, 1.33225, 1.06777, -0.459234, -0.406505, 0.763579, 0.202003],
                 [1.04201, 1.32434, 1.0701, -0.465052, -0.415046, 0.753641, 0.20854],
                 [1.03784, 1.31642, 1.07243, -0.470754, -0.423483, 0.743517, 0.215025],
                 [1.03368, 1.30851, 1.07476, -0.47634, -0.431816, 0.733207, 0.221457],
                 [1.02951, 1.30059, 1.07708, -0.481807, -0.440041, 0.722716, 0.227833],
                 [1.02512, 1.29982, 1.07478, -0.498255, -0.441004, 0.71216, 0.223796],
                 [1.02072, 1.29905, 1.07248, -0.514505, -0.441791, 0.701321, 0.21967],
                 [1.01633, 1.29827, 1.07018, -0.530549, -0.442401, 0.690201, 0.215456],
                 [1.01194, 1.2975, 1.06788, -0.546381, -0.442835, 0.678806, 0.211156],
                 [1.01357, 1.31061, 1.07859, -0.566847, -0.423999, 0.679336, 0.193423],
                 [1.0152, 1.32372, 1.0893, -0.586696, -0.404702, 0.679126, 0.175479],
                 [1.0218243588822036, 1.3022287444887737, 1.0890167121163563, -0.5443540127362463, -0.37556246288232953,
                  0.717389483215993, 0.21905221903189248],
                 [1.0284487177644073, 1.2807374889775476, 1.088733424232713, -0.49874362395899735, -0.34416797969744223,
                  0.7513455854290746, 0.2613101807574065],
                 [1.0350730766466107, 1.2592462334663213, 1.0884501363490693, -0.450138442636695, -0.3107068802184714,
                  0.7807901028320272, 0.3019990602056036],
                 [1.0416974355288142, 1.2377549779550951, 1.0881668484654257, -0.39883032700608095, -0.2753800874175711,
                  0.8055462307160287, 0.34087453394660044],
                 [1.048321794411018, 1.2162637224438688, 1.0878835605817823, -0.34512736554149354, -0.23839972714585714,
                  0.8254653166198348, 0.37770316745942845],
                 [1.0549461532932214, 1.1947724669326425, 1.0876002726981386, -0.28935202698421597,
                  -0.19998785438696182, 0.8404277529392512, 0.4122638168300801],
                 [1.0615705121754249, 1.1732812114214164, 1.087316984814495, -0.23183922402302337, -0.16037511988939399,
                  0.8503436951317258, 0.4443489566484742],
                 [1.0681948710576286, 1.15178995591019, 1.0870336969308516, -0.1729343022528847, -0.11979938518413556,
                  0.8551536012034796, 0.47376592613075436],
                 [1.074819229939832, 1.130298700398964, 1.086750409047208, -0.11299096648747552, -0.07850429430384759,
                  0.8548285892397285, 0.5003380859843637],
                 [1.0814435888220355, 1.1088074448877376, 1.0864671211635644, -0.05236915687731019,
                  -0.03673781078004167, 0.8493706108311487, 0.5239058790692952],
                 [1.088090925162347, 1.0872416441364399, 1.0861828506587201, 0.0087786000250608, 0.0053950000154014315,
                  0.8387670057979981, 0.5443930015541114]]
        tj_14 = [[0.926, 1.452, 1.09, 0.00874018, 0.00537304, 0.838917, 0.544162],
                 [0.928949, 1.4492, 1.08885, 0.00240784, -0.00279536, 0.829016, 0.559213],
                 [0.931898, 1.44639, 1.08769, -0.00392555, -0.0109625, 0.818757, 0.574022],
                 [0.934846, 1.44359, 1.08654, -0.0102572, -0.019125, 0.808145, 0.588584],
                 [0.937795, 1.44078, 1.08539, -0.0165845, -0.0272792, 0.797184, 0.602892],
                 [0.936902, 1.44211, 1.08751, -0.00119612, -0.0190589, 0.78895, 0.614161],
                 [0.936008, 1.44344, 1.08963, 0.0141929, -0.010829, 0.780321, 0.625124],
                 [0.935115, 1.44476, 1.09175, 0.0295748, -0.00259367, 0.771304, 0.635774],
                 [0.934221, 1.44609, 1.09387, 0.0449419, 0.00564292, 0.761901, 0.646107],
                 [0.93602, 1.43951, 1.09056, 0.054414, -0.0023094, 0.753573, 0.655104],
                 [0.937819, 1.43292, 1.08725, 0.0638697, -0.010261, 0.745017, 0.663901],
                 [0.939617, 1.42634, 1.08394, 0.073306, -0.0182095, 0.736235, 0.672498],
                 [0.941416, 1.41975, 1.08063, 0.0827201, -0.0261525, 0.727229, 0.68089],
                 [0.942556, 1.41426, 1.08088, 0.0916824, -0.0400028, 0.733181, 0.672637],
                 [0.943696, 1.40877, 1.08113, 0.10061, -0.053838, 0.738856, 0.664131],
                 [0.944836, 1.40327, 1.08138, 0.1095, -0.0676531, 0.744255, 0.655376],
                 [0.945977, 1.39778, 1.08163, 0.118349, -0.0814427, 0.749373, 0.646375],
                 [0.952439, 1.39874, 1.07957, 0.120592, -0.0983591, 0.744131, 0.649656],
                 [0.958902, 1.3997, 1.07752, 0.122795, -0.115243, 0.738643, 0.652723],
                 [0.965364, 1.40066, 1.07546, 0.124957, -0.132089, 0.732912, 0.655575],
                 [0.971827, 1.40162, 1.07341, 0.127079, -0.148892, 0.726939, 0.658211],
                 [0.976482, 1.39767, 1.07329, 0.116808, -0.163627, 0.722265, 0.661752],
                 [0.981138, 1.39372, 1.07318, 0.106496, -0.178304, 0.717333, 0.665056],
                 [0.985794, 1.38977, 1.07307, 0.096146, -0.192918, 0.712145, 0.668123],
                 [0.990449, 1.38582, 1.07296, 0.0857616, -0.207462, 0.706703, 0.670951],
                 [0.986487, 1.3898, 1.07201, 0.0770479, -0.220348, 0.6974, 0.677602],
                 [0.982525, 1.39377, 1.07106, 0.0683056, -0.233152, 0.687836, 0.684],
                 [0.978563, 1.39775, 1.0701, 0.0595378, -0.245869, 0.678016, 0.690143],
                 [0.9746, 1.40173, 1.06915, 0.0507478, -0.258494, 0.667944, 0.696029],
                 [0.973952, 1.39983, 1.06811, 0.0477944, -0.277321, 0.656223, 0.700129],
                 [0.973303, 1.39794, 1.06707, 0.0448164, -0.296004, 0.644163, 0.703866],
                 [0.972655, 1.39605, 1.06603, 0.0418152, -0.314534, 0.631769, 0.707239],
                 [0.972006, 1.39416, 1.06499, 0.0387923, -0.332902, 0.619049, 0.710246],
                 [0.968448, 1.38732, 1.06491, 0.0359522, -0.346266, 0.623049, 0.700441],
                 [0.96489, 1.38047, 1.06484, 0.0331013, -0.359526, 0.626862, 0.690427],
                 [0.961331, 1.37363, 1.06476, 0.0302405, -0.37268, 0.630488, 0.680206],
                 [0.957773, 1.36678, 1.06469, 0.0273706, -0.385721, 0.633926, 0.669782],
                 [0.961003, 1.36126, 1.06394, 0.0123062, -0.395545, 0.633082, 0.665282],
                 [0.964232, 1.35574, 1.0632, -0.00276248, -0.405231, 0.632021, 0.660552],
                 [0.967462, 1.35021, 1.06245, -0.0178302, -0.414779, 0.630742, 0.655595],
                 [0.970691, 1.34469, 1.06171, -0.0328918, -0.424183, 0.629246, 0.650413],
                 [0.976479, 1.34044, 1.06592, -0.0392797, -0.434772, 0.618461, 0.653404],
                 [0.982266, 1.3362, 1.07013, -0.0456566, -0.44524, 0.607504, 0.656213],
                 [0.988053, 1.33196, 1.07434, -0.0520208, -0.455584, 0.596378, 0.65884],
                 [0.99384, 1.32771, 1.07856, -0.0583705, -0.465802, 0.585087, 0.661283],
                 [0.993093, 1.32223, 1.07565, -0.0531078, -0.481442, 0.582492, 0.652761],
                 [0.992345, 1.31675, 1.07274, -0.0478264, -0.496912, 0.579692, 0.644009],
                 [0.991597, 1.31128, 1.06984, -0.0425282, -0.512208, 0.576689, 0.635031],
                 [0.99085, 1.3058, 1.06693, -0.037215, -0.527324, 0.573482, 0.62583],
                 [0.992772, 1.29995, 1.06716, -0.0427376, -0.537256, 0.559059, 0.630065],
                 [0.994694, 1.29411, 1.0674, -0.0482451, -0.546998, 0.544437, 0.634077],
                 [0.996616, 1.28827, 1.06764, -0.0537354, -0.556546, 0.529622, 0.637863],
                 [0.998538, 1.28243, 1.06787, -0.0592067, -0.565896, 0.514619, 0.641423],
                 [1.00656, 1.27477, 1.06458, -0.0897159, -0.568422, 0.493876, 0.651869],
                 [1.01458, 1.26711, 1.06129, -0.120093, -0.570109, 0.472403, 0.661354],
                 [1.0226105815855506, 1.2474605736972952, 1.064009379026591, -0.10791398689664478, -0.5166150023452838,
                  0.5285468257529803, 0.6649073347243974],
                 [1.0306411631711012, 1.2278111473945905, 1.066728758053182, -0.0950686568276661, -0.4599311547863365,
                  0.5814271112252952, 0.6643551743443402],
                 [1.038671744756652, 1.2081617210918858, 1.0694481370797728, -0.08163630722164061, -0.400407374671642,
                  0.6307172642076077, 0.6597008263464947],
                 [1.0467023263422026, 1.188512294789181, 1.0721675161063637, -0.06769987868076373, -0.3384112029238575,
                  0.6761129332580217, 0.6509730298894661],
                 [1.0547329079277532, 1.1688628684864766, 1.0748868951329547, -0.0533454243386667, -0.2743254467188959,
                  0.7173338141677778, 0.6382256764155717],
                 [1.0627634895133038, 1.1492134421837719, 1.0776062741595456, -0.038661578508946075,
                  -0.20854581577007447, 0.754125380753002, 0.621537476887863],
                 [1.0707940710988546, 1.1295640158810671, 1.0803256531861363, -0.0237390093957046, -0.14147847894308643,
                  0.7862604564751308, 0.6010114757742819],
                 [1.0788246526844052, 1.1099145895783624, 1.0830450322127272, -0.008669859245448223,
                  -0.07353755628894411, 0.8135406171857343, 0.5767744147799633],
                 [1.088090925162347, 1.0872416441364399, 1.0861828506587201, 0.0087786000250608, 0.0053950000154014315,
                  0.8387670057979981, 0.5443930015541114]]
        tj_15 = [[0.926, 1.452, 1.09, 0.00873912, 0.00537498, 0.839042, 0.54397],
                 [0.930779, 1.44625, 1.091, 0.0158633, 0.00873432, 0.847318, 0.530777],
                 [0.935558, 1.44049, 1.09201, 0.0229826, 0.012091, 0.855336, 0.517422],
                 [0.940337, 1.43474, 1.09301, 0.0300949, 0.015444, 0.863094, 0.50391],
                 [0.945116, 1.42898, 1.09402, 0.037198, 0.0187923, 0.870588, 0.490244],
                 [0.944998, 1.42902, 1.09135, 0.0392266, 0.0158671, 0.88117, 0.470902],
                 [0.944881, 1.42905, 1.08869, 0.0412356, 0.012934, 0.891312, 0.451326],
                 [0.944763, 1.42908, 1.08603, 0.0432241, 0.00999442, 0.90101, 0.431524],
                 [0.944646, 1.42911, 1.08336, 0.045191, 0.00704987, 0.910258, 0.411507],
                 [0.948939, 1.42832, 1.08334, 0.0412589, -0.0047006, 0.917098, 0.396494],
                 [0.953233, 1.42752, 1.08332, 0.0373093, -0.0164491, 0.923547, 0.381312],
                 [0.957527, 1.42673, 1.0833, 0.0333438, -0.0281905, 0.929603, 0.365967],
                 [0.961821, 1.42593, 1.08329, 0.0293641, -0.03992, 0.935263, 0.350467],
                 [0.957618, 1.42489, 1.08607, 0.0265955, -0.0439315, 0.941755, 0.332355],
                 [0.953416, 1.42384, 1.08886, 0.0238164, -0.0479257, 0.947876, 0.314112],
                 [0.949213, 1.4228, 1.09165, 0.0210279, -0.051901, 0.953624, 0.295745],
                 [0.945011, 1.42175, 1.09444, 0.0182312, -0.0558558, 0.958996, 0.277261],
                 [0.944504, 1.41736, 1.09123, 0.0195031, -0.0695266, 0.961904, 0.263677],
                 [0.943998, 1.41296, 1.08802, 0.0207676, -0.0831709, 0.964446, 0.249992],
                 [0.943491, 1.40857, 1.0848, 0.0220242, -0.0967834, 0.966619, 0.236212],
                 [0.942984, 1.40418, 1.08159, 0.0232723, -0.110359, 0.968423, 0.222341],
                 [0.943481, 1.40474, 1.08394, 0.00917768, -0.126753, 0.968088, 0.215997],
                 [0.943979, 1.40531, 1.08628, -0.00492161, -0.143082, 0.967262, 0.209542],
                 [0.944476, 1.40588, 1.08863, -0.0190184, -0.159339, 0.965944, 0.202982],
                 [0.944973, 1.40645, 1.09098, -0.0331055, -0.175515, 0.964136, 0.196318],
                 [0.950757, 1.40603, 1.09098, -0.0454349, -0.179883, 0.965573, 0.182336],
                 [0.956541, 1.40561, 1.09098, -0.0577476, -0.184185, 0.966654, 0.168286],
                 [0.962326, 1.4052, 1.09098, -0.070039, -0.188418, 0.967379, 0.154175],
                 [0.96811, 1.40478, 1.09098, -0.0823045, -0.192583, 0.967748, 0.140007],
                 [0.97226, 1.40747, 1.09154, -0.0772917, -0.211409, 0.963859, 0.142507],
                 [0.97641, 1.41016, 1.0921, -0.0722479, -0.230151, 0.959583, 0.14495],
                 [0.98056, 1.41285, 1.09267, -0.0671751, -0.2488, 0.954923, 0.147335],
                 [0.984711, 1.41553, 1.09323, -0.0620755, -0.26735, 0.94988, 0.149661],
                 [0.982506, 1.4137, 1.09159, -0.0699095, -0.286271, 0.944549, 0.144874],
                 [0.980301, 1.41187, 1.08995, -0.0777106, -0.305058, 0.938773, 0.140019],
                 [0.978096, 1.41004, 1.0883, -0.0854752, -0.323701, 0.932556, 0.135099],
                 [0.975892, 1.40821, 1.08666, -0.0931995, -0.342192, 0.925899, 0.130114],
                 [0.975577, 1.40428, 1.08564, -0.0900617, -0.361532, 0.919285, 0.12688],
                 [0.975262, 1.40036, 1.08462, -0.0868845, -0.380713, 0.912269, 0.12359],
                 [0.974947, 1.39643, 1.0836, -0.0836691, -0.399728, 0.904852, 0.120246],
                 [0.974632, 1.3925, 1.08257, -0.0804172, -0.418568, 0.89704, 0.116849],
                 [0.970851, 1.39475, 1.08581, -0.0960058, -0.428129, 0.890639, 0.119376],
                 [0.967071, 1.397, 1.08905, -0.111558, -0.437526, 0.883898, 0.121858],
                 [0.96329, 1.39924, 1.09229, -0.127067, -0.446757, 0.87682, 0.124292],
                 [0.95951, 1.40149, 1.09553, -0.142528, -0.455817, 0.869407, 0.12668],
                 [0.960969, 1.40605, 1.09458, -0.156576, -0.467365, 0.860798, 0.126808],
                 [0.962428, 1.41061, 1.09364, -0.170561, -0.478724, 0.851841, 0.126885],
                 [0.963886, 1.41516, 1.09269, -0.184477, -0.489889, 0.842539, 0.12691],
                 [0.965345, 1.41972, 1.09175, -0.198318, -0.500855, 0.832895, 0.126884],
                 [0.966015, 1.41297, 1.09131, -0.215356, -0.498185, 0.831088, 0.12135],
                 [0.966685, 1.40622, 1.09086, -0.232322, -0.49535, 0.829006, 0.115776],
                 [0.967355, 1.39947, 1.09042, -0.249212, -0.492351, 0.826649, 0.110164],
                 [0.968025, 1.39272, 1.08997, -0.266019, -0.489188, 0.824018, 0.104515],
                 [0.973908, 1.38663, 1.08665, -0.278558, -0.489757, 0.81846, 0.112543],
                 [0.97979, 1.38054, 1.08333, -0.291028, -0.490203, 0.812696, 0.120542],
                 [0.985673, 1.37444, 1.08001, -0.303423, -0.490524, 0.806725, 0.128511],
                 [0.991555, 1.36835, 1.07669, -0.315742, -0.490721, 0.800551, 0.136447],
                 [0.988539, 1.36247, 1.07874, -0.330938, -0.491837, 0.795038, 0.128417],
                 [0.985524, 1.3566, 1.08079, -0.346026, -0.492793, 0.789265, 0.120345],
                 [0.982508, 1.35072, 1.08284, -0.361, -0.493587, 0.783233, 0.112233],
                 [0.979492, 1.34484, 1.08488, -0.375857, -0.494219, 0.776946, 0.104085],
                 [0.975781, 1.33789, 1.08741, -0.382771, -0.50462, 0.766352, 0.107465],
                 [0.972071, 1.33093, 1.08993, -0.389578, -0.51488, 0.755545, 0.110814],
                 [0.96836, 1.32398, 1.09246, -0.396276, -0.524996, 0.744525, 0.114132],
                 [0.964649, 1.31702, 1.09498, -0.402863, -0.534965, 0.733298, 0.117419],
                 [0.972079, 1.30837, 1.0921, -0.410521, -0.537232, 0.726083, 0.125126],
                 [0.97951, 1.29972, 1.08922, -0.418108, -0.539405, 0.718741, 0.132811],
                 [0.98694, 1.29107, 1.08633, -0.425621, -0.541483, 0.711274, 0.140473],
                 [0.994371, 1.28242, 1.08345, -0.433059, -0.543467, 0.703681, 0.14811],
                 [1.00507, 1.27593, 1.08316, -0.443492, -0.536986, 0.698587, 0.164125],
                 [1.01577, 1.26944, 1.08287, -0.453732, -0.530272, 0.693191, 0.180068],
                 [1.02647, 1.26294, 1.08258, -0.463776, -0.523329, 0.687495, 0.195933],
                 [1.0325462126445801, 1.245615031757192, 1.0829352638437573, -0.42652618305740175, -0.4817954079414886,
                  0.7265844155207849, 0.2408810643767956],
                 [1.0386224252891603, 1.228290063514384, 1.0832905276875144, -0.38643544331369023, -0.43705276641232327,
                  0.7608343111358847, 0.2842246972391224],
                 [1.0446986379337404, 1.210965095271576, 1.0836457915312716, -0.34377069679079875, -0.38939896131607404,
                  0.7900163675851377, 0.3256751385990589],
                 [1.0507748505783205, 1.193640127028768, 1.0840010553750288, -0.2988161289812384, -0.33915140975175145,
                  0.8139362061891796, 0.3649562913260066],
                 [1.0568510632229007, 1.1763151587859602, 1.084356319218786, -0.2518711776418171, -0.28664480549706567,
                  0.8324344993609865, 0.40180650769804294],
                 [1.062927275867481, 1.158990190543152, 1.0847115830625431, -0.20324853826951067, -0.23222888964756513,
                  0.84538803187056, 0.4359803322105108],
                 [1.0690034885120612, 1.141665222300344, 1.0850668469063003, -0.15327208126888822, -0.17626612102713732,
                  0.8527105215689126, 0.4672501365299138],
                 [1.0750797011566413, 1.1243402540575362, 1.0854221107500575, -0.10227469468462802, -0.119129261887028,
                  0.8543531941045965, 0.49540763570285407],
                 [1.0811559138012214, 1.1070152858147282, 1.0857773745938146, -0.050596066868457586,
                  -0.06119889497478847, 0.8503051078046471, 0.5202652755207049],
                 [1.088090925162347, 1.0872416441364399, 1.0861828506587201, 0.0087786000250608, 0.0053950000154014315,
                  0.8387670057979981, 0.5443930015541114]]
        tj_16 = [[0.926, 1.453, 1.09, 0.00873916, 0.00537446, 0.839009, 0.544021],
                 [0.930408, 1.4552, 1.08818, -0.00108291, -0.00952789, 0.83439, 0.551092],
                 [0.934816, 1.45741, 1.08635, -0.0109046, -0.0244265, 0.829445, 0.557947],
                 [0.939224, 1.45961, 1.08453, -0.020722, -0.0393156, 0.824178, 0.564585],
                 [0.943632, 1.46181, 1.08271, -0.0305313, -0.0541894, 0.818589, 0.571003],
                 [0.958068, 1.45016, 1.0835, -0.0350934, -0.0586761, 0.818557, 0.570342],
                 [0.972505, 1.43852, 1.0843, -0.0396542, -0.0631604, 0.818491, 0.569659],
                 [0.986941, 1.42687, 1.08509, -0.0442132, -0.0676421, 0.818391, 0.568951],
                 [1.00138, 1.41522, 1.08589, -0.0487705, -0.0721209, 0.818258, 0.56822],
                 [1.0049, 1.41579, 1.08817, -0.0654858, -0.0840705, 0.815373, 0.569044],
                 [1.00842, 1.41636, 1.09045, -0.0821728, -0.0959837, 0.812136, 0.569622],
                 [1.01195, 1.41694, 1.09273, -0.0988245, -0.107856, 0.808549, 0.569955],
                 [1.01547, 1.41751, 1.09501, -0.115433, -0.119681, 0.804614, 0.570042],
                 [1.02123, 1.42038, 1.09285, -0.105899, -0.13477, 0.805916, 0.566677],
                 [1.02699, 1.42326, 1.09069, -0.0963293, -0.149815, 0.80695, 0.563123],
                 [1.03275, 1.42613, 1.08854, -0.0867277, -0.16481, 0.807717, 0.559383],
                 [1.03851, 1.42901, 1.08638, -0.0770973, -0.17975, 0.808216, 0.555457],
                 [1.03927, 1.42257, 1.08431, -0.0739024, -0.192483, 0.813381, 0.543966],
                 [1.04004, 1.41613, 1.08224, -0.0706831, -0.205152, 0.818277, 0.532296],
                 [1.0408, 1.40969, 1.08017, -0.0674403, -0.217753, 0.822902, 0.520449],
                 [1.04156, 1.40326, 1.07809, -0.0641753, -0.230283, 0.827255, 0.50843],
                 [1.0454, 1.40321, 1.07697, -0.082816, -0.233621, 0.829931, 0.499777],
                 [1.04925, 1.40317, 1.07585, -0.10142, -0.236857, 0.832242, 0.490903],
                 [1.0531, 1.40312, 1.07473, -0.11998, -0.239988, 0.834187, 0.481812],
                 [1.05694, 1.40308, 1.0736, -0.138486, -0.243013, 0.835764, 0.47251],
                 [1.06064, 1.40073, 1.07083, -0.150607, -0.257437, 0.829649, 0.471939],
                 [1.06434, 1.39838, 1.06806, -0.162669, -0.271759, 0.823209, 0.471183],
                 [1.06804, 1.39604, 1.06529, -0.174667, -0.285975, 0.816445, 0.470242],
                 [1.07174, 1.39369, 1.06252, -0.186597, -0.300078, 0.809361, 0.469116],
                 [1.06395, 1.39724, 1.06615, -0.190407, -0.314148, 0.803592, 0.468291],
                 [1.05616, 1.4008, 1.06978, -0.194171, -0.32814, 0.797625, 0.467349],
                 [1.04837, 1.40436, 1.07342, -0.197886, -0.342052, 0.791462, 0.466293],
                 [1.04058, 1.40791, 1.07705, -0.201553, -0.35588, 0.785104, 0.465122],
                 [1.03842, 1.40324, 1.07619, -0.201047, -0.372781, 0.775381, 0.4684],
                 [1.03626, 1.39856, 1.07533, -0.200462, -0.389537, 0.765355, 0.471495],
                 [1.03411, 1.39389, 1.07447, -0.199798, -0.40614, 0.75503, 0.474406],
                 [1.03195, 1.38921, 1.07361, -0.199057, -0.422584, 0.744409, 0.477131],
                 [1.02898, 1.38571, 1.07574, -0.200285, -0.439249, 0.733588, 0.478326],
                 [1.02601, 1.38222, 1.07787, -0.201433, -0.455739, 0.722474, 0.479331],
                 [1.02305, 1.37872, 1.08, -0.202501, -0.472047, 0.711074, 0.480145],
                 [1.02008, 1.37523, 1.08213, -0.203489, -0.488168, 0.69939, 0.480768],
                 [1.01655, 1.37263, 1.07803, -0.194781, -0.49799, 0.688802, 0.489509],
                 [1.01302, 1.37003, 1.07393, -0.186003, -0.507631, 0.677965, 0.498073],
                 [1.00949, 1.36742, 1.06982, -0.177158, -0.51709, 0.666883, 0.506458],
                 [1.00595, 1.36482, 1.06572, -0.168249, -0.526362, 0.655561, 0.51466],
                 [1.00472, 1.35951, 1.06551, -0.151447, -0.535151, 0.65618, 0.510005],
                 [1.00348, 1.35419, 1.0653, -0.134588, -0.543736, 0.656548, 0.505156],
                 [1.00224, 1.34887, 1.0651, -0.117677, -0.552114, 0.656665, 0.500113],
                 [1.001, 1.34355, 1.06489, -0.100722, -0.560281, 0.656532, 0.49488],
                 [1.00501, 1.33631, 1.06203, -0.0963821, -0.565554, 0.662618, 0.481453],
                 [1.00902, 1.32907, 1.05918, -0.0920173, -0.570678, 0.668529, 0.467898],
                 [1.01303, 1.32184, 1.05633, -0.0876282, -0.575651, 0.674264, 0.45422],
                 [1.01704, 1.3146, 1.05348, -0.0832159, -0.580472, 0.679821, 0.440422],
                 [1.02287, 1.30978, 1.05185, -0.0729633, -0.587763, 0.682139, 0.428832],
                 [1.0287, 1.30496, 1.05022, -0.062689, -0.594879, 0.684254, 0.417115],
                 [1.03453, 1.30014, 1.04858, -0.052396, -0.601818, 0.686165, 0.405274],
                 [1.04036, 1.29533, 1.04695, -0.0420874, -0.608578, 0.687872, 0.393311],
                 [1.04024, 1.29277, 1.05113, -0.0236428, -0.614897, 0.682994, 0.393524],
                 [1.04011, 1.29022, 1.05531, -0.00518865, -0.620969, 0.67784, 0.393578],
                 [1.03999, 1.28766, 1.05949, 0.0132676, -0.626789, 0.672412, 0.393473],
                 [1.03986, 1.28511, 1.06367, 0.0317185, -0.632356, 0.666713, 0.393209],
                 [1.03451, 1.27571, 1.06497, 0.0186313, -0.660545, 0.648266, 0.378264],
                 [1.02916, 1.26631, 1.06627, 0.0055157, -0.687724, 0.628828, 0.36274],
                 [1.0353323680034034, 1.2475545539059603, 1.068355652684458, 0.0060528266989232455, -0.6275626515819213,
                  0.6713338937141787, 0.3942578912069877],
                 [1.0415047360068068, 1.2287991078119205, 1.0704413053689157, 0.006551094555090425, -0.5633723642404597,
                  0.709529854926159, 0.42324466604873906],
                 [1.0476771040102102, 1.2100436617178807, 1.0725269580533736, 0.007007305656250026,
                  -0.49556533571144873, 0.7431707722400116, 0.4495142923165224],
                 [1.0538494720136136, 1.1912882156238411, 1.0746126107378315, 0.007418531215715761,
                  -0.42457687395184984, 0.7720406774986897, 0.4728981241105117],
                 [1.060021840017017, 1.1725327695298013, 1.0766982634222892, 0.007782131245244194, -0.35086271110222167,
                  0.7959542315161461, 0.49324604177106396],
                 [1.0661942080204203, 1.1537773234357616, 1.0787839161067472, 0.00809577150324646, -0.2748960777748668,
                  0.8147579139189203, 0.5104274156177849],
                 [1.0723665760238237, 1.1350218773417218, 1.080869568791205, 0.008357438480166122, -0.1971646650051895,
                  0.828331008717126, 0.5243319445667127],
                 [1.078538944027227, 1.116266431247682, 1.0829552214756628, 0.008565452324819933, -0.11816749336994378,
                  0.8365863792776681, 0.5348703642418576],
                 [1.088090925162347, 1.0872416441364399, 1.0861828506587201, 0.0087786000250608, 0.0053950000154014315,
                  0.8387670057979981, 0.5443930015541114]]
        tj_17 = [[0.926, 1.452, 1.09, 0.00874141, 0.00537073, 0.838782, 0.544371],
                 [0.929501, 1.4534, 1.0905, 0.0188636, 0.00732849, 0.82843, 0.559727],
                 [0.933002, 1.45479, 1.09101, 0.0289774, 0.00928296, 0.817705, 0.574832],
                 [0.936503, 1.45619, 1.09151, 0.0390781, 0.0112333, 0.806614, 0.589679],
                 [0.940004, 1.45759, 1.09202, 0.0491612, 0.0131785, 0.795159, 0.604261],
                 [0.944872, 1.44808, 1.09206, 0.0538799, 0.0162869, 0.786905, 0.614501],
                 [0.94974, 1.43857, 1.0921, 0.0585875, 0.0193919, 0.77849, 0.624616],
                 [0.954609, 1.42906, 1.09214, 0.0632832, 0.022493, 0.769915, 0.634603],
                 [0.959477, 1.41955, 1.09218, 0.0679658, 0.0255894, 0.761182, 0.644459],
                 [0.963678, 1.41966, 1.09239, 0.070409, 0.0337787, 0.748488, 0.658534],
                 [0.96788, 1.41978, 1.09259, 0.0728217, 0.0419535, 0.73547, 0.672325],
                 [0.972081, 1.4199, 1.0928, 0.0752029, 0.05011, 0.722134, 0.685825],
                 [0.976282, 1.42002, 1.09301, 0.0775516, 0.058245, 0.708487, 0.699028],
                 [0.979277, 1.41424, 1.09112, 0.0913532, 0.049508, 0.702313, 0.704244],
                 [0.982272, 1.40847, 1.08923, 0.105124, 0.0407546, 0.695907, 0.709226],
                 [0.985268, 1.40269, 1.08734, 0.118861, 0.0319877, 0.689269, 0.713973],
                 [0.988263, 1.39691, 1.08546, 0.132558, 0.0232101, 0.682402, 0.718482],
                 [0.994501, 1.39465, 1.08369, 0.14695, 0.0287847, 0.687581, 0.7105],
                 [1.00074, 1.39239, 1.08191, 0.161294, 0.0343499, 0.692533, 0.702284],
                 [1.00698, 1.39013, 1.08014, 0.175586, 0.0399038, 0.697258, 0.693837],
                 [1.01322, 1.38787, 1.07837, 0.189819, 0.0454445, 0.701753, 0.685161],
                 [1.01658, 1.38694, 1.07814, 0.200475, 0.0396742, 0.687968, 0.696373],
                 [1.01994, 1.38602, 1.0779, 0.211038, 0.0338856, 0.673866, 0.707262],
                 [1.02329, 1.38509, 1.07767, 0.221504, 0.0280813, 0.659451, 0.717824],
                 [1.02665, 1.38416, 1.07743, 0.231867, 0.022264, 0.644732, 0.728054],
                 [1.02414, 1.38435, 1.07579, 0.249435, 0.0254488, 0.632738, 0.732651],
                 [1.02163, 1.38454, 1.07416, 0.266882, 0.0286213, 0.620437, 0.736894],
                 [1.01911, 1.38472, 1.07252, 0.2842, 0.0317799, 0.607837, 0.74078],
                 [1.0166, 1.38491, 1.07089, 0.30138, 0.0349232, 0.594943, 0.744307],
                 [1.02017, 1.38471, 1.07032, 0.311802, 0.0486372, 0.583048, 0.748645],
                 [1.02374, 1.38451, 1.06974, 0.32208, 0.062329, 0.570886, 0.752641],
                 [1.02731, 1.3843, 1.06917, 0.332212, 0.0759924, 0.558463, 0.756293],
                 [1.03088, 1.3841, 1.0686, 0.342191, 0.089621, 0.545785, 0.7596],
                 [1.02667, 1.37871, 1.0661, 0.355764, 0.0954095, 0.535986, 0.759637],
                 [1.02246, 1.37331, 1.0636, 0.369225, 0.101168, 0.526019, 0.759435],
                 [1.01825, 1.36792, 1.06109, 0.38257, 0.106895, 0.515887, 0.758995],
                 [1.01404, 1.36252, 1.05859, 0.395796, 0.112588, 0.505593, 0.758317],
                 [1.0158, 1.36401, 1.05988, 0.414033, 0.108041, 0.493532, 0.757186],
                 [1.01756, 1.3655, 1.06116, 0.432063, 0.103441, 0.481224, 0.755675],
                 [1.01931, 1.36698, 1.06245, 0.449877, 0.098788, 0.468676, 0.753787],
                 [1.02107, 1.36847, 1.06374, 0.467466, 0.094086, 0.455893, 0.751522],
                 [1.02105, 1.36625, 1.06098, 0.484294, 0.085648, 0.446388, 0.747571],
                 [1.02102, 1.36403, 1.05822, 0.5009, 0.0771706, 0.436677, 0.743275],
                 [1.02099, 1.36181, 1.05545, 0.517274, 0.0686577, 0.426764, 0.738638],
                 [1.02097, 1.35959, 1.05269, 0.533411, 0.0601132, 0.416656, 0.73366],
                 [1.01762, 1.35675, 1.05163, 0.537937, 0.0506483, 0.40031, 0.740142],
                 [1.01426, 1.3539, 1.05057, 0.542238, 0.0411623, 0.383797, 0.746313],
                 [1.01091, 1.35106, 1.0495, 0.546312, 0.0316589, 0.367122, 0.752172],
                 [1.00756, 1.34821, 1.04844, 0.550156, 0.0221423, 0.350294, 0.757715],
                 [1.01046, 1.34157, 1.05016, 0.555663, 0.0128799, 0.33664, 0.760096],
                 [1.01335, 1.33494, 1.05188, 0.560998, 0.00361351, 0.322882, 0.762244],
                 [1.01625, 1.3283, 1.05361, 0.56616, -0.005654, 0.309025, 0.764156],
                 [1.01915, 1.32166, 1.05533, 0.571148, -0.0149198, 0.295073, 0.765833],
                 [1.02173, 1.32112, 1.05986, 0.580941, -0.0108981, 0.278423, 0.764768],
                 [1.02431, 1.32057, 1.06438, 0.590507, -0.00687214, 0.261664, 0.763404],
                 [1.02689, 1.32002, 1.06891, 0.599843, -0.00284352, 0.244804, 0.761742],
                 [1.02947, 1.31948, 1.07344, 0.608945, 0.00118621, 0.227848, 0.759783],
                 [1.01799, 1.30105, 1.07425, 0.608309, 0.002995, 0.217005, 0.763452],
                 [1.0065, 1.28262, 1.07506, 0.607592, 0.00480339, 0.206134, 0.767019],
                 [0.995014, 1.2642, 1.07587, 0.606793, 0.00661113, 0.195235, 0.770482],
                 [1.0031663110715479, 1.2487007818941893, 1.0767732697036099, 0.5692375280307443, 0.0067418927249689,
                  0.26832349434929226, 0.7771265572248306],
                 [1.0113186221430959, 1.2332015637883786, 1.0776765394072196, 0.527813432638127, 0.006826836345445997,
                  0.33958839168853855, 0.7784896266891819],
                 [1.0194709332146439, 1.217702345682568, 1.0785798091108294, 0.48280204984848196, 0.00686538132869785,
                  0.40854527433480436, 0.774561686387866],
                 [1.027623244286192, 1.2022031275767575, 1.0794830788144392, 0.43450929981506725, 0.006857265703497263,
                  0.47472547639071294, 0.7653694325924588],
                 [1.03577555535774, 1.1867039094709468, 1.0803863485180492, 0.3832634045247631, 0.006802544627743349,
                  0.537679203686416, 0.750975340515131],
                 [1.043927866429288, 1.1712046913651362, 1.081289618221659, 0.32941265703522316, 0.006701590013580174,
                  0.5969785908085703, 0.7314772396954657],
                 [1.052080177500836, 1.1557054732593255, 1.0821928879252687, 0.27332305429886383, 0.00655508799969355,
                  0.6522206090903778, 0.7070076491013612],
                 [1.060232488572384, 1.1402062551535148, 1.0830961576288785, 0.21537580966221648, 0.0063640342879651805,
                  0.703029805798514, 0.6777328764630133],
                 [1.068384799643932, 1.1247070370477041, 1.0839994273324882, 0.15596476194696227, 0.0061297273761782844,
                  0.7490608559000547, 0.6438518879613531],
                 [1.07653711071548, 1.1092078189418935, 1.084902697036098, 0.09549369872184055, 0.005853759732769039,
                  0.7900009090662838, 0.6055949559531176],
                 [1.088090925162347, 1.0872416441364399, 1.0861828506587201, 0.0087786000250608, 0.0053950000154014315,
                  0.8387670057979981, 0.5443930015541114]]
        tj_18 = [[0.926, 1.452, 1.09, 0.00874141, 0.00537073, 0.838782, 0.544371],
                 [0.925948, 1.45204, 1.09067, 0.0059834, -0.0163839, 0.844388, 0.535449],
                 [0.925896, 1.45207, 1.09134, 0.00322184, -0.0381288, 0.849493, 0.52621],
                 [0.925844, 1.45211, 1.092, 0.000458382, -0.0598511, 0.854096, 0.51666],
                 [0.925792, 1.45214, 1.09267, -0.00230535, -0.081538, 0.858194, 0.506803],
                 [0.929393, 1.44229, 1.08994, 0.00247304, -0.094536, 0.858584, 0.503876],
                 [0.932993, 1.43244, 1.0872, 0.00725093, -0.107515, 0.858801, 0.500847],
                 [0.936594, 1.42258, 1.08447, 0.0120274, -0.120473, 0.858847, 0.497718],
                 [0.940195, 1.41273, 1.08174, 0.0168014, -0.133406, 0.85872, 0.49449],
                 [0.941649, 1.4096, 1.08103, 0.00914754, -0.152494, 0.853619, 0.497993],
                 [0.943103, 1.40647, 1.08031, 0.00148946, -0.171512, 0.848123, 0.501267],
                 [0.944558, 1.40335, 1.0796, -0.0061693, -0.190451, 0.842236, 0.50431],
                 [0.946012, 1.40022, 1.07889, -0.0138252, -0.209302, 0.835961, 0.50712],
                 [0.946441, 1.39546, 1.07735, -0.0272707, -0.222573, 0.836121, 0.500619],
                 [0.946871, 1.3907, 1.07581, -0.0407053, -0.235755, 0.835946, 0.493919],
                 [0.9473, 1.38594, 1.07428, -0.0541236, -0.248843, 0.835439, 0.487022],
                 [0.947729, 1.38118, 1.07274, -0.0675204, -0.261831, 0.834597, 0.47993],
                 [0.951431, 1.38483, 1.07288, -0.0607186, -0.279619, 0.832082, 0.475149],
                 [0.955134, 1.38849, 1.07302, -0.053893, -0.297298, 0.82924, 0.470181],
                 [0.958836, 1.39214, 1.07315, -0.0470463, -0.31486, 0.826074, 0.465029],
                 [0.962538, 1.3958, 1.07329, -0.0401812, -0.332299, 0.822584, 0.459694],
                 [0.963333, 1.39808, 1.07601, -0.0479698, -0.350737, 0.817576, 0.45415],
                 [0.964127, 1.40036, 1.07872, -0.0557365, -0.369015, 0.812195, 0.448398],
                 [0.964921, 1.40265, 1.08143, -0.0634778, -0.387125, 0.806443, 0.442442],
                 [0.965716, 1.40493, 1.08415, -0.0711901, -0.405058, 0.800323, 0.436283],
                 [0.968348, 1.39826, 1.08238, -0.0697311, -0.406992, 0.807855, 0.420553],
                 [0.97098, 1.39158, 1.08062, -0.0682505, -0.408801, 0.815137, 0.404692],
                 [0.973612, 1.3849, 1.07885, -0.0667488, -0.410482, 0.822166, 0.388705],
                 [0.976244, 1.37822, 1.07709, -0.0652263, -0.412036, 0.82894, 0.372598],
                 [0.978173, 1.37224, 1.07373, -0.065157, -0.427001, 0.825408, 0.363492],
                 [0.980102, 1.36625, 1.07038, -0.0650669, -0.44183, 0.821612, 0.35427],
                 [0.982031, 1.36026, 1.06703, -0.0649561, -0.456517, 0.817553, 0.344935],
                 [0.983959, 1.35427, 1.06368, -0.0648245, -0.471058, 0.813234, 0.335489],
                 [0.984432, 1.34876, 1.06398, -0.0792426, -0.480937, 0.809264, 0.327891],
                 [0.984905, 1.34326, 1.06429, -0.0936308, -0.490632, 0.804988, 0.320168],
                 [0.985377, 1.33775, 1.06459, -0.107983, -0.500142, 0.800407, 0.312324],
                 [0.98585, 1.33225, 1.06489, -0.122295, -0.509463, 0.795522, 0.304362],
                 [0.989727, 1.32664, 1.06672, -0.119461, -0.52074, 0.785139, 0.313234],
                 [0.993605, 1.32102, 1.06855, -0.116588, -0.53185, 0.774503, 0.322005],
                 [0.997482, 1.31541, 1.07038, -0.113678, -0.542788, 0.763619, 0.330673],
                 [1.00136, 1.3098, 1.07221, -0.110731, -0.553552, 0.752488, 0.339235],
                 [0.997481, 1.30443, 1.07516, -0.124548, -0.560205, 0.744035, 0.342155],
                 [0.993602, 1.29907, 1.0781, -0.138326, -0.566681, 0.735347, 0.344968],
                 [0.989723, 1.2937, 1.08105, -0.152061, -0.572979, 0.726427, 0.347672],
                 [0.985844, 1.28833, 1.08399, -0.165747, -0.579096, 0.717279, 0.350266],
                 [0.982925, 1.27949, 1.08612, -0.183553, -0.586775, 0.712952, 0.337199],
                 [0.980007, 1.27065, 1.08826, -0.201256, -0.594121, 0.708222, 0.323942],
                 [0.977088, 1.26181, 1.09039, -0.218845, -0.601131, 0.703091, 0.310501],
                 [0.9890988468335503, 1.2429211844017138, 1.089934773921317, -0.19769810367080004, -0.5452518314804591,
                  0.7374119562540417, 0.3461784320921251],
                 [1.0011096936671007, 1.2240323688034276, 1.0894795478426342, -0.17536090810421803, -0.4860898155714817,
                  0.7672931399985972, 0.37977161612953675],
                 [1.0131205405006511, 1.2051435532051413, 1.0890243217639513, -0.15196795541914987, -0.4240013048777705,
                  0.7925548495212376, 0.4110783921444179],
                 [1.0251313873342016, 1.186254737606855, 1.0885690956852685, -0.12766008245652363, -0.3593601021347795,
                  0.813044997164902, 0.4399102782681932],
                 [1.037142234167752, 1.1673659220085688, 1.0881138696065855, -0.10258363431795285, -0.2925553785134453,
                  0.8286402223729675, 0.4660936926613087],
                 [1.0491530810013023, 1.1484771064102826, 1.0876586435279028, -0.07688958329774684,
                  -0.22398933062231802, 0.8392466343792089, 0.4894709985595013],
                 [1.0611639278348528, 1.1295882908119963, 1.0872034174492198, -0.05073261995699218,
                  -0.15407475909028365, 0.8448003774754286, 0.5099014533230357],
                 [1.0731747746684033, 1.11069947521371, 1.0867481913705368, -0.02427022181187848, -0.08323258330797983,
                  0.8452680154535721, 0.5272620557751735],
                 [1.088090925162347, 1.0872416441364399, 1.0861828506587201, 0.0087786000250608, 0.005395000015401431,
                  0.8387670057979981, 0.5443930015541114]]
        tj_19 = [[0.926, 1.452, 1.09, 0.00874097, 0.00537172, 0.838833, 0.544292],
                 [0.927429, 1.45325, 1.08873, 0.00149788, -0.0143564, 0.843451, 0.537012],
                 [0.928858, 1.45449, 1.08745, -0.00574598, -0.0340771, 0.847634, 0.529455],
                 [0.930287, 1.45574, 1.08618, -0.0129869, -0.0537802, 0.851379, 0.521625],
                 [0.931716, 1.45699, 1.08491, -0.0202211, -0.0734556, 0.854685, 0.513526],
                 [0.933256, 1.44916, 1.08631, -0.0229372, -0.0832899, 0.846972, 0.524572],
                 [0.934795, 1.44133, 1.08771, -0.0256468, -0.0931004, 0.839017, 0.535467],
                 [0.936334, 1.4335, 1.08911, -0.028349, -0.102884, 0.830822, 0.54621],
                 [0.937873, 1.42567, 1.09052, -0.0310432, -0.112639, 0.822391, 0.556797],
                 [0.94237, 1.42463, 1.08978, -0.0407505, -0.130276, 0.82099, 0.554386],
                 [0.946866, 1.42359, 1.08903, -0.0504409, -0.147859, 0.81925, 0.551746],
                 [0.951363, 1.42254, 1.08829, -0.0601105, -0.165382, 0.817171, 0.548878],
                 [0.95586, 1.4215, 1.08755, -0.0697553, -0.182836, 0.814755, 0.545784],
                 [0.956991, 1.41289, 1.08618, -0.0698589, -0.198195, 0.814603, 0.540611],
                 [0.958123, 1.40428, 1.08482, -0.0699442, -0.213503, 0.814237, 0.535297],
                 [0.959255, 1.39567, 1.08345, -0.0700111, -0.228754, 0.813657, 0.529842],
                 [0.960387, 1.38706, 1.08208, -0.0700596, -0.243946, 0.812863, 0.524248],
                 [0.967494, 1.39077, 1.07995, -0.0865571, -0.245823, 0.811337, 0.52327],
                 [0.974602, 1.39448, 1.07781, -0.10303, -0.247632, 0.809585, 0.522145],
                 [0.981709, 1.39819, 1.07568, -0.119475, -0.249371, 0.807607, 0.520875],
                 [0.988816, 1.4019, 1.07354, -0.135886, -0.251041, 0.805404, 0.519459],
                 [0.992837, 1.39962, 1.07407, -0.132191, -0.259568, 0.81354, 0.503292],
                 [0.996858, 1.39733, 1.07459, -0.128441, -0.267987, 0.821339, 0.486917],
                 [1.00088, 1.39505, 1.07511, -0.124637, -0.276294, 0.828798, 0.47034],
                 [1.0049, 1.39276, 1.07564, -0.120782, -0.284488, 0.835915, 0.453569],
                 [1.00542, 1.39127, 1.08, -0.117662, -0.303738, 0.831856, 0.449348],
                 [1.00595, 1.38978, 1.08436, -0.114493, -0.322861, 0.827453, 0.444942],
                 [1.00647, 1.38829, 1.08872, -0.111277, -0.341851, 0.822707, 0.44035],
                 [1.00699, 1.38681, 1.09309, -0.108014, -0.360699, 0.81762, 0.435576],
                 [1.00219, 1.38566, 1.08812, -0.121651, -0.367438, 0.817769, 0.425963],
                 [0.997383, 1.38451, 1.08315, -0.135248, -0.374057, 0.817654, 0.416211],
                 [0.992578, 1.38336, 1.07819, -0.148801, -0.380556, 0.817273, 0.406325],
                 [0.987773, 1.38221, 1.07322, -0.162306, -0.386931, 0.816628, 0.396307],
                 [0.990048, 1.38204, 1.06941, -0.165688, -0.399425, 0.804653, 0.406866],
                 [0.992323, 1.38188, 1.06559, -0.168999, -0.41175, 0.792338, 0.417254],
                 [0.994599, 1.38171, 1.06177, -0.17224, -0.423902, 0.779689, 0.427465],
                 [0.996874, 1.38154, 1.05796, -0.175407, -0.435874, 0.766709, 0.437496],
                 [1.00167, 1.37462, 1.05993, -0.178747, -0.444207, 0.75611, 0.446125],
                 [1.00647, 1.3677, 1.06191, -0.182038, -0.452421, 0.745308, 0.454635],
                 [1.01127, 1.36079, 1.06388, -0.185281, -0.460514, 0.734307, 0.463023],
                 [1.01607, 1.35387, 1.06586, -0.188475, -0.468484, 0.72311, 0.471288],
                 [1.02077, 1.34886, 1.06249, -0.19586, -0.479603, 0.712276, 0.473585],
                 [1.02547, 1.34384, 1.05912, -0.203186, -0.490578, 0.701228, 0.47574],
                 [1.03017, 1.33883, 1.05575, -0.210451, -0.501405, 0.689968, 0.477752],
                 [1.03488, 1.33382, 1.05237, -0.217653, -0.512082, 0.678501, 0.47962],
                 [1.03556, 1.33925, 1.05107, -0.236019, -0.508818, 0.673566, 0.481362],
                 [1.03623, 1.34468, 1.04976, -0.254296, -0.505362, 0.668377, 0.482923],
                 [1.03691, 1.35011, 1.04846, -0.272478, -0.501717, 0.662938, 0.484303],
                 [1.03759, 1.35554, 1.04715, -0.290557, -0.497884, 0.657249, 0.485501],
                 [1.02835, 1.34979, 1.04515, -0.293076, -0.508112, 0.648135, 0.485644],
                 [1.0191, 1.34403, 1.04315, -0.295538, -0.518242, 0.638895, 0.485692],
                 [1.00986, 1.33828, 1.04115, -0.297942, -0.528271, 0.62953, 0.485645],
                 [1.00062, 1.33252, 1.03915, -0.300289, -0.538198, 0.620044, 0.485505],
                 [1.00337, 1.3309, 1.03842, -0.319527, -0.53878, 0.617467, 0.475765],
                 [1.00612, 1.32928, 1.0377, -0.338615, -0.539109, 0.614599, 0.4658],
                 [1.00887, 1.32765, 1.03697, -0.357542, -0.539182, 0.611441, 0.455615],
                 [1.01162, 1.32603, 1.03624, -0.376301, -0.539002, 0.607995, 0.445216],
                 [1.01704, 1.31992, 1.03966, -0.385032, -0.53247, 0.601076, 0.454899],
                 [1.02246, 1.3138, 1.04308, -0.393663, -0.5258, 0.594001, 0.464463],
                 [1.02788, 1.30769, 1.04651, -0.402191, -0.518993, 0.586771, 0.473907],
                 [1.0333, 1.30158, 1.04993, -0.410614, -0.512051, 0.579389, 0.483227],
                 [1.04374, 1.29205, 1.0585, -0.397539, -0.540055, 0.568122, 0.477011],
                 [1.05417, 1.28253, 1.06706, -0.384019, -0.567454, 0.556219, 0.470261],
                 [1.057660839369471, 1.262432662507433, 1.0690279533979832, -0.3505148577989926, -0.5188080475223508,
                  0.6045807127081635, 0.4924019761429551],
                 [1.0611516787389421, 1.2423353250148659, 1.0709959067959665, -0.3147960770348759, -0.46688413914587423,
                  0.6491225569147283, 0.5114318494207744],
                 [1.0646425181084134, 1.222237987522299, 1.0729638601939497, -0.27708840347366004, -0.4120104401933094,
                  0.6895632217481487, 0.5272304781065366],
                 [1.0681333574778844, 1.202140650029732, 1.074931813591933, -0.237630075536382, -0.3545336446438984,
                  0.7256472016514476, 0.5396980459029945],
                 [1.0716241968473554, 1.182043312537165, 1.0768997669899163, -0.19667039234061362, -0.2948168929336507,
                  0.7571465167577973, 0.5487557822739294],
                 [1.0751150362168265, 1.161945975044598, 1.0788677203878994, -0.15446813861978267, -0.23323747762116936,
                  0.7838621532754159, 0.5543464601191952],
                 [1.0786058755862975, 1.141848637552031, 1.0808356737858826, -0.11128994971203343, -0.1701844596398627,
                  0.8056253208637758, 0.5564347573375691],
                 [1.0820967149557688, 1.121751300059464, 1.082803627183866, -0.06740862694868914, -0.1060562101971477,
                  0.8222985190569668, 0.5550074799930448],
                 [1.088090925162347, 1.0872416441364399, 1.0861828506587201, 0.0087786000250608, 0.0053950000154014315,
                  0.8387670057979981, 0.5443930015541114]]
        tj_20 = [[0.926, 1.453, 1.09, 0.00873968, 0.00537457, 0.838991, 0.544048],
                 [0.929621, 1.44768, 1.08933, -0.00394222, -0.00700532, 0.836056, 0.548585],
                 [0.933243, 1.44236, 1.08866, -0.0166228, -0.0193828, 0.832835, 0.552933],
                 [0.936864, 1.43704, 1.08799, -0.0292976, -0.0317536, 0.829327, 0.557091],
                 [0.940485, 1.43171, 1.08731, -0.0419624, -0.0441136, 0.825535, 0.561058],
                 [0.937441, 1.43255, 1.08666, -0.0531266, -0.0522234, 0.815204, 0.574363],
                 [0.934398, 1.43338, 1.086, -0.0642657, -0.0603086, 0.804486, 0.587396],
                 [0.931354, 1.43422, 1.08534, -0.0753743, -0.0683651, 0.793388, 0.600151],
                 [0.92831, 1.43506, 1.08469, -0.0864472, -0.0763892, 0.781913, 0.612621],
                 [0.93407, 1.43287, 1.08654, -0.0999639, -0.0884723, 0.782143, 0.608632],
                 [0.939831, 1.43068, 1.08839, -0.113446, -0.100525, 0.782104, 0.604433],
                 [0.945591, 1.42849, 1.09024, -0.126889, -0.112543, 0.781795, 0.600025],
                 [0.951352, 1.4263, 1.09209, -0.140289, -0.124522, 0.781216, 0.595411],
                 [0.950105, 1.42177, 1.09215, -0.146906, -0.143196, 0.776787, 0.595413],
                 [0.948858, 1.41724, 1.09221, -0.153463, -0.16181, 0.772037, 0.595169],
                 [0.947611, 1.41271, 1.09226, -0.159956, -0.180358, 0.766969, 0.594679],
                 [0.946365, 1.40818, 1.09232, -0.166384, -0.198831, 0.761585, 0.593945],
                 [0.945321, 1.40812, 1.08916, -0.174568, -0.217929, 0.755597, 0.592543],
                 [0.944278, 1.40805, 1.08599, -0.18267, -0.236925, 0.749254, 0.590862],
                 [0.943235, 1.40799, 1.08283, -0.190686, -0.255809, 0.742559, 0.588903],
                 [0.942192, 1.40793, 1.07967, -0.198612, -0.274574, 0.735515, 0.586669],
                 [0.947585, 1.41089, 1.07787, -0.21658, -0.274247, 0.734174, 0.582125],
                 [0.952979, 1.41386, 1.07606, -0.234474, -0.273826, 0.732579, 0.577381],
                 [0.958372, 1.41683, 1.07426, -0.252286, -0.273311, 0.730732, 0.572437],
                 [0.963766, 1.4198, 1.07246, -0.270011, -0.272701, 0.728632, 0.567296],
                 [0.962477, 1.424, 1.07144, -0.273377, -0.291622, 0.722039, 0.564696],
                 [0.961188, 1.42821, 1.07041, -0.276627, -0.310421, 0.715143, 0.56186],
                 [0.959899, 1.43242, 1.06939, -0.279762, -0.32909, 0.707946, 0.558789],
                 [0.95861, 1.43662, 1.06837, -0.282779, -0.347621, 0.700453, 0.555482],
                 [0.957341, 1.44158, 1.06784, -0.301445, -0.342687, 0.695957, 0.554383],
                 [0.956072, 1.44654, 1.06732, -0.319992, -0.337619, 0.691186, 0.553065],
                 [0.954803, 1.4515, 1.0668, -0.338414, -0.332417, 0.686143, 0.551528],
                 [0.953534, 1.45646, 1.06628, -0.356701, -0.327084, 0.680829, 0.549775],
                 [0.952797, 1.45455, 1.06819, -0.369815, -0.326693, 0.687278, 0.533065],
                 [0.952061, 1.45265, 1.0701, -0.382747, -0.326141, 0.693388, 0.516091],
                 [0.951324, 1.45074, 1.07201, -0.395491, -0.325428, 0.699156, 0.498864],
                 [0.950587, 1.44884, 1.07393, -0.408039, -0.324554, 0.704579, 0.48139],
                 [0.95818, 1.4543, 1.07687, -0.418103, -0.329576, 0.703191, 0.471266],
                 [0.965773, 1.45975, 1.07982, -0.428071, -0.334522, 0.70164, 0.461033],
                 [0.973366, 1.46521, 1.08277, -0.43794, -0.339391, 0.699927, 0.450694],
                 [0.980959, 1.47066, 1.08571, -0.447708, -0.344181, 0.698052, 0.44025],
                 [0.97883, 1.47029, 1.08296, -0.455204, -0.359989, 0.692021, 0.429306],
                 [0.976702, 1.46992, 1.08021, -0.46249, -0.375631, 0.68567, 0.418164],
                 [0.974573, 1.46955, 1.07745, -0.469562, -0.391099, 0.679002, 0.406828],
                 [0.972444, 1.46917, 1.0747, -0.476417, -0.406387, 0.67202, 0.395305],
                 [0.966456, 1.45254, 1.07127, -0.480138, -0.409771, 0.667164, 0.395535],
                 [0.960468, 1.43592, 1.06783, -0.483836, -0.413135, 0.662275, 0.395745],
                 [0.95448, 1.41929, 1.0644, -0.487511, -0.416478, 0.657354, 0.395936],
                 [0.948492, 1.40266, 1.06097, -0.491161, -0.419802, 0.6524, 0.396108],
                 [0.952954, 1.39534, 1.05851, -0.501639, -0.419469, 0.641366, 0.401314],
                 [0.957417, 1.38803, 1.05605, -0.511987, -0.419028, 0.630167, 0.406417],
                 [0.961879, 1.38071, 1.05359, -0.522203, -0.418478, 0.618804, 0.411415],
                 [0.966341, 1.3734, 1.05113, -0.532283, -0.41782, 0.607281, 0.416306],
                 [0.968702, 1.37405, 1.05363, -0.546964, -0.405146, 0.599392, 0.421209],
                 [0.971063, 1.37471, 1.05612, -0.561392, -0.392285, 0.591225, 0.425916],
                 [0.973425, 1.37536, 1.05861, -0.57556, -0.379243, 0.582785, 0.430427],
                 [0.975786, 1.37601, 1.0611, -0.589462, -0.366025, 0.574075, 0.434739],
                 [0.981006, 1.36108, 1.0604, -0.614388, -0.354675, 0.553692, 0.436072],
                 [0.986226, 1.34615, 1.0597, -0.638596, -0.342911, 0.532662, 0.436897],
                 [0.9954330910475714, 1.3227484925467365, 1.0620936602007558, 0.5920402919249865, 0.3178425667672619,
                  -0.5796871878427012, -0.4608982097374608],
                 [1.0046401820951427, 1.299346985093473, 1.0644873204015115, -0.5421792457533312, -0.2909996307755754,
                  0.6234759767096398, 0.48232622448793755],
                 [1.013847273142714, 1.2759454776402095, 1.0668809806022672, -0.48929105833991815, -0.2625319609148655,
                  0.6637837215175784, 0.5010612744715685],
                 [1.0230543641902852, 1.2525439701869459, 1.069274640803023, -0.4336710195073935, -0.23259850029881105,
                  0.7003853726401685, 0.516998756565095],
                 [1.0322614552378566, 1.2291424627336824, 1.0716683010037786, -0.37562967178341183,
                  -0.20136637596860574, 0.7330765726282868, 0.5300496872640531],
                 [1.0414685462854278, 1.205740955280419, 1.0740619612045343, -0.31549107655320463, -0.16900996577505847,
                  0.7616747971959135, 0.540141199503035],
                 [1.0506756373329993, 1.1823394478271554, 1.07645562140529, -0.2535910047325132, -0.1357099247773573,
                  0.7860203743052335, 0.5472169494932875],
                 [1.0598827283805705, 1.158937940373892, 1.0788492816060458, -0.19027506206287806, -0.10165217659439021,
                  0.805977375661133, 0.551237431306133],
                 [1.0690898194281417, 1.1355364329206286, 1.0812429418068015, -0.12589675949629, -0.06702687534019534,
                  0.8214343756376035, 0.5521801974458374],
                 [1.0782969104757132, 1.112134925467365, 1.0836366020075572, -0.060815539442800576, -0.0320273439393511,
                  0.832305073398771, 0.5500399841803669],
                 [1.088090925162347, 1.0872416441364399, 1.0861828506587201, 0.0087786000250608, 0.0053950000154014315,
                  0.8387670057979981, 0.5443930015541114]]
        tj_21 = [[0.926, 1.451, 1.09, 0.00874376, 0.00536792, 0.838578, 0.544685],
                 [0.930655, 1.4474, 1.08994, 0.0203086, 0.00521888, 0.846627, 0.531773],
                 [0.93531, 1.44379, 1.08989, 0.031866, 0.00506793, 0.854367, 0.518667],
                 [0.939965, 1.44019, 1.08983, 0.0434117, 0.00491513, 0.861795, 0.505371],
                 [0.944621, 1.43658, 1.08977, 0.0549416, 0.00476053, 0.868908, 0.491891],
                 [0.950298, 1.43763, 1.08922, 0.0460827, -0.00716892, 0.875208, 0.481493],
                 [0.955976, 1.43869, 1.08866, 0.0372067, -0.0190957, 0.881186, 0.470917],
                 [0.961653, 1.43974, 1.08811, 0.0283171, -0.0310155, 0.886838, 0.460167],
                 [0.967331, 1.44079, 1.08756, 0.019417, -0.0429238, 0.892164, 0.449248],
                 [0.970521, 1.44138, 1.08717, 0.00865166, -0.0613557, 0.889634, 0.45245],
                 [0.973711, 1.44197, 1.08678, -0.00211775, -0.0797586, 0.886685, 0.455439],
                 [0.976901, 1.44256, 1.08639, -0.0128862, -0.0981238, 0.883316, 0.458212],
                 [0.980091, 1.44315, 1.086, -0.0236485, -0.116443, 0.87953, 0.460769],
                 [0.980413, 1.44044, 1.08821, -0.0407197, -0.112284, 0.885139, 0.449738],
                 [0.980735, 1.43773, 1.09041, -0.0577721, -0.108073, 0.890338, 0.438498],
                 [0.981057, 1.43502, 1.09262, -0.0747979, -0.103812, 0.895126, 0.427056],
                 [0.981379, 1.43231, 1.09482, -0.0917891, -0.0995028, 0.899501, 0.415417],
                 [0.977356, 1.43565, 1.09415, -0.0877203, -0.109216, 0.905766, 0.399955],
                 [0.973334, 1.43899, 1.09348, -0.0836173, -0.118887, 0.911679, 0.384338],
                 [0.969311, 1.44233, 1.09281, -0.0794818, -0.128511, 0.917237, 0.36857],
                 [0.965288, 1.44567, 1.09214, -0.0753153, -0.138086, 0.922438, 0.35266],
                 [0.96194, 1.43831, 1.08971, -0.0702304, -0.153708, 0.920386, 0.352607],
                 [0.958592, 1.43095, 1.08727, -0.0651263, -0.169288, 0.918081, 0.352458],
                 [0.955245, 1.4236, 1.08483, -0.0600044, -0.184822, 0.915526, 0.352212],
                 [0.951897, 1.41624, 1.08239, -0.054866, -0.200305, 0.912719, 0.351869],
                 [0.952374, 1.40998, 1.07979, -0.0599338, -0.214451, 0.906099, 0.359727],
                 [0.95285, 1.40372, 1.07718, -0.0649817, -0.228525, 0.899179, 0.367466],
                 [0.953326, 1.39746, 1.07458, -0.0700081, -0.242524, 0.89196, 0.375084],
                 [0.953803, 1.3912, 1.07198, -0.0750113, -0.256442, 0.884447, 0.382577],
                 [0.958342, 1.3911, 1.07216, -0.0667176, -0.274366, 0.881687, 0.378022],
                 [0.962882, 1.391, 1.07233, -0.0583959, -0.292176, 0.878557, 0.373309],
                 [0.967421, 1.39091, 1.07251, -0.0500499, -0.309864, 0.875061, 0.36844],
                 [0.97196, 1.39081, 1.07269, -0.0416828, -0.327422, 0.871198, 0.363417],
                 [0.968704, 1.39032, 1.07377, -0.0506702, -0.344765, 0.862687, 0.366525],
                 [0.965448, 1.38984, 1.07486, -0.059634, -0.361948, 0.853777, 0.369462],
                 [0.962192, 1.38936, 1.07595, -0.0685702, -0.378964, 0.84447, 0.372228],
                 [0.958937, 1.38887, 1.07704, -0.0774746, -0.395804, 0.834773, 0.374822],
                 [0.965323, 1.38289, 1.0768, -0.0837848, -0.398988, 0.826965, 0.387191],
                 [0.971709, 1.3769, 1.07656, -0.0900729, -0.402066, 0.81894, 0.399458],
                 [0.978095, 1.37092, 1.07632, -0.0963373, -0.405039, 0.810698, 0.41162],
                 [0.984482, 1.36494, 1.07608, -0.102576, -0.407904, 0.802243, 0.423673],
                 [0.989655, 1.35951, 1.07561, -0.116224, -0.415863, 0.79509, 0.42589],
                 [0.994828, 1.35407, 1.07513, -0.129836, -0.423694, 0.787694, 0.427978],
                 [1.0, 1.34864, 1.07465, -0.143408, -0.431395, 0.780057, 0.429935],
                 [1.00517, 1.34321, 1.07417, -0.156937, -0.438965, 0.772181, 0.43176],
                 [1.00831, 1.34692, 1.07284, -0.172973, -0.442998, 0.772621, 0.420583],
                 [1.01144, 1.35063, 1.07151, -0.18894, -0.446853, 0.772753, 0.409238],
                 [1.01458, 1.35434, 1.07018, -0.204832, -0.450531, 0.772578, 0.39773],
                 [1.01771, 1.35805, 1.06884, -0.220642, -0.454029, 0.772094, 0.386064],
                 [1.01366, 1.36184, 1.07248, -0.238463, -0.451297, 0.768616, 0.385611],
                 [1.00961, 1.36563, 1.07611, -0.256204, -0.448412, 0.764879, 0.385027],
                 [1.00555, 1.36942, 1.07975, -0.273859, -0.445376, 0.760884, 0.384314],
                 [1.0015, 1.37321, 1.08338, -0.291421, -0.44219, 0.756632, 0.383471],
                 [1.00588, 1.3683, 1.08648, -0.299946, -0.454436, 0.748711, 0.378089],
                 [1.01026, 1.36339, 1.08958, -0.308377, -0.46654, 0.740556, 0.372587],
                 [1.01463, 1.35849, 1.09268, -0.316711, -0.478498, 0.732167, 0.366968],
                 [1.01901, 1.35358, 1.09578, -0.324946, -0.490304, 0.723548, 0.361234],
                 [1.02134, 1.34745, 1.09531, -0.336454, -0.497499, 0.711569, 0.36464],
                 [1.02368, 1.34131, 1.09484, -0.347848, -0.504525, 0.699349, 0.367923],
                 [1.02601, 1.33518, 1.09437, -0.359124, -0.511379, 0.686892, 0.37108],
                 [1.02834, 1.32904, 1.09389, -0.370278, -0.51806, 0.674202, 0.374112],
                 [1.02668, 1.32494, 1.09197, -0.385083, -0.511456, 0.677283, 0.362508],
                 [1.02501, 1.32084, 1.09005, -0.399731, -0.504644, 0.680088, 0.350757],
                 [1.02334, 1.31674, 1.08813, -0.414216, -0.497626, 0.682617, 0.338863],
                 [1.02167, 1.31263, 1.08621, -0.428533, -0.490406, 0.684867, 0.326831],
                 [1.02311, 1.31912, 1.08436, -0.439189, -0.498489, 0.672725, 0.325673],
                 [1.02454, 1.32561, 1.08252, -0.449702, -0.506408, 0.660362, 0.324409],
                 [1.02598, 1.3321, 1.08067, -0.460068, -0.514162, 0.647782, 0.323038],
                 [1.02741, 1.33859, 1.07882, -0.470282, -0.521747, 0.634991, 0.321561],
                 [1.02923, 1.3343, 1.08052, -0.497285, -0.538762, 0.605894, 0.308765],
                 [1.03104, 1.33002, 1.08222, -0.523278, -0.554683, 0.575568, 0.295343],
                 [1.036084458274039, 1.3085534719374226, 1.0825703963298978, -0.48722526930864596, -0.5168611003107662,
                  0.6215191379253805, 0.330424123098108],
                 [1.041128916548078, 1.2870869438748451, 1.0829207926597957, -0.4482136339168813, -0.47590031858232296,
                  0.6636958495398906, 0.36349861684439816],
                 [1.0461733748221171, 1.2656204158122677, 1.0832711889896935, -0.4064802255024216, -0.43204963670227114,
                  0.701842274429818, 0.39436576870224965],
                 [1.0512178330961561, 1.24415388774969, 1.0836215853195912, -0.3622784697903503, -0.3855753375186471,
                  0.7357267687933654, 0.4228381381860175],
                 [1.0562622913701951, 1.2226873596871126, 1.083971981649489, -0.3158767815212041, -0.33675963577057216,
                  0.7651435693612626, 0.4487428271032129],
                 [1.061306749644234, 1.2012208316245352, 1.084322377979387, -0.2675569345050181, -0.2858989643419389,
                  0.7899140428922579, 0.4719225294771298],
                 [1.066351207918273, 1.1797543035619578, 1.0846727743092848, -0.21761235055254846, -0.2333021741755464,
                  0.8098877709217749, 0.49223648678589094],
                 [1.0713956661923123, 1.1582877754993803, 1.0850231706391826, -0.16634631767412084,
                  -0.17928865877869168, 0.8249434631766053, 0.509561342717231],
                 [1.0764401244663513, 1.136821247436803, 1.0853735669690805, -0.11407014836607714, -0.1241864147091402,
                  0.8349896941089389, 0.5237918922485257],
                 [1.0814845827403903, 1.1153547193742255, 1.0857239632989781, -0.06110128916860626,
                  -0.06833004981915138, 0.8399654580771285, 0.5348417205032965],
                 [1.088090925162347, 1.0872416441364399, 1.0861828506587201, 0.0087786000250608, 0.0053950000154014315,
                  0.8387670057979981, 0.5443930015541114]]
        tj_22 = [[0.926, 1.451, 1.09, 0.00874376, 0.00536792, 0.838578, 0.544685],
                 [0.927952, 1.45349, 1.09202, -0.00271589, -0.000774472, 0.829349, 0.558724],
                 [0.929904, 1.45598, 1.09405, -0.0141743, -0.00691651, 0.819746, 0.57251],
                 [0.931856, 1.45847, 1.09607, -0.0256263, -0.0130554, 0.809773, 0.586038],
                 [0.933808, 1.46096, 1.09809, -0.0370668, -0.0191885, 0.799434, 0.599302],
                 [0.934023, 1.45848, 1.09647, -0.0532411, -0.0316844, 0.793686, 0.605165],
                 [0.934238, 1.456, 1.09484, -0.0693896, -0.0441651, 0.787552, 0.610734],
                 [0.934453, 1.45352, 1.09322, -0.0855043, -0.0566242, 0.781036, 0.616008],
                 [0.934668, 1.45104, 1.09159, -0.101578, -0.069056, 0.774141, 0.620982],
                 [0.934594, 1.4533, 1.09132, -0.103058, -0.0729226, 0.759638, 0.637974],
                 [0.934521, 1.45557, 1.09105, -0.104486, -0.0767516, 0.744742, 0.654638],
                 [0.934447, 1.45783, 1.09078, -0.105859, -0.080541, 0.729462, 0.670963],
                 [0.934374, 1.46009, 1.09051, -0.107178, -0.0842888, 0.713806, 0.686942],
                 [0.935265, 1.45496, 1.08895, -0.123776, -0.0941988, 0.71109, 0.68568],
                 [0.936157, 1.44983, 1.08739, -0.140328, -0.104073, 0.708102, 0.684155],
                 [0.937049, 1.4447, 1.08582, -0.156825, -0.113907, 0.704843, 0.682369],
                 [0.93794, 1.43956, 1.08426, -0.173262, -0.123697, 0.701314, 0.680321],
                 [0.940151, 1.43711, 1.08188, -0.183152, -0.1392, 0.704871, 0.670996],
                 [0.942361, 1.43465, 1.07949, -0.192961, -0.154643, 0.708119, 0.661377],
                 [0.944571, 1.43219, 1.0771, -0.202685, -0.170017, 0.711057, 0.651468],
                 [0.946781, 1.42973, 1.07472, -0.212321, -0.185317, 0.713683, 0.641275],
                 [0.951623, 1.42886, 1.07646, -0.203872, -0.199959, 0.705998, 0.648089],
                 [0.956464, 1.42799, 1.07821, -0.195342, -0.214523, 0.698036, 0.65465],
                 [0.961306, 1.42713, 1.07995, -0.186736, -0.229003, 0.689802, 0.660955],
                 [0.966147, 1.42626, 1.08169, -0.178057, -0.243393, 0.681297, 0.667001],
                 [0.969962, 1.42915, 1.07952, -0.174681, -0.261472, 0.674229, 0.668232],
                 [0.973778, 1.43205, 1.07735, -0.171236, -0.279449, 0.666899, 0.669203],
                 [0.977593, 1.43495, 1.07518, -0.167726, -0.297317, 0.659309, 0.669913],
                 [0.981409, 1.43784, 1.07301, -0.164149, -0.315069, 0.651461, 0.670362],
                 [0.980166, 1.43838, 1.07209, -0.18117, -0.328498, 0.648767, 0.662094],
                 [0.978924, 1.43892, 1.07116, -0.198092, -0.341748, 0.645719, 0.653464],
                 [0.977681, 1.43946, 1.07024, -0.214906, -0.354811, 0.642318, 0.644478],
                 [0.976439, 1.44, 1.06932, -0.231602, -0.367681, 0.638567, 0.635141],
                 [0.975191, 1.43415, 1.06866, -0.247261, -0.369035, 0.628473, 0.638512],
                 [0.973943, 1.4283, 1.068, -0.26283, -0.370256, 0.618153, 0.641653],
                 [0.972695, 1.42245, 1.06734, -0.278305, -0.371343, 0.60761, 0.644562],
                 [0.971447, 1.4166, 1.06668, -0.293679, -0.372297, 0.596848, 0.64724],
                 [0.968458, 1.41156, 1.06597, -0.286464, -0.389215, 0.594507, 0.642659],
                 [0.96547, 1.40652, 1.06525, -0.279144, -0.405991, 0.591949, 0.637845],
                 [0.962481, 1.40148, 1.06453, -0.271722, -0.422619, 0.589176, 0.632797],
                 [0.959493, 1.39645, 1.06381, -0.264201, -0.439093, 0.586187, 0.627519],
                 [0.967349, 1.39098, 1.06877, -0.265437, -0.449634, 0.57675, 0.628276],
                 [0.975205, 1.38551, 1.07372, -0.26662, -0.460085, 0.567196, 0.628907],
                 [0.983061, 1.38004, 1.07868, -0.267749, -0.470442, 0.557528, 0.62941],
                 [0.990918, 1.37458, 1.08363, -0.268824, -0.480704, 0.547747, 0.629786],
                 [0.989029, 1.36874, 1.08577, -0.27002, -0.487991, 0.532326, 0.636854],
                 [0.987141, 1.3629, 1.08791, -0.271124, -0.49511, 0.516723, 0.643704],
                 [0.985253, 1.35706, 1.09004, -0.272134, -0.50206, 0.500944, 0.650334],
                 [0.983364, 1.35122, 1.09218, -0.273052, -0.508838, 0.484993, 0.656741],
                 [0.996498, 1.33271, 1.08416, -0.271217, -0.504818, 0.480748, 0.663688],
                 [1.00963, 1.3142, 1.07614, -0.269358, -0.500754, 0.476462, 0.670578],
                 [1.02277, 1.29569, 1.06812, -0.267477, -0.496648, 0.472135, 0.677411],
                 [1.03015, 1.28997, 1.06513, -0.261743, -0.482261, 0.466075, 0.694039],
                 [1.03754, 1.28424, 1.06213, -0.255864, -0.467607, 0.459756, 0.710283],
                 [1.04493, 1.27851, 1.05913, -0.249844, -0.452694, 0.453183, 0.726134],
                 [1.0497868312221736, 1.2569868718678348, 1.0621742148594788, -0.22454914071732496, -0.4081431372671045,
                  0.5114695332936218, 0.7220774054194478],
                 [1.0546436624443476, 1.2354637437356697, 1.065218429718958, -0.19789827923360098, -0.3611275896167205,
                  0.5666674566994501, 0.7136603734352736],
                 [1.0595004936665213, 1.2139406156035044, 1.068262644578437, -0.17005243129100564, -0.3119314137103049,
                  0.6184436225232455, 0.7009339837053373],
                 [1.064357324888695, 1.1924174874713394, 1.0713068594379158, -0.14117974090451502, -0.2608516751123692,
                  0.6664853862221071, 0.6839750830999429],
                 [1.069214156110869, 1.170894359339174, 1.0743510742973947, -0.11145455256149221, -0.2081968130684874,
                  0.710502653019553, 0.6628860760293731],
                 [1.0740709873330427, 1.149371231207009, 1.0773952891568739, -0.08105635846214677, -0.15428477802966853,
                  0.7502296296101378, 0.637794306086396],
                 [1.0789278185552165, 1.1278481030748437, 1.0804395040163528, -0.05016871467610979,
                  -0.09944111174438591, 0.7854264291223766, 0.6088512870963321],
                 [1.088090925162347, 1.0872416441364399, 1.0861828506587201, 0.0087786000250608, 0.005395000015401431,
                  0.8387670057979981, 0.5443930015541114]]
        tj_23 = [[0.926, 1.452, 1.09, 0.0087403, 0.00537316, 0.838923, 0.544154],
                 [0.931171, 1.45233, 1.08813, 0.0109712, -0.00977284, 0.83225, 0.554207],
                 [0.936342, 1.45265, 1.08627, 0.0131979, -0.0249151, 0.82526, 0.564049],
                 [0.941513, 1.45298, 1.0844, 0.0154196, -0.0400479, 0.817957, 0.573677],
                 [0.946684, 1.45331, 1.08254, 0.0176354, -0.0551655, 0.810343, 0.583087],
                 [0.950616, 1.44722, 1.08354, 0.00954935, -0.0666345, 0.803547, 0.591423],
                 [0.954548, 1.44112, 1.08454, 0.00146032, -0.0780826, 0.796501, 0.599573],
                 [0.95848, 1.43503, 1.08554, -0.00662917, -0.0895063, 0.789205, 0.607536],
                 [0.962413, 1.42894, 1.08653, -0.0147166, -0.100902, 0.781663, 0.615309],
                 [0.961268, 1.42429, 1.08889, -0.0324317, -0.102268, 0.776001, 0.62154],
                 [0.960124, 1.41963, 1.09124, -0.0501342, -0.103594, 0.77004, 0.62753],
                 [0.95898, 1.41498, 1.0936, -0.0678174, -0.104881, 0.76378, 0.633278],
                 [0.957836, 1.41033, 1.09595, -0.0854744, -0.106126, 0.757225, 0.638781],
                 [0.960949, 1.41286, 1.09518, -0.0870447, -0.112299, 0.743619, 0.653332],
                 [0.964063, 1.41538, 1.0944, -0.088577, -0.118423, 0.729688, 0.667597],
                 [0.967176, 1.41791, 1.09363, -0.0900705, -0.124495, 0.715438, 0.68157],
                 [0.97029, 1.42044, 1.09285, -0.0915246, -0.130513, 0.700874, 0.695244],
                 [0.975713, 1.42328, 1.09351, -0.106743, -0.12817, 0.692367, 0.702001],
                 [0.981136, 1.42611, 1.09416, -0.121924, -0.125781, 0.683614, 0.708509],
                 [0.986559, 1.42895, 1.09482, -0.137061, -0.123348, 0.674618, 0.714765],
                 [0.991982, 1.43179, 1.09548, -0.15215, -0.120871, 0.665383, 0.720768],
                 [0.997455, 1.43225, 1.0951, -0.168687, -0.125855, 0.669258, 0.7126],
                 [1.00293, 1.43271, 1.09473, -0.18516, -0.130791, 0.672878, 0.704162],
                 [1.0084, 1.43317, 1.09436, -0.201562, -0.135678, 0.676243, 0.695457],
                 [1.01387, 1.43364, 1.09398, -0.217888, -0.140513, 0.67935, 0.686487],
                 [1.01049, 1.43172, 1.09121, -0.22288, -0.157659, 0.682833, 0.677649],
                 [1.00711, 1.42981, 1.08844, -0.22778, -0.17474, 0.686036, 0.668534],
                 [1.00373, 1.4279, 1.08566, -0.232587, -0.191749, 0.688959, 0.659145],
                 [1.00035, 1.42598, 1.08289, -0.237298, -0.20868, 0.691599, 0.649487],
                 [1.00205, 1.42268, 1.08404, -0.248837, -0.225034, 0.686804, 0.644779],
                 [1.00376, 1.41938, 1.08519, -0.260264, -0.241288, 0.681703, 0.639784],
                 [1.00546, 1.41608, 1.08633, -0.271575, -0.257434, 0.676298, 0.634504],
                 [1.00716, 1.41278, 1.08748, -0.282766, -0.273466, 0.670592, 0.628941],
                 [0.998597, 1.4129, 1.08306, -0.287484, -0.286122, 0.670111, 0.621642],
                 [0.99003, 1.41302, 1.07865, -0.292134, -0.298712, 0.669472, 0.614196],
                 [0.981463, 1.41314, 1.07423, -0.296715, -0.31123, 0.668675, 0.606605],
                 [0.972896, 1.41326, 1.06982, -0.301227, -0.323676, 0.66772, 0.598871],
                 [0.9782, 1.40921, 1.06866, -0.292396, -0.332526, 0.676263, 0.588727],
                 [0.983503, 1.40516, 1.0675, -0.283469, -0.341266, 0.684582, 0.578386],
                 [0.988806, 1.40111, 1.06634, -0.274447, -0.349892, 0.692674, 0.567854],
                 [0.99411, 1.39706, 1.06518, -0.265334, -0.358402, 0.700535, 0.557133],
                 [0.990983, 1.39456, 1.06151, -0.275902, -0.37046, 0.698922, 0.546028],
                 [0.987856, 1.39205, 1.05785, -0.286364, -0.382376, 0.697041, 0.534714],
                 [0.984729, 1.38955, 1.05418, -0.296717, -0.394145, 0.694893, 0.523195],
                 [0.981602, 1.38705, 1.05052, -0.306956, -0.405763, 0.692479, 0.511476],
                 [0.984041, 1.37904, 1.05409, -0.306566, -0.419882, 0.688876, 0.50514],
                 [0.98648, 1.37102, 1.05766, -0.3061, -0.433895, 0.685099, 0.498676],
                 [0.988919, 1.36301, 1.06123, -0.305555, -0.447799, 0.681149, 0.492086],
                 [0.991357, 1.355, 1.0648, -0.304934, -0.461589, 0.677026, 0.485372],
                 [0.990023, 1.35368, 1.06883, -0.289551, -0.474908, 0.674175, 0.485912],
                 [0.988688, 1.35237, 1.07286, -0.274045, -0.488026, 0.67104, 0.486247],
                 [0.987353, 1.35105, 1.0769, -0.258423, -0.500938, 0.66762, 0.486376],
                 [0.986018, 1.34974, 1.08093, -0.242693, -0.513639, 0.663919, 0.486299],
                 [0.986635, 1.35645, 1.08096, -0.225879, -0.520757, 0.664039, 0.486666],
                 [0.987251, 1.36316, 1.08099, -0.20899, -0.527702, 0.663937, 0.486869],
                 [0.987868, 1.36987, 1.08102, -0.192031, -0.534471, 0.663614, 0.486911],
                 [0.988485, 1.37658, 1.08106, -0.175009, -0.541061, 0.663069, 0.48679],
                 [0.985909, 1.36974, 1.07827, -0.167858, -0.553177, 0.662048, 0.476982],
                 [0.983334, 1.3629, 1.07548, -0.160658, -0.56513, 0.660831, 0.467033],
                 [0.980758, 1.35607, 1.07269, -0.153411, -0.576916, 0.659419, 0.456946],
                 [0.978182, 1.34923, 1.0699, -0.146118, -0.588532, 0.657812, 0.446725],
                 [0.974783, 1.34156, 1.07151, -0.157196, -0.595409, 0.655649, 0.436924],
                 [0.971384, 1.33389, 1.07312, -0.168232, -0.602125, 0.653307, 0.427004],
                 [0.967985, 1.32622, 1.07473, -0.179222, -0.608678, 0.650789, 0.41697],
                 [0.964586, 1.31854, 1.07634, -0.190164, -0.615067, 0.648095, 0.406822],
                 [0.96708, 1.31371, 1.07593, -0.205284, -0.622186, 0.63836, 0.404029],
                 [0.969575, 1.30887, 1.07552, -0.220326, -0.629067, 0.628382, 0.401082],
                 [0.972069, 1.30403, 1.0751, -0.235284, -0.635708, 0.618164, 0.397982],
                 [0.974563, 1.29919, 1.07469, -0.250152, -0.642107, 0.60771, 0.39473],
                 [0.977329, 1.29591, 1.072, -0.263491, -0.645705, 0.607425, 0.380359],
                 [0.980094, 1.29263, 1.06932, -0.276724, -0.649047, 0.606898, 0.365837],
                 [0.98286, 1.28935, 1.06663, -0.289848, -0.652131, 0.60613, 0.351169],
                 [0.985625, 1.28607, 1.06394, -0.302856, -0.654955, 0.605122, 0.336363],
                 [0.991116, 1.28892, 1.06506, -0.31956, -0.66583, 0.588131, 0.329627],
                 [0.996608, 1.29178, 1.06619, -0.336031, -0.676218, 0.570709, 0.32265],
                 [1.0021, 1.29463, 1.06731, -0.352255, -0.686112, 0.552871, 0.315438],
                 [1.0098034122326964, 1.2760513186512188, 1.069000705715234, 0.32752941070713915, 0.639152142898019,
                  -0.6014710237565233, -0.34991660568196],
                 [1.0175068244653926, 1.2574726373024376, 1.070691411430468, -0.3007184674162445, -0.5881228538960939,
                  0.6462414942133751, 0.3821672974455081],
                 [1.025210236698089, 1.2388939559536565, 1.072382117145702, -0.27199275426733777, -0.5333488001541652,
                  0.6868971401143948, 0.41198460881790894],
                 [1.0329136489307855, 1.2203152746048753, 1.0740728228609357, -0.24153517698669522, -0.4751787454445655,
                  0.7231790940677726, 0.43917868352078565],
                 [1.0406170611634817, 1.201736593256094, 1.0757635285761697, -0.20953966862879841, -0.4139830769548657,
                  0.75485633736549, 0.4635763682556228],
                 [1.0483204733961782, 1.1831579119073128, 1.0774542342914037, -0.17620995474306836, -0.3501514469151787,
                  0.7817271709527053, 0.48502231522573563],
                 [1.0560238856288744, 1.1645792305585316, 1.0791449400066377, -0.14175825618867935,
                  -0.28409029155882926, 0.8036204997096054, 0.5033799712849611],
                 [1.0637272978615708, 1.1460005492097505, 1.0808356457218717, -0.10640393785703944,
                  -0.21622024321394978, 0.8203969218682998, 0.5185324474148503],
                 [1.0714307100942673, 1.1274218678609693, 1.0825263514371057, -0.07037211190595427,
                  -0.14697345200404024, 0.831949616628088, 0.5303832629941194],
                 [1.0791341223269635, 1.1088431865121882, 1.0842170571523395, -0.03389220439912056,
                  -0.07679083421107043, 0.8382050243173781, 0.5388569601213663],
                 [1.088090925162347, 1.0872416441364399, 1.0861828506587201, 0.0087786000250608, 0.0053950000154014315,
                  0.8387670057979981, 0.5443930015541114]]
        tj_24 = [[0.926, 1.452, 1.09, 0.0087396, 0.00537434, 0.838993, 0.544045],
                 [0.927454, 1.45576, 1.09078, 0.00288896, 0.00947289, 0.849515, 0.527471],
                 [0.928907, 1.45953, 1.09155, -0.00296294, 0.0135673, 0.859666, 0.510667],
                 [0.930361, 1.46329, 1.09233, -0.00881355, 0.0176558, 0.869442, 0.49364],
                 [0.931814, 1.46705, 1.09311, -0.0146603, 0.0217366, 0.878839, 0.476398],
                 [0.935067, 1.46863, 1.09235, -0.0164347, 0.00740596, 0.886358, 0.46265],
                 [0.938321, 1.4702, 1.0916, -0.0182017, -0.00692804, 0.893474, 0.448693],
                 [0.941574, 1.47177, 1.09085, -0.0199604, -0.0212589, 0.900185, 0.434531],
                 [0.944827, 1.47335, 1.09009, -0.0217101, -0.0355801, 0.906486, 0.420173],
                 [0.94722, 1.46616, 1.09163, -0.0194939, -0.0233867, 0.911848, 0.409398],
                 [0.949613, 1.45898, 1.09317, -0.0172719, -0.0111863, 0.916937, 0.398502],
                 [0.952006, 1.4518, 1.0947, -0.0150447, 0.00101736, 0.921752, 0.387487],
                 [0.954398, 1.44462, 1.09624, -0.0128131, 0.0132208, 0.926292, 0.376356],
                 [0.956857, 1.44425, 1.09659, -0.0219978, 0.0111586, 0.933661, 0.357307],
                 [0.959315, 1.44389, 1.09694, -0.0311715, 0.00909072, 0.940558, 0.338077],
                 [0.961773, 1.44352, 1.09729, -0.0403294, 0.00701829, 0.946979, 0.318676],
                 [0.964232, 1.44316, 1.09764, -0.0494669, 0.00494231, 0.952921, 0.299114],
                 [0.967056, 1.44287, 1.09701, -0.038941, -0.011082, 0.956547, 0.288754],
                 [0.969881, 1.44259, 1.09638, -0.0283961, -0.0271008, 0.959705, 0.278253],
                 [0.972706, 1.44231, 1.09575, -0.0178373, -0.0431065, 0.962395, 0.267617],
                 [0.975531, 1.44203, 1.09512, -0.00726981, -0.0590911, 0.964616, 0.25685],
                 [0.977415, 1.44537, 1.09404, -0.01572, -0.0532634, 0.969396, 0.23914],
                 [0.9793, 1.4487, 1.09296, -0.0241633, -0.0474123, 0.973747, 0.221325],
                 [0.981184, 1.45204, 1.09188, -0.0325958, -0.0415401, 0.977669, 0.203411],
                 [0.983068, 1.45537, 1.0908, -0.041014, -0.0356497, 0.981158, 0.185408],
                 [0.981806, 1.44652, 1.0883, -0.0315012, -0.0370999, 0.983677, 0.173236],
                 [0.980544, 1.43767, 1.08581, -0.0219807, -0.038541, 0.985953, 0.16102],
                 [0.979282, 1.42883, 1.08331, -0.0124547, -0.0399726, 0.987986, 0.148765],
                 [0.978021, 1.41998, 1.08081, -0.00292569, -0.0413943, 0.989774, 0.136473],
                 [0.974855, 1.41779, 1.08325, -0.00576246, -0.059676, 0.989989, 0.12778],
                 [0.97169, 1.41559, 1.08569, -0.00859683, -0.0779328, 0.98979, 0.119034],
                 [0.968524, 1.4134, 1.08813, -0.0114276, -0.096157, 0.989177, 0.110238],
                 [0.965359, 1.41121, 1.09056, -0.0142536, -0.114341, 0.988151, 0.101395],
                 [0.967519, 1.40658, 1.08815, -0.00760497, -0.125163, 0.988303, 0.0867973],
                 [0.969678, 1.40195, 1.08574, -0.000953502, -0.135937, 0.988085, 0.0721668],
                 [0.971837, 1.39732, 1.08333, 0.00569832, -0.146661, 0.987497, 0.0575093],
                 [0.973997, 1.39269, 1.08091, 0.012348, -0.15733, 0.98654, 0.0428303],
                 [0.97554, 1.38855, 1.08137, 0.0124466, -0.172568, 0.984486, 0.0291929],
                 [0.977084, 1.3844, 1.08183, 0.01254, -0.187733, 0.982017, 0.0155432],
                 [0.978627, 1.38026, 1.08229, 0.0126281, -0.202819, 0.979133, 0.00188692],
                 [0.98017, 1.37611, 1.08275, 0.0127109, -0.217819, 0.975835, -0.0117702],
                 [0.985105, 1.37171, 1.08353, 0.00237367, -0.232549, 0.972495, -0.0130147],
                 [0.990039, 1.36732, 1.08432, -0.00796432, -0.247199, 0.968827, -0.0142548],
                 [0.994974, 1.36292, 1.0851, -0.0182996, -0.261767, 0.964833, -0.0154901],
                 [0.999908, 1.35852, 1.08589, -0.0286288, -0.276247, 0.960515, -0.0167202],
                 [1.00569, 1.36016, 1.08973, -0.0375684, -0.290679, 0.955988, -0.0134503],
                 [1.01146, 1.3618, 1.09357, -0.046496, -0.305018, 0.951156, -0.0101762],
                 [1.01724, 1.36344, 1.09741, -0.0554088, -0.31926, 0.946021, -0.00689877],
                 [1.02302, 1.36508, 1.10125, -0.0643038, -0.3334, 0.940583, -0.00361915],
                 [1.02566, 1.3604, 1.09704, -0.0598411, -0.35004, 0.934816, -0.00319468],
                 [1.0283, 1.35572, 1.09283, -0.0553586, -0.366563, 0.928741, -0.00276916],
                 [1.03095, 1.35104, 1.08862, -0.0508579, -0.382965, 0.922359, -0.00234272],
                 [1.03359, 1.34635, 1.08442, -0.0463403, -0.399241, 0.915672, -0.0019155],
                 [1.03385, 1.34118, 1.08379, -0.0287041, -0.405841, 0.913462, -0.00757123],
                 [1.03411, 1.33601, 1.08317, -0.0110566, -0.412281, 0.910894, -0.013224],
                 [1.03436, 1.33084, 1.08255, 0.00659515, -0.41856, 0.907969, -0.0188716],
                 [1.03462, 1.32567, 1.08193, 0.0242443, -0.424675, 0.904689, -0.0245118],
                 [1.03878, 1.32058, 1.08274, 0.022896, -0.440221, 0.897078, -0.0305366],
                 [1.04293, 1.31548, 1.08355, 0.02154, -0.455618, 0.889164, -0.036551],
                 [1.04708, 1.31039, 1.08436, 0.0201766, -0.470861, 0.88095, -0.0425532],
                 [1.05124, 1.30529, 1.08517, 0.0188065, -0.485945, 0.872438, -0.0485409],
                 [1.05108, 1.30402, 1.08437, 0.00789155, -0.50261, 0.863559, -0.0398212],
                 [1.05093, 1.30275, 1.08357, -0.00302777, -0.518998, 0.854205, -0.0310794],
                 [1.05077, 1.30148, 1.08276, -0.0139454, -0.5351, 0.844379, -0.0223206],
                 [1.05062, 1.30021, 1.08196, -0.0248554, -0.550907, 0.834087, -0.0135494],
                 [1.0584, 1.29525, 1.07615, -0.0206023, -0.562119, 0.826694, -0.0132266],
                 [1.06619, 1.29028, 1.07034, -0.0163452, -0.57322, 0.819137, -0.0129011],
                 [1.07397, 1.28532, 1.06453, -0.0120849, -0.584207, 0.811417, -0.0125731],
                 [1.08176, 1.28036, 1.05872, -0.00782211, -0.595079, 0.803536, -0.0122426],
                 [1.08071, 1.27799, 1.05647, 0.010798, -0.603399, 0.797219, -0.0153238],
                 [1.07966, 1.27562, 1.05423, 0.0294131, -0.611439, 0.79053, -0.0183978],
                 [1.07862, 1.27325, 1.05198, 0.0480145, -0.619195, 0.783474, -0.0214633],
                 [1.07757, 1.27088, 1.04974, 0.0665936, -0.626662, 0.776053, -0.0245188],
                 [1.0719, 1.26512, 1.04959, 0.0784299, -0.633328, 0.769236, -0.0319388],
                 [1.06622, 1.25936, 1.04944, 0.0902438, -0.639814, 0.762198, -0.0393497],
                 [1.06055, 1.2536, 1.04928, 0.102032, -0.646115, 0.754943, -0.0467493],
                 [1.05488, 1.24784, 1.04913, 0.113791, -0.652233, 0.747471, -0.0541356],
                 [1.06196, 1.25082, 1.052, 0.11638, -0.652939, 0.745083, -0.0705471],
                 [1.06904, 1.2538, 1.05487, 0.118936, -0.653461, 0.742485, -0.0869388],
                 [1.07612, 1.25678, 1.05774, 0.121458, -0.653799, 0.739677, -0.103306],
                 [1.0832, 1.25976, 1.06061, 0.123947, -0.653952, 0.736661, -0.119644],
                 [1.08173, 1.26049, 1.06122, 0.138362, -0.647382, 0.736937, -0.13666],
                 [1.08026, 1.26122, 1.06183, 0.152703, -0.640461, 0.736816, -0.153603],
                 [1.0788, 1.26195, 1.06244, 0.166962, -0.633194, 0.736296, -0.170462],
                 [1.07733, 1.26268, 1.06305, 0.18113, -0.625585, 0.735378, -0.187229],
                 [1.07725, 1.25986, 1.05983, 0.186022, -0.637215, 0.721674, -0.196313],
                 [1.07716, 1.25705, 1.05662, 0.190835, -0.648573, 0.70766, -0.205312],
                 [1.07708, 1.25423, 1.0534, 0.195566, -0.659651, 0.693342, -0.214223],
                 [1.077, 1.25141, 1.05019, 0.200212, -0.670446, 0.678726, -0.223042],
                 [1.0786, 1.24708, 1.05637, 0.188656, -0.679674, 0.666874, -0.240272],
                 [1.08021, 1.24274, 1.06255, 0.176976, -0.688457, 0.654585, -0.257344],
                 [1.08182, 1.23841, 1.06874, 0.165179, -0.696787, 0.641866, -0.274248],
                 [1.0822929303037272, 1.2270094339588447, 1.070055476177172, 0.15924983145486366, -0.6677086078268553,
                  0.6959589321303685, -0.21082189403256393],
                 [1.0827658606074542, 1.2156088679176895, 1.0713709523543438, 0.15207380930334333, -0.6334023618674316,
                  0.7446028168719058, -0.1457451529593801],
                 [1.0832387909111814, 1.2042083018765342, 1.0726864285315154, 0.14370710647206805, -0.5941368138543428,
                  0.7874167418577043, -0.07952728220323753],
                 [1.0837117212149083, 1.192807735835379, 1.0740019047086873, 0.13421523109462177, -0.5502193982472996,
                  0.8240654901721184, -0.012686742764916244],
                 [1.0841846515186355, 1.1814071697942237, 1.0753173808858592, 0.12367250098206176, -0.5019939718781098,
                  0.854262115912004, 0.05425312909342133],
                 [1.0846575818223625, 1.1700066037530683, 1.076632857063031, 0.11216146174243458, -0.44983812168112985,
                  0.8777701908651199, 0.12076821937415723],
                 [1.0851305121260897, 1.1586060377119132, 1.077948333240203, 0.09977224047973866, -0.39416020832970783,
                  0.8944056556540287, 0.18633773995951985],
                 [1.0856034424298167, 1.1472054716707578, 1.0792638094173748, 0.08660184013262773, -0.3353961689258361,
                  0.9040382608517341, 0.2504483061864997],
                 [1.0860763727335438, 1.1358049056296025, 1.0805792855945464, 0.07275337997791738, -0.27400610377674744,
                  0.9065925867856631, 0.3125979564555005],
                 [1.086549303037271, 1.1244043395884473, 1.0818947617717183, 0.05833528824546582, -0.21047067398271121,
                  0.9020486340453108, 0.37230008240070395],
                 [1.087022233340998, 1.113003773547292, 1.0832102379488902, 0.04346045316595154, -0.14528733804155977,
                  0.8904419800700953, 0.42908723885039496],
                 [1.0874951636447252, 1.1016032075061366, 1.084525714126062, 0.028245339098522178, -0.07896645693591504,
                  0.8718635005914017, 0.48251480374668804],
                 [1.088090925162347, 1.0872416441364399, 1.0861828506587201, 0.0087786000250608, 0.0053950000154014315,
                  0.8387670057979981, 0.5443930015541114]]
        tj_25 = [[0.926, 1.452, 1.09, 0.00874269, 0.00536934, 0.838693, 0.544508],
                 [0.926892, 1.44902, 1.09021, -0.00840427, -0.00689834, 0.835487, 0.549402],
                 [0.927785, 1.44604, 1.09041, -0.0255472, -0.0191627, 0.831882, 0.554033],
                 [0.928677, 1.44305, 1.09062, -0.0426779, -0.0314179, 0.827878, 0.558399],
                 [0.929569, 1.44007, 1.09083, -0.0597882, -0.0436581, 0.823478, 0.562498],
                 [0.934295, 1.43596, 1.08908, -0.0524317, -0.0483682, 0.814334, 0.575996],
                 [0.93902, 1.43185, 1.08733, -0.0450573, -0.0530618, 0.804911, 0.589298],
                 [0.943746, 1.42774, 1.08558, -0.0376675, -0.0577373, 0.795214, 0.602397],
                 [0.948472, 1.42363, 1.08383, -0.0302648, -0.062393, 0.785244, 0.615291],
                 [0.953221, 1.4209, 1.08617, -0.0431116, -0.0759618, 0.782038, 0.617081],
                 [0.957971, 1.41817, 1.08851, -0.0559427, -0.0895031, 0.778548, 0.618646],
                 [0.96272, 1.41544, 1.09084, -0.0687536, -0.103012, 0.774776, 0.619987],
                 [0.96747, 1.4127, 1.09318, -0.0815395, -0.116483, 0.770722, 0.621104],
                 [0.964691, 1.41559, 1.09152, -0.0944037, -0.124337, 0.760246, 0.630599],
                 [0.961911, 1.41847, 1.08987, -0.107227, -0.132138, 0.749444, 0.639824],
                 [0.959132, 1.42135, 1.08821, -0.120005, -0.139883, 0.738323, 0.648777],
                 [0.956353, 1.42423, 1.08655, -0.132732, -0.147567, 0.726886, 0.657452],
                 [0.960162, 1.42345, 1.0845, -0.132368, -0.157731, 0.713663, 0.669541],
                 [0.963971, 1.42266, 1.08244, -0.131947, -0.167828, 0.700137, 0.681345],
                 [0.96778, 1.42187, 1.08039, -0.131471, -0.177854, 0.686314, 0.692861],
                 [0.971589, 1.42109, 1.07833, -0.130939, -0.187804, 0.6722, 0.704082],
                 [0.968777, 1.41606, 1.07622, -0.136138, -0.205422, 0.66926, 0.70097],
                 [0.965966, 1.41103, 1.0741, -0.141288, -0.222968, 0.666081, 0.69761],
                 [0.963155, 1.406, 1.07198, -0.146389, -0.240434, 0.662665, 0.694001],
                 [0.960343, 1.40097, 1.06987, -0.151437, -0.257814, 0.659014, 0.690145],
                 [0.957343, 1.40262, 1.07027, -0.15525, -0.262016, 0.642675, 0.703003],
                 [0.954343, 1.40428, 1.07067, -0.158991, -0.266097, 0.626039, 0.715535],
                 [0.951343, 1.40593, 1.07108, -0.162659, -0.270053, 0.609111, 0.727734],
                 [0.948344, 1.40758, 1.07148, -0.16625, -0.273885, 0.5919, 0.739596],
                 [0.953858, 1.41075, 1.07372, -0.173602, -0.288135, 0.583177, 0.739422],
                 [0.959373, 1.41393, 1.07595, -0.180896, -0.302288, 0.57426, 0.739002],
                 [0.964887, 1.4171, 1.07819, -0.188129, -0.316342, 0.565151, 0.738336],
                 [0.970402, 1.42028, 1.08043, -0.1953, -0.330289, 0.555854, 0.737424],
                 [0.975556, 1.41756, 1.08175, -0.205041, -0.336504, 0.541627, 0.742538],
                 [0.98071, 1.41485, 1.08306, -0.214708, -0.342597, 0.527205, 0.747384],
                 [0.985864, 1.41213, 1.08437, -0.224298, -0.348566, 0.512591, 0.751959],
                 [0.991019, 1.40942, 1.08569, -0.233806, -0.354409, 0.497792, 0.756262],
                 [0.992031, 1.41357, 1.08446, -0.251046, -0.362016, 0.496018, 0.748256],
                 [0.993043, 1.41773, 1.08322, -0.26818, -0.369471, 0.494034, 0.739933],
                 [0.994055, 1.42188, 1.08199, -0.2852, -0.376769, 0.491842, 0.731299],
                 [0.995067, 1.42603, 1.08076, -0.3021, -0.383908, 0.489442, 0.722355],
                 [0.991796, 1.42242, 1.08332, -0.318149, -0.389547, 0.488733, 0.712863],
                 [0.988524, 1.4188, 1.08588, -0.334078, -0.395037, 0.487839, 0.703101],
                 [0.985253, 1.41519, 1.08844, -0.349879, -0.400377, 0.48676, 0.693071],
                 [0.981982, 1.41158, 1.09101, -0.365547, -0.405565, 0.485496, 0.682778],
                 [0.983249, 1.40614, 1.08983, -0.3759, -0.389754, 0.485071, 0.686656],
                 [0.984515, 1.40071, 1.08866, -0.386112, -0.373798, 0.484467, 0.690278],
                 [0.985782, 1.39527, 1.08749, -0.396181, -0.357703, 0.483681, 0.693644],
                 [0.987048, 1.38984, 1.08631, -0.406102, -0.341475, 0.482716, 0.696751],
                 [0.991804, 1.38227, 1.08554, -0.422159, -0.323813, 0.478255, 0.698712],
                 [0.99656, 1.3747, 1.08476, -0.437964, -0.30596, 0.473511, 0.700259],
                 [1.00132, 1.36713, 1.08398, -0.45351, -0.287925, 0.468486, 0.701391],
                 [1.0005, 1.36357, 1.08829, -0.454986, -0.273054, 0.451492, 0.717345],
                 [0.999688, 1.36001, 1.09259, -0.456114, -0.257973, 0.434152, 0.73275],
                 [0.998874, 1.35645, 1.09689, -0.456892, -0.242694, 0.416479, 0.747593],
                 [1.0077509486866874, 1.3296641770557582, 1.0958246548862753, -0.4175046722852951, -0.2216925794611023,
                  0.47252212575068175, 0.743817914215494],
                 [1.0166278973733747, 1.3028783541115168, 1.0947593097725508, -0.37596840458597236, -0.1995500841829344,
                  0.5261330943142825, 0.7362142960598775],
                 [1.025504846060062, 1.276092531167275, 1.0936939646588262, -0.3324968385499404, -0.17638040432941715,
                  0.577035791411733, 0.7248210129078221],
                 [1.0343817947467493, 1.2493067082230336, 1.0926286195451016, -0.2873137441760007, -0.152302805887627,
                  0.6249681952778399, 0.7096967117138724],
                 [1.0432587434334366, 1.2225208852787919, 1.091563274431377, -0.24065170155950905, -0.12744122835792662,
                  0.669683573739513, 0.6909192448576101],
                 [1.052135692120124, 1.1957350623345502, 1.0904979293176524, -0.19275090368573095, -0.10192364677506388,
                  0.7109517542702671, 0.6685852693983286],
                 [1.0610126408068112, 1.1689492393903087, 1.0894325842039279, -0.14385792003582915,
                  -0.07588141295635088, 0.7485603088038982, 0.6428097495325699],
                 [1.0698895894934985, 1.142163416446067, 1.0883672390902033, -0.09422442736981886, -0.04944857936784252,
                  0.7823156472085031, 0.6137253648156256],
                 [1.0787665381801859, 1.1153775935018255, 1.0873018939764787, -0.044105914219794685,
                  -0.0227612090889211, 0.8120440137921872, 0.5814818271931813],
                 [1.088090925162347, 1.0872416441364399, 1.0861828506587201, 0.0087786000250608, 0.0053950000154014315,
                  0.8387670057979981, 0.5443930015541114]]
        cereal_name = u'cereal'
        drawer_frame_id = u'iai_kitchen/oven_area_area_right_drawer_board_2_link'
        # take milk out of fridge
        kitchen_setup.set_kitchen_js({u'oven_area_area_right_drawer_main_joint': 0.48})
        # spawn milk
        cereal_pose = PoseStamped()
        cereal_pose.header.frame_id = drawer_frame_id
        cereal_pose.pose.position = Point(0.123, 0.0, 0.165)  # big box: Point(0.123, 0.0, 0.165)
        cereal_pose.pose.orientation = Quaternion(0.0087786, 0.005395, -0.838767, -0.544393)
        kitchen_setup.add_box(cereal_name, [0.1028, 0.0634, 0.26894], cereal_pose)  # big box [0.1028, 0.0634, 0.26894]

        drawer_T_box = tf.msg_to_kdl(cereal_pose)

        # grasp milk
        kitchen_setup.open_l_gripper()
        grasp_pose = PoseStamped()
        grasp_pose.header.frame_id = cereal_name
        grasp_pose.pose.position = Point(0.03, 0, 0.05)
        grasp_pose.pose.orientation = Quaternion(0, 0, 1, 0)
        box_T_r_goal = tf.msg_to_kdl(grasp_pose)
        box_T_r_goal_pre = deepcopy(box_T_r_goal)
        box_T_r_goal_pre.p[0] += 0.2

        pre_grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal_pre, drawer_frame_id)

        box_T_r_goal_post = deepcopy(box_T_r_goal)
        box_T_r_goal_post.p[0] += 0.4
        post_grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal_post, drawer_frame_id)

        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup.set_json_goal(u'CartesianPose',
                                    tip_link=kitchen_setup.r_tip,
                                    root_link=kitchen_setup.default_root,
                                    goal=pre_grasp_pose)
        kitchen_setup.plan_and_execute()

        grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal, drawer_frame_id)

        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup.set_cart_goal(grasp_pose,
                                    tip_link=kitchen_setup.r_tip)
        kitchen_setup.plan_and_execute()

        kitchen_setup.attach_object(cereal_name, kitchen_setup.r_tip)

        for tj in [tj_1, tj_2, tj_3, tj_4, tj_5, tj_6, tj_7, tj_8, tj_9, tj_10, tj_11, tj_12,
                   tj_13, tj_14, tj_15, tj_16, tj_17, tj_18, tj_19, tj_20, tj_21, tj_22, tj_23,
                   tj_24, tj_25]:
            poses = []
            for i, point in enumerate(tj_12):
                base_pose = PoseStamped()
                base_pose.header.frame_id = 'map'
                base_pose.pose.position.x = point[0]
                base_pose.pose.position.y = point[1]
                base_pose.pose.position.z = point[2] if len(point) > 3 else 0
                if len(point) > 3:
                    base_pose.pose.orientation = Quaternion(point[3], point[4], point[5], point[6])
                else:
                    arr = quaternion_from_euler(0, 0, point[2])
                    base_pose.pose.orientation = Quaternion(arr[0], arr[1], arr[2], arr[3])
                if i == 0:
                    # important assumption for constraint:
                    # we do not to reach the first pose, since it is the start pose
                    continue
                else:
                    poses.append(base_pose)

            # kitchen_setup.keep_position(kitchen_setup.r_tip)
            # kitchen_setup.close_r_gripper()

            # x = Vector3Stamped()
            # x.header.frame_id = 'milk'
            # x.vector.x = 1
            # x_map = Vector3Stamped()
            # x_map.header.frame_id = 'iai_kitchen/iai_fridge_door'
            # x_map.vector.x = 1
            # z = Vector3Stamped()
            # z.header.frame_id = 'milk'
            # z.vector.z = 1
            # z_map = Vector3Stamped()
            # z_map.header.frame_id = 'map'
            # z_map.vector.z = 1
            # kitchen_setup.align_planes('milk', x, root_normal=x_map)
            # kitchen_setup.align_planes('milk', z, root_normal=z_map)
            # kitchen_setup.keep_orientation(u'milk')
            # kitchen_setup.set_cart_goal(grasp_pose, cereal_name, kitchen_setup.default_root)
            # kitchen_setup.allow_all_collisions()
            kitchen_setup.allow_all_collisions()
            kitchen_setup.set_json_goal(u'CartesianPathCarrot',
                                        tip_link=kitchen_setup.r_tip,
                                        root_link=kitchen_setup.default_root,
                                        goal=post_grasp_pose,
                                        goals=poses,
                                        narrow=True,
                                        narrow_padding=0.2,
                                        predict_f=2.0)
            try:
                kitchen_setup.plan_and_execute()
            except Exception:
                pass

            kitchen_setup.allow_all_collisions()
            kitchen_setup.set_json_goal(u'CartesianPathCarrot',
                                        tip_link=kitchen_setup.r_tip,
                                        root_link=kitchen_setup.default_root,
                                        goal=grasp_pose,
                                        goals=list(reversed(poses)),
                                        narrow=True,
                                        narrow_padding=0.2,
                                        predict_f=2.0)
            try:
                kitchen_setup.plan_and_execute()
            except Exception:
                pass
            # kitchen_setup.set_joint_goal(gaya_pose)

            # place milk back

            # kitchen_setup.add_json_goal(u'BasePointingForward')
            # milk_goal = PoseStamped()
            # milk_goal.header.frame_id = u'iai_kitchen/kitchen_island_surface'
            # milk_goal.pose.position = Point(.1, -.2, .13)
            # milk_goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0,0,1]))

    @pytest.mark.repeat(25)
    def test_cereal_1(self, kitchen_setup):
        # FIXME collision avoidance needs soft_threshholds at 0
        cereal_name = u'cereal'
        drawer_frame_id = u'iai_kitchen/oven_area_area_right_drawer_board_2_link'
        # take milk out of fridge
        kitchen_setup.set_kitchen_js({u'oven_area_area_right_drawer_main_joint': 0.48})
        # spawn milk
        cereal_pose = PoseStamped()
        cereal_pose.header.frame_id = drawer_frame_id
        cereal_pose.pose.position = Point(0.123, 0.0, 0.165)  # big box: Point(0.123, 0.0, 0.165)
        cereal_pose.pose.orientation = Quaternion(0.0087786, 0.005395, -0.838767, -0.544393)
        kitchen_setup.add_box(cereal_name, [0.1028, 0.0634, 0.26894], cereal_pose)  # big box [0.1028, 0.0634, 0.26894]

        drawer_T_box = tf.msg_to_kdl(cereal_pose)

        # grasp milk
        kitchen_setup.open_l_gripper()
        grasp_pose = PoseStamped()
        grasp_pose.header.frame_id = cereal_name
        grasp_pose.pose.position = Point(0.03, 0, 0.05)
        grasp_pose.pose.orientation = Quaternion(0, 0, 1, 0)
        box_T_r_goal = tf.msg_to_kdl(grasp_pose)
        box_T_r_goal_pre = deepcopy(box_T_r_goal)
        box_T_r_goal_pre.p[0] += 0.2

        pre_grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal_pre, drawer_frame_id)

        box_T_r_goal_post = deepcopy(box_T_r_goal)
        box_T_r_goal_post.p[0] += 0.4
        post_grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal_post, drawer_frame_id)

        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup.set_json_goal(u'CartesianPose',
                                    tip_link=kitchen_setup.r_tip,
                                    root_link=kitchen_setup.default_root,
                                    goal=pre_grasp_pose)
        kitchen_setup.plan_and_execute()

        grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal, drawer_frame_id)

        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup.set_cart_goal(grasp_pose,
                                    tip_link=kitchen_setup.r_tip)
        kitchen_setup.plan_and_execute()

        kitchen_setup.attach_object(cereal_name, kitchen_setup.r_tip)
        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        # kitchen_setup.close_r_gripper()

        # x = Vector3Stamped()
        # x.header.frame_id = 'milk'
        # x.vector.x = 1
        # x_map = Vector3Stamped()
        # x_map.header.frame_id = 'iai_kitchen/iai_fridge_door'
        # x_map.vector.x = 1
        # z = Vector3Stamped()
        # z.header.frame_id = 'milk'
        # z.vector.z = 1
        # z_map = Vector3Stamped()
        # z_map.header.frame_id = 'map'
        # z_map.vector.z = 1
        # kitchen_setup.align_planes('milk', x, root_normal=x_map)
        # kitchen_setup.align_planes('milk', z, root_normal=z_map)
        # kitchen_setup.keep_orientation(u'milk')
        # kitchen_setup.set_cart_goal(grasp_pose, cereal_name, kitchen_setup.default_root)
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.set_json_goal(u'CartesianPathCarrot',
                                    tip_link=kitchen_setup.r_tip,
                                    root_link=kitchen_setup.default_root,
                                    goal=post_grasp_pose,
                                    narrow=True,
                                    narrow_padding=0.2,
                                    predict_f=2.0)
        try:
            kitchen_setup.plan_and_execute()
        except Exception:
            pass
        # kitchen_setup.set_joint_goal(gaya_pose)

        # place milk back

        # kitchen_setup.add_json_goal(u'BasePointingForward')
        # milk_goal = PoseStamped()
        # milk_goal.header.frame_id = u'iai_kitchen/kitchen_island_surface'
        # milk_goal.pose.position = Point(.1, -.2, .13)
        # milk_goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0,0,1]))

    def test_ease_cereal_pick_and_place(self, kitchen_setup):
        # FIXME collision avoidance needs soft_threshholds at 0
        cereal_name = u'cereal'
        drawer_frame_id = u'iai_kitchen/oven_area_area_right_drawer_board_2_link'

        # take milk out of fridge
        kitchen_setup.set_kitchen_js({u'oven_area_area_right_drawer_main_joint': 0.48})

        # spawn milk
        cereal_pose = PoseStamped()
        cereal_pose.header.frame_id = drawer_frame_id
        cereal_pose.pose.position = Point(0.123, 0.0, 0.165)  # big box: Point(0.123, 0.0, 0.165)
        cereal_pose.pose.orientation = Quaternion(0.0087786, 0.005395, -0.838767, -0.544393)
        kitchen_setup.add_box(cereal_name, [0.1028, 0.0634, 0.26894], cereal_pose)  # big box [0.1028, 0.0634, 0.26894]

        drawer_T_box = tf.msg_to_kdl(cereal_pose)

        # grasp milk
        kitchen_setup.open_l_gripper()
        grasp_pose = PoseStamped()
        grasp_pose.header.frame_id = cereal_name
        grasp_pose.pose.position = Point(0.03, 0, 0.05)
        grasp_pose.pose.orientation = Quaternion(0, 0, 1, 0)
        box_T_r_goal = tf.msg_to_kdl(grasp_pose)
        box_T_r_goal_pre = deepcopy(box_T_r_goal)
        box_T_r_goal_pre.p[0] += 0.2

        pre_grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal_pre, drawer_frame_id)

        box_T_r_goal_post = deepcopy(box_T_r_goal)
        box_T_r_goal_post.p[0] += 0.4
        post_grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal_post, drawer_frame_id)

        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup.set_json_goal(u'CartesianPose',
                                    tip_link=kitchen_setup.r_tip,
                                    root_link=kitchen_setup.default_root,
                                    goal=pre_grasp_pose)
        kitchen_setup.plan_and_execute()

        grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal, drawer_frame_id)

        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup.set_cart_goal(grasp_pose,
                                    tip_link=kitchen_setup.r_tip)
        kitchen_setup.plan_and_execute()

        kitchen_setup.attach_object(cereal_name, kitchen_setup.r_tip)
        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        kitchen_setup.close_l_gripper()

        # x = Vector3Stamped()
        # x.header.frame_id = 'milk'
        # x.vector.x = 1
        # x_map = Vector3Stamped()
        # x_map.header.frame_id = 'iai_kitchen/iai_fridge_door'
        # x_map.vector.x = 1
        # z = Vector3Stamped()
        # z.header.frame_id = 'milk'
        # z.vector.z = 1
        # z_map = Vector3Stamped()
        # z_map.header.frame_id = 'map'
        # z_map.vector.z = 1
        # kitchen_setup.align_planes('milk', x, root_normal=x_map)
        # kitchen_setup.align_planes('milk', z, root_normal=z_map)
        # kitchen_setup.keep_orientation(u'milk')
        # kitchen_setup.set_cart_goal(grasp_pose, cereal_name, kitchen_setup.default_root)
        kitchen_setup.set_json_goal(u'CartesianPathCarrot',
                                    tip_link=kitchen_setup.r_tip,
                                    root_link=kitchen_setup.default_root,
                                    goal=post_grasp_pose)
        kitchen_setup.plan_and_execute()
        # kitchen_setup.set_joint_goal(gaya_pose)

        # place milk back

        # kitchen_setup.add_json_goal(u'BasePointingForward')
        # milk_goal = PoseStamped()
        # milk_goal.header.frame_id = u'iai_kitchen/kitchen_island_surface'
        # milk_goal.pose.position = Point(.1, -.2, .13)
        # milk_goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0,0,1]))

        kitchen_setup.set_json_goal(u'CartesianPathCarrot',
                                    tip_link=kitchen_setup.r_tip,
                                    root_link=kitchen_setup.default_root,
                                    goal=grasp_pose)
        kitchen_setup.plan_and_execute()

        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        kitchen_setup.open_l_gripper()
        kitchen_setup.detach_object(cereal_name)

        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.plan_and_execute()

    def test_cereal_stuck(self, kitchen_setup):
        from tf.transformations import quaternion_about_axis
        tj_stuck = [[1.10100000e+00, 1.45300000e+00, 1.07500000e+00, -1.22467000e-05,
                     9.79897000e-06, 5.00030000e-01, 8.66008000e-01],
                    [1.09528000e+00, 1.44342000e+00, 1.07904000e+00, 8.97380000e-03,
                     -5.89117000e-03, 4.93460000e-01, 8.69702000e-01],
                    [1.08955000e+00, 1.43384000e+00, 1.08309000e+00, 1.79583000e-02,
                     -1.17911000e-02, 4.86806000e-01, 8.73246000e-01],
                    [1.08383000e+00, 1.42426000e+00, 1.08713000e+00, 2.69397000e-02,
                     -1.76891000e-02, 4.80067000e-01, 8.76639000e-01],
                    [1.07810000e+00, 1.41468000e+00, 1.09118000e+00, 3.59165000e-02,
                     -2.35839000e-02, 4.73246000e-01, 8.79882000e-01],
                    [1.08214000e+00, 1.41881000e+00, 1.09458000e+00, 4.24803000e-02,
                     -4.04748000e-02, 4.74963000e-01, 8.78048000e-01],
                    [1.08618000e+00, 1.42294000e+00, 1.09798000e+00, 4.90299000e-02,
                     -5.73520000e-02, 4.76520000e-01, 8.75920000e-01],
                    [1.09022000e+00, 1.42707000e+00, 1.10138000e+00, 5.55632000e-02,
                     -7.42101000e-02, 4.77918000e-01, 8.73499000e-01],
                    [1.09426000e+00, 1.43121000e+00, 1.10479000e+00, 6.20778000e-02,
                     -9.10433000e-02, 4.79156000e-01, 8.70785000e-01],
                    [1.09182000e+00, 1.42320000e+00, 1.10230000e+00, 6.20418000e-02,
                     -1.05220000e-01, 4.85348000e-01, 8.65746000e-01],
                    [1.08938000e+00, 1.41519000e+00, 1.09982000e+00, 6.19893000e-02,
                     -1.19368000e-01, 4.91411000e-01, 8.60479000e-01],
                    [1.08695000e+00, 1.40719000e+00, 1.09733000e+00, 6.19205000e-02,
                     -1.33485000e-01, 4.97345000e-01, 8.54983000e-01],
                    [1.08451000e+00, 1.39918000e+00, 1.09485000e+00, 6.18352000e-02,
                     -1.47567000e-01, 5.03146000e-01, 8.49261000e-01],
                    [1.08087000e+00, 1.39999000e+00, 1.09588000e+00, 7.84897000e-02,
                     -1.58153000e-01, 4.95685000e-01, 8.50366000e-01],
                    [1.07723000e+00, 1.40081000e+00, 1.09692000e+00, 9.51091000e-02,
                     -1.68668000e-01, 4.88003000e-01, 8.51093000e-01],
                    [1.07358000e+00, 1.40162000e+00, 1.09796000e+00, 1.11686000e-01,
                     -1.79108000e-01, 4.80103000e-01, 8.51439000e-01],
                    [1.06994000e+00, 1.40244000e+00, 1.09899000e+00, 1.28213000e-01,
                     -1.89469000e-01, 4.71988000e-01, 8.51405000e-01],
                    [1.07057000e+00, 1.40270000e+00, 1.09741000e+00, 1.46094000e-01,
                     -1.91992000e-01, 4.57919000e-01, 8.55632000e-01],
                    [1.07120000e+00, 1.40297000e+00, 1.09584000e+00, 1.63896000e-01,
                     -1.94412000e-01, 4.43602000e-01, 8.59395000e-01],
                    [1.07184000e+00, 1.40323000e+00, 1.09426000e+00, 1.81610000e-01,
                     -1.96726000e-01, 4.29045000e-01, 8.62692000e-01],
                    [1.07247000e+00, 1.40349000e+00, 1.09268000e+00, 1.99224000e-01,
                     -1.98933000e-01, 4.14255000e-01, 8.65522000e-01],
                    [1.07592000e+00, 1.39947000e+00, 1.09151000e+00, 2.13779000e-01,
                     -1.87664000e-01, 4.19800000e-01, 8.61887000e-01],
                    [1.07938000e+00, 1.39544000e+00, 1.09035000e+00, 2.28251000e-01,
                     -1.76322000e-01, 4.25184000e-01, 8.57922000e-01],
                    [1.08284000e+00, 1.39141000e+00, 1.08918000e+00, 2.42636000e-01,
                     -1.64913000e-01, 4.30406000e-01, 8.53629000e-01],
                    [1.08630000e+00, 1.38739000e+00, 1.08802000e+00, 2.56928000e-01,
                     -1.53441000e-01, 4.35463000e-01, 8.49009000e-01],
                    [1.08448000e+00, 1.38159000e+00, 1.08767000e+00, 2.66198000e-01,
                     -1.68956000e-01, 4.31270000e-01, 8.45339000e-01],
                    [1.08267000e+00, 1.37579000e+00, 1.08732000e+00, 2.75372000e-01,
                     -1.84410000e-01, 4.26923000e-01, 8.41368000e-01],
                    [1.08085000e+00, 1.36999000e+00, 1.08698000e+00, 2.84448000e-01,
                     -1.99799000e-01, 4.22424000e-01, 8.37095000e-01],
                    [1.07903000e+00, 1.36419000e+00, 1.08663000e+00, 2.93423000e-01,
                     -2.15116000e-01, 4.17773000e-01, 8.32523000e-01],
                    [1.07595000e+00, 1.35221000e+00, 1.08668000e+00, 3.03484000e-01,
                     -2.09455000e-01, 4.12658000e-01, 8.32910000e-01],
                    [1.07287000e+00, 1.34023000e+00, 1.08672000e+00, 3.13497000e-01,
                     -2.03760000e-01, 4.07478000e-01, 8.33164000e-01],
                    [1.06979000e+00, 1.32826000e+00, 1.08677000e+00, 3.23461000e-01,
                     -1.98034000e-01, 4.02232000e-01, 8.33286000e-01],
                    [1.06671000e+00, 1.31628000e+00, 1.08682000e+00, 3.33372000e-01,
                     -1.92275000e-01, 3.96922000e-01, 8.33274000e-01],
                    [1.07113000e+00, 1.31434000e+00, 1.08611000e+00, 3.44694000e-01,
                     -1.95121000e-01, 3.80695000e-01, 8.35575000e-01],
                    [1.07555000e+00, 1.31241000e+00, 1.08541000e+00, 3.55876000e-01,
                     -1.97887000e-01, 3.64313000e-01, 8.37537000e-01],
                    [1.07997000e+00, 1.31047000e+00, 1.08470000e+00, 3.66914000e-01,
                     -2.00574000e-01, 3.47784000e-01, 8.39161000e-01],
                    [1.08439000e+00, 1.30854000e+00, 1.08400000e+00, 3.77803000e-01,
                     -2.03179000e-01, 3.31114000e-01, 8.40444000e-01],
                    [1.08435000e+00, 1.31897000e+00, 1.08154000e+00, 3.87174000e-01,
                     -2.10005000e-01, 3.23306000e-01, 8.37536000e-01],
                    [1.08431000e+00, 1.32941000e+00, 1.07907000e+00, 3.96466000e-01,
                     -2.16788000e-01, 3.15432000e-01, 8.34458000e-01],
                    [1.08427000e+00, 1.33984000e+00, 1.07661000e+00, 4.05678000e-01,
                     -2.23526000e-01, 3.07494000e-01, 8.31209000e-01],
                    [1.08423000e+00, 1.35028000e+00, 1.07415000e+00, 4.14806000e-01,
                     -2.30220000e-01, 2.99493000e-01, 8.27791000e-01],
                    [1.08697000e+00, 1.35193000e+00, 1.07614000e+00, 4.31185000e-01,
                     -2.38880000e-01, 2.97575000e-01, 8.17597000e-01],
                    [1.08971000e+00, 1.35357000e+00, 1.07814000e+00, 4.47369000e-01,
                     -2.47432000e-01, 2.95522000e-01, 8.07035000e-01],
                    [1.09244000e+00, 1.35522000e+00, 1.08013000e+00, 4.63351000e-01,
                     -2.55873000e-01, 2.93336000e-01, 7.96108000e-01],
                    [1.09518000e+00, 1.35686000e+00, 1.08213000e+00, 4.79125000e-01,
                     -2.64199000e-01, 2.91018000e-01, 7.84823000e-01],
                    [1.09245000e+00, 1.34438000e+00, 1.08649000e+00, 4.78703000e-01,
                     -2.78366000e-01, 2.77153000e-01, 7.85202000e-01],
                    [1.08971000e+00, 1.33190000e+00, 1.09085000e+00, 4.78093000e-01,
                     -2.92424000e-01, 2.63179000e-01, 7.85272000e-01],
                    [1.08697000e+00, 1.31941000e+00, 1.09520000e+00, 4.77295000e-01,
                     -3.06367000e-01, 2.49101000e-01, 7.85034000e-01],
                    [1.08524111e+00, 1.31072102e+00, 1.09458802e+00, 4.64751828e-01,
                     -2.98315766e-01, 2.58773769e-01, 7.92495791e-01],
                    [1.08351222e+00, 1.30203204e+00, 1.09397603e+00, 4.52036276e-01,
                     -2.90153883e-01, 2.68350549e-01, 7.99663625e-01],
                    [1.08178333e+00, 1.29334306e+00, 1.09336405e+00, 4.39152840e-01,
                     -2.81884239e-01, 2.77827665e-01, 8.06534468e-01],
                    [1.08005444e+00, 1.28465408e+00, 1.09275207e+00, 4.26106305e-01,
                     -2.73509905e-01, 2.87201597e-01, 8.13105769e-01],
                    [1.07832555e+00, 1.27596510e+00, 1.09214008e+00, 4.12901517e-01,
                     -2.65033991e-01, 2.96468864e-01, 8.19375087e-01],
                    [1.07659667e+00, 1.26727612e+00, 1.09152810e+00, 3.99543379e-01,
                     -2.56459645e-01, 3.05626025e-01, 8.25340095e-01],
                    [1.07486778e+00, 1.25858714e+00, 1.09091612e+00, 3.86036854e-01,
                     -2.47790052e-01, 3.14669677e-01, 8.30998575e-01],
                    [1.07313889e+00, 1.24989816e+00, 1.09030413e+00, 3.72386956e-01,
                     -2.39028430e-01, 3.23596464e-01, 8.36348428e-01],
                    [1.07141000e+00, 1.24120918e+00, 1.08969215e+00, 3.58598757e-01,
                     -2.30178035e-01, 3.32403068e-01, 8.41387666e-01],
                    [1.06968111e+00, 1.23252020e+00, 1.08908017e+00, 3.44677375e-01,
                     -2.21242153e-01, 3.41086220e-01, 8.46114417e-01],
                    [1.06795222e+00, 1.22383122e+00, 1.08846818e+00, 3.30627983e-01,
                     -2.12224103e-01, 3.49642694e-01, 8.50526927e-01],
                    [1.06622333e+00, 1.21514224e+00, 1.08785620e+00, 3.16455797e-01,
                     -2.03127234e-01, 3.58069314e-01, 8.54623555e-01],
                    [1.06449444e+00, 1.20645326e+00, 1.08724422e+00, 3.02166082e-01,
                     -1.93954925e-01, 3.66362948e-01, 8.58402782e-01],
                    [1.06276555e+00, 1.19776427e+00, 1.08663223e+00, 2.87764144e-01,
                     -1.84710582e-01, 3.74520517e-01, 8.61863203e-01],
                    [1.06103666e+00, 1.18907529e+00, 1.08602025e+00, 2.73255332e-01,
                     -1.75397639e-01, 3.82538992e-01, 8.65003533e-01],
                    [1.05930777e+00, 1.18038631e+00, 1.08540827e+00, 2.58645034e-01,
                     -1.66019554e-01, 3.90415394e-01, 8.67822605e-01],
                    [1.05757888e+00, 1.17169733e+00, 1.08479628e+00, 2.43938678e-01,
                     -1.56579811e-01, 3.98146798e-01, 8.70319374e-01],
                    [1.05585000e+00, 1.16300835e+00, 1.08418430e+00, 2.29141723e-01,
                     -1.47081914e-01, 4.05730332e-01, 8.72492910e-01],
                    [1.05412111e+00, 1.15431937e+00, 1.08357232e+00, 2.14259667e-01,
                     -1.37529393e-01, 4.13163181e-01, 8.74342408e-01],
                    [1.05239222e+00, 1.14563039e+00, 1.08296033e+00, 1.99298037e-01,
                     -1.27925793e-01, 4.20442583e-01, 8.75867181e-01],
                    [1.05066333e+00, 1.13694141e+00, 1.08234835e+00, 1.84262388e-01,
                     -1.18274683e-01, 4.27565835e-01, 8.77066661e-01],
                    [1.04893444e+00, 1.12825243e+00, 1.08173637e+00, 1.69158305e-01,
                     -1.08579646e-01, 4.34530292e-01, 8.77940404e-01],
                    [1.04720555e+00, 1.11956345e+00, 1.08112438e+00, 1.53991397e-01,
                     -9.88442837e-02, 4.41333367e-01, 8.78488085e-01],
                    [1.04547666e+00, 1.11087447e+00, 1.08051240e+00, 1.38767298e-01,
                     -8.90722109e-02, 4.47972533e-01, 8.78709501e-01],
                    [1.04374777e+00, 1.10218549e+00, 1.07990042e+00, 1.23491662e-01,
                     -7.92670572e-02, 4.54445325e-01, 8.78604570e-01],
                    [1.04201888e+00, 1.09349651e+00, 1.07928843e+00, 1.08170162e-01,
                     -6.94324642e-02, 4.60749339e-01, 8.78173329e-01],
                    [1.04028999e+00, 1.08480753e+00, 1.07867645e+00, 9.28084879e-02,
                     -5.95720844e-02, 4.66882232e-01, 8.77415940e-01],
                    [1.03856110e+00, 1.07611855e+00, 1.07806446e+00, 7.74123453e-02,
                     -4.96895798e-02, 4.72841729e-01, 8.76332684e-01],
                    [1.03683222e+00, 1.06742957e+00, 1.07745248e+00, 6.19874523e-02,
                     -3.97886209e-02, 4.78625615e-01, 8.74923964e-01],
                    [1.03510333e+00, 1.05874059e+00, 1.07684050e+00, 4.65395374e-02,
                     -2.98728846e-02, 4.84231742e-01, 8.73190301e-01],
                    [1.03337444e+00, 1.05005161e+00, 1.07622851e+00, 3.10743381e-02,
                     -1.99460538e-02, 4.89658028e-01, 8.71132341e-01],
                    [1.03164555e+00, 1.04136263e+00, 1.07561653e+00, 1.55975979e-02,
                     -1.00118151e-02, 4.94902459e-01, 8.68750847e-01],
                    [1.02990381e+00, 1.03260908e+00, 1.07500000e+00, 0.00000000e+00,
                     0.00000000e+00, 5.00000000e-01, 8.66025404e-01]]
        tj_terminte = [[1.10000000e+00, 1.45300000e+00, 1.07500000e+00, -1.55389000e-05,
                        -6.12874000e-06, 4.99801000e-01, 8.66140000e-01],
                       [1.09687000e+00, 1.44620000e+00, 1.07556000e+00, 7.18269000e-03,
                        1.35011000e-02, 4.92388000e-01, 8.70241000e-01],
                       [1.09373000e+00, 1.43940000e+00, 1.07613000e+00, 1.43787000e-02,
                        2.70043000e-02, 4.84825000e-01, 8.74076000e-01],
                       [1.09060000e+00, 1.43261000e+00, 1.07669000e+00, 2.15704000e-02,
                        4.04991000e-02, 4.77113000e-01, 8.77643000e-01],
                       [1.08747000e+00, 1.42581000e+00, 1.07725000e+00, 2.87554000e-02,
                        5.39816000e-02, 4.69255000e-01, 8.80942000e-01],
                       [1.08555000e+00, 1.42725000e+00, 1.07879000e+00, 4.88222000e-02,
                        5.36177000e-02, 4.60640000e-01, 8.84620000e-01],
                       [1.08364000e+00, 1.42869000e+00, 1.08034000e+00, 6.88650000e-02,
                        5.32274000e-02, 4.51800000e-01, 8.87863000e-01],
                       [1.08172000e+00, 1.43013000e+00, 1.08188000e+00, 8.88741000e-02,
                        5.28111000e-02, 4.42738000e-01, 8.90671000e-01],
                       [1.07981000e+00, 1.43157000e+00, 1.08342000e+00, 1.08840000e-01,
                        5.23688000e-02, 4.33458000e-01, 8.93043000e-01],
                       [1.07698000e+00, 1.43438000e+00, 1.08277000e+00, 1.20895000e-01,
                        4.64142000e-02, 4.18499000e-01, 8.98938000e-01],
                       [1.07415000e+00, 1.43719000e+00, 1.08211000e+00, 1.32897000e-01,
                        4.04393000e-02, 4.03356000e-01, 9.04437000e-01],
                       [1.07133000e+00, 1.44001000e+00, 1.08145000e+00, 1.44840000e-01,
                        3.44466000e-02, 3.88036000e-01, 9.09540000e-01],
                       [1.06850000e+00, 1.44282000e+00, 1.08079000e+00, 1.56720000e-01,
                        2.84387000e-02, 3.72545000e-01, 9.14243000e-01],
                       [1.07154000e+00, 1.43612000e+00, 1.07907000e+00, 1.64783000e-01,
                        2.34537000e-02, 3.58540000e-01, 9.18556000e-01],
                       [1.07457000e+00, 1.42943000e+00, 1.07736000e+00, 1.72796000e-01,
                        1.84616000e-02, 3.44426000e-01, 9.22590000e-01],
                       [1.07761000e+00, 1.42273000e+00, 1.07565000e+00, 1.80756000e-01,
                        1.34638000e-02, 3.30206000e-01, 9.26342000e-01],
                       [1.08064000e+00, 1.41604000e+00, 1.07393000e+00, 1.88661000e-01,
                        8.46191000e-03, 3.15887000e-01, 9.29812000e-01],
                       [1.07691000e+00, 1.41290000e+00, 1.07481000e+00, 2.07710000e-01,
                        3.62088000e-03, 3.15550000e-01, 9.25890000e-01],
                       [1.07319000e+00, 1.40975000e+00, 1.07569000e+00, 2.26676000e-01,
                        -1.22161000e-03, 3.15086000e-01, 9.21595000e-01],
                       [1.06946000e+00, 1.40661000e+00, 1.07656000e+00, 2.45550000e-01,
                        -6.06360000e-03, 3.14496000e-01, 9.16930000e-01],
                       [1.06573000e+00, 1.40347000e+00, 1.07744000e+00, 2.64326000e-01,
                        -1.09032000e-02, 3.13779000e-01, 9.11897000e-01],
                       [1.06659000e+00, 1.40058000e+00, 1.07609000e+00, 2.75248000e-01,
                        -2.33929000e-03, 2.97262000e-01, 9.14258000e-01],
                       [1.06745000e+00, 1.39770000e+00, 1.07475000e+00, 2.86041000e-01,
                        6.22567000e-03, 2.80605000e-01, 9.16189000e-01],
                       [1.06832000e+00, 1.39481000e+00, 1.07341000e+00, 2.96698000e-01,
                        1.47877000e-02, 2.63816000e-01, 9.17689000e-01],
                       [1.06918000e+00, 1.39193000e+00, 1.07206000e+00, 3.07216000e-01,
                        2.33428000e-02, 2.46903000e-01, 9.18756000e-01],
                       [1.06371000e+00, 1.39275000e+00, 1.07148000e+00, 3.17821000e-01,
                        1.21852000e-02, 2.35036000e-01, 9.18476000e-01],
                       [1.05824000e+00, 1.39358000e+00, 1.07089000e+00, 3.28307000e-01,
                        1.02297000e-03, 2.23081000e-01, 9.17850000e-01],
                       [1.05277000e+00, 1.39441000e+00, 1.07031000e+00, 3.38668000e-01,
                        -1.01396000e-02, 2.11041000e-01, 9.16877000e-01],
                       [1.04731000e+00, 1.39523000e+00, 1.06972000e+00, 3.48901000e-01,
                        -2.12984000e-02, 1.98922000e-01, 9.15557000e-01],
                       [1.04830000e+00, 1.39865000e+00, 1.06802000e+00, 3.62649000e-01,
                        -7.31281000e-03, 1.92227000e-01, 9.11856000e-01],
                       [1.04929000e+00, 1.40208000e+00, 1.06632000e+00, 3.76238000e-01,
                        6.67599000e-03, 1.85446000e-01, 9.07750000e-01],
                       [1.05028000e+00, 1.40550000e+00, 1.06462000e+00, 3.89659000e-01,
                        2.06618000e-02, 1.78584000e-01, 9.03242000e-01],
                       [1.05127000e+00, 1.40892000e+00, 1.06292000e+00, 4.02908000e-01,
                        3.46385000e-02, 1.71642000e-01, 8.98334000e-01],
                       [1.05084000e+00, 1.40127000e+00, 1.06317000e+00, 4.15287000e-01,
                        4.48119000e-02, 1.73248000e-01, 8.91916000e-01],
                       [1.05041000e+00, 1.39363000e+00, 1.06341000e+00, 4.27541000e-01,
                        5.49719000e-02, 1.74802000e-01, 8.85229000e-01],
                       [1.04997000e+00, 1.38598000e+00, 1.06366000e+00, 4.39667000e-01,
                        6.51153000e-02, 1.76304000e-01, 8.78277000e-01],
                       [1.04954000e+00, 1.37833000e+00, 1.06390000e+00, 4.51661000e-01,
                        7.52392000e-02, 1.77752000e-01, 8.71060000e-01],
                       [1.04436000e+00, 1.38236000e+00, 1.07034000e+00, 4.65403000e-01,
                        7.43863000e-02, 1.73145000e-01, 8.64805000e-01],
                       [1.03919000e+00, 1.38639000e+00, 1.07678000e+00, 4.79028000e-01,
                        7.35149000e-02, 1.68495000e-01, 8.58334000e-01],
                       [1.03401000e+00, 1.39042000e+00, 1.08322000e+00, 4.92534000e-01,
                        7.26250000e-02, 1.63802000e-01, 8.51648000e-01],
                       [1.02884000e+00, 1.39445000e+00, 1.08966000e+00, 5.05916000e-01,
                        7.17170000e-02, 1.59069000e-01, 8.44750000e-01],
                       [1.02812000e+00, 1.39318000e+00, 1.08781000e+00, 5.22477000e-01,
                        7.93428000e-02, 1.65104000e-01, 8.32745000e-01],
                       [1.02739000e+00, 1.39191000e+00, 1.08597000e+00, 5.38769000e-01,
                        8.69279000e-02, 1.71054000e-01, 8.20312000e-01],
                       [1.02667000e+00, 1.39064000e+00, 1.08413000e+00, 5.54785000e-01,
                        9.44685000e-02, 1.76916000e-01, 8.07459000e-01],
                       [1.02595000e+00, 1.38937000e+00, 1.08228000e+00, 5.70517000e-01,
                        1.01961000e-01, 1.82688000e-01, 7.94191000e-01],
                       [1.02273000e+00, 1.39362000e+00, 1.08152000e+00, 5.84023000e-01,
                        1.04962000e-01, 1.71079000e-01, 7.86532000e-01],
                       [1.01952000e+00, 1.39787000e+00, 1.08076000e+00, 5.97304000e-01,
                        1.07923000e-01, 1.59405000e-01, 7.78569000e-01],
                       [1.01630000e+00, 1.40212000e+00, 1.08000000e+00, 6.10356000e-01,
                        1.10842000e-01, 1.47670000e-01, 7.70308000e-01],
                       [1.01308000e+00, 1.40637000e+00, 1.07924000e+00, 6.23172000e-01,
                        1.13718000e-01, 1.35878000e-01, 7.61749000e-01],
                       [1.01343000e+00, 1.40247000e+00, 1.08272000e+00, 6.29533000e-01,
                        1.38571000e-01, 1.40845000e-01, 7.51431000e-01],
                       [1.01377000e+00, 1.39857000e+00, 1.08621000e+00, 6.35397000e-01,
                        1.63314000e-01, 1.45702000e-01, 7.40521000e-01],
                       [1.01411000e+00, 1.39467000e+00, 1.08969000e+00, 6.40759000e-01,
                        1.87928000e-01, 1.50443000e-01, 7.29025000e-01],
                       [1.01452430e+00, 1.38517261e+00, 1.08930466e+00, 6.27430015e-01,
                        1.84018746e-01, 1.61883387e-01, 7.39095695e-01],
                       [1.01493859e+00, 1.37567521e+00, 1.08891932e+00, 6.13834083e-01,
                        1.80031200e-01, 1.73254902e-01, 7.48851937e-01],
                       [1.01535289e+00, 1.36617782e+00, 1.08853398e+00, 5.99977121e-01,
                        1.75967096e-01, 1.84552741e-01, 7.58289734e-01],
                       [1.01576718e+00, 1.35668042e+00, 1.08814864e+00, 5.85865021e-01,
                        1.71828163e-01, 1.95772100e-01, 7.67405072e-01],
                       [1.01618148e+00, 1.34718303e+00, 1.08776330e+00, 5.71503784e-01,
                        1.67616160e-01, 2.06908208e-01, 7.76194074e-01],
                       [1.01659577e+00, 1.33768563e+00, 1.08737796e+00, 5.56899519e-01,
                        1.63332880e-01, 2.17956329e-01, 7.84653003e-01],
                       [1.01701007e+00, 1.32818824e+00, 1.08699262e+00, 5.42058434e-01,
                        1.58980143e-01, 2.28911764e-01, 7.92778262e-01],
                       [1.01742436e+00, 1.31869084e+00, 1.08660728e+00, 5.26986841e-01,
                        1.54559800e-01, 2.39769857e-01, 8.00566396e-01],
                       [1.01783866e+00, 1.30919345e+00, 1.08622194e+00, 5.11691149e-01,
                        1.50073732e-01, 2.50525988e-01, 8.08014092e-01],
                       [1.01825295e+00, 1.29969605e+00, 1.08583660e+00, 4.96177863e-01,
                        1.45523845e-01, 2.61175584e-01, 8.15118183e-01],
                       [1.01866725e+00, 1.29019866e+00, 1.08545125e+00, 4.80453580e-01,
                        1.40912075e-01, 2.71714116e-01, 8.21875650e-01],
                       [1.01908154e+00, 1.28070126e+00, 1.08506591e+00, 4.64524986e-01,
                        1.36240383e-01, 2.82137103e-01, 8.28283617e-01],
                       [1.01949584e+00, 1.27120387e+00, 1.08468057e+00, 4.48398854e-01,
                        1.31510755e-01, 2.92440113e-01, 8.34339361e-01],
                       [1.01991013e+00, 1.26170647e+00, 1.08429523e+00, 4.32082043e-01,
                        1.26725203e-01, 3.02618764e-01, 8.40040305e-01],
                       [1.02032443e+00, 1.25220908e+00, 1.08390989e+00, 4.15581491e-01,
                        1.21885761e-01, 3.12668727e-01, 8.45384026e-01],
                       [1.02073872e+00, 1.24271169e+00, 1.08352455e+00, 3.98904215e-01,
                        1.16994488e-01, 3.22585730e-01, 8.50368252e-01],
                       [1.02115302e+00, 1.23321429e+00, 1.08313921e+00, 3.82057307e-01,
                        1.12053464e-01, 3.32365554e-01, 8.54990862e-01],
                       [1.02156731e+00, 1.22371690e+00, 1.08275387e+00, 3.65047930e-01,
                        1.07064790e-01, 3.42004042e-01, 8.59249891e-01],
                       [1.02198161e+00, 1.21421950e+00, 1.08236853e+00, 3.47883319e-01,
                        1.02030586e-01, 3.51497093e-01, 8.63143528e-01],
                       [1.02239590e+00, 1.20472211e+00, 1.08198319e+00, 3.30570772e-01,
                        9.69529948e-02, 3.60840672e-01, 8.66670117e-01],
                       [1.02281020e+00, 1.19522471e+00, 1.08159785e+00, 3.13117651e-01,
                        9.18341746e-02, 3.70030806e-01, 8.69828158e-01],
                       [1.02322449e+00, 1.18572732e+00, 1.08121251e+00, 2.95531378e-01,
                        8.66763022e-02, 3.79063585e-01, 8.72616309e-01],
                       [1.02363879e+00, 1.17622992e+00, 1.08082717e+00, 2.77819431e-01,
                        8.14815712e-02, 3.87935169e-01, 8.75033383e-01],
                       [1.02405308e+00, 1.16673253e+00, 1.08044183e+00, 2.59989344e-01,
                        7.62521906e-02, 3.96641786e-01, 8.77078354e-01],
                       [1.02446738e+00, 1.15723513e+00, 1.08005649e+00, 2.42048697e-01,
                        7.09903840e-02, 4.05179732e-01, 8.78750351e-01],
                       [1.02488167e+00, 1.14773774e+00, 1.07967115e+00, 2.24005119e-01,
                        6.56983890e-02, 4.13545378e-01, 8.80048663e-01],
                       [1.02529597e+00, 1.13824034e+00, 1.07928581e+00, 2.05866285e-01,
                        6.03784562e-02, 4.21735165e-01, 8.80972738e-01],
                       [1.02571026e+00, 1.12874295e+00, 1.07890047e+00, 1.87639907e-01,
                        5.50328476e-02, 4.29745611e-01, 8.81522184e-01],
                       [1.02612456e+00, 1.11924555e+00, 1.07851513e+00, 1.69333736e-01,
                        4.96638366e-02, 4.37573310e-01, 8.81696766e-01],
                       [1.02653885e+00, 1.10974816e+00, 1.07812979e+00, 1.50955556e-01,
                        4.42737063e-02, 4.45214932e-01, 8.81496411e-01],
                       [1.02695315e+00, 1.10025076e+00, 1.07774445e+00, 1.32513183e-01,
                        3.88647487e-02, 4.52667229e-01, 8.80921204e-01],
                       [1.02736744e+00, 1.09075337e+00, 1.07735910e+00, 1.14014460e-01,
                        3.34392641e-02, 4.59927032e-01, 8.79971388e-01],
                       [1.02778174e+00, 1.08125598e+00, 1.07697376e+00, 9.54672525e-02,
                        2.79995596e-02, 4.66991252e-01, 8.78647369e-01],
                       [1.02819603e+00, 1.07175858e+00, 1.07658842e+00, 7.68794480e-02,
                        2.25479484e-02, 4.73856887e-01, 8.76949709e-01],
                       [1.02861033e+00, 1.06226119e+00, 1.07620308e+00, 5.82589508e-02,
                        1.70867488e-02, 4.80521016e-01, 8.74879130e-01],
                       [1.02902462e+00, 1.05276379e+00, 1.07581774e+00, 3.96136792e-02,
                        1.16182832e-02, 4.86980805e-01, 8.72436512e-01],
                       [1.02943892e+00, 1.04326640e+00, 1.07543240e+00, 2.09515622e-02,
                        6.14487690e-03, 4.93233508e-01, 8.69622895e-01],
                       [1.02990381e+00, 1.03260908e+00, 1.07500000e+00, 0.00000000e+00,
                        0.00000000e+00, 5.00000000e-01, 8.66025404e-01]]
        poses = []
        for i, point in enumerate(tj_terminte):
            base_pose = PoseStamped()
            base_pose.header.frame_id = 'map'
            base_pose.pose.position.x = point[0]
            base_pose.pose.position.y = point[1]
            base_pose.pose.position.z = point[2] if len(point) > 3 else 0
            if len(point) > 3:
                base_pose.pose.orientation = Quaternion(point[3], point[4], point[5], point[6])
            else:
                arr = quaternion_from_euler(0, 0, point[2])
                base_pose.pose.orientation = Quaternion(arr[0], arr[1], arr[2], arr[3])
            if i == 0:
                # important assumption for constraint:
                # we do not to reach the first pose, since it is the start pose
                continue
            else:
                poses.append(base_pose)
        # FIXME collision avoidance needs soft_threshholds at 0
        cereal_name = u'cereal'
        drawer_frame_id = u'iai_kitchen/oven_area_area_right_drawer_board_2_link'

        # take milk out of fridge
        kitchen_setup.set_kitchen_js({u'oven_area_area_right_drawer_main_joint': 0.48})

        # spawn milk
        cereal_pose = PoseStamped()
        cereal_pose.header.frame_id = drawer_frame_id
        cereal_pose.pose.position = Point(-0.08, 0.0, 0.15)
        cereal_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 2 - np.pi / 6, [0, 0, 1]))
        kitchen_setup.add_box(cereal_name, [0.1028, 0.0634, 0.20894], cereal_pose)

        drawer_T_box = tf.msg_to_kdl(cereal_pose)

        # grasp milk
        kitchen_setup.open_l_gripper()
        grasp_pose = PoseStamped()
        grasp_pose.header.frame_id = cereal_name
        grasp_pose.pose.position = Point(0.03, 0, 0.05)
        grasp_pose.pose.orientation = Quaternion(0, 0, 1, 0)
        box_T_r_goal = tf.msg_to_kdl(grasp_pose)
        box_T_r_goal_pre = deepcopy(box_T_r_goal)
        box_T_r_goal_pre.p[0] += 0.2

        pre_grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal_pre, drawer_frame_id)

        box_T_r_goal_post = deepcopy(box_T_r_goal)
        box_T_r_goal_post.p[0] += 0.4
        box_T_r_goal_post.p[1] += 0.15
        post_grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal_post, drawer_frame_id)

        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup.set_json_goal(u'CartesianPose',
                                    tip_link=kitchen_setup.r_tip,
                                    root_link=kitchen_setup.default_root,
                                    goal=pre_grasp_pose)
        kitchen_setup.plan_and_execute()

        grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal, drawer_frame_id)

        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup.set_cart_goal(grasp_pose,
                                    tip_link=kitchen_setup.r_tip)
        kitchen_setup.plan_and_execute()

        kitchen_setup.attach_object(cereal_name, kitchen_setup.r_tip)
        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        kitchen_setup.close_l_gripper()

        # x = Vector3Stamped()
        # x.header.frame_id = 'milk'
        # x.vector.x = 1
        # x_map = Vector3Stamped()
        # x_map.header.frame_id = 'iai_kitchen/iai_fridge_door'
        # x_map.vector.x = 1
        # z = Vector3Stamped()
        # z.header.frame_id = 'milk'
        # z.vector.z = 1
        # z_map = Vector3Stamped()
        # z_map.header.frame_id = 'map'
        # z_map.vector.z = 1
        # kitchen_setup.align_planes('milk', x, root_normal=x_map)
        # kitchen_setup.align_planes('milk', z, root_normal=z_map)
        # kitchen_setup.keep_orientation(u'milk')
        # kitchen_setup.set_cart_goal(grasp_pose, cereal_name, kitchen_setup.default_root)
        kitchen_setup.set_json_goal(u'CartesianPathCarrot',
                                    tip_link=kitchen_setup.r_tip,
                                    root_link=kitchen_setup.default_root,
                                    goal=post_grasp_pose,
                                    narrow=True,
                                    narrow_padding=0.2,
                                    goals=poses,
                                    goal_sampling_axis=[True, False, False])
        kitchen_setup.plan_and_execute()

    @pytest.mark.repeat(4)
    def test_cereal_2(self, kitchen_setup):
        from tf.transformations import quaternion_about_axis
        # FIXME collision avoidance needs soft_threshholds at 0
        cereal_name = u'cereal'
        drawer_frame_id = u'iai_kitchen/oven_area_area_right_drawer_board_2_link'

        # take milk out of fridge
        kitchen_setup.set_kitchen_js({u'oven_area_area_right_drawer_main_joint': 0.48})

        # spawn milk
        cereal_pose = PoseStamped()
        cereal_pose.header.frame_id = drawer_frame_id
        cereal_pose.pose.position = Point(-0.08, 0.0, 0.15)
        cereal_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 2 - np.pi / 6, [0, 0, 1]))
        kitchen_setup.add_box(cereal_name, [0.1028, 0.0634, 0.20894], cereal_pose)

        drawer_T_box = tf.msg_to_kdl(cereal_pose)

        # grasp milk
        kitchen_setup.open_l_gripper()
        grasp_pose = PoseStamped()
        grasp_pose.header.frame_id = cereal_name
        grasp_pose.pose.position = Point(0.03, 0, 0.05)
        grasp_pose.pose.orientation = Quaternion(0, 0, 1, 0)
        box_T_r_goal = tf.msg_to_kdl(grasp_pose)
        box_T_r_goal_pre = deepcopy(box_T_r_goal)
        box_T_r_goal_pre.p[0] += 0.2

        pre_grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal_pre, drawer_frame_id)

        box_T_r_goal_post = deepcopy(box_T_r_goal)
        box_T_r_goal_post.p[0] += 0.4
        box_T_r_goal_post.p[1] += 0.15
        post_grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal_post, drawer_frame_id)

        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup.set_json_goal(u'CartesianPose',
                                    tip_link=kitchen_setup.r_tip,
                                    root_link=kitchen_setup.default_root,
                                    goal=pre_grasp_pose)
        kitchen_setup.plan_and_execute()

        grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal, drawer_frame_id)

        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup.set_cart_goal(grasp_pose,
                                    tip_link=kitchen_setup.r_tip)
        kitchen_setup.plan_and_execute()

        kitchen_setup.attach_object(cereal_name, kitchen_setup.r_tip)
        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        kitchen_setup.close_l_gripper()

        # x = Vector3Stamped()
        # x.header.frame_id = 'milk'
        # x.vector.x = 1
        # x_map = Vector3Stamped()
        # x_map.header.frame_id = 'iai_kitchen/iai_fridge_door'
        # x_map.vector.x = 1
        # z = Vector3Stamped()
        # z.header.frame_id = 'milk'
        # z.vector.z = 1
        # z_map = Vector3Stamped()
        # z_map.header.frame_id = 'map'
        # z_map.vector.z = 1
        # kitchen_setup.align_planes('milk', x, root_normal=x_map)
        # kitchen_setup.align_planes('milk', z, root_normal=z_map)
        # kitchen_setup.keep_orientation(u'milk')
        # kitchen_setup.set_cart_goal(grasp_pose, cereal_name, kitchen_setup.default_root)
        kitchen_setup.set_json_goal(u'CartesianPathCarrot',
                                    tip_link=kitchen_setup.r_tip,
                                    root_link=kitchen_setup.default_root,
                                    goal=post_grasp_pose,
                                    narrow=True,
                                    narrow_padding=0.2,
                                    goal_sampling_axis=[True, False, False])
        kitchen_setup.plan()
        # kitchen_setup.set_joint_goal(gaya_pose)

        # place milk back

        # kitchen_setup.add_json_goal(u'BasePointingForward')
        # milk_goal = PoseStamped()
        # milk_goal.header.frame_id = u'iai_kitchen/kitchen_island_surface'
        # milk_goal.pose.position = Point(.1, -.2, .13)
        # milk_goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0,0,1]))

        # kitchen_setup.set_json_goal(u'CartesianPathCarrot',
        #                            tip_link=kitchen_setup.r_tip,
        #                            root_link=kitchen_setup.default_root,
        #                            goal=grasp_pose)
        # kitchen_setup.plan_and_execute()

        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        # kitchen_setup.open_l_gripper()
        # kitchen_setup.detach_object(cereal_name)

    # kitchen_setup.set_joint_goal(gaya_pose)
    # kitchen_setup.plan_and_execute()

    def test_ease_cereal_different_drawers(self, kitchen_setup):
        # FIXME collision avoidance needs soft_threshholds at 0
        cereal_name = u'cereal'
        drawer_frame_id = 'iai_kitchen/oven_area_area_right_drawer_board_1_link'

        # take milk out of fridge
        kitchen_setup.set_kitchen_js({u'oven_area_area_right_drawer_main_joint': 0.48})

        # spawn milk

        cereal_pose = PoseStamped()
        cereal_pose.header.frame_id = drawer_frame_id
        cereal_pose.pose.position = Point(0.123, 0.0, 0.13)
        cereal_pose.pose.orientation = Quaternion(0.0087786, 0.005395, -0.838767, -0.544393)
        kitchen_setup.add_box(cereal_name, [0.1028, 0.0634, 0.20894], cereal_pose)

        drawer_T_box = tf.msg_to_kdl(cereal_pose)

        # grasp milk
        kitchen_setup.open_l_gripper()
        grasp_pose = PoseStamped()
        grasp_pose.header.frame_id = cereal_name
        grasp_pose.pose.position = Point(0.13, 0, 0.05)
        grasp_pose.pose.orientation = Quaternion(0, 0, 1, 0)
        box_T_r_goal = tf.msg_to_kdl(grasp_pose)
        box_T_r_goal_pre = deepcopy(box_T_r_goal)
        box_T_r_goal_pre.p[0] += 0.1

        pre_grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal_pre, drawer_frame_id)

        box_T_r_goal_post = deepcopy(box_T_r_goal)
        box_T_r_goal_post.p[0] += 0.4
        if drawer_frame_id == 'iai_kitchen/oven_area_area_right_drawer_board_1_link':
            box_T_r_goal_post.p[1] -= 0.15
            box_T_r_goal_post.p[2] += 0.2
        post_grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal_post, drawer_frame_id)

        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup.set_json_goal(u'CartesianPose',
                                    tip_link=kitchen_setup.r_tip,
                                    root_link=kitchen_setup.default_root,
                                    goal=pre_grasp_pose)
        kitchen_setup.plan_and_execute()

        grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal, drawer_frame_id)

        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup.set_cart_goal(grasp_pose,
                                    tip_link=kitchen_setup.r_tip)
        kitchen_setup.plan_and_execute()

        kitchen_setup.attach_object(cereal_name, kitchen_setup.r_tip)
        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        kitchen_setup.close_l_gripper()

        # x = Vector3Stamped()
        # x.header.frame_id = 'milk'
        # x.vector.x = 1
        # x_map = Vector3Stamped()
        # x_map.header.frame_id = 'iai_kitchen/iai_fridge_door'
        # x_map.vector.x = 1
        # z = Vector3Stamped()
        # z.header.frame_id = 'milk'
        # z.vector.z = 1
        # z_map = Vector3Stamped()
        # z_map.header.frame_id = 'map'
        # z_map.vector.z = 1
        # kitchen_setup.align_planes('milk', x, root_normal=x_map)
        # kitchen_setup.align_planes('milk', z, root_normal=z_map)
        # kitchen_setup.keep_orientation(u'milk')
        # kitchen_setup.set_cart_goal(grasp_pose, cereal_name, kitchen_setup.default_root)
        kitchen_setup.set_json_goal(u'CartesianPose',
                                    tip_link=kitchen_setup.r_tip,
                                    root_link=kitchen_setup.default_root,
                                    goal=post_grasp_pose)
        kitchen_setup.plan_and_execute()
        # kitchen_setup.set_joint_goal(gaya_pose)

        # place milk back
        drawer_frame_id = u'iai_kitchen/oven_area_area_right_drawer_board_2_link'
        cereal_pose = PoseStamped()
        cereal_pose.header.frame_id = drawer_frame_id
        cereal_pose.pose.position = Point(0.123, 0.0, 0.13)
        cereal_pose.pose.orientation = Quaternion(0.0087786, 0.005395, -0.838767, -0.544393)

        drawer_T_box = tf.msg_to_kdl(cereal_pose)

        grasp_pose = PoseStamped()
        grasp_pose.header.frame_id = cereal_name
        grasp_pose.pose.position = Point(0.13, 0, 0.05)
        grasp_pose.pose.orientation = Quaternion(0, 0, 1, 0)

        box_T_r_goal = tf.msg_to_kdl(grasp_pose)

        grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal, drawer_frame_id)

        # kitchen_setup.add_json_goal(u'BasePointingForward')
        # milk_goal = PoseStamped()
        # milk_goal.header.frame_id = u'iai_kitchen/kitchen_island_surface'
        # milk_goal.pose.position = Point(.1, -.2, .13)
        # milk_goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0,0,1]))

        kitchen_setup.set_json_goal(u'CartesianPose',
                                    tip_link=kitchen_setup.r_tip,
                                    root_link=kitchen_setup.default_root,
                                    goal=grasp_pose)
        kitchen_setup.plan_and_execute()

        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        kitchen_setup.open_l_gripper()
        kitchen_setup.detach_object(cereal_name)

        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.plan_and_execute()


class TestShaking(object):
    def test_wiggle_prismatic_joint_neglectable_shaking(self, kitchen_setup: PR2):
        # FIXME
        sample_period = kitchen_setup.god_map.get_data(identifier.sample_period)
        frequency_range = kitchen_setup.god_map.get_data(identifier.frequency_range)
        amplitude_threshold = kitchen_setup.god_map.get_data(identifier.amplitude_threshold)
        max_detectable_freq = int(1 / (2 * sample_period))
        min_wiggle_frequency = int(frequency_range * max_detectable_freq)
        while np.fmod(min_wiggle_frequency, 5.0) != 0.0:
            min_wiggle_frequency += 1
        distance_between_frequencies = 5

        for i, t in enumerate([('torso_lift_joint', 0.05), ('odom_x_joint', 0.5)]):  # max vel: 0.015 and 0.5
            for f in range(min_wiggle_frequency, max_detectable_freq, distance_between_frequencies):
                target_freq = float(f)
                joint = t[0]
                goal = t[1]
                kitchen_setup.set_json_goal('JointPositionPrismatic',
                                            joint_name=joint,
                                            goal=0.0,
                                            )
                kitchen_setup.plan_and_execute()
                kitchen_setup.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
                kitchen_setup.set_json_goal('ShakyJointPositionRevoluteOrPrismatic',
                                            joint_name=joint,
                                            noise_amplitude=amplitude_threshold - 0.05,
                                            goal=goal,
                                            frequency=target_freq
                                            )
                kitchen_setup.plan_and_execute()

    def test_wiggle_revolute_joint_neglectable_shaking(self, kitchen_setup: PR2):
        # FIXME
        sample_period = kitchen_setup.god_map.get_data(identifier.sample_period)
        frequency_range = kitchen_setup.god_map.get_data(identifier.frequency_range)
        amplitude_threshold = kitchen_setup.god_map.get_data(identifier.amplitude_threshold)
        max_detectable_freq = int(1 / (2 * sample_period))
        min_wiggle_frequency = int(frequency_range * max_detectable_freq)
        while np.fmod(min_wiggle_frequency, 5.0) != 0.0:
            min_wiggle_frequency += 1
        distance_between_frequencies = 5

        for i, joint in enumerate(['r_wrist_flex_joint', 'head_pan_joint']):  # max vel: 1.0 and 0.5
            for f in range(min_wiggle_frequency, max_detectable_freq, distance_between_frequencies):
                target_freq = float(f)
                kitchen_setup.set_json_goal('JointPositionRevolute',
                                            joint_name=joint,
                                            goal=0.0,
                                            )
                kitchen_setup.plan_and_execute()
                kitchen_setup.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
                kitchen_setup.set_json_goal('ShakyJointPositionRevoluteOrPrismatic',
                                            joint_name=joint,
                                            noise_amplitude=amplitude_threshold - 0.05,
                                            goal=-1.0,
                                            frequency=target_freq
                                            )
                kitchen_setup.plan_and_execute()

    def test_wiggle_continuous_joint_neglectable_shaking(self, kitchen_setup: PR2):
        sample_period = kitchen_setup.god_map.get_data(identifier.sample_period)
        frequency_range = kitchen_setup.god_map.get_data(identifier.frequency_range)
        amplitude_threshold = kitchen_setup.god_map.get_data(identifier.amplitude_threshold)
        max_detectable_freq = int(1 / (2 * sample_period))
        min_wiggle_frequency = int(frequency_range * max_detectable_freq)
        while np.fmod(min_wiggle_frequency, 5.0) != 0.0:
            min_wiggle_frequency += 1
        distance_between_frequencies = 5

        for continuous_joint in ['l_wrist_roll_joint', 'r_forearm_roll_joint']:  # max vel. of 1.0 and 1.0
            for f in range(min_wiggle_frequency, max_detectable_freq, distance_between_frequencies):
                kitchen_setup.set_json_goal('JointPositionContinuous',
                                            joint_name=continuous_joint,
                                            goal=5.0,
                                            )
                kitchen_setup.plan_and_execute()
                target_freq = float(f)
                kitchen_setup.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
                kitchen_setup.set_json_goal('ShakyJointPositionContinuous',
                                            joint_name=continuous_joint,
                                            goal=-5.0,
                                            noise_amplitude=amplitude_threshold - 0.05,
                                            frequency=target_freq
                                            )
                kitchen_setup.plan_and_execute()

    def test_wiggle_revolute_joint_shaking(self, kitchen_setup: PR2):
        sample_period = kitchen_setup.god_map.get_data(identifier.sample_period)
        frequency_range = kitchen_setup.god_map.get_data(identifier.frequency_range)
        max_detectable_freq = int(1 / (2 * sample_period))
        min_wiggle_frequency = int(frequency_range * max_detectable_freq)
        while np.fmod(min_wiggle_frequency, 5.0) != 0.0:
            min_wiggle_frequency += 1
        distance_between_frequencies = 5

        for joint in ['head_pan_joint', 'r_wrist_flex_joint']:  # max vel: 1.0 and 0.5
            for f in range(min_wiggle_frequency, max_detectable_freq, distance_between_frequencies):
                kitchen_setup.set_json_goal('JointPositionRevolute',
                                            joint_name=joint,
                                            goal=0.5,
                                            )
                kitchen_setup.plan_and_execute()
                target_freq = float(f)
                kitchen_setup.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
                kitchen_setup.set_json_goal('ShakyJointPositionRevoluteOrPrismatic',
                                            joint_name=joint,
                                            goal=0.0,
                                            frequency=target_freq
                                            )
                r = kitchen_setup.plan_and_execute(expected_error_codes=[MoveResult.SHAKING])
                assert len(r.error_codes) != 0
                error_code = r.error_codes[0]
                assert error_code == MoveResult.SHAKING
                error_message = r.error_messages[0]
                freqs_str = re.findall("[0-9]+\.[0-9]+ hertz", error_message)
                assert any(map(lambda f_str: float(f_str[:-6]) == target_freq, freqs_str))

    def test_wiggle_prismatic_joint_shaking(self, kitchen_setup: PR2):
        sample_period = kitchen_setup.god_map.get_data(identifier.sample_period)
        frequency_range = kitchen_setup.god_map.get_data(identifier.frequency_range)
        max_detectable_freq = int(1 / (2 * sample_period))
        min_wiggle_frequency = int(frequency_range * max_detectable_freq)
        while np.fmod(min_wiggle_frequency, 5.0) != 0.0:
            min_wiggle_frequency += 1
        distance_between_frequencies = 5

        for joint in ['odom_x_joint']:  # , 'torso_lift_joint']: # max vel: 0.015 and 0.5
            for f in range(min_wiggle_frequency, max_detectable_freq, distance_between_frequencies):
                kitchen_setup.set_json_goal('JointPositionPrismatic',
                                            joint_name=joint,
                                            goal=0.02,
                                            )
                kitchen_setup.plan_and_execute()
                target_freq = float(f)
                kitchen_setup.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
                kitchen_setup.set_json_goal('ShakyJointPositionRevoluteOrPrismatic',
                                            joint_name=joint,
                                            goal=0.0,
                                            frequency=target_freq
                                            )
                r = kitchen_setup.plan_and_execute(expected_error_codes=[MoveResult.SHAKING])
                assert len(r.error_codes) != 0
                error_code = r.error_codes[0]
                assert error_code == MoveResult.SHAKING
                error_message = r.error_messages[0]
                freqs_str = re.findall("[0-9]+\.[0-9]+ hertz", error_message)
                assert any(map(lambda f_str: float(f_str[:-6]) == target_freq, freqs_str))

    def test_wiggle_continuous_joint_shaking(self, kitchen_setup: PR2):
        sample_period = kitchen_setup.god_map.get_data(identifier.sample_period)
        frequency_range = kitchen_setup.god_map.get_data(identifier.frequency_range)
        max_detectable_freq = int(1 / (2 * sample_period))
        min_wiggle_frequency = int(frequency_range * max_detectable_freq)
        while np.fmod(min_wiggle_frequency, 5.0) != 0.0:
            min_wiggle_frequency += 1
        distance_between_frequencies = 5

        for continuous_joint in ['l_wrist_roll_joint', 'r_forearm_roll_joint']:  # max vel. of 1.0 and 1.0
            for f in range(min_wiggle_frequency, max_detectable_freq, distance_between_frequencies):
                kitchen_setup.set_json_goal('JointPositionContinuous',
                                            joint_name=continuous_joint,
                                            goal=5.0,
                                            )
                kitchen_setup.plan_and_execute()
                target_freq = float(f)
                kitchen_setup.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
                kitchen_setup.set_json_goal('ShakyJointPositionContinuous',
                                            joint_name=continuous_joint,
                                            goal=-5.0,
                                            frequency=target_freq
                                            )
                r = kitchen_setup.plan_and_execute(expected_error_codes=[MoveResult.SHAKING])
                assert len(r.error_codes) != 0
                error_code = r.error_codes[0]
                assert error_code == MoveResult.SHAKING
                error_message = r.error_messages[0]
                freqs_str = re.findall("[0-9]+\.[0-9]+ hertz", error_message)
                assert any(map(lambda f_str: float(f_str[:-6]) == target_freq, freqs_str))

    def test_only_revolute_joint_shaking(self, kitchen_setup: PR2):
        sample_period = kitchen_setup.god_map.get_data(identifier.sample_period)
        frequency_range = kitchen_setup.god_map.get_data(identifier.frequency_range)
        amplitude_threshold = kitchen_setup.god_map.get_data(identifier.amplitude_threshold)
        max_detectable_freq = int(1 / (2 * sample_period))
        min_wiggle_frequency = int(frequency_range * max_detectable_freq)
        while np.fmod(min_wiggle_frequency, 5.0) != 0.0:
            min_wiggle_frequency += 1
        distance_between_frequencies = 5

        for revolute_joint in ['r_wrist_flex_joint', 'head_pan_joint']:  # max vel. of 1.0 and 1.0
            for f in range(min_wiggle_frequency, max_detectable_freq, distance_between_frequencies):
                target_freq = float(f)

                if f == min_wiggle_frequency:
                    kitchen_setup.set_json_goal('JointPositionRevolute',
                                                joint_name=revolute_joint,
                                                goal=0.0,
                                                )
                    kitchen_setup.plan_and_execute()

                kitchen_setup.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
                kitchen_setup.set_json_goal('ShakyJointPositionRevoluteOrPrismatic',
                                            joint_name=revolute_joint,
                                            goal=0.0,
                                            noise_amplitude=amplitude_threshold + 0.02,
                                            frequency=target_freq
                                            )
                r = kitchen_setup.plan_and_execute(expected_error_codes=[MoveResult.SHAKING])
                assert len(r.error_codes) != 0
                error_code = r.error_codes[0]
                assert error_code == MoveResult.SHAKING
                error_message = r.error_messages[0]
                freqs_str = re.findall("[0-9]+\.[0-9]+ hertz", error_message)
                assert any(map(lambda f_str: float(f_str[:-6]) == target_freq, freqs_str))

    def test_only_revolute_joint_neglectable_shaking(self, kitchen_setup: PR2):
        # FIXME
        sample_period = kitchen_setup.god_map.get_data(identifier.sample_period)
        frequency_range = kitchen_setup.god_map.get_data(identifier.frequency_range)
        amplitude_threshold = kitchen_setup.god_map.get_data(identifier.amplitude_threshold)
        max_detectable_freq = int(1 / (2 * sample_period))
        min_wiggle_frequency = int(frequency_range * max_detectable_freq)
        while np.fmod(min_wiggle_frequency, 5.0) != 0.0:
            min_wiggle_frequency += 1
        distance_between_frequencies = 5

        for revolute_joint in ['r_wrist_flex_joint', 'head_pan_joint']:  # max vel. of 1.0 and 0.5
            for f in range(min_wiggle_frequency, max_detectable_freq, distance_between_frequencies):
                target_freq = float(f)
                if f == min_wiggle_frequency:
                    kitchen_setup.set_json_goal('JointPositionRevolute',
                                                joint_name=revolute_joint,
                                                goal=0.0,
                                                )
                    kitchen_setup.plan_and_execute()
                kitchen_setup.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
                kitchen_setup.set_json_goal('ShakyJointPositionRevoluteOrPrismatic',
                                            joint_name=revolute_joint,
                                            goal=0.0,
                                            noise_amplitude=amplitude_threshold - 0.02,
                                            frequency=target_freq
                                            )
                r = kitchen_setup.plan_and_execute()
                if any(map(lambda c: c == MoveResult.SHAKING, r.error_codes)):
                    error_message = r.error_messages[0]
                    freqs_str = re.findall("[0-9]+\.[0-9]+ hertz", error_message)
                    assert all(map(lambda f_str: float(f_str[:-6]) != target_freq, freqs_str))
                else:
                    assert True


class TestWorldManipulation(object):

    def test_dye_group(self, kitchen_setup: PR2):
        kitchen_setup.dye_group(kitchen_setup.get_robot_name(), (1,0,0,1))
        kitchen_setup.dye_group('kitchen', (0,1,0,1))
        kitchen_setup.dye_group(kitchen_setup.r_gripper_group, (0,0,1,1))
        kitchen_setup.set_joint_goal(kitchen_setup.default_pose)
        kitchen_setup.plan_and_execute()

    def test_clear_world(self, zero_pose: PR2):
        object_name = 'muh'
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position = Point(1.2, 0, 1.6)
        p.pose.orientation = Quaternion(0.0, 0.0, 0.47942554, 0.87758256)
        zero_pose.add_box(object_name, size=(1, 1, 1), pose=p)
        zero_pose.clear_world()
        object_name = 'muh2'
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position = Point(1.2, 0, 1.6)
        p.pose.orientation = Quaternion(0.0, 0.0, 0.47942554, 0.87758256)
        zero_pose.add_box(object_name, size=(1, 1, 1), pose=p)
        zero_pose.clear_world()
        zero_pose.plan_and_execute()

    def test_attach_remove_box(self, better_pose: PR2):
        pocky = 'http://muh#pocky'
        p = PoseStamped()
        p.header.frame_id = better_pose.r_tip
        p.pose.orientation.w = 1
        better_pose.add_box(pocky, size=(1, 1, 1), pose=p)
        for i in range(3):
            better_pose.update_parent_link_of_group(name=pocky, parent_link=better_pose.r_tip)
            better_pose.detach_group(pocky)
        better_pose.remove_group(pocky)

    def test_reattach_box(self, zero_pose: PR2):
        pocky = 'http://muh#pocky'
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.position = Point(0.05, 0, 0)
        p.pose.orientation = Quaternion(0., 0., 0.47942554, 0.87758256)
        zero_pose.add_box(pocky, (0.1, 0.02, 0.02), pose=p)
        zero_pose.update_parent_link_of_group(pocky, parent_link=zero_pose.r_tip)
        relative_pose = zero_pose.robot.compute_fk_pose(zero_pose.r_tip, pocky).pose
        compare_poses(p.pose, relative_pose)

    def test_add_box_twice(self, zero_pose: PR2):
        object_name = 'muh'
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position = Point(1.2, 0, 1.6)
        p.pose.orientation = Quaternion(0.0, 0.0, 0.47942554, 0.87758256)
        zero_pose.add_box(object_name, size=(1, 1, 1), pose=p)
        zero_pose.add_box(object_name, size=(1, 1, 1), pose=p,
                          expected_error_code=UpdateWorldResponse.DUPLICATE_GROUP_ERROR)

    def test_add_remove_sphere(self, zero_pose: PR2):
        object_name = 'muh'
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position.x = 1.2
        p.pose.position.y = 0
        p.pose.position.z = 1.6
        p.pose.orientation.w = 1
        zero_pose.add_sphere(object_name, radius=1, pose=p)
        zero_pose.remove_group(object_name)

    def test_add_remove_cylinder(self, zero_pose: PR2):
        object_name = 'muh'
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position.x = 0.5
        p.pose.position.y = 0
        p.pose.position.z = 0
        p.pose.orientation.w = 1
        zero_pose.add_cylinder(object_name, height=1, radius=1, pose=p)
        zero_pose.remove_group(object_name)

    def test_add_urdf_body(self, kitchen_setup: PR2):
        object_name = 'kitchen'
        kitchen_setup.clear_world()
        kitchen_setup.add_urdf(name=object_name,
                               urdf=rospy.get_param('kitchen_description'),
                               pose=tf.lookup_pose('map', 'iai_kitchen/world'),
                               js_topic='/kitchen/joint_states',
                               set_js_topic='/kitchen/cram_joint_states')
        kitchen_setup.remove_group(object_name)
        kitchen_setup.add_urdf(name=object_name,
                               urdf=rospy.get_param('kitchen_description'),
                               pose=tf.lookup_pose('map', 'iai_kitchen/world'),
                               js_topic='/kitchen/joint_states',
                               set_js_topic='/kitchen/cram_joint_states')

    def test_add_mesh(self, zero_pose: PR2):
        object_name = 'muh'
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.add_mesh(object_name, mesh='package://giskardpy/test/urdfs/meshes/bowl_21.obj', pose=p)

    def test_add_non_existing_mesh(self, zero_pose: PR2):
        object_name = 'muh'
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.add_mesh(object_name, mesh='package://giskardpy/test/urdfs/meshes/muh.obj', pose=p,
                           expected_error_code=UpdateWorldResponse.CORRUPT_MESH_ERROR)

    def test_add_attach_detach_remove_add(self, zero_pose: PR2):
        object_name = 'muh'
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position = Point(1.2, 0, 1.6)
        p.pose.orientation = Quaternion(0.0, 0.0, 0.47942554, 0.87758256)
        zero_pose.add_box(object_name, size=(1, 1, 1), pose=p, timeout=0)
        zero_pose.update_parent_link_of_group(object_name, parent_link=zero_pose.r_tip, timeout=0)
        zero_pose.detach_group(object_name, timeout=0)
        zero_pose.remove_group(object_name, timeout=0)
        zero_pose.add_box(object_name, size=(1, 1, 1), pose=p, timeout=0)

    def test_attach_to_kitchen(self, kitchen_setup: PR2):
        object_name = 'muh'
        drawer_joint = 'sink_area_left_middle_drawer_main_joint'

        cup_pose = PoseStamped()
        cup_pose.header.frame_id = 'iai_kitchen/sink_area_left_middle_drawer_main'
        cup_pose.pose.position = Point(0.1, 0.2, -.05)
        cup_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup.add_cylinder(object_name, height=0.07, radius=0.04, pose=cup_pose, parent_link_group='kitchen',
                                   parent_link='sink_area_left_middle_drawer_main')
        kitchen_setup.set_kitchen_js({drawer_joint: 0.48})
        kitchen_setup.plan_and_execute()
        kitchen_setup.detach_group(object_name)
        kitchen_setup.set_kitchen_js({drawer_joint: 0})
        kitchen_setup.plan_and_execute()

    def test_update_group_pose1(self, zero_pose: PR2):
        group_name = 'muh'
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position = Point(1.2, 0, 1.6)
        p.pose.orientation = Quaternion(0.0, 0.0, 0.47942554, 0.87758256)
        zero_pose.add_box(group_name, size=(1, 1, 1), pose=p)
        p.pose.position = Point(1, 0, 0)
        zero_pose.update_group_pose('asdf', p, expected_error_code=UpdateWorldResponse.UNKNOWN_GROUP_ERROR)
        zero_pose.update_group_pose(group_name, p)

    def test_update_group_pose2(self, zero_pose: PR2):
        group_name = 'muh'
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position = Point(1.2, 0, 1.6)
        p.pose.orientation = Quaternion(0.0, 0.0, 0.47942554, 0.87758256)
        zero_pose.add_box(group_name, size=(1, 1, 1), pose=p, parent_link='r_gripper_tool_frame')
        p.pose.position = Point(1, 0, 0)
        zero_pose.update_group_pose('asdf', p, expected_error_code=UpdateWorldResponse.UNKNOWN_GROUP_ERROR)
        zero_pose.update_group_pose(group_name, p)
        zero_pose.set_joint_goal(zero_pose.better_pose)
        # TODO test that attached object moved?
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

    def test_attach_existing_box2(self, zero_pose: PR2):
        pocky = 'http://muh#pocky'
        old_p = PoseStamped()
        old_p.header.frame_id = zero_pose.r_tip
        old_p.pose.position = Point(0.05, 0, 0)
        old_p.pose.orientation = Quaternion(0., 0., 0.47942554, 0.87758256)
        zero_pose.add_box(pocky, (0.1, 0.02, 0.02), pose=old_p)
        zero_pose.update_parent_link_of_group(pocky, parent_link=zero_pose.r_tip)
        relative_pose = zero_pose.robot.compute_fk_pose(zero_pose.r_tip, pocky).pose
        compare_poses(old_p.pose, relative_pose)

        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.position.x = -0.1
        p.pose.orientation.w = 1.0
        zero_pose.set_cart_goal(p, zero_pose.r_tip, zero_pose.default_root)
        zero_pose.plan_and_execute()
        p.header.frame_id = 'map'
        p.pose.position.y = -1
        p.pose.orientation = Quaternion(0, 0, 0.47942554, 0.87758256)
        zero_pose.move_base(p)
        rospy.sleep(.5)

        zero_pose.detach_group(pocky)

    def test_attach_to_nonexistant_robot_link(self, zero_pose: PR2):
        pocky = 'http://muh#pocky'
        p = PoseStamped()
        zero_pose.add_box(name=pocky,
                          size=(0.1, 0.02, 0.02),
                          pose=p,
                          parent_link='muh',
                          expected_error_code=UpdateWorldResponse.UNKNOWN_LINK_ERROR)

    def test_reattach_unknown_object(self, zero_pose: PR2):
        zero_pose.update_parent_link_of_group('muh',
                                              parent_link='',
                                              parent_link_group='',
                                              expected_response=UpdateWorldResponse.UNKNOWN_GROUP_ERROR)

    def test_add_remove_box(self, zero_pose: PR2):
        object_name = 'muh'
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position.x = 1.2
        p.pose.position.y = 0
        p.pose.position.z = 1.6
        p.pose.orientation.w = 1
        zero_pose.add_box(object_name, size=(1, 1, 1), pose=p)
        zero_pose.remove_group(object_name)

    def test_invalid_update_world(self, zero_pose: PR2):
        req = UpdateWorldRequest()
        req.timeout = 500
        req.body = WorldBody()
        req.pose = PoseStamped()
        req.parent_link = zero_pose.r_tip
        req.operation = 42
        assert zero_pose._update_world_srv.call(req).error_codes == UpdateWorldResponse.INVALID_OPERATION

    def test_remove_unkown_group(self, zero_pose: PR2):
        zero_pose.remove_group('muh', expected_response=UpdateWorldResponse.UNKNOWN_GROUP_ERROR)

    def test_corrupt_shape_error(self, zero_pose: PR2):
        p = PoseStamped()
        p.header.frame_id = 'base_link'
        req = UpdateWorldRequest()
        req.body = WorldBody(type=WorldBody.PRIMITIVE_BODY,
                             shape=SolidPrimitive(type=42))
        req.pose = PoseStamped()
        req.pose.header.frame_id = 'map'
        req.parent_link = 'base_link'
        req.operation = UpdateWorldRequest.ADD
        assert zero_pose._update_world_srv.call(req).error_codes == UpdateWorldResponse.CORRUPT_SHAPE_ERROR

    def test_tf_error(self, zero_pose: PR2):
        req = UpdateWorldRequest()
        req.body = WorldBody(type=WorldBody.PRIMITIVE_BODY,
                             shape=SolidPrimitive(type=1))
        req.pose = PoseStamped()
        req.parent_link = 'base_link'
        req.operation = UpdateWorldRequest.ADD
        assert zero_pose._update_world_srv.call(req).error_codes == UpdateWorldResponse.TF_ERROR

    def test_unsupported_options(self, kitchen_setup: PR2):
        wb = WorldBody()
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = str('base_link')
        pose.pose.position = Point()
        pose.pose.orientation = Quaternion(w=1)
        wb.type = WorldBody.URDF_BODY

        req = UpdateWorldRequest()
        req.body = wb
        req.pose = pose
        req.parent_link = 'base_link'
        req.operation = UpdateWorldRequest.ADD
        assert kitchen_setup._update_world_srv.call(req).error_codes == UpdateWorldResponse.CORRUPT_URDF_ERROR


class TestCollisionAvoidanceGoals(object):

    def test_handover(self, kitchen_setup: PR2):
        js = {
            "l_shoulder_pan_joint": 1.0252138037286773,
            "l_shoulder_lift_joint": - 0.06966848987919201,
            "l_upper_arm_roll_joint": 1.1765832782526544,
            "l_elbow_flex_joint": - 1.9323726623855864,
            "l_forearm_roll_joint": 1.3824994377973336,
            "l_wrist_flex_joint": - 1.8416233909065576,
            "l_wrist_roll_joint": 2.907373693068033,
        }
        kitchen_setup.set_joint_goal(js)
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()

        p = PoseStamped()
        p.header.frame_id = kitchen_setup.l_tip
        p.pose.position.y = -0.08
        p.pose.orientation.w = 1
        kitchen_setup.add_box(name='box',
                              size=(0.08, 0.16, 0.16),
                              parent_link=kitchen_setup.l_tip,
                              parent_link_group=kitchen_setup.get_robot_name(),
                              pose=p)
        kitchen_setup.close_l_gripper()
        r_goal = PoseStamped()
        r_goal.header.frame_id = kitchen_setup.l_tip
        r_goal.pose.position.x = 0.05
        r_goal.pose.position.y = -0.08
        r_goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0, 0, 1]))
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.set_cart_goal(r_goal,
                                    tip_link=kitchen_setup.r_tip,
                                    root_link=kitchen_setup.l_tip,
                                    linear_velocity=0.2,
                                    angular_velocity=1
                                    )
        kitchen_setup.allow_collision(group1=kitchen_setup.get_robot_name(), group2='box')
        kitchen_setup.plan_and_execute()

        kitchen_setup.update_parent_link_of_group('box', kitchen_setup.r_tip)

        r_goal2 = PoseStamped()
        r_goal2.header.frame_id = 'box'
        r_goal2.pose.position.x -= -.1
        r_goal2.pose.orientation.w = 1

        kitchen_setup.set_cart_goal(r_goal2, 'box', root_link=kitchen_setup.l_tip)
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()
        # kitchen_setup.check_cart_goal('box', r_goal2)

    def test_only_collision_avoidance(self, zero_pose: PR2):
        zero_pose.plan_and_execute()

    def test_mesh_collision_avoidance(self, zero_pose: PR2):
        zero_pose.close_r_gripper()
        object_name = 'muh'
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.position = Point(0.01, 0, 0)
        p.pose.orientation = Quaternion(*quaternion_about_axis(-np.pi / 2, [0, 1, 0]))
        zero_pose.add_mesh(object_name, mesh='package://giskardpy/test/urdfs/meshes/bowl_21.obj', pose=p)
        zero_pose.plan_and_execute()

    def test_attach_box_as_eef(self, zero_pose: PR2):
        pocky = 'http://muh#pocky'
        box_pose = PoseStamped()
        box_pose.header.frame_id = zero_pose.r_tip
        box_pose.pose.position = Point(0.05, 0, 0, )
        box_pose.pose.orientation = Quaternion(1, 0, 0, 0)
        zero_pose.add_box(name=pocky, size=(0.1, 0.02, 0.02), pose=box_pose, parent_link=zero_pose.r_tip,
                          parent_link_group=zero_pose.get_robot_name())
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.orientation.w = 1
        zero_pose.set_cart_goal(p, pocky, zero_pose.default_root)
        p = tf.transform_pose(zero_pose.default_root, p)
        zero_pose.plan_and_execute()
        p2 = zero_pose.robot.compute_fk_pose(zero_pose.default_root, pocky)
        compare_poses(p2.pose, p.pose)
        zero_pose.detach_group(pocky)
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.orientation.w = 1
        p.pose.position.x = -.1
        zero_pose.set_cart_goal(p, zero_pose.r_tip, zero_pose.default_root)
        zero_pose.plan_and_execute()

    def test_infeasible(self, kitchen_setup: PR2):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position = Point(2, 0, 0)
        pose.pose.orientation = Quaternion(w=1)
        kitchen_setup.teleport_base(pose)
        kitchen_setup.plan_and_execute(expected_error_codes=[MoveResult.HARD_CONSTRAINTS_VIOLATED])

    def test_unknown_group1(self, box_setup: PR2):
        box_setup.avoid_collision(min_distance=0.05, group1='muh')
        box_setup.plan_and_execute([MoveResult.UNKNOWN_GROUP])

    def test_unknown_group2(self, box_setup: PR2):
        box_setup.avoid_collision(group2='muh')
        box_setup.plan_and_execute([MoveResult.UNKNOWN_GROUP])

    def test_base_link_in_collision(self, zero_pose: PR2):
        zero_pose.allow_self_collision()
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position.x = 0
        p.pose.position.y = 0
        p.pose.position.z = -0.2
        p.pose.orientation.w = 1
        zero_pose.add_box(name='box', size=(1, 1, 1), pose=p)
        zero_pose.set_joint_goal(pocky_pose)
        zero_pose.plan_and_execute()

    def test_allow_self_collision(self, zero_pose: PR2):
        zero_pose.check_cpi_geq(zero_pose.get_r_gripper_links(), 0.05)
        zero_pose.check_cpi_geq(zero_pose.get_l_gripper_links(), 0.05)

    def test_allow_self_collision2(self, zero_pose: PR2):
        goal_js = {
            'l_elbow_flex_joint': -1.43286344265,
            'l_forearm_roll_joint': 1.26465060073,
            'l_shoulder_lift_joint': 0.47990329056,
            'l_shoulder_pan_joint': 0.281272240139,
            'l_upper_arm_roll_joint': 0.528415402668,
            'l_wrist_flex_joint': -1.18811419869,
            'l_wrist_roll_joint': 2.26884630124,
        }
        zero_pose.allow_all_collisions()
        zero_pose.set_joint_goal(goal_js)
        zero_pose.plan_and_execute()

        p = PoseStamped()
        p.header.frame_id = zero_pose.l_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.2
        p.pose.orientation.w = 1
        zero_pose.allow_self_collision()
        zero_pose.set_cart_goal(p, zero_pose.l_tip, zero_pose.default_root)
        zero_pose.plan_and_execute()
        zero_pose.check_cpi_leq(zero_pose.get_l_gripper_links(), 0.01)
        zero_pose.check_cpi_leq(['r_forearm_link'], 0.01)
        zero_pose.check_cpi_geq(zero_pose.get_r_gripper_links(), 0.05)

    def test_allow_self_collision3(self, zero_pose: PR2):
        # fixme
        goal_js = {
            'l_elbow_flex_joint': -1.43286344265,
            'l_forearm_roll_joint': 1.26465060073,
            'l_shoulder_lift_joint': 0.47990329056,
            'l_shoulder_pan_joint': 0.281272240139,
            'l_upper_arm_roll_joint': 0.528415402668,
            'l_wrist_flex_joint': -1.18811419869,
            'l_wrist_roll_joint': 2.26884630124,
        }
        zero_pose.allow_all_collisions()
        zero_pose.set_joint_goal(goal_js)
        zero_pose.plan_and_execute()

        p = PoseStamped()
        p.header.frame_id = zero_pose.l_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.18
        p.pose.position.z = 0.02
        p.pose.orientation.w = 1

        zero_pose.allow_self_collision()

        zero_pose.set_cart_goal(p, zero_pose.l_tip, 'base_footprint')
        zero_pose.plan_and_execute()
        zero_pose.check_cpi_leq(zero_pose.get_l_gripper_links(), 0.01)
        zero_pose.check_cpi_leq(['r_forearm_link'], 0.01)
        zero_pose.check_cpi_geq(zero_pose.get_r_gripper_links(), 0.05)

    def test_avoid_self_collision(self, zero_pose: PR2):
        goal_js = {
            'l_elbow_flex_joint': -1.43286344265,
            'l_forearm_roll_joint': 1.26465060073,
            'l_shoulder_lift_joint': 0.47990329056,
            'l_shoulder_pan_joint': 0.281272240139,
            'l_upper_arm_roll_joint': 0.528415402668,
            'l_wrist_flex_joint': -1.18811419869,
            'l_wrist_roll_joint': 2.26884630124,
        }
        zero_pose.allow_all_collisions()
        zero_pose.set_joint_goal(goal_js)
        zero_pose.plan_and_execute()

        p = PoseStamped()
        p.header.frame_id = zero_pose.l_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.2
        p.pose.orientation.w = 1
        zero_pose.set_cart_goal(p, zero_pose.l_tip, 'base_footprint')
        zero_pose.plan_and_execute()
        zero_pose.check_cpi_geq(zero_pose.get_l_gripper_links(), 0.048)

    def test_avoid_self_collision2(self, zero_pose: PR2):
        goal_js = {
            'r_elbow_flex_joint': -1.43286344265,
            'r_forearm_roll_joint': -1.26465060073,
            'r_shoulder_lift_joint': 0.47990329056,
            'r_shoulder_pan_joint': -0.281272240139,
            'r_upper_arm_roll_joint': -0.528415402668,
            'r_wrist_flex_joint': -1.18811419869,
            'r_wrist_roll_joint': 2.26884630124,
        }
        zero_pose.allow_all_collisions()
        zero_pose.set_joint_goal(goal_js)
        zero_pose.plan_and_execute()

        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.2
        p.pose.orientation.w = 1
        zero_pose.set_cart_goal(p, zero_pose.r_tip, zero_pose.default_root)
        zero_pose.send_goal()
        zero_pose.check_cpi_geq(zero_pose.get_r_gripper_links(), 0.048)

    def test_avoid_self_collision3(self, zero_pose: PR2):
        goal_js = {
            'r_shoulder_pan_joint': -0.0672581793019,
            'r_shoulder_lift_joint': 0.429650469244,
            'r_upper_arm_roll_joint': -0.580889703636,
            'r_forearm_roll_joint': -101.948215412,
            'r_elbow_flex_joint': -1.35221928696,
            'r_wrist_flex_joint': -0.986144640142,
            'r_wrist_roll_joint': 2.31051794404,
        }
        zero_pose.allow_all_collisions()
        zero_pose.set_joint_goal(goal_js)
        zero_pose.plan_and_execute()

        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = -0.2
        p.pose.orientation.w = 1
        zero_pose.set_cart_goal(p, zero_pose.r_tip, zero_pose.default_root)
        zero_pose.plan_and_execute()
        zero_pose.check_cpi_geq(zero_pose.get_r_gripper_links(), 0.048)

    def test_get_out_of_self_collision(self, zero_pose: PR2):
        goal_js = {
            'l_elbow_flex_joint': -1.43286344265,
            'l_forearm_roll_joint': 1.26465060073,
            'l_shoulder_lift_joint': 0.47990329056,
            'l_shoulder_pan_joint': 0.281272240139,
            'l_upper_arm_roll_joint': 0.528415402668,
            'l_wrist_flex_joint': -1.18811419869,
            'l_wrist_roll_joint': 2.26884630124,
        }
        zero_pose.allow_all_collisions()
        zero_pose.set_joint_goal(goal_js)
        zero_pose.plan_and_execute()

        p = PoseStamped()
        p.header.frame_id = zero_pose.l_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.15
        p.pose.orientation.w = 1
        zero_pose.set_cart_goal(p, zero_pose.l_tip, 'base_footprint')
        zero_pose.allow_all_collisions()
        zero_pose.send_goal()
        zero_pose.send_goal()
        zero_pose.check_cpi_geq(zero_pose.get_l_gripper_links(), 0.048)

    def test_avoid_collision(self, box_setup: PR2):
        box_setup.avoid_collision(min_distance=0.05, group1=box_setup.get_robot_name())
        box_setup.avoid_collision(min_distance=0.15, group1=box_setup.l_gripper_group, group2='box')
        box_setup.avoid_collision(min_distance=0.10, group1=box_setup.r_gripper_group, group2='box')
        box_setup.allow_self_collision()
        box_setup.plan_and_execute()
        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.148)
        box_setup.check_cpi_geq(box_setup.get_r_gripper_links(), 0.088)

    def test_collision_override(self, box_setup: PR2):
        # FIXME
        p = PoseStamped()
        p.header.frame_id = box_setup.default_root
        p.pose.position.x += 0.5
        p.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0, 0, 1]))
        box_setup.teleport_base(p)
        # ce = CollisionEntry()
        # ce.type = CollisionEntry.AVOID_COLLISION
        # ce.body_b = 'box'
        # ce.min_dist = 0.05
        # box_setup.add_collision_entries([ce])
        box_setup.plan_and_execute()
        box_setup.check_cpi_geq(['base_link'], 0.099)

    def test_avoid_collision2(self, fake_table_setup: PR2):
        r_goal = PoseStamped()
        r_goal.header.frame_id = 'map'
        r_goal.pose.position.x = 0.8
        r_goal.pose.position.y = -0.38
        r_goal.pose.position.z = 0.84
        r_goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 2, [0, 1, 0]))
        # fake_table_setup.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
        fake_table_setup.avoid_all_collisions(0.1)
        fake_table_setup.set_cart_goal(r_goal, fake_table_setup.r_tip)
        fake_table_setup.plan_and_execute()
        fake_table_setup.check_cpi_geq(fake_table_setup.get_l_gripper_links(), 0.05)
        fake_table_setup.check_cpi_leq(['r_gripper_l_finger_tip_link'], 0.04)
        fake_table_setup.check_cpi_leq(['r_gripper_r_finger_tip_link'], 0.04)

    def test_allow_collision(self, box_setup: PR2):
        p = PoseStamped()
        p.header.frame_id = 'base_footprint'
        p.header.stamp = rospy.get_rostime()
        p.pose.position = Point(0.15, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)

        box_setup.allow_collision(group2='box')

        box_setup.allow_self_collision()
        box_setup.set_cart_goal(p, 'base_footprint', box_setup.default_root)
        box_setup.plan_and_execute()

        box_setup.check_cpi_leq(box_setup.get_l_gripper_links(), 0.0)
        box_setup.check_cpi_leq(box_setup.get_r_gripper_links(), 0.0)

    def test_avoid_collision3(self, pocky_pose_setup: PR2):
        p = PoseStamped()
        p.header.frame_id = pocky_pose_setup.r_tip
        p.pose.position.x = 0.08
        p.pose.orientation.w = 1
        pocky_pose_setup.add_box(name='box',
                                 size=(0.2, 0.05, 0.05),
                                 parent_link=pocky_pose_setup.r_tip,
                                 pose=p)
        p = PoseStamped()
        p.header.frame_id = pocky_pose_setup.r_tip
        p.pose.position.x = 0.15
        p.pose.position.y = 0.04
        p.pose.position.z = 0
        p.pose.orientation.w = 1
        pocky_pose_setup.add_box('bl', (0.1, 0.01, 0.2), pose=p)
        p = PoseStamped()
        p.header.frame_id = pocky_pose_setup.r_tip
        p.pose.position.x = 0.15
        p.pose.position.y = -0.04
        p.pose.position.z = 0
        p.pose.orientation.w = 1
        pocky_pose_setup.add_box('br', (0.1, 0.01, 0.2), pose=p)

        p = PoseStamped()
        p.header.frame_id = pocky_pose_setup.r_tip
        p.pose.position = Point(-0.15, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        pocky_pose_setup.set_cart_goal(p, pocky_pose_setup.r_tip, pocky_pose_setup.default_root)

        pocky_pose_setup.allow_self_collision()

        pocky_pose_setup.plan_and_execute()
        # TODO check traj length?
        pocky_pose_setup.check_cpi_geq(['box'], 0.048)

    def test_avoid_collision4(self, pocky_pose_setup: PR2):
        p = PoseStamped()
        p.header.frame_id = pocky_pose_setup.r_tip
        p.pose.position.x = 0.08
        p.pose.orientation.w = 1
        pocky_pose_setup.add_box(name='box',
                                 size=(0.2, 0.05, 0.05),
                                 parent_link=pocky_pose_setup.r_tip,
                                 pose=p)
        p = PoseStamped()
        p.header.frame_id = pocky_pose_setup.r_tip
        p.pose.position.x = 0.2
        p.pose.position.y = 0
        p.pose.position.z = 0
        p.pose.orientation.w = 1
        pocky_pose_setup.add_box('b1', (0.01, 0.2, 0.2), pose=p)
        p = PoseStamped()
        p.header.frame_id = pocky_pose_setup.r_tip
        p.pose.position.x = 0.15
        p.pose.position.y = 0.04
        p.pose.position.z = 0
        p.pose.orientation.w = 1
        pocky_pose_setup.add_box('bl', (0.1, 0.01, 0.2), pose=p)
        p = PoseStamped()
        p.header.frame_id = pocky_pose_setup.r_tip
        p.pose.position.x = 0.15
        p.pose.position.y = -0.04
        p.pose.position.z = 0
        p.pose.orientation.w = 1
        pocky_pose_setup.add_box('br', (0.1, 0.01, 0.2), pose=p)

        # p = PoseStamped()
        # p.header.frame_id = pocky_pose_setup.r_tip
        # p.pose.position = Point(-0.15, 0, 0)
        # p.pose.orientation = Quaternion(0, 0, 0, 1)
        # pocky_pose_setup.set_cart_goal(p, pocky_pose_setup.r_tip, pocky_pose_setup.default_root)
        x = Vector3Stamped()
        x.header.frame_id = 'box'
        x.vector.x = 1
        y = Vector3Stamped()
        y.header.frame_id = 'box'
        y.vector.y = 1
        x_map = Vector3Stamped()
        x_map.header.frame_id = 'map'
        x_map.vector.x = 1
        y_map = Vector3Stamped()
        y_map.header.frame_id = 'map'
        y_map.vector.y = 1
        pocky_pose_setup.set_align_planes_goal('box', x, root_normal=x_map)
        pocky_pose_setup.set_align_planes_goal('box', y, root_normal=y_map)
        pocky_pose_setup.allow_self_collision()
        # pocky_pose_setup.allow_all_collisions()

        pocky_pose_setup.plan_and_execute()
        assert ('box', 'bl') not in pocky_pose_setup.collision_scene.black_list
        pocky_pose_setup.check_cpi_geq(pocky_pose_setup.get_group_info('r_gripper').links, 0.04)

    def test_avoid_collision_two_sticks(self, pocky_pose_setup: PR2):
        p = PoseStamped()
        p.header.frame_id = pocky_pose_setup.r_tip
        p.pose.position.x = 0.08
        p.pose.orientation = Quaternion(*quaternion_about_axis(0.01, [1, 0, 0]).tolist())
        pocky_pose_setup.add_box(name='box',
                                 size=(0.2, 0.05, 0.05),
                                 parent_link=pocky_pose_setup.r_tip,
                                 pose=p)
        p = PoseStamped()
        p.header.frame_id = pocky_pose_setup.r_tip
        p.pose.position.x = 0.12
        p.pose.position.y = 0.04
        p.pose.position.z = 0
        p.pose.orientation.w = 1
        pocky_pose_setup.add_cylinder('bl', height=0.2, radius=0.01, pose=p)
        p = PoseStamped()
        p.header.frame_id = pocky_pose_setup.r_tip
        p.pose.position.x = 0.12
        p.pose.position.y = -0.04
        p.pose.position.z = 0
        p.pose.orientation.w = 1
        pocky_pose_setup.add_cylinder('br', height=0.2, radius=0.01, pose=p)
        pocky_pose_setup.allow_self_collision()
        pocky_pose_setup.plan_and_execute()

    def test_avoid_collision5_cut_off(self, pocky_pose_setup: PR2):
        p = PoseStamped()
        p.header.frame_id = pocky_pose_setup.r_tip
        p.pose.position.x = 0.08
        p.pose.orientation = Quaternion(*quaternion_about_axis(0.01, [1, 0, 0]).tolist())
        pocky_pose_setup.add_box(name='box',
                                 size=(0.2, 0.05, 0.05),
                                 parent_link=pocky_pose_setup.r_tip,
                                 pose=p)
        p = PoseStamped()
        p.header.frame_id = pocky_pose_setup.r_tip
        p.pose.position.x = 0.12
        p.pose.position.y = 0.04
        p.pose.position.z = 0
        p.pose.orientation.w = 1
        pocky_pose_setup.add_cylinder('bl', height=0.2, radius=0.01, pose=p)
        p = PoseStamped()
        p.header.frame_id = pocky_pose_setup.r_tip
        p.pose.position.x = 0.12
        p.pose.position.y = -0.04
        p.pose.position.z = 0
        p.pose.orientation.w = 1
        pocky_pose_setup.add_cylinder('br', height=0.2, radius=0.01, pose=p)

        pocky_pose_setup.send_goal(goal_type=MoveGoal.PLAN_AND_EXECUTE_AND_CUT_OFF_SHAKING)

    # def test_avoid_collision6(self, fake_table_setup: PR2):
    #     #fixme
    #     js = {
    #         'r_shoulder_pan_joint': -0.341482794236,
    #         'r_shoulder_lift_joint': 0.0301123643508,
    #         'r_upper_arm_roll_joint': -2.67555547662,
    #         'r_forearm_roll_joint': -0.472653283346,
    #         'r_elbow_flex_joint': -0.149999999999,
    #         'r_wrist_flex_joint': -1.40685144215,
    #         'r_wrist_roll_joint': 2.87855178783,
    #         'odom_x_joint': 0.0708087929675,
    #         'odom_y_joint': 0.052896931145,
    #         'odom_z_joint': 0.0105784287694,
    #         'torso_lift_joint': 0.277729421077,
    #     }
    #     # fake_table_setup.allow_all_collisions()
    #     fake_table_setup.send_and_check_joint_goal(js, weight=WEIGHT_ABOVE_CA)
    #     fake_table_setup.check_cpi_geq(fake_table_setup.get_l_gripper_links(), 0.048)
    #     fake_table_setup.check_cpi_leq(['r_gripper_l_finger_tip_link'], 0.04)
    #     fake_table_setup.check_cpi_leq(['r_gripper_r_finger_tip_link'], 0.04)

    def test_avoid_collision7(self, kitchen_setup: PR2):
        base_pose = PoseStamped()
        base_pose.header.frame_id = 'map'
        base_pose.pose.position.x = 0.8
        base_pose.pose.position.y = 1
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        kitchen_setup.teleport_base(base_pose)
        base_pose = PoseStamped()
        base_pose.header.frame_id = 'map'
        base_pose.pose.position.x = 0.64
        base_pose.pose.position.y = 0.64
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(-np.pi / 4, [0, 0, 1]))
        kitchen_setup.set_joint_goal(kitchen_setup.better_pose)
        kitchen_setup.set_cart_goal(base_pose, 'base_footprint')
        kitchen_setup.plan_and_execute()

    def test_avoid_collision_at_kitchen_corner(self, kitchen_setup: PR2):
        base_pose = PoseStamped()
        base_pose.header.stamp = rospy.get_rostime()
        base_pose.header.frame_id = 'map'
        base_pose.pose.position.x = 0.75
        base_pose.pose.position.y = 0.9
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 2, [0, 0, 1]))
        kitchen_setup.teleport_base(base_pose)
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0, 0, 1]))
        kitchen_setup.set_joint_goal(kitchen_setup.better_pose)  # , weight=WEIGHT_ABOVE_CA)
        kitchen_setup.set_rotation_goal(base_pose, 'base_footprint')
        kitchen_setup.set_translation_goal(base_pose, 'base_footprint', weight=WEIGHT_BELOW_CA)
        kitchen_setup.plan_and_execute()

    def test_avoid_collision8(self, kitchen_setup: PR2):
        base_pose = PoseStamped()
        base_pose.header.frame_id = 'map'
        base_pose.pose.position.x = 0.8
        base_pose.pose.position.y = 0.9
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        kitchen_setup.teleport_base(base_pose)
        base_pose = PoseStamped()
        base_pose.header.frame_id = 'map'
        base_pose.pose.position.x = 0.64
        base_pose.pose.position.y = 0.64
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(-np.pi / 4, [0, 0, 1]))
        kitchen_setup.set_joint_goal(kitchen_setup.better_pose)
        kitchen_setup.set_cart_goal(base_pose, 'base_footprint')
        kitchen_setup.plan_and_execute()

    def test_avoid_collision_drive_under_drawer(self, kitchen_setup: PR2):
        kitchen_js = {'sink_area_left_middle_drawer_main_joint': 0.45}
        kitchen_setup.set_kitchen_js(kitchen_js)
        base_pose = PoseStamped()
        base_pose.header.frame_id = 'map'
        base_pose.pose.position.x = 0.57
        base_pose.pose.position.y = 0.5
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        kitchen_setup.teleport_base(base_pose)
        base_pose = PoseStamped()
        base_pose.header.frame_id = 'base_footprint'
        base_pose.pose.position.y = 1
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        kitchen_setup.set_cart_goal(base_pose, tip_link='base_footprint')
        kitchen_setup.plan_and_execute()

    def test_avoid_collision_with_far_object(self, pocky_pose_setup: PR2):
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position.x = 25
        p.pose.position.y = 25
        p.pose.position.z = 25
        p.pose.orientation.w = 1
        pocky_pose_setup.add_box(name='box', size=(1, 1, 1), pose=p)
        p = PoseStamped()
        p.header.frame_id = pocky_pose_setup.r_tip
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        pocky_pose_setup.set_cart_goal(p, pocky_pose_setup.r_tip, pocky_pose_setup.default_root)

        pocky_pose_setup.avoid_collision(0.05, pocky_pose_setup.get_robot_name(), 'box')

        pocky_pose_setup.plan_and_execute()
        pocky_pose_setup.check_cpi_geq(pocky_pose_setup.get_l_gripper_links(), 0.048)
        pocky_pose_setup.check_cpi_geq(pocky_pose_setup.get_r_gripper_links(), 0.048)

    def test_avoid_collision_touch(self, box_setup: PR2):
        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        box_setup.set_cart_goal(p, box_setup.r_tip, box_setup.default_root, check=False)

        box_setup.avoid_all_collisions(0.05)

        box_setup.plan_and_execute()

        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.0)
        box_setup.check_cpi_geq(box_setup.get_r_gripper_links(), -0.008)
        box_setup.check_cpi_leq(box_setup.get_r_gripper_links(), 0.04)

    def test_get_out_of_collision(self, box_setup: PR2):
        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.pose.position = Point(0.15, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        box_setup.set_cart_goal(p, box_setup.r_tip, box_setup.default_root)

        box_setup.allow_all_collisions()

        box_setup.plan_and_execute()

        box_setup.avoid_all_collisions(0.05)

        box_setup.plan_and_execute()

        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.0)
        box_setup.check_cpi_geq(box_setup.get_r_gripper_links(), 0.0)

    def test_allow_collision_gripper(self, box_setup: PR2):
        box_setup.allow_collision(box_setup.l_gripper_group, 'box')
        p = PoseStamped()
        p.header.frame_id = box_setup.l_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.11
        p.pose.orientation.w = 1
        box_setup.set_cart_goal(p, box_setup.l_tip, box_setup.default_root)
        box_setup.plan_and_execute()
        box_setup.check_cpi_leq(box_setup.get_l_gripper_links(), 0.0)
        box_setup.check_cpi_geq(box_setup.get_r_gripper_links(), 0.048)

    def test_attached_get_below_soft_threshold(self, box_setup: PR2):
        attached_link_name = 'pocky'
        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.pose.position.x = 0.05
        p.pose.orientation.w = 1
        box_setup.add_box(attached_link_name,
                          size=(0.2, 0.04, 0.04),
                          parent_link=box_setup.r_tip,
                          pose=p)
        box_setup.add_box(attached_link_name,
                          size=(0.2, 0.04, 0.04),
                          parent_link=box_setup.r_tip,
                          pose=p,
                          expected_error_code=UpdateWorldResponse.DUPLICATE_GROUP_ERROR)
        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = -0.15
        p.pose.orientation.w = 1
        box_setup.set_cart_goal(p, box_setup.r_tip, box_setup.default_root)
        box_setup.plan_and_execute()
        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.048)
        box_setup.check_cpi_geq([attached_link_name], 0.048)

        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.1
        p.pose.orientation.w = 1
        box_setup.set_cart_goal(p, box_setup.r_tip, box_setup.default_root, check=False)
        box_setup.plan_and_execute()
        box_setup.check_cpi_geq([attached_link_name], -0.008)
        box_setup.check_cpi_leq([attached_link_name], 0.01)
        box_setup.detach_group(attached_link_name)

    def test_attached_get_out_of_collision_below(self, box_setup: PR2):
        attached_link_name = 'pocky'
        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.pose.position.x = 0.05
        p.pose.orientation.w = 1
        box_setup.add_box(attached_link_name,
                          size=(0.2, 0.04, 0.04),
                          parent_link=box_setup.r_tip,
                          pose=p)
        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = -0.15
        p.pose.orientation.w = 1
        box_setup.set_cart_goal(p, box_setup.r_tip, box_setup.default_root)
        box_setup.plan_and_execute()
        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.048)
        box_setup.check_cpi_geq([attached_link_name], 0.048)

        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.05
        p.pose.orientation.w = 1
        box_setup.set_cart_goal(p, box_setup.r_tip, box_setup.default_root, weight=WEIGHT_BELOW_CA, check=False)
        box_setup.plan_and_execute()
        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.048)
        box_setup.check_cpi_geq([attached_link_name], 0.048)
        box_setup.detach_group(attached_link_name)

    def test_attached_get_out_of_collision_and_stay_in_hard_threshold(self, box_setup: PR2):
        attached_link_name = 'pocky'
        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.pose.position.x = 0.05
        p.pose.orientation.w = 1
        box_setup.add_box(attached_link_name,
                          size=(0.2, 0.04, 0.04),
                          parent_link=box_setup.r_tip,
                          pose=p)
        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = -0.08
        p.pose.orientation.w = 1
        box_setup.set_cart_goal(p, box_setup.r_tip, box_setup.default_root)
        box_setup.plan_and_execute()
        box_setup.check_cpi_geq([attached_link_name], -1e-3)

        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.08
        p.pose.orientation.w = 1
        box_setup.set_cart_goal(p, box_setup.r_tip, box_setup.default_root, check=False)
        box_setup.plan_and_execute()
        box_setup.check_cpi_geq([attached_link_name], -0.002)
        box_setup.check_cpi_leq([attached_link_name], 0.01)
        box_setup.detach_group(attached_link_name)

    def test_attached_get_out_of_collision_stay_in(self, box_setup: PR2):
        attached_link_name = 'pocky'
        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.pose.position.x = 0.05
        p.pose.orientation.w = 1
        box_setup.add_box(attached_link_name,
                          size=(0.2, 0.04, 0.04),
                          parent_link=box_setup.r_tip,
                          pose=p)
        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.
        p.pose.orientation.w = 1
        box_setup.set_cart_goal(p, box_setup.r_tip, box_setup.default_root)
        box_setup.plan_and_execute()
        box_setup.check_cpi_geq([attached_link_name], -0.082)
        box_setup.detach_group(attached_link_name)

    def test_attached_get_out_of_collision_passive(self, box_setup: PR2):
        attached_link_name = 'pocky'
        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.pose.position.x = 0.05
        p.pose.orientation.w = 1
        box_setup.add_box(attached_link_name,
                          size=(0.2, 0.04, 0.04),
                          parent_link=box_setup.r_tip,
                          pose=p)
        box_setup.plan_and_execute()
        box_setup.check_cpi_geq([attached_link_name], 0.049)
        box_setup.detach_group(attached_link_name)

    def test_attached_collision2(self, box_setup: PR2):
        # FIXME
        attached_link_name = 'pocky'
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position.x = 0.01
        p.pose.orientation.w = 1
        box_setup.add_box(attached_link_name,
                          size=(0.2, 0.04, 0.04),
                          parent_link=box_setup.r_tip,
                          pose=p)
        box_setup.plan_and_execute()
        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.048)
        box_setup.check_cpi_geq([attached_link_name], 0.048)
        box_setup.detach_group(attached_link_name)

    def test_attached_self_collision(self, zero_pose: PR2):
        collision_pose = {
            'l_elbow_flex_joint': - 1.1343683863086362,
            'l_forearm_roll_joint': 7.517553513504836,
            'l_shoulder_lift_joint': 0.5726770101613905,
            'l_shoulder_pan_joint': 0.1592669164939349,
            'l_upper_arm_roll_joint': 0.5532568387077381,
            'l_wrist_flex_joint': - 1.215660155912625,
            'l_wrist_roll_joint': 4.249300323527076,
            'torso_lift_joint': 0.2}

        zero_pose.set_joint_goal(collision_pose)
        zero_pose.plan_and_execute()

        attached_link_name = 'pocky'
        p = PoseStamped()
        p.header.frame_id = zero_pose.l_tip
        p.pose.position.x = 0.04
        p.pose.orientation.w = 1
        zero_pose.add_box(attached_link_name,
                          size=(0.16, 0.04, 0.04),
                          parent_link=zero_pose.l_tip,
                          pose=p)

        # zero_pose.set_prediction_horizon(1)
        zero_pose.set_joint_goal({'r_forearm_roll_joint': 0.0,
                                  'r_shoulder_lift_joint': 0.0,
                                  'r_shoulder_pan_joint': 0.0,
                                  'r_upper_arm_roll_joint': 0.0,
                                  'r_wrist_flex_joint': -0.10001,
                                  'r_wrist_roll_joint': 0.0,
                                  'r_elbow_flex_joint': -0.15,
                                  'torso_lift_joint': 0.2})

        p = PoseStamped()
        p.header.frame_id = zero_pose.l_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.z = 0.20
        p.pose.orientation.w = 1
        zero_pose.set_cart_goal(p, zero_pose.l_tip, zero_pose.default_root)
        zero_pose.plan_and_execute()

        zero_pose.check_cpi_geq(zero_pose.get_l_gripper_links(), 0.048)
        zero_pose.check_cpi_geq([attached_link_name], 0.048)
        zero_pose.detach_group(attached_link_name)

    def test_attached_self_collision2(self, zero_pose: PR2):
        collision_pose = {
            'r_elbow_flex_joint': - 1.1343683863086362,
            'r_forearm_roll_joint': -7.517553513504836,
            'r_shoulder_lift_joint': 0.5726770101613905,
            'r_shoulder_pan_joint': -0.1592669164939349,
            'r_upper_arm_roll_joint': -0.5532568387077381,
            'r_wrist_flex_joint': - 1.215660155912625,
            'r_wrist_roll_joint': -4.249300323527076,
            'torso_lift_joint': 0.2
        }

        zero_pose.set_joint_goal(collision_pose)
        zero_pose.plan_and_execute()

        attached_link_name = 'box'
        p = PoseStamped()
        p.header.frame_id = zero_pose.l_tip
        p.pose.position.x = 0.04
        p.pose.orientation.w = 1
        zero_pose.add_box(attached_link_name,
                          size=(0.16, 0.04, 0.04),
                          parent_link=zero_pose.l_tip,
                          pose=p)

        js_goal = {'l_forearm_roll_joint': 0.0,
                   'l_shoulder_lift_joint': 0.0,
                   'odom_x_joint': 0.0,
                   'odom_y_joint': 0.0,
                   'odom_z_joint': 0.0,
                   'l_shoulder_pan_joint': 0.0,
                   'l_upper_arm_roll_joint': 0.0,
                   'l_wrist_flex_joint': -0.11,
                   'l_wrist_roll_joint': 0.0,
                   'l_elbow_flex_joint': -0.16,
                   'torso_lift_joint': 0.2}
        zero_pose.set_joint_goal(js_goal)

        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.z = 0.20
        p.pose.orientation.w = 1
        zero_pose.set_cart_goal(p, zero_pose.r_tip, zero_pose.default_root)
        zero_pose.plan_and_execute()

        zero_pose.check_cpi_geq(zero_pose.get_r_gripper_links(), 0.048)
        zero_pose.check_cpi_geq([attached_link_name], 0.048)
        zero_pose.detach_group(attached_link_name)

    def test_attached_self_collision3(self, zero_pose: PR2):
        collision_pose = {
            'l_elbow_flex_joint': - 1.1343683863086362,
            'l_forearm_roll_joint': 7.517553513504836,
            'l_shoulder_lift_joint': 0.5726770101613905,
            'l_shoulder_pan_joint': 0.1592669164939349,
            'l_upper_arm_roll_joint': 0.5532568387077381,
            'l_wrist_flex_joint': - 1.215660155912625,
            'l_wrist_roll_joint': 4.249300323527076,
            'torso_lift_joint': 0.2}

        zero_pose.set_joint_goal(collision_pose)
        zero_pose.plan_and_execute()

        attached_link_name = 'pocky'
        p = PoseStamped()
        p.header.frame_id = zero_pose.l_tip
        p.pose.position.x = 0.02
        p.pose.orientation.w = 1
        zero_pose.add_box(attached_link_name,
                          size=(0.1, 0.04, 0.04),
                          parent_link=zero_pose.l_tip,
                          pose=p)

        js_goal = {'r_forearm_roll_joint': 0.0,
                   'r_shoulder_lift_joint': 0.0,
                   'odom_x_joint': 0.0,
                   'odom_y_joint': 0.0,
                   'odom_z_joint': 0.0,
                   'r_shoulder_pan_joint': 0.0,
                   'r_upper_arm_roll_joint': 0.0,
                   'r_wrist_flex_joint': -0.11,
                   'r_wrist_roll_joint': 0.0,
                   'r_elbow_flex_joint': -0.16,
                   'torso_lift_joint': 0.2}

        zero_pose.set_joint_goal(js_goal)

        p = PoseStamped()
        p.header.frame_id = zero_pose.l_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.z = 0.25
        p.pose.orientation.w = 1
        zero_pose.set_cart_goal(p, zero_pose.l_tip, zero_pose.default_root)
        zero_pose.plan_and_execute()

        zero_pose.check_cpi_geq(zero_pose.get_l_gripper_links(), 0.048)
        zero_pose.check_cpi_geq([attached_link_name], 0.048)
        zero_pose.detach_group(attached_link_name)

    def test_attached_collision_allow(self, box_setup: PR2):
        pocky = 'http://muh#pocky'
        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.pose.position.x = 0.05
        p.pose.orientation.w = 1
        box_setup.add_box(pocky,
                          size=(0.1, 0.02, 0.02),
                          parent_link=box_setup.r_tip,
                          pose=p)

        box_setup.allow_collision(group1=pocky, group2='box')

        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.y = -0.11
        p.pose.orientation.w = 1
        box_setup.set_cart_goal(p, box_setup.r_tip, box_setup.default_root)
        box_setup.plan_and_execute()
        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.048)
        box_setup.check_cpi_leq([pocky], 0.0)

    def test_avoid_collision_gripper(self, box_setup: PR2):
        box_setup.allow_all_collisions()
        box_setup.avoid_collision(0.05, box_setup.l_gripper_group, 'box')
        p = PoseStamped()
        p.header.frame_id = box_setup.l_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.15
        p.pose.orientation.w = 1
        box_setup.set_cart_goal(p, box_setup.l_tip, box_setup.default_root, check=False)
        box_setup.plan_and_execute()
        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), -1e-3)

    # def test_end_state_collision(self, box_setup: PR2):
    #     # TODO endstate impossible as long as we check for path collision?
    #     pass

    # def test_filled_vel_values(self, box_setup: PR2):
    #     pass
    #
    # def test_undefined_goal(self, box_setup: PR2):
    #     pass

    # TODO test plan only

    def test_attached_two_items(self, zero_pose: PR2):
        box1_name = 'box1'
        box2_name = 'box2'

        js = {
            'r_elbow_flex_joint': -1.58118094489,
            'r_forearm_roll_joint': -0.904933033043,
            'r_shoulder_lift_joint': 0.822412440711,
            'r_shoulder_pan_joint': -1.07866800992,
            'r_upper_arm_roll_joint': -1.34905471854,
            'r_wrist_flex_joint': -1.20182042644,
            'r_wrist_roll_joint': 0.190433188769,
        }
        zero_pose.set_joint_goal(js)
        zero_pose.plan_and_execute()

        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.r_tip
        r_goal.pose.position.x = 0.4
        r_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[-1, 0, 0, 0],
                                                                      [0, 1, 0, 0],
                                                                      [0, 0, -1, 0],
                                                                      [0, 0, 0, 1]]))
        zero_pose.set_cart_goal(r_goal, zero_pose.l_tip, 'torso_lift_link')
        zero_pose.plan_and_execute()

        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.position.x = 0.1
        p.pose.orientation.w = 1
        zero_pose.add_box(box1_name,
                          size=(.2, .04, .04),
                          parent_link=zero_pose.r_tip,
                          pose=p)
        p.header.frame_id = zero_pose.l_tip
        zero_pose.add_box(box2_name,
                          size=(.2, .04, .04),
                          parent_link=zero_pose.l_tip,
                          pose=p)

        zero_pose.plan_and_execute()

        zero_pose.check_cpi_geq([box1_name, box2_name], 0.049)

        zero_pose.detach_group(box1_name)
        zero_pose.detach_group(box2_name)
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'base_footprint'
        base_goal.pose.position.x = -.1
        base_goal.pose.orientation.w = 1
        zero_pose.move_base(base_goal)

    # def test_pick_and_place(self, kitchen_setup: PR2):
    #
    #     base_pose = PoseStamped()
    #     base_pose.header.frame_id = 'map'
    #     base_pose.pose.position = Point(0.760, 0.480, 0.000)
    #     base_pose.pose.orientation = Quaternion(0.000, 0.000, 0.230, 0.973)
    #     kitchen_setup.move_pr2_base(base_pose)
    #     attached_link_name = 'edekabowl'
    #     p = PoseStamped()
    #     p.header.frame_id = 'map'
    #     p.pose.position = Point(1.39985, 0.799920, 0.888)
    #     p.pose.orientation = Quaternion(-0.0037, -0.00476, 0.3921, 0.9198)
    #     kitchen_setup.add_box(attached_link_name, [.145, .145, .072], pose=p)
    #
    #     pick_pose = PoseStamped()
    #     pick_pose.header.frame_id = 'base_footprint'
    #     pick_pose.pose.position = Point(0.649, -0.023, 0.918)
    #     pick_pose.pose.orientation = Quaternion(0.407, 0.574, -0.408, 0.582)
    #
    #     # pregrasp
    #     pick_pose.pose.position.z += 0.2
    #     kitchen_setup.set_and_check_cart_goal(pick_pose, kitchen_setup.l_tip, kitchen_setup.default_root)
    #
    #     # grasp
    #     pick_pose.pose.position.z -= 0.2
    #     kitchen_setup.avoid_collision(kitchen_setup.get_l_gripper_links(), 'kitchen', [], 0)
    #     kitchen_setup.allow_collision(kitchen_setup.get_l_gripper_links(), attached_link_name, [])
    #     kitchen_setup.set_and_check_cart_goal(pick_pose, kitchen_setup.l_tip, kitchen_setup.default_root)
    #     kitchen_setup.attach_existing(attached_link_name, frame_id=kitchen_setup.l_tip)
    #
    #     # post grasp
    #     pick_pose.pose.position.z += 0.2
    #     kitchen_setup.avoid_all_collisions(0.05)
    #     kitchen_setup.set_and_check_cart_goal(pick_pose, kitchen_setup.l_tip, kitchen_setup.default_root)
    #     # kitchen_setup.remove_object(attached_link_name)
    #     kitchen_setup.send_and_check_joint_goal(gaya_pose)
    #
    #     # place============================
    #     base_pose.pose.position = Point(-0.200, 1.120, 0.000)
    #     base_pose.pose.orientation = Quaternion(0.000, 0.000, 0.994, -0.105)
    #     kitchen_setup.move_pr2_base(base_pose)
    #
    #     # pre place
    #     place_pose = PoseStamped()
    #     place_pose.header.frame_id = 'base_footprint'
    #     place_pose.pose.position = Point(0.587, 0.068, 0.920)
    #     place_pose.pose.orientation = Quaternion(0.703, -0.074, -0.703, -0.074)
    #     place_pose.pose.position.z += 0.2
    #     kitchen_setup.set_and_check_cart_goal(place_pose, kitchen_setup.l_tip, kitchen_setup.default_root)
    #
    #     # place
    #     place_pose.pose.position.z -= 0.19
    #     kitchen_setup.avoid_all_collisions(0.)
    #     kitchen_setup.set_cart_goal(place_pose, kitchen_setup.l_tip, kitchen_setup.default_root)
    #     kitchen_setup.send_goal()
    #     rospy.sleep(1)
    #
    #     # post place
    #     kitchen_setup.detach_object(attached_link_name)
    #     place_pose.pose.position.z += 0.2
    #     kitchen_setup.avoid_all_collisions(0.)
    #     kitchen_setup.set_and_check_cart_goal(place_pose, kitchen_setup.l_tip, kitchen_setup.default_root)

    # def test_hand_in_kitchen(self, kitchen_setup: PR2):
    #
    #     kitchen_setup.send_and_check_joint_goal(pick_up_pose)
    #
    #     base_pose = PoseStamped()
    #     base_pose.header.frame_id = 'map'
    #     base_pose.pose.position = Point(0.743, 0.586, 0.000)
    #     base_pose.pose.orientation.w = 1
    #     kitchen_setup.teleport_base(base_pose)
    #
    #     # grasp
    #     p = PoseStamped()
    #     p.header.frame_id = kitchen_setup.l_tip
    #     p.pose.position.x = 0.2
    #     p.pose.orientation.w = 1
    #     kitchen_setup.allow_collision(kitchen_setup.get_l_gripper_links(), 'kitchen',
    #                                           ['sink_area', 'sink_area_surface'])
    #     kitchen_setup.set_and_check_cart_goal(p, kitchen_setup.l_tip, kitchen_setup.default_root)
    #
    #     # post grasp
    #     pregrasp_pose = PoseStamped()
    #     pregrasp_pose.header.frame_id = 'base_footprint'
    #     pregrasp_pose.pose.position.x = 0.611175722907
    #     pregrasp_pose.pose.position.y = -0.0244662287535
    #     pregrasp_pose.pose.position.z = 1.10803325995
    #     pregrasp_pose.pose.orientation.x = -0.0128682380997
    #     pregrasp_pose.pose.orientation.y = -0.710292569338
    #     pregrasp_pose.pose.orientation.z = 0.0148339707762
    #     pregrasp_pose.pose.orientation.w = -0.703632573456
    #     kitchen_setup.avoid_all_collisions(0.05)
    #     kitchen_setup.set_and_check_cart_goal(pregrasp_pose, kitchen_setup.l_tip, kitchen_setup.default_root)

    def test_set_kitchen_joint_state(self, kitchen_setup: PR2):
        kitchen_js = {'sink_area_left_upper_drawer_main_joint': 0.45}
        kitchen_setup.set_kitchen_js(kitchen_js)

    def test_ease_fridge(self, kitchen_setup: PR2):
        milk_name = 'milk'

        # take milk out of fridge
        kitchen_setup.set_kitchen_js({'iai_fridge_door_joint': 1.56})

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 0.565
        base_goal.pose.position.y = -0.5
        base_goal.pose.orientation.z = -0.51152562713
        base_goal.pose.orientation.w = 0.85926802151
        kitchen_setup.teleport_base(base_goal)
        # kitchen_setup.add_json_goal('BasePointingForward')

        # spawn milk
        milk_pose = PoseStamped()
        milk_pose.header.frame_id = 'iai_kitchen/iai_fridge_door_shelf1_bottom'
        milk_pose.pose.position = Point(0, 0, 0.12)
        milk_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        milk_pre_pose = PoseStamped()
        milk_pre_pose.header.frame_id = 'iai_kitchen/iai_fridge_door_shelf1_bottom'
        milk_pre_pose.pose.position = Point(0, 0, 0.22)
        milk_pre_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup.add_box(milk_name, (0.05, 0.05, 0.2), pose=milk_pose)

        # grasp milk
        kitchen_setup.open_l_gripper()

        # l_goal = deepcopy(milk_pose)
        # l_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[1, 0, 0, 0],
        #                                                               [0, 1, 0, 0],
        #                                                               [0, 0, 1, 0],
        #                                                               [0, 0, 0, 1]]))
        # kitchen_setup.set_cart_goal(l_goal, kitchen_setup.l_tip, kitchen_setup.default_root)
        # kitchen_setup.send_and_check_goal()

        # handle_name = 'map'
        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = 'map'
        bar_axis.vector.z = 1

        bar_center = PointStamped()
        bar_center.header.frame_id = milk_pose.header.frame_id
        bar_center.point = deepcopy(milk_pose.pose.position)

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = kitchen_setup.l_tip
        tip_grasp_axis.vector.z = 1

        kitchen_setup.set_json_goal('GraspBar',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=kitchen_setup.l_tip,
                                    tip_grasp_axis=tip_grasp_axis,
                                    bar_center=bar_center,
                                    bar_axis=bar_axis,
                                    bar_length=.12)

        x = Vector3Stamped()
        x.header.frame_id = kitchen_setup.l_tip
        x.vector.x = 1
        x_map = Vector3Stamped()
        x_map.header.frame_id = 'iai_kitchen/iai_fridge_door'
        x_map.vector.x = 1
        # z = Vector3Stamped()
        # z.header.frame_id = 'milk'
        # z.vector.z = 1
        # z_map = Vector3Stamped()
        # z_map.header.frame_id = 'map'
        # z_map.vector.z = 1
        kitchen_setup.set_align_planes_goal(kitchen_setup.l_tip, x, root_normal=x_map)

        # kitchen_setup.allow_collision([], milk_name, [])
        # kitchen_setup.add_json_goal('AvoidJointLimits', percentage=15)
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()

        kitchen_setup.update_parent_link_of_group(milk_name, kitchen_setup.l_tip)
        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        kitchen_setup.close_l_gripper()

        # x = Vector3Stamped()
        # x.header.frame_id = 'milk'
        # x.vector.x = 1
        # x_map = Vector3Stamped()
        # x_map.header.frame_id = 'iai_kitchen/iai_fridge_door'
        # x_map.vector.x = 1
        # z = Vector3Stamped()
        # z.header.frame_id = 'milk'
        # z.vector.z = 1
        # z_map = Vector3Stamped()
        # z_map.header.frame_id = 'map'
        # z_map.vector.z = 1
        # kitchen_setup.align_planes('milk', x, root_normal=x_map)
        # kitchen_setup.align_planes('milk', z, root_normal=z_map)
        # kitchen_setup.keep_orientation('milk')
        kitchen_setup.set_cart_goal(milk_pre_pose, milk_name, kitchen_setup.default_root)
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_joint_goal(kitchen_setup.better_pose)
        kitchen_setup.plan_and_execute()

        # place milk back

        # kitchen_setup.add_json_goal('BasePointingForward')
        # milk_goal = PoseStamped()
        # milk_goal.header.frame_id = 'iai_kitchen/kitchen_island_surface'
        # milk_goal.pose.position = Point(.1, -.2, .13)
        # milk_goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0,0,1]))
        kitchen_setup.set_cart_goal(milk_pre_pose, milk_name, kitchen_setup.default_root)
        kitchen_setup.plan_and_execute()

        kitchen_setup.set_cart_goal(milk_pose, milk_name, kitchen_setup.default_root)
        kitchen_setup.plan_and_execute()

        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        kitchen_setup.open_l_gripper()

        kitchen_setup.detach_group(milk_name)

        kitchen_setup.set_joint_goal(kitchen_setup.better_pose)
        kitchen_setup.plan_and_execute()

    def test_ease_cereal_1(self, kitchen_setup):
        # FIXME collision avoidance needs soft_threshholds at 0
        cereal_name = u'cereal'
        drawer_frame_id = u'iai_kitchen/oven_area_area_right_drawer_board_2_link'
        drawer_frame = u'oven_area_area_right_drawer_board_2_link'

        # spawn milk
        cereal_pose = PoseStamped()
        cereal_pose.header.frame_id = drawer_frame_id
        cereal_pose.pose.position = Point(0.123, 0.0, 0.15) #placing z 0.15
        cereal_pose.pose.orientation = Quaternion(0.0087786, 0.005395, -0.838767, -0.544393)
        kitchen_setup.add_box(cereal_name, [0.1028, 0.0634, 0.20894], cereal_pose)# placing size: [0.1028, 0.0634, 0.20894]
        kitchen_setup.update_parent_link_of_group(cereal_name, parent_link=drawer_frame, parent_link_group='kitchen')

        # take milk out of fridge
        kitchen_setup.set_kitchen_js({u'oven_area_area_right_drawer_main_joint': 0.48})

        drawer_T_box = tf.np_to_kdl(kitchen_setup.world.get_fk(drawer_frame, cereal_name))

        # grasp milk
        kitchen_setup.open_l_gripper()
        grasp_pose = PoseStamped()
        grasp_pose.header.frame_id = cereal_name
        grasp_pose.pose.position = Point(0.03, 0, 0.05)
        grasp_pose.pose.orientation = Quaternion(0, 0, 1, 0)
        box_T_r_goal = tf.msg_to_kdl(grasp_pose)
        box_T_r_goal_pre = deepcopy(box_T_r_goal)
        box_T_r_goal_pre.p[0] += 0.2

        pre_grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal_pre, drawer_frame)

        box_T_r_goal_post = deepcopy(box_T_r_goal)
        box_T_r_goal_post.p[0] += 0.4
        post_grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal_post, drawer_frame)

        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup.set_json_goal(u'CartesianPose',
                                    tip_link=kitchen_setup.r_tip,
                                    root_link=kitchen_setup.default_root,
                                    goal_pose=pre_grasp_pose)
        kitchen_setup.plan_and_execute()

        grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal, drawer_frame)

        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup.set_cart_goal(grasp_pose,
                                    tip_link=kitchen_setup.r_tip)
        kitchen_setup.plan_and_execute()

        #kitchen_setup.attach_object(cereal_name, kitchen_setup.r_tip)
        kitchen_setup.detach_group(cereal_name)
        kitchen_setup.update_parent_link_of_group(cereal_name, parent_link=kitchen_setup.r_tip)
        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        kitchen_setup.close_l_gripper()

        # x = Vector3Stamped()
        # x.header.frame_id = 'milk'
        # x.vector.x = 1
        # x_map = Vector3Stamped()
        # x_map.header.frame_id = 'iai_kitchen/iai_fridge_door'
        # x_map.vector.x = 1
        # z = Vector3Stamped()
        # z.header.frame_id = 'milk'
        # z.vector.z = 1
        # z_map = Vector3Stamped()
        # z_map.header.frame_id = 'map'
        # z_map.vector.z = 1
        # kitchen_setup.align_planes('milk', x, root_normal=x_map)
        # kitchen_setup.align_planes('milk', z, root_normal=z_map)
        # kitchen_setup.keep_orientation(u'milk')
        # kitchen_setup.set_cart_goal(grasp_pose, cereal_name, kitchen_setup.default_root)
        kitchen_setup.set_json_goal(u'CartesianPose',
                                    tip_link=kitchen_setup.r_tip,
                                    root_link=kitchen_setup.default_root,
                                    goal_pose=post_grasp_pose)
        kitchen_setup.plan_and_execute()
        #kitchen_setup.set_joint_goal(gaya_pose)

        # place milk back

        # kitchen_setup.add_json_goal(u'BasePointingForward')
        # milk_goal = PoseStamped()
        # milk_goal.header.frame_id = u'iai_kitchen/kitchen_island_surface'
        # milk_goal.pose.position = Point(.1, -.2, .13)
        # milk_goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0,0,1]))

        kitchen_setup.set_json_goal(u'CartesianPose',
                                    tip_link=kitchen_setup.r_tip,
                                    root_link=kitchen_setup.default_root,
                                    goal_pose=grasp_pose)
        kitchen_setup.plan_and_execute()

        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        kitchen_setup.open_l_gripper()
        kitchen_setup.detach_object(cereal_name)

        kitchen_setup.set_joint_goal(kitchen_setup.better_pose)
        kitchen_setup.plan_and_execute()

    def test_ease_cereal_3(self, kitchen_setup: PR2):
        # FIXME
        cereal_name = 'cereal'
        drawer_frame_id = 'iai_kitchen/oven_area_area_right_drawer_board_2_link'

        # take milk out of fridge
        kitchen_setup.set_kitchen_js({'oven_area_area_right_drawer_main_joint': 0.48})

        # spawn milk

        cereal_pose = PoseStamped()
        cereal_pose.header.frame_id = drawer_frame_id
        cereal_pose.pose.position = Point(0.123, -0.03, 0.11)
        cereal_pose.pose.orientation = Quaternion(0.0087786, 0.005395, -0.838767, -0.544393)
        kitchen_setup.add_box(cereal_name, (0.1528, 0.0634, 0.22894), pose=cereal_pose)

        cereal_pose_in_map = tf.msg_to_kdl(tf.transform_pose(u'map', cereal_pose))

        drawer_T_box = tf.msg_to_kdl(cereal_pose)

        # grasp milk
        kitchen_setup.open_l_gripper()
        grasp_pose = PoseStamped()
        grasp_pose.header.frame_id = cereal_name
        grasp_pose.pose.position = Point(0.13, 0, 0.05)
        grasp_pose.pose.orientation = Quaternion(0, 0, 1, 0)
        box_T_r_goal = tf.msg_to_kdl(grasp_pose)
        box_T_r_goal_pre = deepcopy(box_T_r_goal)
        box_T_r_goal_pre.p[0] += 0.1

        pre_grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal_pre, drawer_frame_id)

        kitchen_setup.set_json_goal('AvoidJointLimits', percentage=40)
        kitchen_setup.set_cart_goal(pre_grasp_pose, tip_link=kitchen_setup.r_tip)
        kitchen_setup.plan_and_execute()

        grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal, drawer_frame_id)

        kitchen_setup.set_json_goal('AvoidJointLimits', percentage=40)
        kitchen_setup.set_cart_goal(grasp_pose, tip_link=kitchen_setup.r_tip)
        kitchen_setup.plan_and_execute()

        kitchen_setup.update_parent_link_of_group(cereal_name, kitchen_setup.r_tip)
        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        kitchen_setup.close_l_gripper()

        # x = Vector3Stamped()
        # x.header.frame_id = 'milk'
        # x.vector.x = 1
        # x_map = Vector3Stamped()
        # x_map.header.frame_id = 'iai_kitchen/iai_fridge_door'
        # x_map.vector.x = 1
        # z = Vector3Stamped()
        # z.header.frame_id = 'milk'
        # z.vector.z = 1
        # z_map = Vector3Stamped()
        # z_map.header.frame_id = 'map'
        # z_map.vector.z = 1
        # kitchen_setup.align_planes('milk', x, root_normal=x_map)
        # kitchen_setup.align_planes('milk', z, root_normal=z_map)
        # kitchen_setup.keep_orientation('milk')
        # kitchen_setup.set_cart_goal(grasp_pose, cereal_name, kitchen_setup.default_root)
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_joint_goal(kitchen_setup.better_pose)
        kitchen_setup.plan_and_execute()

        # place milk back

        # kitchen_setup.add_json_goal('BasePointingForward')
        # milk_goal = PoseStamped()
        # milk_goal.header.frame_id = 'iai_kitchen/kitchen_island_surface'
        # milk_goal.pose.position = Point(.1, -.2, .13)
        # milk_goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0,0,1]))

        kitchen_setup.set_cart_goal(cereal_pose, cereal_name)  # , kitchen_setup.default_root)
        kitchen_setup.plan_and_execute()

        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        kitchen_setup.open_l_gripper()

        kitchen_setup.detach_group(cereal_name)

        kitchen_setup.set_joint_goal(kitchen_setup.better_pose)
        kitchen_setup.plan_and_execute()

    def test_bowl_and_cup(self, kitchen_setup: PR2):
        # kernprof -lv py.test -s test/test_integration_pr2.py::TestCollisionAvoidanceGoals::test_bowl_and_cup
        bowl_name = 'bowl'
        cup_name = 'cup'
        percentage = 50
        drawer_handle = 'sink_area_left_middle_drawer_handle'
        drawer_joint = 'sink_area_left_middle_drawer_main_joint'
        # spawn cup
        cup_pose = PoseStamped()
        cup_pose.header.frame_id = 'iai_kitchen/sink_area_left_middle_drawer_main'
        cup_pose.pose.position = Point(0.1, 0.2, -.05)
        cup_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup.add_cylinder(cup_name, height=0.07, radius=0.04, pose=cup_pose, parent_link_group='kitchen',
                                   parent_link='sink_area_left_middle_drawer_main')

        # spawn bowl
        bowl_pose = PoseStamped()
        bowl_pose.header.frame_id = 'iai_kitchen/sink_area_left_middle_drawer_main'
        bowl_pose.pose.position = Point(0.1, -0.2, -.05)
        bowl_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup.add_cylinder(bowl_name, height=0.05, radius=0.07, pose=bowl_pose, parent_link_group='kitchen',
                                   parent_link='sink_area_left_middle_drawer_main')

        # grasp drawer handle
        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = drawer_handle
        bar_axis.vector.y = 1

        bar_center = PointStamped()
        bar_center.header.frame_id = drawer_handle

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = kitchen_setup.l_tip
        tip_grasp_axis.vector.z = 1

        kitchen_setup.set_json_goal('GraspBar',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=kitchen_setup.l_tip,
                                    tip_grasp_axis=tip_grasp_axis,
                                    bar_center=bar_center,
                                    bar_axis=bar_axis,
                                    bar_length=0.4)  # TODO: check for real length
        x_gripper = Vector3Stamped()
        x_gripper.header.frame_id = kitchen_setup.l_tip
        x_gripper.vector.x = 1

        x_goal = Vector3Stamped()
        x_goal.header.frame_id = drawer_handle
        x_goal.vector.x = -1

        kitchen_setup.set_align_planes_goal(kitchen_setup.l_tip,
                                            x_gripper,
                                            root_normal=x_goal)
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()

        # open drawer
        kitchen_setup.set_json_goal('Open',
                                    tip_link=kitchen_setup.l_tip,
                                    environment_link=drawer_handle)
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_kitchen_js({drawer_joint: 0.48})

        kitchen_setup.set_joint_goal(kitchen_setup.better_pose)
        kitchen_setup.plan_and_execute()

        # grasp bowl
        l_goal = deepcopy(bowl_pose)
        l_goal.pose.position.z += .2
        l_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 1, 0, 0],
                                                                      [0, 0, -1, 0],
                                                                      [-1, 0, 0, 0],
                                                                      [0, 0, 0, 1]]))
        kitchen_setup.set_cart_goal(l_goal, kitchen_setup.l_tip, kitchen_setup.default_root)

        # grasp cup
        r_goal = deepcopy(cup_pose)
        r_goal.pose.position.z += .2
        r_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 1, 0, 0],
                                                                      [0, 0, -1, 0],
                                                                      [-1, 0, 0, 0],
                                                                      [0, 0, 0, 1]]))
        kitchen_setup.set_json_goal('AvoidJointLimits', percentage=percentage)
        kitchen_setup.set_cart_goal(r_goal, kitchen_setup.r_tip, kitchen_setup.default_root)
        kitchen_setup.plan_and_execute()

        l_goal.pose.position.z -= .2
        r_goal.pose.position.z -= .2
        kitchen_setup.allow_collision(group1=kitchen_setup.get_robot_name(), group2=bowl_name)
        kitchen_setup.allow_collision(group1=kitchen_setup.get_robot_name(), group2=cup_name)
        kitchen_setup.set_cart_goal(l_goal, kitchen_setup.l_tip, kitchen_setup.default_root)
        kitchen_setup.set_cart_goal(r_goal, kitchen_setup.r_tip, kitchen_setup.default_root)
        kitchen_setup.set_json_goal('AvoidJointLimits', percentage=percentage)
        kitchen_setup.plan_and_execute()

        kitchen_setup.update_parent_link_of_group(bowl_name, kitchen_setup.l_tip)
        kitchen_setup.update_parent_link_of_group(cup_name, kitchen_setup.r_tip)

        kitchen_setup.set_joint_goal(kitchen_setup.better_pose)
        kitchen_setup.plan_and_execute()
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'base_footprint'
        base_goal.pose.position.x = -.1
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        kitchen_setup.teleport_base(base_goal)

        # place bowl and cup
        bowl_goal = PoseStamped()
        bowl_goal.header.frame_id = 'iai_kitchen/kitchen_island_surface'
        bowl_goal.pose.position = Point(.2, 0, .05)
        bowl_goal.pose.orientation = Quaternion(0, 0, 0, 1)

        cup_goal = PoseStamped()
        cup_goal.header.frame_id = 'iai_kitchen/kitchen_island_surface'
        cup_goal.pose.position = Point(.15, 0.25, .07)
        cup_goal.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup.set_cart_goal(bowl_goal, bowl_name, kitchen_setup.default_root)
        kitchen_setup.set_cart_goal(cup_goal, cup_name, kitchen_setup.default_root)
        kitchen_setup.set_json_goal('AvoidJointLimits', percentage=percentage)
        kitchen_setup.plan_and_execute()

        kitchen_setup.detach_group(bowl_name)
        kitchen_setup.detach_group(cup_name)
        kitchen_setup.allow_collision(group1=kitchen_setup.get_robot_name(), group2=cup_name)
        kitchen_setup.allow_collision(group1=kitchen_setup.get_robot_name(), group2=bowl_name)
        kitchen_setup.set_joint_goal(kitchen_setup.better_pose)
        kitchen_setup.plan_and_execute()

    def test_ease_grasp_bowl(self, kitchen_setup: PR2):
        percentage = 40

        base_pose = PoseStamped()
        base_pose.header.frame_id = 'map'
        base_pose.pose.position = Point(0.314, 0.818, 0.000)
        base_pose.pose.orientation = Quaternion(-0.001, 0.000, 0.037, 0.999)
        kitchen_setup.teleport_base(base_pose)

        js = {
            'torso_lift_joint': 0.262156255996,
            'head_pan_joint': 0.0694220762479,
            'head_tilt_joint': 1.01903547689,
            'r_upper_arm_roll_joint': -1.5717499752,
            'r_shoulder_pan_joint': -0.00156068057783,
            'r_shoulder_lift_joint': 0.252786184819,
            'r_forearm_roll_joint': -89.673490548,
            'r_elbow_flex_joint': -0.544166310929,
            'r_wrist_flex_joint': -1.32591140165,
            'r_wrist_roll_joint': 65.7348048877,
            'l_upper_arm_roll_joint': 1.38376171392,
            'l_shoulder_pan_joint': 1.59536261129,
            'l_shoulder_lift_joint': -0.0236488517104,
            'l_forearm_roll_joint': 23.2795803857,
            'l_elbow_flex_joint': -1.72694302293,
            'l_wrist_flex_joint': -0.48001173639,
            'l_wrist_roll_joint': -6.28312737965,
        }
        kitchen_setup.allow_all_collisions()
        kitchen_setup.set_joint_goal(js)
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_kitchen_js({'sink_area_left_middle_drawer_main_joint': 0.45})

        r_goal = PoseStamped()
        r_goal.header.frame_id = kitchen_setup.r_tip
        r_goal.pose.position.x += 0.25
        r_goal.pose.orientation.w = 1

        kitchen_setup.set_json_goal('AvoidJointLimits', percentage=percentage)
        kitchen_setup.set_cart_goal(r_goal, tip_link=kitchen_setup.r_tip)
        kitchen_setup.plan_and_execute()

        # spawn cup

    # def test_avoid_self_collision2(self, kitchen_setup: PR2):
    #     base_goal = PoseStamped()
    #     base_goal.header.frame_id = 'base_footprint'
    #     base_goal.pose.position.x = -.1
    #     base_goal.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
    #     kitchen_setup.teleport_base(base_goal)
    #
    #     # place bowl and cup
    #     bowl_goal = PoseStamped()
    #     bowl_goal.header.frame_id = 'iai_kitchen/kitchen_island_surface'
    #     bowl_goal.pose.position = Point(.2, 0, .05)
    #     bowl_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 0, -1, 0],
    #                                                                      [0, -1, 0, 0],
    #                                                                      [-1, 0, 0, 0],
    #                                                                      [0, 0, 0, 1]]))
    #
    #     cup_goal = PoseStamped()
    #     cup_goal.header.frame_id = 'iai_kitchen/kitchen_island_surface'
    #     cup_goal.pose.position = Point(.15, 0.25, .07)
    #     cup_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 0, -1, 0],
    #                                                                     [0, -1, 0, 0],
    #                                                                     [-1, 0, 0, 0],
    #                                                                     [0, 0, 0, 1]]))
    #
    #     kitchen_setup.set_cart_goal(bowl_goal, kitchen_setup.l_tip, kitchen_setup.default_root)
    #     kitchen_setup.set_cart_goal(cup_goal, kitchen_setup.r_tip, kitchen_setup.default_root)
    #     kitchen_setup.send_and_check_goal()

    def test_ease_spoon(self, kitchen_setup: PR2):
        spoon_name = 'spoon'
        percentage = 40

        # spawn cup
        cup_pose = PoseStamped()
        cup_pose.header.frame_id = 'iai_kitchen/sink_area_surface'
        cup_pose.pose.position = Point(0.1, -.5, .02)
        cup_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup.add_box(spoon_name, (0.1, 0.02, 0.01), pose=cup_pose)

        # kitchen_setup.send_and_check_joint_goal(gaya_pose)

        # grasp spoon
        l_goal = deepcopy(cup_pose)
        l_goal.pose.position.z += .2
        l_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 0, -1, 0],
                                                                      [0, -1, 0, 0],
                                                                      [-1, 0, 0, 0],
                                                                      [0, 0, 0, 1]]))
        kitchen_setup.set_json_goal('AvoidJointLimits', percentage=percentage)
        kitchen_setup.set_cart_goal(l_goal, kitchen_setup.l_tip, kitchen_setup.default_root)
        kitchen_setup.plan_and_execute()

        l_goal.pose.position.z -= .2
        # kitchen_setup.allow_collision([CollisionEntry.ALL], spoon_name, [CollisionEntry.ALL])
        kitchen_setup.set_cart_goal(l_goal, kitchen_setup.l_tip, kitchen_setup.default_root)
        kitchen_setup.set_json_goal('AvoidJointLimits', percentage=percentage)
        kitchen_setup.plan_and_execute()
        kitchen_setup.update_parent_link_of_group(spoon_name, kitchen_setup.l_tip)

        l_goal.pose.position.z += .2
        # kitchen_setup.allow_collision([CollisionEntry.ALL], spoon_name, [CollisionEntry.ALL])
        kitchen_setup.set_cart_goal(l_goal, kitchen_setup.l_tip, kitchen_setup.default_root)
        kitchen_setup.set_json_goal('AvoidJointLimits', percentage=percentage)
        kitchen_setup.plan_and_execute()

        l_goal.pose.position.z -= .2
        # kitchen_setup.allow_collision([CollisionEntry.ALL], spoon_name, [CollisionEntry.ALL])
        kitchen_setup.set_cart_goal(l_goal, kitchen_setup.l_tip, kitchen_setup.default_root)
        kitchen_setup.set_json_goal('AvoidJointLimits', percentage=percentage)
        kitchen_setup.plan_and_execute()

        kitchen_setup.set_joint_goal(kitchen_setup.better_pose)
        kitchen_setup.plan_and_execute()

    def test_ease_place_on_new_table(self, kitchen_setup: PR2):
        percentage = 40
        js = {
            'torso_lift_joint': 0.262343532164,
            'head_pan_joint': 0.0308852063639,
            'head_tilt_joint': 0.710418818732,
            'r_upper_arm_roll_joint': -1.4635104674,
            'r_shoulder_pan_joint': -1.59535749265,
            'r_shoulder_lift_joint': -0.0235854289628,
            'r_forearm_roll_joint': -123.897562601,
            'r_elbow_flex_joint': -1.72694302293,
            'r_wrist_flex_joint': -0.480010977079,
            'r_wrist_roll_joint': 88.0157228707,
            'l_upper_arm_roll_joint': 1.90635809306,
            'l_shoulder_pan_joint': 0.352841136964,
            'l_shoulder_lift_joint': -0.35035444474,
            'l_forearm_roll_joint': 32.5396842176,
            'l_elbow_flex_joint': -0.543731998795,
            'l_wrist_flex_joint': -1.68825444756,
            'l_wrist_roll_joint': -12.6846818117,
        }
        kitchen_setup.set_joint_goal(js)
        kitchen_setup.plan_and_execute()
        base_pose = PoseStamped()
        base_pose.header.frame_id = 'map'
        base_pose.pose.position = Point(-2.8, 0.188, -0.000)  # -2.695
        base_pose.pose.orientation = Quaternion(-0.001, -0.001, 0.993, -0.114)
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.teleport_base(base_pose)

        object_name = 'box'
        p = PoseStamped()
        p.header.frame_id = kitchen_setup.l_tip
        p.pose.position = Point(0.0175, 0.025, 0)
        p.pose.orientation.w = 1
        kitchen_setup.add_box(name=object_name,
                              size=(0.10, 0.14, 0.14),
                              parent_link=kitchen_setup.l_tip,
                              pose=p)

        l_goal = PoseStamped()
        l_goal.header.stamp = rospy.get_rostime()
        l_goal.header.frame_id = kitchen_setup.l_tip
        l_goal.pose.position.x += 0.2
        # l_goal.pose.position.z -= 0.1
        l_goal.pose.orientation.w = 1
        kitchen_setup.set_json_goal('AvoidJointLimits', percentage=percentage)

        js = {
            'r_upper_arm_roll_joint': -1.4635104674,
            'r_shoulder_pan_joint': -1.59535749265,
            'r_shoulder_lift_joint': -0.0235854289628,
            'r_forearm_roll_joint': -123.897562601,
            'r_elbow_flex_joint': -1.72694302293,
            'r_wrist_flex_joint': -0.480010977079,
            'r_wrist_roll_joint': 88.0157228707,
        }
        kitchen_setup.set_joint_goal(js)

        # base_pose.header.frame_id = 'base_footprint'
        # base_pose.pose.position = Point(0,0,0)
        # base_pose.pose.orientation = Quaternion(0,0,0,1)
        # kitchen_setup.set_cart_goal(base_pose, 'base_footprint')

        kitchen_setup.set_cart_goal(l_goal, tip_link=kitchen_setup.l_tip)
        kitchen_setup.plan_and_execute()

    def test_tray(self, kitchen_setup: PR2):
        # FIXME
        tray_name = 'tray'
        percentage = 50

        tray_pose = PoseStamped()
        tray_pose.header.frame_id = 'iai_kitchen/sink_area_surface'
        tray_pose.pose.position = Point(0.1, -0.4, 0.07)
        tray_pose.pose.orientation.w = 1

        kitchen_setup.add_box(tray_name, (.2, .4, .1), pose=tray_pose)

        l_goal = deepcopy(tray_pose)
        l_goal.pose.position.y -= 0.18
        l_goal.pose.position.z += 0.06
        l_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 0, -1, 0],
                                                                      [1, 0, 0, 0],
                                                                      [0, -1, 0, 0],
                                                                      [0, 0, 0, 1]]))

        r_goal = deepcopy(tray_pose)
        r_goal.pose.position.y += 0.18
        r_goal.pose.position.z += 0.06
        r_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 0, 1, 0],
                                                                      [-1, 0, 0, 0],
                                                                      [0, -1, 0, 0],
                                                                      [0, 0, 0, 1]]))

        kitchen_setup.set_cart_goal(l_goal, kitchen_setup.l_tip)
        kitchen_setup.set_cart_goal(r_goal, kitchen_setup.r_tip)
        kitchen_setup.allow_collision(kitchen_setup.get_robot_name(), tray_name)
        kitchen_setup.avoid_joint_limits(percentage=percentage)
        # grasp tray
        kitchen_setup.plan_and_execute()

        kitchen_setup.update_parent_link_of_group(tray_name, kitchen_setup.r_tip)

        r_goal = PoseStamped()
        r_goal.header.frame_id = kitchen_setup.l_tip
        r_goal.pose.orientation.w = 1
        kitchen_setup.set_cart_goal(r_goal, kitchen_setup.l_tip, tray_name)

        tray_goal = tf.lookup_pose('base_footprint', tray_name)
        tray_goal.pose.position.y = 0
        tray_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[-1, 0, 0, 0],
                                                                         [0, -1, 0, 0],
                                                                         [0, 0, 1, 0],
                                                                         [0, 0, 0, 1]]))
        kitchen_setup.set_cart_goal(tray_goal, tray_name, 'base_footprint')

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x -= 0.5
        base_goal.pose.position.y -= 0.3
        base_goal.pose.orientation.w = 1
        kitchen_setup.avoid_joint_limits(percentage=percentage)
        kitchen_setup.allow_collision(group1=tray_name,
                                      group2=kitchen_setup.l_gripper_group)
        # kitchen_setup.allow_self_collision()
        # drive back
        kitchen_setup.move_base(base_goal)

        r_goal = PoseStamped()
        r_goal.header.frame_id = kitchen_setup.l_tip
        r_goal.pose.orientation.w = 1
        kitchen_setup.set_cart_goal(r_goal, kitchen_setup.l_tip, tray_name)

        expected_pose = tf.lookup_pose(tray_name, kitchen_setup.l_tip)
        expected_pose.header.stamp = rospy.Time()

        tray_goal = PoseStamped()
        tray_goal.header.frame_id = tray_name
        tray_goal.pose.position.z = .1
        tray_goal.pose.position.x = .1
        tray_goal.pose.orientation = Quaternion(*quaternion_about_axis(-1, [0, 1, 0]))
        kitchen_setup.avoid_joint_limits(percentage=percentage)
        kitchen_setup.allow_collision(group1=tray_name,
                                      group2=kitchen_setup.l_gripper_group)
        kitchen_setup.set_cart_goal(tray_goal, tray_name, 'base_footprint')
        kitchen_setup.plan_and_execute()

    # TODO FIXME attaching and detach of urdf objects that listen to joint states

    def test_iis(self, kitchen_setup: PR2):
        # rosrun tf static_transform_publisher 0 - 0.2 0.93 1.5707963267948966 0 0 iai_kitchen/table_area_main lid 10
        # rosrun tf static_transform_publisher 0 - 0.15 0 0 0 0 lid goal 10
        # kitchen_setup.set_joint_goal(pocky_pose)
        # kitchen_setup.send_and_check_goal()
        object_name = 'lid'
        pot_pose = PoseStamped()
        pot_pose.header.frame_id = 'lid'
        pot_pose.pose.position.z = -0.22
        # pot_pose.pose.orientation.w = 1
        pot_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 2, [0, 0, 1]))
        kitchen_setup.add_mesh(object_name,
                               mesh='package://cad_models/kitchen/cooking-vessels/cookingpot.dae',
                               pose=pot_pose)

        base_pose = PoseStamped()
        base_pose.header.frame_id = 'iai_kitchen/table_area_main'
        base_pose.pose.position.y = -1.1
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 2, [0, 0, 1]))
        kitchen_setup.teleport_base(base_pose)
        # m = zero_pose.world.get_object(object_name).as_marker_msg()
        # compare_poses(m.pose, p.pose)

        hand_goal = PoseStamped()
        hand_goal.header.frame_id = 'lid'
        hand_goal.pose.position.y = -0.15
        hand_goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 2, [0, 0, 1]))
        # kitchen_setup.allow_all_collisions()
        # kitchen_setup.avoid_collision([], 'kitchen', ['table_area_main'], 0.05)
        kitchen_setup.set_cart_goal(hand_goal, 'r_gripper_tool_frame')
        kitchen_setup.send_goal(goal_type=MoveGoal.PLAN_ONLY)
        kitchen_setup.set_cart_goal(hand_goal, 'r_gripper_tool_frame')
        kitchen_setup.send_goal()

        hand_goal = PoseStamped()
        hand_goal.header.frame_id = 'r_gripper_tool_frame'
        hand_goal.pose.position.x = 0.15
        hand_goal.pose.orientation.w = 1
        # kitchen_setup.allow_all_collisions()
        # kitchen_setup.avoid_collision([], 'kitchen', ['table_area_main'], 0.05)
        kitchen_setup.set_cart_goal(hand_goal, 'r_gripper_tool_frame')
        kitchen_setup.allow_all_collisions()
        kitchen_setup.send_goal(goal_type=MoveGoal.PLAN_ONLY)
        kitchen_setup.set_cart_goal(hand_goal, 'r_gripper_tool_frame')
        kitchen_setup.allow_all_collisions()
        kitchen_setup.send_goal()

        # kitchen_setup.add_cylinder('pot', size=[0.2,0.2], pose=pot_pose)

    def test_ease_dishwasher(self, kitchen_setup: PR2):
        # FIXME
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.orientation.w = 1
        p.pose.position.x = 0.5
        p.pose.position.y = 0.2
        kitchen_setup.teleport_base(p)

        hand = kitchen_setup.r_tip

        goal_angle = np.pi / 4
        handle_frame_id = 'iai_kitchen/sink_area_dish_washer_door_handle'
        handle_name = 'sink_area_dish_washer_door_handle'
        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = handle_frame_id
        bar_axis.vector.y = 1

        bar_center = PointStamped()
        bar_center.header.frame_id = handle_frame_id

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = hand
        tip_grasp_axis.vector.z = 1

        kitchen_setup.set_json_goal('GraspBar',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=hand,
                                    tip_grasp_axis=tip_grasp_axis,
                                    bar_center=bar_center,
                                    bar_axis=bar_axis,
                                    bar_length=.3)
        # kitchen_setup.allow_collision([], 'kitchen', [handle_name])
        kitchen_setup.allow_all_collisions()

        gripper_axis = Vector3Stamped()
        gripper_axis.header.frame_id = hand
        gripper_axis.vector.x = 1

        world_axis = Vector3Stamped()
        world_axis.header.frame_id = handle_frame_id
        world_axis.vector.x = -1
        kitchen_setup.set_align_planes_goal(hand, gripper_axis, root_normal=world_axis)
        # kitchen_setup.allow_all_collisions()

        kitchen_setup.plan_and_execute()

        kitchen_setup.set_json_goal('Open',
                                    tip_link=hand,
                                    environment_link=handle_name,
                                    goal_joint_state=goal_angle,
                                    # weight=100
                                    )
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_kitchen_js({'sink_area_dish_washer_door_joint': goal_angle})
        # ----------------------------------------------------------------------------------------
        kitchen_setup.set_joint_goal(kitchen_setup.better_pose)
        kitchen_setup.plan_and_execute()

        tray_handle_frame_id = 'iai_kitchen/sink_area_dish_washer_tray_handle_front_side'
        tray_handle_name = 'sink_area_dish_washer_tray_handle_front_side'
        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = tray_handle_frame_id
        bar_axis.vector.y = 1
        bar_axis.vector.z = -0.1

        bar_center = PointStamped()
        bar_center.header.frame_id = tray_handle_frame_id

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = hand
        tip_grasp_axis.vector.z = 1

        kitchen_setup.set_json_goal('GraspBar',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=hand,
                                    tip_grasp_axis=tip_grasp_axis,
                                    bar_center=bar_center,
                                    bar_axis=bar_axis,
                                    bar_length=.3)
        # kitchen_setup.allow_collision([], 'kitchen', [handle_name])
        kitchen_setup.plan_and_execute()

        p = tf.lookup_pose(tray_handle_frame_id, hand)
        p.pose.position.x += 0.3

        # p = tf.transform_pose(hand, p)

        # kitchen_setup.add_json_goal('CartesianPosition',
        #                             root_link=kitchen_setup.default_root,
        #                             tip_link=hand,
        #                             goal=p)
        kitchen_setup.set_cart_goal(p, hand)
        kitchen_setup.plan_and_execute()

        # gripper_axis = Vector3Stamped()
        # gripper_axis.header.frame_id = hand
        # gripper_axis.vector.z = 1
        #
        # world_axis = Vector3Stamped()
        # world_axis.header.frame_id = tray_handle_frame_id
        # world_axis.vector.y = 1
        # kitchen_setup.align_planes(hand, gripper_axis, root_normal=world_axis)
        # kitchen_setup.send_and_check_goal()

        # ------------------------------------------------------------------------------------------
        # kitchen_setup.add_json_goal('Close',
        #                             tip_link=hand,
        #                             object_name='kitchen',
        #                             object_link_name=handle_name)
        # kitchen_setup.allow_all_collisions()
        # kitchen_setup.send_and_check_goal()
        # kitchen_setup.set_kitchen_js({u'sink_area_dish_washer_door_joint': 0})


class TestEASE():

    def test_bowl(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        :return:
        """
        # kernprof -lv py.test -s test/test_integration_pr2.py::TestCollisionAvoidanceGoals::test_bowl_and_cup
        bowl_name = u'bowl'
        cup_name = u'cup'
        percentage = 50
        drawer_handle = u'sink_area_left_middle_drawer_handle'
        drawer_joint = u'sink_area_left_middle_drawer_main_joint'

        # grasp drawer handle
        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = drawer_handle
        bar_axis.vector.y = 1

        bar_center = PointStamped()
        bar_center.header.frame_id = drawer_handle

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = kitchen_setup.l_tip
        tip_grasp_axis.vector.z = 1

        kitchen_setup.set_json_goal(u'GraspBar',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=kitchen_setup.l_tip,
                                    tip_grasp_axis=tip_grasp_axis,
                                    bar_center=bar_center,
                                    bar_axis=bar_axis,
                                    bar_length=0.4)  # TODO: check for real length
        x_gripper = Vector3Stamped()
        x_gripper.header.frame_id = kitchen_setup.l_tip
        x_gripper.vector.x = 1

        x_goal = Vector3Stamped()
        x_goal.header.frame_id = drawer_handle
        x_goal.vector.x = -1

        kitchen_setup.set_align_planes_goal(kitchen_setup.l_tip,
                                            x_gripper,
                                            root_normal=x_goal)
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()

        # open drawer
        kitchen_setup.set_json_goal(u'Open',
                                    tip_link=kitchen_setup.l_tip,
                                    environment_link=drawer_handle)
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_kitchen_js({drawer_joint: 0.48})

        # spawn bowl
        bowl_pose = PoseStamped()
        bowl_pose.header.frame_id = u'iai_kitchen/sink_area_left_middle_drawer_main'
        bowl_pose.pose.position = Point(0.1, -0.2, -.05)
        bowl_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup.add_cylinder(bowl_name, height=0.05, radius=0.07, pose=bowl_pose)
        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.plan_and_execute()

        # grasp bowl
        l_goal = deepcopy(bowl_pose)
        l_goal.pose.position.z += .2
        l_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 1, 0, 0],
                                                                      [0, 0, -1, 0],
                                                                      [-1, 0, 0, 0],
                                                                      [0, 0, 0, 1]]))
        kitchen_setup.set_cart_goal(l_goal, kitchen_setup.l_tip, kitchen_setup.default_root)

        kitchen_setup.plan_and_execute()

        l_goal.pose.position.z -= .2
        r_goal.pose.position.z -= .2
        kitchen_setup.allow_collision([CollisionEntry.ALL], bowl_name, [CollisionEntry.ALL])
        kitchen_setup.allow_collision([CollisionEntry.ALL], cup_name, [CollisionEntry.ALL])
        kitchen_setup.set_cart_goal(l_goal, kitchen_setup.l_tip, kitchen_setup.default_root)
        kitchen_setup.set_cart_goal(r_goal, kitchen_setup.r_tip, kitchen_setup.default_root)
        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=percentage)
        kitchen_setup.plan_and_execute()

        kitchen_setup.attach_object(bowl_name, kitchen_setup.l_tip)
        kitchen_setup.attach_object(cup_name, kitchen_setup.r_tip)

        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.plan_and_execute()
        base_goal = PoseStamped()
        base_goal.header.frame_id = u'base_footprint'
        base_goal.pose.position.x = -.1
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        kitchen_setup.teleport_base(base_goal)

        # place bowl and cup
        bowl_goal = PoseStamped()
        bowl_goal.header.frame_id = u'iai_kitchen/kitchen_island_surface'
        bowl_goal.pose.position = Point(.2, 0, .05)
        bowl_goal.pose.orientation = Quaternion(0, 0, 0, 1)

        cup_goal = PoseStamped()
        cup_goal.header.frame_id = u'iai_kitchen/kitchen_island_surface'
        cup_goal.pose.position = Point(.15, 0.25, .07)
        cup_goal.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup.set_cart_goal(bowl_goal, bowl_name, kitchen_setup.default_root)
        kitchen_setup.set_cart_goal(cup_goal, cup_name, kitchen_setup.default_root)
        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=percentage)
        kitchen_setup.plan_and_execute()

        kitchen_setup.detach_object(bowl_name)
        kitchen_setup.detach_object(cup_name)
        kitchen_setup.allow_collision([], cup_name, [])
        kitchen_setup.allow_collision([], bowl_name, [])
        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.plan_and_execute()

    def look_at(self, kitchen_setup, tip_link=None, goal_point=None):
        tip = u'head_mount_kinect_rgb_link'
        if goal_point is None:
            goal_point = tf.lookup_point(u'map', tip_link)
        pointing_axis = Vector3Stamped()
        pointing_axis.header.frame_id = tip
        pointing_axis.vector.x = 1
        kitchen_setup.pointing(tip, goal_point, pointing_axis=pointing_axis)

    def look_at_while_driving(self, kitchen_setup, tip_link):
        tip = u'head_mount_kinect_rgb_link'
        goal_point = tf.lookup_point(u'map', tip_link)
        pointing_axis = Vector3Stamped()
        pointing_axis.header.frame_id = tip
        pointing_axis.vector.x = 1
        kitchen_setup.pointing(tip, goal_point, pointing_axis=pointing_axis, root_link=tip_link)
        dgaya_pose = deepcopy(gaya_pose)
        del dgaya_pose[u'head_pan_joint']
        del dgaya_pose[u'head_tilt_joint']
        kitchen_setup.set_joint_goal(dgaya_pose)

    def test_grasp_cereal(self, kitchen_setup):
        # FIXME collision avoidance needs soft_threshholds at 0
        cereal_name = u'cereal'
        drawer_frame_id = u'iai_kitchen/oven_area_area_right_drawer_board_2_link'

        # take milk out of fridge
        kitchen_setup.set_kitchen_js({u'oven_area_area_right_drawer_main_joint': 0.48})

        # spawn milk

        cereal_pose = PoseStamped()
        cereal_pose.header.frame_id = drawer_frame_id
        cereal_pose.pose.position = Point(0.123, 0.0, 0.15)
        cereal_pose.pose.orientation = Quaternion(0.0087786, 0.005395, -0.838767, -0.544393)
        kitchen_setup.add_box(cereal_name, [0.1028, 0.0634, 0.20894], cereal_pose)

        drawer_T_box = tf.msg_to_kdl(cereal_pose)

        # grasp milk
        kitchen_setup.open_l_gripper()
        grasp_pose = PoseStamped()
        grasp_pose.header.frame_id = cereal_name
        grasp_pose.pose.position = Point(0.03, 0, 0.05)
        grasp_pose.pose.orientation = Quaternion(0, 0, 1, 0)
        box_T_r_goal = tf.msg_to_kdl(grasp_pose)
        box_T_r_goal_pre = deepcopy(box_T_r_goal)
        box_T_r_goal_pre.p[0] += 0.2

        pre_grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal_pre, drawer_frame_id)

        box_T_r_goal_post = deepcopy(box_T_r_goal)
        box_T_r_goal_post.p[0] += 0.4
        post_grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal_post, drawer_frame_id)

        point = tf.np_to_pose(kitchen_setup.world.get_fk('map', cereal_name)).position
        point_s = PointStamped()
        point_s.header.frame_id = 'map'
        point_s.point = point
        self.look_at(kitchen_setup, goal_point=point_s)

        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup.set_json_goal(u'CartesianPose',
                                    tip_link=kitchen_setup.r_tip,
                                    root_link=kitchen_setup.default_root,
                                    goal=pre_grasp_pose)
        kitchen_setup.plan_and_execute()

        grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal, drawer_frame_id)

        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup.set_cart_goal(grasp_pose, tip_link=kitchen_setup.r_tip)
        kitchen_setup.plan_and_execute()

        kitchen_setup.attach_object(cereal_name, kitchen_setup.r_tip)
        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        kitchen_setup.close_r_gripper()

        # x = Vector3Stamped()
        # x.header.frame_id = 'milk'
        # x.vector.x = 1
        # x_map = Vector3Stamped()
        # x_map.header.frame_id = 'iai_kitchen/iai_fridge_door'
        # x_map.vector.x = 1
        # z = Vector3Stamped()
        # z.header.frame_id = 'milk'
        # z.vector.z = 1
        # z_map = Vector3Stamped()
        # z_map.header.frame_id = 'map'
        # z_map.vector.z = 1
        # kitchen_setup.align_planes('milk', x, root_normal=x_map)
        # kitchen_setup.align_planes('milk', z, root_normal=z_map)
        # kitchen_setup.keep_orientation(u'milk')
        # kitchen_setup.set_cart_goal(grasp_pose, cereal_name, kitchen_setup.default_root)

        point = tf.np_to_pose(kitchen_setup.world.get_fk('map', cereal_name)).position
        point_s = PointStamped()
        point_s.header.frame_id = 'map'
        point_s.point = point
        self.look_at(kitchen_setup, goal_point=point_s)

        kitchen_setup.set_json_goal(u'CartesianPathCarrot',
                                    tip_link=kitchen_setup.r_tip,
                                    root_link=kitchen_setup.default_root,
                                    goal=post_grasp_pose,
                                    goal_sampling_axis=[True, False, False])
        kitchen_setup.plan_and_execute()

        # gaya pose
        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.plan_and_execute()

    def test_grasp_cup(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        :return:
        """
        # kernprof -lv py.test -s test/test_integration_pr2.py::TestCollisionAvoidanceGoals::test_bowl_and_cup
        tip = kitchen_setup.l_tip
        cup_name = u'cup'
        percentage = 50
        drawer_handle = u'kitchen_island_left_upper_drawer_handle'
        drawer_joint = u'kitchen_island_left_upper_drawer_main_joint'

        # grasp drawer handle
        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = drawer_handle
        bar_axis.vector.y = 1

        bar_center = PointStamped()
        bar_center.header.frame_id = drawer_handle

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = tip
        tip_grasp_axis.vector.z = 1

        kitchen_setup.set_json_goal(u'GraspBar',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=tip,
                                    tip_grasp_axis=tip_grasp_axis,
                                    bar_center=bar_center,
                                    bar_axis=bar_axis,
                                    bar_length=0.1)
        x_gripper = Vector3Stamped()
        x_gripper.header.frame_id = tip
        x_gripper.vector.x = 1

        x_goal = Vector3Stamped()
        x_goal.header.frame_id = drawer_handle
        x_goal.vector.x = -1

        kitchen_setup.set_align_planes_goal(tip,
                                            x_gripper,
                                            root_normal=x_goal)
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()

        # open drawer
        kitchen_setup.set_json_goal(u'Open',
                                    tip_link=tip,
                                    environment_link=drawer_handle)
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_kitchen_js({drawer_joint: 0.48})

        # move a bit away from the drawer
        kitchen_setup.allow_collision()
        base_goal = PoseStamped()
        base_goal.header.frame_id = u'base_footprint'
        base_goal.pose.position.x = -0.2
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        kitchen_setup.move_base(base_goal)

        # spawn cup
        cup_pose = PoseStamped()
        cup_pose.header.frame_id = u'iai_kitchen/kitchen_island_left_upper_drawer_main'
        cup_pose.pose.position = Point(0.1, 0.1, 0)
        cup_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup.add_cylinder(cup_name, height=0.07, radius=0.04, pose=cup_pose)

        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.plan_and_execute()

        point = tf.np_to_pose(kitchen_setup.world.get_fk('map', cup_name)).position
        point_s = PointStamped()
        point_s.header.frame_id = 'map'
        point_s.point = point
        self.look_at(kitchen_setup, goal_point=point_s)
        kitchen_setup.plan_and_execute()

        # grasp cup
        # pregrasp
        pregrasp_r_goal = deepcopy(cup_pose)
        pregrasp_r_goal.pose.position.z += .2
        kitchen_setup.set_json_goal('CartesianPreGrasp',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=tip,
                                    grasping_goal=cup_pose,
                                    goal_position=pregrasp_r_goal)
        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=percentage)
        kitchen_setup.plan_and_execute()

        # grasp
        r_goal = deepcopy(tf.np_to_pose_stamped(kitchen_setup.robot.get_fk(kitchen_setup.default_root, tip),
                                                kitchen_setup.default_root))
        r_goal.pose.position.z -= .2
        kitchen_setup.allow_collision([CollisionEntry.ALL], cup_name, [CollisionEntry.ALL])
        kitchen_setup.set_cart_goal(r_goal, tip, kitchen_setup.default_root)
        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=percentage)
        kitchen_setup.plan_and_execute()

        # attach cup
        kitchen_setup.attach_object(cup_name, tip)
        kitchen_setup.close_l_gripper()

        # pregrasp
        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=percentage)
        kitchen_setup.set_json_goal('CartesianPreGrasp',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=tip,
                                    grasping_goal=cup_pose,
                                    goal_position=pregrasp_r_goal)
        kitchen_setup.plan_and_execute()

        # gaya pose
        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.plan_and_execute()

        # base_goal = PoseStamped()
        # base_goal.header.frame_id = u'base_footprint'
        # base_goal.pose.position.y = -.6
        # base_goal.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        # kitchen_setup.teleport_base(base_goal)
        # kitchen_setup.wait_heartbeats(10)

        # place cup
        # cup_goal = PoseStamped()
        # cup_goal.header.frame_id = u'iai_kitchen/kitchen_island_surface'
        # cup_goal.pose.position = Point(.15, 0.25, .07)
        # cup_goal.pose.orientation = Quaternion(0, 0, 0, 1)

        # kitchen_setup.set_cart_goal(cup_goal, cup_name, kitchen_setup.default_root)
        # kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=percentage)
        # kitchen_setup.plan_and_execute()

        # kitchen_setup.detach_object(cup_name)
        # kitchen_setup.allow_collision([], cup_name, [])
        # kitchen_setup.set_joint_goal(gaya_pose)
        # kitchen_setup.plan_and_execute()

    def test_grasp_milk(self, kitchen_setup):
        tip_link = kitchen_setup.l_tip
        milk_name = u'milk'

        # take milk out of fridge
        kitchen_setup.set_kitchen_js({u'iai_fridge_door_joint': 1.56})

        # spawn milk
        milk_pose = PoseStamped()
        milk_pose.header.frame_id = u'iai_kitchen/iai_fridge_door_shelf1_bottom'
        milk_pose.pose.position = Point(0, 0, 0.13)
        milk_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup.add_box(milk_name, [0.05, 0.05, 0.2], milk_pose)

        point = tf.np_to_pose(kitchen_setup.world.get_fk('map', milk_name)).position
        point_s = PointStamped()
        point_s.header.frame_id = 'map'
        point_s.point = point
        self.look_at(kitchen_setup, goal_point=point_s)
        kitchen_setup.plan_and_execute()

        # move arm towards milk
        if tip_link == kitchen_setup.l_tip:
            kitchen_setup.open_l_gripper()
        elif tip_link == kitchen_setup.r_tip:
            kitchen_setup.open_r_gripper()
        else:
            raise Exception('Wrong tip link')

        kitchen_setup.set_json_goal('CartesianPreGrasp',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=tip_link,
                                    grasping_goal=milk_pose)
        kitchen_setup.plan_and_execute()

        kitchen_setup.set_json_goal(u'CartesianPose',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=tip_link,
                                    goal=milk_pose)
        kitchen_setup.plan_and_execute()

        # grasp milk
        kitchen_setup.attach_object(milk_name, tip_link)

        if tip_link == kitchen_setup.l_tip:
            kitchen_setup.close_l_gripper()
        elif tip_link == kitchen_setup.r_tip:
            kitchen_setup.close_r_gripper()
        else:
            raise Exception('Wrong tip link')

        kitchen_setup.set_json_goal('CartesianPreGrasp',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=tip_link,
                                    grasping_goal=milk_pose,
                                    dist=0.1)
        kitchen_setup.plan_and_execute()

        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.plan_and_execute()

    def test_move_to_cup(self, kitchen_setup):
        # teleport to cup drawer
        base_goal = PoseStamped()
        base_goal.header.frame_id = u'map'
        base_goal.pose.position.x = 0.5
        base_goal.pose.position.y = 1.0
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        kitchen_setup.set_cart_goal(base_goal, tip_link='base_footprint')
        kitchen_setup.plan_and_execute()

    def test_move_to_fridge(self, kitchen_setup):
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 0.3
        base_goal.pose.position.y = -0.7
        base_goal.pose.orientation.z = -0.51152562713
        base_goal.pose.orientation.w = 0.85926802151
        kitchen_setup.set_cart_goal(base_goal, tip_link='base_footprint')
        kitchen_setup.plan_and_execute()

    def test_open_cereal_drawer(self, kitchen_setup):
        handle_name = [
            # u'oven_area_area_middle_upper_drawer_handle',
            # u'oven_area_area_middle_lower_drawer_handle',
            # u'sink_area_left_upper_drawer_handle',
            # u'sink_area_left_middle_drawer_handle',
            # u'sink_area_left_bottom_drawer_handle',
            # u'sink_area_trash_drawer_handle',
            # u'fridge_area_lower_drawer_handle',
            # u'kitchen_island_left_upper_drawer_handle',
            # u'kitchen_island_left_lower_drawer_handle',
            # u'kitchen_island_middle_upper_drawer_handle',
            # u'kitchen_island_middle_lower_drawer_handle',
            # u'kitchen_island_right_upper_drawer_handle',
            # u'kitchen_island_right_lower_drawer_handle',
            u'oven_area_area_right_drawer_handle',
            # u'oven_area_area_right_drawer_handle'
        ]

        handle_frame_id = [u'iai_kitchen/' + item for item in handle_name]
        joint_name = [item.replace(u'handle', u'main_joint') for item in handle_name]

        for i_handle_id, i_handle_name, i_joint_name in zip(handle_frame_id, handle_name, joint_name):
            bar_axis = Vector3Stamped()
            bar_axis.header.frame_id = i_handle_id
            bar_axis.vector.y = 1

            bar_center = PointStamped()
            bar_center.header.frame_id = i_handle_id

            tip_grasp_axis = Vector3Stamped()
            tip_grasp_axis.header.frame_id = kitchen_setup.l_tip
            tip_grasp_axis.vector.z = 1

            self.look_at(kitchen_setup, tip_link=i_handle_id)

            kitchen_setup.set_json_goal(u'GraspBar',
                                        root_link=kitchen_setup.default_root,
                                        tip_link=kitchen_setup.l_tip,
                                        tip_grasp_axis=tip_grasp_axis,
                                        bar_center=bar_center,
                                        bar_axis=bar_axis,
                                        bar_length=0.4)  # TODO: check for real length
            x_gripper = Vector3Stamped()
            x_gripper.header.frame_id = kitchen_setup.l_tip
            x_gripper.vector.x = 1

            x_goal = Vector3Stamped()
            x_goal.header.frame_id = i_handle_id
            x_goal.vector.x = -1

            kitchen_setup.set_align_planes_goal(kitchen_setup.l_tip,
                                                x_gripper,
                                                root_normal=x_goal)
            kitchen_setup.allow_all_collisions()
            kitchen_setup.avoid_self_collision()
            kitchen_setup.plan_and_execute()

            kitchen_setup.set_json_goal(u'Open',
                                        tip_link=kitchen_setup.l_tip,
                                        environment_link=i_handle_name)
            kitchen_setup.allow_all_collisions()  # makes execution faster
            kitchen_setup.avoid_self_collision()
            kitchen_setup.plan_and_execute()  # send goal to Giskard
            # Update kitchen object
            kitchen_setup.set_kitchen_js({i_joint_name: 0.48})  # TODO: get real value from URDF
            kitchen_setup.allow_collision()
            r_goal = deepcopy(tf.lookup_pose(kitchen_setup.default_root, 'base_footprint'))
            r_goal.pose.position.x -= 0.2
            kitchen_setup.move_base(r_goal)
            kitchen_setup.set_joint_goal(gaya_pose)
            kitchen_setup.plan_and_execute()

    def test_close_cereal_drawer(self, kitchen_setup):
        handle_name = [
            # u'oven_area_area_middle_upper_drawer_handle',
            # u'oven_area_area_middle_lower_drawer_handle',
            # u'sink_area_left_upper_drawer_handle',
            # u'sink_area_left_middle_drawer_handle',
            # u'sink_area_left_bottom_drawer_handle',
            # u'sink_area_trash_drawer_handle',
            # u'fridge_area_lower_drawer_handle',
            # u'kitchen_island_left_upper_drawer_handle',
            # u'kitchen_island_left_lower_drawer_handle',
            # u'kitchen_island_middle_upper_drawer_handle',
            # u'kitchen_island_middle_lower_drawer_handle',
            # u'kitchen_island_right_upper_drawer_handle',
            # u'kitchen_island_right_lower_drawer_handle',
            u'oven_area_area_right_drawer_handle',
            # u'oven_area_area_right_drawer_handle'
        ]

        handle_frame_id = [u'iai_kitchen/' + item for item in handle_name]
        joint_name = [item.replace(u'handle', u'main_joint') for item in handle_name]

        for i_handle_id, i_handle_name, i_joint_name in zip(handle_frame_id, handle_name, joint_name):
            bar_axis = Vector3Stamped()
            bar_axis.header.frame_id = i_handle_id
            bar_axis.vector.y = 1

            bar_center = PointStamped()
            bar_center.header.frame_id = i_handle_id

            tip_grasp_axis = Vector3Stamped()
            tip_grasp_axis.header.frame_id = kitchen_setup.l_tip
            tip_grasp_axis.vector.z = 1

            self.look_at(kitchen_setup, tip_link=i_handle_id)

            kitchen_setup.set_json_goal(u'GraspBar',
                                        root_link=kitchen_setup.default_root,
                                        tip_link=kitchen_setup.l_tip,
                                        tip_grasp_axis=tip_grasp_axis,
                                        bar_center=bar_center,
                                        bar_axis=bar_axis,
                                        bar_length=0.4)  # TODO: check for real length
            x_gripper = Vector3Stamped()
            x_gripper.header.frame_id = kitchen_setup.l_tip
            x_gripper.vector.x = 1

            x_goal = Vector3Stamped()
            x_goal.header.frame_id = i_handle_id
            x_goal.vector.x = -1

            kitchen_setup.set_align_planes_goal(kitchen_setup.l_tip,
                                                x_gripper,
                                                root_normal=x_goal)
            kitchen_setup.allow_all_collisions()
            kitchen_setup.avoid_self_collision()
            kitchen_setup.plan_and_execute()

            kitchen_setup.set_json_goal(u'Close',
                                        tip_link=kitchen_setup.l_tip,
                                        object_name=u'kitchen',
                                        environment_link=i_handle_name)
            kitchen_setup.allow_all_collisions()  # makes execution faster
            kitchen_setup.avoid_self_collision()
            kitchen_setup.plan_and_execute()  # send goal to Giskard
            # Update kitchen object
            kitchen_setup.set_kitchen_js({i_joint_name: 0.0})  # TODO: get real value from URDF
            kitchen_setup.allow_collision()
            r_goal = deepcopy(tf.lookup_pose(kitchen_setup.default_root, 'base_footprint'))
            r_goal.pose.position.x -= 0.2
            kitchen_setup.move_base(r_goal)
            kitchen_setup.set_joint_goal(gaya_pose)
            kitchen_setup.plan_and_execute()

    def test_open_fridge(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        handle_frame_id = u'iai_kitchen/iai_fridge_door_handle'
        handle_name = u'iai_fridge_door_handle'

        base_goal = PoseStamped()
        base_goal.header.frame_id = u'map'
        base_goal.pose.position = Point(0.3, -0.5, 0)
        base_goal.pose.orientation.w = 1
        kitchen_setup.teleport_base(base_goal)

        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = handle_frame_id
        bar_axis.vector.z = 1

        bar_center = PointStamped()
        bar_center.header.frame_id = handle_frame_id

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = kitchen_setup.r_tip
        tip_grasp_axis.vector.z = 1

        self.look_at(kitchen_setup, tip_link=handle_frame_id)

        kitchen_setup.set_json_goal(u'GraspBar',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=kitchen_setup.r_tip,
                                    tip_grasp_axis=tip_grasp_axis,
                                    bar_center=bar_center,
                                    bar_axis=bar_axis,
                                    bar_length=.4)
        x_gripper = Vector3Stamped()
        x_gripper.header.frame_id = kitchen_setup.r_tip
        x_gripper.vector.x = 1

        x_goal = Vector3Stamped()
        x_goal.header.frame_id = handle_frame_id
        x_goal.vector.x = -1
        kitchen_setup.set_align_planes_goal(kitchen_setup.r_tip, x_gripper, root_normal=x_goal)
        kitchen_setup.allow_all_collisions()
        # kitchen_setup.add_json_goal(u'AvoidJointLimits', percentage=10)
        kitchen_setup.plan_and_execute()

        kitchen_setup.set_json_goal(u'Open',
                                    tip_link=kitchen_setup.r_tip,
                                    environment_link=handle_name,
                                    goal_joint_state=1.5)
        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup.allow_all_collisions()
        # kitchen_setup.add_json_goal(u'AvoidJointLimits')
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_kitchen_js({u'iai_fridge_door_joint': 1.5})

        kitchen_setup.avoid_all_collisions()
        r_goal = deepcopy(tf.lookup_pose(kitchen_setup.default_root, handle_frame_id))
        kitchen_setup.set_json_goal('CartesianPreGrasp',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=kitchen_setup.r_tip,
                                    grasping_goal=r_goal,
                                    dist=0.025)
        kitchen_setup.plan_and_execute()

        base_goal = PoseStamped()
        base_goal.header.frame_id = u'base_footprint'
        base_goal.pose.position.x = -0.2
        base_goal.pose.orientation.w = 1
        kitchen_setup.set_cart_goal(base_goal, tip_link='base_footprint')
        kitchen_setup.plan_and_execute()

        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.plan_and_execute()

    def test_move_to_dining_area(self, kitchen_setup, left):
        table_navigation_link = 'iai_kitchen/dining_area_footprint'
        tip_link = u'base_footprint'

        # spawn milk
        table_navigation_goal = PoseStamped()
        table_navigation_goal.header.frame_id = table_navigation_link
        if left:
            table_navigation_goal.pose.position = Point(0.15, -0.95, 0.0)
        else:
            table_navigation_goal.pose.position = Point(-0.15, -0.95, 0.0)
        table_navigation_goal.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, np.pi / 2.))

        kitchen_setup.set_json_goal(u'CartesianPose',
                                    tip_link=tip_link,
                                    root_link=kitchen_setup.default_root,
                                    goal=table_navigation_goal
                                    )

        kitchen_setup.plan_and_execute()

    def test_place_on_dining_area_table(self, kitchen_setup, z_offset, left=False, detach_object_name=None):
        # move to dining table
        self.test_move_to_dining_area(kitchen_setup, left)
        # go in gaya pose
        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.plan_and_execute()
        # place
        # preplacing
        if left:
            tip = kitchen_setup.l_tip
        else:
            tip = kitchen_setup.r_tip
        g = PoseStamped()
        g.header.frame_id = 'iai_kitchen/dining_area_jokkmokk_table_main'
        g.pose.position.y = -0.3
        if left:
            g.pose.position.x = -0.2
        else:
            g.pose.position.x = 0.2
        g.pose.position.z = 0.35 + z_offset
        pregrasp_r_goal = deepcopy(g)
        pregrasp_r_goal.pose.position.z += .2
        kitchen_setup.set_json_goal('CartesianPreGrasp',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=tip,
                                    grasping_goal=g,
                                    goal_position=pregrasp_r_goal)
        # kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=percentage)
        kitchen_setup.plan_and_execute()

        # placing
        r_goal = deepcopy(tf.np_to_pose_stamped(kitchen_setup.robot.get_fk(kitchen_setup.default_root, tip),
                                                kitchen_setup.default_root))
        r_goal.pose.position.z -= 0.2
        kitchen_setup.set_cart_goal(r_goal, tip)
        kitchen_setup.plan_and_execute()
        if detach_object_name:
            kitchen_setup.detach_object(detach_object_name)
        if left:
            kitchen_setup.open_l_gripper()
        else:
            kitchen_setup.open_r_gripper()
        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.plan_and_execute()

    def test_fetch_milk(self, kitchen_setup):
        self.test_open_fridge(kitchen_setup)
        self.test_move_to_fridge(kitchen_setup)
        self.test_grasp_milk(kitchen_setup)

    def test_fetch_cereal(self, kitchen_setup):
        self.test_open_cereal_drawer(kitchen_setup)
        self.test_grasp_cereal(kitchen_setup)
        self.test_close_cereal_drawer(kitchen_setup)

    def test_deliver_milk(self, kitchen_setup):
        self.test_fetch_milk(kitchen_setup)
        self.test_move_to_dining_area(kitchen_setup)

    def test_deliver_cup(self, kitchen_setup):
        left = True
        tip_link = 'l_gripper_tool_frame'
        self.test_move_to_cup(kitchen_setup)
        self.test_grasp_cup(kitchen_setup)
        self.look_at_while_driving(kitchen_setup, tip_link)
        self.test_place_on_dining_area_table(kitchen_setup, 0.1, left=left)

    def test_deliver_cereal_cup(self, kitchen_setup):
        self.test_fetch_cereal(kitchen_setup)
        self.test_move_to_cup(kitchen_setup)
        self.test_grasp_cup(kitchen_setup)
        self.look_at_while_driving(kitchen_setup, 'l_gripper_tool_frame')
        self.test_place_on_dining_area_table(kitchen_setup, 0.1, left=True, detach_object_name='cup')
        self.test_place_on_dining_area_table(kitchen_setup, 0.2, left=False, detach_object_name='cereal')


class TestInfoServices(object):
    def test_get_object_info(self, zero_pose: PR2):
        result = zero_pose.get_group_info('robot')
        expected = {'head_pan_joint',
                    'head_tilt_joint',
                    'l_elbow_flex_joint',
                    'l_forearm_roll_joint',
                    'l_shoulder_lift_joint',
                    'l_shoulder_pan_joint',
                    'l_upper_arm_roll_joint',
                    'l_wrist_flex_joint',
                    'l_wrist_roll_joint',
                    'odom_x_joint',
                    'odom_y_joint',
                    'odom_z_joint',
                    'r_elbow_flex_joint',
                    'r_forearm_roll_joint',
                    'r_shoulder_lift_joint',
                    'r_shoulder_pan_joint',
                    'r_upper_arm_roll_joint',
                    'r_wrist_flex_joint',
                    'r_wrist_roll_joint',
                    'torso_lift_joint'}
        assert set(result.controlled_joints) == expected

# time: *[1-9].
# import pytest
# pytest.main(['-s', __file__ + '::TestJointGoals::test_joint_movement1'])
# pytest.main(['-s', __file__ + '::TestCollisionAvoidanceGoals::test_bowl_and_cup'])
# pytest.main(['-s', __file__ + '::TestCollisionAvoidanceGoals::test_attached_collision2'])
# pytest.main(['-s', __file__ + '::TestCollisionAvoidanceGoals::test_avoid_self_collision'])
# pytest.main(['-s', __file__ + '::TestCollisionAvoidanceGoals::test_avoid_collision_at_kitchen_corner'])
# pytest.main(['-s', __file__ + '::TestCartesianPath::test_ease_fridge_with_cart_goals_and_global_planner'])
# pytest.main(['-s', __file__ + '::TestCartGoals::test_keep_position3'])
