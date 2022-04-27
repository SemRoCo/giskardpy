from __future__ import division

import itertools
import re
from copy import deepcopy

import numpy as np
import pytest
import roslaunch
import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Vector3Stamped, PointStamped
from numpy import pi
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from tf.transformations import quaternion_from_matrix, quaternion_about_axis, quaternion_from_euler

import giskardpy.utils.tfwrapper as tf
from giskard_msgs.msg import CollisionEntry, MoveResult, WorldBody, MoveGoal
from giskard_msgs.srv import UpdateWorldResponse, UpdateWorldRequest
from giskardpy import identifier, RobotName
from giskardpy.goals.goal import WEIGHT_ABOVE_CA, WEIGHT_BELOW_CA, WEIGHT_COLLISION_AVOIDANCE
from giskardpy.identifier import fk_pose
from giskardpy.python_interface import DEFAULT_WORLD_TIMEOUT
from giskardpy.utils import logging
from giskardpy.utils.tfwrapper import init as tf_init
from test_integration_pr2_without_base import gaya_pose
from utils_for_tests import PR2, compare_poses, compare_points, compare_orientations, publish_marker_vector, \
    JointGoalChecker

# TODO roslaunch iai_pr2_sim ros_control_sim_with_base.launch
# TODO roslaunch iai_kitchen upload_kitchen_obj.launch

# scopes = ['module', 'class', 'function']
pocky_pose = {u'r_elbow_flex_joint': -1.29610152504,
              u'r_forearm_roll_joint': -0.0301682323805,
              u'r_shoulder_lift_joint': 1.20324921318,
              u'r_shoulder_pan_joint': -0.73456435706,
              u'r_upper_arm_roll_joint': -0.70790051778,
              u'r_wrist_flex_joint': -0.10001,
              u'r_wrist_roll_joint': 0.258268529825,

              u'l_elbow_flex_joint': -1.29610152504,
              u'l_forearm_roll_joint': 0.0301682323805,
              u'l_shoulder_lift_joint': 1.20324921318,
              u'l_shoulder_pan_joint': 0.73456435706,
              u'l_upper_arm_roll_joint': 0.70790051778,
              u'l_wrist_flex_joint': -0.1001,
              u'l_wrist_roll_joint': -0.258268529825,

              u'torso_lift_joint': 0.2,
              u'head_pan_joint': 0,
              u'head_tilt_joint': 0,
              }

default_pose = {u'r_elbow_flex_joint': -0.15,
                u'r_forearm_roll_joint': 0,
                u'r_shoulder_lift_joint': 0,
                u'r_shoulder_pan_joint': 0,
                u'r_upper_arm_roll_joint': 0,
                u'r_wrist_flex_joint': -0.10001,
                u'r_wrist_roll_joint': 0,

                u'l_elbow_flex_joint': -0.15,
                u'l_forearm_roll_joint': 0,
                u'l_shoulder_lift_joint': 0,
                u'l_shoulder_pan_joint': 0,
                u'l_upper_arm_roll_joint': 0,
                u'l_wrist_flex_joint': -0.10001,
                u'l_wrist_roll_joint': 0,

                u'torso_lift_joint': 0.2,
                u'head_pan_joint': 0,
                u'head_tilt_joint': 0,
                }

gaya_pose = {u'r_shoulder_pan_joint': -1.7125,
             u'r_shoulder_lift_joint': -0.25672,
             u'r_upper_arm_roll_joint': -1.46335,
             u'r_elbow_flex_joint': -2.12,
             u'r_forearm_roll_joint': 1.76632,
             u'r_wrist_flex_joint': -0.10001,
             u'r_wrist_roll_joint': 0.05106,
             u'l_shoulder_pan_joint': 1.9652,
             u'l_shoulder_lift_joint': - 0.26499,
             u'l_upper_arm_roll_joint': 1.3837,
             u'l_elbow_flex_joint': -2.12,
             u'l_forearm_roll_joint': 16.99,
             u'l_wrist_flex_joint': - 0.10001,
             u'l_wrist_roll_joint': 0,
             u'torso_lift_joint': 0.2,

             u'head_pan_joint': 0,
             u'head_tilt_joint': 0,
             }

pick_up_pose = {
    u'head_pan_joint': -2.46056758502e-16,
    u'head_tilt_joint': -1.97371778181e-16,
    u'l_elbow_flex_joint': -0.962150355946,
    u'l_forearm_roll_joint': 1.44894622393,
    u'l_shoulder_lift_joint': -0.273579583084,
    u'l_shoulder_pan_joint': 0.0695426768038,
    u'l_upper_arm_roll_joint': 1.3591238067,
    u'l_wrist_flex_joint': -1.9004529902,
    u'l_wrist_roll_joint': 2.23732576003,
    u'r_elbow_flex_joint': -2.1207193579,
    u'r_forearm_roll_joint': 1.76628402882,
    u'r_shoulder_lift_joint': -0.256729037039,
    u'r_shoulder_pan_joint': -1.71258744959,
    u'r_upper_arm_roll_joint': -1.46335011257,
    u'r_wrist_flex_joint': -0.100010762609,
    u'r_wrist_roll_joint': 0.0509923457388,
    u'torso_lift_joint': 0.261791330751,
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


folder_name = u'tmp_data/'


@pytest.fixture(scope=u'module')
def ros(request):
    try:
        logging.loginfo(u'deleting tmp test folder')
        # shutil.rmtree(folder_name)
    except Exception:
        pass

    logging.loginfo(u'init ros')
    # rospy.init_node('tests', log_level=rospy.DEBUG)
    rospy.init_node('tests')
    tf_init(60)
    launch = roslaunch.scriptapi.ROSLaunch()
    launch.start()

    rospy.set_param('/joint_trajectory_splitter/state_topics',
                    ['/whole_body_controller/base/state',
                     '/whole_body_controller/body/state'])
    rospy.set_param('/joint_trajectory_splitter/client_topics',
                    ['/whole_body_controller/base/follow_joint_trajectory',
                     '/whole_body_controller/body/follow_joint_trajectory'])
    node = roslaunch.core.Node('giskardpy', 'joint_trajectory_splitter.py', name='joint_trajectory_splitter')
    joint_trajectory_splitter = launch.launch(node)

    def kill_ros():
        joint_trajectory_splitter.stop()
        rospy.delete_param('/joint_trajectory_splitter/state_topics')
        rospy.delete_param('/joint_trajectory_splitter/client_topics')
        logging.loginfo(u'shutdown ros')
        rospy.signal_shutdown('die')
        try:
            logging.loginfo(u'deleting tmp test folder')
            # shutil.rmtree(folder_name)
        except Exception:
            pass

    request.addfinalizer(kill_ros)


@pytest.fixture(scope=u'module')
def giskard(request, ros):
    c = PR2()
    request.addfinalizer(c.tear_down)
    return c


@pytest.fixture()
def resetted_giskard(giskard):
    """
    :type giskard: PR2
    """
    logging.loginfo(u'resetting giskard')
    giskard.open_l_gripper()
    giskard.open_r_gripper()
    giskard.clear_world()
    giskard.reset_base()
    return giskard


@pytest.fixture()
def zero_pose(resetted_giskard):
    """
    :type resetted_giskard: PR2
    """
    resetted_giskard.allow_all_collisions()
    resetted_giskard.set_joint_goal(default_pose)
    resetted_giskard.plan_and_execute()
    return resetted_giskard


@pytest.fixture()
def pocky_pose_setup(resetted_giskard):
    """
    :type resetted_giskard: PR2
    """
    resetted_giskard.set_joint_goal(pocky_pose)
    resetted_giskard.allow_all_collisions()
    resetted_giskard.plan_and_execute()
    return resetted_giskard


@pytest.fixture()
def box_setup(pocky_pose_setup):
    """
    :type pocky_pose_setup: PR2
    :rtype: PR2
    """
    p = PoseStamped()
    p.header.frame_id = u'map'
    p.pose.position.x = 1.2
    p.pose.position.y = 0
    p.pose.position.z = 0.5
    p.pose.orientation.w = 1
    pocky_pose_setup.add_box(size=[1, 1, 1], pose=p)
    return pocky_pose_setup


@pytest.fixture()
def fake_table_setup(pocky_pose_setup):
    """
    :type pocky_pose_setup: PR2
    :rtype: PR2
    """
    p = PoseStamped()
    p.header.frame_id = u'map'
    p.pose.position.x = 1.2
    p.pose.position.y = 0
    p.pose.position.z = 0.3
    p.pose.orientation.w = 1
    pocky_pose_setup.add_box(pose=p)
    return pocky_pose_setup

@pytest.fixture()
def kitchen_setup_avoid_collisions(resetted_giskard):
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
def kitchen_setup(resetted_giskard):
    """
    :type resetted_giskard: PR2
    :return:
    """
    resetted_giskard.allow_all_collisions()
    resetted_giskard.set_joint_goal(gaya_pose)
    resetted_giskard.plan_and_execute()
    object_name = u'kitchen'
    resetted_giskard.add_urdf(object_name, rospy.get_param(u'kitchen_description'),
                              tf.lookup_pose(u'map', u'iai_kitchen/world'), u'/kitchen/joint_states',
                              set_js_topic=u'/kitchen/cram_joint_states')
    js = {str(k): 0.0 for k in resetted_giskard.world.groups[object_name].movable_joints}
    resetted_giskard.set_kitchen_js(js)
    return resetted_giskard


def decrease_external_collision_avoidance(kitchen_setup_avoid_collisions):
    eca = kitchen_setup_avoid_collisions.god_map.get_data(identifier.external_collision_avoidance)
    n_eca = deepcopy(eca)
    for joint_name, _ in eca.items():
        if eca[joint_name] == eca.default_factory():
            n_eca.pop(joint_name)
        else:
            n_eca[joint_name]['soft_threshold'] /= 10
    default = deepcopy(eca.default_factory())
    default['soft_threshold'] /= 10
    n_eca.default_factory = lambda: default
    kitchen_setup_avoid_collisions.god_map.set_data(identifier.external_collision_avoidance, n_eca)


def increase_external_collision_avoidance(kitchen_setup_avoid_collisions):
    eca = kitchen_setup_avoid_collisions.god_map.get_data(identifier.external_collision_avoidance)
    n_eca = deepcopy(eca)
    for joint_name, _ in eca.items():
        if eca[joint_name] == eca.default_factory():
            n_eca.pop(joint_name)
        else:
            n_eca[joint_name]['soft_threshold'] *= 10
    default = deepcopy(eca.default_factory())
    default['soft_threshold'] *= 10
    n_eca.default_factory = lambda: default
    kitchen_setup_avoid_collisions.god_map.set_data(identifier.external_collision_avoidance, n_eca)


class TestFk(object):
    def test_fk(self, zero_pose):
        for root, tip in itertools.product(zero_pose.robot.link_names, repeat=2):
            fk1 = zero_pose.god_map.get_data(fk_pose + [(root, tip)])
            fk2 = tf.lookup_pose(str(root), str(tip))
            compare_poses(fk1.pose, fk2.pose)

    def test_fk_attached(self, zero_pose):
        pocky = u'box'
        zero_pose.attach_box(pocky, [0.1, 0.02, 0.02], zero_pose.r_tip, [0.05, 0, 0], [1, 0, 0, 0])
        for root, tip in itertools.product(zero_pose.robot.link_names, [pocky]):
            fk1 = zero_pose.god_map.get_data(fk_pose + [(root, tip)])
            fk2 = tf.lookup_pose(str(root), str(tip))
            compare_poses(fk1.pose, fk2.pose)


class TestJointGoals(object):
    def test_joint_movement1(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.allow_all_collisions()
        zero_pose.set_joint_goal(pocky_pose)
        zero_pose.plan_and_execute()

    def test_partial_joint_state_goal1(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.allow_self_collision()
        js = dict(list(pocky_pose.items())[:3])
        zero_pose.set_joint_goal(js)
        zero_pose.plan_and_execute()

    def test_continuous_joint1(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.allow_self_collision()
        js = {u'r_wrist_roll_joint': -pi,
              u'l_wrist_roll_joint': -2.1 * pi, }
        zero_pose.set_joint_goal(js)
        zero_pose.plan_and_execute()

    def test_prismatic_joint1(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.allow_self_collision()
        js = {u'torso_lift_joint': 0.1}
        zero_pose.set_joint_goal(js)
        zero_pose.plan_and_execute()

    def test_hard_joint_limits(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.allow_self_collision()
        r_elbow_flex_joint_limits = zero_pose.robot.get_joint_position_limits('r_elbow_flex_joint')
        torso_lift_joint_limits = zero_pose.robot.get_joint_position_limits('torso_lift_joint')
        head_pan_joint_limits = zero_pose.robot.get_joint_position_limits('head_pan_joint')

        goal_js = {u'r_elbow_flex_joint': r_elbow_flex_joint_limits[0] - 0.2,
                   u'torso_lift_joint': torso_lift_joint_limits[0] - 0.2,
                   u'head_pan_joint': head_pan_joint_limits[0] - 0.2}
        zero_pose.set_joint_goal(goal_js, check=False)
        zero_pose.plan_and_execute()

        goal_js = {u'r_elbow_flex_joint': r_elbow_flex_joint_limits[1] + 0.2,
                   u'torso_lift_joint': torso_lift_joint_limits[1] + 0.2,
                   u'head_pan_joint': head_pan_joint_limits[1] + 0.2}

        zero_pose.set_joint_goal(goal_js, check=False)
        zero_pose.plan_and_execute()

    # TODO test goal for unknown joint


class TestConstraints(object):
    # TODO write buggy constraints that test sanity checks
    def test_JointPositionRange(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        joint_name = u'head_pan_joint'
        lower_limit, upper_limit = zero_pose.robot.joints[joint_name].position_limits
        lower_limit *= 0.5
        upper_limit *= 0.5
        zero_pose.set_json_goal(u'JointPositionRange',
                                joint_name=joint_name,
                                upper_limit=upper_limit,
                                lower_limit=lower_limit)
        zero_pose.set_joint_goal({
            joint_name: 2
        }, check=False)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()
        assert zero_pose.robot.state[joint_name].position <= upper_limit + 2e-3
        assert zero_pose.robot.state[joint_name].position >= lower_limit - 2e-3

        zero_pose.set_json_goal(u'JointPositionRange',
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

        zero_pose.set_json_goal(u'JointPositionRange',
                                joint_name=joint_name,
                                upper_limit=10,
                                lower_limit=9)
        zero_pose.set_joint_goal({
            joint_name: 0
        }, check=False)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.CONSTRAINT_INITIALIZATION_ERROR])

    def test_CollisionAvoidanceHint(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        # FIXME bouncy
        tip = u'base_footprint'
        base_pose = PoseStamped()
        base_pose.header.frame_id = u'map'
        base_pose.pose.position.x = 0
        base_pose.pose.position.y = 1.5
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0, 0, 1]))
        kitchen_setup.teleport_base(base_pose)
        base_pose = PoseStamped()
        base_pose.header.frame_id = tip
        base_pose.pose.position.x = 2.3
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))

        avoidance_hint = Vector3Stamped()
        avoidance_hint.header.frame_id = u'map'
        avoidance_hint.vector.y = -1
        kitchen_setup.avoid_all_collisions(0.1)
        kitchen_setup.set_json_goal(u'CollisionAvoidanceHint',
                                    tip_link=u'base_link',
                                    max_threshold=0.4,
                                    spring_threshold=0.5,
                                    # max_linear_velocity=1,
                                    object_name=u'kitchen',
                                    object_link_name=u'kitchen_island',
                                    weight=WEIGHT_COLLISION_AVOIDANCE,
                                    avoidance_hint=avoidance_hint)
        kitchen_setup.set_joint_goal(gaya_pose)

        kitchen_setup.set_cart_goal(base_pose, tip, weight=WEIGHT_BELOW_CA, linear_velocity=0.5)
        kitchen_setup.plan_and_execute()

    def test_CartesianPosition(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        tip = zero_pose.r_tip
        p = PoseStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = tip
        p.pose.position = Point(-0.4, -0.2, -0.3)
        p.pose.orientation = Quaternion(0, 0, 0, 1)

        expected = tf.transform_pose('map', p)

        zero_pose.allow_all_collisions()
        zero_pose.set_json_goal(u'CartesianPosition',
                                root_link=zero_pose.default_root,
                                tip_link=tip,
                                goal=p)
        zero_pose.plan_and_execute()
        new_pose = tf.lookup_pose('map', tip)
        compare_points(expected.pose.position, new_pose.pose.position)

    def test_CartesianPose(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        tip = zero_pose.r_tip
        p = PoseStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = tip
        p.pose.position = Point(-0.4, -0.2, -0.3)
        p.pose.orientation = Quaternion(0, 0, 1, 0)

        expected = tf.transform_pose('map', p)

        zero_pose.allow_all_collisions()
        zero_pose.set_json_goal(u'CartesianPose',
                                root_link=zero_pose.default_root,
                                tip_link=tip,
                                goal=p)
        zero_pose.plan_and_execute()
        new_pose = tf.lookup_pose('map', tip)
        compare_points(expected.pose.position, new_pose.pose.position)

    def test_JointPositionRevolute(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        joint = 'r_shoulder_lift_joint'
        joint_goal = 1
        zero_pose.allow_all_collisions()
        zero_pose.set_json_goal(u'JointPositionRevolute',
                                joint_name=joint,
                                goal=joint_goal,
                                max_velocity=0.5)
        zero_pose.plan_and_execute()
        np.testing.assert_almost_equal(zero_pose.robot.state[joint].position, joint_goal, decimal=3)

    def test_JointPositionContinuous(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        joint = 'odom_z_joint'
        joint_goal = 4
        zero_pose.allow_all_collisions()
        zero_pose.set_json_goal(u'JointPositionContinuous',
                                joint_name=joint,
                                goal=joint_goal,
                                max_velocity=1)
        zero_pose.plan_and_execute()
        np.testing.assert_almost_equal(zero_pose.robot.state[joint].position, -2.283, decimal=2)

    def test_JointPosition_kitchen(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        joint_name1 = 'iai_fridge_door_joint'
        joint_name2 = 'sink_area_left_upper_drawer_main_joint'
        joint_goal = 0.4
        kitchen_setup.allow_all_collisions()
        kitchen_setup.set_json_goal(u'JointPosition',
                                    joint_name=joint_name1,
                                    goal=joint_goal,
                                    max_velocity=1)
        kitchen_setup.set_json_goal(u'JointPosition',
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

    def test_CartesianOrientation(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        tip = 'base_footprint'
        root = 'odom_combined'
        p = PoseStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = tip
        p.pose.orientation = Quaternion(*quaternion_about_axis(4, [0, 0, 1]))

        expected = tf.transform_pose('map', p)

        zero_pose.allow_all_collisions()
        zero_pose.set_json_goal(u'CartesianOrientation',
                                root_link=root,
                                tip_link=tip,
                                goal=p,
                                max_velocity=0.15
                                )
        zero_pose.plan_and_execute()
        new_pose = tf.lookup_pose('map', tip)
        compare_orientations(expected.pose.orientation, new_pose.pose.orientation)

    def test_CartesianPoseStraight(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.close_l_gripper()
        goal_position = PoseStamped()
        goal_position.header.frame_id = u'base_link'
        goal_position.pose.position.x = 0.3
        goal_position.pose.position.y = 0.5
        goal_position.pose.position.z = 1
        goal_position.pose.orientation.w = 1

        start_pose = tf.lookup_pose(u'map', zero_pose.l_tip)
        map_T_goal_position = tf.transform_pose(u'map', goal_position)

        object_pose = PoseStamped()
        object_pose.header.frame_id = u'map'
        object_pose.pose.position.x = (start_pose.pose.position.x + map_T_goal_position.pose.position.x) / 2.
        object_pose.pose.position.y = (start_pose.pose.position.y + map_T_goal_position.pose.position.y) / 2.
        object_pose.pose.position.z = (start_pose.pose.position.z + map_T_goal_position.pose.position.z) / 2.
        object_pose.pose.position.z += 0.08
        object_pose.pose.orientation.w = 1

        zero_pose.add_sphere(u'sphere', 0.05, pose=object_pose)

        publish_marker_vector(start_pose.pose.position, map_T_goal_position.pose.position)
        zero_pose.allow_self_collision()
        zero_pose.set_straight_cart_goal(goal_position, zero_pose.l_tip)
        zero_pose.plan_and_execute()

    def test_CartesianVelocityLimit(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        base_linear_velocity = 0.1
        base_angular_velocity = 0.2
        zero_pose.limit_cartesian_velocity(
            root_link=zero_pose.default_root,
            tip_link=u'base_footprint',
            max_linear_velocity=base_linear_velocity,
            max_angular_velocity=base_angular_velocity,
            hard=True,
        )
        eef_linear_velocity = 1
        eef_angular_velocity = 1
        goal_position = PoseStamped()
        goal_position.header.frame_id = u'r_gripper_tool_frame'
        goal_position.pose.position.x = 1
        goal_position.pose.position.y = 0
        goal_position.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 4, [0, 0, 1]))
        zero_pose.allow_all_collisions()
        zero_pose.set_cart_goal(goal_pose=goal_position,
                                tip_link=u'r_gripper_tool_frame',
                                linear_velocity=eef_linear_velocity,
                                angular_velocity=eef_angular_velocity,
                                weight=WEIGHT_BELOW_CA)
        zero_pose.plan_and_execute()

        for time, state in zero_pose.god_map.get_data(identifier.debug_trajectory).items():
            key = '{}/{}/{}/{}/trans_error'.format('CartesianVelocityLimit',
                                                   'TranslationVelocityLimit',
                                                   zero_pose.default_root,
                                                   u'base_footprint')
            assert key in state
            assert state[key].position <= base_linear_velocity + 2e3
            assert state[key].position >= -base_linear_velocity - 2e3

    def test_AvoidJointLimits1(self, zero_pose):
        """
        :type zero_pose: PR2
        """
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

    def test_AvoidJointLimits2(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        percentage = 10
        joints = [j for j in zero_pose.robot.controlled_joints if
                  not zero_pose.robot.is_joint_continuous(j)]
        goal_state = {j: zero_pose.robot.get_joint_position_limits(j)[1] for j in joints}
        del goal_state[u'odom_x_joint']
        del goal_state[u'odom_y_joint']
        zero_pose.set_json_goal(u'AvoidJointLimits',
                                percentage=percentage)
        zero_pose.set_joint_goal(goal_state, check=False)
        zero_pose.allow_self_collision()
        zero_pose.plan_and_execute()

        zero_pose.set_json_goal(u'AvoidJointLimits',
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
            assert position <= upper_limit2 and position >= lower_limit2

    def test_UpdateGodMap(self, pocky_pose_setup):
        """
        :type pocky_pose_setup: PR2
        """
        joint_velocity_weight = identifier.joint_weights + [u'velocity', u'override']
        old_torso_value = pocky_pose_setup.god_map.get_data(
            joint_velocity_weight + [u'torso_lift_joint'])
        old_odom_x_value = pocky_pose_setup.god_map.get_data(joint_velocity_weight + [u'odom_x_joint'])

        r_goal = PoseStamped()
        r_goal.header.frame_id = pocky_pose_setup.r_tip
        r_goal.pose.orientation.w = 1
        r_goal.pose.position.x += 0.1
        updates = {
            u'rosparam': {
                u'general_options': {
                    u'joint_weights': {
                        u'velocity': {
                            u'override': {
                                u'odom_x_joint': 1000000,
                                u'odom_y_joint': 1000000,
                                u'odom_z_joint': 1000000
                            }
                        }
                    }
                }
            }
        }

        old_pose = tf.lookup_pose(u'map', u'base_footprint')

        pocky_pose_setup.update_god_map(updates)
        pocky_pose_setup.set_cart_goal(r_goal, pocky_pose_setup.r_tip, check=False)
        pocky_pose_setup.plan_and_execute()

        new_pose = tf.lookup_pose(u'map', u'base_footprint')
        compare_poses(new_pose.pose, old_pose.pose)

        assert pocky_pose_setup.god_map.unsafe_get_data(
            joint_velocity_weight + [u'odom_x_joint']) == 1000000
        assert pocky_pose_setup.god_map.unsafe_get_data(
            joint_velocity_weight + [u'torso_lift_joint']) == old_torso_value

        updates = {
            u'rosparam': {
                u'general_options': {
                    u'joint_weights': {
                        u'velocity': {
                            u'override': {
                                u'odom_x_joint': 0.0001,
                                u'odom_y_joint': 0.0001,
                                u'odom_z_joint': 0.0001
                            }
                        }
                    }
                }
            }
        }
        # old_pose = tf.lookup_pose(u'map', u'base_footprint')
        # old_pose.pose.position.x += 0.1
        pocky_pose_setup.update_god_map(updates)
        pocky_pose_setup.set_cart_goal(r_goal, pocky_pose_setup.r_tip)
        pocky_pose_setup.plan_and_execute()

        new_pose = tf.lookup_pose(u'map', u'base_footprint')

        # compare_poses(old_pose.pose, new_pose.pose)
        assert new_pose.pose.position.x >= 0.03
        assert pocky_pose_setup.god_map.unsafe_get_data(
            joint_velocity_weight + [u'odom_x_joint']) == 0.0001
        assert pocky_pose_setup.god_map.unsafe_get_data(
            joint_velocity_weight + [u'torso_lift_joint']) == old_torso_value
        pocky_pose_setup.plan_and_execute()
        assert pocky_pose_setup.god_map.unsafe_get_data(
            joint_velocity_weight + [u'odom_x_joint']) == old_odom_x_value
        assert pocky_pose_setup.god_map.unsafe_get_data(
            joint_velocity_weight + [u'torso_lift_joint']) == old_torso_value

    def test_UpdateGodMap2(self, pocky_pose_setup):
        """
        :type pocky_pose_setup: PR2
        """
        joint_velocity_weight = identifier.joint_weights + [u'velocity', u'override']
        old_torso_value = pocky_pose_setup.god_map.get_data(
            joint_velocity_weight + [u'torso_lift_joint'])
        old_odom_x_value = pocky_pose_setup.god_map.get_data(joint_velocity_weight + [u'odom_x_joint'])
        old_odom_y_value = pocky_pose_setup.god_map.get_data(joint_velocity_weight + [u'odom_y_joint'])

        r_goal = PoseStamped()
        r_goal.header.frame_id = pocky_pose_setup.r_tip
        r_goal.pose.orientation.w = 1
        r_goal.pose.position.x += 0.1
        updates = {
            u'rosparam': {
                u'general_options': {
                    u'joint_weights': {
                        u'velocity': {
                            u'override': {
                                u'odom_x_joint': u'asdf',
                                u'odom_y_joint': 0.0001,
                                u'odom_z_joint': 0.0001
                            }
                        }
                    }
                }
            }
        }
        pocky_pose_setup.update_god_map(updates)
        pocky_pose_setup.set_cart_goal(r_goal, pocky_pose_setup.r_tip)
        pocky_pose_setup.plan_and_execute(expected_error_codes=[MoveResult.ERROR])
        assert pocky_pose_setup.god_map.unsafe_get_data(
            joint_velocity_weight + [u'odom_x_joint']) == old_odom_x_value
        assert pocky_pose_setup.god_map.unsafe_get_data(
            joint_velocity_weight + [u'odom_y_joint']) == old_odom_y_value
        assert pocky_pose_setup.god_map.get_data(
            joint_velocity_weight + [u'torso_lift_joint']) == old_torso_value

    def test_UpdateGodMap3(self, pocky_pose_setup):
        """
        :type pocky_pose_setup: PR2
        """
        joint_velocity_weight = identifier.joint_weights + [u'velocity', u'override']
        old_torso_value = pocky_pose_setup.god_map.get_data(
            joint_velocity_weight + [u'torso_lift_joint'])
        old_odom_x_value = pocky_pose_setup.god_map.get_data(joint_velocity_weight + [u'odom_x_joint'])

        r_goal = PoseStamped()
        r_goal.header.frame_id = pocky_pose_setup.r_tip
        r_goal.pose.orientation.w = 1
        r_goal.pose.position.x += 0.1
        updates = {
            u'rosparam': {
                u'general_options': {
                    u'joint_weights': u'asdf'
                }
            }
        }
        pocky_pose_setup.update_god_map(updates)
        pocky_pose_setup.set_cart_goal(r_goal, pocky_pose_setup.r_tip)
        pocky_pose_setup.plan_and_execute(expected_error_codes=[MoveResult.ERROR])
        assert pocky_pose_setup.god_map.unsafe_get_data(
            joint_velocity_weight + [u'odom_x_joint']) == old_odom_x_value
        assert pocky_pose_setup.god_map.unsafe_get_data(
            joint_velocity_weight + [u'torso_lift_joint']) == old_torso_value

    def test_pointing(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """

        rospy.loginfo("Starting looking")
        tip = u'head_mount_kinect_rgb_link'
        goal_point = tf.lookup_point(u'map', kitchen_setup.r_tip)
        pointing_axis = Vector3Stamped()
        pointing_axis.header.frame_id = tip
        pointing_axis.vector.x = 1
        kitchen_setup.pointing(tip, goal_point, pointing_axis=pointing_axis, root_link=kitchen_setup.r_tip)
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

        kitchen_setup.set_cart_goal(r_goal, kitchen_setup.r_tip, u'base_footprint', weight=WEIGHT_BELOW_CA)
        kitchen_setup.plan_and_execute()

        rospy.loginfo("Starting testing")
        current_x = Vector3Stamped()
        current_x.header.frame_id = tip
        current_x.vector.x = 1

        expected_x = tf.lookup_point(tip, kitchen_setup.r_tip)
        np.testing.assert_almost_equal(expected_x.point.y, 0, 2)
        np.testing.assert_almost_equal(expected_x.point.z, 0, 2)

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

        kitchen_setup.set_json_goal(u'Open',
                                    tip_link=kitchen_setup.r_tip,
                                    environment_link=handle_name,
                                    goal_joint_state=0)
        kitchen_setup.allow_all_collisions()
        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_kitchen_js({u'iai_fridge_door_joint': 0})

        kitchen_setup.plan_and_execute()

        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.plan_and_execute()

    def test_open_drawer(self, kitchen_setup):
        """"
        :type kitchen_setup: PR2
        """
        handle_frame_id = u'iai_kitchen/sink_area_left_middle_drawer_handle'
        handle_name = u'sink_area_left_middle_drawer_handle'
        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = handle_frame_id
        bar_axis.vector.y = 1

        bar_center = PointStamped()
        bar_center.header.frame_id = handle_frame_id

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
        x_goal.header.frame_id = handle_frame_id
        x_goal.vector.x = -1

        kitchen_setup.set_align_planes_goal(kitchen_setup.l_tip,
                                            x_gripper,
                                            root_normal=x_goal)
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()

        kitchen_setup.set_json_goal(u'Open',
                                    tip_link=kitchen_setup.l_tip,
                                    environment_link=handle_name)
        kitchen_setup.allow_all_collisions()  # makes execution faster
        kitchen_setup.plan_and_execute()  # send goal to Giskard
        # Update kitchen object
        kitchen_setup.set_kitchen_js({u'sink_area_left_middle_drawer_main_joint': 0.48})

        # Close drawer partially
        kitchen_setup.set_json_goal(u'Open',
                                    tip_link=kitchen_setup.l_tip,
                                    environment_link=handle_name,
                                    goal_joint_state=0.2)
        kitchen_setup.allow_all_collisions()  # makes execution faster
        kitchen_setup.plan_and_execute()  # send goal to Giskard
        # Update kitchen object
        kitchen_setup.set_kitchen_js({u'sink_area_left_middle_drawer_main_joint': 0.2})

        kitchen_setup.set_json_goal(u'Close',
                                    tip_link=kitchen_setup.l_tip,
                                    environment_link=handle_name)
        kitchen_setup.allow_all_collisions()  # makes execution faster
        kitchen_setup.plan_and_execute()  # send goal to Giskard
        # Update kitchen object
        kitchen_setup.set_kitchen_js({u'sink_area_left_middle_drawer_main_joint': 0.0})

        # TODO: calculate real and desired value and compare
        pass

    def test_open_close_dishwasher(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        p = PoseStamped()
        p.header.frame_id = u'map'
        p.pose.orientation.w = 1
        p.pose.position.x = 0.5
        p.pose.position.y = 0.2
        kitchen_setup.teleport_base(p)

        hand = kitchen_setup.r_tip

        goal_angle = np.pi / 4
        handle_frame_id = u'iai_kitchen/sink_area_dish_washer_door_handle'
        handle_name = u'sink_area_dish_washer_door_handle'
        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = handle_frame_id
        bar_axis.vector.y = 1

        bar_center = PointStamped()
        bar_center.header.frame_id = handle_frame_id

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = hand
        tip_grasp_axis.vector.z = 1

        kitchen_setup.set_json_goal(u'GraspBar',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=hand,
                                    tip_grasp_axis=tip_grasp_axis,
                                    bar_center=bar_center,
                                    bar_axis=bar_axis,
                                    bar_length=.3)
        # kitchen_setup.allow_collision([], u'kitchen', [handle_name])
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

        kitchen_setup.set_json_goal(u'Open',
                                    tip_link=hand,
                                    environment_link=handle_name,
                                    goal_joint_state=goal_angle,
                                    )
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_kitchen_js({u'sink_area_dish_washer_door_joint': goal_angle})

        kitchen_setup.set_json_goal(u'Open',
                                    tip_link=hand,
                                    environment_link=handle_name,
                                    goal_joint_state=0)
        kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_kitchen_js({u'sink_area_dish_washer_door_joint': 0})

    def test_open_close_dishwasher_palm(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        # FIXME
        handle_frame_id = u'iai_kitchen/sink_area_dish_washer_door_handle'
        handle_name = u'sink_area_dish_washer_door_handle'
        hand = kitchen_setup.r_tip
        goal_angle = np.pi / 3.5

        p = PoseStamped()
        p.header.frame_id = u'map'
        p.pose.orientation.w = 1
        p.pose.position.x = 0.5
        p.pose.position.y = 0.2
        kitchen_setup.teleport_base(p)

        kitchen_setup.set_kitchen_js({u'sink_area_dish_washer_door_joint': 0.})

        hand_goal = PoseStamped()
        hand_goal.header.frame_id = handle_frame_id
        hand_goal.pose.position.x -= 0.03
        hand_goal.pose.position.z = 0.03
        hand_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 0, 1, 0],
                                                                         [0, -1, 0, 0],
                                                                         [1, 0, 0, 0],
                                                                         [0, 0, 0, 1]]))
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.set_json_goal(u'CartesianVelocityLimit',
                                    root_link=u'odom_combined',
                                    tip_link='base_footprint',
                                    max_linear_velocity=0.05,
                                    max_angular_velocity=0.08,
                                    )
        kitchen_setup.set_cart_goal(hand_goal, hand)
        kitchen_setup.send_goal(goal_type=MoveGoal.PLAN_AND_EXECUTE_AND_CUT_OFF_SHAKING)

        kitchen_setup.set_json_goal(u'Open',
                                    tip_link=hand,
                                    object_name=u'kitchen',
                                    handle_link=handle_name,
                                    goal_joint_state=goal_angle,
                                    )

        kitchen_setup.set_json_goal(u'CartesianVelocityLimit',
                                    root_link=u'odom_combined',
                                    tip_link='base_footprint',
                                    max_linear_velocity=0.05,
                                    max_angular_velocity=0.1,
                                    )

        kitchen_setup.set_json_goal(u'CartesianVelocityLimit',
                                    root_link=u'odom_combined',
                                    tip_link='r_forearm_link',
                                    max_linear_velocity=0.1,
                                    max_angular_velocity=0.5,
                                    )

        # kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_kitchen_js({u'sink_area_dish_washer_door_joint': goal_angle})

        kitchen_setup.set_json_goal(u'Open',
                                    tip_link=hand,
                                    object_name=u'kitchen',
                                    handle_link=handle_name,
                                    goal_joint_state=0,
                                    )

        kitchen_setup.set_json_goal(u'CartesianVelocityLimit',
                                    root_link=u'odom_combined',
                                    tip_link='base_footprint',
                                    max_linear_velocity=0.05,
                                    max_angular_velocity=0.1,
                                    )

        kitchen_setup.set_json_goal(u'CartesianVelocityLimit',
                                    root_link=u'odom_combined',
                                    tip_link='r_forearm_link',
                                    max_linear_velocity=0.05,
                                    max_angular_velocity=0.1,
                                    )

        # kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_kitchen_js({u'sink_area_dish_washer_door_joint': 0})

        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.plan_and_execute()

    def test_align_planes1(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        x_gripper = Vector3Stamped()
        x_gripper.header.frame_id = zero_pose.r_tip
        x_gripper.vector.x = 1
        y_gripper = Vector3Stamped()
        y_gripper.header.frame_id = zero_pose.r_tip
        y_gripper.vector.y = 1

        x_goal = Vector3Stamped()
        x_goal.header.frame_id = u'map'
        x_goal.vector.x = 1
        y_goal = Vector3Stamped()
        y_goal.header.frame_id = u'map'
        y_goal.vector.z = 1
        zero_pose.set_align_planes_goal(zero_pose.r_tip, x_gripper, root_normal=x_goal)
        zero_pose.set_align_planes_goal(zero_pose.r_tip, y_gripper, root_normal=y_goal)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

    def test_wrong_constraint_type(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        goal_state = JointState()
        goal_state.name = [u'r_elbow_flex_joint']
        goal_state.position = [-1.0]
        kwargs = {u'goal_state': goal_state}
        zero_pose.set_json_goal(u'jointpos', **kwargs)
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.UNKNOWN_CONSTRAINT])

    def test_python_code_in_constraint_type(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        goal_state = JointState()
        goal_state.name = [u'r_elbow_flex_joint']
        goal_state.position = [-1.0]
        kwargs = {u'goal_state': goal_state}
        zero_pose.set_json_goal(u'print("asd")', **kwargs)
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.UNKNOWN_CONSTRAINT])

    def test_wrong_params1(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        goal_state = JointState()
        goal_state.name = u'r_elbow_flex_joint'
        goal_state.position = [-1.0]
        kwargs = {u'goal_state': goal_state}
        zero_pose.set_json_goal(u'JointPositionList', **kwargs)
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.CONSTRAINT_INITIALIZATION_ERROR])

    def test_wrong_params2(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        goal_state = JointState()
        goal_state.name = [5432]
        goal_state.position = u'test'
        kwargs = {u'goal_state': goal_state}
        zero_pose.set_json_goal(u'JointPositionList', **kwargs)
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.CONSTRAINT_INITIALIZATION_ERROR])

    def test_align_planes2(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        x_gripper = Vector3Stamped()
        x_gripper.header.frame_id = zero_pose.r_tip
        x_gripper.vector.y = 1

        x_goal = Vector3Stamped()
        x_goal.header.frame_id = u'map'
        x_goal.vector.y = -1
        x_goal.vector = tf.normalize(x_goal.vector)
        zero_pose.set_align_planes_goal(zero_pose.r_tip, x_gripper, root_normal=x_goal)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

    def test_align_planes3(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        eef_vector = Vector3Stamped()
        eef_vector.header.frame_id = 'base_footprint'
        eef_vector.vector.y = 1

        goal_vector = Vector3Stamped()
        goal_vector.header.frame_id = u'map'
        goal_vector.vector.y = -1
        goal_vector.vector = tf.normalize(goal_vector.vector)
        zero_pose.set_align_planes_goal('base_footprint', eef_vector, root_normal=goal_vector)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

    def test_align_planes4(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        elbow = u'r_elbow_flex_link'
        handle_frame_id = u'iai_kitchen/iai_fridge_door_handle'

        tip_axis = Vector3Stamped()
        tip_axis.header.frame_id = elbow
        tip_axis.vector.x = 1

        env_axis = Vector3Stamped()
        env_axis.header.frame_id = handle_frame_id
        env_axis.vector.z = 1
        kitchen_setup.set_align_planes_goal(elbow, tip_axis, root_normal=env_axis, weight=WEIGHT_ABOVE_CA)
        kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()

    def test_grasp_fridge_handle(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        handle_name = u'iai_kitchen/iai_fridge_door_handle'
        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = handle_name
        bar_axis.vector.z = 1

        bar_center = PointStamped()
        bar_center.header.frame_id = handle_name

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = kitchen_setup.r_tip
        tip_grasp_axis.vector.z = 1

        kitchen_setup.set_json_goal(u'GraspBar',
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
        x_goal.header.frame_id = u'iai_kitchen/iai_fridge_door_handle'
        x_goal.vector.x = -1
        kitchen_setup.set_align_planes_goal(kitchen_setup.r_tip, x_gripper, root_normal=x_goal)
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()

    def test_close_fridge_with_elbow(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        base_pose = PoseStamped()
        base_pose.header.frame_id = u'map'
        base_pose.pose.position.y = -1.5
        base_pose.pose.orientation.w = 1
        kitchen_setup.teleport_base(base_pose)

        handle_frame_id = u'iai_kitchen/iai_fridge_door_handle'
        handle_name = u'iai_fridge_door_handle'

        kitchen_setup.set_kitchen_js({u'iai_fridge_door_joint': np.pi / 2})

        elbow = u'r_elbow_flex_link'

        tip_axis = Vector3Stamped()
        tip_axis.header.frame_id = elbow
        tip_axis.vector.x = 1

        env_axis = Vector3Stamped()
        env_axis.header.frame_id = handle_frame_id
        env_axis.vector.z = 1
        kitchen_setup.set_align_planes_goal(elbow, tip_axis, root_normal=env_axis, weight=WEIGHT_ABOVE_CA)
        kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()
        elbow_pose = PoseStamped()
        elbow_pose.header.frame_id = handle_frame_id
        elbow_pose.pose.position.x += 0.1
        elbow_pose.pose.orientation.w = 1
        kitchen_setup.set_translation_goal(elbow_pose, elbow)
        kitchen_setup.set_align_planes_goal(elbow, tip_axis, root_normal=env_axis, weight=WEIGHT_ABOVE_CA)
        kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()

        kitchen_setup.set_json_goal(u'Close',
                                    tip_link=elbow,
                                    environment_link=handle_name)
        kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_kitchen_js({u'iai_fridge_door_joint': 0})

    def test_open_close_oven(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        goal_angle = 0.5
        handle_frame_id = u'iai_kitchen/oven_area_oven_door_handle'
        handle_name = u'oven_area_oven_door_handle'
        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = handle_frame_id
        bar_axis.vector.y = 1

        bar_center = PointStamped()
        bar_center.header.frame_id = handle_frame_id

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = kitchen_setup.l_tip
        tip_grasp_axis.vector.z = 1

        kitchen_setup.set_json_goal(u'GraspBar',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=kitchen_setup.l_tip,
                                    tip_grasp_axis=tip_grasp_axis,
                                    bar_center=bar_center,
                                    bar_axis=bar_axis,
                                    bar_length=.3)
        # kitchen_setup.allow_collision([], u'kitchen', [handle_name])
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

        kitchen_setup.set_json_goal(u'Open',
                                    tip_link=kitchen_setup.l_tip,
                                    environment_link=handle_name,
                                    goal_joint_state=goal_angle)
        kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_kitchen_js({u'oven_area_oven_door_joint': goal_angle})

        kitchen_setup.set_json_goal(u'Close',
                                    tip_link=kitchen_setup.l_tip,
                                    environment_link=handle_name)
        kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_kitchen_js({u'oven_area_oven_door_joint': 0})

    def test_grasp_dishwasher_handle(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        handle_name = u'iai_kitchen/sink_area_dish_washer_door_handle'
        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = handle_name
        bar_axis.vector.y = 1

        bar_center = PointStamped()
        bar_center.header.frame_id = handle_name

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = kitchen_setup.r_tip
        tip_grasp_axis.vector.z = 1

        kitchen_setup.grasp_bar(root_link=kitchen_setup.default_root,
                                tip_link=kitchen_setup.r_tip,
                                tip_grasp_axis=tip_grasp_axis,
                                bar_center=bar_center,
                                bar_axis=bar_axis,
                                bar_length=.3)
        kitchen_setup.allow_collision([], u'kitchen', [u'sink_area_dish_washer_door_handle'])
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()

    def test_open_all_drawers(self, kitchen_setup):
        """"
        :type kitchen_setup: PR2
        """
        handle_name = [
            # u'oven_area_area_middle_upper_drawer_handle',
            # u'oven_area_area_middle_lower_drawer_handle',
            #u'sink_area_left_upper_drawer_handle',
            # u'sink_area_left_middle_drawer_handle',
            # u'sink_area_left_bottom_drawer_handle',
            #u'sink_area_trash_drawer_handle',
            #u'fridge_area_lower_drawer_handle',
            # u'kitchen_island_left_upper_drawer_handle',
            # u'kitchen_island_left_lower_drawer_handle',
            # u'kitchen_island_middle_upper_drawer_handle',
            # u'kitchen_island_middle_lower_drawer_handle',
            # u'kitchen_island_right_upper_drawer_handle',
            #u'kitchen_island_right_lower_drawer_handle',
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
    def test_move_base(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        map_T_odom = PoseStamped()
        map_T_odom.pose.position.x = 1
        map_T_odom.pose.position.y = 1
        map_T_odom.pose.orientation = Quaternion(*quaternion_about_axis(np.pi/3, [0,0,1]))
        zero_pose.set_localization(map_T_odom)

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 1
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        zero_pose.set_cart_goal(base_goal, 'base_footprint')
        zero_pose.plan_and_execute()

    def test_rotate_gripper(self, zero_pose):
        """
        :type zero_pose: PR2
        """
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
        #zero_pose.set_align_planes_goal(zero_pose.r_tip, r_gripper_vec, root_normal=gripper_goal_vec)
        #zero_pose.set_align_planes_goal(zero_pose.l_tip, l_gripper_vec, root_normal=gripper_goal_vec)
        zero_pose.plan_and_execute()


    def test_keep_position1(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.allow_self_collision()
        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.r_tip
        r_goal.pose.position.x = -.1
        r_goal.pose.orientation.w = 1
        zero_pose.set_cart_goal(r_goal, zero_pose.r_tip, u'torso_lift_link')
        zero_pose.plan_and_execute()

        js = {u'torso_lift_joint': 0.1}
        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.r_tip
        r_goal.pose.orientation.w = 1
        zero_pose.set_cart_goal(r_goal, zero_pose.r_tip, u'torso_lift_link')
        zero_pose.set_joint_goal(js)
        zero_pose.allow_self_collision()
        zero_pose.plan_and_execute()

    def test_keep_position2(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.allow_self_collision()
        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.r_tip
        r_goal.pose.position.x = -.1
        r_goal.pose.orientation.w = 1
        zero_pose.set_cart_goal(r_goal, zero_pose.r_tip, u'torso_lift_link')
        zero_pose.plan_and_execute()

        zero_pose.allow_self_collision()
        js = {u'torso_lift_joint': 0.1}
        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.r_tip
        r_goal.pose.orientation.w = 1
        expected_pose = tf.lookup_pose(zero_pose.default_root, zero_pose.r_tip)
        expected_pose.header.stamp = rospy.Time()
        zero_pose.set_cart_goal(r_goal, zero_pose.r_tip, zero_pose.default_root)
        zero_pose.set_joint_goal(js)
        zero_pose.plan_and_execute()

    def test_keep_position3(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        js = {
            u'r_elbow_flex_joint': -1.58118094489,
            u'r_forearm_roll_joint': -0.904933033043,
            u'r_shoulder_lift_joint': 0.822412440711,
            u'r_shoulder_pan_joint': -1.07866800992,
            u'r_upper_arm_roll_joint': -1.34905471854,
            u'r_wrist_flex_joint': -1.20182042644,
            u'r_wrist_roll_joint': 0.190433188769,
        }
        zero_pose.send_and_check_joint_goal(js)

        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.r_tip
        r_goal.pose.position.x = 0.3
        r_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[-1, 0, 0, 0],
                                                                      [0, 1, 0, 0],
                                                                      [0, 0, -1, 0],
                                                                      [0, 0, 0, 1]]))
        zero_pose.set_cart_goal(r_goal, zero_pose.l_tip, u'torso_lift_link')
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
        zero_pose.plan_and_execute()

    def test_cart_goal_1eef(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        p = PoseStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = zero_pose.r_tip
        p.pose.position = Point(-0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.allow_self_collision()
        zero_pose.set_cart_goal(p, zero_pose.r_tip, u'base_footprint')
        zero_pose.plan_and_execute()

    def test_cart_goal_1eef2(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        p = PoseStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = u'base_footprint'
        p.pose.position = Point(0.599, -0.009, 0.983)
        p.pose.orientation = Quaternion(0.524, -0.495, 0.487, -0.494)
        zero_pose.allow_all_collisions()
        zero_pose.set_cart_goal(p, zero_pose.l_tip, u'base_footprint')
        zero_pose.plan_and_execute()

    def test_cart_goal_1eef3(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        self.test_cart_goal_1eef(zero_pose)
        self.test_cart_goal_1eef2(zero_pose)

    def test_cart_goal_1eef4(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        p = PoseStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = u'map'
        p.pose.position = Point(2., 0, 1.)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.allow_all_collisions()
        zero_pose.set_cart_goal(p, zero_pose.r_tip, zero_pose.default_root)
        zero_pose.plan_and_execute()

    def test_cart_goal_orientation_singularity(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        root = u'base_link'
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
        #l_goal = PoseStamped()
        #l_goal.header.frame_id = zero_pose.l_tip
        #l_goal.header.stamp = rospy.get_rostime()
        #l_goal.pose.position = Point(0, 0, -0.1)
        #l_goal.pose.orientation = Quaternion(0, 0, 0, 1)
        #zero_pose.set_json_goal(u'CartesianPoseChanging',
        #                        root_link=root,
        #                        tip_link=zero_pose.l_tip,
        #                        goal=l_goal
        #                        )
        zero_pose.allow_self_collision()
        zero_pose.send_goal()
        #zero_pose.check_cart_goal(zero_pose.r_tip, r_goal)
        #zero_pose.check_cart_goal(zero_pose.l_tip, l_goal)

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
        #kitchen_setup.send_goal()
        #kitchen_setup.check_cart_goal(tip_link, goal_a)
        #zero_pose.check_cart_goal(zero_pose.l_tip, l_goal)

    def test_cart_goal_2eef2(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        root = u'odom_combined'

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

    def test_cart_goal_left_right_chain(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        r_goal = tf.lookup_pose(zero_pose.l_tip, zero_pose.r_tip)
        r_goal.pose.position.x -= 0.1
        zero_pose.allow_all_collisions()
        zero_pose.set_cart_goal(r_goal, zero_pose.r_tip, zero_pose.l_tip)
        zero_pose.plan_and_execute()

    def test_wiggle1(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        tray_pose = PoseStamped()
        tray_pose.header.frame_id = u'iai_kitchen/sink_area_surface'
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
        kitchen_setup.set_json_goal(u'CartesianVelocityLimit',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=u'base_footprint',
                                    max_linear_velocity=0.1,
                                    max_angular_velocity=0.2
                                    )
        kitchen_setup.plan_and_execute()

    def test_wiggle2(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        goal_js = {
            u'l_upper_arm_roll_joint': 1.63487737202,
            u'l_shoulder_pan_joint': 1.36222920328,
            u'l_shoulder_lift_joint': 0.229120778526,
            u'l_forearm_roll_joint': 13.7578920265,
            u'l_elbow_flex_joint': -1.48141189643,
            u'l_wrist_flex_joint': -1.22662876066,
            u'l_wrist_roll_joint': -53.6150824007,
        }
        zero_pose.allow_all_collisions()
        zero_pose.send_and_check_joint_goal(goal_js)

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

    def test_wiggle3(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        goal_js = {
            u'r_upper_arm_roll_joint': -0.0812729778068,
            u'r_shoulder_pan_joint': -1.20939684714,
            u'r_shoulder_lift_joint': 0.135095147908,
            u'r_forearm_roll_joint': -1.50201448056,
            u'r_elbow_flex_joint': -0.404527363115,
            u'r_wrist_flex_joint': -1.11738043795,
            u'r_wrist_roll_joint': 8.0946050982,
        }
        zero_pose.allow_all_collisions()
        zero_pose.send_and_check_joint_goal(goal_js)

        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.5
        p.pose.orientation.w = 1
        zero_pose.set_cart_goal(p, zero_pose.r_tip, zero_pose.default_root)
        zero_pose.plan_and_execute()

    def test_hot_init_failed(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.r_tip
        r_goal.header.stamp = rospy.get_rostime()
        r_goal.pose.position = Point(-0.0, 0, 0)
        r_goal.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_cart_goal(r_goal, zero_pose.r_tip, zero_pose.default_root)
        l_goal = PoseStamped()
        l_goal.header.frame_id = zero_pose.l_tip
        l_goal.header.stamp = rospy.get_rostime()
        l_goal.pose.position = Point(-0.0, 0, 0)
        l_goal.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_cart_goal(l_goal, zero_pose.l_tip, zero_pose.default_root)
        zero_pose.allow_self_collision()
        zero_pose.plan_and_execute()

        zero_pose.allow_all_collisions()
        zero_pose.send_and_check_joint_goal(default_pose)

        goal_js = {
            u'r_upper_arm_roll_joint': -0.0812729778068,
            u'r_shoulder_pan_joint': -1.20939684714,
            u'r_shoulder_lift_joint': 0.135095147908,
            u'r_forearm_roll_joint': -1.50201448056,
            u'r_elbow_flex_joint': -0.404527363115,
            u'r_wrist_flex_joint': -1.11738043795,
            u'r_wrist_roll_joint': 8.0946050982,
        }
        zero_pose.allow_all_collisions()
        zero_pose.send_and_check_joint_goal(goal_js)

    def test_root_link_not_equal_chain_root(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        p = PoseStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = u'base_footprint'
        p.pose.position.x = 0.8
        p.pose.position.y = -0.5
        p.pose.position.z = 1
        p.pose.orientation.w = 1
        zero_pose.allow_self_collision()
        zero_pose.set_cart_goal(p, zero_pose.r_tip, u'torso_lift_link')
        zero_pose.plan_and_execute()


class TestActionServerEvents(object):
    def test_interrupt1(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        p = PoseStamped()
        p.header.frame_id = u'base_footprint'
        p.pose.position = Point(2, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_cart_goal(p, u'base_footprint')
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.PREEMPTED], stop_after=1)

    def test_interrupt2(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        p = PoseStamped()
        p.header.frame_id = u'base_footprint'
        p.pose.position = Point(2, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_cart_goal(p, u'base_footprint')
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.PREEMPTED], stop_after=6)

    def test_undefined_type(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.allow_all_collisions()
        zero_pose.send_goal(goal_type=MoveGoal.UNDEFINED,
                            expected_error_codes=[MoveResult.INVALID_GOAL])

    def test_empty_goal(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.cmd_seq = []
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.INVALID_GOAL])

    def test_plan_only(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.allow_self_collision()
        zero_pose.set_joint_goal(pocky_pose, check=False)
        zero_pose.add_goal_check(JointGoalChecker(zero_pose.god_map, default_pose))
        zero_pose.send_goal(goal_type=MoveGoal.PLAN_ONLY)


class TestWayPoints(object):
    def test_interrupt_way_points1(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        p = PoseStamped()
        p.header.frame_id = u'base_footprint'
        p.pose.position = Point(0, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_cart_goal(deepcopy(p), u'base_footprint')
        zero_pose.add_cmd()
        p.pose.position.x += 1
        zero_pose.set_cart_goal(deepcopy(p), u'base_footprint')
        zero_pose.add_cmd()
        p.pose.position.x += 1
        zero_pose.set_cart_goal(p, u'base_footprint')
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

    def test_waypoints2(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.set_joint_goal(pocky_pose)
        zero_pose.allow_all_collisions()
        zero_pose.add_cmd()
        zero_pose.set_joint_goal(pick_up_pose)
        zero_pose.allow_all_collisions()
        zero_pose.add_cmd()
        zero_pose.set_joint_goal(gaya_pose)
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
            assert False, u'pocky pose not in trajectory'

        traj.points = traj.points[i:]
        for i, p in enumerate(traj.points):
            js = {joint_name: position for joint_name, position in zip(traj.joint_names, p.positions)}
            try:
                zero_pose.compare_joint_state(js, pick_up_pose)
                break
            except AssertionError:
                pass
        else:  # if no break
            assert False, u'pick_up_pose not in trajectory'

        traj.points = traj.points[i:]
        for i, p in enumerate(traj.points):
            js = {joint_name: position for joint_name, position in zip(traj.joint_names, p.positions)}
            try:
                zero_pose.compare_joint_state(js, gaya_pose)
                break
            except AssertionError:
                pass
        else:  # if no break
            assert False, u'gaya_pose not in trajectory'

        pass

    def test_waypoints_with_fail(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.set_joint_goal(pocky_pose)
        zero_pose.add_cmd()
        zero_pose.set_json_goal(u'muh')
        zero_pose.add_cmd()
        zero_pose.set_joint_goal(gaya_pose)

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
            assert False, u'pocky pose not in trajectory'

        traj.points = traj.points[i:]
        for i, p in enumerate(traj.points):
            js = {joint_name: position for joint_name, position in zip(traj.joint_names, p.positions)}
            try:
                zero_pose.compare_joint_state(js, gaya_pose)
                break
            except AssertionError:
                pass
        else:  # if no break
            assert False, u'gaya_pose not in trajectory'

    def test_waypoints_with_fail1(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.set_json_goal(u'muh')
        zero_pose.add_cmd()
        zero_pose.set_joint_goal(pocky_pose)
        zero_pose.add_cmd()
        zero_pose.set_joint_goal(gaya_pose)

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
            assert False, u'pocky pose not in trajectory'

        traj.points = traj.points[i:]
        for i, p in enumerate(traj.points):
            js = {joint_name: position for joint_name, position in zip(traj.joint_names, p.positions)}
            try:
                zero_pose.compare_joint_state(js, gaya_pose)
                break
            except AssertionError:
                pass
        else:  # if no break
            assert False, u'gaya_pose not in trajectory'

    def test_waypoints_with_fail2(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.set_joint_goal(pocky_pose)
        zero_pose.add_cmd()
        zero_pose.set_joint_goal(gaya_pose)
        zero_pose.add_cmd()
        zero_pose.set_json_goal(u'muh')

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
            assert False, u'pocky pose not in trajectory'

        traj.points = traj.points[i:]
        for i, p in enumerate(traj.points):
            js = {joint_name: position for joint_name, position in zip(traj.joint_names, p.positions)}
            try:
                zero_pose.compare_joint_state(js, gaya_pose)
                break
            except AssertionError:
                pass
        else:  # if no break
            assert False, u'gaya_pose not in trajectory'

    def test_waypoints_with_fail3(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.set_joint_goal(pocky_pose)
        zero_pose.add_cmd()
        zero_pose.set_json_goal(u'muh')
        zero_pose.add_cmd()
        zero_pose.set_joint_goal(gaya_pose)

        traj = zero_pose.send_goal(expected_error_codes=[MoveResult.SUCCESS,
                                                         MoveResult.UNKNOWN_CONSTRAINT,
                                                         MoveResult.ERROR],
                                   goal_type=MoveGoal.PLAN_AND_EXECUTE)

        for i, p in enumerate(traj.points):
            js = {joint_name: position for joint_name, position in zip(traj.joint_names, p.positions)}
            try:
                zero_pose.compare_joint_state(js, default_pose)
                break
            except AssertionError:
                pass
        else:  # if no break
            assert False, u'pocky pose not in trajectory'

    def test_skip_failures1(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.set_json_goal(u'muh')
        zero_pose.send_goal(expected_error_codes=[MoveResult.UNKNOWN_CONSTRAINT, ],
                            goal_type=MoveGoal.PLAN_AND_EXECUTE_AND_SKIP_FAILURES)

    def test_skip_failures2(self, zero_pose):
        """
        :type zero_pose: PR2
        """
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
            assert False, u'pocky pose not in trajectory'

    # TODO test translation and orientation goal in different frame


class TestCartesianPath(object):

    def test_pathAroundKitchenIsland_without_global_planner(self, kitchen_setup_avoid_collisions):
        # kernprof -lv py.test -s test/test_integration_pr2.py::TestCartGoals::test_pathAroundKitchenIsland_with_global_planner
        """
        :type kitchen_setup_avoid_collisions: PR2
        """
        tip_link = u'base_footprint'

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = 0
        base_pose.pose.position.y = 2.0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        kitchen_setup_avoid_collisions.teleport_base(base_pose)

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = -1.2
        base_pose.pose.position.y = -2.0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        goal_c = base_pose

        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPose',
                                                     tip_link=tip_link,
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     goal=goal_c
                                                     )

        kitchen_setup_avoid_collisions.plan_and_execute()
                #kitchen_setup_avoid_collisions.send_goal()
        #kitchen_setup_avoid_collisions.check_cart_goal(tip_link, goal_c)
        #zero_pose.check_cart_goal(zero_pose.l_tip, l_goal)

    def test_pathAroundKitchenIsland_with_global_planner(self, kitchen_setup_avoid_collisions):
        # kernprof -lv py.test -s test/test_integration_pr2.py::TestCartesianPath::test_pathAroundKitchenIsland_with_global_planner
        """
        :type kitchen_setup_avoid_collisions: PR2
        """
        tip_link = u'base_footprint'

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = 0
        base_pose.pose.position.y = 2.0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        kitchen_setup_avoid_collisions.teleport_base(base_pose)

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = -2
        base_pose.pose.position.y = 0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        goal_c = base_pose

        kitchen_setup_avoid_collisions.set_json_goal(u'SetPredictionHorizon', prediction_horizon=1)
        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPose',
                                                     tip_link=tip_link,
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     goal=goal_c
                                                     )

        kitchen_setup_avoid_collisions.plan_and_execute()
        #kitchen_setup_avoid_collisions.send_goal()
        #kitchen_setup_avoid_collisions.check_cart_goal(tip_link, goal_c)
        #zero_pose.check_cart_goal(zero_pose.l_tip, l_goal)

    def test_pathAroundKitchenIslandChangedOrientation_with_global_planner1(self, kitchen_setup_avoid_collisions):
        # kernprof -lv py.test -s test/test_integration_pr2.py::TestCartGoals::test_pathAroundKitchenIsland_with_global_planner
        """
        :type kitchen_setup_avoid_collisions: PR2
        """
        tip_link = u'base_footprint'

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = 0
        base_pose.pose.position.y = 2.0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        kitchen_setup_avoid_collisions.teleport_base(base_pose)

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = 2.1
        base_pose.pose.position.y = 0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        goal_c = base_pose

        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPose',
                                                     tip_link=tip_link,
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     goal=goal_c
                                                     )

        kitchen_setup_avoid_collisions.plan_and_execute()
        #kitchen_setup_avoid_collisions.send_goal()
        #kitchen_setup_avoid_collisions.check_cart_goal(tip_link, goal_c)
        #zero_pose.check_cart_goal(zero_pose.l_tip, l_goal)

    def test_pathAroundKitchenIslandChangedOrientation_with_global_planner2(self, kitchen_setup_avoid_collisions):
        # kernprof -lv py.test -s test/test_integration_pr2.py::TestCartGoals::test_pathAroundKitchenIsland_with_global_planner
        """
        :type kitchen_setup_avoid_collisions: PR2
        """
        table_navigation_link = 'iai_kitchen/dining_area_footprint'
        # spawn milk
        table_navigation_goal = PoseStamped()
        table_navigation_goal.header.frame_id = table_navigation_link
        table_navigation_goal.pose.position = Point(-0.24, -0.80, 0.0)
        table_navigation_goal.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, np.pi/2.))

        tip_link = u'base_footprint'

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = 0
        base_pose.pose.position.y = 0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        kitchen_setup_avoid_collisions.teleport_base(base_pose)

        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPose',
                                                     tip_link=tip_link,
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     goal=table_navigation_goal
                                                     )

        kitchen_setup_avoid_collisions.plan_and_execute()
        #kitchen_setup_avoid_collisions.send_goal()
        #kitchen_setup_avoid_collisions.check_cart_goal(tip_link, goal_c)
        #zero_pose.check_cart_goal(zero_pose.l_tip, l_goal)

    def test_pathAroundKitchenIslandChangedOrientation_with_global_planner_align_planes1(self, kitchen_setup_avoid_collisions):
        """
        :type zero_pose: PR2
        """
        tip_link = u'base_footprint'

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = 0
        base_pose.pose.position.y = 2.0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        kitchen_setup_avoid_collisions.teleport_base(base_pose)

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = 2.1
        base_pose.pose.position.y = 0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        goal_c = base_pose

        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPose',
                                                     tip_link=tip_link,
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     goal=goal_c
                                                     )

        r_gripper_vec = Vector3Stamped()
        r_gripper_vec.header.frame_id = kitchen_setup_avoid_collisions.r_tip
        r_gripper_vec.vector.z = 1
        l_gripper_vec = Vector3Stamped()
        l_gripper_vec.header.frame_id = kitchen_setup_avoid_collisions.l_tip
        l_gripper_vec.vector.z = 1

        gripper_goal_vec = Vector3Stamped()
        gripper_goal_vec.header.frame_id = u'map'
        gripper_goal_vec.vector.z = 1
        kitchen_setup_avoid_collisions.set_align_planes_goal(kitchen_setup_avoid_collisions.r_tip, r_gripper_vec, root_normal=gripper_goal_vec)
        kitchen_setup_avoid_collisions.set_align_planes_goal(kitchen_setup_avoid_collisions.l_tip, l_gripper_vec, root_normal=gripper_goal_vec)
        kitchen_setup_avoid_collisions.plan_and_execute()

    def test_pathAroundKitchenIslandChangedOrientation_with_global_planner_align_planes2(self, kitchen_setup_avoid_collisions):
        """
        :type zero_pose: PR2
        """
        tip_link = u'base_footprint'

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = 0
        base_pose.pose.position.y = 2.0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        kitchen_setup_avoid_collisions.teleport_base(base_pose)

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = 2.1
        base_pose.pose.position.y = 0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        goal_c = base_pose

        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPose',
                                                     tip_link=tip_link,
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     goal=goal_c
                                                     )

        r_gripper_vec = Vector3Stamped()
        r_gripper_vec.header.frame_id = kitchen_setup_avoid_collisions.r_tip
        r_gripper_vec.vector.z = -1
        l_gripper_vec = Vector3Stamped()
        l_gripper_vec.header.frame_id = kitchen_setup_avoid_collisions.l_tip
        l_gripper_vec.vector.z = 1

        gripper_goal_vec = Vector3Stamped()
        gripper_goal_vec.header.frame_id = u'map'
        gripper_goal_vec.vector.z = 1
        kitchen_setup_avoid_collisions.set_align_planes_goal(kitchen_setup_avoid_collisions.r_tip, r_gripper_vec, root_normal=gripper_goal_vec)
        kitchen_setup_avoid_collisions.set_align_planes_goal(kitchen_setup_avoid_collisions.l_tip, l_gripper_vec, root_normal=gripper_goal_vec)
        kitchen_setup_avoid_collisions.plan_and_execute()

    def test_pathAroundKitchenIslandChangedOrientation_with_global_planner_shaky_grippers(self,
                                                                                         kitchen_setup_avoid_collisions):
        """
        :type zero_pose: PR2
        """
        tip_link = u'base_footprint'

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = 0
        base_pose.pose.position.y = 2.0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        kitchen_setup_avoid_collisions.teleport_base(base_pose)

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = 2.1
        base_pose.pose.position.y = 0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        goal_c = base_pose

        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPose',
                                                     tip_link=tip_link,
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     goal=goal_c)

        f = 0.5
        amp = 1.0
        axis = 'z'

        kitchen_setup_avoid_collisions.set_json_goal(u'ShakyCartesianPosition',
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     tip_link=kitchen_setup_avoid_collisions.r_tip,
                                                     frequency=f,
                                                     noise_amplitude=amp,
                                                     shaking_axis=axis)
        kitchen_setup_avoid_collisions.set_json_goal(u'ShakyCartesianPosition',
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     tip_link=kitchen_setup_avoid_collisions.l_tip,
                                                     frequency=f,
                                                     noise_amplitude=amp,
                                                     shaking_axis=axis)

        kitchen_setup_avoid_collisions.plan_and_execute()

    def test_pathAroundKitchenIsland_with_global_planner_and_box(self, kitchen_setup_avoid_collisions):
        # kernprof -lv py.test -s test/test_integration_pr2.py::TestCartGoals::test_pathAroundKitchenIsland_with_global_planner
        """
        :type kitchen_setup_avoid_collisions: PR2
        """
        tip_link = u'base_footprint'

        box_pose = PoseStamped()
        box_pose.header.frame_id = tip_link
        box_pose.pose.position.x = -0.5
        box_pose.pose.position.y = 0
        box_pose.pose.position.z = 0
        box_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        kitchen_setup_avoid_collisions.add_box('box', [0.5, 0.5, 1], pose=box_pose)

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = 0
        base_pose.pose.position.y = 2.0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        kitchen_setup_avoid_collisions.teleport_base(base_pose)

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = -2
        base_pose.pose.position.y = 0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        goal_c = base_pose

        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPose',
                                                     tip_link=tip_link,
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     goal=goal_c
                                                     )
        kitchen_setup_avoid_collisions.plan_and_execute()
        #kitchen_setup_avoid_collisions.send_goal()
        #kitchen_setup_avoid_collisions.check_cart_goal(tip_link, goal_c)
        #zero_pose.check_cart_goal(zero_pose.l_tip, l_goal)


    def test_ease_fridge_with_cart_goals_lifting_and_global_planner(self, kitchen_setup_avoid_collisions):
        rospy.sleep(10.0)#0.5

        tip_link = kitchen_setup_avoid_collisions.r_tip
        milk_name = u'milk'

        # take milk out of fridge
        kitchen_setup_avoid_collisions.set_kitchen_js({u'iai_fridge_door_joint': 1.56})

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 0.4
        base_goal.pose.position.y = -1.05
        base_goal.pose.orientation.x = 0
        base_goal.pose.orientation.y = 0
        base_goal.pose.orientation.z = 0
        base_goal.pose.orientation.w = 1
        kitchen_setup_avoid_collisions.teleport_base(base_goal)

        # spawn milk
        milk_pose = PoseStamped()
        milk_pose.header.frame_id = u'iai_kitchen/iai_fridge_door_shelf1_bottom'
        milk_pose.pose.position = Point(0, 0, 0.12)
        milk_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        milk_lift_pre_pose = PoseStamped()
        milk_lift_pre_pose.header.frame_id = u'iai_kitchen/iai_fridge_door_shelf1_bottom'
        milk_lift_pre_pose.pose.position = Point(0, 0, 0.22)
        milk_lift_pre_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        #milk_grasp_pre_pose = PoseStamped()
        #milk_grasp_pre_pose.header.frame_id = u'iai_kitchen/iai_fridge_door_shelf1_bottom'
        #milk_grasp_pre_pose.pose.position = Point(-0.2, 0, 0.12)
        #milk_grasp_pre_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup_avoid_collisions.add_box(milk_name, [0.05, 0.05, 0.2], milk_pose)

        # move arm towards milk
        if tip_link == kitchen_setup_avoid_collisions.l_tip:
            kitchen_setup_avoid_collisions.open_l_gripper()
        elif tip_link == kitchen_setup_avoid_collisions.r_tip:
            kitchen_setup_avoid_collisions.open_r_gripper()
        else:
            raise Exception('Wrong tip link')

        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPose',
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     tip_link=tip_link,
                                                     goal=milk_pose)

        kitchen_setup_avoid_collisions.plan_and_execute()

        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPose',
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     tip_link=tip_link,
                                                     goal=milk_pose)

        kitchen_setup_avoid_collisions.plan_and_execute()

        # grasp milk
        kitchen_setup_avoid_collisions.attach_object(milk_name, tip_link)

        if tip_link == kitchen_setup_avoid_collisions.l_tip:
            kitchen_setup_avoid_collisions.close_l_gripper()
        elif tip_link == kitchen_setup_avoid_collisions.r_tip:
            kitchen_setup_avoid_collisions.close_r_gripper()
        else:
            raise Exception('Wrong tip link')

        kitchen_setup_avoid_collisions.set_cart_goal(milk_lift_pre_pose, milk_name, kitchen_setup_avoid_collisions.default_root)
        kitchen_setup_avoid_collisions.plan_and_execute()
        kitchen_setup_avoid_collisions.set_joint_goal(gaya_pose)
        kitchen_setup_avoid_collisions.plan_and_execute()

        # dont allow robot to move
        js = kitchen_setup_avoid_collisions.god_map.get_data(identifier.joint_states)
        odom_joints = ['odom_x_joint', 'odom_y_joint', 'odom_z_joint']
        kitchen_setup_avoid_collisions.set_joint_goal({j_n: js[j_n].position for j_n in odom_joints})

        # place milk back
        kitchen_setup_avoid_collisions.set_cart_goal(milk_lift_pre_pose, milk_name, kitchen_setup_avoid_collisions.default_root)
        kitchen_setup_avoid_collisions.plan_and_execute()

        kitchen_setup_avoid_collisions.set_cart_goal(milk_pose, milk_name, kitchen_setup_avoid_collisions.default_root)
        kitchen_setup_avoid_collisions.plan_and_execute()

        # kitchen_setup.keep_position(kitchen_setup_avoid_collisions.r_tip)
        if tip_link == kitchen_setup_avoid_collisions.l_tip:
            kitchen_setup_avoid_collisions.open_l_gripper()
        elif tip_link == kitchen_setup_avoid_collisions.r_tip:
            kitchen_setup_avoid_collisions.open_r_gripper()
        else:
            raise Exception('Wrong tip link')

        kitchen_setup_avoid_collisions.detach_object(milk_name)

        #kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPose',
        #                                             root_link=kitchen_setup_avoid_collisions.default_root,
        #                                             tip_link=tip_link,
        #                                             goal=milk_grasp_pre_pose)
        # kitchen_setup_avoid_collisions.send_and_check_goal()

        kitchen_setup_avoid_collisions.set_joint_goal(gaya_pose)

    def test_ease_fridge_with_cart_goals_and_global_planner(self, kitchen_setup_avoid_collisions):
        rospy.sleep(10.0)#0.5

        tip_link = kitchen_setup_avoid_collisions.r_tip
        milk_name = u'milk'

        # take milk out of fridge
        kitchen_setup_avoid_collisions.set_kitchen_js({u'iai_fridge_door_joint': 1.56})

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 0.4
        base_goal.pose.position.y = -1.05
        base_goal.pose.orientation.x = 0
        base_goal.pose.orientation.y = 0
        base_goal.pose.orientation.z = 0
        base_goal.pose.orientation.w = 1
        kitchen_setup_avoid_collisions.teleport_base(base_goal)

        # spawn milk
        milk_pose = PoseStamped()
        milk_pose.header.frame_id = u'iai_kitchen/iai_fridge_door_shelf1_bottom'
        milk_pose.pose.position = Point(0, 0, 0.13)
        milk_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup_avoid_collisions.add_box(milk_name, [0.05, 0.05, 0.2], milk_pose)

        # move arm towards milk
        if tip_link == kitchen_setup_avoid_collisions.l_tip:
            kitchen_setup_avoid_collisions.open_l_gripper()
        elif tip_link == kitchen_setup_avoid_collisions.r_tip:
            kitchen_setup_avoid_collisions.open_r_gripper()
        else:
            raise Exception('Wrong tip link')

        kitchen_setup_avoid_collisions.set_json_goal('CartesianPreGrasp',
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     tip_link=tip_link,
                                                     grasping_goal=milk_pose)
        rospy.logerr('Pregrasping')
        kitchen_setup_avoid_collisions.plan_and_execute()

        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPose',
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     tip_link=tip_link,
                                                     goal=milk_pose)
        kitchen_setup_avoid_collisions.plan_and_execute()

        # grasp milk
        kitchen_setup_avoid_collisions.attach_object(milk_name, tip_link)

        if tip_link == kitchen_setup_avoid_collisions.l_tip:
            kitchen_setup_avoid_collisions.close_l_gripper()
        elif tip_link == kitchen_setup_avoid_collisions.r_tip:
            kitchen_setup_avoid_collisions.close_r_gripper()
        else:
            raise Exception('Wrong tip link')

        kitchen_setup_avoid_collisions.set_json_goal('CartesianPreGrasp',
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     tip_link=tip_link,
                                                     grasping_goal=milk_pose)
        rospy.logerr('Pregrasping')
        kitchen_setup_avoid_collisions.plan_and_execute()

        kitchen_setup_avoid_collisions.set_joint_goal(gaya_pose)
        kitchen_setup_avoid_collisions.plan_and_execute()

        # place milk back
        kitchen_setup_avoid_collisions.set_json_goal('CartesianPreGrasp',
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     tip_link=tip_link,
                                                     grasping_goal=milk_pose)
        rospy.logerr('Pregrasping')
        kitchen_setup_avoid_collisions.plan_and_execute()
        kitchen_setup_avoid_collisions.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup_avoid_collisions.set_cart_goal(milk_pose, milk_name, kitchen_setup_avoid_collisions.default_root)
        kitchen_setup_avoid_collisions.plan_and_execute()

        # kitchen_setup.keep_position(kitchen_setup_avoid_collisions.r_tip)
        if tip_link == kitchen_setup_avoid_collisions.l_tip:
            kitchen_setup_avoid_collisions.open_l_gripper()
        elif tip_link == kitchen_setup_avoid_collisions.r_tip:
            kitchen_setup_avoid_collisions.open_r_gripper()
        else:
            raise Exception('Wrong tip link')

        kitchen_setup_avoid_collisions.detach_object(milk_name)

        #kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPose',
        #                                             root_link=kitchen_setup_avoid_collisions.default_root,
        #                                             tip_link=tip_link,
        #                                             goal=milk_grasp_pre_pose)
        # kitchen_setup_avoid_collisions.send_and_check_goal()

        kitchen_setup_avoid_collisions.set_joint_goal(gaya_pose)

    def test_ease_fridge_with_cart_goals_and_global_planner_aligned_planes(self, kitchen_setup_avoid_collisions):
        rospy.sleep(10.0)#0.5

        tip_link = kitchen_setup_avoid_collisions.r_tip
        milk_name = u'milk'

        # take milk out of fridge
        kitchen_setup_avoid_collisions.set_kitchen_js({u'iai_fridge_door_joint': 1.56})

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 0.4
        base_goal.pose.position.y = -1.05
        base_goal.pose.orientation.x = 0
        base_goal.pose.orientation.y = 0
        base_goal.pose.orientation.z = 0
        base_goal.pose.orientation.w = 1
        kitchen_setup_avoid_collisions.teleport_base(base_goal)

        # spawn milk
        milk_pose = PoseStamped()
        milk_pose.header.frame_id = u'iai_kitchen/iai_fridge_door_shelf1_bottom'
        milk_pose.pose.position = Point(0, 0, 0.12)
        milk_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        milk_lift_pre_pose = PoseStamped()
        milk_lift_pre_pose.header.frame_id = u'iai_kitchen/iai_fridge_door_shelf1_bottom'
        milk_lift_pre_pose.pose.position = Point(0, 0, 0.22)
        milk_lift_pre_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        #milk_grasp_pre_pose = PoseStamped()
        #milk_grasp_pre_pose.header.frame_id = u'iai_kitchen/iai_fridge_door_shelf1_bottom'
        #milk_grasp_pre_pose.pose.position = Point(-0.2, 0, 0.12)
        #milk_grasp_pre_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup_avoid_collisions.add_box(milk_name, [0.05, 0.05, 0.2], milk_pose)

        # move arm towards milk
        if tip_link == kitchen_setup_avoid_collisions.l_tip:
            kitchen_setup_avoid_collisions.open_l_gripper()
        elif tip_link == kitchen_setup_avoid_collisions.r_tip:
            kitchen_setup_avoid_collisions.open_r_gripper()
        else:
            raise Exception('Wrong tip link')

        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPose',
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     tip_link=tip_link,
                                                     goal=milk_pose)

        kitchen_setup_avoid_collisions.plan_and_execute()

        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPose',
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     tip_link=tip_link,
                                                     goal=milk_pose)

        kitchen_setup_avoid_collisions.plan_and_execute()

        # grasp milk
        kitchen_setup_avoid_collisions.attach_object(milk_name, tip_link)

        if tip_link == kitchen_setup_avoid_collisions.l_tip:
            kitchen_setup_avoid_collisions.close_l_gripper()
        elif tip_link == kitchen_setup_avoid_collisions.r_tip:
            kitchen_setup_avoid_collisions.close_r_gripper()
        else:
            raise Exception('Wrong tip link')

        gripper_vec = Vector3Stamped()
        gripper_vec.header.frame_id = tip_link
        gripper_vec.vector.x = 1

        gripper_goal_vec = Vector3Stamped()
        gripper_goal_vec.header.frame_id = u'milk'
        gripper_goal_vec.vector.z = 1
        kitchen_setup_avoid_collisions.set_align_planes_goal(tip_link, gripper_vec, root_normal=gripper_goal_vec)

        kitchen_setup_avoid_collisions.set_cart_goal(milk_lift_pre_pose, milk_name, kitchen_setup_avoid_collisions.default_root)
        kitchen_setup_avoid_collisions.plan_and_execute()
        kitchen_setup_avoid_collisions.set_joint_goal(gaya_pose)
        kitchen_setup_avoid_collisions.plan_and_execute()

        # place milk back
        kitchen_setup_avoid_collisions.set_cart_goal(milk_lift_pre_pose, milk_name, kitchen_setup_avoid_collisions.default_root)
        kitchen_setup_avoid_collisions.plan_and_execute()

        kitchen_setup_avoid_collisions.set_cart_goal(milk_pose, milk_name, kitchen_setup_avoid_collisions.default_root)
        kitchen_setup_avoid_collisions.plan_and_execute()

        # kitchen_setup.keep_position(kitchen_setup_avoid_collisions.r_tip)
        if tip_link == kitchen_setup_avoid_collisions.l_tip:
            kitchen_setup_avoid_collisions.open_l_gripper()
        elif tip_link == kitchen_setup_avoid_collisions.r_tip:
            kitchen_setup_avoid_collisions.open_r_gripper()
        else:
            raise Exception('Wrong tip link')

        kitchen_setup_avoid_collisions.detach_object(milk_name)

        #kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPose',
        #                                             root_link=kitchen_setup_avoid_collisions.default_root,
        #                                             tip_link=tip_link,
        #                                             goal=milk_grasp_pre_pose)
        kitchen_setup_avoid_collisions.send_and_check_goal()

        kitchen_setup_avoid_collisions.send_and_check_joint_goal(gaya_pose)

    def test_faster_ease_cereal_with_planner(self, kitchen_setup_avoid_collisions):
        """
        :type kitchen_setup_avoid_collisions: PR2
        """
        # FIXME collision avoidance needs soft_threshholds at 0
        cereal_name = u'cereal'
        drawer_frame_id = u'iai_kitchen/oven_area_area_right_drawer_board_2_link'

        # take milk out of fridge
        kitchen_setup_avoid_collisions.set_kitchen_js({u'oven_area_area_right_drawer_main_joint': 0.48})

        # spawn milk
        cereal_pose = PoseStamped()
        cereal_pose.header.frame_id = drawer_frame_id
        cereal_pose.pose.position = Point(0.123, 0.0, 0.15)
        cereal_pose.pose.orientation = Quaternion(0.0087786, 0.005395, -0.838767, -0.544393)
        kitchen_setup_avoid_collisions.add_box(cereal_name, [0.1028, 0.0634, 0.20894], cereal_pose)

        cereal_pose_in_map = tf.msg_to_kdl(tf.transform_pose(u'map', cereal_pose))

        drawer_T_box = tf.msg_to_kdl(cereal_pose)

        # grasp milk
        kitchen_setup_avoid_collisions.open_l_gripper()
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

        kitchen_setup_avoid_collisions.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup_avoid_collisions.set_joint_goal(oven_area_cereal, check=False)
        kitchen_setup_avoid_collisions.plan_and_execute()

        kitchen_setup_avoid_collisions.attach_object(cereal_name, kitchen_setup_avoid_collisions.r_tip)
        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        kitchen_setup_avoid_collisions.close_l_gripper()

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
        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPathCarrot',
                                                     tip_link=kitchen_setup_avoid_collisions.r_tip,
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     goal=post_grasp_pose,
                                                     goal_sampling_axis=[True,False,False])
        kitchen_setup_avoid_collisions.plan_and_execute()
        #kitchen_setup.set_joint_goal(gaya_pose)

        # place milk back

        # kitchen_setup.add_json_goal(u'BasePointingForward')
        # milk_goal = PoseStamped()
        # milk_goal.header.frame_id = u'iai_kitchen/kitchen_island_surface'
        # milk_goal.pose.position = Point(.1, -.2, .13)
        # milk_goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0,0,1]))

        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPathCarrot',
                                                     tip_link=kitchen_setup_avoid_collisions.r_tip,
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     goal=grasp_pose)
        kitchen_setup_avoid_collisions.plan_and_execute()

        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        kitchen_setup_avoid_collisions.open_l_gripper()
        kitchen_setup_avoid_collisions.detach_object(cereal_name)

        kitchen_setup_avoid_collisions.set_joint_goal(gaya_pose)
        kitchen_setup_avoid_collisions.plan_and_execute()

    def test_ease_cereal_with_planner(self, kitchen_setup_avoid_collisions):
        # FIXME collision avoidance needs soft_threshholds at 0
        cereal_name = u'cereal'
        drawer_frame_id = u'iai_kitchen/oven_area_area_right_drawer_board_2_link'

        # take milk out of fridge
        kitchen_setup_avoid_collisions.set_kitchen_js({u'oven_area_area_right_drawer_main_joint': 0.48})

        # spawn milk

        cereal_pose = PoseStamped()
        cereal_pose.header.frame_id = drawer_frame_id
        cereal_pose.pose.position = Point(0.123, 0.0, 0.15)
        cereal_pose.pose.orientation = Quaternion(0.0087786, 0.005395, -0.838767, -0.544393)
        kitchen_setup_avoid_collisions.add_box(cereal_name, [0.1028, 0.0634, 0.20894], cereal_pose)

        drawer_T_box = tf.msg_to_kdl(cereal_pose)

        # grasp milk
        kitchen_setup_avoid_collisions.open_l_gripper()
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

        kitchen_setup_avoid_collisions.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPose',
                                                     tip_link=kitchen_setup_avoid_collisions.r_tip,
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     goal=pre_grasp_pose)
        kitchen_setup_avoid_collisions.plan_and_execute()

        grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal, drawer_frame_id)

        kitchen_setup_avoid_collisions.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup_avoid_collisions.set_cart_goal(grasp_pose,
                                                     tip_link=kitchen_setup_avoid_collisions.r_tip)
        kitchen_setup_avoid_collisions.plan_and_execute()

        kitchen_setup_avoid_collisions.attach_object(cereal_name, kitchen_setup_avoid_collisions.r_tip)
        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        kitchen_setup_avoid_collisions.close_l_gripper()

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
        kitchen_setup_avoid_collisions.god_map.set_data(identifier.rosparam + ['reset_god_map'], False)
        decrease_external_collision_avoidance(kitchen_setup_avoid_collisions)
        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPathCarrot',
                                                     tip_link=kitchen_setup_avoid_collisions.r_tip,
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     goal=post_grasp_pose,
                                                     goal_sampling_axis=[True, False, False])
        kitchen_setup_avoid_collisions.plan_and_execute()
        #kitchen_setup.set_joint_goal(gaya_pose)

        # place milk back

        # kitchen_setup.add_json_goal(u'BasePointingForward')
        # milk_goal = PoseStamped()
        # milk_goal.header.frame_id = u'iai_kitchen/kitchen_island_surface'
        # milk_goal.pose.position = Point(.1, -.2, .13)
        # milk_goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0,0,1]))

        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPathCarrot',
                                                     tip_link=kitchen_setup_avoid_collisions.r_tip,
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     goal=grasp_pose)
        kitchen_setup_avoid_collisions.plan_and_execute()
        kitchen_setup_avoid_collisions.god_map.set_data(identifier.rosparam + ['reset_god_map'], True)

        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        kitchen_setup_avoid_collisions.open_l_gripper()
        kitchen_setup_avoid_collisions.detach_object(cereal_name)

        kitchen_setup_avoid_collisions.set_joint_goal(gaya_pose)
        kitchen_setup_avoid_collisions.plan_and_execute()

    def test_ease_cereal_different_drawers(self, kitchen_setup_avoid_collisions):
        # FIXME collision avoidance needs soft_threshholds at 0
        cereal_name = u'cereal'
        drawer_frame_id = u'iai_kitchen/oven_area_area_right_drawer_board_3_link'

        # take milk out of fridge
        kitchen_setup_avoid_collisions.set_kitchen_js({u'oven_area_area_right_drawer_main_joint': 0.48})

        # spawn milk

        cereal_pose = PoseStamped()
        cereal_pose.header.frame_id = drawer_frame_id
        cereal_pose.pose.position = Point(0.123, 0.0, 0.13)
        cereal_pose.pose.orientation = Quaternion(0.0087786, 0.005395, -0.838767, -0.544393)
        kitchen_setup_avoid_collisions.add_box(cereal_name, [0.1028, 0.0634, 0.20894], cereal_pose)

        drawer_T_box = tf.msg_to_kdl(cereal_pose)

        # grasp milk
        kitchen_setup_avoid_collisions.open_l_gripper()
        grasp_pose = PoseStamped()
        grasp_pose.header.frame_id = cereal_name
        grasp_pose.pose.position = Point(0.13, 0, 0.05)
        grasp_pose.pose.orientation = Quaternion(0, 0, 1, 0)
        box_T_r_goal = tf.msg_to_kdl(grasp_pose)
        box_T_r_goal_pre = deepcopy(box_T_r_goal)
        box_T_r_goal_pre.p[0] += 0.1

        pre_grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal_pre, drawer_frame_id)

        box_T_r_goal_post = deepcopy(box_T_r_goal)
        box_T_r_goal_post.p[0] += 0.3
        post_grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal_post, drawer_frame_id)

        kitchen_setup_avoid_collisions.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPose',
                                                     tip_link=kitchen_setup_avoid_collisions.r_tip,
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     goal=pre_grasp_pose)
        kitchen_setup_avoid_collisions.plan_and_execute()

        grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal, drawer_frame_id)

        kitchen_setup_avoid_collisions.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup_avoid_collisions.set_cart_goal(grasp_pose,
                                                     tip_link=kitchen_setup_avoid_collisions.r_tip)
        kitchen_setup_avoid_collisions.plan_and_execute()

        kitchen_setup_avoid_collisions.attach_object(cereal_name, kitchen_setup_avoid_collisions.r_tip)
        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        kitchen_setup_avoid_collisions.close_l_gripper()

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
        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPose',
                                                     tip_link=kitchen_setup_avoid_collisions.r_tip,
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     goal=post_grasp_pose)
        kitchen_setup_avoid_collisions.plan_and_execute()
        #kitchen_setup.set_joint_goal(gaya_pose)

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

        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPose',
                                                     tip_link=kitchen_setup_avoid_collisions.r_tip,
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     goal=grasp_pose)
        kitchen_setup_avoid_collisions.plan_and_execute()

        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        kitchen_setup_avoid_collisions.open_l_gripper()
        kitchen_setup_avoid_collisions.detach_object(cereal_name)

        kitchen_setup_avoid_collisions.set_joint_goal(gaya_pose)
        kitchen_setup_avoid_collisions.plan_and_execute()


class TestShaking(object):
    def test_wiggle_prismatic_joint_neglectable_shaking(self, kitchen_setup):
        sample_period = kitchen_setup.god_map.get_data(identifier.sample_period)
        frequency_range = kitchen_setup.god_map.get_data(identifier.frequency_range)
        amplitude_threshold = kitchen_setup.god_map.get_data(identifier.amplitude_threshold)
        max_detectable_freq = int(1 / (2 * sample_period))
        min_wiggle_frequency = int(frequency_range * max_detectable_freq)
        while np.fmod(min_wiggle_frequency, 5.0) != 0.0:
            min_wiggle_frequency += 1
        distance_between_frequencies = 5

        for i, t in enumerate([(u'torso_lift_joint', 0.05), (u'odom_x_joint', 0.5)]):  # max vel: 0.015 and 0.5
            for f in range(min_wiggle_frequency, max_detectable_freq, distance_between_frequencies):
                target_freq = float(f)
                joint = t[0]
                goal = t[1]
                kitchen_setup.set_json_goal(u'JointPositionPrismatic',
                                            joint_name=joint,
                                            goal=0.0,
                                            )
                kitchen_setup.send_goal()
                kitchen_setup.set_json_goal(u'ShakyJointPositionRevoluteOrPrismatic',
                                            joint_name=joint,
                                            noise_amplitude=amplitude_threshold - 0.05,
                                            goal=goal,
                                            frequency=target_freq
                                            )
                kitchen_setup.send_and_check_goal()

    def test_wiggle_revolute_joint_neglectable_shaking(self, kitchen_setup):
        sample_period = kitchen_setup.god_map.get_data(identifier.sample_period)
        frequency_range = kitchen_setup.god_map.get_data(identifier.frequency_range)
        amplitude_threshold = kitchen_setup.god_map.get_data(identifier.amplitude_threshold)
        max_detectable_freq = int(1 / (2 * sample_period))
        min_wiggle_frequency = int(frequency_range * max_detectable_freq)
        while np.fmod(min_wiggle_frequency, 5.0) != 0.0:
            min_wiggle_frequency += 1
        distance_between_frequencies = 5

        for i, joint in enumerate([u'r_wrist_flex_joint', u'head_pan_joint']):  # max vel: 1.0 and 0.5
            for f in range(min_wiggle_frequency, max_detectable_freq, distance_between_frequencies):
                target_freq = float(f)
                kitchen_setup.set_json_goal(u'JointPositionRevolute',
                                            joint_name=joint,
                                            goal=0.0,
                                            )
                kitchen_setup.send_goal()
                kitchen_setup.set_json_goal(u'ShakyJointPositionRevoluteOrPrismatic',
                                            joint_name=joint,
                                            noise_amplitude=amplitude_threshold - 0.05,
                                            goal=-1.0,
                                            frequency=target_freq
                                            )
                kitchen_setup.send_and_check_goal()

    def test_wiggle_continuous_joint_neglectable_shaking(self, kitchen_setup):
        sample_period = kitchen_setup.god_map.get_data(identifier.sample_period)
        frequency_range = kitchen_setup.god_map.get_data(identifier.frequency_range)
        amplitude_threshold = kitchen_setup.god_map.get_data(identifier.amplitude_threshold)
        max_detectable_freq = int(1 / (2 * sample_period))
        min_wiggle_frequency = int(frequency_range * max_detectable_freq)
        while np.fmod(min_wiggle_frequency, 5.0) != 0.0:
            min_wiggle_frequency += 1
        distance_between_frequencies = 5

        for continuous_joint in [u'l_wrist_roll_joint', u'r_forearm_roll_joint']:  # max vel. of 1.0 and 1.0
            for f in range(min_wiggle_frequency, max_detectable_freq, distance_between_frequencies):
                kitchen_setup.set_json_goal(u'JointPositionContinuous',
                                            joint_name=continuous_joint,
                                            goal=5.0,
                                            )
                kitchen_setup.send_goal()
                target_freq = float(f)
                kitchen_setup.set_json_goal(u'ShakyJointPositionContinuous',
                                            joint_name=continuous_joint,
                                            goal=-5.0,
                                            noise_amplitude=amplitude_threshold - 0.05,
                                            frequency=target_freq
                                            )
                kitchen_setup.send_and_check_goal()

    def test_wiggle_revolute_joint_shaking(self, kitchen_setup):
        sample_period = kitchen_setup.god_map.get_data(identifier.sample_period)
        frequency_range = kitchen_setup.god_map.get_data(identifier.frequency_range)
        max_detectable_freq = int(1 / (2 * sample_period))
        min_wiggle_frequency = int(frequency_range * max_detectable_freq)
        while np.fmod(min_wiggle_frequency, 5.0) != 0.0:
            min_wiggle_frequency += 1
        distance_between_frequencies = 5

        for joint in [u'head_pan_joint', u'r_wrist_flex_joint']:  # max vel: 1.0 and 0.5
            for f in range(min_wiggle_frequency, max_detectable_freq, distance_between_frequencies):
                kitchen_setup.set_json_goal(u'JointPositionRevolute',
                                            joint_name=joint,
                                            goal=0.5,
                                            )
                kitchen_setup.send_goal()
                target_freq = float(f)
                kitchen_setup.set_json_goal(u'ShakyJointPositionRevoluteOrPrismatic',
                                            joint_name=joint,
                                            goal=0.0,
                                            frequency=target_freq
                                            )
                kitchen_setup.send_and_check_goal()
                # r = kitchen_setup.send_goal(goal=None, goal_type=MoveGoal.PLAN_AND_EXECUTE)
                # assert len(r.error_codes) != 0
                # error_code = r.error_codes[0]
                # assert error_code == MoveResult.SHAKING
                # error_message = r.error_messages[0]
                # freqs_str = re.findall("[0-9]+\.[0-9]+ hertz", error_message)
                # assert any(map(lambda f_str: float(f_str[:-6]) == target_freq, freqs_str))

    def test_wiggle_prismatic_joint_shaking(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        sample_period = kitchen_setup.god_map.get_data(identifier.sample_period)
        frequency_range = kitchen_setup.god_map.get_data(identifier.frequency_range)
        max_detectable_freq = int(1 / (2 * sample_period))
        min_wiggle_frequency = int(frequency_range * max_detectable_freq)
        while np.fmod(min_wiggle_frequency, 5.0) != 0.0:
            min_wiggle_frequency += 1
        distance_between_frequencies = 5

        for joint in [u'odom_x_joint']:  # , u'torso_lift_joint']: # max vel: 0.015 and 0.5
            for f in range(min_wiggle_frequency, max_detectable_freq, distance_between_frequencies):
                kitchen_setup.set_json_goal(u'JointPositionPrismatic',
                                            joint_name=joint,
                                            goal=0.02,
                                            )
                kitchen_setup.send_goal()
                target_freq = float(f)
                kitchen_setup.set_json_goal(u'ShakyJointPositionRevoluteOrPrismatic',
                                            joint_name=joint,
                                            goal=0.0,
                                            frequency=target_freq
                                            )
                kitchen_setup.send_and_check_goal()
                # r = kitchen_setup.send_goal(goal=None, goal_type=MoveGoal.PLAN_AND_EXECUTE)
                # assert len(r.error_codes) != 0
                # error_code = r.error_codes[0]
                # assert error_code == MoveResult.SHAKING
                # error_message = r.error_messages[0]
                # freqs_str = re.findall("[0-9]+\.[0-9]+ hertz", error_message)
                # assert any(map(lambda f_str: float(f_str[:-6]) == target_freq, freqs_str))

    def test_wiggle_continuous_joint_shaking(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        sample_period = kitchen_setup.god_map.get_data(identifier.sample_period)
        frequency_range = kitchen_setup.god_map.get_data(identifier.frequency_range)
        max_detectable_freq = int(1 / (2 * sample_period))
        min_wiggle_frequency = int(frequency_range * max_detectable_freq)
        while np.fmod(min_wiggle_frequency, 5.0) != 0.0:
            min_wiggle_frequency += 1
        distance_between_frequencies = 5

        for continuous_joint in [u'l_wrist_roll_joint', u'r_forearm_roll_joint']:  # max vel. of 1.0 and 1.0
            for f in range(min_wiggle_frequency, max_detectable_freq, distance_between_frequencies):
                kitchen_setup.set_json_goal(u'JointPositionContinuous',
                                            joint_name=continuous_joint,
                                            goal=5.0,
                                            )
                kitchen_setup.send_goal()
                target_freq = float(f)
                kitchen_setup.set_json_goal(u'ShakyJointPositionContinuous',
                                            joint_name=continuous_joint,
                                            goal=-5.0,
                                            frequency=target_freq
                                            )
                kitchen_setup.send_and_check_goal()
                # r = kitchen_setup.send_goal(goal=None, goal_type=MoveGoal.PLAN_AND_EXECUTE)
                # assert len(r.error_codes) != 0
                # error_code = r.error_codes[0]
                # assert error_code == MoveResult.SUCCESS
                # error_message = r.error_messages[0]
                # freqs_str = re.findall("[0-9]+\.[0-9]+ hertz", error_message)
                # assert any(map(lambda f_str: float(f_str[:-6]) == target_freq, freqs_str))

    def test_only_revolute_joint_shaking(self, kitchen_setup):
        sample_period = kitchen_setup.god_map.get_data(identifier.sample_period)
        frequency_range = kitchen_setup.god_map.get_data(identifier.frequency_range)
        amplitude_threshold = kitchen_setup.god_map.get_data(identifier.amplitude_threshold)
        max_detectable_freq = int(1 / (2 * sample_period))
        min_wiggle_frequency = int(frequency_range * max_detectable_freq)
        while np.fmod(min_wiggle_frequency, 5.0) != 0.0:
            min_wiggle_frequency += 1
        distance_between_frequencies = 5

        for revolute_joint in [u'r_wrist_flex_joint', u'head_pan_joint']:  # max vel. of 1.0 and 1.0
            for f in range(min_wiggle_frequency, max_detectable_freq, distance_between_frequencies):
                target_freq = float(f)

                if f == min_wiggle_frequency:
                    kitchen_setup.set_json_goal(u'JointPositionRevolute',
                                                joint_name=revolute_joint,
                                                goal=0.0,
                                                )
                    kitchen_setup.send_goal()

                kitchen_setup.set_json_goal(u'ShakyJointPositionRevoluteOrPrismatic',
                                            joint_name=revolute_joint,
                                            goal=0.0,
                                            noise_amplitude=amplitude_threshold + 0.02,
                                            frequency=target_freq
                                            )
                kitchen_setup.send_and_check_goal()
                # r = kitchen_setup.send_goal(goal=None, goal_type=MoveGoal.PLAN_AND_EXECUTE)
                # assert len(r.error_codes) != 0
                # error_code = r.error_codes[0]
                # assert error_code == MoveResult.SHAKING
                # error_message = r.error_messages[0]
                # freqs_str = re.findall("[0-9]+\.[0-9]+ hertz", error_message)
                # assert any(map(lambda f_str: float(f_str[:-6]) == target_freq, freqs_str))

    def test_only_revolute_joint_neglectable_shaking(self, kitchen_setup):
        sample_period = kitchen_setup.god_map.get_data(identifier.sample_period)
        frequency_range = kitchen_setup.god_map.get_data(identifier.frequency_range)
        amplitude_threshold = kitchen_setup.god_map.get_data(identifier.amplitude_threshold)
        max_detectable_freq = int(1 / (2 * sample_period))
        min_wiggle_frequency = int(frequency_range * max_detectable_freq)
        while np.fmod(min_wiggle_frequency, 5.0) != 0.0:
            min_wiggle_frequency += 1
        distance_between_frequencies = 5

        for revolute_joint in [u'r_wrist_flex_joint', u'head_pan_joint']:  # max vel. of 1.0 and 0.5
            for f in range(min_wiggle_frequency, max_detectable_freq, distance_between_frequencies):
                target_freq = float(f)

                if f == min_wiggle_frequency:
                    kitchen_setup.set_json_goal(u'JointPositionRevolute',
                                                joint_name=revolute_joint,
                                                goal=0.0,
                                                )
                    kitchen_setup.send_goal()

                kitchen_setup.set_json_goal(u'ShakyJointPositionRevoluteOrPrismatic',
                                            joint_name=revolute_joint,
                                            goal=0.0,
                                            noise_amplitude=amplitude_threshold - 0.02,
                                            frequency=target_freq
                                            )
                r = kitchen_setup.send_goal(goal=None, goal_type=MoveGoal.PLAN_AND_EXECUTE)
                if any(map(lambda c: c == MoveResult.SHAKING, r.error_codes)):
                    error_message = r.error_messages[0]
                    freqs_str = re.findall("[0-9]+\.[0-9]+ hertz", error_message)
                    assert all(map(lambda f_str: float(f_str[:-6]) != target_freq, freqs_str))
                else:
                    assert True


class TestCollisionAvoidanceGoals(object):

    # def test_wiggle4(self, pocky_pose_setup):
    #     """
    #     :type pocky_pose_setup: PR2
    #     """
    #     # FIXME
    #     p = PoseStamped()
    #     p.header.frame_id = u'map'
    #     p.pose.position.x = 1.1
    #     p.pose.position.y = 0
    #     p.pose.position.z = 0.6
    #     p.pose.orientation.w = 1
    #     pocky_pose_setup.add_box(size=[1, 1, 0.01], pose=p)
    #
    #     p = PoseStamped()
    #     p.header.frame_id = pocky_pose_setup.r_tip
    #     p.pose.position = Point(0.1, 0, 0)
    #     p.pose.orientation = Quaternion(0, 0, 0, 1)
    #     pocky_pose_setup.set_and_check_cart_goal(p, pocky_pose_setup.r_tip, pocky_pose_setup.default_root,
    #                                              expected_error_codes=[MoveResult.SHAKING])

    # box_setup.avoid_collision()

    # collision_entry = CollisionEntry()
    # collision_entry.type = CollisionEntry.AVOID_COLLISION
    # collision_entry.min_dist = 0.05
    # collision_entry.body_b = u'box'
    # pocky_pose_setup.add_collision_entries([collision_entry])
    #
    # pocky_pose_setup.send_and_check_goal(expected_error_code=MoveResult.INSOLVABLE)

    def test_handover(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
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
        kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()

        kitchen_setup.attach_box(size=[0.08, 0.16, 0.16],
                                 frame_id=kitchen_setup.l_tip,
                                 position=[0.0, -0.08, 0],
                                 orientation=[0, 0, 0, 1])
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
        kitchen_setup.plan_and_execute()

        kitchen_setup.detach_object('box')
        kitchen_setup.attach_object('box', kitchen_setup.r_tip)

        r_goal2 = PoseStamped()
        r_goal2.header.frame_id = 'box'
        r_goal2.pose.position.x -= -.1
        r_goal2.pose.orientation.w = 1

        kitchen_setup.set_cart_goal(r_goal2, u'box', root_link=kitchen_setup.l_tip)
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()
        # kitchen_setup.check_cart_goal(u'box', r_goal2)

    def test_clear_world(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        object_name = u'muh'
        p = PoseStamped()
        p.header.frame_id = u'map'
        p.pose.position = Point(1.2, 0, 1.6)
        p.pose.orientation = Quaternion(0.0, 0.0, 0.47942554, 0.87758256)
        zero_pose.add_box(object_name, pose=p)
        zero_pose.clear_world()
        object_name = u'muh2'
        p = PoseStamped()
        p.header.frame_id = u'map'
        p.pose.position = Point(1.2, 0, 1.6)
        p.pose.orientation = Quaternion(0.0, 0.0, 0.47942554, 0.87758256)
        zero_pose.add_box(object_name, pose=p)
        zero_pose.clear_world()
        zero_pose.plan_and_execute()

    def test_only_collision_avoidance(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.plan_and_execute()

    def test_add_mesh(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        object_name = u'muh'
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.add_mesh(object_name, mesh=u'package://giskardpy/test/urdfs/meshes/bowl_21.obj', pose=p)

    def test_add_non_existing_mesh(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        object_name = u'muh'
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.add_mesh(object_name, mesh=u'package://giskardpy/test/urdfs/meshes/muh.obj', pose=p,
                           expected_error_code=UpdateWorldResponse.CORRUPT_MESH_ERROR)

    def test_mesh_collision_avoidance(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.close_r_gripper()
        object_name = u'muh'
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.position = Point(0.01, 0, 0)
        p.pose.orientation = Quaternion(*quaternion_about_axis(-np.pi / 2, [0, 1, 0]))
        zero_pose.add_mesh(object_name, mesh=u'package://giskardpy/test/urdfs/meshes/bowl_21.obj', pose=p)
        zero_pose.plan_and_execute()

    def test_add_box_twice(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        object_name = u'muh'
        p = PoseStamped()
        p.header.frame_id = u'map'
        p.pose.position = Point(1.2, 0, 1.6)
        p.pose.orientation = Quaternion(0.0, 0.0, 0.47942554, 0.87758256)
        zero_pose.add_box(object_name, size=[1, 1, 1], pose=p)
        zero_pose.add_box(object_name, size=[1, 1, 1], pose=p,
                          expected_error_code=UpdateWorldResponse.DUPLICATE_BODY_ERROR)

    def test_add_remove_sphere(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        object_name = u'muh'
        p = PoseStamped()
        p.header.frame_id = u'map'
        p.pose.position.x = 1.2
        p.pose.position.y = 0
        p.pose.position.z = 1.6
        p.pose.orientation.w = 1
        zero_pose.add_sphere(object_name, radius=1, pose=p)
        zero_pose.remove_object(object_name)

    def test_add_remove_cylinder(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        object_name = u'muh'
        p = PoseStamped()
        p.header.frame_id = u'map'
        p.pose.position.x = 0.5
        p.pose.position.y = 0
        p.pose.position.z = 0
        p.pose.orientation.w = 1
        zero_pose.add_cylinder(object_name, height=1, radius=1, pose=p)
        zero_pose.remove_object(object_name)

    def test_add_urdf_body(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        object_name = u'kitchen'
        kitchen_setup.clear_world()
        kitchen_setup.add_urdf(object_name, rospy.get_param(u'kitchen_description'),
                               tf.lookup_pose(u'map', u'iai_kitchen/world'), u'/kitchen/joint_states',
                               set_js_topic=u'/kitchen/cram_joint_states')
        kitchen_setup.remove_object(object_name)
        kitchen_setup.add_urdf(object_name, rospy.get_param(u'kitchen_description'),
                               tf.lookup_pose(u'map', u'iai_kitchen/world'), u'/kitchen/joint_states',
                               set_js_topic=u'/kitchen/cram_joint_states')

    def test_attach_box_as_eef(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        pocky = u'http://muh#pocky'
        box_pose = PoseStamped()
        box_pose.header.frame_id = zero_pose.r_tip
        box_pose.pose.position = Point(0.05, 0, 0, )
        box_pose.pose.orientation = Quaternion(1, 0, 0, 0)
        zero_pose.attach_box(pocky, [0.1, 0.02, 0.02], zero_pose.r_tip, box_pose)
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.orientation.w = 1
        zero_pose.set_cart_goal(p, pocky, zero_pose.default_root)
        p = tf.transform_pose(zero_pose.default_root, p)
        zero_pose.plan_and_execute()
        p2 = zero_pose.robot.compute_fk_pose(zero_pose.default_root, pocky)
        compare_poses(p2.pose, p.pose)
        zero_pose.detach_object(pocky)
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.orientation.w = 1
        p.pose.position.x = -.1
        zero_pose.set_cart_goal(p, zero_pose.r_tip, zero_pose.default_root)
        zero_pose.plan_and_execute()

    def test_attach_remove_box(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        pocky = u'http://muh#pocky'
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.position = Point(0.05, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.attach_box(pocky, [0.1, 0.02, 0.02], zero_pose.r_tip, p)
        zero_pose.detach_object(pocky)
        zero_pose.remove_object(pocky)

    def test_attach_remove_sphere(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        pocky = u'http://muh#pocky'
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.position = Point(0.05, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.attach_sphere(pocky, 1, zero_pose.r_tip, p)
        zero_pose.detach_object(pocky)
        zero_pose.remove_object(pocky)

    def test_attach_remove_cylinder(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        pocky = u'http://muh#pocky'
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.position = Point(0.05, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.attach_cylinder(pocky, 1, 1, zero_pose.r_tip, p)
        zero_pose.detach_object(pocky)
        zero_pose.remove_object(pocky)

    def test_attach_remove_box2(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.set_joint_goal(gaya_pose)
        zero_pose.plan_and_execute()
        pocky = u'http://muh#pocky'
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.orientation.w = 1
        zero_pose.add_box(pocky, size=[1,1,1], pose=p)
        for i in range(3):
            zero_pose.attach_object(pocky, zero_pose.r_tip)
            zero_pose.detach_object(pocky)
        zero_pose.remove_object(pocky)

    def test_remove_attached_box(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        pocky = u'http://muh#pocky'
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.position = Point(0.05, 0, 0)
        p.pose.orientation.w = 1
        zero_pose.attach_box(pocky, [0.1, 0.02, 0.02], zero_pose.r_tip, p)
        zero_pose.remove_object(pocky)

    def test_attach_existing_box(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        pocky = u'http://muh#pocky'
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.position = Point(0.05, 0, 0)
        p.pose.orientation = Quaternion(0., 0., 0.47942554, 0.87758256)
        zero_pose.add_box(pocky, [0.1, 0.02, 0.02], pose=p)
        zero_pose.attach_object(pocky, frame_id=zero_pose.r_tip)
        relative_pose = zero_pose.robot.compute_fk_pose(zero_pose.r_tip, pocky).pose
        compare_poses(p.pose, relative_pose)

    def test_add_attach_detach_remove_add(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        object_name = u'muh'
        p = PoseStamped()
        p.header.frame_id = u'map'
        p.pose.position = Point(1.2, 0, 1.6)
        p.pose.orientation = Quaternion(0.0, 0.0, 0.47942554, 0.87758256)
        zero_pose.add_box(object_name, size=[1,1,1], pose=p)
        zero_pose.attach_object(object_name, frame_id=zero_pose.r_tip)
        zero_pose.detach_object(object_name)
        zero_pose.remove_object(object_name)
        zero_pose.add_box(object_name, size=[1,1,1], pose=p)
        assert zero_pose.get_attached_objects().object_names == []

    def test_attach_existing_box2(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        pocky = u'http://muh#pocky'
        old_p = PoseStamped()
        old_p.header.frame_id = zero_pose.r_tip
        old_p.pose.position = Point(0.05, 0, 0)
        old_p.pose.orientation = Quaternion(0., 0., 0.47942554, 0.87758256)
        zero_pose.add_box(pocky, [0.1, 0.02, 0.02], pose=old_p)
        zero_pose.attach_object(pocky, frame_id=zero_pose.r_tip)
        relative_pose = zero_pose.robot.compute_fk_pose(zero_pose.r_tip, pocky).pose
        compare_poses(old_p.pose, relative_pose)

        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.position.x = -0.1
        p.pose.orientation.w = 1.0
        zero_pose.set_cart_goal(p, zero_pose.r_tip, zero_pose.default_root)
        zero_pose.plan_and_execute()
        p.header.frame_id = u'map'
        p.pose.position.y = -1
        p.pose.orientation = Quaternion(0, 0, 0.47942554, 0.87758256)
        zero_pose.move_base(p)
        rospy.sleep(.5)

        zero_pose.detach_object(pocky)

    def test_attach_detach_twice(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        pocky = u'http://muh#pocky'
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.position = Point(0.05, 0, 0)
        p.pose.orientation = Quaternion(1, 0, 0, 0)
        zero_pose.attach_box(pocky, [0.1, 0.02, 0.02], zero_pose.r_tip, p)
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.orientation.w = 1
        zero_pose.set_cart_goal(p, pocky)
        p = tf.transform_pose(zero_pose.default_root, p)
        zero_pose.plan_and_execute()
        p2 = zero_pose.robot.compute_fk_pose(zero_pose.default_root, pocky)
        compare_poses(p2.pose, p.pose)

        zero_pose.clear_world()

        old_p = PoseStamped()
        old_p.header.frame_id = zero_pose.r_tip
        old_p.pose.position = Point(0.05, 0, 0)
        old_p.pose.orientation = Quaternion(0., 0., 0.47942554, 0.87758256)
        zero_pose.add_box(pocky, [0.1, 0.02, 0.02], pose=old_p)
        zero_pose.attach_object(pocky, frame_id=zero_pose.r_tip)
        relative_pose = zero_pose.robot.compute_fk_pose(zero_pose.r_tip, pocky).pose
        compare_poses(actual_pose=relative_pose, desired_pose=old_p.pose)

        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.position.x = -0.1
        p.pose.orientation.w = 1.0
        zero_pose.set_cart_goal(p, zero_pose.r_tip, zero_pose.default_root)
        zero_pose.plan_and_execute()

    def test_attach_to_nonexistant_robot_link(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        pocky = u'http://muh#pocky'
        zero_pose.attach_box(pocky, [0.1, 0.02, 0.02], u'', [0.05, 0, 0],
                             expected_response=UpdateWorldResponse.MISSING_BODY_ERROR)

    def test_detach_unknown_object(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.detach_object(u'nil', expected_response=UpdateWorldResponse.MISSING_BODY_ERROR)

    def test_add_remove_box(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        object_name = u'muh'
        p = PoseStamped()
        p.header.frame_id = u'map'
        p.pose.position.x = 1.2
        p.pose.position.y = 0
        p.pose.position.z = 1.6
        p.pose.orientation.w = 1
        zero_pose.add_box(object_name, size=[1,1,1], pose=p)
        zero_pose.remove_object(object_name)

    def test_invalid_update_world(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        req = UpdateWorldRequest(42, DEFAULT_WORLD_TIMEOUT, WorldBody(), True, PoseStamped())
        assert zero_pose._update_world_srv.call(req).error_codes == UpdateWorldResponse.INVALID_OPERATION

    def test_missing_body_error(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.remove_object(u'muh', expected_response=UpdateWorldResponse.MISSING_BODY_ERROR)

    def test_corrupt_shape_error(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        p = PoseStamped()
        p.header.frame_id = 'base_link'
        req = UpdateWorldRequest(UpdateWorldRequest.ADD, DEFAULT_WORLD_TIMEOUT,
                                 WorldBody(type=WorldBody.PRIMITIVE_BODY,
                                           shape=SolidPrimitive(type=42)), True, p)
        assert zero_pose._update_world_srv.call(req).error_codes == UpdateWorldResponse.CORRUPT_SHAPE_ERROR

    def test_tf_error(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        req = UpdateWorldRequest(UpdateWorldRequest.ADD, DEFAULT_WORLD_TIMEOUT,
                                 WorldBody(type=WorldBody.PRIMITIVE_BODY,
                                           shape=SolidPrimitive(type=42)), False, PoseStamped())
        assert zero_pose._update_world_srv.call(req).error_codes == UpdateWorldResponse.TF_ERROR

    def test_unsupported_options(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        wb = WorldBody()
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = str(u'base_link')
        pose.pose.position = Point()
        pose.pose.orientation = Quaternion(w=1)
        wb.type = WorldBody.URDF_BODY

        req = UpdateWorldRequest(UpdateWorldRequest.ADD, DEFAULT_WORLD_TIMEOUT, wb, True, pose)
        assert kitchen_setup._update_world_srv.call(req).error_codes == UpdateWorldResponse.CORRUPT_URDF_ERROR

    def test_infeasible(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        pose = PoseStamped()
        pose.header.frame_id = u'map'
        pose.pose.position = Point(2, 0, 0)
        pose.pose.orientation = Quaternion(w=1)
        kitchen_setup.teleport_base(pose)
        kitchen_setup.plan_and_execute(expected_error_codes=[MoveResult.HARD_CONSTRAINTS_VIOLATED])

    def test_link_b_set_but_body_b_not(self, box_setup):
        """
        :type box_setup: PR2
        """
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.link_bs = [u'asdf']
        box_setup.set_collision_entries([ce])
        box_setup.plan_and_execute(expected_error_codes=[MoveResult.WORLD_ERROR])

    def test_unknown_robot_link(self, box_setup):
        """
        :type box_setup: PR2
        """
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.robot_links = [u'asdf']
        box_setup.set_collision_entries([ce])
        box_setup.plan_and_execute([MoveResult.UNKNOWN_OBJECT])

    def test_unknown_body_b(self, box_setup):
        """
        :type box_setup: PR2
        """
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.body_b = u'asdf'
        box_setup.set_collision_entries([ce])
        box_setup.plan_and_execute([MoveResult.UNKNOWN_OBJECT])

    def test_unknown_link_b(self, box_setup):
        """
        :type box_setup: PR2
        """
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.body_b = u'box'
        ce.link_bs = [u'asdf']
        box_setup.set_collision_entries([ce])
        box_setup.plan_and_execute([MoveResult.UNKNOWN_OBJECT])

    def test_base_link_in_collision(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.allow_self_collision()
        p = PoseStamped()
        p.header.frame_id = u'map'
        p.pose.position.x = 0
        p.pose.position.y = 0
        p.pose.position.z = -0.2
        p.pose.orientation.w = 1
        zero_pose.add_box(pose=p)
        zero_pose.set_joint_goal(pocky_pose)
        zero_pose.plan_and_execute()

    def test_unknown_object1(self, box_setup):
        """
        :type box_setup: PR2
        """
        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        box_setup.set_cart_goal(p, box_setup.r_tip, box_setup.default_root)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_COLLISION
        collision_entry.min_dist = 0.05
        collision_entry.body_b = u'muh'
        box_setup.set_collision_entries([collision_entry])

        box_setup.plan_and_execute([MoveResult.UNKNOWN_OBJECT])

    def test_allow_self_collision(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.check_cpi_geq(zero_pose.get_r_gripper_links(), 0.05)
        zero_pose.check_cpi_geq(zero_pose.get_l_gripper_links(), 0.05)

    def test_allow_self_collision2(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        goal_js = {
            u'l_elbow_flex_joint': -1.43286344265,
            u'l_forearm_roll_joint': 1.26465060073,
            u'l_shoulder_lift_joint': 0.47990329056,
            u'l_shoulder_pan_joint': 0.281272240139,
            u'l_upper_arm_roll_joint': 0.528415402668,
            u'l_wrist_flex_joint': -1.18811419869,
            u'l_wrist_roll_joint': 2.26884630124,
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
        zero_pose.check_cpi_leq([u'r_forearm_link'], 0.01)
        zero_pose.check_cpi_geq(zero_pose.get_r_gripper_links(), 0.05)

    def test_allow_self_collision3(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        goal_js = {
            u'l_elbow_flex_joint': -1.43286344265,
            u'l_forearm_roll_joint': 1.26465060073,
            u'l_shoulder_lift_joint': 0.47990329056,
            u'l_shoulder_pan_joint': 0.281272240139,
            u'l_upper_arm_roll_joint': 0.528415402668,
            u'l_wrist_flex_joint': -1.18811419869,
            u'l_wrist_roll_joint': 2.26884630124,
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

        ces = []
        ces.append(CollisionEntry(type=CollisionEntry.ALLOW_COLLISION,
                                  robot_links=zero_pose.get_l_gripper_links(),
                                  body_b=u'robot',
                                  link_bs=zero_pose.get_r_forearm_links()))
        zero_pose.set_collision_entries(ces)

        zero_pose.set_cart_goal(p, zero_pose.l_tip, zero_pose.default_root)
        zero_pose.plan_and_execute()
        zero_pose.check_cpi_leq(zero_pose.get_l_gripper_links(), 0.01)
        zero_pose.check_cpi_leq([u'r_forearm_link'], 0.01)
        zero_pose.check_cpi_geq(zero_pose.get_r_gripper_links(), 0.05)

    def test_avoid_self_collision(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        goal_js = {
            u'l_elbow_flex_joint': -1.43286344265,
            u'l_forearm_roll_joint': 1.26465060073,
            u'l_shoulder_lift_joint': 0.47990329056,
            u'l_shoulder_pan_joint': 0.281272240139,
            u'l_upper_arm_roll_joint': 0.528415402668,
            u'l_wrist_flex_joint': -1.18811419869,
            u'l_wrist_roll_joint': 2.26884630124,
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

    def test_avoid_self_collision2(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        goal_js = {
            u'r_elbow_flex_joint': -1.43286344265,
            u'r_forearm_roll_joint': -1.26465060073,
            u'r_shoulder_lift_joint': 0.47990329056,
            u'r_shoulder_pan_joint': -0.281272240139,
            u'r_upper_arm_roll_joint': -0.528415402668,
            u'r_wrist_flex_joint': -1.18811419869,
            u'r_wrist_roll_joint': 2.26884630124,
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

    def test_avoid_self_collision3(self, zero_pose):
        """
        :type zero_pose: PR2
        """
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
        zero_pose.send_goal()
        zero_pose.check_cpi_geq(zero_pose.get_r_gripper_links(), 0.048)

    def test_get_out_of_self_collision(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        goal_js = {
            u'l_elbow_flex_joint': -1.43286344265,
            u'l_forearm_roll_joint': 1.26465060073,
            u'l_shoulder_lift_joint': 0.47990329056,
            u'l_shoulder_pan_joint': 0.281272240139,
            u'l_upper_arm_roll_joint': 0.528415402668,
            u'l_wrist_flex_joint': -1.18811419869,
            u'l_wrist_roll_joint': 2.26884630124,
        }
        zero_pose.allow_all_collisions()
        zero_pose.set_joint_goal(goal_js)
        zero_pose.plan_and_execute()

        p = PoseStamped()
        p.header.frame_id = zero_pose.l_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.15
        p.pose.orientation.w = 1
        zero_pose.set_cart_goal(p, zero_pose.l_tip, u'base_footprint')
        zero_pose.allow_all_collisions()
        zero_pose.send_goal()
        zero_pose.send_goal()
        zero_pose.check_cpi_geq(zero_pose.get_l_gripper_links(), 0.048)

    def test_avoid_collision(self, box_setup):
        """
        :type box_setup: PR2
        """
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.body_b = u'box'
        # ce.min_dist = 0.05
        box_setup.set_collision_entries([ce])
        box_setup.allow_self_collision()
        # box_setup.set_joint_goal(gaya_pose)
        box_setup.plan_and_execute()
        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.048)
        box_setup.check_cpi_geq(box_setup.get_r_gripper_links(), 0.048)

    def test_collision_override(self, box_setup):
        """
        :type box_setup: PR2
        """
        p = PoseStamped()
        p.header.frame_id = box_setup.default_root
        p.pose.position.x += 0.5
        p.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0, 0, 1]))
        box_setup.teleport_base(p)
        # ce = CollisionEntry()
        # ce.type = CollisionEntry.AVOID_COLLISION
        # ce.body_b = u'box'
        # ce.min_dist = 0.05
        # box_setup.add_collision_entries([ce])
        box_setup.plan_and_execute()
        box_setup.check_cpi_geq([u'base_link'], 0.099)

    def test_avoid_collision2(self, fake_table_setup):
        """
        :type fake_table_setup: PR2
        """
        r_goal = PoseStamped()
        r_goal.header.frame_id = u'map'
        r_goal.pose.position.x = 0.8
        r_goal.pose.position.y = -0.38
        r_goal.pose.position.z = 0.84
        r_goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 2, [0, 1, 0]))
        fake_table_setup.avoid_all_collisions(0.1)
        fake_table_setup.set_cart_goal(r_goal, fake_table_setup.r_tip)
        fake_table_setup.plan_and_execute()
        fake_table_setup.check_cpi_geq(fake_table_setup.get_l_gripper_links(), 0.05)
        fake_table_setup.check_cpi_leq([u'r_gripper_l_finger_tip_link'], 0.04)
        fake_table_setup.check_cpi_leq([u'r_gripper_r_finger_tip_link'], 0.04)

    def test_allow_collision(self, box_setup):
        """
        :type box_setup: PR2
        """
        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position = Point(0.15, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_COLLISION
        collision_entry.body_b = u'box'
        collision_entry.link_bs = [u'box']
        box_setup.set_collision_entries([collision_entry])

        box_setup.allow_self_collision()
        box_setup.set_cart_goal(p, box_setup.r_tip, box_setup.default_root)
        box_setup.plan_and_execute()

        box_setup.check_cpi_leq(box_setup.get_l_gripper_links(), 0.0)
        box_setup.check_cpi_leq(box_setup.get_r_gripper_links(), 0.0)

    def test_avoid_collision3(self, pocky_pose_setup):
        """
        :type pocky_pose_setup: PR2
        """
        pocky_pose_setup.attach_box(size=[0.2, 0.05, 0.05],
                                    frame_id=pocky_pose_setup.r_tip,
                                    position=[0.08, 0, 0],
                                    orientation=[0, 0, 0, 1])
        p = PoseStamped()
        p.header.frame_id = pocky_pose_setup.r_tip
        p.pose.position.x = 0.15
        p.pose.position.y = 0.04
        p.pose.position.z = 0
        p.pose.orientation.w = 1
        pocky_pose_setup.add_box('bl', [0.1, 0.01, 0.2], pose=p)
        p = PoseStamped()
        p.header.frame_id = pocky_pose_setup.r_tip
        p.pose.position.x = 0.15
        p.pose.position.y = -0.04
        p.pose.position.z = 0
        p.pose.orientation.w = 1
        pocky_pose_setup.add_box('br', [0.1, 0.01, 0.2], pose=p)

        p = PoseStamped()
        p.header.frame_id = pocky_pose_setup.r_tip
        p.pose.position = Point(-0.15, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        pocky_pose_setup.set_cart_goal(p, pocky_pose_setup.r_tip, pocky_pose_setup.default_root)

        pocky_pose_setup.allow_self_collision()

        pocky_pose_setup.plan_and_execute()
        # TODO check traj length?
        pocky_pose_setup.check_cpi_geq(['box'], 0.048)

    def test_avoid_collision4(self, pocky_pose_setup):
        """
        :type pocky_pose_setup: PR2
        """
        pocky_pose_setup.attach_box(size=[0.2, 0.05, 0.05],
                                    frame_id=pocky_pose_setup.r_tip,
                                    position=[0.08, 0, 0],
                                    orientation=[0, 0, 0, 1])
        p = PoseStamped()
        p.header.frame_id = pocky_pose_setup.r_tip
        p.pose.position.x = 0.2
        p.pose.position.y = 0
        p.pose.position.z = 0
        p.pose.orientation.w = 1
        pocky_pose_setup.add_box('b1', [0.01, 0.2, 0.2], pose=p)
        p = PoseStamped()
        p.header.frame_id = pocky_pose_setup.r_tip
        p.pose.position.x = 0.15
        p.pose.position.y = 0.04
        p.pose.position.z = 0
        p.pose.orientation.w = 1
        pocky_pose_setup.add_box('bl', [0.1, 0.01, 0.2], pose=p)
        p = PoseStamped()
        p.header.frame_id = pocky_pose_setup.r_tip
        p.pose.position.x = 0.15
        p.pose.position.y = -0.04
        p.pose.position.z = 0
        p.pose.orientation.w = 1
        pocky_pose_setup.add_box('br', [0.1, 0.01, 0.2], pose=p)

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

    def test_avoid_collision_two_sticks(self, pocky_pose_setup):
        """
        :type pocky_pose_setup: PR2
        """
        pocky_pose_setup.attach_box(size=[0.2, 0.05, 0.05],
                                    frame_id=pocky_pose_setup.r_tip,
                                    position=[0.08, 0, 0],
                                    orientation=quaternion_about_axis(0.01, [1, 0, 0]).tolist())
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

    def test_avoid_collision5_cut_off(self, pocky_pose_setup):
        """
        :type pocky_pose_setup: PR2
        """
        pocky_pose_setup.attach_box(size=[0.2, 0.05, 0.05],
                                    frame_id=pocky_pose_setup.r_tip,
                                    position=[0.08, 0, 0],
                                    orientation=quaternion_about_axis(0.01, [1, 0, 0]).tolist())
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

    # def test_avoid_collision6(self, fake_table_setup):
    #     """
    #     :type fake_table_setup: PR2
    #     """
    #     #fixme
    #     js = {
    #         u'r_shoulder_pan_joint': -0.341482794236,
    #         u'r_shoulder_lift_joint': 0.0301123643508,
    #         u'r_upper_arm_roll_joint': -2.67555547662,
    #         u'r_forearm_roll_joint': -0.472653283346,
    #         u'r_elbow_flex_joint': -0.149999999999,
    #         u'r_wrist_flex_joint': -1.40685144215,
    #         u'r_wrist_roll_joint': 2.87855178783,
    #         u'odom_x_joint': 0.0708087929675,
    #         u'odom_y_joint': 0.052896931145,
    #         u'odom_z_joint': 0.0105784287694,
    #         u'torso_lift_joint': 0.277729421077,
    #     }
    #     # fake_table_setup.allow_all_collisions()
    #     fake_table_setup.send_and_check_joint_goal(js, weight=WEIGHT_ABOVE_CA)
    #     fake_table_setup.check_cpi_geq(fake_table_setup.get_l_gripper_links(), 0.048)
    #     fake_table_setup.check_cpi_leq([u'r_gripper_l_finger_tip_link'], 0.04)
    #     fake_table_setup.check_cpi_leq([u'r_gripper_r_finger_tip_link'], 0.04)

    def test_avoid_collision7(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        base_pose = PoseStamped()
        base_pose.header.frame_id = u'map'
        base_pose.pose.position.x = 0.8
        base_pose.pose.position.y = 1
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        kitchen_setup.teleport_base(base_pose)
        base_pose = PoseStamped()
        base_pose.header.frame_id = u'map'
        base_pose.pose.position.x = 0.64
        base_pose.pose.position.y = 0.64
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(-np.pi / 4, [0, 0, 1]))
        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.set_cart_goal(base_pose, u'base_footprint')
        kitchen_setup.plan_and_execute()

    def test_avoid_collision_at_kitchen_corner(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        base_pose = PoseStamped()
        base_pose.header.stamp = rospy.get_rostime()
        base_pose.header.frame_id = u'map'
        base_pose.pose.position.x = 0.75
        base_pose.pose.position.y = 0.9
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 2, [0, 0, 1]))
        kitchen_setup.teleport_base(base_pose)
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0, 0, 1]))
        kitchen_setup.set_joint_goal(gaya_pose)  # , weight=WEIGHT_ABOVE_CA)
        kitchen_setup.set_rotation_goal(base_pose, u'base_footprint')
        kitchen_setup.set_translation_goal(base_pose, u'base_footprint', weight=WEIGHT_BELOW_CA)
        kitchen_setup.plan_and_execute()

    def test_avoid_collision8(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        base_pose = PoseStamped()
        base_pose.header.frame_id = u'map'
        base_pose.pose.position.x = 0.8
        base_pose.pose.position.y = 0.9
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        kitchen_setup.teleport_base(base_pose)
        base_pose = PoseStamped()
        base_pose.header.frame_id = u'map'
        base_pose.pose.position.x = 0.64
        base_pose.pose.position.y = 0.64
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(-np.pi / 4, [0, 0, 1]))
        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.set_cart_goal(base_pose, 'base_footprint')
        kitchen_setup.plan_and_execute()

    def test_avoid_collision_drive_under_drawer(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        kitchen_js = {u'sink_area_left_middle_drawer_main_joint': 0.45}
        kitchen_setup.set_kitchen_js(kitchen_js)
        base_pose = PoseStamped()
        base_pose.header.frame_id = u'map'
        base_pose.pose.position.x = 0.57
        base_pose.pose.position.y = 0.5
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        kitchen_setup.teleport_base(base_pose)
        base_pose = PoseStamped()
        base_pose.header.frame_id = u'base_footprint'
        base_pose.pose.position.y = 1
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        kitchen_setup.set_cart_goal(base_pose, tip_link=u'base_footprint')
        kitchen_setup.plan_and_execute()

    def test_drive_into_wall_with_CollisionAvoidanceHint(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        tip = u'base_footprint'
        base_pose = PoseStamped()
        base_pose.header.frame_id = u'map'
        base_pose.pose.position.x = 0
        base_pose.pose.position.y = 1.5
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        kitchen_setup.teleport_base(base_pose)
        base_pose = PoseStamped()
        base_pose.header.frame_id = tip
        base_pose.pose.position.x = 1
        base_pose.pose.position.y = 0
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))

        avoidance_hint = Vector3Stamped()
        avoidance_hint.header.frame_id = u'map'
        avoidance_hint.vector.y = -1
        kitchen_setup.avoid_all_collisions(0.1)
        kitchen_setup.set_json_goal(u'CollisionAvoidanceHint',
                                    tip_link=u'base_footprint',
                                    max_threshold=0.25,
                                    spring_threshold=0.3,
                                    max_linear_velocity=1,
                                    object_name=u'kitchen',
                                    object_link_name=u'kitchen_island',
                                    avoidance_hint=avoidance_hint)
        kitchen_setup.set_joint_goal(gaya_pose)

        kitchen_setup.set_cart_goal(base_pose, tip, weight=WEIGHT_BELOW_CA)
        kitchen_setup.plan_and_execute()
        # TODO check only y distance, and that there is no collision

    def test_avoid_collision_with_far_object(self, pocky_pose_setup):
        """
        :type pocky_pose_setup: PR2
        """
        p = PoseStamped()
        p.header.frame_id = u'map'
        p.pose.position.x = 25
        p.pose.position.y = 25
        p.pose.position.z = 25
        p.pose.orientation.w = 1
        pocky_pose_setup.add_box(pose=p)
        p = PoseStamped()
        p.header.frame_id = pocky_pose_setup.r_tip
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        pocky_pose_setup.set_cart_goal(p, pocky_pose_setup.r_tip, pocky_pose_setup.default_root)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_COLLISION
        collision_entry.min_dist = 0.05
        collision_entry.body_b = u'box'
        pocky_pose_setup.set_collision_entries([collision_entry])

        pocky_pose_setup.plan_and_execute()
        pocky_pose_setup.check_cpi_geq(pocky_pose_setup.get_l_gripper_links(), 0.048)
        pocky_pose_setup.check_cpi_geq(pocky_pose_setup.get_r_gripper_links(), 0.048)

    def test_avoid_collision_touch(self, box_setup):
        """
        :type box_setup: PR2
        """
        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        box_setup.set_cart_goal(p, box_setup.r_tip, box_setup.default_root)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_ALL_COLLISIONS
        collision_entry.min_dist = 0.05
        box_setup.set_collision_entries([collision_entry])

        box_setup.plan_and_execute()

        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.0)
        box_setup.check_cpi_geq(box_setup.get_r_gripper_links(), -0.008)
        box_setup.check_cpi_leq(box_setup.get_r_gripper_links(), 0.04)

    def test_get_out_of_collision(self, box_setup):
        """
        :type box_setup: PR2
        """
        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.pose.position = Point(0.15, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        box_setup.set_cart_goal(p, box_setup.r_tip, box_setup.default_root)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_ALL_COLLISIONS
        collision_entry.min_dist = 0.05
        box_setup.set_collision_entries([collision_entry])

        box_setup.plan_and_execute()

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_ALL_COLLISIONS
        collision_entry.min_dist = 0.05
        box_setup.set_collision_entries([collision_entry])

        box_setup.plan_and_execute()

        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.0)
        box_setup.check_cpi_geq(box_setup.get_r_gripper_links(), 0.0)

    def test_allow_collision_gripper(self, box_setup):
        """
        :type box_setup: PR2
        """
        ces = box_setup.get_allow_l_gripper(u'box')
        box_setup.set_collision_entries(ces)
        p = PoseStamped()
        p.header.frame_id = box_setup.l_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.11
        p.pose.orientation.w = 1
        box_setup.set_cart_goal(p, box_setup.l_tip, box_setup.default_root)
        box_setup.plan_and_execute()
        box_setup.check_cpi_leq(box_setup.get_l_gripper_links(), 0.0)
        box_setup.check_cpi_geq(box_setup.get_r_gripper_links(), 0.048)

    def test_attached_get_below_soft_threshold(self, box_setup):
        """
        :type box_setup: PR2
        """
        attached_link_name = u'pocky'
        box_setup.attach_box(attached_link_name, [0.2, 0.04, 0.04], box_setup.r_tip, [0.05, 0, 0])
        box_setup.attach_box(attached_link_name, [0.2, 0.04, 0.04], box_setup.r_tip, [0.05, 0, 0],
                             expected_response=UpdateWorldResponse.DUPLICATE_BODY_ERROR)
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
        box_setup.set_cart_goal(p, box_setup.r_tip, box_setup.default_root)
        box_setup.plan_and_execute()
        box_setup.check_cpi_geq([attached_link_name], -0.008)
        box_setup.check_cpi_leq([attached_link_name], 0.01)
        box_setup.detach_object(attached_link_name)

    def test_attached_get_out_of_collision_below(self, box_setup):
        """
        :type box_setup: PR2
        """
        attached_link_name = u'pocky'
        box_setup.attach_box(attached_link_name, [0.2, 0.04, 0.04], box_setup.r_tip, [0.05, 0, 0])
        box_setup.attach_box(attached_link_name, [0.2, 0.04, 0.04], box_setup.r_tip, [0.05, 0, 0],
                             expected_response=UpdateWorldResponse.DUPLICATE_BODY_ERROR)
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
        box_setup.set_cart_goal(p, box_setup.r_tip, box_setup.default_root, weight=WEIGHT_BELOW_CA)
        box_setup.plan_and_execute()
        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.048)
        box_setup.check_cpi_geq([attached_link_name], 0.048)
        box_setup.detach_object(attached_link_name)

    def test_attached_get_out_of_collision_and_stay_in_hard_threshold(self, box_setup):
        """
        :type box_setup: PR2
        """
        # fixme
        attached_link_name = u'pocky'
        box_setup.attach_box(attached_link_name, [0.2, 0.04, 0.04], box_setup.r_tip, [0.05, 0, 0])
        box_setup.attach_box(attached_link_name, [0.2, 0.04, 0.04], box_setup.r_tip, [0.05, 0, 0],
                             expected_response=UpdateWorldResponse.DUPLICATE_BODY_ERROR)
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
        box_setup.set_cart_goal(p, box_setup.r_tip, box_setup.default_root)
        box_setup.plan_and_execute()
        box_setup.check_cpi_geq([attached_link_name], -0.002)
        box_setup.check_cpi_leq([attached_link_name], 0.01)
        box_setup.detach_object(attached_link_name)

    def test_attached_get_out_of_collision_stay_in(self, box_setup):
        """
        :type box_setup: PR2
        """
        attached_link_name = u'pocky'
        box_setup.attach_box(attached_link_name, [0.2, 0.04, 0.04], box_setup.r_tip, [0.05, 0, 0])
        box_setup.attach_box(attached_link_name, [0.2, 0.04, 0.04], box_setup.r_tip, [0.05, 0, 0],
                             expected_response=UpdateWorldResponse.DUPLICATE_BODY_ERROR)
        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.
        p.pose.orientation.w = 1
        box_setup.set_cart_goal(p, box_setup.r_tip, box_setup.default_root)
        box_setup.plan_and_execute()
        box_setup.check_cpi_geq([attached_link_name], -0.082)
        box_setup.detach_object(attached_link_name)

    def test_attached_get_out_of_collision_passive(self, box_setup):
        """
        :type box_setup: PR2
        """
        attached_link_name = u'pocky'
        box_setup.attach_box(attached_link_name, [0.2, 0.04, 0.04], box_setup.r_tip, [0.05, 0, 0])
        box_setup.attach_box(attached_link_name, [0.2, 0.04, 0.04], box_setup.r_tip, [0.05, 0, 0],
                             expected_response=UpdateWorldResponse.DUPLICATE_BODY_ERROR)
        box_setup.plan_and_execute()
        box_setup.check_cpi_geq([attached_link_name], 0.049)
        box_setup.detach_object(attached_link_name)

    def test_attached_collision2(self, box_setup):
        """
        :type box_setup: PR2
        """
        attached_link_name = u'pocky'
        box_setup.attach_box(attached_link_name, [0.2, 0.04, 0.04], box_setup.r_tip, [0.05, 0, 0])
        box_setup.attach_box(attached_link_name, [0.2, 0.04, 0.04], box_setup.r_tip, [0.05, 0, 0],
                             expected_response=UpdateWorldResponse.DUPLICATE_BODY_ERROR)
        box_setup.plan_and_execute()
        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.048)
        box_setup.check_cpi_geq([attached_link_name], 0.048)
        box_setup.detach_object(attached_link_name)

    def test_attached_self_collision(self, zero_pose):
        """
        :type zero_pose: PR2
        """

        collision_pose = {
            u'l_elbow_flex_joint': - 1.1343683863086362,
            u'l_forearm_roll_joint': 7.517553513504836,
            u'l_shoulder_lift_joint': 0.5726770101613905,
            u'l_shoulder_pan_joint': 0.1592669164939349,
            u'l_upper_arm_roll_joint': 0.5532568387077381,
            u'l_wrist_flex_joint': - 1.215660155912625,
            u'l_wrist_roll_joint': 4.249300323527076,
            u'torso_lift_joint': 0.2}

        zero_pose.set_joint_goal(collision_pose)
        zero_pose.plan_and_execute()

        attached_link_name = u'pocky'
        zero_pose.attach_box(attached_link_name, [0.16, 0.04, 0.04], zero_pose.l_tip, [0.04, 0, 0], [0, 0, 0, 1])

        zero_pose.set_joint_goal({u'r_forearm_roll_joint': 0.0,
                                  u'r_shoulder_lift_joint': 0.0,
                                  u'r_shoulder_pan_joint': 0.0,
                                  u'r_upper_arm_roll_joint': 0.0,
                                  u'r_wrist_flex_joint': 0.0,
                                  u'r_wrist_roll_joint': 0.0,
                                  u'r_elbow_flex_joint': 0.0,
                                  u'torso_lift_joint': 0.2})

        p = PoseStamped()
        p.header.frame_id = zero_pose.l_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.z = 0.20
        p.pose.orientation.w = 1
        zero_pose.set_cart_goal(p, zero_pose.l_tip, zero_pose.default_root)
        zero_pose.plan_and_execute()

        zero_pose.check_cpi_geq(zero_pose.get_l_gripper_links(), 0.048)
        zero_pose.check_cpi_geq([attached_link_name], 0.048)
        zero_pose.detach_object(attached_link_name)

    def test_attached_self_collision2(self, zero_pose):
        """
        :type zero_pose: PR2
        """

        collision_pose = {
            u'r_elbow_flex_joint': - 1.1343683863086362,
            u'r_forearm_roll_joint': -7.517553513504836,
            u'r_shoulder_lift_joint': 0.5726770101613905,
            u'r_shoulder_pan_joint': -0.1592669164939349,
            u'r_upper_arm_roll_joint': -0.5532568387077381,
            u'r_wrist_flex_joint': - 1.215660155912625,
            u'r_wrist_roll_joint': -4.249300323527076,
            u'torso_lift_joint': 0.2
        }

        zero_pose.set_joint_goal(collision_pose)
        zero_pose.plan_and_execute()

        attached_link_name = u'box'
        zero_pose.attach_box(attached_link_name, [0.16, 0.04, 0.04], zero_pose.r_tip, [0.04, 0, 0], [0, 0, 0, 1])

        js_goal = {u'l_forearm_roll_joint': 0.0,
                   u'l_shoulder_lift_joint': 0.0,
                   u'odom_x_joint': 0.0,
                   u'odom_y_joint': 0.0,
                   u'odom_z_joint': 0.0,
                   u'l_shoulder_pan_joint': 0.0,
                   u'l_upper_arm_roll_joint': 0.0,
                   u'l_wrist_flex_joint': -0.11,
                   u'l_wrist_roll_joint': 0.0,
                   u'l_elbow_flex_joint': -0.16,
                   u'torso_lift_joint': 0.2}
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
        zero_pose.detach_object(attached_link_name)

    def test_attached_self_collision3(self, zero_pose):
        """
        :type zero_pose: PR2
        """

        collision_pose = {
            u'l_elbow_flex_joint': - 1.1343683863086362,
            u'l_forearm_roll_joint': 7.517553513504836,
            u'l_shoulder_lift_joint': 0.5726770101613905,
            u'l_shoulder_pan_joint': 0.1592669164939349,
            u'l_upper_arm_roll_joint': 0.5532568387077381,
            u'l_wrist_flex_joint': - 1.215660155912625,
            u'l_wrist_roll_joint': 4.249300323527076,
            u'torso_lift_joint': 0.2}

        zero_pose.set_joint_goal(collision_pose)
        zero_pose.plan_and_execute()

        attached_link_name = u'pocky'
        zero_pose.attach_box(attached_link_name, [0.1, 0.04, 0.04], zero_pose.l_tip, [0.02, 0, 0], [0, 0, 0, 1])

        js_goal = {u'r_forearm_roll_joint': 0.0,
                   u'r_shoulder_lift_joint': 0.0,
                   u'odom_x_joint': 0.0,
                   u'odom_y_joint': 0.0,
                   u'odom_z_joint': 0.0,
                   u'r_shoulder_pan_joint': 0.0,
                   u'r_upper_arm_roll_joint': 0.0,
                   u'r_wrist_flex_joint': -0.11,
                   u'r_wrist_roll_joint': 0.0,
                   u'r_elbow_flex_joint': -0.16,
                   u'torso_lift_joint': 0.2}

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
        zero_pose.detach_object(attached_link_name)

    def test_attached_collision_allow(self, box_setup):
        """
        :type box_setup: PR2
        """
        pocky = u'http://muh#pocky'
        box_setup.attach_box(pocky, [0.1, 0.02, 0.02], box_setup.r_tip, [0.05, 0, 0])

        ces = []
        ce = CollisionEntry()
        ce.type = CollisionEntry.ALLOW_COLLISION
        ce.robot_links = [pocky]
        ce.body_b = u'box'
        ces.append(ce)
        box_setup.set_collision_entries(ces)

        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.y = -0.11
        p.pose.orientation.w = 1
        box_setup.set_cart_goal(p, box_setup.r_tip, box_setup.default_root)
        box_setup.plan_and_execute()
        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.048)
        box_setup.check_cpi_leq([pocky], 0.0)

    def test_avoid_collision_gripper(self, box_setup):
        """
        :type box_setup: PR2
        """
        box_setup.allow_all_collisions()
        ces = box_setup.get_l_gripper_collision_entries(u'box', 0.05, CollisionEntry.AVOID_COLLISION)
        box_setup.set_collision_entries(ces)
        p = PoseStamped()
        p.header.frame_id = box_setup.l_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.15
        p.pose.orientation.w = 1
        box_setup.set_cart_goal(p, box_setup.l_tip, box_setup.default_root)
        box_setup.plan_and_execute()
        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), -1e-3)

    # def test_end_state_collision(self, box_setup):
    #     """
    #     :type box_setup: PR2
    #     """
    #     # TODO endstate impossible as long as we check for path collision?
    #     pass

    # def test_filled_vel_values(self, box_setup):
    #     """
    #     :type box_setup: PR2
    #     """
    #     pass
    #
    # def test_undefined_goal(self, box_setup):
    #     """
    #     :type box_setup: PR2
    #     """
    #     pass

    # TODO test plan only

    def test_attached_two_items(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        box1_name = u'box1'
        box2_name = u'box2'

        js = {
            u'r_elbow_flex_joint': -1.58118094489,
            u'r_forearm_roll_joint': -0.904933033043,
            u'r_shoulder_lift_joint': 0.822412440711,
            u'r_shoulder_pan_joint': -1.07866800992,
            u'r_upper_arm_roll_joint': -1.34905471854,
            u'r_wrist_flex_joint': -1.20182042644,
            u'r_wrist_roll_joint': 0.190433188769,
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
        zero_pose.set_cart_goal(r_goal, zero_pose.l_tip, u'torso_lift_link')
        zero_pose.plan_and_execute()

        zero_pose.attach_box(box1_name, [.2, .04, .04], zero_pose.r_tip, [.1, 0, 0], [0, 0, 0, 1])
        zero_pose.attach_box(box2_name, [.2, .04, .04], zero_pose.l_tip, [.1, 0, 0], [0, 0, 0, 1])

        zero_pose.plan_and_execute()

        zero_pose.check_cpi_geq([box1_name, box2_name], 0.049)

        zero_pose.detach_object(box1_name)
        zero_pose.detach_object(box2_name)
        base_goal = PoseStamped()
        base_goal.header.frame_id = u'base_footprint'
        base_goal.pose.position.x = -.1
        base_goal.pose.orientation.w = 1
        zero_pose.move_base(base_goal)

    # def test_pick_and_place(self, kitchen_setup):
    #     """
    #     :type kitchen_setup: PR2
    #     :return:
    #     """
    #
    #     base_pose = PoseStamped()
    #     base_pose.header.frame_id = u'map'
    #     base_pose.pose.position = Point(0.760, 0.480, 0.000)
    #     base_pose.pose.orientation = Quaternion(0.000, 0.000, 0.230, 0.973)
    #     kitchen_setup.move_pr2_base(base_pose)
    #     attached_link_name = u'edekabowl'
    #     p = PoseStamped()
    #     p.header.frame_id = u'map'
    #     p.pose.position = Point(1.39985, 0.799920, 0.888)
    #     p.pose.orientation = Quaternion(-0.0037, -0.00476, 0.3921, 0.9198)
    #     kitchen_setup.add_box(attached_link_name, [.145, .145, .072], pose=p)
    #
    #     pick_pose = PoseStamped()
    #     pick_pose.header.frame_id = u'base_footprint'
    #     pick_pose.pose.position = Point(0.649, -0.023, 0.918)
    #     pick_pose.pose.orientation = Quaternion(0.407, 0.574, -0.408, 0.582)
    #
    #     # pregrasp
    #     pick_pose.pose.position.z += 0.2
    #     kitchen_setup.set_and_check_cart_goal(pick_pose, kitchen_setup.l_tip, kitchen_setup.default_root)
    #
    #     # grasp
    #     pick_pose.pose.position.z -= 0.2
    #     kitchen_setup.avoid_collision(kitchen_setup.get_l_gripper_links(), u'kitchen', [], 0)
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
    #     place_pose.header.frame_id = u'base_footprint'
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

    # def test_hand_in_kitchen(self, kitchen_setup):
    #     """
    #     :type kitchen_setup: PR2
    #     :return:
    #     """
    #
    #     kitchen_setup.send_and_check_joint_goal(pick_up_pose)
    #
    #     base_pose = PoseStamped()
    #     base_pose.header.frame_id = u'map'
    #     base_pose.pose.position = Point(0.743, 0.586, 0.000)
    #     base_pose.pose.orientation.w = 1
    #     kitchen_setup.teleport_base(base_pose)
    #
    #     # grasp
    #     p = PoseStamped()
    #     p.header.frame_id = kitchen_setup.l_tip
    #     p.pose.position.x = 0.2
    #     p.pose.orientation.w = 1
    #     kitchen_setup.allow_collision(kitchen_setup.get_l_gripper_links(), u'kitchen',
    #                                           [u'sink_area', u'sink_area_surface'])
    #     kitchen_setup.set_and_check_cart_goal(p, kitchen_setup.l_tip, kitchen_setup.default_root)
    #
    #     # post grasp
    #     pregrasp_pose = PoseStamped()
    #     pregrasp_pose.header.frame_id = u'base_footprint'
    #     pregrasp_pose.pose.position.x = 0.611175722907
    #     pregrasp_pose.pose.position.y = -0.0244662287535
    #     pregrasp_pose.pose.position.z = 1.10803325995
    #     pregrasp_pose.pose.orientation.x = -0.0128682380997
    #     pregrasp_pose.pose.orientation.y = -0.710292569338
    #     pregrasp_pose.pose.orientation.z = 0.0148339707762
    #     pregrasp_pose.pose.orientation.w = -0.703632573456
    #     kitchen_setup.avoid_all_collisions(0.05)
    #     kitchen_setup.set_and_check_cart_goal(pregrasp_pose, kitchen_setup.l_tip, kitchen_setup.default_root)

    def test_set_kitchen_joint_state(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        kitchen_js = {u'sink_area_left_upper_drawer_main_joint': 0.45}
        kitchen_setup.set_kitchen_js(kitchen_js)

    def test_ease_fridge(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        milk_name = u'milk'

        # take milk out of fridge
        kitchen_setup.set_kitchen_js({u'iai_fridge_door_joint': 1.56})

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 0.565
        base_goal.pose.position.y = -0.5
        base_goal.pose.orientation.z = -0.51152562713
        base_goal.pose.orientation.w = 0.85926802151
        kitchen_setup.teleport_base(base_goal)
        # kitchen_setup.add_json_goal(u'BasePointingForward')

        # spawn milk
        milk_pose = PoseStamped()
        milk_pose.header.frame_id = u'iai_kitchen/iai_fridge_door_shelf1_bottom'
        milk_pose.pose.position = Point(0, 0, 0.12)
        milk_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        milk_pre_pose = PoseStamped()
        milk_pre_pose.header.frame_id = u'iai_kitchen/iai_fridge_door_shelf1_bottom'
        milk_pre_pose.pose.position = Point(0, 0, 0.22)
        milk_pre_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup.add_box(milk_name, [0.05, 0.05, 0.2], pose=milk_pose)

        # grasp milk
        kitchen_setup.open_l_gripper()

        # l_goal = deepcopy(milk_pose)
        # l_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[1, 0, 0, 0],
        #                                                               [0, 1, 0, 0],
        #                                                               [0, 0, 1, 0],
        #                                                               [0, 0, 0, 1]]))
        # kitchen_setup.set_cart_goal(l_goal, kitchen_setup.l_tip, kitchen_setup.default_root)
        # kitchen_setup.send_and_check_goal()

        # handle_name = u'map'
        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = u'map'
        bar_axis.vector.z = 1

        bar_center = PointStamped()
        bar_center.header.frame_id = milk_pose.header.frame_id
        bar_center.point = deepcopy(milk_pose.pose.position)

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = kitchen_setup.l_tip
        tip_grasp_axis.vector.z = 1

        kitchen_setup.set_json_goal(u'GraspBar',
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
        x_map.header.frame_id = u'iai_kitchen/iai_fridge_door'
        x_map.vector.x = 1
        # z = Vector3Stamped()
        # z.header.frame_id = 'milk'
        # z.vector.z = 1
        # z_map = Vector3Stamped()
        # z_map.header.frame_id = 'map'
        # z_map.vector.z = 1
        kitchen_setup.set_align_planes_goal(kitchen_setup.l_tip, x, root_normal=x_map)

        # kitchen_setup.allow_collision([], milk_name, [])
        # kitchen_setup.add_json_goal(u'AvoidJointLimits', percentage=15)
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()

        kitchen_setup.attach_object(milk_name, kitchen_setup.l_tip)
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
        kitchen_setup.set_cart_goal(milk_pre_pose, milk_name, kitchen_setup.default_root)
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.plan_and_execute()

        # place milk back

        # kitchen_setup.add_json_goal(u'BasePointingForward')
        # milk_goal = PoseStamped()
        # milk_goal.header.frame_id = u'iai_kitchen/kitchen_island_surface'
        # milk_goal.pose.position = Point(.1, -.2, .13)
        # milk_goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0,0,1]))
        kitchen_setup.set_cart_goal(milk_pre_pose, milk_name, kitchen_setup.default_root)
        kitchen_setup.plan_and_execute()

        kitchen_setup.set_cart_goal(milk_pose, milk_name, kitchen_setup.default_root)
        kitchen_setup.plan_and_execute()

        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        kitchen_setup.open_l_gripper()

        kitchen_setup.detach_object(milk_name)

        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.plan_and_execute()

    def test_ease_cereal(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        # FIXME
        cereal_name = u'cereal'
        drawer_frame_id = u'iai_kitchen/oven_area_area_right_drawer_board_2_link'

        # take milk out of fridge
        kitchen_setup.set_kitchen_js({u'oven_area_area_right_drawer_main_joint': 0.48})

        # spawn milk

        cereal_pose = PoseStamped()
        cereal_pose.header.frame_id = drawer_frame_id
        cereal_pose.pose.position = Point(0.123, -0.03, 0.11)
        cereal_pose.pose.orientation = Quaternion(0.0087786, 0.005395, -0.838767, -0.544393)
        kitchen_setup.add_box(cereal_name, [0.1528, 0.0634, 0.22894], pose=cereal_pose)

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

        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup.set_cart_goal(pre_grasp_pose, tip_link=kitchen_setup.r_tip)
        kitchen_setup.plan_and_execute()

        grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal, drawer_frame_id)

        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup.set_cart_goal(grasp_pose, tip_link=kitchen_setup.r_tip)
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
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.plan_and_execute()

        # place milk back

        # kitchen_setup.add_json_goal(u'BasePointingForward')
        # milk_goal = PoseStamped()
        # milk_goal.header.frame_id = u'iai_kitchen/kitchen_island_surface'
        # milk_goal.pose.position = Point(.1, -.2, .13)
        # milk_goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0,0,1]))

        kitchen_setup.set_cart_goal(cereal_pose, cereal_name)#, kitchen_setup.default_root)
        kitchen_setup.plan_and_execute()

        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        kitchen_setup.open_l_gripper()

        kitchen_setup.detach_object(cereal_name)

        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.plan_and_execute()

    def test_bowl_and_cup(self, kitchen_setup):
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

        # spawn cup
        cup_pose = PoseStamped()
        cup_pose.header.frame_id = u'iai_kitchen/sink_area_left_middle_drawer_main'
        cup_pose.pose.position = Point(0.1, 0.2, -.05)
        cup_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup.add_cylinder(cup_name, height=0.07, radius=0.04, pose=cup_pose)

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

        # grasp cup
        r_goal = deepcopy(cup_pose)
        r_goal.pose.position.z += .2
        r_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 1, 0, 0],
                                                                      [0, 0, -1, 0],
                                                                      [-1, 0, 0, 0],
                                                                      [0, 0, 0, 1]]))
        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=percentage)
        kitchen_setup.set_cart_goal(r_goal, kitchen_setup.r_tip, kitchen_setup.default_root)
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

    def test_ease_grasp_bowl(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        bowl_name = u'bowl'
        percentage = 40

        base_pose = PoseStamped()
        base_pose.header.frame_id = u'map'
        base_pose.pose.position = Point(0.314, 0.818, 0.000)
        base_pose.pose.orientation = Quaternion(-0.001, 0.000, 0.037, 0.999)
        kitchen_setup.teleport_base(base_pose)

        js = {
            u'torso_lift_joint': 0.262156255996,
            u'head_pan_joint': 0.0694220762479,
            u'head_tilt_joint': 1.01903547689,
            u'r_upper_arm_roll_joint': -1.5717499752,
            u'r_shoulder_pan_joint': -0.00156068057783,
            u'r_shoulder_lift_joint': 0.252786184819,
            u'r_forearm_roll_joint': -89.673490548,
            u'r_elbow_flex_joint': -0.544166310929,
            u'r_wrist_flex_joint': -1.32591140165,
            u'r_wrist_roll_joint': 65.7348048877,
            u'l_upper_arm_roll_joint': 1.38376171392,
            u'l_shoulder_pan_joint': 1.59536261129,
            u'l_shoulder_lift_joint': -0.0236488517104,
            u'l_forearm_roll_joint': 23.2795803857,
            u'l_elbow_flex_joint': -1.72694302293,
            u'l_wrist_flex_joint': -0.48001173639,
            u'l_wrist_roll_joint': -6.28312737965,
        }
        kitchen_setup.allow_all_collisions()
        kitchen_setup.set_joint_goal(js)
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_kitchen_js({u'sink_area_left_middle_drawer_main_joint': 0.45})

        r_goal = PoseStamped()
        r_goal.header.frame_id = kitchen_setup.r_tip
        r_goal.pose.position.x += 0.25
        r_goal.pose.orientation.w = 1

        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=percentage)
        kitchen_setup.set_cart_goal(r_goal, tip_link=kitchen_setup.r_tip)
        kitchen_setup.plan_and_execute()

        # spawn cup

    # def test_avoid_self_collision2(self, kitchen_setup):
    #     base_goal = PoseStamped()
    #     base_goal.header.frame_id = u'base_footprint'
    #     base_goal.pose.position.x = -.1
    #     base_goal.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
    #     kitchen_setup.teleport_base(base_goal)
    #
    #     # place bowl and cup
    #     bowl_goal = PoseStamped()
    #     bowl_goal.header.frame_id = u'iai_kitchen/kitchen_island_surface'
    #     bowl_goal.pose.position = Point(.2, 0, .05)
    #     bowl_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 0, -1, 0],
    #                                                                      [0, -1, 0, 0],
    #                                                                      [-1, 0, 0, 0],
    #                                                                      [0, 0, 0, 1]]))
    #
    #     cup_goal = PoseStamped()
    #     cup_goal.header.frame_id = u'iai_kitchen/kitchen_island_surface'
    #     cup_goal.pose.position = Point(.15, 0.25, .07)
    #     cup_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 0, -1, 0],
    #                                                                     [0, -1, 0, 0],
    #                                                                     [-1, 0, 0, 0],
    #                                                                     [0, 0, 0, 1]]))
    #
    #     kitchen_setup.set_cart_goal(bowl_goal, kitchen_setup.l_tip, kitchen_setup.default_root)
    #     kitchen_setup.set_cart_goal(cup_goal, kitchen_setup.r_tip, kitchen_setup.default_root)
    #     kitchen_setup.send_and_check_goal()

    def test_ease_spoon(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        spoon_name = u'spoon'
        percentage = 40

        # spawn cup
        cup_pose = PoseStamped()
        cup_pose.header.frame_id = u'iai_kitchen/sink_area_surface'
        cup_pose.pose.position = Point(0.1, -.5, .02)
        cup_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup.add_box(spoon_name, [0.1, 0.02, 0.01], pose=cup_pose)

        # kitchen_setup.send_and_check_joint_goal(gaya_pose)

        # grasp spoon
        l_goal = deepcopy(cup_pose)
        l_goal.pose.position.z += .2
        l_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 0, -1, 0],
                                                                      [0, -1, 0, 0],
                                                                      [-1, 0, 0, 0],
                                                                      [0, 0, 0, 1]]))
        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=percentage)
        kitchen_setup.set_cart_goal(l_goal, kitchen_setup.l_tip, kitchen_setup.default_root)
        kitchen_setup.plan_and_execute()

        l_goal.pose.position.z -= .2
        # kitchen_setup.allow_collision([CollisionEntry.ALL], spoon_name, [CollisionEntry.ALL])
        kitchen_setup.set_cart_goal(l_goal, kitchen_setup.l_tip, kitchen_setup.default_root)
        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=percentage)
        kitchen_setup.plan_and_execute()
        kitchen_setup.attach_object(spoon_name, kitchen_setup.l_tip)

        l_goal.pose.position.z += .2
        # kitchen_setup.allow_collision([CollisionEntry.ALL], spoon_name, [CollisionEntry.ALL])
        kitchen_setup.set_cart_goal(l_goal, kitchen_setup.l_tip, kitchen_setup.default_root)
        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=percentage)
        kitchen_setup.plan_and_execute()

        l_goal.pose.position.z -= .2
        # kitchen_setup.allow_collision([CollisionEntry.ALL], spoon_name, [CollisionEntry.ALL])
        kitchen_setup.set_cart_goal(l_goal, kitchen_setup.l_tip, kitchen_setup.default_root)
        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=percentage)
        kitchen_setup.plan_and_execute()

        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.plan_and_execute()

    def test_ease_place_on_new_table(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        percentage = 40
        js = {
            u'torso_lift_joint': 0.262343532164,
            u'head_pan_joint': 0.0308852063639,
            u'head_tilt_joint': 0.710418818732,
            u'r_upper_arm_roll_joint': -1.4635104674,
            u'r_shoulder_pan_joint': -1.59535749265,
            u'r_shoulder_lift_joint': -0.0235854289628,
            u'r_forearm_roll_joint': -123.897562601,
            u'r_elbow_flex_joint': -1.72694302293,
            u'r_wrist_flex_joint': -0.480010977079,
            u'r_wrist_roll_joint': 88.0157228707,
            u'l_upper_arm_roll_joint': 1.90635809306,
            u'l_shoulder_pan_joint': 0.352841136964,
            u'l_shoulder_lift_joint': -0.35035444474,
            u'l_forearm_roll_joint': 32.5396842176,
            u'l_elbow_flex_joint': -0.543731998795,
            u'l_wrist_flex_joint': -1.68825444756,
            u'l_wrist_roll_joint': -12.6846818117,
        }
        kitchen_setup.set_joint_goal(js)
        kitchen_setup.plan_and_execute()
        base_pose = PoseStamped()
        base_pose.header.frame_id = u'map'
        base_pose.pose.position = Point(-2.8, 0.188, -0.000)  # -2.695
        base_pose.pose.orientation = Quaternion(-0.001, -0.001, 0.993, -0.114)
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.teleport_base(base_pose)

        object_name = u'box'
        kitchen_setup.attach_box(name=object_name,
                                 size=[0.10, 0.14, 0.14],
                                 frame_id=kitchen_setup.l_tip,
                                 position=[0.0175, 0.025, 0],
                                 orientation=[0, 0, 0, 1])

        l_goal = PoseStamped()
        l_goal.header.stamp = rospy.get_rostime()
        l_goal.header.frame_id = kitchen_setup.l_tip
        l_goal.pose.position.x += 0.2
        # l_goal.pose.position.z -= 0.1
        l_goal.pose.orientation.w = 1
        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=percentage)

        js = {
            u'r_upper_arm_roll_joint': -1.4635104674,
            u'r_shoulder_pan_joint': -1.59535749265,
            u'r_shoulder_lift_joint': -0.0235854289628,
            u'r_forearm_roll_joint': -123.897562601,
            u'r_elbow_flex_joint': -1.72694302293,
            u'r_wrist_flex_joint': -0.480010977079,
            u'r_wrist_roll_joint': 88.0157228707,
        }
        kitchen_setup.set_joint_goal(js)

        # base_pose.header.frame_id = u'base_footprint'
        # base_pose.pose.position = Point(0,0,0)
        # base_pose.pose.orientation = Quaternion(0,0,0,1)
        # kitchen_setup.set_cart_goal(base_pose, u'base_footprint')

        kitchen_setup.set_cart_goal(l_goal, tip_link=kitchen_setup.l_tip)
        kitchen_setup.plan_and_execute()

    def test_tray(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        # FIXME
        tray_name = u'tray'
        percentage = 50

        tray_pose = PoseStamped()
        tray_pose.header.frame_id = u'iai_kitchen/sink_area_surface'
        tray_pose.pose.position = Point(0.1, -0.4, 0.07)
        tray_pose.pose.orientation.w = 1

        kitchen_setup.add_box(tray_name, [.2, .4, .1], pose=tray_pose)

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
        kitchen_setup.allow_collision([], tray_name, [])
        kitchen_setup.avoid_joint_limits(percentage=percentage)
        # grasp tray
        kitchen_setup.plan_and_execute()

        kitchen_setup.attach_object(tray_name, kitchen_setup.r_tip)

        r_goal = PoseStamped()
        r_goal.header.frame_id = kitchen_setup.l_tip
        r_goal.pose.orientation.w = 1
        kitchen_setup.set_cart_goal(r_goal, kitchen_setup.l_tip, tray_name)

        tray_goal = tf.lookup_pose(u'base_footprint', tray_name)
        tray_goal.pose.position.y = 0
        tray_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[-1, 0, 0, 0],
                                                                         [0, -1, 0, 0],
                                                                         [0, 0, 1, 0],
                                                                         [0, 0, 0, 1]]))
        kitchen_setup.set_cart_goal(tray_goal, tray_name, u'base_footprint')

        base_goal = PoseStamped()
        base_goal.header.frame_id = u'map'
        base_goal.pose.position.x -= 0.5
        base_goal.pose.position.y -= 0.3
        base_goal.pose.orientation.w = 1
        kitchen_setup.avoid_joint_limits(percentage=percentage)
        kitchen_setup.allow_collision(robot_links=[tray_name],
                                      body_b=kitchen_setup.robot.name,
                                      link_bs=kitchen_setup.get_l_gripper_links())
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
        kitchen_setup.allow_collision(robot_links=[tray_name],
                                      body_b=kitchen_setup.robot.name,
                                      link_bs=kitchen_setup.get_l_gripper_links())
        kitchen_setup.set_cart_goal(tray_goal, tray_name, u'base_footprint')
        kitchen_setup.plan_and_execute()

    # TODO FIXME attaching and detach of urdf objects that listen to joint states

    def test_iis(self, kitchen_setup):
        """
        :TYPE kitchen_setup: PR2
        """
        # rosrun tf static_transform_publisher 0 - 0.2 0.93 1.5707963267948966 0 0 iai_kitchen/table_area_main lid 10
        # rosrun tf static_transform_publisher 0 - 0.15 0 0 0 0 lid goal 10
        # kitchen_setup.set_joint_goal(pocky_pose)
        # kitchen_setup.send_and_check_goal()
        object_name = u'lid'
        pot_pose = PoseStamped()
        pot_pose.header.frame_id = u'lid'
        pot_pose.pose.position.z = -0.22
        # pot_pose.pose.orientation.w = 1
        pot_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 2, [0, 0, 1]))
        kitchen_setup.add_mesh(object_name, path=u'package://cad_models/kitchen/cooking-vessels/cookingpot.dae',
                               pose=pot_pose)

        base_pose = PoseStamped()
        base_pose.header.frame_id = u'iai_kitchen/table_area_main'
        base_pose.pose.position.y = -1.1
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 2, [0, 0, 1]))
        kitchen_setup.teleport_base(base_pose)
        # m = zero_pose.world.get_object(object_name).as_marker_msg()
        # compare_poses(m.pose, p.pose)

        hand_goal = PoseStamped()
        hand_goal.header.frame_id = u'lid'
        hand_goal.pose.position.y = -0.15
        hand_goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 2, [0, 0, 1]))
        # kitchen_setup.allow_all_collisions()
        # kitchen_setup.avoid_collision([], 'kitchen', ['table_area_main'], 0.05)
        kitchen_setup.set_cart_goal(hand_goal, u'r_gripper_tool_frame')
        kitchen_setup.send_goal(goal_type=MoveGoal.PLAN_ONLY)
        kitchen_setup.set_cart_goal(hand_goal, u'r_gripper_tool_frame')
        kitchen_setup.send_goal()

        hand_goal = PoseStamped()
        hand_goal.header.frame_id = u'r_gripper_tool_frame'
        hand_goal.pose.position.x = 0.15
        hand_goal.pose.orientation.w = 1
        # kitchen_setup.allow_all_collisions()
        # kitchen_setup.avoid_collision([], 'kitchen', ['table_area_main'], 0.05)
        kitchen_setup.set_cart_goal(hand_goal, u'r_gripper_tool_frame')
        kitchen_setup.allow_all_collisions()
        kitchen_setup.send_goal(goal_type=MoveGoal.PLAN_ONLY)
        kitchen_setup.set_cart_goal(hand_goal, u'r_gripper_tool_frame')
        kitchen_setup.allow_all_collisions()
        kitchen_setup.send_goal()

        # kitchen_setup.add_cylinder('pot', size=[0.2,0.2], pose=pot_pose)

    def test_ease_dishwasher(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        # FIXME
        p = PoseStamped()
        p.header.frame_id = u'map'
        p.pose.orientation.w = 1
        p.pose.position.x = 0.5
        p.pose.position.y = 0.2
        kitchen_setup.teleport_base(p)

        hand = kitchen_setup.r_tip

        goal_angle = np.pi / 4
        handle_frame_id = u'iai_kitchen/sink_area_dish_washer_door_handle'
        handle_name = u'sink_area_dish_washer_door_handle'
        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = handle_frame_id
        bar_axis.vector.y = 1

        bar_center = PointStamped()
        bar_center.header.frame_id = handle_frame_id

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = hand
        tip_grasp_axis.vector.z = 1

        kitchen_setup.set_json_goal(u'GraspBar',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=hand,
                                    tip_grasp_axis=tip_grasp_axis,
                                    bar_center=bar_center,
                                    bar_axis=bar_axis,
                                    bar_length=.3)
        # kitchen_setup.allow_collision([], u'kitchen', [handle_name])
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

        kitchen_setup.set_json_goal(u'Open',
                                    tip_link=hand,
                                    object_name=u'kitchen',
                                    object_link_name=handle_name,
                                    goal_joint_state=goal_angle,
                                    # weight=100
                                    )
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_kitchen_js({u'sink_area_dish_washer_door_joint': goal_angle})
        # ----------------------------------------------------------------------------------------
        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.plan_and_execute()

        tray_handle_frame_id = u'iai_kitchen/sink_area_dish_washer_tray_handle_front_side'
        tray_handle_name = u'sink_area_dish_washer_tray_handle_front_side'
        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = tray_handle_frame_id
        bar_axis.vector.y = 1
        bar_axis.vector.z = -0.1

        bar_center = PointStamped()
        bar_center.header.frame_id = tray_handle_frame_id

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = hand
        tip_grasp_axis.vector.z = 1

        kitchen_setup.set_json_goal(u'GraspBar',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=hand,
                                    tip_grasp_axis=tip_grasp_axis,
                                    bar_center=bar_center,
                                    bar_axis=bar_axis,
                                    bar_length=.3)
        # kitchen_setup.allow_collision([], u'kitchen', [handle_name])
        kitchen_setup.plan_and_execute()

        p = tf.lookup_pose(tray_handle_frame_id, hand)
        p.pose.position.x += 0.3

        # p = tf.transform_pose(hand, p)

        # kitchen_setup.add_json_goal(u'CartesianPosition',
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
        # kitchen_setup.add_json_goal(u'Close',
        #                             tip_link=hand,
        #                             object_name=u'kitchen',
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
        kitchen_setup.set_cart_goal(grasp_pose,tip_link=kitchen_setup.r_tip)
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

        kitchen_setup.god_map.set_data(identifier.rosparam + ['reset_god_map'], False)
        decrease_external_collision_avoidance(kitchen_setup)
        kitchen_setup.set_json_goal(u'CartesianPathCarrot',
                                    tip_link=kitchen_setup.r_tip,
                                    root_link=kitchen_setup.default_root,
                                    goal=post_grasp_pose,
                                    goal_sampling_axis=[True, False, False])
        kitchen_setup.plan_and_execute()
        kitchen_setup.god_map.set_data(identifier.rosparam + ['reset_god_map'], True)

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

        #base_goal = PoseStamped()
        #base_goal.header.frame_id = u'base_footprint'
        #base_goal.pose.position.y = -.6
        #base_goal.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        #kitchen_setup.teleport_base(base_goal)
        #kitchen_setup.wait_heartbeats(10)

        # place cup
        #cup_goal = PoseStamped()
        #cup_goal.header.frame_id = u'iai_kitchen/kitchen_island_surface'
        #cup_goal.pose.position = Point(.15, 0.25, .07)
        #cup_goal.pose.orientation = Quaternion(0, 0, 0, 1)

        #kitchen_setup.set_cart_goal(cup_goal, cup_name, kitchen_setup.default_root)
        #kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=percentage)
        #kitchen_setup.plan_and_execute()

        #kitchen_setup.detach_object(cup_name)
        #kitchen_setup.allow_collision([], cup_name, [])
        #kitchen_setup.set_joint_goal(gaya_pose)
        #kitchen_setup.plan_and_execute()

    def test_grasp_milk(self, kitchen_setup_avoid_collisions):
        tip_link = kitchen_setup_avoid_collisions.l_tip
        milk_name = u'milk'

        # take milk out of fridge
        kitchen_setup_avoid_collisions.set_kitchen_js({u'iai_fridge_door_joint': 1.56})

        # spawn milk
        milk_pose = PoseStamped()
        milk_pose.header.frame_id = u'iai_kitchen/iai_fridge_door_shelf1_bottom'
        milk_pose.pose.position = Point(0, 0, 0.13)
        milk_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup_avoid_collisions.add_box(milk_name, [0.05, 0.05, 0.2], milk_pose)

        point = tf.np_to_pose(kitchen_setup_avoid_collisions.world.get_fk('map', milk_name)).position
        point_s = PointStamped()
        point_s.header.frame_id = 'map'
        point_s.point = point
        self.look_at(kitchen_setup_avoid_collisions, goal_point=point_s)
        kitchen_setup.plan_and_execute()

        # move arm towards milk
        if tip_link == kitchen_setup_avoid_collisions.l_tip:
            kitchen_setup_avoid_collisions.open_l_gripper()
        elif tip_link == kitchen_setup_avoid_collisions.r_tip:
            kitchen_setup_avoid_collisions.open_r_gripper()
        else:
            raise Exception('Wrong tip link')

        kitchen_setup_avoid_collisions.set_json_goal('CartesianPreGrasp',
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     tip_link=tip_link,
                                                     grasping_goal=milk_pose)
        kitchen_setup_avoid_collisions.plan_and_execute()

        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPose',
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     tip_link=tip_link,
                                                     goal=milk_pose)
        kitchen_setup_avoid_collisions.plan_and_execute()

        # grasp milk
        kitchen_setup_avoid_collisions.attach_object(milk_name, tip_link)

        if tip_link == kitchen_setup_avoid_collisions.l_tip:
            kitchen_setup_avoid_collisions.close_l_gripper()
        elif tip_link == kitchen_setup_avoid_collisions.r_tip:
            kitchen_setup_avoid_collisions.close_r_gripper()
        else:
            raise Exception('Wrong tip link')

        kitchen_setup_avoid_collisions.set_json_goal('CartesianPreGrasp',
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     tip_link=tip_link,
                                                     grasping_goal=milk_pose,
                                                     dist=0.1)
        kitchen_setup_avoid_collisions.plan_and_execute()

        kitchen_setup_avoid_collisions.set_joint_goal(gaya_pose)
        kitchen_setup_avoid_collisions.plan_and_execute()

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
            #u'sink_area_left_upper_drawer_handle',
            # u'sink_area_left_middle_drawer_handle',
            # u'sink_area_left_bottom_drawer_handle',
            #u'sink_area_trash_drawer_handle',
            #u'fridge_area_lower_drawer_handle',
            # u'kitchen_island_left_upper_drawer_handle',
            # u'kitchen_island_left_lower_drawer_handle',
            # u'kitchen_island_middle_upper_drawer_handle',
            # u'kitchen_island_middle_lower_drawer_handle',
            # u'kitchen_island_right_upper_drawer_handle',
            #u'kitchen_island_right_lower_drawer_handle',
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
            #u'sink_area_left_upper_drawer_handle',
            # u'sink_area_left_middle_drawer_handle',
            # u'sink_area_left_bottom_drawer_handle',
            #u'sink_area_trash_drawer_handle',
            #u'fridge_area_lower_drawer_handle',
            # u'kitchen_island_left_upper_drawer_handle',
            # u'kitchen_island_left_lower_drawer_handle',
            # u'kitchen_island_middle_upper_drawer_handle',
            # u'kitchen_island_middle_lower_drawer_handle',
            # u'kitchen_island_right_upper_drawer_handle',
            #u'kitchen_island_right_lower_drawer_handle',
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
        table_navigation_goal.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, np.pi/2.))

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
        #kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=percentage)
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

class TestReachability():
    def test_unreachable_goal_0(self, zero_pose):
        js = {}
        js['r_shoulder_lift_joint'] = 10
        zero_pose.set_joint_goal(js)
        zero_pose.check_reachability(expected_error_codes=[MoveResult.UNREACHABLE])

    def test_unreachable_goal_1(self, zero_pose):
        pose = PoseStamped()
        pose.header.frame_id = zero_pose.r_tip
        pose.pose.position.z = -2
        pose.pose.orientation.w = 1
        zero_pose.set_cart_goal(root_link=zero_pose.default_root, tip_link=zero_pose.r_tip, goal_pose=pose)
        zero_pose.check_reachability(expected_error_codes=[MoveResult.UNREACHABLE])

    def test_unreachable_goal_2(self, zero_pose):
        pose = PoseStamped()
        pose.pose.position.z = 5
        pose.header.frame_id = 'map'
        pose.pose.orientation.w = 1
        zero_pose.set_cart_goal(root_link='odom_combined', tip_link='r_gripper_tool_frame', goal_pose=pose)
        zero_pose.check_reachability(expected_error_codes=[MoveResult.SHAKING])

    def test_unreachable_goal_3(self, zero_pose):
        pose = PoseStamped()
        pose.pose.position.z = 1.2
        pose.header.frame_id = 'map'
        pose.pose.orientation.w = 1
        zero_pose.set_translation_goal(goal_pose=pose, tip_link='r_gripper_tool_frame', root_link='odom_combined')
        js = {}
        js['r_shoulder_lift_joint'] = 1.0  # soft lower -0.3536 soft upper 1.2963
        js['r_elbow_flex_joint'] = -0.2  # soft lower -2.1213 soft upper -0.15
        js['torso_lift_joint'] = 0.15  # soft lower 0.0115 soft upper 0.325
        zero_pose.set_joint_goal(js)
        zero_pose.check_reachability(expected_error_codes=[MoveResult.UNREACHABLE])

    def test_unreachable_goal_4(self, zero_pose):
        pose = PoseStamped()
        pose.pose.position.y = -2.0
        pose.pose.position.z = 1
        pose.header.frame_id = 'map'
        pose.pose.orientation.w = 1
        zero_pose.set_translation_goal(goal_pose=pose, tip_link='r_gripper_tool_frame', root_link='odom_combined')
        pose2 = PoseStamped()
        pose2.pose.position.y = 2.0
        pose2.pose.position.z = 1
        pose2.header.frame_id = 'map'
        pose.pose.orientation.w = 1
        zero_pose.set_translation_goal(goal_pose=pose2, tip_link='l_gripper_tool_frame', root_link='odom_combined')
        zero_pose.check_reachability(expected_error_codes=[MoveResult.UNREACHABLE])

    def test_unreachable_goal_5(self, zero_pose):
        pose = PoseStamped()
        pose.pose.position.x = 2
        pose.header.frame_id = 'map'
        pose.pose.orientation.w = 1
        zero_pose.set_cart_goal(root_link='r_shoulder_lift_link', tip_link='r_gripper_tool_frame', goal_pose=pose)
        zero_pose.check_reachability(expected_error_codes=[MoveResult.UNREACHABLE])

    def test_unreachable_goal_6(self, zero_pose):  # TODO torso lift joint xdot has a wrong value
        pose = PoseStamped()
        pose.pose.position.x = 0.0
        pose.pose.position.y = 0.0
        pose.pose.position.z = 0.0
        pose.header.frame_id = 'map'
        pose.pose.orientation.w = 1
        zero_pose.set_translation_goal(goal_pose=pose, tip_link='base_footprint', root_link='odom_combined')
        js = {}
        js['odom_x_joint'] = 0.01
        zero_pose.set_joint_goal(js)
        zero_pose.check_reachability(expected_error_codes=[MoveResult.UNREACHABLE])

    def test_unreachable_goal_7(self, zero_pose):
        js = {}
        js['r_shoulder_lift_joint'] = 0.2
        js['r_elbow_flex_joint'] = -2.5
        js['torso_lift_joint'] = 0.15
        zero_pose.set_joint_goal(js)
        zero_pose.check_reachability(expected_error_codes=[MoveResult.UNREACHABLE])

    def test_reachable_goal_0(self, zero_pose):
        pose = PoseStamped()
        pose.pose.position.x = 3.0
        pose.pose.position.z = 1.0
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.707
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 0.707
        pose.header.frame_id = 'map'
        zero_pose.set_cart_goal(root_link='odom_combined', tip_link='r_gripper_tool_frame', goal_pose=pose)
        zero_pose.check_reachability()
        zero_pose.set_cart_goal(root_link='odom_combined', tip_link='r_gripper_tool_frame', goal_pose=pose)
        zero_pose.allow_all_collisions()
        zero_pose.send_and_check_goal(goal_type=MoveGoal.PLAN_ONLY)

    def test_reachable_goal_1(self, zero_pose):
        pose = PoseStamped()
        pose.pose.orientation.x = 0
        pose.pose.orientation.y = 0
        pose.pose.orientation.z = 0
        pose.pose.orientation.w = 1
        pose.pose.position.x = 3.0
        pose.pose.position.z = 1.0
        pose.header.frame_id = 'map'
        zero_pose.set_cart_goal(root_link='odom_combined', tip_link='r_gripper_tool_frame', goal_pose=pose)
        zero_pose.check_reachability()
        zero_pose.set_cart_goal(root_link='odom_combined', tip_link='r_gripper_tool_frame', goal_pose=pose)
        zero_pose.send_and_check_goal(goal_type=MoveGoal.PLAN_ONLY)

    def test_reachable_goal_2(self, zero_pose):
        pose = PoseStamped()
        pose.pose.position.x = -1.0
        pose.pose.position.z = 1.0
        pose.header.frame_id = 'map'
        zero_pose.set_cart_goal(root_link='odom_combined', tip_link='r_gripper_tool_frame', goal_pose=pose)
        zero_pose.check_reachability()
        zero_pose.set_cart_goal(root_link='odom_combined', tip_link='r_gripper_tool_frame', goal_pose=pose)
        zero_pose.send_and_check_goal(goal_type=MoveGoal.PLAN_ONLY)

    def test_reachable_goal_3(self, zero_pose):
        pose = PoseStamped()
        pose.pose.position.x = 1.0
        pose.pose.position.y = 0.3
        pose.pose.position.z = 1.0
        pose.pose.orientation.x = -0.697
        pose.pose.orientation.y = -0.557
        pose.pose.orientation.z = 0.322
        pose.pose.orientation.w = -0.317
        pose.header.frame_id = 'map'
        zero_pose.set_cart_goal(root_link='odom_combined', tip_link='r_gripper_tool_frame', goal_pose=pose)
        zero_pose.check_reachability()
        zero_pose.set_cart_goal(root_link='odom_combined', tip_link='r_gripper_tool_frame', goal_pose=pose)
        zero_pose.send_and_check_goal(goal_type=MoveGoal.PLAN_ONLY)

    def test_reachable_goal_4(self, zero_pose):
        pose = PoseStamped()
        pose.pose.position.x = 1.0
        pose.pose.position.y = 0.3
        pose.pose.position.z = 1.0
        pose.pose.orientation.x = -0.697
        pose.pose.orientation.y = -0.557
        pose.pose.orientation.z = 0.322
        pose.pose.orientation.w = -0.317
        pose.header.frame_id = 'map'
        zero_pose.set_cart_goal(root_link='odom_combined', tip_link='r_gripper_tool_frame', goal_pose=pose)
        js = {}
        js['r_elbow_flex_joint'] = -0.2
        js['torso_lift_joint'] = 0.15
        zero_pose.set_joint_goal(js)
        zero_pose.allow_all_collisions()
        zero_pose.check_reachability()

        zero_pose.set_cart_goal(root_link='odom_combined', tip_link='r_gripper_tool_frame', goal_pose=pose)
        zero_pose.set_joint_goal(js)
        zero_pose.allow_all_collisions()
        zero_pose.send_and_check_goal(goal_type=MoveGoal.PLAN_ONLY)

    def test_reachable_goal_5(self, zero_pose):
        js = {}
        js['r_shoulder_lift_joint'] = 0.2
        js['r_elbow_flex_joint'] = -0.2
        js['torso_lift_joint'] = 0.15
        zero_pose.set_joint_goal(js)
        zero_pose.check_reachability()
        zero_pose.set_joint_goal(js)
        zero_pose.send_and_check_goal(goal_type=MoveGoal.PLAN_ONLY)


class TestConfigFile(object):
    def test_prediction_horizon1(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
        zero_pose.set_joint_goal(gaya_pose)
        zero_pose.plan_and_execute()
        zero_pose.set_joint_goal(default_pose)
        zero_pose.plan_and_execute()

    def test_bowl_and_cup_prediction_horizon1(self, kitchen_setup):
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

        kitchen_setup.set_json_goal('SetPredictionHorizon', prediction_horizon=1)

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
        kitchen_setup.set_json_goal('SetPredictionHorizon', prediction_horizon=1)

        # open drawer
        kitchen_setup.set_json_goal(u'Open',
                                    tip_link=kitchen_setup.l_tip,
                                    environment_link=drawer_handle)
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
        kitchen_setup.set_kitchen_js({drawer_joint: 0.48})

        # spawn cup
        cup_pose = PoseStamped()
        cup_pose.header.frame_id = u'iai_kitchen/sink_area_left_middle_drawer_main'
        cup_pose.pose.position = Point(0.1, 0.2, -.05)
        cup_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup.add_cylinder(cup_name, height=0.07, radius=0.04, pose=cup_pose)

        # spawn bowl
        bowl_pose = PoseStamped()
        bowl_pose.header.frame_id = u'iai_kitchen/sink_area_left_middle_drawer_main'
        bowl_pose.pose.position = Point(0.1, -0.2, -.05)
        bowl_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup.add_cylinder(bowl_name, height=0.05, radius=0.07, pose=bowl_pose)
        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_json_goal('SetPredictionHorizon', prediction_horizon=1)

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
        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=percentage)
        kitchen_setup.set_cart_goal(r_goal, kitchen_setup.r_tip, kitchen_setup.default_root)
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_json_goal('SetPredictionHorizon', prediction_horizon=1)

        l_goal.pose.position.z -= .2
        r_goal.pose.position.z -= .2
        kitchen_setup.allow_collision([CollisionEntry.ALL], bowl_name, [CollisionEntry.ALL])
        kitchen_setup.allow_collision([CollisionEntry.ALL], cup_name, [CollisionEntry.ALL])
        kitchen_setup.set_cart_goal(l_goal, kitchen_setup.l_tip, kitchen_setup.default_root)
        kitchen_setup.set_cart_goal(r_goal, kitchen_setup.r_tip, kitchen_setup.default_root)
        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=percentage)
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_json_goal('SetPredictionHorizon', prediction_horizon=1)

        kitchen_setup.attach_object(bowl_name, kitchen_setup.l_tip)
        kitchen_setup.attach_object(cup_name, kitchen_setup.r_tip)

        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
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
        kitchen_setup.set_json_goal('SetPredictionHorizon', prediction_horizon=1)

        kitchen_setup.detach_object(bowl_name)
        kitchen_setup.detach_object(cup_name)
        kitchen_setup.allow_collision([], cup_name, [])
        kitchen_setup.allow_collision([], bowl_name, [])
        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.plan_and_execute()

# import pytest
# pytest.main(['-s', __file__ + '::TestJointGoals::test_joint_movement1'])
# pytest.main(['-s', __file__ + '::TestCollisionAvoidanceGoals::test_bowl_and_cup'])
# pytest.main(['-s', __file__ + '::TestCollisionAvoidanceGoals::test_attached_collision2'])
# pytest.main(['-s', __file__ + '::TestCollisionAvoidanceGoals::test_avoid_self_collision'])
# pytest.main(['-s', __file__ + '::TestCartesianPath::test_ease_fridge_with_cart_goals_and_global_planner'])
