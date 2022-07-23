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

    #@pytest.mark.repeat(3)
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

    @pytest.mark.repeat(20)
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

        #kitchen_setup_avoid_collisions.set_json_goal(u'SetPredictionHorizon', prediction_horizon=1)
        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPathCarrot',
                                                     tip_link=tip_link,
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     goal=goal_c)

        kitchen_setup_avoid_collisions.plan_and_execute()
        #kitchen_setup_avoid_collisions.send_goal()
        #kitchen_setup_avoid_collisions.check_cart_goal(tip_link, goal_c)
        #zero_pose.check_cart_goal(zero_pose.l_tip, l_goal)

    @pytest.mark.repeat(5)
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

    @pytest.mark.repeat(5)
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

        try:
            kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPose',
                                                         tip_link=tip_link,
                                                         root_link=kitchen_setup_avoid_collisions.default_root,
                                                         goal=goal_c
                                                         )
            kitchen_setup_avoid_collisions.plan_and_execute()
        except Exception:
            pass
        #kitchen_setup_avoid_collisions.send_goal()
        #kitchen_setup_avoid_collisions.check_cart_goal(tip_link, goal_c)
        #zero_pose.check_cart_goal(zero_pose.l_tip, l_goal)

    @pytest.mark.repeat(5)
    def test_pathAroundKitchenIsland_with_open_stuff(self, kitchen_setup_avoid_collisions):
        # kernprof -lv py.test -s test/test_integration_pr2.py::TestCartGoals::test_pathAroundKitchenIsland_with_global_planner
        """
        :type kitchen_setup_avoid_collisions: PR2
        """
        tip_link = u'base_footprint'
        kitchen_setup_avoid_collisions.set_kitchen_js({'kitchen_island_left_upper_drawer_main_joint': 0.48})
        kitchen_setup_avoid_collisions.set_kitchen_js({u'oven_area_area_right_drawer_main_joint': 0.28})

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = 0.3
        base_pose.pose.position.y = 2.1
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi/2, [0, 0, 1]))
        kitchen_setup_avoid_collisions.teleport_base(base_pose)

        base_pose = PoseStamped()
        base_pose.header.frame_id = 'map'
        base_pose.pose.position.x = -2.5
        base_pose.pose.position.y = 1
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0, 0, 1]))
        goal_c = base_pose

        try:
            kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPose',
                                                         tip_link=tip_link,
                                                         root_link=kitchen_setup_avoid_collisions.default_root,
                                                         goal=goal_c)
            kitchen_setup_avoid_collisions.plan_and_execute()
        except Exception:
            pass
        #kitchen_setup_avoid_collisions.send_goal()
        #kitchen_setup_avoid_collisions.check_cart_goal(tip_link, goal_c)
        #zero_pose.check_cart_goal(zero_pose.l_tip, l_goal)

    #@pytest.mark.repeat(5)
    def test_pathAroundKitchenIsland_with_open_stuff_and_box(self, kitchen_setup_avoid_collisions):
        tj_a = [[-0.        ,  2.1       ,  0.392705  ], # cost 12.11348
       [-0.0802661 ,  1.96285   ,  0.310531  ],
       [-0.232431  ,  1.92845   ,  0.39852   ],
       [-0.266267  ,  1.74423   ,  0.423928  ],
       [-0.297121  ,  1.59974   ,  0.319424  ],
       [-0.243799  ,  1.61411   ,  0.0298742 ],
       [-0.218986  ,  1.6151    , -0.320462  ],
       [-0.0769564 ,  1.56538   , -0.419501  ],
       [ 0.041516  ,  1.53989   , -0.577133  ],
       [ 0.135408  ,  1.41418   , -0.663335  ],
       [ 0.132241  ,  1.2957    , -0.500389  ],
       [ 0.246383  ,  1.16907   , -0.441349  ],
       [ 0.192859  ,  1.06091   , -0.600004  ],
       [ 0.194196  ,  0.916292  , -0.48926   ],
       [ 0.254225  ,  0.731022  , -0.478763  ],
       [ 0.126667  ,  0.595153  , -0.45149   ],
       [ 0.207964  ,  0.434421  , -0.411734  ],
       [ 0.235332  ,  0.285364  , -0.508637  ],
       [ 0.235267  ,  0.131167  , -0.417033  ],
       [ 0.161633  , -0.0253799 , -0.363032  ],
       [ 0.26869   , -0.107964  , -0.233449  ],
       [ 0.377213  , -0.2157    , -0.139287  ],
       [ 0.320443  , -0.369975  , -0.0680656 ],
       [ 0.263673  , -0.524251  ,  0.00315595],
       [ 0.206903  , -0.678526  ,  0.0743775 ],
       [ 0.150133  , -0.832802  ,  0.145599  ],
       [ 0.093363  , -0.987078  ,  0.216821  ],
       [ 0.0365929 , -1.14135   ,  0.288042  ],
       [ 0.00438163, -1.22889   ,  0.328453  ],
       [-0.040047  , -1.09063   ,  0.218901  ],
       [-0.166175  , -1.08318   ,  0.0715956 ],
       [-0.304505  , -1.01639   , -0.0211831 ],
       [-0.416302  , -1.10381   ,  0.0949787 ],
       [-0.584223  , -0.995921  ,  0.0957923 ],
       [-0.63015   , -0.965385  ,  0.385489  ],
       [-0.67664   , -0.827247  ,  0.493986  ],
       [-0.826078  , -0.819651  ,  0.594724  ],
       [-0.936039  , -0.755494  ,  0.740106  ],
       [-1.03406   , -0.583549  ,  0.744263  ],
       [-1.05712   , -0.459216  ,  0.891357  ],
       [-1.19344   , -0.358359  ,  0.9522    ],
       [-1.269     , -0.457331  ,  1.10316   ],
       [-1.42486   , -0.436468  ,  1.18866   ],
       [-1.55751   , -0.493249  ,  1.30009   ],
       [-1.69183   , -0.373243  ,  1.33984   ],
       [-1.81488   , -0.313819  ,  1.46656   ],
       [-1.80537   , -0.203449  ,  1.645     ],
       [-1.79732   , -0.0631315 ,  1.7639    ],
       [-1.78322   ,  0.0674739 ,  1.90117   ],
       [-1.82952   ,  0.209675  ,  2.00208   ],
       [-1.91443   ,  0.367467  ,  2.04371   ],
       [-1.89544   ,  0.561629  ,  2.05353   ],
       [-2.05176   ,  0.622268  ,  2.11819   ],
       [-1.97649   ,  0.798679  ,  2.13459   ],
       [-1.94454   ,  0.916244  ,  2.29093   ],
       [-1.91639   ,  1.07168   ,  2.375     ],
       [-1.89805   ,  1.23953   ,  2.43731   ],
       [-2.01844   ,  1.19162   ,  2.57817   ],
       [-2.13883   ,  1.14372   ,  2.71902   ],
       [-2.25922   ,  1.09581   ,  2.85988   ],
       [-2.37961   ,  1.04791   ,  3.00074   ],
       [-2.5       ,  1.        , -3.14159   ]]
        tj_b = [[-0.        ,  2.1       ,  0.392701  ], # cost: 9.85511
       [ 0.00467793,  1.95786   ,  0.277136  ],
       [-0.0254839 ,  1.76226   ,  0.272961  ],
       [ 0.0593704 ,  1.64567   ,  0.384568  ],
       [ 0.144519  ,  1.47528   ,  0.403598  ],
       [ 0.24026   ,  1.40588   ,  0.240088  ],
       [ 0.32897   ,  1.29505   ,  0.124016  ],
       [ 0.233237  ,  1.2132    , -0.0240789 ],
       [ 0.185689  ,  1.07631   ,  0.0860996 ],
       [ 0.218911  ,  0.920867  ,  0.00401163],
       [ 0.297386  ,  0.745154  ,  0.0191308 ],
       [ 0.354747  ,  0.557326  ,  0.0119145 ],
       [ 0.274374  ,  0.413704  , -0.0589234 ],
       [ 0.250227  ,  0.248627  ,  0.00740821],
       [ 0.173682  ,  0.141097  , -0.128609  ],
       [ 0.190416  ,  0.0151832 ,  0.0173499 ],
       [ 0.177986  , -0.127474  , -0.0962549 ],
       [ 0.144721  , -0.222212  , -0.295437  ],
       [ 0.283602  , -0.27488   , -0.398371  ],
       [ 0.194931  , -0.406516  , -0.480941  ],
       [ 0.0850136 , -0.514325  , -0.573015  ],
       [-0.00466381, -0.608402  , -0.713072  ],
       [ 0.0158795 , -0.791212  , -0.74515   ],
       [-0.16418   , -0.828904  , -0.777226  ],
       [-0.328351  , -0.736285  , -0.800236  ],
       [-0.426551  , -0.701317  , -0.991756  ],
       [-0.524751  , -0.666349  , -1.18327   ],
       [-0.701068  , -0.739115  , -1.20179   ],
       [-0.723341  , -0.725232  , -1.25952   ],
       [-0.804176  , -0.674848  , -1.46901   ],
       [-0.885011  , -0.624463  , -1.67851   ],
       [-0.965846  , -0.574078  , -1.888     ],
       [-1.04668   , -0.523693  , -2.0975    ],
       [-1.12752   , -0.473309  , -2.307     ],
       [-1.20835   , -0.422924  , -2.51649   ],
       [-1.26138   , -0.246933  , -2.48411   ],
       [-1.35992   , -0.1759    , -2.64116   ],
       [-1.45846   , -0.104867  , -2.79821   ],
       [-1.557     , -0.0338334 , -2.95527   ],
       [-1.65554   ,  0.0371998 , -3.11232   ],
       [-1.75408   ,  0.108233  ,  3.01381   ],
       [-1.85262   ,  0.179266  ,  2.85676   ],
       [-1.80791   ,  0.296471  ,  2.70764   ],
       [-1.84295   ,  0.407824  ,  2.54112   ],
       [-1.90793   ,  0.532592  ,  2.65978   ],
       [-2.03086   ,  0.67301   ,  2.68653   ],
       [-2.05935   ,  0.812553  ,  2.80168   ],
       [-2.12985   ,  0.968706  ,  2.85902   ],
       [-2.26689   ,  1.11357   ,  2.8602    ],
       [-2.38344   ,  1.05678   ,  3.0009    ],
       [-2.5       ,  1.        , -3.14159   ]]
        tj_c = [[-0.00000e+00,  2.10000e+00,  3.92701e-01], # 10.90804
       [ 1.03621e-03,  2.14905e+00,  9.08181e-02],
       [ 1.10533e-01,  2.29301e+00,  5.25616e-02],
       [ 2.11731e-01,  2.26691e+00, -1.38418e-01],
       [ 3.69641e-01,  2.20312e+00, -7.90313e-02],
       [ 4.26120e-01,  2.12237e+00, -2.81950e-01],
       [ 4.94427e-01,  1.94130e+00, -2.94895e-01],
       [ 4.60001e-01,  1.75638e+00, -3.18703e-01],
       [ 3.72188e-01,  1.61177e+00, -3.80336e-01],
       [ 3.09346e-01,  1.48267e+00, -4.93166e-01],
       [ 3.20707e-01,  1.32623e+00, -5.79467e-01],
       [ 2.28102e-01,  1.22140e+00, -6.99721e-01],
       [ 1.67046e-01,  1.08949e+00, -8.09000e-01],
       [ 1.98143e-01,  8.93903e-01, -8.12920e-01],
       [ 1.77782e-01,  6.95968e-01, -8.14960e-01],
       [ 2.19842e-01,  5.47219e-01, -9.05798e-01],
       [ 3.39845e-01,  3.99106e-01, -8.87050e-01],
       [ 5.14364e-01,  3.68445e-01, -8.41433e-01],
       [ 5.60645e-01,  2.37766e-01, -7.18698e-01],
       [ 6.03366e-01,  7.07443e-02, -7.73901e-01],
       [ 6.95048e-01, -9.26733e-02, -7.99144e-01],
       [ 6.07192e-01, -2.57417e-01, -8.25732e-01],
       [ 5.16146e-01, -3.06489e-01, -1.01888e+00],
       [ 4.25100e-01, -3.55560e-01, -1.21202e+00],
       [ 3.34054e-01, -4.04632e-01, -1.40516e+00],
       [ 2.43008e-01, -4.53704e-01, -1.59831e+00],
       [ 1.51962e-01, -5.02776e-01, -1.79145e+00],
       [ 6.09160e-02, -5.51847e-01, -1.98459e+00],
       [-3.01300e-02, -6.00919e-01, -2.17774e+00],
       [-1.21176e-01, -6.49991e-01, -2.37088e+00],
       [-2.12222e-01, -6.99063e-01, -2.56402e+00],
       [-3.03268e-01, -7.48134e-01, -2.75717e+00],
       [-3.94314e-01, -7.97206e-01, -2.95031e+00],
       [-4.85360e-01, -8.46278e-01,  3.13973e+00],
       [-5.34543e-01, -8.72786e-01,  3.03540e+00],
       [-6.62254e-01, -9.22624e-01, -3.12197e+00],
       [-8.50192e-01, -8.64544e-01, -3.11538e+00],
       [-1.04266e+00, -9.17213e-01, -3.11446e+00],
       [-1.09378e+00, -8.83666e-01, -2.83676e+00],
       [-1.23260e+00, -7.68052e-01, -2.79807e+00],
       [-1.27442e+00, -6.62199e-01, -2.62570e+00],
       [-1.26521e+00, -4.69011e-01, -2.63889e+00],
       [-1.42527e+00, -3.66965e-01, -2.65925e+00],
       [-1.44458e+00, -2.56842e-01, -2.83565e+00],
       [-1.46256e+00, -1.08439e-01, -2.73462e+00],
       [-1.60793e+00, -4.36694e-02, -2.65292e+00],
       [-1.78401e+00,  4.23054e-02, -2.64481e+00],
       [-1.78416e+00,  2.20887e-01, -2.68765e+00],
       [-1.79546e+00,  3.62032e-01, -2.80445e+00],
       [-1.88684e+00,  5.16376e-01, -2.76318e+00],
       [-1.89678e+00,  6.52217e-01, -2.89078e+00],
       [-2.01680e+00,  7.44677e-01, -2.98777e+00],
       [-2.17626e+00,  6.97958e-01, -3.05544e+00],
       [-2.35331e+00,  7.17207e-01, -3.09926e+00],
       [-2.36643e+00,  9.02993e-01, -3.07176e+00],
       [-2.50000e+00,  1.00000e+00, -3.14159e+00]]
        tj_d = [[-0.        ,  2.1       ,  0.392701  ], # 12.48887
       [ 0.139569  ,  2.08543   ,  0.273356  ],
       [ 0.227537  ,  1.98913   ,  0.134224  ],
       [ 0.229875  ,  1.83587   ,  0.0407701 ],
       [ 0.323961  ,  1.70722   , -0.0404562 ],
       [ 0.465547  ,  1.60167   ,  0.00634619],
       [ 0.482292  ,  1.41878   ,  0.0390481 ],
       [ 0.446123  ,  1.24544   , -0.00680645],
       [ 0.495396  ,  1.1319    ,  0.14565   ],
       [ 0.512665  ,  0.972999  ,  0.22597   ],
       [ 0.572157  ,  0.795254  ,  0.251097  ],
       [ 0.614694  ,  0.649204  ,  0.155335  ],
       [ 0.615436  ,  0.493344  ,  0.0670587 ],
       [ 0.623793  ,  0.308478  ,  0.037168  ],
       [ 0.463789  ,  0.235456  , -0.0110722 ],
       [ 0.348092  ,  0.0848329 , -0.0312141 ],
       [ 0.267299  , -0.0880793 , -0.012927  ],
       [ 0.338009  , -0.207769  , -0.134895  ],
       [ 0.42238   , -0.387306  , -0.138147  ],
       [ 0.443422  , -0.47824   ,  0.0751795 ],
       [ 0.378783  , -0.599233  ,  0.200826  ],
       [ 0.215894  , -0.68623   ,  0.231495  ],
       [ 0.0440172 , -0.787697  ,  0.23068   ],
       [ 0.0774157 , -0.984575  ,  0.230062  ],
       [ 0.0232548 , -1.07376   ,  0.42138   ],
       [-0.0293856 , -1.05316   ,  0.486056  ],
       [-0.147857  , -1.00679   ,  0.631614  ],
       [-0.266329  , -0.960429  ,  0.777172  ],
       [-0.3848    , -0.914064  ,  0.92273   ],
       [-0.503272  , -0.8677    ,  1.06829   ],
       [-0.621743  , -0.821336  ,  1.21385   ],
       [-0.740215  , -0.774971  ,  1.3594    ],
       [-0.793995  , -0.63376   ,  1.26162   ],
       [-0.947701  , -0.666206  ,  1.1758    ],
       [-0.905737  , -0.554368  ,  1.01471   ],
       [-0.992933  , -0.54829   ,  0.789522  ],
       [-1.14496   , -0.46919   ,  0.732262  ],
       [-1.30162   , -0.378942  ,  0.693857  ],
       [-1.27722   , -0.256353  ,  0.543843  ],
       [-1.19517   , -0.0937672 ,  0.579605  ],
       [-1.30792   , -0.0300795 ,  0.720609  ],
       [-1.42068   ,  0.0336081 ,  0.861613  ],
       [-1.44436   ,  0.220015  ,  0.885801  ],
       [-1.59205   ,  0.348168  ,  0.894721  ],
       [-1.70986   ,  0.41645   ,  1.02239   ],
       [-1.86617   ,  0.494108  ,  1.07332   ],
       [-2.02248   ,  0.571766  ,  1.12424   ],
       [-1.94818   ,  0.692789  ,  1.24022   ],
       [-2.09447   ,  0.774015  ,  1.30557   ],
       [-2.20166   ,  0.922504  ,  1.3393    ],
       [-2.08703   ,  1.04677   ,  1.40118   ],
       [-1.94289   ,  1.08284   ,  1.50401   ],
       [-1.93897   ,  1.2627    ,  1.5442    ],
       [-1.93374   ,  1.45926   ,  1.55093   ],
       [-1.88461   ,  1.45611   ,  1.85247   ],
       [-2.02921   ,  1.45839   ,  1.96325   ],
       [-2.0768    ,  1.41755   ,  2.23782   ],
       [-1.96365   ,  1.38859   ,  2.40424   ],
       [-1.86944   ,  1.28708   ,  2.52726   ],
       [-1.99555   ,  1.22966   ,  2.65013   ],
       [-2.12167   ,  1.17225   ,  2.77299   ],
       [-2.24778   ,  1.11483   ,  2.89586   ],
       [-2.37389   ,  1.05742   ,  3.01873   ],
       [-2.5       ,  1.        , -3.14159   ]]
        tj_e = [[-0.       ,  2.1      ,  0.392701 ], # 11.44077
       [-0.0962288,  1.97623  ,  0.479153 ],
       [-0.065071 ,  1.83039  ,  0.377432 ],
       [-0.0208806,  1.68173  ,  0.287606 ],
       [ 0.0219156,  1.50957  ,  0.242402 ],
       [ 0.195807 ,  1.52915  ,  0.192383 ],
       [ 0.254148 ,  1.39435  ,  0.298611 ],
       [ 0.333058 ,  1.36579  ,  0.530771 ],
       [ 0.312525 ,  1.21316  ,  0.622776 ],
       [ 0.230496 ,  1.09709  ,  0.738514 ],
       [ 0.190517 ,  0.950689 ,  0.834985 ],
       [ 0.199867 ,  0.784739 ,  0.767413 ],
       [ 0.151796 ,  0.622718 ,  0.82941  ],
       [ 0.217371 ,  0.536659 ,  1.01302  ],
       [ 0.276327 ,  0.430404 ,  0.856051 ],
       [ 0.440119 ,  0.426643 ,  0.783721 ],
       [ 0.458402 ,  0.306025 ,  0.627713 ],
       [ 0.478758 ,  0.15728  ,  0.527975 ],
       [ 0.464033 ,  0.128491 ,  0.544846 ],
       [ 0.391803 , -0.0127291,  0.627605 ],
       [ 0.319572 , -0.15395  ,  0.710364 ],
       [ 0.247342 , -0.29517  ,  0.793123 ],
       [ 0.175112 , -0.436391 ,  0.875882 ],
       [ 0.102881 , -0.577611 ,  0.958641 ],
       [ 0.030651 , -0.718832 ,  1.0414   ],
       [-0.0415793, -0.860053 ,  1.12416  ],
       [-0.11381  , -1.00127  ,  1.20692  ],
       [-0.18604  , -1.14249  ,  1.28968  ],
       [-0.25827  , -1.28371  ,  1.37244  ],
       [-0.330501 , -1.42493  ,  1.45519  ],
       [-0.388852 , -1.31916  ,  1.6136   ],
       [-0.542595 , -1.36854  ,  1.69064  ],
       [-0.512172 , -1.30173  ,  1.94381  ],
       [-0.631762 , -1.24959  ,  2.08288  ],
       [-0.721006 , -1.16256  ,  1.93218  ],
       [-0.746526 , -1.03269  ,  1.79688  ],
       [-0.822826 , -0.904442 ,  1.89841  ],
       [-0.955289 , -0.777773 ,  1.93185  ],
       [-0.92488  , -0.669029 ,  2.10602  ],
       [-1.04587  , -0.591118 ,  2.2182   ],
       [-1.19017  , -0.483003 ,  2.17882  ],
       [-1.16233  , -0.384816 ,  1.98293  ],
       [-1.27521  , -0.26142  ,  1.91741  ],
       [-1.35986  , -0.14339  ,  2.02692  ],
       [-1.55561  , -0.184117 ,  2.02702  ],
       [-1.69383  , -0.0835249,  2.08512  ],
       [-1.82101  ,  0.051435 ,  2.056    ],
       [-1.71645  ,  0.202719 ,  2.0882   ],
       [-1.82239  ,  0.326574 ,  2.16222  ],
       [-1.91847  ,  0.397925 ,  2.32288  ],
       [-1.95058  ,  0.594845 ,  2.32192  ],
       [-1.92461  ,  0.731331 ,  2.44405  ],
       [-1.89778  ,  0.874189 ,  2.55334  ],
       [-1.88217  ,  1.04831  ,  2.60371  ],
       [-2.01014  ,  1.19894  ,  2.59902  ],
       [-2.13261  ,  1.14921  ,  2.73466  ],
       [-2.25507  ,  1.09947  ,  2.87031  ],
       [-2.37754  ,  1.04974  ,  3.00595  ],
       [-2.5      ,  1.       , -3.14159  ]]

        # kernprof -lv py.test -s test/test_integration_pr2.py::TestCartGoals::test_pathAroundKitchenIsland_with_global_planner
        """
        :type kitchen_setup_avoid_collisions: PR2
        """
        poses = []
        for i, point in enumerate(tj_e):
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
                pass
            else:
                poses.append(base_pose)
        tip_link = u'base_footprint'
        kitchen_setup_avoid_collisions.set_kitchen_js({'kitchen_island_left_upper_drawer_main_joint': 0.48})
        kitchen_setup_avoid_collisions.set_kitchen_js({u'oven_area_area_right_drawer_main_joint': 0.28})

        box_pose = PoseStamped()
        box_pose.header.frame_id = tip_link
        box_pose.pose.position.x = -2.5
        box_pose.pose.position.y = 0
        box_pose.pose.position.z = 0
        box_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
        kitchen_setup_avoid_collisions.add_box('box1', [0.5, 0.5, 1], pose=box_pose)
        box_pose.pose.position.x = -0.5
        kitchen_setup_avoid_collisions.add_box('box2', [0.5, 0.5, 1], pose=box_pose)

        base_pose = PoseStamped()
        base_pose.header.frame_id = tip_link
        base_pose.pose.position.x = 0
        base_pose.pose.position.y = 2.1
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi/8, [0, 0, 1]))
        kitchen_setup_avoid_collisions.teleport_base(base_pose)

        base_pose = PoseStamped()
        base_pose.header.frame_id = 'map'
        base_pose.pose.position.x = -2.5
        base_pose.pose.position.y = 1
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0, 0, 1]))
        goal_c = base_pose
        kitchen_setup_avoid_collisions.allow_all_collisions()
        try:
            kitchen_setup_avoid_collisions.set_json_goal('CartesianPathCarrot',
                                                         tip_link=tip_link,
                                                         root_link=kitchen_setup_avoid_collisions.default_root,
                                                         goal=goal_c,
                                                         goals=poses,
                                                         predict_f=10.0)
            kitchen_setup_avoid_collisions.plan_and_execute()
        except Exception:
            pass
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

        # pregrasping
        kitchen_setup_avoid_collisions.allow_collision(body_b=milk_name)
        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPose',
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     tip_link=tip_link,
                                                     goal=milk_pose)

        kitchen_setup_avoid_collisions.plan_and_execute()

        # grasping
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

    def test_ease_fridge_picking_1(self, kitchen_setup_avoid_collisions):
        rospy.sleep(10.0)#0.5

        tip_link = kitchen_setup_avoid_collisions.l_tip
        milk_name = u'milk'

        # take milk out of fridge
        kitchen_setup_avoid_collisions.set_kitchen_js({u'iai_fridge_door_joint': 1.56})

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 0.0
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

        milk_pre_pose = PoseStamped()
        milk_pre_pose.header.frame_id = u'iai_kitchen/iai_fridge_door_shelf1_bottom'
        milk_pre_pose.pose.position = Point(-0.3, -0.1, 0.22)
        milk_pre_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup_avoid_collisions.add_box(milk_name, [0.05, 0.05, 0.2], milk_pose)

        # move arm towards milk
        if tip_link == kitchen_setup_avoid_collisions.l_tip:
            kitchen_setup_avoid_collisions.open_l_gripper()
        elif tip_link == kitchen_setup_avoid_collisions.r_tip:
            kitchen_setup_avoid_collisions.open_r_gripper()
        else:
            raise Exception('Wrong tip link')

        # Grasp
        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPose',
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     tip_link=tip_link,
                                                     goal=milk_pose)
        kitchen_setup_avoid_collisions.plan_and_execute()

        # attach milk
        kitchen_setup_avoid_collisions.attach_object(milk_name, tip_link)

        # pick up
        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPathCarrot',
                                                     tip_link=tip_link,
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     goal=milk_pre_pose)
        kitchen_setup_avoid_collisions.plan_and_execute()

        if tip_link == kitchen_setup_avoid_collisions.l_tip:
            kitchen_setup_avoid_collisions.close_l_gripper()
        elif tip_link == kitchen_setup_avoid_collisions.r_tip:
            kitchen_setup_avoid_collisions.close_r_gripper()
        else:
            raise Exception('Wrong tip link')

        kitchen_setup_avoid_collisions.set_joint_goal(gaya_pose)
        kitchen_setup_avoid_collisions.plan_and_execute()

        # dont allow robot to move
        js = kitchen_setup_avoid_collisions.god_map.get_data(identifier.joint_states)
        odom_joints = ['odom_x_joint', 'odom_y_joint', 'odom_z_joint']
        kitchen_setup_avoid_collisions.set_joint_goal({j_n: js[j_n].position for j_n in odom_joints})

        # place milk back
        # pregrasping
        kitchen_setup_avoid_collisions.allow_collision(body_b=milk_name)
        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPose',
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     tip_link=tip_link,
                                                     goal=milk_pose)
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

    def test_ease_fridge_placing_global_planner(self, kitchen_setup_avoid_collisions):
        rospy.sleep(10.0)#0.5

        tip_link = kitchen_setup_avoid_collisions.l_tip
        milk_name = u'milk'

        # take milk out of fridge
        kitchen_setup_avoid_collisions.set_kitchen_js({u'iai_fridge_door_joint': 1.56})

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 0.0
        base_goal.pose.position.y = -1.05
        base_goal.pose.orientation.x = 0
        base_goal.pose.orientation.y = 0
        base_goal.pose.orientation.z = 0
        base_goal.pose.orientation.w = 1
        kitchen_setup_avoid_collisions.teleport_base(base_goal)

        # spawn milk
        milk_pose = PoseStamped()
        milk_pose.header.frame_id = u'iai_kitchen/iai_fridge_door_shelf1_bottom'
        milk_pose.pose.position = Point(0, 0, 0.125)
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

        # Pregrasp
        kitchen_setup_avoid_collisions.allow_collision(body_b=milk_name)
        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPose',
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     tip_link=tip_link,
                                                     goal=milk_pose)

        kitchen_setup_avoid_collisions.plan_and_execute()

        # Grasp
        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPose',
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     tip_link=tip_link,
                                                     goal=milk_pose)

        kitchen_setup_avoid_collisions.plan_and_execute()

        # attach milk
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
        # pregrasping
        kitchen_setup_avoid_collisions.allow_collision(body_b=milk_name)
        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPathCarrot',
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     tip_link=tip_link,
                                                     goal=milk_pose)
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

    def test_ease_fridge_placing_presampling(self, kitchen_setup_avoid_collisions):
        rospy.sleep(10.0)#0.5

        tip_link = kitchen_setup_avoid_collisions.l_tip
        milk_name = u'milk'

        # take milk out of fridge
        kitchen_setup_avoid_collisions.set_kitchen_js({u'iai_fridge_door_joint': 1.56})

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 0.0
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

        # Pregrasp
        kitchen_setup_avoid_collisions.allow_collision(body_b=milk_name)
        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPose',
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     tip_link=tip_link,
                                                     goal=milk_pose)

        kitchen_setup_avoid_collisions.plan_and_execute()

        # Grasp
        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPose',
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     tip_link=tip_link,
                                                     goal=milk_pose)

        kitchen_setup_avoid_collisions.plan_and_execute()

        # attach milk
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
        # pregrasping
        kitchen_setup_avoid_collisions.allow_collision(body_b=milk_name)
        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPreGrasp',
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     tip_link=tip_link,
                                                     grasping_object=milk_name,
                                                     grasping_orientation=milk_pose.pose.orientation,
                                                     grasping_goal=milk_pose)
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

    @pytest.mark.repeat(5)
    def test_ease_fridge_pregrasp_1(self, kitchen_setup_avoid_collisions):
        rospy.sleep(10.0)

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

        milk_pre_pose = PoseStamped()
        milk_pre_pose.header.frame_id = u'iai_kitchen/iai_fridge_door_shelf1_bottom'
        milk_pre_pose.pose.position = Point(-0.2, -0.1, 0.22)
        milk_pre_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup_avoid_collisions.add_box(milk_name, [0.05, 0.05, 0.2], milk_pose)

        # move arm towards milk
        if tip_link == kitchen_setup_avoid_collisions.l_tip:
            kitchen_setup_avoid_collisions.open_l_gripper()
        elif tip_link == kitchen_setup_avoid_collisions.r_tip:
            kitchen_setup_avoid_collisions.open_r_gripper()
        else:
            raise Exception('Wrong tip link')

        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPathCarrot',
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     tip_link=tip_link,
                                                     goal=milk_pre_pose)
        kitchen_setup_avoid_collisions.plan_and_execute()

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

    def test_ease_cereal_with_planner_placing(self, kitchen_setup_avoid_collisions):
        # FIXME collision avoidance needs soft_threshholds at 0
        cereal_name = u'cereal'
        drawer_frame_id = u'iai_kitchen/oven_area_area_right_drawer_board_2_link'

        # take milk out of fridge
        kitchen_setup_avoid_collisions.set_kitchen_js({u'oven_area_area_right_drawer_main_joint': 0.48})

        cereal_pose = PoseStamped()
        cereal_pose.header.frame_id = drawer_frame_id
        cereal_pose.pose.position = Point(0.123, 0.25, 0.15)
        cereal_pose.pose.orientation = Quaternion(0.0087786, 0.005395, -0.838767, -0.544393)
        kitchen_setup_avoid_collisions.add_box(cereal_name, [0.1028, 0.0634, 0.20894], cereal_pose)

        drawer_T_pick_box = tf.msg_to_kdl(cereal_pose)
        cereal_pose.pose.position = Point(0.123, 0.0, 0.15)
        drawer_T_place_box = tf.msg_to_kdl(cereal_pose)

        # grasp milk
        kitchen_setup_avoid_collisions.open_l_gripper()
        grasp_pose = PoseStamped()
        grasp_pose.header.frame_id = cereal_name
        grasp_pose.pose.position = Point(0.03, 0, 0.05)
        grasp_pose.pose.orientation = Quaternion(0, 0, 1, 0)
        box_T_r_goal = tf.msg_to_kdl(grasp_pose)

        grasp_pose = tf.kdl_to_pose_stamped(drawer_T_pick_box * box_T_r_goal, drawer_frame_id)

        kitchen_setup_avoid_collisions.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup_avoid_collisions.set_cart_goal(grasp_pose,tip_link=kitchen_setup_avoid_collisions.r_tip)
        kitchen_setup_avoid_collisions.plan_and_execute()

        kitchen_setup_avoid_collisions.attach_object(cereal_name, kitchen_setup_avoid_collisions.r_tip)
        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        kitchen_setup_avoid_collisions.close_l_gripper()

        grasp_pose = tf.kdl_to_pose_stamped(drawer_T_place_box * box_T_r_goal, drawer_frame_id)
        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPathCarrot',
                                                     tip_link=kitchen_setup_avoid_collisions.r_tip,
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     goal=grasp_pose)
        kitchen_setup_avoid_collisions.plan_and_execute()
        #kitchen_setup_avoid_collisions.god_map.set_data(identifier.rosparam + ['reset_god_map'], True)

        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        kitchen_setup_avoid_collisions.open_l_gripper()
        kitchen_setup_avoid_collisions.detach_object(cereal_name)

        kitchen_setup_avoid_collisions.set_joint_goal(gaya_pose)
        kitchen_setup_avoid_collisions.plan_and_execute()

    #@pytest.mark.repeat(10)
    def test_ease_cereal_with_planner_1_pick(self, kitchen_setup_avoid_collisions):
        tj_a = [[0.926, 1.453, 1.09, 0.00873676, 0.00537855,
          0.839261, 0.543633],
         [0.924767, 1.43488, 1.08704, -0.0197094, -0.00763073,
          0.795805, 0.605183],
         [0.939336, 1.41977, 1.09107, -0.0718906, -0.059543,
          0.775747, 0.624102],
         [0.95405, 1.42687, 1.08886, -0.134803, -0.11318,
          0.765101, 0.619385],
         [0.973875, 1.41395, 1.09299, -0.137585, -0.185483,
          0.743478, 0.62762],
         [0.962465, 1.40553, 1.09279, -0.167463, -0.222918,
          0.687045, 0.670994],
         [0.980964, 1.42372, 1.07323, -0.2151, -0.2247,
          0.64663, 0.6965],
         [0.997692, 1.42536, 1.06782, -0.171931, -0.220107,
          0.600615, 0.749169],
         [1.02751, 1.44197, 1.0798, -0.184796, -0.274952,
          0.570708, 0.751362],
         [1.01039, 1.44075, 1.06623, -0.238538, -0.278445,
          0.518295, 0.772618],
         [0.995644, 1.43728, 1.07101, -0.187906, -0.296294,
          0.466775, 0.811802],
         [0.982401, 1.44527, 1.08122, -0.226686, -0.360485,
          0.448531, 0.785802],
         [0.988509, 1.42962, 1.07476, -0.190508, -0.422415,
          0.408842, 0.786206],
         [0.992892, 1.40333, 1.07564, -0.249291, -0.445875,
          0.373852, 0.774135],
         [0.998308, 1.41604, 1.08756, -0.310081, -0.429826,
          0.322618, 0.78423],
         [1.00619, 1.40645, 1.08215, -0.3877, -0.437372,
          0.315053, 0.747754],
         [1.02766, 1.36858, 1.08451, -0.40264, -0.47731,
          0.322155, 0.711528],
         [1.03031726, 1.35620905, 1.08458356, -0.38882257, -0.46118023,
          0.35331032, 0.71505358],
         [1.03297451, 1.34383809, 1.08465712, -0.37444745, -0.44438898,
          0.38395889, 0.71755356],
         [1.03563177, 1.33146714, 1.08473067, -0.35953531, -0.4269604,
          0.4140568, 0.71902444],
         [1.03828903, 1.31909619, 1.08480423, -0.34410753, -0.40891949,
          0.44356088, 0.71946411],
         [1.04094628, 1.30672523, 1.08487779, -0.32818624, -0.39029211,
          0.47242881, 0.71887195],
         [1.04360354, 1.29435428, 1.08495135, -0.31179427, -0.37110499,
          0.5006192, 0.71724879],
         [1.0462608, 1.28198333, 1.08502491, -0.29495514, -0.35138564,
          0.52809161, 0.71459698],
         [1.04891805, 1.26961237, 1.08509847, -0.27769299, -0.33116234,
          0.55480665, 0.71092031],
         [1.05157531, 1.25724142, 1.08517202, -0.26003258, -0.31046409,
          0.580726, 0.70622406],
         [1.05423256, 1.24487047, 1.08524558, -0.24199923, -0.28932059,
          0.60581248, 0.70051496],
         [1.05688982, 1.23249951, 1.08531914, -0.22361882, -0.26776215,
          0.63003013, 0.69380119],
         [1.05954708, 1.22012856, 1.0853927, -0.2049177, -0.24581969,
          0.6533442, 0.6860924],
         [1.06220433, 1.20775761, 1.08546626, -0.18592269, -0.22352468,
          0.67572126, 0.67739962],
         [1.06486159, 1.19538665, 1.08553982, -0.16666103, -0.2009091,
          0.69712922, 0.66773534],
         [1.06751885, 1.1830157, 1.08561337, -0.14716036, -0.17800538,
          0.71753737, 0.65711341],
         [1.0701761, 1.17064474, 1.08568693, -0.12744863, -0.15484637,
          0.73691645, 0.64554907],
         [1.07283336, 1.15827379, 1.08576049, -0.10755412, -0.13146528,
          0.75523866, 0.63305889],
         [1.07549062, 1.14590284, 1.08583405, -0.08750535, -0.10789565,
          0.77247773, 0.6196608],
         [1.07814787, 1.13353188, 1.08590761, -0.06733109, -0.08417128,
          0.78860893, 0.605374],
         [1.08080513, 1.12116093, 1.08598117, -0.04706026, -0.06032619,
          0.80360912, 0.59021899],
         [1.08346239, 1.10878998, 1.08605472, -0.02672194, -0.03639458,
          0.8174568, 0.57421751],
         [1.08809093, 1.08724164, 1.08618285, 0.0087786, 0.005395,
          0.83876701, 0.544393]]
        tj_b =  [[0.926, 1.452, 1.09, 0.00874084, 0.00537131,
          0.838827, 0.544301],
         [0.922181, 1.45679, 1.09301, 0.0143654, -0.0535375,
          0.874501, 0.481845],
         [0.951395, 1.4434, 1.08215, -0.0115004, -0.112476,
          0.863365, 0.49175],
         [0.93573, 1.40442, 1.07948, -0.0233309, -0.167639,
          0.851312, 0.496609],
         [0.943229, 1.40641, 1.07352, -0.0735979, -0.238196,
          0.845482, 0.472235],
         [0.956869, 1.39425, 1.08902, -0.0965844, -0.264701,
          0.868083, 0.4087],
         [0.97617, 1.37671, 1.08455, -0.0740051, -0.311388,
          0.877389, 0.357421],
         [0.993055, 1.36458, 1.08085, -0.0254254, -0.351056,
          0.883197, 0.309962],
         [1.00169, 1.35857, 1.07208, -0.00371679, -0.418711,
          0.84294, 0.337815],
         [0.999879, 1.36393, 1.07157, 0.0553312, -0.445256,
          0.802545, 0.393201],
         [0.994198, 1.31988, 1.07208, 0.103172, -0.418372,
          0.810936, 0.395857],
         [0.997454, 1.29001, 1.08911, 0.1648, -0.422639,
          0.808764, 0.374323],
         [1.00528, 1.27962, 1.07887, 0.223283, -0.452137,
          0.800751, 0.323286],
         [1.00382, 1.26705, 1.08957, 0.281059, -0.454713,
          0.760745, 0.368116],
         [1.00861, 1.23581, 1.09965, 0.290085, -0.45249,
          0.727988, 0.425603],
         [1.01104175, 1.2312645, 1.09923797, 0.28242901, -0.43993104,
          0.73516316, 0.43154334],
         [1.0134735, 1.22671899, 1.09882594, 0.27468755, -0.42723895,
          0.74211583, 0.43735308],
         [1.01590525, 1.22217349, 1.0984139, 0.26686283, -0.41441736,
          0.74884356, 0.44303026],
         [1.01833699, 1.21762798, 1.09800187, 0.25895721, -0.40147015,
          0.7553443, 0.44857314],
         [1.02076874, 1.21308248, 1.09758984, 0.25097311, -0.38840126,
          0.76161609, 0.45398006],
         [1.02320049, 1.20853698, 1.09717781, 0.24291293, -0.37521463,
          0.76765702, 0.45924937],
         [1.02563224, 1.20399147, 1.09676577, 0.23477913, -0.36191427,
          0.77346527, 0.46437948],
         [1.02806399, 1.19944597, 1.09635374, 0.22657415, -0.34850421,
          0.77903907, 0.46936882],
         [1.03049574, 1.19490046, 1.09594171, 0.2183005, -0.33498852,
          0.78437674, 0.4742159],
         [1.03292748, 1.19035496, 1.09552968, 0.20996069, -0.32137128,
          0.78947665, 0.47891923],
         [1.03535923, 1.18580946, 1.09511764, 0.20155723, -0.30765664,
          0.79433726, 0.4834774],
         [1.03779098, 1.18126395, 1.09470561, 0.19309267, -0.29384874,
          0.79895709, 0.48788903],
         [1.04022273, 1.17671845, 1.09429358, 0.18456959, -0.27995177,
          0.80333476, 0.49215276],
         [1.04265448, 1.17217294, 1.09388155, 0.17599056, -0.26596994,
          0.80746892, 0.49626732],
         [1.04508623, 1.16762744, 1.09346951, 0.16735819, -0.2519075,
          0.81135833, 0.50023145],
         [1.04751797, 1.16308194, 1.09305748, 0.15867508, -0.2377687,
          0.81500181, 0.50404396],
         [1.04994972, 1.15853643, 1.09264545, 0.14994389, -0.22355783,
          0.81839825, 0.50770369],
         [1.05238147, 1.15399093, 1.09223342, 0.14116724, -0.20927919,
          0.82154662, 0.51120952],
         [1.05481322, 1.14944543, 1.09182138, 0.1323478, -0.19493712,
          0.82444598, 0.51456041],
         [1.05724497, 1.14489992, 1.09140935, 0.12348825, -0.18053597,
          0.82709543, 0.51775532],
         [1.05967672, 1.14035442, 1.09099732, 0.11459126, -0.16608009,
          0.82949418, 0.52079329],
         [1.06210846, 1.13580891, 1.09058529, 0.10565954, -0.15157387,
          0.8316415, 0.52367341],
         [1.06454021, 1.13126341, 1.09017325, 0.0966958, -0.1370217,
          0.83353674, 0.52639479],
         [1.06697196, 1.12671791, 1.08976122, 0.08770274, -0.122428,
          0.83517933, 0.52895662],
         [1.06940371, 1.1221724, 1.08934919, 0.07868311, -0.1077972,
          0.83656876, 0.53135811],
         [1.07183546, 1.1176269, 1.08893716, 0.06963962, -0.09313372,
          0.83770462, 0.53359854],
         [1.07426721, 1.11308139, 1.08852512, 0.06057502, -0.07844201,
          0.83858656, 0.53567723],
         [1.07669895, 1.10853589, 1.08811309, 0.05149206, -0.06372652,
          0.83921431, 0.53759356],
         [1.0791307, 1.10399039, 1.08770106, 0.0423935, -0.04899171,
          0.83958769, 0.53934693],
         [1.08156245, 1.09944488, 1.08728903, 0.03328208, -0.03424206,
          0.83970658, 0.54093682],
         [1.0839942, 1.09489938, 1.08687699, 0.02416058, -0.01948202,
          0.83957095, 0.54236274],
         [1.08809093, 1.08724164, 1.08618285, 0.0087786, 0.005395,
          0.83876701, 0.544393]]
        tj_c = [[9.26000000e-01, 1.45300000e+00, 1.09000000e+00,
          8.73823000e-03, 5.37644000e-03, 8.39126000e-01,
          5.43841000e-01],
         [9.33996000e-01, 1.45459000e+00, 1.09255000e+00,
          1.29798000e-03, -4.94932000e-02, 7.96408000e-01,
          6.02730000e-01],
         [9.52261000e-01, 1.43676000e+00, 1.08412000e+00,
          7.52382000e-03, -1.21925000e-01, 7.88763000e-01,
          6.02437000e-01],
         [9.38384000e-01, 1.40772000e+00, 1.08275000e+00,
          -4.78447000e-02, -1.59622000e-01, 7.78723000e-01,
          6.04833000e-01],
         [9.33168000e-01, 1.40032000e+00, 1.07677000e+00,
          -1.19013000e-01, -2.01572000e-01, 7.86004000e-01,
          5.72190000e-01],
         [9.50019000e-01, 1.41200000e+00, 1.06858000e+00,
          -1.68142000e-01, -2.57843000e-01, 7.64800000e-01,
          5.65974000e-01],
         [9.69627000e-01, 1.41291000e+00, 1.08167000e+00,
          -2.16568000e-01, -2.99559000e-01, 7.67177000e-01,
          5.24216000e-01],
         [9.87996000e-01, 1.40710000e+00, 1.07740000e+00,
          -2.93443000e-01, -3.03316000e-01, 7.47262000e-01,
          5.13313000e-01],
         [1.00875000e+00, 1.41683000e+00, 1.05868000e+00,
          -2.94492000e-01, -3.60788000e-01, 7.48109000e-01,
          4.72693000e-01],
         [9.97204000e-01, 1.42263000e+00, 1.05931000e+00,
          -3.57943000e-01, -3.97640000e-01, 7.01333000e-01,
          4.71054000e-01],
         [9.77645000e-01, 1.38039000e+00, 1.05451000e+00,
          -3.69143000e-01, -4.19075000e-01, 7.12392000e-01,
          4.24978000e-01],
         [9.79995000e-01, 1.35957000e+00, 1.06629000e+00,
          -3.59625000e-01, -4.83372000e-01, 6.73279000e-01,
          4.28622000e-01],
         [9.64143000e-01, 1.37410000e+00, 1.08057000e+00,
          -3.37398000e-01, -5.30613000e-01, 6.28983000e-01,
          4.57157000e-01],
         [9.59996000e-01, 1.35213000e+00, 1.07564000e+00,
          -3.34656000e-01, -5.87335000e-01, 5.76868000e-01,
          4.58548000e-01],
         [9.67521000e-01, 1.33524000e+00, 1.07541000e+00,
          -2.98902000e-01, -6.36156000e-01, 5.26145000e-01,
          4.78680000e-01],
         [9.59806000e-01, 1.31795000e+00, 1.07156000e+00,
          -3.59865000e-01, -6.30564000e-01, 4.76502000e-01,
          4.95815000e-01],
         [9.64603738e-01, 1.30932172e+00, 1.07210688e+00,
          -3.49231721e-01, -6.12354122e-01, 4.98355819e-01,
          5.04679217e-01],
         [9.69401477e-01, 1.30069344e+00, 1.07265376e+00,
          -3.38249001e-01, -5.93531523e-01, 5.19710974e-01,
          5.13038447e-01],
         [9.74199215e-01, 1.29206516e+00, 1.07320064e+00,
          -3.26927751e-01, -5.74114899e-01, 5.40545988e-01,
          5.20884213e-01],
         [9.78996954e-01, 1.28343688e+00, 1.07374752e+00,
          -3.15279302e-01, -5.54123684e-01, 5.60840007e-01,
          5.28208662e-01],
         [9.83794692e-01, 1.27480860e+00, 1.07429441e+00,
          -3.03315312e-01, -5.33577886e-01, 5.80572720e-01,
          5.35004465e-01],
         [9.88592431e-01, 1.26618032e+00, 1.07484129e+00,
          -2.91047755e-01, -5.12498067e-01, 5.99724380e-01,
          5.41264819e-01],
         [9.93390169e-01, 1.25755203e+00, 1.07538817e+00,
          -2.78488909e-01, -4.90905325e-01, 6.18275817e-01,
          5.46983459e-01],
         [9.98187908e-01, 1.24892375e+00, 1.07593505e+00,
          -2.65651342e-01, -4.68821270e-01, 6.36208465e-01,
          5.52154662e-01],
         [1.00298565e+00, 1.24029547e+00, 1.07648193e+00,
          -2.52547904e-01, -4.46268005e-01, 6.53504377e-01,
          5.56773251e-01],
         [1.00778338e+00, 1.23166719e+00, 1.07702881e+00,
          -2.39191709e-01, -4.23268101e-01, 6.70146242e-01,
          5.60834606e-01],
         [1.01258112e+00, 1.22303891e+00, 1.07757569e+00,
          -2.25596123e-01, -3.99844578e-01, 6.86117405e-01,
          5.64334661e-01],
         [1.01737886e+00, 1.21441063e+00, 1.07812257e+00,
          -2.11774754e-01, -3.76020879e-01, 7.01401881e-01,
          5.67269912e-01],
         [1.02217660e+00, 1.20578235e+00, 1.07866946e+00,
          -1.97741435e-01, -3.51820847e-01, 7.15984373e-01,
          5.69637423e-01],
         [1.02697434e+00, 1.19715407e+00, 1.07921634e+00,
          -1.83510210e-01, -3.27268702e-01, 7.29850287e-01,
          5.71434824e-01],
         [1.03177208e+00, 1.18852579e+00, 1.07976322e+00,
          -1.69095323e-01, -3.02389017e-01, 7.42985745e-01,
          5.72660316e-01],
         [1.03656982e+00, 1.17989751e+00, 1.08031010e+00,
          -1.54511200e-01, -2.77206692e-01, 7.55377600e-01,
          5.73312673e-01],
         [1.04136755e+00, 1.17126923e+00, 1.08085698e+00,
          -1.39772437e-01, -2.51746930e-01, 7.67013451e-01,
          5.73391241e-01],
         [1.04616529e+00, 1.16264095e+00, 1.08140386e+00,
          -1.24893786e-01, -2.26035212e-01, 7.77881652e-01,
          5.72895942e-01],
         [1.05096303e+00, 1.15401267e+00, 1.08195074e+00,
          -1.09890138e-01, -2.00097272e-01, 7.87971326e-01,
          5.71827271e-01],
         [1.05576077e+00, 1.14538438e+00, 1.08249762e+00,
          -9.47765082e-02, -1.73959068e-01, 7.97272375e-01,
          5.70186299e-01],
         [1.06055851e+00, 1.13675610e+00, 1.08304451e+00,
          -7.95680233e-02, -1.47646761e-01, 8.05775490e-01,
          5.67974667e-01],
         [1.06535625e+00, 1.12812782e+00, 1.08359139e+00,
          -6.42799045e-02, -1.21186685e-01, 8.13472161e-01,
          5.65194590e-01],
         [1.07015398e+00, 1.11949954e+00, 1.08413827e+00,
          -4.89274524e-02, -9.46053209e-02, 8.20354685e-01,
          5.61848849e-01],
         [1.07495172e+00, 1.11087126e+00, 1.08468515e+00,
          -3.35260322e-02, -6.79292734e-02, 8.26416173e-01,
          5.57940793e-01],
         [1.07974946e+00, 1.10224298e+00, 1.08523203e+00,
          -1.80910582e-02, -4.11852402e-02, 8.31650559e-01,
          5.53474333e-01],
         [1.08809093e+00, 1.08724164e+00, 1.08618285e+00,
          8.77860003e-03, 5.39500002e-03, 8.38767006e-01,
          5.44393002e-01]]
        tj_d = [[0.926, 1.452, 1.09, 0.00874023, 0.00537267,
          0.838907, 0.544179],
         [0.934417, 1.42573, 1.091, -0.0131335, -0.0635549,
          0.835977, 0.544913],
         [0.954224, 1.40735, 1.08852, -0.0567584, -0.119802,
          0.822665, 0.552854],
         [0.952468, 1.42154, 1.08114, -0.123649, -0.165218,
          0.822171, 0.530518],
         [0.94678, 1.42172, 1.08695, -0.177465, -0.201655,
          0.773756, 0.57371],
         [0.954886, 1.40678, 1.08267, -0.198966, -0.244486,
          0.721802, 0.61615],
         [0.976322, 1.40998, 1.07852, -0.148351, -0.284487,
          0.691303, 0.647425],
         [0.97628, 1.39776, 1.069, -0.130073, -0.36323,
          0.673449, 0.630564],
         [0.991388, 1.36989, 1.05834, -0.0815189, -0.401022,
          0.650042, 0.640298],
         [1.0013, 1.34869, 1.06703, -0.0100392, -0.411914,
          0.634998, 0.653455],
         [0.995316, 1.34314, 1.05683, 0.0447232, -0.384631,
          0.597312, 0.702337],
         [0.980561, 1.34417, 1.07223, 0.112821, -0.403662,
          0.611584, 0.671039],
         [0.992509, 1.32086, 1.07081, 0.183947, -0.410352,
          0.60958, 0.65283],
         [0.993872, 1.29281, 1.05919, 0.207569, -0.466283,
          0.58175, 0.633294],
         [0.987282, 1.25994, 1.07442, 0.226461, -0.514081,
          0.547369, 0.620341],
         [1.00324, 1.24732, 1.08055, 0.244992, -0.556498,
          0.484288, 0.62909],
         [1.0057933, 1.24250299, 1.0807195, 0.23919388, -0.54249395,
          0.49985485, 0.6313729],
         [1.00834659, 1.23768599, 1.080889, 0.23328162, -0.52823103,
          0.51518318, 0.63335453],
         [1.01089989, 1.23286898, 1.0810585, 0.22725803, -0.51371601,
          0.53026562, 0.63503387],
         [1.01345318, 1.22805198, 1.081228, 0.22112597, -0.4989558,
          0.54509499, 0.63641013],
         [1.01600648, 1.22323497, 1.08139751, 0.21488838, -0.48395746,
          0.5596642, 0.63748266],
         [1.01855977, 1.21841797, 1.08156701, 0.20854822, -0.46872814,
          0.5739663, 0.63825093],
         [1.02111307, 1.21360096, 1.08173651, 0.20210854, -0.45327512,
          0.58799447, 0.63871458],
         [1.02366637, 1.20878396, 1.08190601, 0.19557239, -0.43760575,
          0.601742, 0.6388734],
         [1.02621966, 1.20396695, 1.08207551, 0.1889429, -0.42172753,
          0.61520233, 0.63872729],
         [1.02877296, 1.19914995, 1.08224501, 0.18222323, -0.40564803,
          0.62836905, 0.63827635],
         [1.03132625, 1.19433294, 1.08241451, 0.1754166, -0.38937493,
          0.64123587, 0.63752077],
         [1.03387955, 1.18951593, 1.08258401, 0.16852624, -0.37291599,
          0.65379665, 0.63646093],
         [1.03643284, 1.18469893, 1.08275352, 0.16155545, -0.35627907,
          0.66604539, 0.63509732],
         [1.03898614, 1.17988192, 1.08292302, 0.15450756, -0.33947211,
          0.67797624, 0.63343059],
         [1.04153944, 1.17506492, 1.08309252, 0.14738592, -0.32250313,
          0.68958352, 0.63146155],
         [1.04409273, 1.17024791, 1.08326202, 0.14019394, -0.30538023,
          0.70086168, 0.62919114],
         [1.04664603, 1.16543091, 1.08343152, 0.13293505, -0.28811158,
          0.71180533, 0.62662042],
         [1.04919932, 1.1606139, 1.08360102, 0.12561272, -0.27070542,
          0.72240927, 0.62375065],
         [1.05175262, 1.1557969, 1.08377052, 0.11823043, -0.25317006,
          0.73266842, 0.62058317],
         [1.05430591, 1.15097989, 1.08394002, 0.11079172, -0.23551387,
          0.74257789, 0.61711951],
         [1.05685921, 1.14616288, 1.08410953, 0.10330012, -0.21774528,
          0.75213295, 0.61336131],
         [1.0594125, 1.14134588, 1.08427903, 0.09575923, -0.19987277,
          0.76132904, 0.60931038],
         [1.0619658, 1.13652887, 1.08444853, 0.08817263, -0.18190486,
          0.77016177, 0.60496864],
         [1.0645191, 1.13171187, 1.08461803, 0.08054395, -0.16385013,
          0.77862693, 0.60033817],
         [1.06707239, 1.12689486, 1.08478753, 0.07287683, -0.1457172,
          0.78672047, 0.59542117],
         [1.06962569, 1.12207786, 1.08495703, 0.06517493, -0.12751473,
          0.79443853, 0.59022],
         [1.07217898, 1.11726085, 1.08512653, 0.05744192, -0.1092514,
          0.80177743, 0.58473713],
         [1.07473228, 1.11244385, 1.08529603, 0.0496815, -0.09093592,
          0.80873367, 0.57897519],
         [1.07728557, 1.10762684, 1.08546553, 0.04189736, -0.07257705,
          0.81530392, 0.57293691],
         [1.07983887, 1.10280984, 1.08563504, 0.03409323, -0.05418353,
          0.82148505, 0.5666252],
         [1.08239217, 1.09799283, 1.08580454, 0.02627283, -0.03576416,
          0.82727411, 0.56004304],
         [1.08494546, 1.09317582, 1.08597404, 0.01843988, -0.01732771,
          0.83266834, 0.5531936],
         [1.08809093, 1.08724164, 1.08618285, 0.0087786, 0.005395,
          0.83876701, 0.544393]]
        tj_e =  [[9.25000000e-01, 1.45200000e+00, 1.09000000e+00,
          8.73908000e-03, 5.37519000e-03, 8.39040000e-01,
          5.43974000e-01],
         [9.45218000e-01, 1.44351000e+00, 1.09350000e+00,
          1.38044000e-03, 1.04140000e-02, 8.78532000e-01,
          4.77569000e-01],
         [9.34512000e-01, 1.42300000e+00, 1.09642000e+00,
          -3.64166000e-02, -4.60855000e-02, 8.93404000e-01,
          4.45397000e-01],
         [9.79882000e-01, 1.42267000e+00, 1.09173000e+00,
          -6.15065000e-02, -8.01426000e-02, 9.04997000e-01,
          4.13248000e-01],
         [9.90893000e-01, 1.43586000e+00, 1.08069000e+00,
          -4.96424000e-02, -1.57931000e-01, 9.00299000e-01,
          4.02562000e-01],
         [9.76139000e-01, 1.41524000e+00, 1.09356000e+00,
          -9.04090000e-02, -1.98394000e-01, 9.06951000e-01,
          3.60424000e-01],
         [9.88547000e-01, 1.41948000e+00, 1.08228000e+00,
          -8.42892000e-02, -1.89064000e-01, 9.36238000e-01,
          2.83918000e-01],
         [1.00604000e+00, 1.40300000e+00, 1.08842000e+00,
          -1.38538000e-01, -2.08858000e-01, 9.38913000e-01,
          2.35855000e-01],
         [1.02868000e+00, 1.39075000e+00, 1.09280000e+00,
          -9.33327000e-02, -2.18361000e-01, 9.54505000e-01,
          1.80354000e-01],
         [1.01027000e+00, 1.40117000e+00, 1.08323000e+00,
          -4.82024000e-02, -2.12637000e-01, 9.68520000e-01,
          1.20131000e-01],
         [1.01310000e+00, 1.37655000e+00, 1.09170000e+00,
          1.65175000e-02, -2.22059000e-01, 9.71089000e-01,
          8.60425000e-02],
         [1.02789000e+00, 1.37835000e+00, 1.09802000e+00,
          -9.56434000e-03, -2.89088000e-01, 9.56161000e-01,
          4.57408000e-02],
         [1.02980000e+00, 1.35273000e+00, 1.09862000e+00,
          -4.71633000e-02, -3.37578000e-01, 9.40089000e-01,
          7.07758000e-03],
         [1.03874000e+00, 1.33091000e+00, 1.07752000e+00,
          -3.32602000e-02, -3.95407000e-01, 9.17348000e-01,
          3.19133000e-02],
         [1.04011000e+00, 1.31970000e+00, 1.07103000e+00,
          1.50725000e-03, -4.39105000e-01, 8.97871000e-01,
          -3.18188000e-02],
         [1.01677000e+00, 1.35936000e+00, 1.04596000e+00,
          4.31263000e-02, -4.55203000e-01, 8.88192000e-01,
          -4.52367000e-02],
         [1.00530000e+00, 1.35244000e+00, 1.05503000e+00,
          9.96332000e-02, -4.80630000e-01, 8.65825000e-01,
          -9.70284000e-02],
         [1.00007000e+00, 1.29824000e+00, 1.07180000e+00,
          8.55418000e-02, -5.15200000e-01, 8.46230000e-01,
          -1.05576000e-01],
         [1.02846000e+00, 1.27150000e+00, 1.09373000e+00,
          5.70754000e-02, -5.35239000e-01, 8.30142000e-01,
          -1.45347000e-01],
         [1.01010000e+00, 1.27794000e+00, 1.10640000e+00,
          8.14397000e-02, -5.12199000e-01, 8.27690000e-01,
          -2.14357000e-01],
         [1.01214316e+00, 1.27294420e+00, 1.10587036e+00,
          8.02457991e-02, -5.02831287e-01, 8.38546158e-01,
          -1.93808282e-01],
         [1.01418631e+00, 1.26794841e+00, 1.10534073e+00,
          7.90013966e-02, -4.93147123e-01, 8.48874589e-01,
          -1.73137592e-01],
         [1.01622947e+00, 1.26295261e+00, 1.10481109e+00,
          7.77072796e-02, -4.83152630e-01, 8.58668836e-01,
          -1.52357949e-01],
         [1.01827263e+00, 1.25795682e+00, 1.10428146e+00,
          7.63642626e-02, -4.72854095e-01, 8.67922735e-01,
          -1.31482429e-01],
         [1.02031579e+00, 1.25296102e+00, 1.10375182e+00,
          7.49731907e-02, -4.62258001e-01, 8.76630463e-01,
          -1.10524170e-01],
         [1.02235894e+00, 1.24796523e+00, 1.10322218e+00,
          7.35349393e-02, -4.51371013e-01, 8.84786541e-01,
          -8.94963593e-02],
         [1.02440210e+00, 1.24296943e+00, 1.10269255e+00,
          7.20504133e-02, -4.40199985e-01, 8.92385835e-01,
          -6.84122299e-02],
         [1.02644526e+00, 1.23797364e+00, 1.10216291e+00,
          7.05205471e-02, -4.28751946e-01, 8.99423563e-01,
          -4.72850496e-02],
         [1.02848841e+00, 1.23297784e+00, 1.10163327e+00,
          6.89463034e-02, -4.17034099e-01, 9.05895297e-01,
          -2.61281135e-02],
         [1.03053157e+00, 1.22798205e+00, 1.10110364e+00,
          6.73286727e-02, -4.05053818e-01, 9.11796965e-01,
          -4.95473537e-03],
         [1.03257473e+00, 1.22298625e+00, 1.10057400e+00,
          6.56686731e-02, -3.92818643e-01, 9.17124852e-01,
          1.62217607e-02],
         [1.03461788e+00, 1.21799046e+00, 1.10004437e+00,
          6.39673491e-02, -3.80336273e-01, 9.21875605e-01,
          3.73880486e-02],
         [1.03666104e+00, 1.21299466e+00, 1.09951473e+00,
          6.22257714e-02, -3.67614563e-01, 9.26046236e-01,
          5.85308088e-02],
         [1.03870420e+00, 1.20799887e+00, 1.09898509e+00,
          6.04450359e-02, -3.54661518e-01, 9.29634119e-01,
          7.96367364e-02],
         [1.04074736e+00, 1.20300307e+00, 1.09845546e+00,
          5.86262633e-02, -3.41485290e-01, 9.32636997e-01,
          1.00692550e-01],
         [1.04279051e+00, 1.19800728e+00, 1.09792582e+00,
          5.67705980e-02, -3.28094170e-01, 9.35052980e-01,
          1.21684999e-01],
         [1.04483367e+00, 1.19301148e+00, 1.09739619e+00,
          5.48792078e-02, -3.14496585e-01, 9.36880548e-01,
          1.42600873e-01],
         [1.04687683e+00, 1.18801569e+00, 1.09686655e+00,
          5.29532829e-02, -3.00701093e-01, 9.38118551e-01,
          1.63427011e-01],
         [1.04891998e+00, 1.18301989e+00, 1.09633691e+00,
          5.09940352e-02, -2.86716373e-01, 9.38766209e-01,
          1.84150306e-01],
         [1.05096314e+00, 1.17802410e+00, 1.09580728e+00,
          4.90026978e-02, -2.72551227e-01, 9.38823115e-01,
          2.04757719e-01],
         [1.05300630e+00, 1.17302830e+00, 1.09527764e+00,
          4.69805236e-02, -2.58214568e-01, 9.38289233e-01,
          2.25236280e-01],
         [1.05504946e+00, 1.16803250e+00, 1.09474801e+00,
          4.49287854e-02, -2.43715419e-01, 9.37164900e-01,
          2.45573104e-01],
         [1.05709261e+00, 1.16303671e+00, 1.09421837e+00,
          4.28487740e-02, -2.29062903e-01, 9.35450822e-01,
          2.65755391e-01],
         [1.05913577e+00, 1.15804091e+00, 1.09368873e+00,
          4.07417986e-02, -2.14266241e-01, 9.33148079e-01,
          2.85770443e-01],
         [1.06117893e+00, 1.15304512e+00, 1.09315910e+00,
          3.86091849e-02, -1.99334745e-01, 9.30258119e-01,
          3.05605664e-01],
         [1.06322208e+00, 1.14804932e+00, 1.09262946e+00,
          3.64522751e-02, -1.84277810e-01, 9.26782762e-01,
          3.25248571e-01],
         [1.06526524e+00, 1.14305353e+00, 1.09209982e+00,
          3.42724264e-02, -1.69104912e-01, 9.22724193e-01,
          3.44686804e-01],
         [1.06730840e+00, 1.13805773e+00, 1.09157019e+00,
          3.20710105e-02, -1.53825598e-01, 9.18084968e-01,
          3.63908131e-01],
         [1.06935155e+00, 1.13306194e+00, 1.09104055e+00,
          2.98494128e-02, -1.38449485e-01, 9.12868004e-01,
          3.82900456e-01],
         [1.07139471e+00, 1.12806614e+00, 1.09051092e+00,
          2.76090313e-02, -1.22986247e-01, 9.07076587e-01,
          4.01651827e-01],
         [1.07343787e+00, 1.12307035e+00, 1.08998128e+00,
          2.53512758e-02, -1.07445615e-01, 9.00714359e-01,
          4.20150444e-01],
         [1.07548103e+00, 1.11807455e+00, 1.08945164e+00,
          2.30775671e-02, -9.18373697e-02, 8.93785325e-01,
          4.38384667e-01],
         [1.07752418e+00, 1.11307876e+00, 1.08892201e+00,
          2.07893361e-02, -7.61713324e-02, 8.86293845e-01,
          4.56343021e-01],
         [1.07956734e+00, 1.10808296e+00, 1.08839237e+00,
          1.84880227e-02, -6.04573616e-02, 8.78244632e-01,
          4.74014205e-01],
         [1.08161050e+00, 1.10308717e+00, 1.08786274e+00,
          1.61750750e-02, -4.47053458e-02, 8.69642754e-01,
          4.91387098e-01],
         [1.08365365e+00, 1.09809137e+00, 1.08733310e+00,
          1.38519485e-02, -2.89251976e-02, 8.60493621e-01,
          5.08450769e-01],
         [1.08569681e+00, 1.09309558e+00, 1.08680346e+00,
          1.15201052e-02, -1.31268472e-02, 8.50802993e-01,
          5.25194479e-01],
         [1.08809093e+00, 1.08724164e+00, 1.08618285e+00,
          8.77860003e-03, 5.39500002e-03, 8.38767006e-01,
          5.44393002e-01]]
        # tj_f
        tj_f = [[9.26000000e-01, 1.45200000e+00, 1.09000000e+00,
          8.74282000e-03, 5.36897000e-03, 8.38665000e-01,
          5.44552000e-01],
         [9.42919000e-01, 1.43145000e+00, 1.09072000e+00,
          2.99493000e-02, -5.37885000e-02, 8.15961000e-01,
          5.74820000e-01],
         [9.34537000e-01, 1.40521000e+00, 1.08192000e+00,
          6.44541000e-02, -1.05795000e-01, 8.29984000e-01,
          5.43857000e-01],
         [9.39226000e-01, 1.40183000e+00, 1.08127000e+00,
          2.37045000e-02, -1.39873000e-01, 8.67920000e-01,
          4.76012000e-01],
         [9.38080000e-01, 1.40384000e+00, 1.07867000e+00,
          1.43779000e-03, -2.13522000e-01, 8.82240000e-01,
          4.19594000e-01],
         [9.57440000e-01, 1.39960000e+00, 1.08492000e+00,
          -5.73969000e-02, -2.60613000e-01, 8.78854000e-01,
          3.95477000e-01],
         [9.66622000e-01, 1.41092000e+00, 1.09350000e+00,
          -1.14171000e-01, -2.85110000e-01, 8.88532000e-01,
          3.40863000e-01],
         [9.58925000e-01, 1.39686000e+00, 1.09456000e+00,
          -8.37916000e-02, -3.38186000e-01, 8.93405000e-01,
          2.83614000e-01],
         [9.67917000e-01, 1.41247000e+00, 1.08663000e+00,
          -2.14284000e-02, -3.85116000e-01, 8.82836000e-01,
          2.68004000e-01],
         [9.93246000e-01, 1.40149000e+00, 1.07521000e+00,
          4.84630000e-02, -3.89723000e-01, 8.79865000e-01,
          2.67592000e-01],
         [9.88162000e-01, 1.38028000e+00, 1.07719000e+00,
          1.20172000e-01, -3.98090000e-01, 8.77766000e-01,
          2.37928000e-01],
         [9.75234000e-01, 1.35021000e+00, 1.08274000e+00,
          1.30775000e-01, -3.50880000e-01, 9.05204000e-01,
          2.00964000e-01],
         [9.92500000e-01, 1.33536000e+00, 1.08402000e+00,
          1.03926000e-01, -3.90452000e-01, 9.03899000e-01,
          1.40407000e-01],
         [9.84972000e-01, 1.30399000e+00, 1.09613000e+00,
          7.90666000e-02, -3.89318000e-01, 9.14152000e-01,
          8.06636000e-02],
         [9.75530000e-01, 1.28186000e+00, 1.08583000e+00,
          1.22748000e-01, -4.30319000e-01, 8.93261000e-01,
          4.29423000e-02],
         [9.88813000e-01, 1.27538000e+00, 1.09300000e+00,
          1.51026000e-01, -4.48783000e-01, 8.80192000e-01,
          -3.23512000e-02],
         [9.75873000e-01, 1.24634000e+00, 1.09606000e+00,
          9.14500000e-02, -4.41835000e-01, 8.90204000e-01,
          -6.28873000e-02],
         [9.91458000e-01, 1.20413000e+00, 1.11909000e+00,
          5.00678000e-02, -4.21383000e-01, 9.04020000e-01,
          -5.17506000e-02],
         [9.94225898e-01, 1.20078192e+00, 1.11814743e+00,
          4.91825225e-02, -4.11430678e-01, 9.09501102e-01,
          -3.33709673e-02],
         [9.96993796e-01, 1.19743383e+00, 1.11720485e+00,
          4.82742607e-02, -4.01286084e-01, 9.14557161e-01,
          -1.49757457e-02],
         [9.99761694e-01, 1.19408575e+00, 1.11626228e+00,
          4.73434217e-02, -3.90953814e-01, 9.19185495e-01,
          3.42647981e-03],
         [1.00252959e+00, 1.19073767e+00, 1.11531971e+00,
          4.63904408e-02, -3.80438701e-01, 9.23383939e-01,
          2.18271028e-02],
         [1.00529749e+00, 1.18738958e+00, 1.11437713e+00,
          4.54157638e-02, -3.69745662e-01, 9.27150530e-01,
          4.02175176e-02],
         [1.00806539e+00, 1.18404150e+00, 1.11343456e+00,
          4.44198465e-02, -3.58879698e-01, 9.30483505e-01,
          5.85891233e-02],
         [1.01083329e+00, 1.18069342e+00, 1.11249199e+00,
          4.34031547e-02, -3.47845892e-01, 9.33381307e-01,
          7.69333277e-02],
         [1.01360118e+00, 1.17734534e+00, 1.11154941e+00,
          4.23661639e-02, -3.36649403e-01, 9.35842580e-01,
          9.52415514e-02],
         [1.01636908e+00, 1.17399725e+00, 1.11060684e+00,
          4.13093590e-02, -3.25295467e-01, 9.37866173e-01,
          1.13505232e-01],
         [1.01913698e+00, 1.17064917e+00, 1.10966426e+00,
          4.02332343e-02, -3.13789396e-01, 9.39451139e-01,
          1.31715828e-01],
         [1.02190488e+00, 1.16730109e+00, 1.10872169e+00,
          3.91382931e-02, -3.02136570e-01, 9.40596737e-01,
          1.49864822e-01],
         [1.02467278e+00, 1.16395300e+00, 1.10777912e+00,
          3.80250474e-02, -2.90342439e-01, 9.41302432e-01,
          1.67943727e-01],
         [1.02744068e+00, 1.16060492e+00, 1.10683654e+00,
          3.68940181e-02, -2.78412519e-01, 9.41567893e-01,
          1.85944087e-01],
         [1.03020857e+00, 1.15725684e+00, 1.10589397e+00,
          3.57457339e-02, -2.66352390e-01, 9.41392996e-01,
          2.03857483e-01],
         [1.03297647e+00, 1.15390875e+00, 1.10495140e+00,
          3.45807319e-02, -2.54167691e-01, 9.40777824e-01,
          2.21675538e-01],
         [1.03574437e+00, 1.15056067e+00, 1.10400882e+00,
          3.33995570e-02, -2.41864122e-01, 9.39722663e-01,
          2.39389919e-01],
         [1.03851227e+00, 1.14721259e+00, 1.10306625e+00,
          3.22027617e-02, -2.29447436e-01, 9.38228007e-01,
          2.56992340e-01],
         [1.04128017e+00, 1.14386451e+00, 1.10212368e+00,
          3.09909056e-02, -2.16923441e-01, 9.36294556e-01,
          2.74474570e-01],
         [1.04404806e+00, 1.14051642e+00, 1.10118110e+00,
          2.97645554e-02, -2.04297994e-01, 9.33923213e-01,
          2.91828432e-01],
         [1.04681596e+00, 1.13716834e+00, 1.10023853e+00,
          2.85242848e-02, -1.91577000e-01, 9.31115087e-01,
          3.09045810e-01],
         [1.04958386e+00, 1.13382026e+00, 1.09929596e+00,
          2.72706739e-02, -1.78766408e-01, 9.27871492e-01,
          3.26118652e-01],
         [1.05235176e+00, 1.13047217e+00, 1.09835338e+00,
          2.60043088e-02, -1.65872210e-01, 9.24193946e-01,
          3.43038973e-01],
         [1.05511966e+00, 1.12712409e+00, 1.09741081e+00,
          2.47257818e-02, -1.52900435e-01, 9.20084167e-01,
          3.59798859e-01],
         [1.05788755e+00, 1.12377601e+00, 1.09646824e+00,
          2.34356910e-02, -1.39857151e-01, 9.15544077e-01,
          3.76390473e-01],
         [1.06065545e+00, 1.12042792e+00, 1.09552566e+00,
          2.21346397e-02, -1.26748458e-01, 9.10575801e-01,
          3.92806054e-01],
         [1.06342335e+00, 1.11707984e+00, 1.09458309e+00,
          2.08232362e-02, -1.13580487e-01, 9.05181662e-01,
          4.09037926e-01],
         [1.06619125e+00, 1.11373176e+00, 1.09364051e+00,
          1.95020941e-02, -1.00359395e-01, 8.99364182e-01,
          4.25078496e-01],
         [1.06895915e+00, 1.11038367e+00, 1.09269794e+00,
          1.81718311e-02, -8.70913673e-02, 8.93126083e-01,
          4.40920263e-01],
         [1.07172705e+00, 1.10703559e+00, 1.09175537e+00,
          1.68330694e-02, -7.37826079e-02, 8.86470281e-01,
          4.56555818e-01],
         [1.07449494e+00, 1.10368751e+00, 1.09081279e+00,
          1.54864351e-02, -6.04393414e-02, 8.79399890e-01,
          4.71977849e-01],
         [1.07726284e+00, 1.10033943e+00, 1.08987022e+00,
          1.41325581e-02, -4.70678082e-02, 8.71918216e-01,
          4.87179142e-01],
         [1.08003074e+00, 1.09699134e+00, 1.08892765e+00,
          1.27720714e-02, -3.36742622e-02, 8.64028759e-01,
          5.02152588e-01],
         [1.08279864e+00, 1.09364326e+00, 1.08798507e+00,
          1.14056115e-02, -2.02649672e-02, 8.55735208e-01,
          5.16891185e-01],
         [1.08809093e+00, 1.08724164e+00, 1.08618285e+00,
          8.77860003e-03, 5.39500002e-03, 8.38767006e-01,
          5.44393002e-01]]
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
                pass
            else:
                poses.append(base_pose)
        # FIXME collision avoidance needs soft_threshholds at 0
        cereal_name = u'cereal'
        drawer_frame_id = u'iai_kitchen/oven_area_area_right_drawer_board_2_link'

        # take milk out of fridge
        kitchen_setup_avoid_collisions.set_kitchen_js({u'oven_area_area_right_drawer_main_joint': 0.48})

        # spawn milk
        cereal_pose = PoseStamped()
        cereal_pose.header.frame_id = drawer_frame_id
        cereal_pose.pose.position = Point(0.123, 0.0, 0.165) # big box: Point(0.123, 0.0, 0.165)
        cereal_pose.pose.orientation = Quaternion(0.0087786, 0.005395, -0.838767, -0.544393)
        kitchen_setup_avoid_collisions.add_box(cereal_name, [0.1028, 0.0634, 0.26894], cereal_pose) #big box [0.1028, 0.0634, 0.26894]

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
        #kitchen_setup_avoid_collisions.god_map.set_data(identifier.rosparam + ['reset_god_map'], False)
        #decrease_external_collision_avoidance(kitchen_setup_avoid_collisions)
        #kitchen_setup_avoid_collisions.allow_all_collisions()
        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPathCarrot',
                                                     tip_link=kitchen_setup_avoid_collisions.r_tip,
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     goal=post_grasp_pose,
                                                     goals=poses,
                                                     predict_f=2.0)
        kitchen_setup_avoid_collisions.plan_and_execute()
        #kitchen_setup.set_joint_goal(gaya_pose)

        # place milk back

        # kitchen_setup.add_json_goal(u'BasePointingForward')
        # milk_goal = PoseStamped()
        # milk_goal.header.frame_id = u'iai_kitchen/kitchen_island_surface'
        # milk_goal.pose.position = Point(.1, -.2, .13)
        # milk_goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0,0,1]))

    def test_ease_cereal_with_planner_1(self, kitchen_setup_avoid_collisions):
        # FIXME collision avoidance needs soft_threshholds at 0
        cereal_name = u'cereal'
        drawer_frame_id = u'iai_kitchen/oven_area_area_right_drawer_board_2_link'

        # take milk out of fridge
        kitchen_setup_avoid_collisions.set_kitchen_js({u'oven_area_area_right_drawer_main_joint': 0.48})

        # spawn milk
        cereal_pose = PoseStamped()
        cereal_pose.header.frame_id = drawer_frame_id
        cereal_pose.pose.position = Point(0.123, 0.0, 0.165) # big box: Point(0.123, 0.0, 0.165)
        cereal_pose.pose.orientation = Quaternion(0.0087786, 0.005395, -0.838767, -0.544393)
        kitchen_setup_avoid_collisions.add_box(cereal_name, [0.1028, 0.0634, 0.26894], cereal_pose) #big box [0.1028, 0.0634, 0.26894]

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
        #kitchen_setup_avoid_collisions.god_map.set_data(identifier.rosparam + ['reset_god_map'], False)
        #decrease_external_collision_avoidance(kitchen_setup_avoid_collisions)
        kitchen_setup_avoid_collisions.set_json_goal(u'CartesianPathCarrot',
                                                     tip_link=kitchen_setup_avoid_collisions.r_tip,
                                                     root_link=kitchen_setup_avoid_collisions.default_root,
                                                     goal=post_grasp_pose)
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
        #kitchen_setup_avoid_collisions.god_map.set_data(identifier.rosparam + ['reset_god_map'], True)

        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        kitchen_setup_avoid_collisions.open_l_gripper()
        kitchen_setup_avoid_collisions.detach_object(cereal_name)

        kitchen_setup_avoid_collisions.set_joint_goal(gaya_pose)
        kitchen_setup_avoid_collisions.plan_and_execute()

    def test_ease_cereal_with_planner_2(self, kitchen_setup_avoid_collisions):
        from tf.transformations import quaternion_about_axis
        # FIXME collision avoidance needs soft_threshholds at 0
        cereal_name = u'cereal'
        drawer_frame_id = u'iai_kitchen/oven_area_area_right_drawer_board_2_link'

        # take milk out of fridge
        kitchen_setup_avoid_collisions.set_kitchen_js({u'oven_area_area_right_drawer_main_joint': 0.48})

        # spawn milk

        cereal_pose = PoseStamped()
        cereal_pose.header.frame_id = drawer_frame_id
        cereal_pose.pose.position = Point(-0.08, 0.0, 0.15)
        cereal_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 2 - np.pi / 6, [0, 0, 1]))
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
        box_T_r_goal_post.p[1] += 0.15
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
        #kitchen_setup_avoid_collisions.god_map.set_data(identifier.rosparam + ['reset_god_map'], False)
        #decrease_external_collision_avoidance(kitchen_setup_avoid_collisions)
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
        #kitchen_setup_avoid_collisions.god_map.set_data(identifier.rosparam + ['reset_god_map'], True)

        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        kitchen_setup_avoid_collisions.open_l_gripper()
        kitchen_setup_avoid_collisions.detach_object(cereal_name)

        kitchen_setup_avoid_collisions.set_joint_goal(gaya_pose)
        kitchen_setup_avoid_collisions.plan_and_execute()

    def test_ease_cereal_different_drawers(self, kitchen_setup_avoid_collisions):
        # FIXME collision avoidance needs soft_threshholds at 0
        cereal_name = u'cereal'
        drawer_frame_id = 'iai_kitchen/oven_area_area_right_drawer_board_1_link'

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
        box_T_r_goal_post.p[0] += 0.4
        if drawer_frame_id == 'iai_kitchen/oven_area_area_right_drawer_board_1_link':
            box_T_r_goal_post.p[1] -= 0.15
            box_T_r_goal_post.p[2] += 0.2
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
        kitchen_setup_avoid_collisions.god_map.set_data(identifier.rosparam + ['reset_god_map'], True)

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
