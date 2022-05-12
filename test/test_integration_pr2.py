from __future__ import division

import re
import itertools
from copy import deepcopy

import numpy as np
import pytest
import roslaunch
import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Vector3Stamped, PointStamped, Transform
from giskard_msgs.msg import CollisionEntry, MoveActionGoal, MoveResult, WorldBody, MoveGoal, MoveCmd
from giskard_msgs.srv import UpdateWorldResponse, UpdateWorldRequest
from numpy import pi
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from tf.transformations import quaternion_from_matrix, quaternion_about_axis

import giskardpy.tfwrapper as tf
from giskardpy import logging, identifier
from giskardpy.constraints import WEIGHT_ABOVE_CA, WEIGHT_BELOW_CA, WEIGHT_COLLISION_AVOIDANCE
from giskardpy.identifier import fk_pose
from giskardpy.robot import Robot
from giskardpy.tfwrapper import init as tf_init
from giskardpy.utils import to_joint_state_position_dict, publish_marker_vector
from rospy_message_converter.message_converter import convert_dictionary_to_ros_message
from utils_for_tests import PR2, compare_poses

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

folder_name = u'tmp_data/'


@pytest.fixture(scope=u'module')
def ros(request):
    try:
        logging.loginfo(u'deleting tmp test folder')
        # shutil.rmtree(folder_name)
    except Exception:
        pass

    logging.loginfo(u'init ros')
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
    resetted_giskard.send_and_check_joint_goal(default_pose)
    return resetted_giskard


@pytest.fixture()
def pocky_pose_setup(resetted_giskard):
    resetted_giskard.set_joint_goal(pocky_pose)
    resetted_giskard.allow_all_collisions()
    resetted_giskard.send_and_check_goal()
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
def kitchen_setup(resetted_giskard):
    """
    :type resetted_giskard: GiskardTestWrapper
    :return:
    """
    resetted_giskard.allow_all_collisions()
    resetted_giskard.send_and_check_joint_goal(gaya_pose)
    object_name = u'kitchen'
    resetted_giskard.add_urdf(object_name, rospy.get_param(u'kitchen_description'),
                              tf.lookup_pose(u'map', u'iai_kitchen/world'), u'/kitchen/joint_states',
                              set_js_topic=u'/kitchen/cram_joint_states')
    js = {k: 0.0 for k in resetted_giskard.get_world().get_object(object_name).get_movable_joints()}
    resetted_giskard.set_kitchen_js(js)
    return resetted_giskard


class TestInitialization(object):
    def test_load_config_yaml(self, zero_pose):
        gm = zero_pose.get_god_map()
        robot = zero_pose.get_robot()
        assert isinstance(robot, Robot)
        sample_period = gm.unsafe_get_data(identifier.sample_period)
        odom_x_index = gm.unsafe_get_data(identifier.b_keys).index(u'j -- (\'pr2\', \'odom_x_joint\')')
        odom_x_lb = gm.unsafe_get_data(identifier.lb)[odom_x_index] / sample_period
        odom_x_ub = gm.unsafe_get_data(identifier.ub)[odom_x_index] / sample_period
        np.testing.assert_almost_equal(odom_x_lb, -0.5)
        np.testing.assert_almost_equal(odom_x_ub, 0.5)

        odom_z_index = gm.unsafe_get_data(identifier.b_keys).index(u'j -- (\'pr2\', \'odom_z_joint\')')
        odom_z_lb = gm.unsafe_get_data(identifier.lb)[odom_z_index] / sample_period
        odom_z_ub = gm.unsafe_get_data(identifier.ub)[odom_z_index] / sample_period
        np.testing.assert_almost_equal(odom_z_lb, -0.6)
        np.testing.assert_almost_equal(odom_z_ub, 0.6)


class TestFk(object):
    def test_fk1(self, zero_pose):
        for root, tip in itertools.product(zero_pose.get_robot().get_link_names(), repeat=2):
            fk1 = zero_pose.get_god_map().get_data(fk_pose + [(root, tip)])
            fk2 = tf.lookup_pose(root, tip)
            compare_poses(fk1.pose, fk2.pose)

    def test_fk2(self, zero_pose):
        pocky = u'box'
        zero_pose.attach_box(pocky, [0.1, 0.02, 0.02], zero_pose.r_tip, [0.05, 0, 0], [1, 0, 0, 0])
        for root, tip in itertools.product(zero_pose.get_robot().get_link_names(), [pocky]):
            fk1 = zero_pose.get_god_map().get_data(fk_pose + [(root, tip)])
            fk2 = tf.lookup_pose(root, tip)
            compare_poses(fk1.pose, fk2.pose)


class TestJointGoals(object):
    def test_move_base(self, zero_pose):
        p = PoseStamped()
        p.header.frame_id = u'map'
        p.pose.position.y = -1
        p.pose.orientation = Quaternion(0, 0, 0.47942554, 0.87758256)
        zero_pose.move_base(p)

    def test_joint_movement1(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.allow_self_collision()
        # zero_pose.send_and_check_goal()
        zero_pose.send_and_check_joint_goal(pocky_pose)

    def test_partial_joint_state_goal1(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.allow_self_collision()
        js = dict(pocky_pose.items()[:3])
        zero_pose.send_and_check_joint_goal(js)

    def test_continuous_joint1(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.allow_self_collision()
        js = {u'r_wrist_roll_joint': -pi,
              u'l_wrist_roll_joint': -2.1 * pi, }
        zero_pose.send_and_check_joint_goal(js)

    def test_undefined_type(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.allow_self_collision()
        goal = MoveActionGoal()
        goal.goal.type = MoveGoal.UNDEFINED
        result = zero_pose.send_goal(goal)
        assert result.error_codes[0] == MoveResult.INVALID_GOAL

    def test_empty_goal(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.allow_self_collision()
        goal = MoveActionGoal()
        goal.goal.type = MoveGoal.PLAN_AND_EXECUTE
        result = zero_pose.send_goal(goal)
        assert result.error_codes[0] == MoveResult.INVALID_GOAL

    def test_plan_only(self, zero_pose):
        zero_pose.allow_self_collision()
        zero_pose.set_joint_goal(pocky_pose)
        zero_pose.send_and_check_goal(goal_type=MoveGoal.PLAN_ONLY)
        zero_pose.check_current_joint_state(default_pose)

    def test_prismatic_joint1(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.allow_self_collision()
        js = {u'torso_lift_joint': 0.1}
        zero_pose.send_and_check_joint_goal(js)
        js = {u'torso_lift_joint': 0.32}
        zero_pose.send_and_check_joint_goal(js)

    def test_hard_joint_limits(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.allow_self_collision()
        r_elbow_flex_joint_limits = zero_pose.get_robot().get_joint_limits('r_elbow_flex_joint')
        torso_lift_joint_limits = zero_pose.get_robot().get_joint_limits('torso_lift_joint')
        head_pan_joint_limits = zero_pose.get_robot().get_joint_limits('head_pan_joint')

        goal_js = {u'r_elbow_flex_joint': r_elbow_flex_joint_limits[0] - 0.2,
                   u'torso_lift_joint': torso_lift_joint_limits[0] - 0.2,
                   u'head_pan_joint': head_pan_joint_limits[0] - 0.2}
        zero_pose.set_joint_goal(goal_js)
        zero_pose.send_goal()
        assert (not zero_pose.are_joint_limits_violated())

        goal_js = {u'r_elbow_flex_joint': r_elbow_flex_joint_limits[1] + 0.2,
                   u'torso_lift_joint': torso_lift_joint_limits[1] + 0.2,
                   u'head_pan_joint': head_pan_joint_limits[1] + 0.2}

        zero_pose.set_joint_goal(goal_js)
        zero_pose.send_goal()
        assert (not zero_pose.are_joint_limits_violated())

    # TODO test goal for unknown joint


class TestConstraints(object):

    def test_CartesianPoseStraight(self, zero_pose):
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
        # zero_pose.set_straight_translation_goal(goal_position, zero_pose.l_tip)  # FIXME: starts wiggling
        zero_pose.set_straight_cart_goal(goal_position, zero_pose.l_tip)
        zero_pose.send_and_check_goal()

    def test_CartesianVelocityLimit(self, zero_pose):
        linear_velocity = 1
        angular_velocity = 1
        zero_pose.limit_cartesian_velocity(
            root_link=zero_pose.default_root,
            tip_link=u'base_footprint',
            max_linear_velocity=0.1,
            max_angular_velocity=0.2
        )
        goal_position = PoseStamped()
        goal_position.header.frame_id = u'r_gripper_tool_frame'
        goal_position.pose.position.x = 1
        goal_position.pose.position.y = 0
        goal_position.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 4, [0, 0, 1]))

        zero_pose.set_and_check_cart_goal(goal_pose=goal_position,
                                          tip_link=u'r_gripper_tool_frame',
                                          linear_velocity=linear_velocity,
                                          angular_velocity=angular_velocity,
                                          weight=WEIGHT_BELOW_CA
                                          )

    def test_AvoidJointLimits1(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        percentage = 10
        zero_pose.allow_self_collision()
        zero_pose.avoid_joint_limits(percentage=percentage)
        zero_pose.send_and_check_goal()

        joint_non_continuous = [j for j in zero_pose.get_robot().controlled_joints if
                                not zero_pose.get_robot().is_joint_continuous(j)]

        current_joint_state = to_joint_state_position_dict(zero_pose.get_current_joint_state())
        percentage *= 0.99  # if will not reach the exact percentager, because the weight is so low
        for joint in joint_non_continuous:
            position = current_joint_state[joint]
            lower_limit, upper_limit = zero_pose.get_robot().get_joint_limits(joint)
            joint_range = upper_limit - lower_limit
            center = (upper_limit + lower_limit) / 2.
            upper_limit2 = center + joint_range / 2. * (1 - percentage / 100.)
            lower_limit2 = center - joint_range / 2. * (1 - percentage / 100.)
            assert position <= upper_limit2 and position >= lower_limit2

    def test_AvoidJointLimits2(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        percentage = 10
        joints = [j for j in zero_pose.get_robot().controlled_joints if
                  not zero_pose.get_robot().is_joint_continuous(j)]
        goal_state = {j: zero_pose.get_robot().get_joint_limits(j)[1] for j in joints}
        del goal_state[u'odom_x_joint']
        del goal_state[u'odom_y_joint']
        zero_pose.allow_self_collision()
        zero_pose.set_json_goal(u'AvoidJointLimits',
                                percentage=percentage)
        zero_pose.set_joint_goal(goal_state)
        zero_pose.send_and_check_goal()

        zero_pose.allow_self_collision()
        zero_pose.set_json_goal(u'AvoidJointLimits',
                                percentage=percentage)
        zero_pose.send_and_check_goal()

        joint_non_continuous = [j for j in zero_pose.get_robot().controlled_joints if
                                not zero_pose.get_robot().is_joint_continuous(j)]

        current_joint_state = to_joint_state_position_dict(zero_pose.get_current_joint_state())
        percentage *= 0.9  # if will not reach the exact percentage, because the weight is so low
        for joint in joint_non_continuous:
            position = current_joint_state[joint]
            lower_limit, upper_limit = zero_pose.get_robot().get_joint_limits(joint)
            joint_range = upper_limit - lower_limit
            center = (upper_limit + lower_limit) / 2.
            upper_limit2 = center + joint_range / 2. * (1 - percentage / 100.)
            lower_limit2 = center - joint_range / 2. * (1 - percentage / 100.)
            assert position <= upper_limit2 and position >= lower_limit2

    def test_UpdateGodMap(self, pocky_pose_setup):
        """
        :type pocky_pose_setup: PR2
        """
        old_torso_value = pocky_pose_setup.get_god_map().get_data(identifier.joint_weight + [u'torso_lift_joint'])
        old_odom_x_value = pocky_pose_setup.get_god_map().get_data(identifier.joint_weight + [u'odom_x_joint'])

        r_goal = PoseStamped()
        r_goal.header.frame_id = pocky_pose_setup.r_tip
        r_goal.pose.orientation.w = 1
        r_goal.pose.position.x += 0.1
        updates = {
            u'rosparam': {
                u'general_options': {
                    u'joint_weights': {
                        u'odom_x_joint': 1000000,
                        u'odom_y_joint': 1000000,
                        u'odom_z_joint': 1000000
                    }
                }
            }
        }

        old_pose = tf.lookup_pose(u'map', u'base_footprint')

        pocky_pose_setup.update_god_map(updates)
        pocky_pose_setup.set_and_check_cart_goal(r_goal, pocky_pose_setup.r_tip)

        new_pose = tf.lookup_pose(u'map', u'base_footprint')
        compare_poses(old_pose.pose, new_pose.pose)

        assert pocky_pose_setup.get_god_map().unsafe_get_data(identifier.joint_weight + [u'odom_x_joint']) == 1000000
        assert pocky_pose_setup.get_god_map().unsafe_get_data(
            identifier.joint_weight + [u'torso_lift_joint']) == old_torso_value

        updates = {
            u'rosparam': {
                u'general_options': {
                    u'joint_weights': {
                        u'odom_x_joint': 0.0001,
                        u'odom_y_joint': 0.0001,
                        u'odom_z_joint': 0.0001
                    }
                }
            }
        }
        # old_pose = tf.lookup_pose(u'map', u'base_footprint')
        # old_pose.pose.position.x += 0.1
        pocky_pose_setup.update_god_map(updates)
        pocky_pose_setup.set_and_check_cart_goal(r_goal, pocky_pose_setup.r_tip)

        new_pose = tf.lookup_pose(u'map', u'base_footprint')

        # compare_poses(old_pose.pose, new_pose.pose)
        assert new_pose.pose.position.x >= 0.08
        assert pocky_pose_setup.get_god_map().unsafe_get_data(identifier.joint_weight + [u'odom_x_joint']) == 0.0001
        assert pocky_pose_setup.get_god_map().unsafe_get_data(
            identifier.joint_weight + [u'torso_lift_joint']) == old_torso_value
        pocky_pose_setup.send_and_check_goal()
        assert pocky_pose_setup.get_god_map().unsafe_get_data(
            identifier.joint_weight + [u'odom_x_joint']) == old_odom_x_value
        assert pocky_pose_setup.get_god_map().unsafe_get_data(
            identifier.joint_weight + [u'torso_lift_joint']) == old_torso_value

    def test_base_pointing_forward(self, pocky_pose_setup):
        """
        :param pocky_pose_setup: PR2
        """
        # FIXME idk
        pocky_pose_setup.set_json_goal(u'BasePointingForward')
        r_goal = PoseStamped()
        r_goal.header.frame_id = pocky_pose_setup.r_tip
        r_goal.pose.position.y = -2
        r_goal.pose.orientation.w = 1
        pocky_pose_setup.set_json_goal(u'CartesianVelocityLimit',
                                       root_link=pocky_pose_setup.default_root,
                                       tip_link=u'base_footprint',
                                       max_linear_velocity=0.1,
                                       max_angular_velocity=0.2
                                       )
        pocky_pose_setup.set_and_check_cart_goal(r_goal, pocky_pose_setup.r_tip, weight=WEIGHT_BELOW_CA)

    def test_UpdateGodMap2(self, pocky_pose_setup):
        """
        :type pocky_pose_setup: PR2
        """
        old_torso_value = pocky_pose_setup.get_god_map().get_data(identifier.joint_weight + [u'torso_lift_joint'])
        old_odom_x_value = pocky_pose_setup.get_god_map().get_data(identifier.joint_weight + [u'odom_x_joint'])
        old_odom_y_value = pocky_pose_setup.get_god_map().get_data(identifier.joint_weight + [u'odom_y_joint'])

        r_goal = PoseStamped()
        r_goal.header.frame_id = pocky_pose_setup.r_tip
        r_goal.pose.orientation.w = 1
        r_goal.pose.position.x += 0.1
        updates = {
            u'rosparam': {
                u'general_options': {
                    u'joint_weights': {
                        u'odom_x_joint': u'asdf',
                        u'odom_y_joint': 0.0001,
                        u'odom_z_joint': 0.0001
                    }
                }
            }
        }
        pocky_pose_setup.update_god_map(updates)
        pocky_pose_setup.set_cart_goal(r_goal, pocky_pose_setup.r_tip)
        pocky_pose_setup.send_and_check_goal(expected_error_codes=[MoveResult.ERROR])
        assert pocky_pose_setup.get_god_map().unsafe_get_data(
            identifier.joint_weight + [u'odom_x_joint']) == old_odom_x_value
        assert pocky_pose_setup.get_god_map().unsafe_get_data(identifier.joint_weight + [u'odom_y_joint']) == 0.0001
        assert pocky_pose_setup.get_god_map().get_data(
            identifier.joint_weight + [u'torso_lift_joint']) == old_torso_value

    def test_UpdateGodMap3(self, pocky_pose_setup):
        """
        :type pocky_pose_setup: PR2
        """
        old_torso_value = pocky_pose_setup.get_god_map().get_data(identifier.joint_weight + [u'torso_lift_joint'])
        old_odom_x_value = pocky_pose_setup.get_god_map().get_data(identifier.joint_weight + [u'odom_x_joint'])

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
        pocky_pose_setup.send_and_check_goal(expected_error_codes=[MoveResult.ERROR])
        assert pocky_pose_setup.get_god_map().unsafe_get_data(
            identifier.joint_weight + [u'odom_x_joint']) == old_odom_x_value
        assert pocky_pose_setup.get_god_map().unsafe_get_data(
            identifier.joint_weight + [u'torso_lift_joint']) == old_torso_value

    def test_pointing(self, kitchen_setup):
        base_goal = PoseStamped()
        base_goal.header.frame_id = u'base_footprint'
        base_goal.pose.position.y = -1
        base_goal.pose.orientation.w = 1
        kitchen_setup.teleport_base(base_goal)

        tip = u'head_mount_kinect_rgb_link'
        goal_point = tf.lookup_point(u'map', u'iai_kitchen/iai_fridge_door_handle')
        pointing_axis = Vector3Stamped()
        pointing_axis.header.frame_id = tip
        pointing_axis.vector.x = 1
        kitchen_setup.pointing(tip, goal_point, pointing_axis=pointing_axis)
        kitchen_setup.send_and_check_goal()

        base_goal = PoseStamped()
        base_goal.header.frame_id = u'base_footprint'
        base_goal.pose.position.y = 2
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(1, [0, 0, 1]))
        kitchen_setup.pointing(tip, goal_point, pointing_axis=pointing_axis)
        kitchen_setup.set_json_goal(u'CartesianVelocityLimit',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=u'base_footprint',
                                    max_linear_velocity=0.1,
                                    max_angular_velocity=0.2
                                    )
        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.move_base(base_goal)

        current_x = Vector3Stamped()
        current_x.header.frame_id = tip
        current_x.vector.x = 1

        expected_x = tf.transform_point(tip, goal_point)
        np.testing.assert_almost_equal(expected_x.point.y, 0, 1)
        np.testing.assert_almost_equal(expected_x.point.z, 0, 1)

        rospy.loginfo("Starting looking")
        tip = u'head_mount_kinect_rgb_link'
        goal_point = tf.lookup_point(u'map', kitchen_setup.r_tip)
        pointing_axis = Vector3Stamped()
        pointing_axis.header.frame_id = tip
        pointing_axis.vector.x = 1
        kitchen_setup.pointing(tip, goal_point, pointing_axis=pointing_axis, root_link=kitchen_setup.r_tip)

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

        kitchen_setup.set_and_check_cart_goal(r_goal, kitchen_setup.r_tip, u'base_footprint',
                                              weight=WEIGHT_BELOW_CA)

        rospy.loginfo("Starting testing")
        current_x = Vector3Stamped()
        current_x.header.frame_id = tip
        current_x.vector.x = 1

        expected_x = tf.lookup_point(tip, kitchen_setup.r_tip)
        np.testing.assert_almost_equal(expected_x.point.y, 0, 2)
        np.testing.assert_almost_equal(expected_x.point.z, 0, 2)

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
        zero_pose.align_planes(zero_pose.r_tip, x_gripper, root_normal=x_goal)
        zero_pose.align_planes(zero_pose.r_tip, y_gripper, root_normal=y_goal)
        zero_pose.send_and_check_goal()
        map_T_gripper = tf.lookup_pose(u'map', u'r_gripper_tool_frame')
        np.testing.assert_almost_equal(map_T_gripper.pose.orientation.x, 0.7071, decimal=3)
        np.testing.assert_almost_equal(map_T_gripper.pose.orientation.y, 0.0, decimal=3)
        np.testing.assert_almost_equal(map_T_gripper.pose.orientation.z, 0.0, decimal=3)
        np.testing.assert_almost_equal(map_T_gripper.pose.orientation.w, 0.7071, decimal=3)

    def test_wrong_constraint_type(self, zero_pose):
        goal_state = JointState()
        goal_state.name = [u'r_elbow_flex_joint']
        goal_state.position = [-1.0]
        kwargs = {u'goal_state': goal_state}
        zero_pose.set_json_goal(u'jointpos', **kwargs)
        zero_pose.send_and_check_goal(expected_error_codes=[MoveResult.UNKNOWN_CONSTRAINT])

    def test_python_code_in_constraint_type(self, zero_pose):
        goal_state = JointState()
        goal_state.name = [u'r_elbow_flex_joint']
        goal_state.position = [-1.0]
        kwargs = {u'goal_state': goal_state}
        zero_pose.set_json_goal(u'print("asd")', **kwargs)
        zero_pose.send_and_check_goal(expected_error_codes=[MoveResult.UNKNOWN_CONSTRAINT])

    def test_wrong_params1(self, zero_pose):
        goal_state = JointState()
        goal_state.name = u'r_elbow_flex_joint'
        goal_state.position = [-1.0]
        kwargs = {u'goal_state': goal_state}
        zero_pose.set_json_goal(u'JointPositionList', **kwargs)
        zero_pose.send_and_check_goal(expected_error_codes=[MoveResult.CONSTRAINT_INITIALIZATION_ERROR])

    def test_wrong_params2(self, zero_pose):
        goal_state = JointState()
        goal_state.name = [5432]
        goal_state.position = u'test'
        kwargs = {u'goal_state': goal_state}
        zero_pose.set_json_goal(u'JointPositionList', **kwargs)
        zero_pose.send_and_check_goal(expected_error_codes=[MoveResult.CONSTRAINT_INITIALIZATION_ERROR])

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
        # x_goal.vector.z = 0.2
        x_goal.vector = tf.normalize(x_goal.vector)
        zero_pose.align_planes(zero_pose.r_tip, x_gripper, root_normal=x_goal)
        zero_pose.send_and_check_goal()

        map_T_gripper = tf.transform_vector(u'map', x_gripper)
        np.testing.assert_almost_equal(map_T_gripper.vector.x, x_goal.vector.x, decimal=2)
        np.testing.assert_almost_equal(map_T_gripper.vector.y, x_goal.vector.y, decimal=2)
        np.testing.assert_almost_equal(map_T_gripper.vector.z, x_goal.vector.z, decimal=2)

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
        kitchen_setup.align_planes(kitchen_setup.r_tip, x_gripper, root_normal=x_goal)
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.send_and_check_goal()

    def test_open_fridge(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        percentage = 50
        handle_frame_id = u'iai_kitchen/iai_fridge_door_handle'
        handle_name = u'iai_fridge_door_handle'
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
        kitchen_setup.align_planes(kitchen_setup.r_tip, x_gripper, root_normal=x_goal, weight=WEIGHT_BELOW_CA)
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=percentage)
        kitchen_setup.limit_cartesian_velocity(u'odom_combined', u'base_footprint', max_linear_velocity=0.1,
                                               max_angular_velocity=0.2)
        kitchen_setup.send_and_check_goal()

        kitchen_setup.set_json_goal(u'OpenDoor',
                                    tip_link=kitchen_setup.r_tip,
                                    object_name=u'kitchen',
                                    object_link_name=handle_name,
                                    angle_goal=1.5)
        kitchen_setup.allow_all_collisions()
        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=percentage)
        kitchen_setup.send_and_check_goal()
        kitchen_setup.set_kitchen_js({u'iai_fridge_door_joint': 1.5})

        kitchen_setup.set_json_goal(u'OpenDoor',
                                    tip_link=kitchen_setup.r_tip,
                                    object_name=u'kitchen',
                                    object_link_name=handle_name,
                                    angle_goal=0)
        kitchen_setup.allow_all_collisions()
        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=percentage)
        kitchen_setup.send_and_check_goal()
        kitchen_setup.set_kitchen_js({u'iai_fridge_door_joint': 0})

        kitchen_setup.send_and_check_goal()

        kitchen_setup.send_and_check_joint_goal(gaya_pose)

    def test_open_close_fridge2(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        handle_frame_id = u'iai_kitchen/iai_fridge_door_handle'
        handle_name = u'iai_fridge_door_handle'
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
        # kitchen_setup.align_planes(kitchen_setup.r_tip, x_gripper, root_normal=x_goal)
        kitchen_setup.allow_all_collisions()
        kitchen_setup.send_and_check_goal()

        kitchen_setup.set_json_goal(u'Open',
                                    tip_link=kitchen_setup.r_tip,
                                    object_name=u'kitchen',
                                    object_link_name=handle_name)
        kitchen_setup.allow_all_collisions()

        kitchen_setup.send_and_check_goal()
        kitchen_setup.set_kitchen_js({u'iai_fridge_door_joint': np.pi / 2})

        kitchen_setup.set_json_goal(u'Close',
                                    tip_link=kitchen_setup.r_tip,
                                    object_name=u'kitchen',
                                    object_link_name=handle_name)
        kitchen_setup.allow_all_collisions()
        kitchen_setup.send_and_check_goal()
        kitchen_setup.set_kitchen_js({u'iai_fridge_door_joint': 0})

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
        tip_axis.header.frame_id = kitchen_setup.r_tip
        tip_axis.vector.x = 1

        env_axis = Vector3Stamped()
        env_axis.header.frame_id = handle_frame_id
        env_axis.vector.z = 1
        kitchen_setup.align_planes(elbow, tip_axis, root_normal=env_axis, weight=WEIGHT_ABOVE_CA)
        kitchen_setup.send_and_check_goal()
        elbow_pose = PoseStamped()
        elbow_pose.header.frame_id = handle_frame_id
        elbow_pose.pose.position.x += 0.1
        elbow_pose.pose.orientation.w = 1
        kitchen_setup.set_translation_goal(elbow_pose, elbow)
        kitchen_setup.align_planes(elbow, tip_axis, root_normal=env_axis, weight=WEIGHT_ABOVE_CA)
        kitchen_setup.allow_all_collisions()
        kitchen_setup.send_and_check_goal()

        kitchen_setup.set_json_goal(u'Close',
                                    tip_link=elbow,
                                    object_name=u'kitchen',
                                    object_link_name=handle_name)
        kitchen_setup.allow_all_collisions()
        kitchen_setup.send_and_check_goal()
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
        kitchen_setup.align_planes(kitchen_setup.l_tip, x_gripper, root_normal=x_goal)
        # kitchen_setup.allow_all_collisions()

        kitchen_setup.send_and_check_goal()

        kitchen_setup.set_json_goal(u'OpenDoor',
                                    tip_link=kitchen_setup.l_tip,
                                    object_name=u'kitchen',
                                    object_link_name=handle_name,
                                    angle_goal=goal_angle)
        kitchen_setup.allow_all_collisions()
        kitchen_setup.send_and_check_goal()
        kitchen_setup.set_kitchen_js({u'oven_area_oven_door_joint': goal_angle})

        kitchen_setup.set_json_goal(u'OpenDoor',
                                    tip_link=kitchen_setup.l_tip,
                                    object_name=u'kitchen',
                                    object_link_name=handle_name,
                                    angle_goal=0)
        kitchen_setup.allow_all_collisions()
        kitchen_setup.send_and_check_goal()
        kitchen_setup.set_kitchen_js({u'oven_area_oven_door_joint': 0})

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
        kitchen_setup.align_planes(hand, x_gripper, root_normal=x_goal)
        # kitchen_setup.allow_all_collisions()

        kitchen_setup.send_and_check_goal()

        kitchen_setup.set_json_goal(u'Open',
                                    tip_link=hand,
                                    object_name=u'kitchen',
                                    object_link_name=handle_name,
                                    goal_joint_state=goal_angle,
                                    # weight=100
                                    )
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.send_and_check_goal()
        kitchen_setup.set_kitchen_js({u'sink_area_dish_washer_door_joint': goal_angle})

        kitchen_setup.set_json_goal(u'Close',
                                    tip_link=hand,
                                    object_name=u'kitchen',
                                    object_link_name=handle_name)
        kitchen_setup.allow_all_collisions()
        kitchen_setup.send_and_check_goal()
        kitchen_setup.set_kitchen_js({u'sink_area_dish_washer_door_joint': 0})

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
        kitchen_setup.send_and_check_goal()

    def test_open_drawer(self, kitchen_setup):
        """"
        :type kitchen_setup: Boxy
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

        kitchen_setup.align_planes(kitchen_setup.l_tip,
                                   x_gripper,
                                   root_normal=x_goal)
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.send_and_check_goal()

        kitchen_setup.set_json_goal(u'Open',
                                    tip_link=kitchen_setup.l_tip,
                                    object_name=u'kitchen',
                                    object_link_name=handle_name)
        kitchen_setup.allow_all_collisions()  # makes execution faster
        kitchen_setup.send_and_check_goal()  # send goal to Giskard
        # Update kitchen object
        kitchen_setup.set_kitchen_js({u'sink_area_left_middle_drawer_main_joint': 0.48})

        # Close drawer partially
        kitchen_setup.set_json_goal(u'OpenDrawer',
                                    tip_link=kitchen_setup.l_tip,
                                    object_name=u'kitchen',
                                    object_link_name=handle_name,
                                    distance_goal=0.2)
        kitchen_setup.allow_all_collisions()  # makes execution faster
        kitchen_setup.send_and_check_goal()  # send goal to Giskard
        # Update kitchen object
        kitchen_setup.set_kitchen_js({u'sink_area_left_middle_drawer_main_joint': 0.2})

        kitchen_setup.set_json_goal(u'Close',
                                    tip_link=kitchen_setup.l_tip,
                                    object_name=u'kitchen',
                                    object_link_name=handle_name)
        kitchen_setup.allow_all_collisions()  # makes execution faster
        kitchen_setup.send_and_check_goal()  # send goal to Giskard
        # Update kitchen object
        kitchen_setup.set_kitchen_js({u'sink_area_left_middle_drawer_main_joint': 0.0})

        pass

    def test_open_all_drawers(self, kitchen_setup):
        """"
        :type kitchen_setup: Boxy
        """
        handle_name = [u'oven_area_area_middle_upper_drawer_handle',
                       u'oven_area_area_middle_lower_drawer_handle',
                       u'sink_area_left_upper_drawer_handle',
                       u'sink_area_left_middle_drawer_handle',
                       u'sink_area_left_bottom_drawer_handle',
                       u'sink_area_trash_drawer_handle',
                       u'fridge_area_lower_drawer_handle',
                       u'kitchen_island_left_upper_drawer_handle',
                       u'kitchen_island_left_lower_drawer_handle',
                       u'kitchen_island_middle_upper_drawer_handle',
                       u'kitchen_island_middle_lower_drawer_handle',
                       u'kitchen_island_right_upper_drawer_handle',
                       u'kitchen_island_right_lower_drawer_handle',
                       u'oven_area_area_left_drawer_handle',
                       u'oven_area_area_right_drawer_handle']

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

            kitchen_setup.add_json_goal(u'GraspBar',
                                        root=kitchen_setup.default_root,
                                        tip=kitchen_setup.l_tip,
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

            kitchen_setup.align_planes(kitchen_setup.l_tip,
                                       x_gripper,
                                       root_normal=x_goal)
            kitchen_setup.allow_all_collisions()
            kitchen_setup.avoid_self_collision()
            kitchen_setup.send_and_check_goal()

            kitchen_setup.add_json_goal(u'Open',
                                        tip=kitchen_setup.l_tip,
                                        object_name=u'kitchen',
                                        handle_link=i_handle_name)
            kitchen_setup.allow_all_collisions()  # makes execution faster
            kitchen_setup.avoid_self_collision()
            kitchen_setup.send_and_check_goal()  # send goal to Giskard
            # Update kitchen object
            kitchen_setup.set_kitchen_js({i_joint_name: 0.48})  # TODO: get real value from URDF

            # Close drawer partially
            kitchen_setup.add_json_goal(u'OpenDrawer',
                                        tip=kitchen_setup.l_tip,
                                        object_name=u'kitchen',
                                        handle_link=i_handle_name,
                                        distance_goal=0.2)
            kitchen_setup.allow_all_collisions()  # makes execution faster
            kitchen_setup.avoid_self_collision()
            kitchen_setup.send_and_check_goal()  # send goal to Giskard
            # Update kitchen object
            kitchen_setup.set_kitchen_js({i_joint_name: 0.2})

            kitchen_setup.add_json_goal(u'Close',
                                        tip=kitchen_setup.l_tip,
                                        object_name=u'kitchen',
                                        handle_link=i_handle_name)
            kitchen_setup.allow_all_collisions()  # makes execution faster
            kitchen_setup.avoid_self_collision()
            kitchen_setup.send_and_check_goal()  # send goal to Giskard
            # Update kitchen object
            kitchen_setup.set_kitchen_js({i_joint_name: 0.0})

        pass


class TestCartGoals(object):

    def test_rotate_gripper(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.r_tip
        r_goal.pose.orientation = Quaternion(*quaternion_about_axis(pi, [1, 0, 0]))
        zero_pose.set_and_check_cart_goal(r_goal, zero_pose.r_tip)

    def test_keep_position1(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.allow_self_collision()
        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.r_tip
        r_goal.pose.position.x = -.1
        r_goal.pose.orientation.w = 1
        zero_pose.set_and_check_cart_goal(r_goal, zero_pose.r_tip, u'torso_lift_link')

        zero_pose.allow_self_collision()
        js = {u'torso_lift_joint': 0.1}
        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.r_tip
        r_goal.pose.orientation.w = 1
        expected_pose = tf.lookup_pose(u'torso_lift_link', zero_pose.r_tip)
        expected_pose.header.stamp = rospy.Time()
        zero_pose.set_cart_goal(r_goal, zero_pose.r_tip, u'torso_lift_link')
        zero_pose.set_joint_goal(js)
        zero_pose.send_and_check_goal()
        zero_pose.check_cart_goal(zero_pose.r_tip, expected_pose)

    def test_keep_position2(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.allow_self_collision()
        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.r_tip
        r_goal.pose.position.x = -.1
        r_goal.pose.orientation.w = 1
        zero_pose.set_and_check_cart_goal(r_goal, zero_pose.r_tip, u'torso_lift_link')

        zero_pose.allow_self_collision()
        js = {u'torso_lift_joint': 0.1}
        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.r_tip
        r_goal.pose.orientation.w = 1
        expected_pose = tf.lookup_pose(zero_pose.default_root, zero_pose.r_tip)
        expected_pose.header.stamp = rospy.Time()
        zero_pose.set_cart_goal(r_goal, zero_pose.r_tip, zero_pose.default_root)
        zero_pose.set_joint_goal(js)
        zero_pose.send_and_check_goal()
        zero_pose.check_cart_goal(zero_pose.r_tip, expected_pose)

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
        zero_pose.set_and_check_cart_goal(r_goal, zero_pose.l_tip, u'torso_lift_link')

        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.r_tip
        r_goal.pose.orientation.w = 1
        zero_pose.set_cart_goal(r_goal, zero_pose.r_tip, zero_pose.l_tip)

        l_goal = PoseStamped()
        l_goal.header.frame_id = zero_pose.r_tip
        l_goal.pose.position.y = -.1
        l_goal.pose.orientation.w = 1
        zero_pose.set_and_check_cart_goal(l_goal, zero_pose.r_tip, zero_pose.default_root)

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
        zero_pose.set_and_check_cart_goal(p, zero_pose.r_tip, u'base_footprint')

    def test_cart_goal_1eef2(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        p = PoseStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = zero_pose.default_root
        p.pose.position = Point(0.599, -0.009, 0.983)
        p.pose.orientation = Quaternion(0.524, -0.495, 0.487, -0.494)
        zero_pose.allow_self_collision()
        zero_pose.set_and_check_cart_goal(p, zero_pose.l_tip, zero_pose.default_root)

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
        zero_pose.set_and_check_cart_goal(p, zero_pose.r_tip, zero_pose.default_root)

    def test_cart_goal_2eef(self, zero_pose):
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
        zero_pose.allow_self_collision()
        zero_pose.send_and_check_goal()
        zero_pose.check_cart_goal(zero_pose.r_tip, r_goal)
        zero_pose.check_cart_goal(zero_pose.l_tip, l_goal)

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
        zero_pose.send_and_check_goal()
        zero_pose.check_cart_goal(zero_pose.r_tip, r_goal)
        zero_pose.check_cart_goal(zero_pose.l_tip, l_goal)

    def test_wiggle1(self, kitchen_setup):
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
        kitchen_setup.send_and_check_goal()

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
        zero_pose.set_and_check_cart_goal(p, zero_pose.l_tip, zero_pose.default_root)

        p = PoseStamped()
        p.header.frame_id = zero_pose.l_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.2
        p.pose.orientation.w = 1
        # zero_pose.allow_all_collisions()
        zero_pose.set_and_check_cart_goal(p, zero_pose.l_tip, zero_pose.default_root)

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
        zero_pose.send_and_check_goal()

    def test_wiggle4(self, pocky_pose_setup):
        """
        :type pocky_pose_setup: PR2
        """
        p = PoseStamped()
        p.header.frame_id = u'map'
        p.pose.position.x = 1.1
        p.pose.position.y = 0
        p.pose.position.z = 0.6
        p.pose.orientation.w = 1
        pocky_pose_setup.add_box(size=[1, 1, 0.01], pose=p)

        p = PoseStamped()
        p.header.frame_id = pocky_pose_setup.r_tip
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        pocky_pose_setup.set_and_check_cart_goal(p, pocky_pose_setup.r_tip, pocky_pose_setup.default_root,
                                                 expected_error_codes=[MoveResult.SHAKING])

        # box_setup.avoid_collision()

        # collision_entry = CollisionEntry()
        # collision_entry.type = CollisionEntry.AVOID_COLLISION
        # collision_entry.min_dist = 0.05
        # collision_entry.body_b = u'box'
        # pocky_pose_setup.add_collision_entries([collision_entry])
        #
        # pocky_pose_setup.send_and_check_goal(expected_error_code=MoveResult.INSOLVABLE)

    def test_interrupt1(self, zero_pose):
        p = PoseStamped()
        p.header.frame_id = u'base_footprint'
        p.pose.position = Point(2, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_cart_goal(p, u'base_footprint')
        result = zero_pose.send_goal_and_dont_wait(stop_after=20)
        assert result.error_codes[0] == MoveResult.PREEMPTED

    def test_interrupt_way_points1(self, zero_pose):
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
        result = zero_pose.send_goal_and_dont_wait(stop_after=10)
        assert result.error_codes[0] == MoveResult.SUCCESS
        assert result.error_codes[1] == MoveResult.PREEMPTED
        assert result.error_codes[2] == MoveResult.PREEMPTED

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
        zero_pose.send_and_check_goal()
        zero_pose.check_cart_goal(zero_pose.r_tip, r_goal)
        zero_pose.check_cart_goal(zero_pose.l_tip, l_goal)

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
        zero_pose.send_and_check_goal()

    def test_waypoints(self, zero_pose):
        """
        :type zero_pose: PR2
        """
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

        zero_pose.send_and_check_goal()

    def test_waypoints2(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.set_joint_goal(pocky_pose)
        zero_pose.add_cmd()
        zero_pose.set_joint_goal(pick_up_pose)
        zero_pose.add_cmd()
        zero_pose.set_joint_goal(gaya_pose)

        traj = zero_pose.send_and_check_goal()
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

        traj = zero_pose.send_and_check_goal(expected_error_codes=[MoveResult.SUCCESS,
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

        traj = zero_pose.send_and_check_goal(expected_error_codes=[MoveResult.UNKNOWN_CONSTRAINT,
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

        traj = zero_pose.send_and_check_goal(expected_error_codes=[MoveResult.SUCCESS,
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

        traj = zero_pose.send_and_check_goal(expected_error_codes=[MoveResult.SUCCESS,
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
        zero_pose.send_and_check_goal(expected_error_codes=[MoveResult.UNKNOWN_CONSTRAINT, ],
                                      goal_type=MoveGoal.PLAN_AND_EXECUTE_AND_SKIP_FAILURES)

    def test_skip_failures2(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.set_joint_goal(pocky_pose)
        traj = zero_pose.send_and_check_goal(expected_error_codes=[MoveResult.SUCCESS, ],
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


class TestShaking(object):
    def test_wiggle_prismatic_joint_neglectable_shaking(self, kitchen_setup):
        sample_period = kitchen_setup.get_god_map().get_data(identifier.sample_period)
        frequency_range = kitchen_setup.get_god_map().get_data(identifier.frequency_range)
        amplitude_threshold = kitchen_setup.get_god_map().get_data(identifier.amplitude_threshold)
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
        sample_period = kitchen_setup.get_god_map().get_data(identifier.sample_period)
        frequency_range = kitchen_setup.get_god_map().get_data(identifier.frequency_range)
        amplitude_threshold = kitchen_setup.get_god_map().get_data(identifier.amplitude_threshold)
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
        sample_period = kitchen_setup.get_god_map().get_data(identifier.sample_period)
        frequency_range = kitchen_setup.get_god_map().get_data(identifier.frequency_range)
        amplitude_threshold = kitchen_setup.get_god_map().get_data(identifier.amplitude_threshold)
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
        sample_period = kitchen_setup.get_god_map().get_data(identifier.sample_period)
        frequency_range = kitchen_setup.get_god_map().get_data(identifier.frequency_range)
        max_detectable_freq = int(1 / (2 * sample_period))
        min_wiggle_frequency = int(frequency_range * max_detectable_freq)
        while np.fmod(min_wiggle_frequency, 5.0) != 0.0:
            min_wiggle_frequency += 1
        distance_between_frequencies = 5

        for joint in [ u'head_pan_joint', u'r_wrist_flex_joint']:  # max vel: 1.0 and 0.5
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
                r = kitchen_setup.send_goal(goal=None, goal_type=MoveGoal.PLAN_AND_EXECUTE)
                assert len(r.error_codes) != 0
                error_code = r.error_codes[0]
                assert error_code == MoveResult.SHAKING
                error_message = r.error_messages[0]
                freqs_str = re.findall("[0-9]+\.[0-9]+ hertz", error_message)
                assert any(map(lambda f_str: float(f_str[:-6]) == target_freq, freqs_str))

    def test_wiggle_prismatic_joint_shaking(self, kitchen_setup):
        sample_period = kitchen_setup.get_god_map().get_data(identifier.sample_period)
        frequency_range = kitchen_setup.get_god_map().get_data(identifier.frequency_range)
        max_detectable_freq = int(1 / (2 * sample_period))
        min_wiggle_frequency = int(frequency_range * max_detectable_freq)
        while np.fmod(min_wiggle_frequency, 5.0) != 0.0:
            min_wiggle_frequency += 1
        distance_between_frequencies = 5

        for joint in [u'odom_x_joint']: #, u'torso_lift_joint']: # max vel: 0.015 and 0.5
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
                r = kitchen_setup.send_goal(goal=None, goal_type=MoveGoal.PLAN_AND_EXECUTE)
                assert len(r.error_codes) != 0
                error_code = r.error_codes[0]
                assert error_code == MoveResult.SHAKING
                error_message = r.error_messages[0]
                freqs_str = re.findall("[0-9]+\.[0-9]+ hertz", error_message)
                assert any(map(lambda f_str: float(f_str[:-6]) == target_freq, freqs_str))

    def test_wiggle_continuous_joint_shaking(self, kitchen_setup):
        sample_period = kitchen_setup.get_god_map().get_data(identifier.sample_period)
        frequency_range = kitchen_setup.get_god_map().get_data(identifier.frequency_range)
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
                r = kitchen_setup.send_goal(goal=None, goal_type=MoveGoal.PLAN_AND_EXECUTE)
                assert len(r.error_codes) != 0
                error_code = r.error_codes[0]
                assert error_code == MoveResult.SHAKING
                error_message = r.error_messages[0]
                freqs_str = re.findall("[0-9]+\.[0-9]+ hertz", error_message)
                assert any(map(lambda f_str: float(f_str[:-6]) == target_freq, freqs_str))

    def test_only_revolute_joint_shaking(self, kitchen_setup):
        sample_period = kitchen_setup.get_god_map().get_data(identifier.sample_period)
        frequency_range = kitchen_setup.get_god_map().get_data(identifier.frequency_range)
        amplitude_threshold = kitchen_setup.get_god_map().get_data(identifier.amplitude_threshold)
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
                r = kitchen_setup.send_goal(goal=None, goal_type=MoveGoal.PLAN_AND_EXECUTE)
                assert len(r.error_codes) != 0
                error_code = r.error_codes[0]
                assert error_code == MoveResult.SHAKING
                error_message = r.error_messages[0]
                freqs_str = re.findall("[0-9]+\.[0-9]+ hertz", error_message)
                assert any(map(lambda f_str: float(f_str[:-6]) == target_freq, freqs_str))

    def test_only_revolute_joint_neglectable_shaking(self, kitchen_setup):
        sample_period = kitchen_setup.get_god_map().get_data(identifier.sample_period)
        frequency_range = kitchen_setup.get_god_map().get_data(identifier.frequency_range)
        amplitude_threshold = kitchen_setup.get_god_map().get_data(identifier.amplitude_threshold)
        max_detectable_freq = int(1 / (2 * sample_period))
        min_wiggle_frequency = int(frequency_range * max_detectable_freq)
        while np.fmod(min_wiggle_frequency, 5.0) != 0.0:
            min_wiggle_frequency += 1
        distance_between_frequencies = 5

        for revolute_joint in [ u'r_wrist_flex_joint', u'head_pan_joint']:  # max vel. of 1.0 and 0.5
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
    def test_bug2020_09_02_11_52_21_dump(self, kitchen_setup):
        map_T_odom = tf.pose_to_kdl(convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                                      {
                                                                          "header": {
                                                                              "frame_id": "map",
                                                                              "seq": 0,
                                                                              "stamp": {
                                                                                  "nsecs": 0,
                                                                                  "secs": 0
                                                                              }
                                                                          },
                                                                          "pose": {
                                                                              "orientation": {
                                                                                  "w": 0.9907124990298876,
                                                                                  "x": -0.004530583180313736,
                                                                                  "y": 0.003707430414081215,
                                                                                  "z": -0.13584724156833777
                                                                              },
                                                                              "position": {
                                                                                  "x": -0.2989784139083874,
                                                                                  "y": -0.644299033123408,
                                                                                  "z": -0.0046400968427768976
                                                                              }
                                                                          }
                                                                      }
                                                                      ).pose)

        odom_T_base_footprint = PoseStamped()
        odom_T_base_footprint.pose.position = Point(
            -1.4807876704123586,
            0.44216787977956584,
            0
        )
        odom_T_base_footprint.pose.orientation = Quaternion(*quaternion_about_axis(
            0.2796229462716709,
            [0, 0, 1]))
        odom_T_base_footprint = tf.pose_to_kdl(odom_T_base_footprint.pose)
        map_T_base_footprint = tf.kdl_to_pose_stamped(map_T_odom * odom_T_base_footprint, u'map')

        kitchen_setup.teleport_base(map_T_base_footprint)

        js = {
            "l_elbow_flex_joint": -2.121008899318323,
            "l_forearm_roll_joint": -1.8527228579572208,
            "l_shoulder_lift_joint": -0.26381383840391553,
            "l_shoulder_pan_joint": 2.1042434286298475,
            "l_upper_arm_roll_joint": 1.3839220687489173,
            "l_wrist_flex_joint": -0.09996131931979324,
            "l_wrist_roll_joint": -6.282344219090744,
            "r_elbow_flex_joint": -2.1208641286069394,
            "r_forearm_roll_joint": -17.08314793464878,
            "r_shoulder_lift_joint": -0.25706741673108136,
            "r_shoulder_pan_joint": -1.712587449591307,
            "r_upper_arm_roll_joint": -1.4633501125737376,
            "r_wrist_flex_joint": -0.10002450226778681,
            "r_wrist_roll_joint": 25.183178835783142,
            "torso_lift_joint": 0.2631247287575463,
        }
        kitchen_setup.send_and_check_joint_goal(js)

        box_pose = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                     {
                                                         "header": {
                                                             "frame_id": "r_wrist_roll_link",
                                                             "seq": 0,
                                                             "stamp": {
                                                                 "nsecs": 0,
                                                                 "secs": 0
                                                             }
                                                         },
                                                         "pose": {
                                                             "orientation": {
                                                                 "w": 0.7110270195399259,
                                                                 "x": -0.013000494027886993,
                                                                 "y": -0.7030267155229984,
                                                                 "z": -0.005000190010958694
                                                             },
                                                             "position": {
                                                                 "x": 0.195,
                                                                 "y": 0.042,
                                                                 "z": -0.025
                                                             }
                                                         }
                                                     }
                                                     )
        kitchen_setup.attach_box(u'bowl_1',
                                 size=[0.171740325292, 0.171745332082, 0.10578233401],
                                 frame_id='r_wrist_roll_link', pose=box_pose)

        wrong_odom_T_goal = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                              {
                                                                  "header": {
                                                                      "frame_id": "odom_combined",
                                                                      "seq": 0,
                                                                      "stamp": {
                                                                          "nsecs": 0,
                                                                          "secs": 0
                                                                      }
                                                                  },
                                                                  "pose": {
                                                                      "orientation": {
                                                                          "w": 1,
                                                                          "x": 0,
                                                                          "y": 0,
                                                                          "z": 0,
                                                                      },
                                                                      "position": {
                                                                          "x": 0,
                                                                          "y": 2,
                                                                          "z": 0
                                                                      }
                                                                  }
                                                              }
                                                              )

        map_T_cart_goal = tf.kdl_to_pose_stamped(map_T_odom * tf.pose_to_kdl(wrong_odom_T_goal.pose), u'map')
        kitchen_setup.avoid_all_collisions(0.2)
        kitchen_setup.set_and_check_cart_goal(map_T_cart_goal, u'base_footprint',
                                              expected_error_codes=[MoveResult.SHAKING])

    def test_bug2020_09_01_11_22_51_dump(self, kitchen_setup):
        map_T_odom = tf.pose_to_kdl(convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                                      {
                                                                          "header": {
                                                                              "frame_id": "map",
                                                                              "seq": 0,
                                                                              "stamp": {
                                                                                  "nsecs": 0,
                                                                                  "secs": 0
                                                                              }
                                                                          },
                                                                          "pose": {
                                                                              "orientation": {
                                                                                  "w": 0.2402549656091816,
                                                                                  "x": -0.0024454268609120245,
                                                                                  "y": -0.002852994168366019,
                                                                                  "z": 0.9707025454854202
                                                                              },
                                                                              "position": {
                                                                                  "x": -2.1180501887755714,
                                                                                  "y": 0.7919687518861623,
                                                                                  "z": -0.017022869821609603
                                                                              }
                                                                          }
                                                                      }
                                                                      ).pose)

        odom_T_base_footprint = PoseStamped()
        odom_T_base_footprint.pose.position = Point(
            -2.3774013091929227,
            -1.3334991298858108,
            0
        )
        odom_T_base_footprint.pose.orientation = Quaternion(*quaternion_about_axis(
            -1.7025919013405562,
            [0, 0, 1]))
        odom_T_base_footprint = tf.pose_to_kdl(odom_T_base_footprint.pose)
        map_T_base_footprint = tf.kdl_to_pose_stamped(map_T_odom * odom_T_base_footprint, u'map')

        kitchen_setup.teleport_base(map_T_base_footprint)

        js = {
            "l_elbow_flex_joint": -1.727087793643744,
            "l_forearm_roll_joint": -1.852954246305023,
            "l_shoulder_lift_joint": -0.023564256787264204,
            "l_shoulder_pan_joint": 1.5953626112926687,
            "l_upper_arm_roll_joint": 1.384242778401657,
            "l_wrist_flex_joint": -0.4799682274696515,
            "l_wrist_roll_joint": -6.283649486693731,
            "r_elbow_flex_joint": -1.2901697866882575,
            "r_forearm_roll_joint": -8.797767517806948,
            "r_shoulder_lift_joint": -0.007343203726731174,
            "r_shoulder_pan_joint": -0.015571898911165505,
            "r_upper_arm_roll_joint": -0.28570426771329305,
            "r_wrist_flex_joint": -1.1989728061433444,
            "r_wrist_roll_joint": 15.769936954395696,
            "torso_lift_joint": 0.26230693418016243,
        }
        kitchen_setup.send_and_check_joint_goal(js)

        kitchen_setup.set_kitchen_js(
            {
                "fridge_area_lower_drawer_main_joint": 0.0,
                "iai_fridge_door_joint": 0.0,
                "kitchen_island_left_lower_drawer_main_joint": 0.0,
                "kitchen_island_left_upper_drawer_main_joint": 0.0,
                "kitchen_island_middle_lower_drawer_main_joint": 0.0,
                "kitchen_island_middle_upper_drawer_main_joint": 0.0,
                "kitchen_island_right_lower_drawer_main_joint": 0.0,
                "kitchen_island_right_upper_drawer_main_joint": 0.0,
                "oven_area_area_left_drawer_main_joint": 0.0,
                "oven_area_area_middle_lower_drawer_main_joint": 0.0,
                "oven_area_area_middle_upper_drawer_main_joint": 0.0,
                "oven_area_area_right_drawer_main_joint": 0.48,
                "oven_area_oven_door_joint": 0.0,
                "oven_area_oven_knob_oven_joint": 0.0,
                "oven_area_oven_knob_stove_1_joint": 0.0,
                "oven_area_oven_knob_stove_2_joint": 0.0,
                "oven_area_oven_knob_stove_3_joint": 0.0,
                "oven_area_oven_knob_stove_4_joint": 0.0,
                "sink_area_dish_washer_door_joint": 0.0,
                "sink_area_dish_washer_main_joint": 0.0,
                "sink_area_dish_washer_tray_main": 0.0,
                "sink_area_left_bottom_drawer_main_joint": 0.0,
                "sink_area_left_middle_drawer_main_joint": 0.0,
                "sink_area_left_upper_drawer_main_joint": 0.0,
                "sink_area_trash_drawer_main_joint": 0.0
            }
        )

        box_pose = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                     {
                                                         "header": {
                                                             "frame_id": "r_wrist_roll_link",
                                                             "seq": 0,
                                                             "stamp": {
                                                                 "nsecs": 0,
                                                                 "secs": 0
                                                             }
                                                         },
                                                         "pose": {
                                                             "orientation": {
                                                                 "w": 0.9999890001814967,
                                                                 "x": -0.0029999670005432843,
                                                                 "y": -0.0029999670005467277,
                                                                 "z": 0.001999978000361496
                                                             },
                                                             "position": {
                                                                 "x": 0.208,
                                                                 "y": 0.01,
                                                                 "z": -0.033
                                                             }
                                                         }
                                                     }
                                                     )
        kitchen_setup.attach_box(u'bowl_1',
                                 size=[0.152831586202, 0.06345692873, 0.228943316142],
                                 frame_id='r_wrist_roll_link', pose=box_pose)

        wrong_odom_T_goal = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                              {
                                                                  "header": {
                                                                      "frame_id": "odom_combined",
                                                                      "seq": 0,
                                                                      "stamp": {
                                                                          "nsecs": 0,
                                                                          "secs": 0
                                                                      }
                                                                  },
                                                                  "pose": {
                                                                      "orientation": {
                                                                          "w": 0.8487663514523268,
                                                                          "x": -0.0026119368041485624,
                                                                          "y": 0.009552505316512186,
                                                                          "z": -0.5286753547595905
                                                                      },
                                                                      "position": {
                                                                          "x": -2.475719357850976,
                                                                          "y": -2.0207340645247847,
                                                                          "z": 1.4500665049676453
                                                                      }
                                                                  }
                                                              }
                                                              )

        map_T_cart_goal = tf.kdl_to_pose_stamped(map_T_odom * tf.pose_to_kdl(wrong_odom_T_goal.pose), u'map')

        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup.set_and_check_cart_goal(map_T_cart_goal, kitchen_setup.r_tip)

    def test_bug2020_09_08_09_09_53_dump(self, kitchen_setup):
        map_T_odom = tf.pose_to_kdl(convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                                      {
                                                                          "header": {
                                                                              "frame_id": "map",
                                                                              "seq": 0,
                                                                              "stamp": {
                                                                                  "nsecs": 0,
                                                                                  "secs": 0
                                                                              }
                                                                          },
                                                                          "pose": {
                                                                              "orientation": {
                                                                                  "w": 0.9973904146827032,
                                                                                  "x": 0.00024320557199049972,
                                                                                  "y": -0.0049445563148697875,
                                                                                  "z": -0.07202675137033579
                                                                              },
                                                                              "position": {
                                                                                  "x": -0.060075821282061875,
                                                                                  "y": -0.25338950882393374,
                                                                                  "z": -0.002745786211418995
                                                                              }
                                                                          }
                                                                      }
                                                                      ).pose)

        odom_T_base_footprint = PoseStamped()
        odom_T_base_footprint.pose.position = Point(
            0.12146126817667068,
            1.294902645008183,
            0
        )
        odom_T_base_footprint.pose.orientation = Quaternion(*quaternion_about_axis(
            0.31502998439474206,
            [0, 0, 1]))
        odom_T_base_footprint = tf.pose_to_kdl(odom_T_base_footprint.pose)
        map_T_base_footprint = tf.kdl_to_pose_stamped(map_T_odom * odom_T_base_footprint, u'map')

        kitchen_setup.teleport_base(map_T_base_footprint)

        js = {
            "l_elbow_flex_joint": -1.727087793643744,
            "l_forearm_roll_joint": 23.279522538583297,
            "l_shoulder_lift_joint": -0.02364885171036865,
            "l_shoulder_pan_joint": 1.5954455179100255,
            "l_upper_arm_roll_joint": 1.3836013590961775,
            "l_wrist_flex_joint": -0.47992471854955054,
            "l_wrist_roll_joint": -25.132627361160118,
            "r_elbow_flex_joint": -0.5385202531849814,
            "r_forearm_roll_joint": -58.22706206949578,
            "r_shoulder_lift_joint": 0.2722430171334916,
            "r_shoulder_pan_joint": -0.2927287207355688,
            "r_upper_arm_roll_joint": -1.6156871976251295,
            "r_wrist_flex_joint": -1.3164468904122453,
            "r_wrist_roll_joint": 42.526965619404066,
            "torso_lift_joint": 0.2623284753355404,
        }
        kitchen_setup.send_and_check_joint_goal(js)

        kitchen_setup.set_kitchen_js(
            {
                "fridge_area_lower_drawer_main_joint": 0.0,
                "iai_fridge_door_joint": 0.0,
                "kitchen_island_left_lower_drawer_main_joint": 0.0,
                "kitchen_island_left_upper_drawer_main_joint": 0.0,
                "kitchen_island_middle_lower_drawer_main_joint": 0.0,
                "kitchen_island_middle_upper_drawer_main_joint": 0.0,
                "kitchen_island_right_lower_drawer_main_joint": 0.0,
                "kitchen_island_right_upper_drawer_main_joint": 0.0,
                "oven_area_area_left_drawer_main_joint": 0.0,
                "oven_area_area_middle_lower_drawer_main_joint": 0.0,
                "oven_area_area_middle_upper_drawer_main_joint": 0.0,
                "oven_area_area_right_drawer_main_joint": 0.0,
                "oven_area_oven_door_joint": 0.0,
                "oven_area_oven_knob_oven_joint": 0.0,
                "oven_area_oven_knob_stove_1_joint": 0.0,
                "oven_area_oven_knob_stove_2_joint": 0.0,
                "oven_area_oven_knob_stove_3_joint": 0.0,
                "oven_area_oven_knob_stove_4_joint": 0.0,
                "sink_area_dish_washer_door_joint": 0.0,
                "sink_area_dish_washer_main_joint": 0.0,
                "sink_area_dish_washer_tray_main": 0.0,
                "sink_area_left_bottom_drawer_main_joint": 0.0,
                "sink_area_left_middle_drawer_main_joint": 0.48,
                "sink_area_left_upper_drawer_main_joint": 0
            }
        )

        box_pose = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                     {
                                                         "header": {
                                                             "frame_id": "map",
                                                             "seq": 0,
                                                             "stamp": {
                                                                 "nsecs": 0,
                                                                 "secs": 0
                                                             }
                                                         },
                                                         "pose": {
                                                             "orientation": {
                                                                 "w": 0.8058056165210137,
                                                                 "x": 2.2993024607250623e-05,
                                                                 "y": -2.226165262633548e-05,
                                                                 "z": -0.5921801308376775
                                                             },
                                                             "position": {
                                                                 "x": 1.0608560562133789,
                                                                 "y": 0.8797644933064779,
                                                                 "z": 0.48437986373901365
                                                             }
                                                         }
                                                     }
                                                     )
        kitchen_setup.add_box(u'bowl_1',
                              size=[0.171740325292, 0.171745332082, 0.10578233401], pose=box_pose)

        wrong_odom_T_goal = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                              {
                                                                  "header": {
                                                                      "frame_id": "odom_combined",
                                                                      "seq": 0,
                                                                      "stamp": {
                                                                          "nsecs": 935841084,
                                                                          "secs": 1599556173
                                                                      }
                                                                  },
                                                                  "pose": {
                                                                      "orientation": {
                                                                          "w": 0.5957425979708209,
                                                                          "x": 0.37446614102720405,
                                                                          "y": 0.6011850022426514,
                                                                          "z": -0.37873795523713827
                                                                      },
                                                                      "position": {
                                                                          "x": 0.8881003806839934,
                                                                          "y": 1.252664054414375,
                                                                          "z": 0.4769618101890212
                                                                      }
                                                                  }
                                                              }
                                                              )

        map_T_cart_goal = tf.kdl_to_pose_stamped(map_T_odom * tf.pose_to_kdl(wrong_odom_T_goal.pose), u'map')

        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup.avoid_all_collisions(0.1)
        kitchen_setup.allow_collision([
            "r_gripper_l_finger_tip_link",
            "r_gripper_r_finger_tip_link",
            "r_gripper_l_finger_link",
            "r_gripper_r_finger_link",
            "r_gripper_l_finger_tip_frame",
            "r_gripper_palm_link"
        ], 'bowl_1', [])
        kitchen_setup.allow_collision([
            "r_gripper_l_finger_tip_link",
            "r_gripper_r_finger_tip_link",
            "r_gripper_l_finger_link",
            "r_gripper_r_finger_link",
            "r_gripper_l_finger_tip_frame",
            "r_gripper_palm_link"
        ], 'kitchen', [])
        kitchen_setup.set_and_check_cart_goal(map_T_cart_goal, kitchen_setup.r_tip,
                                              expected_error_codes=[MoveResult.SHAKING])

    def test_bug2020_09_08_10_01_32_dump(self, kitchen_setup):
        map_T_odom = tf.pose_to_kdl(convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                                      {
                                                                          "header": {
                                                                              "frame_id": "map",
                                                                              "seq": 0,
                                                                              "stamp": {
                                                                                  "nsecs": 0,
                                                                                  "secs": 0
                                                                              }
                                                                          },
                                                                          "pose": {
                                                                              "orientation": {
                                                                                  "w": 0.87164687748554,
                                                                                  "x": -0.0009706537169409304,
                                                                                  "y": 0.0014611973254151415,
                                                                                  "z": -0.49013125150662035
                                                                              },
                                                                              "position": {
                                                                                  "x": -0.9453400065712703,
                                                                                  "y": 0.018803105577148,
                                                                                  "z": 0.0039435292503660615
                                                                              }
                                                                          }
                                                                      }
                                                                      ).pose)

        odom_T_base_footprint = PoseStamped()
        odom_T_base_footprint.pose.position = Point(
            -0.13824759683245186,
            1.338527401416233,
            0
        )
        odom_T_base_footprint.pose.orientation = Quaternion(*quaternion_about_axis(
            -2.168491195651172,
            [0, 0, 1]))
        odom_T_base_footprint = tf.pose_to_kdl(odom_T_base_footprint.pose)
        map_T_base_footprint = tf.kdl_to_pose_stamped(map_T_odom * odom_T_base_footprint, u'map')

        kitchen_setup.teleport_base(map_T_base_footprint)

        js = {
            "l_elbow_flex_joint": -1.727087793643744,
            "l_forearm_roll_joint": 29.562930970240128,
            "l_shoulder_lift_joint": -0.02364885171036865,
            "l_shoulder_pan_joint": 1.5953626112926687,
            "l_upper_arm_roll_joint": 1.3837617139225473,
            "l_wrist_flex_joint": -0.48370999459821595,
            "l_wrist_roll_joint": -25.132844905760614,
            "r_elbow_flex_joint": -0.8486191169684979,
            "r_forearm_roll_joint": -70.33561430998711,
            "r_shoulder_lift_joint": 0.23299097281302453,
            "r_shoulder_pan_joint": -0.06539877594271382,
            "r_upper_arm_roll_joint": -2.303769757578311,
            "r_wrist_flex_joint": -0.8650853532990554,
            "r_wrist_roll_joint": 46.15404323458665,
            "torso_lift_joint": 0.2617033422141643,
        }
        kitchen_setup.send_and_check_joint_goal(js)

        kitchen_setup.set_kitchen_js(
            {
                "fridge_area_lower_drawer_main_joint": 0.0,
                "iai_fridge_door_joint": 0.0,
                "kitchen_island_left_lower_drawer_main_joint": 0.0,
                "kitchen_island_left_upper_drawer_main_joint": 0.48,
                "kitchen_island_middle_lower_drawer_main_joint": 0.0,
                "kitchen_island_middle_upper_drawer_main_joint": 0.0,
                "kitchen_island_right_lower_drawer_main_joint": 0.0,
                "kitchen_island_right_upper_drawer_main_joint": 0.0,
                "oven_area_area_left_drawer_main_joint": 0.0,
                "oven_area_area_middle_lower_drawer_main_joint": 0.0,
                "oven_area_area_middle_upper_drawer_main_joint": 0.0,
                "oven_area_area_right_drawer_main_joint": 0.0,
                "oven_area_oven_door_joint": 0.0,
                "oven_area_oven_knob_oven_joint": 0.0,
                "oven_area_oven_knob_stove_1_joint": 0.0,
                "oven_area_oven_knob_stove_2_joint": 0.0,
                "oven_area_oven_knob_stove_3_joint": 0.0,
                "oven_area_oven_knob_stove_4_joint": 0.0,
                "sink_area_dish_washer_door_joint": 0.0,
                "sink_area_dish_washer_main_joint": 0.0,
                "sink_area_dish_washer_tray_main": 0.0,
                "sink_area_left_bottom_drawer_main_joint": 0.0,
                "sink_area_left_middle_drawer_main_joint": 0.0,
                "sink_area_left_upper_drawer_main_joint": 0.0,
                "sink_area_trash_drawer_main_joint": 0.0
            }
        )

        box_pose = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                     {
                                                         "header": {
                                                             "frame_id": "map",
                                                             "seq": 0,
                                                             "stamp": {
                                                                 "nsecs": 0,
                                                                 "secs": 0
                                                             }
                                                         },
                                                         "pose": {
                                                             "orientation": {
                                                                 "w": 0.9989048744924449,
                                                                 "x": -0.006304425623536842,
                                                                 "y": 0.00309621704223198,
                                                                 "z": 0.04625710078266182
                                                             },
                                                             "position": {
                                                                 "x": -0.4601353963216146,
                                                                 "y": 0.9215604782104492,
                                                                 "z": 0.4846065521240234
                                                             }
                                                         }
                                                     }
                                                     )
        kitchen_setup.add_box(u'bowl_1',
                              size=[0.0556323369344, 0.0556323369344, 0.105665334066], pose=box_pose)

        wrong_odom_T_goal = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                              {
                                                                  "header": {
                                                                      "frame_id": "odom_combined",
                                                                      "seq": 0,
                                                                      "stamp": {
                                                                          "nsecs": 917441130,
                                                                          "secs": 1599559276
                                                                      }
                                                                  },
                                                                  "pose": {
                                                                      "orientation": {
                                                                          "w": 0.6929397605331455,
                                                                          "x": 0.15709194634891274,
                                                                          "y": 0.685000298795508,
                                                                          "z": -0.1610317299566266
                                                                      },
                                                                      "position": {
                                                                          "x": -0.5333840771704949,
                                                                          "y": 0.8550977845704844,
                                                                          "z": 0.5024671620545541
                                                                      }
                                                                  }
                                                              }
                                                              )

        map_T_cart_goal = tf.kdl_to_pose_stamped(map_T_odom * tf.pose_to_kdl(wrong_odom_T_goal.pose), u'map')

        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=40)
        # kitchen_setup.avoid_all_collisions(0.05)
        kitchen_setup.allow_collision([
            "r_gripper_l_finger_tip_link",
            "r_gripper_r_finger_tip_link",
            "r_gripper_l_finger_link",
            "r_gripper_r_finger_link",
            "r_gripper_l_finger_tip_frame",
            "r_gripper_palm_link"
        ], 'bowl_1', [])
        kitchen_setup.allow_collision([
            "r_gripper_l_finger_tip_link",
            "r_gripper_r_finger_tip_link",
            "r_gripper_l_finger_link",
            "r_gripper_r_finger_link",
            "r_gripper_l_finger_tip_frame",
            "r_gripper_palm_link"
        ], 'kitchen', [])
        kitchen_setup.set_and_check_cart_goal(map_T_cart_goal, kitchen_setup.r_tip)

    def test_bug2020_09_15_11_23_35_dump_grasp_wiggle_drawer(self, kitchen_setup):
        # fixme  AssertionError: in goal 0; got: SUCCESS, expected: SHAKING | error_massage: None
        map_T_odom = tf.pose_to_kdl(convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                                      {
                                                                          "header": {
                                                                              "frame_id": "map",
                                                                              "seq": 0,
                                                                              "stamp": {
                                                                                  "nsecs": 0,
                                                                                  "secs": 0
                                                                              }
                                                                          },
                                                                          "pose": {
                                                                              "orientation": {
                                                                                  "w": 0.8160261838115752,
                                                                                  "x": -0.0023231211294409216,
                                                                                  "y": 6.790936736852906e-05,
                                                                                  "z": -0.5780102644680742
                                                                              },
                                                                              "position": {
                                                                                  "x": -0.24385805687427703,
                                                                                  "y": -0.42599976704485054,
                                                                                  "z": 0.0015019955625650764
                                                                              }
                                                                          }
                                                                      }
                                                                      ).pose)

        odom_T_base_footprint = PoseStamped()
        odom_T_base_footprint.pose.position = Point(
            -0.5126761856270632,
            0.04739263008351229,
            0
        )
        odom_T_base_footprint.pose.orientation = Quaternion(*quaternion_about_axis(
            3.0674680749162633,
            [0, 0, 1]))
        odom_T_base_footprint = tf.pose_to_kdl(odom_T_base_footprint.pose)
        map_T_base_footprint = tf.kdl_to_pose_stamped(map_T_odom * odom_T_base_footprint, u'map')

        kitchen_setup.teleport_base(map_T_base_footprint)

        js = {
            "l_elbow_flex_joint": -2.12086412860694,
            "l_forearm_roll_joint": 10.713341993226086,
            "l_shoulder_lift_joint": -0.26482897748116896,
            "l_shoulder_pan_joint": 1.965374844556896,
            "l_upper_arm_roll_joint": 1.3837617139225473,
            "l_wrist_flex_joint": -0.10000482823989382,
            "l_wrist_roll_joint": -7.146912844735454e-05,
            "r_elbow_flex_joint": -2.1211536700297065,
            "r_forearm_roll_joint": -4.516678153856816,
            "r_shoulder_lift_joint": -0.25706741673108136,
            "r_shoulder_pan_joint": -1.7130019826780922,
            "r_upper_arm_roll_joint": -1.4633501125737374,
            "r_wrist_flex_joint": -0.10001410944881695,
            "r_wrist_roll_joint": 12.616569004141143,
            "torso_lift_joint": 0.26241453005319815,
        }
        kitchen_setup.send_and_check_joint_goal(js)

        kitchen_setup.set_kitchen_js(
            {
                "fridge_area_lower_drawer_main_joint": 0.0,
                "iai_fridge_door_joint": 0.0,
                "kitchen_island_left_lower_drawer_main_joint": 0.0,
                "kitchen_island_left_upper_drawer_main_joint": 0.48,
                "kitchen_island_middle_lower_drawer_main_joint": 0.0,
                "kitchen_island_middle_upper_drawer_main_joint": 0.0,
                "kitchen_island_right_lower_drawer_main_joint": 0.0,
                "kitchen_island_right_upper_drawer_main_joint": 0.0,
                "oven_area_area_left_drawer_main_joint": 0.0,
                "oven_area_area_middle_lower_drawer_main_joint": 0.0,
                "oven_area_area_middle_upper_drawer_main_joint": 0.0,
                "oven_area_area_right_drawer_main_joint": 0.0,
                "oven_area_oven_door_joint": 0.0,
                "oven_area_oven_knob_oven_joint": 0.0,
                "oven_area_oven_knob_stove_1_joint": 0.0,
                "oven_area_oven_knob_stove_2_joint": 0.0,
                "oven_area_oven_knob_stove_3_joint": 0.0,
                "oven_area_oven_knob_stove_4_joint": 0.0,
                "sink_area_dish_washer_door_joint": 0.0,
                "sink_area_dish_washer_main_joint": 0.0,
                "sink_area_dish_washer_tray_main": 0.0,
                "sink_area_left_bottom_drawer_main_joint": 0.0,
                "sink_area_left_middle_drawer_main_joint": 0.0,
                "sink_area_left_upper_drawer_main_joint": 0.0,
                "sink_area_trash_drawer_main_joint": 0.0
            }
        )

        box_pose = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                     {
                                                         "header": {
                                                             "frame_id": "map",
                                                             "seq": 0,
                                                             "stamp": {
                                                                 "nsecs": 0,
                                                                 "secs": 0
                                                             }
                                                         },
                                                         "pose": {
                                                             "orientation": {
                                                                 "w": 0.9882279918997449,
                                                                 "x": -0.007652552213539503,
                                                                 "y": 0.0018437565017171212,
                                                                 "z": -0.15278571606135005
                                                             },
                                                             "position": {
                                                                 "x": -0.4951539675394694,
                                                                 "y": 0.878000005086263,
                                                                 "z": 0.5103498776753743
                                                             }
                                                         }
                                                     }
                                                     )
        kitchen_setup.add_box(u'bowl_1',
                              size=[0.0858103354772, 0.0858093341192, 0.155754343669], pose=box_pose)

        wrong_odom_T_goal = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                              {
                                                                  "header": {
                                                                      "frame_id": "odom_combined",
                                                                      "seq": 0,
                                                                      "stamp": {
                                                                          "nsecs": 687142849,
                                                                          "secs": 1600169000
                                                                      }
                                                                  },
                                                                  "pose": {
                                                                      "orientation": {
                                                                          "w": 0.6741574547197648,
                                                                          "x": 0.2227110833246904,
                                                                          "y": 0.6671277357205174,
                                                                          "z": -0.22550420426923523
                                                                      },
                                                                      "position": {
                                                                          "x": -1.3306134040577549,
                                                                          "y": 0.17057334456353979,
                                                                          "z": 0.6237496353389539
                                                                      }
                                                                  }
                                                              }
                                                              )

        map_T_cart_goal = tf.kdl_to_pose_stamped(map_T_odom * tf.pose_to_kdl(wrong_odom_T_goal.pose), u'map')

        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup.avoid_all_collisions(0.1)
        kitchen_setup.set_json_goal(u'CartesianVelocityLimit',
                                    root_link=u'odom_combined',
                                    tip_link=u'base_footprint',
                                    max_linear_velocity=0.1,
                                    max_angular_velocity=0.2,
                                    )
        kitchen_setup.set_and_check_cart_goal(map_T_cart_goal, kitchen_setup.r_tip,
                                              expected_error_codes=[MoveResult.SHAKING])

    def test_bug2020_09_16_14_21_09_dump_grasp_wiggle_drawer(self, kitchen_setup):
        map_T_odom = tf.pose_to_kdl(convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                                      {
                                                                          "header": {
                                                                              "frame_id": "map",
                                                                              "seq": 0,
                                                                              "stamp": {
                                                                                  "nsecs": 0,
                                                                                  "secs": 0
                                                                              }
                                                                          },
                                                                          "pose": {
                                                                              "orientation": {
                                                                                  "w": 0.9968175080018548,
                                                                                  "x": 0.0011748416001041227,
                                                                                  "y": -0.004088826846478855,
                                                                                  "z": 0.0796037498061895
                                                                              },
                                                                              "position": {
                                                                                  "x": -0.08506470011480943,
                                                                                  "y": 0.012034760341859746,
                                                                                  "z": -0.006754371386816493
                                                                              }
                                                                          }
                                                                      }
                                                                      ).pose)

        odom_T_base_footprint = PoseStamped()
        odom_T_base_footprint.pose.position = Point(
            0.5904346300256861,
            1.0826631089335181,
            0
        )
        odom_T_base_footprint.pose.orientation = Quaternion(*quaternion_about_axis(
            -0.8210085042084164,
            [0, 0, 1]))
        odom_T_base_footprint = tf.pose_to_kdl(odom_T_base_footprint.pose)
        map_T_base_footprint = tf.kdl_to_pose_stamped(map_T_odom * odom_T_base_footprint, u'map')

        kitchen_setup.teleport_base(map_T_base_footprint)

        js = {
            "l_elbow_flex_joint": -2.098424668342493,
            "l_forearm_roll_joint": 86.11817677502071,
            "l_shoulder_lift_joint": -0.2643214079425423,
            "l_shoulder_pan_joint": 1.9710954011545279,
            "l_upper_arm_roll_joint": 1.3941847776365899,
            "l_wrist_flex_joint": -0.10004833715999412,
            "l_wrist_roll_joint": 0.0011032717142405168,
            "r_elbow_flex_joint": -2.1211536700297065,
            "r_forearm_roll_joint": -42.215798260797946,
            "r_shoulder_lift_joint": -0.25689822688487246,
            "r_shoulder_pan_joint": -1.7125874495913074,
            "r_upper_arm_roll_joint": -1.4633501125737374,
            "r_wrist_flex_joint": -0.10014463620911673,
            "r_wrist_roll_joint": 25.184338119509484,
            "torso_lift_joint": 0.26079355811100835,
        }
        kitchen_setup.send_and_check_joint_goal(js)

        kitchen_setup.set_kitchen_js(
            {
                "fridge_area_lower_drawer_main_joint": 0.0,
                "iai_fridge_door_joint": 0.0,
                "kitchen_island_left_lower_drawer_main_joint": 0.0,
                "kitchen_island_left_upper_drawer_main_joint": 0.0,
                "kitchen_island_middle_lower_drawer_main_joint": 0.0,
                "kitchen_island_middle_upper_drawer_main_joint": 0.0,
                "kitchen_island_right_lower_drawer_main_joint": 0.0,
                "kitchen_island_right_upper_drawer_main_joint": 0.0,
                "oven_area_area_left_drawer_main_joint": 0.0,
                "oven_area_area_middle_lower_drawer_main_joint": 0.0,
                "oven_area_area_middle_upper_drawer_main_joint": 0.0,
                "oven_area_area_right_drawer_main_joint": 0.0,
                "oven_area_oven_door_joint": 0.0,
                "oven_area_oven_knob_oven_joint": 0.0,
                "oven_area_oven_knob_stove_1_joint": 0.0,
                "oven_area_oven_knob_stove_2_joint": 0.0,
                "oven_area_oven_knob_stove_3_joint": 0.0,
                "oven_area_oven_knob_stove_4_joint": 0.0,
                "sink_area_dish_washer_door_joint": 0.0,
                "sink_area_dish_washer_main_joint": 0.0,
                "sink_area_dish_washer_tray_main": 0.0,
                "sink_area_left_bottom_drawer_main_joint": 0.0,
                "sink_area_left_middle_drawer_main_joint": 0.0,
                "sink_area_left_upper_drawer_main_joint": 0.48,
                "sink_area_trash_drawer_main_joint": 0.0
            }
        )

        wrong_odom_T_goal = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                              {
                                                                  "header": {
                                                                      "frame_id": "odom_combined",
                                                                      "seq": 0,
                                                                      "stamp": {
                                                                          "nsecs": 978833199,
                                                                          "secs": 1600266051
                                                                      }
                                                                  },
                                                                  "pose": {
                                                                      "orientation": {
                                                                          "w": -0.09828102468482453,
                                                                          "x": -0.6993380143702271,
                                                                          "y": -0.08220840563426661,
                                                                          "z": 0.7032133567930817
                                                                      },
                                                                      "position": {
                                                                          "x": 1.4459541048718576,
                                                                          "y": 0.9500928201148762,
                                                                          "z": 0.9194207479813763
                                                                      }
                                                                  }
                                                              }
                                                              )

        map_T_cart_goal = tf.kdl_to_pose_stamped(map_T_odom * tf.pose_to_kdl(wrong_odom_T_goal.pose), u'map')

        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=40)
        # kitchen_setup.avoid_all_collisions(0.1)
        kitchen_setup.set_json_goal(u'CartesianVelocityLimit',
                                    root_link=u'odom_combined',
                                    tip_link=u'base_footprint',
                                    max_linear_velocity=0.05,
                                    max_angular_velocity=0.1,
                                    )
        kitchen_setup.set_and_check_cart_goal(map_T_cart_goal, kitchen_setup.r_tip)

    def test_bug2020_09_22_10_18_17_dump_shaking_fridge(self, kitchen_setup):
        # fixme shaking detected shaking of joint: 'l_wrist_roll_joint' at 10.0 hertz: 0.310023530937 > 0.23
        map_T_odom = tf.pose_to_kdl(convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                                      {
                                                                          "header": {
                                                                              "frame_id": "map",
                                                                              "seq": 0,
                                                                              "stamp": {
                                                                                  "nsecs": 0,
                                                                                  "secs": 0
                                                                              }
                                                                          },
                                                                          "pose": {
                                                                              "orientation": {
                                                                                  "w": 0.2193882926886813,
                                                                                  "x": -0.0023274834025038566,
                                                                                  "y": -0.008655212394012138,
                                                                                  "z": 0.9755964571228061
                                                                              },
                                                                              "position": {
                                                                                  "x": 0.02668980573710477,
                                                                                  "y": -0.6692684739103011,
                                                                                  "z": 0.0014505867624489428
                                                                              }
                                                                          }
                                                                      }
                                                                      ).pose)

        odom_T_base_footprint = PoseStamped()
        odom_T_base_footprint.pose.position = Point(
            -0.38675678965051236,
            0.09772457084285956,
            0
        )
        odom_T_base_footprint.pose.orientation = Quaternion(*quaternion_about_axis(
            -2.9095261862586757,
            [0, 0, 1]))
        odom_T_base_footprint = tf.pose_to_kdl(odom_T_base_footprint.pose)
        map_T_base_footprint = tf.kdl_to_pose_stamped(map_T_odom * odom_T_base_footprint, u'map')

        kitchen_setup.teleport_base(map_T_base_footprint)

        js = {
            "l_elbow_flex_joint": -0.8455789320294437,
            "l_forearm_roll_joint": 113.1755732252759,
            "l_shoulder_lift_joint": 0.2904520977764714,
            "l_shoulder_pan_joint": 0.2007904007308109,
            "l_upper_arm_roll_joint": 1.4884734155420831,
            "l_wrist_flex_joint": -0.7362792757763624,
            "l_wrist_roll_joint": -1.3458893856491017,
            "r_elbow_flex_joint": -1.727087793643744,
            "r_forearm_roll_joint": -61.06550293198588,
            "r_shoulder_lift_joint": -0.023670023885890915,
            "r_shoulder_pan_joint": -1.595440399265886,
            "r_upper_arm_roll_joint": -1.4631897577473676,
            "r_wrist_flex_joint": -0.47993399975846973,
            "r_wrist_roll_joint": 31.466634601610018,
            "torso_lift_joint": 0.26236881004994694,
        }
        kitchen_setup.allow_all_collisions()
        kitchen_setup.send_and_check_joint_goal(js)

        kitchen_setup.set_kitchen_js(
            {
                "fridge_area_lower_drawer_main_joint": 0.0,
                "iai_fridge_door_joint": 1.5707,
                "kitchen_island_left_lower_drawer_main_joint": 0.0,
                "kitchen_island_left_upper_drawer_main_joint": 0.0,
                "kitchen_island_middle_lower_drawer_main_joint": 0.0,
                "kitchen_island_middle_upper_drawer_main_joint": 0.0,
                "kitchen_island_right_lower_drawer_main_joint": 0.0,
                "kitchen_island_right_upper_drawer_main_joint": 0.0,
                "oven_area_area_left_drawer_main_joint": 0.0,
                "oven_area_area_middle_lower_drawer_main_joint": 0.0,
                "oven_area_area_middle_upper_drawer_main_joint": 0.0,
                "oven_area_area_right_drawer_main_joint": 0.0,
                "oven_area_oven_door_joint": 0.0,
                "oven_area_oven_knob_oven_joint": 0.0,
                "oven_area_oven_knob_stove_1_joint": 0.0,
                "oven_area_oven_knob_stove_2_joint": 0.0,
                "oven_area_oven_knob_stove_3_joint": 0.0,
                "oven_area_oven_knob_stove_4_joint": 0.0,
                "sink_area_dish_washer_door_joint": 0.0,
                "sink_area_dish_washer_main_joint": 0.0,
                "sink_area_dish_washer_tray_main": 0.0,
                "sink_area_left_bottom_drawer_main_joint": 0.0,
                "sink_area_left_middle_drawer_main_joint": 0.0,
                "sink_area_left_upper_drawer_main_joint": 0.0,
                "sink_area_trash_drawer_main_joint": 0.0
            }
        )

        tip = "l_wrist_roll_link"
        box_pose = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                     {
                                                         "header": {
                                                             "frame_id": "l_wrist_roll_link",
                                                             "seq": 0,
                                                             "stamp": {
                                                                 "nsecs": 0,
                                                                 "secs": 0
                                                             }
                                                         },
                                                         "pose": {
                                                             "orientation": {
                                                                 "w": 0.000999970501792039,
                                                                 "x": 0.006999793509130254,
                                                                 "y": -0.002999911503917002,
                                                                 "z": 0.9999705013053104
                                                             },
                                                             "position": {
                                                                 "x": 0.22,
                                                                 "y": 0.007,
                                                                 "z": -0.031
                                                             }
                                                         }
                                                     }
                                                     )
        kitchen_setup.attach_box(u'bowl_1',
                                 size=[0.0695257345835, 0.0704464276632, 0.199114735921],
                                 frame_id=tip, pose=box_pose)

        kitchen_setup.set_json_goal(u'CartesianVelocityLimit',
                                    root_link=u'base_footprint',
                                    tip_link=tip,
                                    max_linear_velocity=0.05,
                                    max_angular_velocity=0.25,
                                    # hard=False,
                                    weight=WEIGHT_BELOW_CA,
                                    )

        kitchen_setup.send_and_check_joint_goal(gaya_pose)

    def test_bug2020_09_22_14_15_44_dump_shaking_fridge(self, kitchen_setup):
        map_T_odom = tf.pose_to_kdl(convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                                      {
                                                                          "header": {
                                                                              "frame_id": "map",
                                                                              "seq": 0,
                                                                              "stamp": {
                                                                                  "nsecs": 0,
                                                                                  "secs": 0
                                                                              }
                                                                          },
                                                                          "pose": {
                                                                              "orientation": {
                                                                                  "w": 0.5653827135915083,
                                                                                  "x": -0.0028604266427044666,
                                                                                  "y": -0.008027907133118711,
                                                                                  "z": 0.8247846736199614
                                                                              },
                                                                              "position": {
                                                                                  "x": -0.37898651855755205,
                                                                                  "y": -0.2566121796947799,
                                                                                  "z": -0.0057903495420781196
                                                                              }
                                                                          }
                                                                      }
                                                                      ).pose)

        odom_T_base_footprint = PoseStamped()
        odom_T_base_footprint.pose.position = Point(
            -0.7757012410466074,
            -0.5520039046144408,
            0
        )
        odom_T_base_footprint.pose.orientation = Quaternion(*quaternion_about_axis(
            -2.406811212054331,
            [0, 0, 1]))
        odom_T_base_footprint = tf.pose_to_kdl(odom_T_base_footprint.pose)
        map_T_base_footprint = tf.kdl_to_pose_stamped(map_T_odom * odom_T_base_footprint, u'map')

        kitchen_setup.teleport_base(map_T_base_footprint)

        js = {
            "l_elbow_flex_joint": -1.0220544292059641,
            "l_forearm_roll_joint": 119.48443437519097,
            "l_shoulder_lift_joint": 0.34476203840953135,
            "l_shoulder_pan_joint": 0.2865987496952911,
            "l_upper_arm_roll_joint": 1.4642598367602302,
            "l_wrist_flex_joint": -0.4885829936493704,
            "l_wrist_roll_joint": 4.94650116591454,
            "r_elbow_flex_joint": -1.7269430229323608,
            "r_forearm_roll_joint": -61.06509800237723,
            "r_shoulder_lift_joint": -0.023670023885890915,
            "r_shoulder_pan_joint": -1.595440399265886,
            "r_upper_arm_roll_joint": -1.4718489183713415,
            "r_wrist_flex_joint": -0.47997750867857025,
            "r_wrist_roll_joint": 31.466765128370312,
            "torso_lift_joint": 0.2623769429351407,
        }
        kitchen_setup.allow_all_collisions()
        kitchen_setup.send_and_check_joint_goal(js)

        kitchen_setup.set_kitchen_js(
            {
                "fridge_area_lower_drawer_main_joint": 0.0,
                "iai_fridge_door_joint": 1.5707,
                "kitchen_island_left_lower_drawer_main_joint": 0.0,
                "kitchen_island_left_upper_drawer_main_joint": 0.0,
                "kitchen_island_middle_lower_drawer_main_joint": 0.0,
                "kitchen_island_middle_upper_drawer_main_joint": 0.0,
                "kitchen_island_right_lower_drawer_main_joint": 0.0,
                "kitchen_island_right_upper_drawer_main_joint": 0.0,
                "oven_area_area_left_drawer_main_joint": 0.0,
                "oven_area_area_middle_lower_drawer_main_joint": 0.0,
                "oven_area_area_middle_upper_drawer_main_joint": 0.0,
                "oven_area_area_right_drawer_main_joint": 0.0,
                "oven_area_oven_door_joint": 0.0,
                "oven_area_oven_knob_oven_joint": 0.0,
                "oven_area_oven_knob_stove_1_joint": 0.0,
                "oven_area_oven_knob_stove_2_joint": 0.0,
                "oven_area_oven_knob_stove_3_joint": 0.0,
                "oven_area_oven_knob_stove_4_joint": 0.0,
                "sink_area_dish_washer_door_joint": 0.0,
                "sink_area_dish_washer_main_joint": 0.0,
                "sink_area_dish_washer_tray_main": 0.0,
                "sink_area_left_bottom_drawer_main_joint": 0.0,
                "sink_area_left_middle_drawer_main_joint": 0.0,
                "sink_area_left_upper_drawer_main_joint": 0.0,
                "sink_area_trash_drawer_main_joint": 0.0
            }
        )

        tip = "l_wrist_roll_link"
        box_pose = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                     {
                                                         "header": {
                                                             "frame_id": "l_wrist_roll_link",
                                                             "seq": 0,
                                                             "stamp": {
                                                                 "nsecs": 0,
                                                                 "secs": 0
                                                             }
                                                         },
                                                         "pose": {
                                                             "orientation": {
                                                                 "w": 0.001999896006030089,
                                                                 "x": 0.005999688024340295,
                                                                 "y": -0.007999584032438233,
                                                                 "z": 0.9999480040556527
                                                             },
                                                             "position": {
                                                                 "x": 0.224,
                                                                 "y": -0.004,
                                                                 "z": -0.026
                                                             }
                                                         }
                                                     }
                                                     )
        kitchen_setup.attach_box(u'bowl_1',
                                 size=[0.0695257345835, 0.0704464276632, 0.199114735921],
                                 frame_id=tip, pose=box_pose)

        kitchen_setup.set_json_goal(u'CartesianVelocityLimit',
                                    root_link=u'base_footprint',
                                    tip_link='l_wrist_roll_link',
                                    max_linear_velocity=0.05,
                                    max_angular_velocity=0.25,
                                    # hard=False,
                                    # weight=WEIGHT_BELOW_CA,
                                    )

        kitchen_setup.set_json_goal(u'CartesianVelocityLimit',
                                    root_link=u'base_footprint',
                                    tip_link='r_wrist_roll_link',
                                    max_linear_velocity=0.05,
                                    max_angular_velocity=0.25,
                                    # hard=False,
                                    # weight=WEIGHT_BELOW_CA,
                                    )
        kitchen_setup.send_and_check_joint_goal({
            "l_shoulder_pan_joint": 1.9652919379395388,
            "l_shoulder_lift_joint": -0.26499816732737785,
            "l_upper_arm_roll_joint": 1.3837617139225473,
            "l_elbow_flex_joint": -2.1224566064321584,
            "l_forearm_roll_joint": 16.99646118944817,
            "l_wrist_flex_joint": -0.07350789589924167,
            "l_wrist_roll_joint": 0.0,
            "r_shoulder_pan_joint": -1.712587449591307,
            "r_shoulder_lift_joint": -0.2567290370386635,
            "r_upper_arm_roll_joint": -1.4633501125737374,
            "r_elbow_flex_joint": -2.1221670650093913,
            "r_forearm_roll_joint": 1.7663253481913623,
            "r_wrist_flex_joint": -0.07942669250968948,
            "r_wrist_roll_joint": 0.05106258161229582
        })

    def test_bug2020_09_22_16_31_19_dump_shaking_fridge(self, kitchen_setup):
        # fixme shaking detected shaking of joint: 'l_shoulder_lift_joint' at 10.0 hertz: 0.245926307205 > 0.23
        # shaking of joint: 'l_wrist_roll_joint' at 10.0 hertz: 0.261248685913 > 0.23
        map_T_odom = tf.pose_to_kdl(convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                                      {
                                                                          "header": {
                                                                              "frame_id": "map",
                                                                              "seq": 0,
                                                                              "stamp": {
                                                                                  "nsecs": 0,
                                                                                  "secs": 0
                                                                              }
                                                                          },
                                                                          "pose": {
                                                                              "orientation": {
                                                                                  "w": 0.9961110256200812,
                                                                                  "x": 0.006263023182666383,
                                                                                  "y": -0.005177686432095072,
                                                                                  "z": -0.087731355523178
                                                                              },
                                                                              "position": {
                                                                                  "x": -0.02804723533322183,
                                                                                  "y": 1.521016808728904,
                                                                                  "z": 0.024661551440549212
                                                                              }
                                                                          }
                                                                      }
                                                                      ).pose)

        odom_T_base_footprint = PoseStamped()
        odom_T_base_footprint.pose.position = Point(
            0.749872459564353,
            -2.3662978157179393,
            0
        )
        odom_T_base_footprint.pose.orientation = Quaternion(*quaternion_about_axis(
            -0.02447645526466663,
            [0, 0, 1]))
        odom_T_base_footprint = tf.pose_to_kdl(odom_T_base_footprint.pose)
        map_T_base_footprint = tf.kdl_to_pose_stamped(map_T_odom * odom_T_base_footprint, u'map')

        kitchen_setup.teleport_base(map_T_base_footprint)

        js = {
            "l_elbow_flex_joint": -1.1999776334963197,
            "l_forearm_roll_joint": 125.80793083810454,
            "l_shoulder_lift_joint": 0.3304654964048785,
            "l_shoulder_pan_joint": 0.31122201505031577,
            "l_upper_arm_roll_joint": 1.501462156478044,
            "l_wrist_flex_joint": -0.4758783889802891,
            "l_wrist_roll_joint": 17.44974655277518,
            "r_elbow_flex_joint": -1.727087793643744,
            "r_forearm_roll_joint": -67.3347966744268,
            "r_shoulder_lift_joint": -0.023585428962786495,
            "r_shoulder_pan_joint": -1.595440399265886,
            "r_upper_arm_roll_joint": -1.4726506925031908,
            "r_wrist_flex_joint": -0.4799775086785738,
            "r_wrist_roll_joint": 37.75032336915375,
            "torso_lift_joint": 0.2481805522141022,
        }
        kitchen_setup.allow_all_collisions()
        kitchen_setup.send_and_check_joint_goal(js)

        kitchen_setup.set_kitchen_js(
            {
                "fridge_area_lower_drawer_main_joint": 0.0,
                "iai_fridge_door_joint": 1.5707,
                "kitchen_island_left_lower_drawer_main_joint": 0.0,
                "kitchen_island_left_upper_drawer_main_joint": 0.0,
                "kitchen_island_middle_lower_drawer_main_joint": 0.0,
                "kitchen_island_middle_upper_drawer_main_joint": 0.0,
                "kitchen_island_right_lower_drawer_main_joint": 0.0,
                "kitchen_island_right_upper_drawer_main_joint": 0.0,
                "oven_area_area_left_drawer_main_joint": 0.0,
                "oven_area_area_middle_lower_drawer_main_joint": 0.0,
                "oven_area_area_middle_upper_drawer_main_joint": 0.0,
                "oven_area_area_right_drawer_main_joint": 0.0,
                "oven_area_oven_door_joint": 0.0,
                "oven_area_oven_knob_oven_joint": 0.0,
                "oven_area_oven_knob_stove_1_joint": 0.0,
                "oven_area_oven_knob_stove_2_joint": 0.0,
                "oven_area_oven_knob_stove_3_joint": 0.0,
                "oven_area_oven_knob_stove_4_joint": 0.0,
                "sink_area_dish_washer_door_joint": 0.0,
                "sink_area_dish_washer_main_joint": 0.0,
                "sink_area_dish_washer_tray_main": 0.0,
                "sink_area_left_bottom_drawer_main_joint": 0.0,
                "sink_area_left_middle_drawer_main_joint": 0.0,
                "sink_area_left_upper_drawer_main_joint": 0.0,
                "sink_area_trash_drawer_main_joint": 0.0
            }
        )

        tip = "l_wrist_roll_link"
        box_pose = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                     {
                                                         "header": {
                                                             "frame_id": "l_wrist_roll_link",
                                                             "seq": 0,
                                                             "stamp": {
                                                                 "nsecs": 0,
                                                                 "secs": 0
                                                             }
                                                         },
                                                         "pose": {
                                                             "orientation": {
                                                                 "w": 0.003999824012358274,
                                                                 "x": 0.005999736017402663,
                                                                 "y": -0.005999736017408128,
                                                                 "z": 0.9999560029037843
                                                             },
                                                             "position": {
                                                                 "x": 0.229,
                                                                 "y": -0.006,
                                                                 "z": -0.028
                                                             }
                                                         }
                                                     }
                                                     )
        kitchen_setup.attach_box(u'bowl_1',
                                 size=[0.0695257345835, 0.0704464276632, 0.199114735921],
                                 frame_id=tip, pose=box_pose)

        kitchen_setup.set_json_goal(u'CartesianVelocityLimit',
                                    root_link=u'odom_combined',
                                    # root_link=u'base_footprint',
                                    tip_link='l_wrist_roll_link',
                                    max_linear_velocity=0.05,
                                    max_angular_velocity=0.25,
                                    )

        kitchen_setup.set_json_goal(u'CartesianVelocityLimit',
                                    root_link=u'odom_combined',
                                    # root_link=u'base_footprint',
                                    tip_link='r_wrist_roll_link',
                                    max_linear_velocity=0.05,
                                    max_angular_velocity=0.25,
                                    )

        # kitchen_setup.add_json_goal(u'CartesianVelocityLimit',
        #                             root_link=u'odom_combined',
        #                             tip_link='base_footprint',
        #                             max_linear_velocity=1,
        #                             max_angular_velocity=1,
        #                             )

        # base_goal = PoseStamped()
        # base_goal.header.frame_id = 'base_footprint'
        # base_goal.pose.orientation.w = 1
        # kitchen_setup.set_cart_goal(base_goal, 'base_footprint')

        kitchen_setup.send_and_check_joint_goal({
            "l_shoulder_pan_joint": 1.9652919379395388,
            "l_shoulder_lift_joint": -0.26499816732737785,
            "l_upper_arm_roll_joint": 1.3837617139225473,
            "l_elbow_flex_joint": -2.1224566064321584,
            "l_forearm_roll_joint": 16.99646118944817,
            "l_wrist_flex_joint": -0.07350789589924167,
            "l_wrist_roll_joint": 0.0,
            "r_shoulder_pan_joint": -1.712587449591307,
            "r_shoulder_lift_joint": -0.2567290370386635,
            "r_upper_arm_roll_joint": -1.4633501125737374,
            "r_elbow_flex_joint": -2.1221670650093913,
            "r_forearm_roll_joint": 1.7663253481913623,
            "r_wrist_flex_joint": -0.07942669250968948,
            "r_wrist_roll_joint": 0.05106258161229582
        })

    def test_bug2020_09_23_14_41_39_dump_shaking_dishwasher_pregrasp(self, kitchen_setup):
        map_T_odom = tf.pose_to_kdl(convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                                      {
                                                                          "header": {
                                                                              "frame_id": "map",
                                                                              "seq": 0,
                                                                              "stamp": {
                                                                                  "nsecs": 0,
                                                                                  "secs": 0
                                                                              }
                                                                          },
                                                                          "pose": {
                                                                              "orientation": {
                                                                                  "w": 0.11408216549397629,
                                                                                  "x": -0.002433897841084015,
                                                                                  "y": -0.004554878786011255,
                                                                                  "z": 0.9934578947981384
                                                                              },
                                                                              "position": {
                                                                                  "x": -0.16320947833792052,
                                                                                  "y": -0.21081622263243344,
                                                                                  "z": -0.006415371063977849
                                                                              }
                                                                          }
                                                                      }
                                                                      ).pose)

        odom_T_base_footprint = PoseStamped()
        odom_T_base_footprint.pose.position = Point(
            -0.28343443012929864,
            -0.5648738290845918,
            0
        )
        odom_T_base_footprint.pose.orientation = Quaternion(*quaternion_about_axis(
            -2.967321533971128,
            [0, 0, 1]))
        odom_T_base_footprint = tf.pose_to_kdl(odom_T_base_footprint.pose)
        map_T_base_footprint = tf.kdl_to_pose_stamped(map_T_odom * odom_T_base_footprint, u'map')

        kitchen_setup.teleport_base(map_T_base_footprint)

        js = {
            "l_elbow_flex_joint": -2.121443211452474,
            "l_forearm_roll_joint": 130.0879211074026,
            "l_shoulder_lift_joint": -0.2985823517998466,
            "l_shoulder_pan_joint": 1.9855211525746432,
            "l_upper_arm_roll_joint": 1.3622741671889826,
            "l_wrist_flex_joint": -0.10043991744088698,
            "l_wrist_roll_joint": 25.13268219072186,
            "r_elbow_flex_joint": -2.1211536700297065,
            "r_forearm_roll_joint": -48.49943808080258,
            "r_shoulder_lift_joint": -0.2567290370386635,
            "r_shoulder_pan_joint": -1.715406274581445,
            "r_upper_arm_roll_joint": -1.4633501125737374,
            "r_wrist_flex_joint": -0.1007102521704093,
            "r_wrist_roll_joint": 25.183946539228586,
            "torso_lift_joint": 0.16467285637205487,
        }
        kitchen_setup.allow_all_collisions()
        kitchen_setup.send_and_check_joint_goal(js)

        kitchen_setup.set_kitchen_js(
            {
                "fridge_area_lower_drawer_main_joint": 0.0,
                "iai_fridge_door_joint": 0.0,
                "kitchen_island_left_lower_drawer_main_joint": 0.0,
                "kitchen_island_left_upper_drawer_main_joint": 0.0,
                "kitchen_island_middle_lower_drawer_main_joint": 0.0,
                "kitchen_island_middle_upper_drawer_main_joint": 0.0,
                "kitchen_island_right_lower_drawer_main_joint": 0.0,
                "kitchen_island_right_upper_drawer_main_joint": 0.0,
                "oven_area_area_left_drawer_main_joint": 0.0,
                "oven_area_area_middle_lower_drawer_main_joint": 0.0,
                "oven_area_area_middle_upper_drawer_main_joint": 0.0,
                "oven_area_area_right_drawer_main_joint": 0.0,
                "oven_area_oven_door_joint": 0.0,
                "oven_area_oven_knob_oven_joint": 0.0,
                "oven_area_oven_knob_stove_1_joint": 0.0,
                "oven_area_oven_knob_stove_2_joint": 0.0,
                "oven_area_oven_knob_stove_3_joint": 0.0,
                "oven_area_oven_knob_stove_4_joint": 0.0,
                "sink_area_dish_washer_door_joint": 0.95,
                "sink_area_dish_washer_main_joint": 0.0,
                "sink_area_dish_washer_tray_main": 0.0,
                "sink_area_left_bottom_drawer_main_joint": 0.0,
                "sink_area_left_middle_drawer_main_joint": 0.0,
                "sink_area_left_upper_drawer_main_joint": 0.0,
                "sink_area_trash_drawer_main_joint": 0.0
            }
        )

        tip = "l_wrist_roll_link"
        # box_pose = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
        #                                              {
        #                                                  "header": {
        #                                                      "frame_id": "l_wrist_roll_link",
        #                                                      "seq": 0,
        #                                                      "stamp": {
        #                                                          "nsecs": 0,
        #                                                          "secs": 0
        #                                                      }
        #                                                  },
        #                                                  "pose": {
        #                                                      "orientation": {
        #                                                          "w": 0.22845752187233953,
        #                                                          "x": 0.5246708312712841,
        #                                                          "y": 0.09269007153360943,
        #                                                          "z": 0.8148228213250679
        #                                                      },
        #                                                      "position": {
        #                                                          "x": 1.24302126566569,
        #                                                          "y": 0.34845867156982424,
        #                                                          "z": 0.6850657145182292
        #                                                      }
        #                                                  }
        #                                              }
        #                                              )
        # kitchen_setup.add_box(u'bowl_1',
        #                       size=[0.10578233401, 0.171745332082, 0.171740325292],
        #                       pose=box_pose)

        wrong_odom_T_goal = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                              {
                                                                  "header": {
                                                                      "frame_id": "odom_combined",
                                                                      "seq": 0,
                                                                      "stamp": {
                                                                          "nsecs": 0,
                                                                          "secs": 0
                                                                      }
                                                                  },
                                                                  "pose": {
                                                                      "orientation": {
                                                                          "w": -0.24653031696884192,
                                                                          "x": -0.39295171047859817,
                                                                          "y": 0.5914696789782642,
                                                                          "z": -0.6595266473901238
                                                                      },
                                                                      "position": {
                                                                          "x": -0.7612966899290812,
                                                                          "y": -0.6022479234367364,
                                                                          "z": 0.3707253045365369
                                                                      }
                                                                  }
                                                              }
                                                              )

        map_T_cart_goal = tf.kdl_to_pose_stamped(map_T_odom * tf.pose_to_kdl(wrong_odom_T_goal.pose), u'map')

        # base_goal = PoseStamped()
        # base_goal.header.frame_id = u'base_footprint'
        # base_goal.pose.position.y += 0.3
        # base_goal.pose.orientation.w = 1
        # kitchen_setup.move_base(base_goal)

        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=40)

        kitchen_setup.set_json_goal(u'CartesianVelocityLimit',
                                    root_link=u'odom_combined',
                                    tip_link='base_footprint',
                                    max_linear_velocity=0.05,
                                    max_angular_velocity=0.1,
                                    )
        avoidance_hint = Vector3Stamped()
        avoidance_hint.header.frame_id = u'base_footprint'
        avoidance_hint.vector.y = 1

        kitchen_setup.set_json_goal(u'CollisionAvoidanceHint',
                                    link_name=u'base_link',
                                    max_threshold=0.05,
                                    spring_threshold=0.1,
                                    max_linear_velocity=1,
                                    body_b=u'pr2',
                                    link_b=u'r_wrist_flex_link',
                                    weight=WEIGHT_COLLISION_AVOIDANCE,
                                    avoidance_hint=avoidance_hint)
        kitchen_setup.set_and_check_cart_goal(map_T_cart_goal, kitchen_setup.r_tip)

    def test_bug2020_09_24_12_42_38_dump_cereal_shaking(self, kitchen_setup):
        map_T_odom = tf.pose_to_kdl(convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                                      {
                                                                          "header": {
                                                                              "frame_id": "map",
                                                                              "seq": 0,
                                                                              "stamp": {
                                                                                  "nsecs": 0,
                                                                                  "secs": 0
                                                                              }
                                                                          },
                                                                          "pose": {
                                                                              "orientation": {
                                                                                  "w": 0.9244367729204006,
                                                                                  "x": 0.0030363552712611747,
                                                                                  "y": -0.0011548192616205582,
                                                                                  "z": 0.3813215176352568
                                                                              },
                                                                              "position": {
                                                                                  "x": 0.33601843640452117,
                                                                                  "y": 0.6726240499464146,
                                                                                  "z": -0.002390805709227884
                                                                              }
                                                                          }
                                                                      }
                                                                      ).pose)

        odom_T_base_footprint = PoseStamped()
        odom_T_base_footprint.pose.position = Point(
            0.250466603803726,
            0.2679258445571168,
            0
        )
        odom_T_base_footprint.pose.orientation = Quaternion(*quaternion_about_axis(
            -0.18458974948173348,
            [0, 0, 1]))
        odom_T_base_footprint = tf.pose_to_kdl(odom_T_base_footprint.pose)
        map_T_base_footprint = tf.kdl_to_pose_stamped(map_T_odom * odom_T_base_footprint, u'map')

        kitchen_setup.teleport_base(map_T_base_footprint)

        js = {
            "head_pan_joint": 0.13790879609620701,
            "head_tilt_joint": 0.29296063589059607,
            "l_elbow_flex_joint": -2.121153670029707,
            "l_forearm_roll_joint": 167.79311515847354,
            "l_shoulder_lift_joint": -0.26152977548009537,
            "l_shoulder_pan_joint": 1.9668671636693216,
            "l_upper_arm_roll_joint": 1.384082423575287,
            "l_wrist_flex_joint": -0.10096202448208258,
            "l_wrist_roll_joint": 25.13233411936106,
            "r_elbow_flex_joint": -2.121008899318323,
            "r_forearm_roll_joint": -54.76560808054815,
            "r_shoulder_lift_joint": -0.28557590581728254,
            "r_shoulder_pan_joint": -1.7038822547688237,
            "r_upper_arm_roll_joint": -1.4841962400018227,
            "r_wrist_flex_joint": -0.1102822145923199,
            "r_wrist_roll_joint": 0.0619830915629791,
            "torso_lift_joint": 0.3249810356561252,
        }
        kitchen_setup.allow_all_collisions()
        kitchen_setup.send_and_check_joint_goal(js)

        kitchen_setup.set_kitchen_js(
            {
                "fridge_area_lower_drawer_main_joint": 0.0,
                "iai_fridge_door_joint": 0.0,
                "kitchen_island_left_lower_drawer_main_joint": 0.0,
                "kitchen_island_left_upper_drawer_main_joint": 0.0,
                "kitchen_island_middle_lower_drawer_main_joint": 0.0,
                "kitchen_island_middle_upper_drawer_main_joint": 0.0,
                "kitchen_island_right_lower_drawer_main_joint": 0.0,
                "kitchen_island_right_upper_drawer_main_joint": 0.0,
                "oven_area_area_left_drawer_main_joint": 0.0,
                "oven_area_area_middle_lower_drawer_main_joint": 0.0,
                "oven_area_area_middle_upper_drawer_main_joint": 0.0,
                "oven_area_area_right_drawer_main_joint": 0.48,
                "oven_area_oven_door_joint": 0.0,
                "oven_area_oven_knob_oven_joint": 0.0,
                "oven_area_oven_knob_stove_1_joint": 0.0,
                "oven_area_oven_knob_stove_2_joint": 0.0,
                "oven_area_oven_knob_stove_3_joint": 0.0,
                "oven_area_oven_knob_stove_4_joint": 0.0,
                "sink_area_dish_washer_door_joint": 0.0,
                "sink_area_dish_washer_main_joint": 0.0,
                "sink_area_dish_washer_tray_main": 0.0,
                "sink_area_left_bottom_drawer_main_joint": 0.0,
                "sink_area_left_middle_drawer_main_joint": 0.0,
                "sink_area_left_upper_drawer_main_joint": 0.0,
                "sink_area_trash_drawer_main_joint": 0.0
            }
        )

        tip = "l_wrist_roll_link"
        box_pose = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                     {
                                                         "header": {
                                                             "frame_id": "r_wrist_roll_link",
                                                             "seq": 0,
                                                             "stamp": {
                                                                 "nsecs": 0,
                                                                 "secs": 0
                                                             }
                                                         },
                                                         "pose": {
                                                             "orientation": {
                                                                 "w": -0.003999918000550195,
                                                                 "x": -0.003999918002514649,
                                                                 "y": 0.002999938501901191,
                                                                 "z": 0.9999795006303613
                                                             },
                                                             "position": {
                                                                 "x": 0.188,
                                                                 "y": 0.016,
                                                                 "z": -0.034
                                                             }
                                                         }
                                                     }
                                                     )
        kitchen_setup.attach_box(u'bowl_1',
                                 size=[0.152831586202, 0.06345692873, 0.228943316142],
                                 frame_id=u'r_wrist_roll_link',
                                 pose=box_pose)

        wrong_odom_T_goal = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                              {
                                                                  "header": {
                                                                      "frame_id": "odom_combined",
                                                                      "seq": 0,
                                                                      "stamp": {
                                                                          "nsecs": 572013378,
                                                                          "secs": 1600951355
                                                                      }
                                                                  },
                                                                  "pose": {
                                                                      "orientation": {
                                                                          "w": 0.9233107716447588,
                                                                          "x": -0.001343573090303712,
                                                                          "y": 0.0028855260882010617,
                                                                          "z": 0.3840404764022997
                                                                      },
                                                                      "position": {
                                                                          "x": 1.0246044281713114,
                                                                          "y": 0.14666530903909242,
                                                                          "z": 1.4647596925502677
                                                                      }
                                                                  }
                                                              }
                                                              )

        map_T_cart_goal = tf.kdl_to_pose_stamped(map_T_odom * tf.pose_to_kdl(wrong_odom_T_goal.pose), u'map')

        # base_goal = PoseStamped()
        # base_goal.header.frame_id = u'base_footprint'
        # base_goal.pose.position.y += 0.3
        # base_goal.pose.orientation.w = 1
        # kitchen_setup.move_base(base_goal)

        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=40)

        kitchen_setup.set_json_goal(u'CartesianVelocityLimit',
                                    root_link=u'odom_combined',
                                    tip_link='base_footprint',
                                    max_linear_velocity=0.05,
                                    max_angular_velocity=0.1,
                                    )
        # avoidance_hint = Vector3Stamped()
        # avoidance_hint.header.frame_id = u'base_footprint'
        # avoidance_hint.vector.y = 1
        #
        # kitchen_setup.add_json_goal(u'CollisionAvoidanceHint',
        #                             link_name=u'base_link',
        #                             max_threshold=0.05,
        #                             spring_threshold=0.1,
        #                             max_velocity=1,
        #                             body_b=u'pr2',
        #                             link_b=u'r_wrist_flex_link',
        #                             weight=WEIGHT_COLLISION_AVOIDANCE,
        #                             avoidance_hint=avoidance_hint)
        kitchen_setup.set_cart_goal(map_T_cart_goal, kitchen_setup.r_tip)
        kitchen_setup.send_and_check_goal(goal_type=MoveGoal.PLAN_AND_EXECUTE_AND_CUT_OFF_SHAKING)

    def test_bug2020_09_24_23_37_34_dump_shaky_navigation(self, kitchen_setup):
        map_T_odom = tf.pose_to_kdl(convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                                      {
                                                                          "header": {
                                                                              "frame_id": "map",
                                                                              "seq": 0,
                                                                              "stamp": {
                                                                                  "nsecs": 0,
                                                                                  "secs": 0
                                                                              }
                                                                          },
                                                                          "pose": {
                                                                              "orientation": {
                                                                                  "w": 0.9998619603433222,
                                                                                  "x": 0.0,
                                                                                  "y": 0.0,
                                                                                  "z": 0.01661506119184534
                                                                              },
                                                                              "position": {
                                                                                  "x": -0.2916058355299374,
                                                                                  "y": -0.12452888287703048,
                                                                                  "z": 0.0
                                                                              }
                                                                          }
                                                                      }
                                                                      ).pose)

        odom_T_base_footprint = PoseStamped()
        odom_T_base_footprint.pose.position = Point(
            0.26437903064558654,
            1.9169635448789215,
            0
        )
        odom_T_base_footprint.pose.orientation = Quaternion(*quaternion_about_axis(
            -2.668025563175793,
            [0, 0, 1]))
        odom_T_base_footprint = tf.pose_to_kdl(odom_T_base_footprint.pose)
        map_T_base_footprint = tf.kdl_to_pose_stamped(map_T_odom * odom_T_base_footprint, u'map')

        kitchen_setup.teleport_base(map_T_base_footprint)

        js = {
            "head_pan_joint": -0.060430420100428574,
            "head_tilt_joint": 1.1056797645197434,
            "l_elbow_flex_joint": -1.9330965159425042,
            "l_forearm_roll_joint": 76.74538681768225,
            "l_shoulder_lift_joint": -0.1149267737400752,
            "l_shoulder_pan_joint": 1.0535678668647663,
            "l_upper_arm_roll_joint": 1.4567231599208457,
            "l_wrist_flex_joint": -1.9148799812342445,
            "l_wrist_roll_joint": -34.79822097912522,
            "r_elbow_flex_joint": -2.121008899318323,
            "r_forearm_roll_joint": -17.083321475909628,
            "r_shoulder_lift_joint": -0.25664444211555915,
            "r_shoulder_pan_joint": -1.712670356208664,
            "r_upper_arm_roll_joint": -1.4633501125737376,
            "r_wrist_flex_joint": -0.10001076260880781,
            "r_wrist_roll_joint": -6.23214323691832,
            "torso_lift_joint": 0.32500004902286184,
        }
        kitchen_setup.allow_all_collisions()
        kitchen_setup.send_and_check_joint_goal(js)

        kitchen_setup.set_kitchen_js(
            {
                "fridge_area_lower_drawer_main_joint": 0.0,
                "iai_fridge_door_joint": 0.0,
                "kitchen_island_left_lower_drawer_main_joint": 0.0,
                "kitchen_island_left_upper_drawer_main_joint": 0.48,
                "kitchen_island_middle_lower_drawer_main_joint": 0.0,
                "kitchen_island_middle_upper_drawer_main_joint": 0.0,
                "kitchen_island_right_lower_drawer_main_joint": 0.0,
                "kitchen_island_right_upper_drawer_main_joint": 0.0,
                "oven_area_area_left_drawer_main_joint": 0.0,
                "oven_area_area_middle_lower_drawer_main_joint": 0.0,
                "oven_area_area_middle_upper_drawer_main_joint": 0.0,
                "oven_area_area_right_drawer_main_joint": 0.0,
                "oven_area_oven_door_joint": 0.0,
                "oven_area_oven_knob_oven_joint": 0.0,
                "oven_area_oven_knob_stove_1_joint": 0.0,
                "oven_area_oven_knob_stove_2_joint": 0.0,
                "oven_area_oven_knob_stove_3_joint": 0.0,
                "oven_area_oven_knob_stove_4_joint": 0.0,
                "sink_area_dish_washer_door_joint": 0.0,
                "sink_area_dish_washer_main_joint": 0.0,
                "sink_area_dish_washer_tray_main": 0.0,
                "sink_area_left_bottom_drawer_main_joint": 0.0,
                "sink_area_left_middle_drawer_main_joint": 0.0,
                "sink_area_left_upper_drawer_main_joint": 0.0,
                "sink_area_trash_drawer_main_joint": 0.0
            }
        )

        tip = "l_wrist_roll_link"
        box_pose = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                     {
                                                         "header": {
                                                             "frame_id": "l_wrist_roll_link",
                                                             "seq": 0,
                                                             "stamp": {
                                                                 "nsecs": 0,
                                                                 "secs": 0
                                                             }
                                                         },
                                                         "pose": {
                                                             "orientation": {
                                                                 "w": -0.49697863037765766,
                                                                 "x": 0.5049782864014414,
                                                                 "y": 0.49397875937069785,
                                                                 "z": -0.5039783293968079
                                                             },
                                                             "position": {
                                                                 "x": 0.258,
                                                                 "y": 0.067,
                                                                 "z": -0.016
                                                             }
                                                         }
                                                     }
                                                     )
        kitchen_setup.add_mesh(u'bowl_1',
                               path=u'package://giskardpy/test/urdfs/meshes/cup_11.obj',
                               pose=box_pose)
        kitchen_setup.attach_object(u'bowl_1',
                                    frame_id=tip)

        map_T_cart_goal = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                            {"header": {"stamp": {"secs": 1600990649,
                                                                                  "nsecs": 781974.0772247314},
                                                                        "frame_id": "map", "seq": 0}, "pose": {
                                                                "position": {"x": -2.4800000190734863,
                                                                             "y": 0.440000057220459, "z": 0.0},
                                                                "orientation": {"x": 0.0, "y": 0.0,
                                                                                "z": -0.971928447021743,
                                                                                "w": 0.23527663264740736}}}
                                                            )

        goal_point = convert_dictionary_to_ros_message(u'geometry_msgs/PointStamped',
                                                       {"header": {"stamp": {"secs": 0, "nsecs": 0.0},
                                                                   "frame_id": "base_footprint",
                                                                   "seq": 0},
                                                        "point": {"x": 1.0, "y": 0.0, "z": 0.0}})
        kitchen_setup.set_json_goal(u'Pointing',
                                    root_link=u'base_footprint',
                                    tip_link=u'narrow_stereo_optical_frame',
                                    goal_point=goal_point)

        # kitchen_setup.add_json_goal(u'AvoidJointLimits', percentage=40)

        kitchen_setup.set_json_goal(u'CartesianVelocityLimit',
                                    root_link=u'odom_combined',
                                    tip_link='base_footprint',
                                    max_linear_velocity=0.25,
                                    max_angular_velocity=0.4,
                                    )

        avoidance_hint = Vector3Stamped()
        avoidance_hint.header.frame_id = u'base_footprint'
        avoidance_hint.vector.y = 1
        kitchen_setup.set_json_goal(u'CollisionAvoidanceHint',
                                    link_name=u'base_link',
                                    max_threshold=0.05,
                                    spring_threshold=0.1,
                                    max_linear_velocity=1,
                                    body_b=u'pr2',
                                    link_b=u'r_wrist_flex_link',
                                    weight=WEIGHT_COLLISION_AVOIDANCE,
                                    avoidance_hint=avoidance_hint)
        kitchen_setup.set_cart_goal(map_T_cart_goal, u'base_footprint', linear_velocity=0.25,
                                    weight=WEIGHT_BELOW_CA)
        kitchen_setup.set_joint_goal({
            "l_shoulder_pan_joint": 1.0251308971113202,
            "l_shoulder_lift_joint": -0.051142201719316396,
            "l_upper_arm_roll_joint": 1.4663444495030389,
            "l_elbow_flex_joint": -1.9328069745197372,
            "l_forearm_roll_joint": 76.74521327642138,
            "l_wrist_flex_joint": -1.8806384611158626,
            "l_wrist_roll_joint": -34.788866561303806,
            "r_shoulder_pan_joint": -1.712587449591307,
            "r_shoulder_lift_joint": -0.25664444211555915,
            "r_upper_arm_roll_joint": -1.4633501125737376,
            "r_elbow_flex_joint": -2.121153670029707,
            "r_forearm_roll_joint": -17.083263628822678,
            "r_wrist_flex_joint": -0.10001076260880781,
            "r_wrist_roll_joint": -6.23214323691832
        })
        kitchen_setup.avoid_all_collisions(0.2)
        kitchen_setup.send_and_check_goal()

    def test_bug2020_09_23_14_41_39_dump_shaking_dishwasher(self, kitchen_setup):
        # fixme
        map_T_odom = tf.pose_to_kdl(convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                                      {
                                                                          "header": {
                                                                              "frame_id": "map",
                                                                              "seq": 0,
                                                                              "stamp": {
                                                                                  "nsecs": 0,
                                                                                  "secs": 0
                                                                              }
                                                                          },
                                                                          "pose": {
                                                                              "orientation": {
                                                                                  "w": 0.09031727466666487,
                                                                                  "x": -0.0004497013398613231,
                                                                                  "y": 0.0004948283836485173,
                                                                                  "z": 0.9959128188804289
                                                                              },
                                                                              "position": {
                                                                                  "x": -0.12686795283770538,
                                                                                  "y": -0.2572401017912599,
                                                                                  "z": 0.0006729246475819154
                                                                              }
                                                                          }
                                                                      }
                                                                      ).pose)

        odom_T_base_footprint = PoseStamped()
        odom_T_base_footprint.pose.position = Point(
            -0.4493578039830778,
            -1.2370549411982148,
            0
        )
        odom_T_base_footprint.pose.orientation = Quaternion(*quaternion_about_axis(
            2.6732432670398554,
            [0, 0, 1]))
        odom_T_base_footprint = tf.pose_to_kdl(odom_T_base_footprint.pose)
        map_T_base_footprint = tf.kdl_to_pose_stamped(map_T_odom * odom_T_base_footprint, u'map')

        kitchen_setup.teleport_base(map_T_base_footprint)

        js = {
            "l_elbow_flex_joint": -2.120719357895556,
            "l_forearm_roll_joint": 130.09538338161923,
            "l_shoulder_lift_joint": -0.28267850625620894,
            "l_shoulder_pan_joint": 1.8997128036101634,
            "l_upper_arm_roll_joint": 1.36580197336912,
            "l_wrist_flex_joint": -0.10104904232228185,
            "l_wrist_roll_joint": 25.131899030160067,
            "r_elbow_flex_joint": -2.121008899318323,
            "r_forearm_roll_joint": -48.49920669245478,
            "r_shoulder_lift_joint": -0.2955581067436082,
            "r_shoulder_pan_joint": -1.6014925823329464,
            "r_upper_arm_roll_joint": -1.4224596318494167,
            "r_wrist_flex_joint": -0.10388640333767984,
            "r_wrist_roll_joint": 25.18512128007127,
            "torso_lift_joint": 0.14616438794053993,
        }
        kitchen_setup.allow_all_collisions()
        kitchen_setup.send_and_check_joint_goal(js)

        kitchen_setup.set_kitchen_js(
            {
                "fridge_area_lower_drawer_main_joint": 0.0,
                "iai_fridge_door_joint": 0.0,
                "kitchen_island_left_lower_drawer_main_joint": 0.0,
                "kitchen_island_left_upper_drawer_main_joint": 0.0,
                "kitchen_island_middle_lower_drawer_main_joint": 0.0,
                "kitchen_island_middle_upper_drawer_main_joint": 0.0,
                "kitchen_island_right_lower_drawer_main_joint": 0.0,
                "kitchen_island_right_upper_drawer_main_joint": 0.0,
                "oven_area_area_left_drawer_main_joint": 0.0,
                "oven_area_area_middle_lower_drawer_main_joint": 0.0,
                "oven_area_area_middle_upper_drawer_main_joint": 0.0,
                "oven_area_area_right_drawer_main_joint": 0.0,
                "oven_area_oven_door_joint": 0.0,
                "oven_area_oven_knob_oven_joint": 0.0,
                "oven_area_oven_knob_stove_1_joint": 0.0,
                "oven_area_oven_knob_stove_2_joint": 0.0,
                "oven_area_oven_knob_stove_3_joint": 0.0,
                "oven_area_oven_knob_stove_4_joint": 0.0,
                "sink_area_dish_washer_door_joint": 0.95,
                "sink_area_dish_washer_main_joint": 0.0,
                "sink_area_dish_washer_tray_main": 0.0,
                "sink_area_left_bottom_drawer_main_joint": 0.0,
                "sink_area_left_middle_drawer_main_joint": 0.0,
                "sink_area_left_upper_drawer_main_joint": 0.0,
                "sink_area_trash_drawer_main_joint": 0.0
            }
        )

        tip = "l_wrist_roll_link"
        box_pose = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                     {
                                                         "header": {
                                                             "frame_id": "l_wrist_roll_link",
                                                             "seq": 0,
                                                             "stamp": {
                                                                 "nsecs": 0,
                                                                 "secs": 0
                                                             }
                                                         },
                                                         "pose": {
                                                             "orientation": {
                                                                 "w": 0.22845752187233953,
                                                                 "x": 0.5246708312712841,
                                                                 "y": 0.09269007153360943,
                                                                 "z": 0.8148228213250679
                                                             },
                                                             "position": {
                                                                 "x": 1.24302126566569,
                                                                 "y": 0.34845867156982424,
                                                                 "z": 0.6850657145182292
                                                             }
                                                         }
                                                     }
                                                     )
        kitchen_setup.add_box(u'bowl_1',
                              size=[0.10578233401, 0.171745332082, 0.171740325292],
                              pose=box_pose)

        wrong_odom_T_goal = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                              {
                                                                  "header": {
                                                                      "frame_id": "odom_combined",
                                                                      "seq": 0,
                                                                      "stamp": {
                                                                          "nsecs": 346399307,
                                                                          "secs": 1600872096
                                                                      }
                                                                  },
                                                                  "pose": {
                                                                      "orientation": {
                                                                          "w": -0.2651488857595898,
                                                                          "x": -0.3784163478285344,
                                                                          "y": 0.5968816112245744,
                                                                          "z": -0.6559188227057068
                                                                      },
                                                                      "position": {
                                                                          "x": -0.8011662710392221,
                                                                          "y": -0.6108487047237943,
                                                                          "z": 0.4534888328274996
                                                                      }
                                                                  }
                                                              }
                                                              )

        map_T_cart_goal = tf.kdl_to_pose_stamped(map_T_odom * tf.pose_to_kdl(wrong_odom_T_goal.pose), u'map')

        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup.set_json_goal(u'CartesianVelocityLimit',
                                    root_link=u'odom_combined',
                                    tip_link='base_footprint',
                                    max_linear_velocity=0.05,
                                    max_angular_velocity=0.1,
                                    )

        kitchen_setup.set_and_check_cart_goal(map_T_cart_goal, kitchen_setup.r_tip)

    def test_bug2020_09_16_15_09_36_dump_grasp_wiggle_drawer(self, kitchen_setup):
        map_T_odom = tf.pose_to_kdl(convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                                      {
                                                                          "header": {
                                                                              "frame_id": "map",
                                                                              "seq": 0,
                                                                              "stamp": {
                                                                                  "nsecs": 0,
                                                                                  "secs": 0
                                                                              }
                                                                          },
                                                                          "pose": {
                                                                              "orientation": {
                                                                                  "w": 0.9727439238312539,
                                                                                  "x": 0.0022333636476489127,
                                                                                  "y": -0.004829125045602607,
                                                                                  "z": 0.23182094445387627
                                                                              },
                                                                              "position": {
                                                                                  "x": 0.024580947578082497,
                                                                                  "y": 0.1013560658118105,
                                                                                  "z": 5.192380582593782e-05
                                                                              }
                                                                          }
                                                                      }
                                                                      ).pose)

        odom_T_base_footprint = PoseStamped()
        odom_T_base_footprint.pose.position = Point(
            0.14115931987581143,
            -0.7236910043884692,
            0
        )
        odom_T_base_footprint.pose.orientation = Quaternion(*quaternion_about_axis(
            -1.3665574203571444,
            [0, 0, 1]))
        odom_T_base_footprint = tf.pose_to_kdl(odom_T_base_footprint.pose)
        map_T_base_footprint = tf.kdl_to_pose_stamped(map_T_odom * odom_T_base_footprint, u'map')

        kitchen_setup.teleport_base(map_T_base_footprint)

        js = {
            "l_elbow_flex_joint": -2.1147837587288314,
            "l_forearm_roll_joint": 98.677242128683,
            "l_shoulder_lift_joint": -0.25273190347723196,
            "l_shoulder_pan_joint": 2.134670157199861,
            "l_upper_arm_roll_joint": 1.3845634880543967,
            "l_wrist_flex_joint": -0.1070532732960286,
            "l_wrist_roll_joint": -0.0011156832108381032,
            "r_elbow_flex_joint": -2.1194164214931046,
            "r_forearm_roll_joint": -48.49857037449833,
            "r_shoulder_lift_joint": -0.25224550611412744,
            "r_shoulder_pan_joint": -1.7388688472934715,
            "r_upper_arm_roll_joint": -1.4655950801429158,
            "r_wrist_flex_joint": -0.10005761836891214,
            "r_wrist_roll_joint": 31.466591092689917,
            "torso_lift_joint": 0.2624334335160808,
        }
        kitchen_setup.send_and_check_joint_goal(js)

        kitchen_setup.set_kitchen_js(
            {
                "fridge_area_lower_drawer_main_joint": 0.0,
                "iai_fridge_door_joint": 1.5707,
                "kitchen_island_left_lower_drawer_main_joint": 0.0,
                "kitchen_island_left_upper_drawer_main_joint": 0.0,
                "kitchen_island_middle_lower_drawer_main_joint": 0.0,
                "kitchen_island_middle_upper_drawer_main_joint": 0.0,
                "kitchen_island_right_lower_drawer_main_joint": 0.0,
                "kitchen_island_right_upper_drawer_main_joint": 0.0,
                "oven_area_area_left_drawer_main_joint": 0.0,
                "oven_area_area_middle_lower_drawer_main_joint": 0.0,
                "oven_area_area_middle_upper_drawer_main_joint": 0.0,
                "oven_area_area_right_drawer_main_joint": 0.0,
                "oven_area_oven_door_joint": 0.0,
                "oven_area_oven_knob_oven_joint": 0.0,
                "oven_area_oven_knob_stove_1_joint": 0.0,
                "oven_area_oven_knob_stove_2_joint": 0.0,
                "oven_area_oven_knob_stove_3_joint": 0.0,
                "oven_area_oven_knob_stove_4_joint": 0.0,
                "sink_area_dish_washer_door_joint": 0.0,
                "sink_area_dish_washer_main_joint": 0.0,
                "sink_area_dish_washer_tray_main": 0.0,
                "sink_area_left_bottom_drawer_main_joint": 0.0,
                "sink_area_left_middle_drawer_main_joint": 0.0,
                "sink_area_left_upper_drawer_main_joint": 0.0,
                "sink_area_trash_drawer_main_joint": 0.0
            }
        )

        box_pose = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                     {
                                                         "header": {
                                                             "frame_id": "map",
                                                             "seq": 0,
                                                             "stamp": {
                                                                 "nsecs": 0,
                                                                 "secs": 0
                                                             }
                                                         },
                                                         "pose": {
                                                             "orientation": {
                                                                 "w": 0.6634436986211376,
                                                                 "x": -0.692860946417934,
                                                                 "y": 0.17933559041317249,
                                                                 "z": -0.21823133070186967
                                                             },
                                                             "position": {
                                                                 "x": 0.926089096069336,
                                                                 "y": -1.0452009836832683,
                                                                 "z": 0.0329125980536143
                                                             }
                                                         }
                                                     }
                                                     )
        kitchen_setup.add_box(u'bowl_1',
                              size=[0.0695257345835, 0.199114735921, 0.0704464276632], pose=box_pose)

        wrong_odom_T_goal = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                              {
                                                                  "header": {
                                                                      "frame_id": "odom_combined",
                                                                      "seq": 0,
                                                                      "stamp": {
                                                                          "nsecs": 962248325,
                                                                          "secs": 1600268947
                                                                      }
                                                                  },
                                                                  "pose": {
                                                                      "orientation": {
                                                                          "w": 0.6756608321219002,
                                                                          "x": -0.21033058845437325,
                                                                          "y": 0.6877822510017452,
                                                                          "z": 0.161861492207766
                                                                      },
                                                                      "position": {
                                                                          "x": 0.44265312758557507,
                                                                          "y": -1.328706668357887,
                                                                          "z": 0.16673579458516688
                                                                      }
                                                                  }
                                                              }
                                                              )

        map_T_cart_goal = tf.kdl_to_pose_stamped(map_T_odom * tf.pose_to_kdl(wrong_odom_T_goal.pose), u'map')

        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=40)
        # kitchen_setup.avoid_all_collisions(0.1)
        kitchen_setup.set_json_goal(u'CartesianVelocityLimit',
                                    root_link=u'odom_combined',
                                    tip_link=u'base_footprint',
                                    max_linear_velocity=0.05,
                                    max_angular_velocity=0.1,
                                    )
        kitchen_setup.set_and_check_cart_goal(map_T_cart_goal, 'l_gripper_tool_frame')

    def test_bug2020_09_09_13_41_21_dump_corner_drawer_shaking(self, kitchen_setup):
        map_T_odom = tf.pose_to_kdl(convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                                      {
                                                                          "header": {
                                                                              "frame_id": "map",
                                                                              "seq": 0,
                                                                              "stamp": {
                                                                                  "nsecs": 0,
                                                                                  "secs": 0
                                                                              }
                                                                          },
                                                                          "pose": {
                                                                              "orientation": {
                                                                                  "w": 0.8779201932099772,
                                                                                  "x": 0.005801928233696417,
                                                                                  "y": 0.0021891835979195693,
                                                                                  "z": -0.47876683203632847
                                                                              },
                                                                              "position": {
                                                                                  "x": -1.2813072440232847,
                                                                                  "y": -0.000900015892780554,
                                                                                  "z": -0.0053088233454366375
                                                                              }
                                                                          }
                                                                      }
                                                                      ).pose)

        odom_T_base_footprint = PoseStamped()
        odom_T_base_footprint.pose.position = Point(
            0.5065161536263582,
            1.2428430519008682,
            0
        )
        odom_T_base_footprint.pose.orientation = Quaternion(*quaternion_about_axis(
            -2.9179043687346833,
            [0, 0, 1]))
        odom_T_base_footprint = tf.pose_to_kdl(odom_T_base_footprint.pose)
        map_T_base_footprint = tf.kdl_to_pose_stamped(map_T_odom * odom_T_base_footprint, u'map')

        kitchen_setup.teleport_base(map_T_base_footprint)

        js = {
            "l_elbow_flex_joint": -2.121008899318323,
            "l_forearm_roll_joint": 60.97875833969833,
            "l_shoulder_lift_joint": -0.26499816732737785,
            "l_shoulder_pan_joint": 1.965291937939539,
            "l_upper_arm_roll_joint": 1.3837617139225473,
            "l_wrist_flex_joint": -0.09896061415750301,
            "l_wrist_roll_joint": -31.415924548422947,
            "r_elbow_flex_joint": -2.121153670029707,
            "r_forearm_roll_joint": -67.34873782238188,
            "r_shoulder_lift_joint": -0.25672903703866357,
            "r_shoulder_pan_joint": -1.712587449591307,
            "r_upper_arm_roll_joint": -1.4633501125737376,
            "r_wrist_flex_joint": -0.09998099334768273,
            "r_wrist_roll_joint": 50.31658512943495,
            "torso_lift_joint": 0.2623074836994323,
        }
        kitchen_setup.send_and_check_joint_goal(js)

        kitchen_setup.set_kitchen_js(
            {
                "fridge_area_lower_drawer_main_joint": 0.0,
                "iai_fridge_door_joint": 0.0,
                "kitchen_island_left_lower_drawer_main_joint": 0.0,
                "kitchen_island_left_upper_drawer_main_joint": 0.48,
                "kitchen_island_middle_lower_drawer_main_joint": 0.0,
                "kitchen_island_middle_upper_drawer_main_joint": 0.0,
                "kitchen_island_right_lower_drawer_main_joint": 0.0,
                "kitchen_island_right_upper_drawer_main_joint": 0.0,
                "oven_area_area_left_drawer_main_joint": 0.0,
                "oven_area_area_middle_lower_drawer_main_joint": 0.0,
                "oven_area_area_middle_upper_drawer_main_joint": 0.0,
                "oven_area_area_right_drawer_main_joint": 0.0,
                "oven_area_oven_door_joint": 0.0,
                "oven_area_oven_knob_oven_joint": 0.0,
                "oven_area_oven_knob_stove_1_joint": 0.0,
                "oven_area_oven_knob_stove_2_joint": 0.0,
                "oven_area_oven_knob_stove_3_joint": 0.0,
                "oven_area_oven_knob_stove_4_joint": 0.0,
                "sink_area_dish_washer_door_joint": 0.0,
                "sink_area_dish_washer_main_joint": 0.0,
                "sink_area_dish_washer_tray_main": 0.0,
                "sink_area_left_bottom_drawer_main_joint": 0.0,
                "sink_area_left_middle_drawer_main_joint": 0.0,
                "sink_area_left_upper_drawer_main_joint": 0.0,
                "sink_area_trash_drawer_main_joint": 0.0
            }
        )

        map_T_cart_goal = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                            {"header": {"stamp":
                                                                            {"secs": 0,
                                                                             "nsecs": 43884.0},
                                                                        "frame_id": "map", "seq": 0},
                                                             "pose": {
                                                                 "position":
                                                                     {"x": -0.32000017166137695,
                                                                      "y": 0.31999969482421875, "z": 0.0},
                                                                 "orientation":
                                                                     {"x": 0.0, "y": 0.0,
                                                                      "z": 0.8443331886423444,
                                                                      "w": 0.5358185015068547}}}
                                                            )

        avoidance_hint = Vector3Stamped()
        avoidance_hint.header.frame_id = 'map'
        avoidance_hint.vector.y = -1
        kitchen_setup.set_json_goal(u'CollisionAvoidanceHint',
                                    link_name=u'base_link',
                                    max_threshold=0.3,
                                    # spring_threshold=0.5,
                                    max_linear_velocity=1,
                                    body_b=u'kitchen',
                                    link_b=u'kitchen_island',
                                    weight=WEIGHT_COLLISION_AVOIDANCE,
                                    avoidance_hint=avoidance_hint)
        kitchen_setup.set_json_goal(u'CartesianVelocityLimit',
                                    root_link=u'odom_combined',
                                    tip_link=u'base_footprint',
                                    max_linear_velocity=0.5,
                                    max_angular_velocity=0.2,
                                    )
        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.avoid_all_collisions(0.2)
        kitchen_setup.set_cart_goal(map_T_cart_goal, 'base_footprint', linear_velocity=0.5,
                                    weight=WEIGHT_BELOW_CA)
        kitchen_setup.send_and_check_goal()

    def test_bug2020_09_15_09_59_29_dump_corner_drawer_shaking(self, kitchen_setup):
        map_T_odom = tf.pose_to_kdl(convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                                      {
                                                                          "header": {
                                                                              "frame_id": "map",
                                                                              "seq": 0,
                                                                              "stamp": {
                                                                                  "nsecs": 0,
                                                                                  "secs": 0
                                                                              }
                                                                          },
                                                                          "pose": {
                                                                              "orientation": {
                                                                                  "w": 0.7579127413061545,
                                                                                  "x": 0.002869387844298475,
                                                                                  "y": 0.0003542513596710703,
                                                                                  "z": -0.6523495364336234
                                                                              },
                                                                              "position": {
                                                                                  "x": -0.32870411899363455,
                                                                                  "y": -0.28615040678711406,
                                                                                  "z": -0.005243998544918415
                                                                              }
                                                                          }
                                                                      }
                                                                      ).pose)

        odom_T_base_footprint = PoseStamped()
        odom_T_base_footprint.pose.position = Point(
            -0.25354562842435296,
            1.0658860186856896,
            0
        )
        odom_T_base_footprint.pose.orientation = Quaternion(*quaternion_about_axis(
            2.692050598317841,
            [0, 0, 1]))
        odom_T_base_footprint = tf.pose_to_kdl(odom_T_base_footprint.pose)
        map_T_base_footprint = tf.kdl_to_pose_stamped(map_T_odom * odom_T_base_footprint, u'map')

        kitchen_setup.teleport_base(map_T_base_footprint)

        js = {
            "head_pan_joint": 0.1268085020535231,
            "head_tilt_joint": 0.903331038339086,
            "l_elbow_flex_joint": -2.121008899318323,
            "l_forearm_roll_joint": 4.430280644090958,
            "l_shoulder_lift_joint": -0.2654211419429001,
            "l_shoulder_pan_joint": 1.965374844556896,
            "l_upper_arm_roll_joint": 1.3837617139225473,
            "l_wrist_flex_joint": -0.10174518504387692,
            "l_wrist_roll_joint": -7.146912844735454e-05,
            "r_elbow_flex_joint": -2.120429816472789,
            "r_forearm_roll_joint": 1.7662675011044127,
            "r_shoulder_lift_joint": -0.26104337811699074,
            "r_shoulder_pan_joint": -1.7173960333980123,
            "r_upper_arm_roll_joint": -1.4633501125737374,
            "r_wrist_flex_joint": -0.11341485683949104,
            "r_wrist_roll_joint": 0.05058375449688546,
            "torso_lift_joint": 0.2329842562296531,
        }
        kitchen_setup.send_and_check_joint_goal(js)

        kitchen_setup.set_kitchen_js(
            {
                "fridge_area_lower_drawer_main_joint": 0.0,
                "iai_fridge_door_joint": 0.0,
                "kitchen_island_left_lower_drawer_main_joint": 0.0,
                "kitchen_island_left_upper_drawer_main_joint": 0.0,
                "kitchen_island_middle_lower_drawer_main_joint": 0.0,
                "kitchen_island_middle_upper_drawer_main_joint": 0.0,
                "kitchen_island_right_lower_drawer_main_joint": 0.0,
                "kitchen_island_right_upper_drawer_main_joint": 0.0,
                "oven_area_area_left_drawer_main_joint": 0.0,
                "oven_area_area_middle_lower_drawer_main_joint": 0.0,
                "oven_area_area_middle_upper_drawer_main_joint": 0.0,
                "oven_area_area_right_drawer_main_joint": 0.0,
                "oven_area_oven_door_joint": 0.0,
                "oven_area_oven_knob_oven_joint": 0.0,
                "oven_area_oven_knob_stove_1_joint": 0.0,
                "oven_area_oven_knob_stove_2_joint": 0.0,
                "oven_area_oven_knob_stove_3_joint": 0.0,
                "oven_area_oven_knob_stove_4_joint": 0.0,
                "sink_area_dish_washer_door_joint": 0.0,
                "sink_area_dish_washer_main_joint": 0.0,
                "sink_area_dish_washer_tray_main": 0.0,
                "sink_area_left_bottom_drawer_main_joint": 0.0,
                "sink_area_left_middle_drawer_main_joint": 0.0,
                "sink_area_left_upper_drawer_main_joint": 0.0,
                "sink_area_trash_drawer_main_joint": 0.0
            }
        )

        map_T_cart_goal = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                            {"header": {"stamp": {"secs": 1600163934,
                                                                                  "nsecs": 326478.0044555664},
                                                                        "frame_id": "map", "seq": 0}, "pose": {
                                                                "position": {"x": -0.440000057220459,
                                                                             "y": 0.15999984741210938, "z": 0.0},
                                                                "orientation": {"x": 0.0, "y": 0.0,
                                                                                "z": 0.9524610762750731,
                                                                                "w": 0.30466029964688424}}}
                                                            )

        avoidance_hint = Vector3Stamped()
        avoidance_hint.header.frame_id = 'map'
        avoidance_hint.vector.y = -1
        kitchen_setup.set_json_goal(u'CollisionAvoidanceHint',
                                    link_name=u'base_link',
                                    max_threshold=0.25,
                                    spring_threshold=0.3,
                                    max_linear_velocity=1,
                                    body_b=u'kitchen',
                                    link_b=u'kitchen_island',
                                    weight=WEIGHT_COLLISION_AVOIDANCE,
                                    avoidance_hint=avoidance_hint)
        kitchen_setup.set_json_goal(u'CartesianVelocityLimit',
                                    root_link=u'odom_combined',
                                    tip_link=u'base_footprint',
                                    max_linear_velocity=0.5,
                                    max_angular_velocity=0.2,
                                    )
        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.avoid_all_collisions(0.2)
        kitchen_setup.set_cart_goal(map_T_cart_goal, 'base_footprint', linear_velocity=0.5,
                                    weight=WEIGHT_BELOW_CA)
        kitchen_setup.send_and_check_goal()

    def test_bug2020_09_15_15_35_15_dump_shaky_nav(self, kitchen_setup):
        map_T_odom = tf.pose_to_kdl(convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                                      {
                                                                          "header": {
                                                                              "frame_id": "map",
                                                                              "seq": 0,
                                                                              "stamp": {
                                                                                  "nsecs": 0,
                                                                                  "secs": 0
                                                                              }
                                                                          },
                                                                          "pose": {
                                                                              "orientation": {
                                                                                  "w": 0.9903702071222805,
                                                                                  "x": -0.0034922295028117414,
                                                                                  "y": -0.00012004497948064491,
                                                                                  "z": -0.13840029901294992
                                                                              },
                                                                              "position": {
                                                                                  "x": 0.022483169988823295,
                                                                                  "y": -0.09823512894746209,
                                                                                  "z": 0.0036762520107436565
                                                                              }
                                                                          }
                                                                      }
                                                                      ).pose)

        odom_T_base_footprint = PoseStamped()
        odom_T_base_footprint.pose.position = Point(
            0.4355780076030453,
            -0.10753565484855881,
            0
        )
        odom_T_base_footprint.pose.orientation = Quaternion(*quaternion_about_axis(
            -2.398246925955068,
            [0, 0, 1]))
        odom_T_base_footprint = tf.pose_to_kdl(odom_T_base_footprint.pose)
        map_T_base_footprint = tf.kdl_to_pose_stamped(map_T_odom * odom_T_base_footprint, u'map')

        kitchen_setup.teleport_base(map_T_base_footprint)

        js = {
            "l_elbow_flex_joint": -2.121008899318323,
            "l_forearm_roll_joint": 48.41234640599331,
            "l_shoulder_lift_joint": -0.2649981673273779,
            "l_shoulder_pan_joint": 1.965457751174253,
            "l_upper_arm_roll_joint": 1.3837617139225473,
            "l_wrist_flex_joint": -0.09965675687909759,
            "l_wrist_roll_joint": 6.283225718134377,
            "r_elbow_flex_joint": -2.1211536700297065,
            "r_forearm_roll_joint": -29.64961771544074,
            "r_shoulder_lift_joint": -0.2566444421155591,
            "r_shoulder_pan_joint": -1.7157379010508729,
            "r_upper_arm_roll_joint": -1.4633501125737374,
            "r_wrist_flex_joint": -0.1001446362091114,
            "r_wrist_roll_joint": 25.18355495894769,
        }
        kitchen_setup.send_and_check_joint_goal(js)

        kitchen_setup.set_kitchen_js(
            {
                "fridge_area_lower_drawer_main_joint": 0.0,
                "iai_fridge_door_joint": 0.0,
                "kitchen_island_left_lower_drawer_main_joint": 0.0,
                "kitchen_island_left_upper_drawer_main_joint": 0.0,
                "kitchen_island_middle_lower_drawer_main_joint": 0.0,
                "kitchen_island_middle_upper_drawer_main_joint": 0.0,
                "kitchen_island_right_lower_drawer_main_joint": 0.0,
                "kitchen_island_right_upper_drawer_main_joint": 0.0,
                "oven_area_area_left_drawer_main_joint": 0.0,
                "oven_area_area_middle_lower_drawer_main_joint": 0.0,
                "oven_area_area_middle_upper_drawer_main_joint": 0.0,
                "oven_area_area_right_drawer_main_joint": 0.0,
                "oven_area_oven_door_joint": 0.0,
                "oven_area_oven_knob_oven_joint": 0.0,
                "oven_area_oven_knob_stove_1_joint": 0.0,
                "oven_area_oven_knob_stove_2_joint": 0.0,
                "oven_area_oven_knob_stove_3_joint": 0.0,
                "oven_area_oven_knob_stove_4_joint": 0.0,
                "sink_area_dish_washer_door_joint": 0.0,
                "sink_area_dish_washer_main_joint": 0.0,
                "sink_area_dish_washer_tray_main": 0.0,
                "sink_area_left_bottom_drawer_main_joint": 0.0,
                "sink_area_left_middle_drawer_main_joint": 0.0,
                "sink_area_left_upper_drawer_main_joint": 0.0,
                "sink_area_trash_drawer_main_joint": 0.0
            }
        )

        map_T_cart_goal = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                            {"header": {"stamp": {"secs": 1600184101,
                                                                                  "nsecs": 976460.9336853027},
                                                                        "frame_id": "map", "seq": 0}, "pose": {
                                                                "position": {"x": 0.48000001907348633,
                                                                             "y": -0.5999999046325684, "z": 0.0},
                                                                "orientation": {"x": 0.0, "y": 0.0,
                                                                                "z": -0.3561185110796128,
                                                                                "w": 0.9344407985883534}}}
                                                            )

        avoidance_hint = Vector3Stamped()
        avoidance_hint.header.frame_id = 'map'
        avoidance_hint.vector.y = -1
        kitchen_setup.set_json_goal(u'CollisionAvoidanceHint',
                                    link_name=u'base_link',
                                    max_threshold=0.25,
                                    spring_threshold=0.3,
                                    max_linear_velocity=1,
                                    body_b=u'kitchen',
                                    link_b=u'kitchen_island',
                                    weight=WEIGHT_COLLISION_AVOIDANCE,
                                    avoidance_hint=avoidance_hint)
        kitchen_setup.set_json_goal(u'CartesianVelocityLimit',
                                    root_link=u'odom_combined',
                                    tip_link=u'base_footprint',
                                    max_linear_velocity=0.5,
                                    max_angular_velocity=0.2,
                                    )
        kitchen_setup.set_joint_goal(
            {
                "l_shoulder_pan_joint": 1.965457751174253,
                "l_shoulder_lift_joint": -0.26482897748116896,
                "l_upper_arm_roll_joint": 1.3837617139225473,
                "l_elbow_flex_joint": -2.121008899318323,
                "l_forearm_roll_joint": 48.412635641428075,
                "l_wrist_flex_joint": -0.09891710523740382,
                "l_wrist_roll_joint": 6.283008173533879,
                "r_shoulder_pan_joint": -1.7159037142855866,
                "r_shoulder_lift_joint": -0.2564752522693502,
                "r_upper_arm_roll_joint": -1.4633501125737374,
                "r_elbow_flex_joint": -2.121443211452474,
                "r_forearm_roll_joint": -29.649386327092937,
                "r_wrist_flex_joint": -0.09992709160861413,
                "r_wrist_roll_joint": 25.182815307305997,
            }
        )

        kitchen_setup.avoid_all_collisions(0.2)
        kitchen_setup.set_and_check_cart_goal(map_T_cart_goal, 'base_footprint', linear_velocity=0.5,
                                              weight=WEIGHT_BELOW_CA, expected_error_codes=[MoveResult.SHAKING])

    def test_bug2020_09_15_15_35_15_dump_shaky_nav_cut_off(self, kitchen_setup):
        map_T_odom = tf.pose_to_kdl(convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                                      {
                                                                          "header": {
                                                                              "frame_id": "map",
                                                                              "seq": 0,
                                                                              "stamp": {
                                                                                  "nsecs": 0,
                                                                                  "secs": 0
                                                                              }
                                                                          },
                                                                          "pose": {
                                                                              "orientation": {
                                                                                  "w": 0.9903702071222805,
                                                                                  "x": -0.0034922295028117414,
                                                                                  "y": -0.00012004497948064491,
                                                                                  "z": -0.13840029901294992
                                                                              },
                                                                              "position": {
                                                                                  "x": 0.022483169988823295,
                                                                                  "y": -0.09823512894746209,
                                                                                  "z": 0.0036762520107436565
                                                                              }
                                                                          }
                                                                      }
                                                                      ).pose)

        odom_T_base_footprint = PoseStamped()
        odom_T_base_footprint.pose.position = Point(
            0.4355780076030453,
            -0.10753565484855881,
            0
        )
        odom_T_base_footprint.pose.orientation = Quaternion(*quaternion_about_axis(
            -2.398246925955068,
            [0, 0, 1]))
        odom_T_base_footprint = tf.pose_to_kdl(odom_T_base_footprint.pose)
        map_T_base_footprint = tf.kdl_to_pose_stamped(map_T_odom * odom_T_base_footprint, u'map')

        kitchen_setup.teleport_base(map_T_base_footprint)

        js = {
            "l_elbow_flex_joint": -2.121008899318323,
            "l_forearm_roll_joint": 48.41234640599331,
            "l_shoulder_lift_joint": -0.2649981673273779,
            "l_shoulder_pan_joint": 1.965457751174253,
            "l_upper_arm_roll_joint": 1.3837617139225473,
            "l_wrist_flex_joint": -0.09965675687909759,
            "l_wrist_roll_joint": 6.283225718134377,
            "r_elbow_flex_joint": -2.1211536700297065,
            "r_forearm_roll_joint": -29.64961771544074,
            "r_shoulder_lift_joint": -0.2566444421155591,
            "r_shoulder_pan_joint": -1.7157379010508729,
            "r_upper_arm_roll_joint": -1.4633501125737374,
            "r_wrist_flex_joint": -0.1001446362091114,
            "r_wrist_roll_joint": 25.18355495894769,
        }
        kitchen_setup.send_and_check_joint_goal(js)

        kitchen_setup.set_kitchen_js(
            {
                "fridge_area_lower_drawer_main_joint": 0.0,
                "iai_fridge_door_joint": 0.0,
                "kitchen_island_left_lower_drawer_main_joint": 0.0,
                "kitchen_island_left_upper_drawer_main_joint": 0.0,
                "kitchen_island_middle_lower_drawer_main_joint": 0.0,
                "kitchen_island_middle_upper_drawer_main_joint": 0.0,
                "kitchen_island_right_lower_drawer_main_joint": 0.0,
                "kitchen_island_right_upper_drawer_main_joint": 0.0,
                "oven_area_area_left_drawer_main_joint": 0.0,
                "oven_area_area_middle_lower_drawer_main_joint": 0.0,
                "oven_area_area_middle_upper_drawer_main_joint": 0.0,
                "oven_area_area_right_drawer_main_joint": 0.0,
                "oven_area_oven_door_joint": 0.0,
                "oven_area_oven_knob_oven_joint": 0.0,
                "oven_area_oven_knob_stove_1_joint": 0.0,
                "oven_area_oven_knob_stove_2_joint": 0.0,
                "oven_area_oven_knob_stove_3_joint": 0.0,
                "oven_area_oven_knob_stove_4_joint": 0.0,
                "sink_area_dish_washer_door_joint": 0.0,
                "sink_area_dish_washer_main_joint": 0.0,
                "sink_area_dish_washer_tray_main": 0.0,
                "sink_area_left_bottom_drawer_main_joint": 0.0,
                "sink_area_left_middle_drawer_main_joint": 0.0,
                "sink_area_left_upper_drawer_main_joint": 0.0,
                "sink_area_trash_drawer_main_joint": 0.0
            }
        )

        map_T_cart_goal = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                            {"header": {"stamp": {"secs": 1600184101,
                                                                                  "nsecs": 976460.9336853027},
                                                                        "frame_id": "map", "seq": 0}, "pose": {
                                                                "position": {"x": 0.48000001907348633,
                                                                             "y": -0.5999999046325684, "z": 0.0},
                                                                "orientation": {"x": 0.0, "y": 0.0,
                                                                                "z": -0.3561185110796128,
                                                                                "w": 0.9344407985883534}}}
                                                            )

        avoidance_hint = Vector3Stamped()
        avoidance_hint.header.frame_id = 'map'
        avoidance_hint.vector.y = -1
        kitchen_setup.set_json_goal(u'CollisionAvoidanceHint',
                                    link_name=u'base_link',
                                    max_threshold=0.25,
                                    spring_threshold=0.3,
                                    max_linear_velocity=1,
                                    body_b=u'kitchen',
                                    link_b=u'kitchen_island',
                                    weight=WEIGHT_COLLISION_AVOIDANCE,
                                    avoidance_hint=avoidance_hint)
        kitchen_setup.set_json_goal(u'CartesianVelocityLimit',
                                    root_link=u'odom_combined',
                                    tip_link=u'base_footprint',
                                    max_linear_velocity=0.5,
                                    max_angular_velocity=0.2,
                                    )
        kitchen_setup.set_joint_goal(
            {
                "l_shoulder_pan_joint": 1.965457751174253,
                "l_shoulder_lift_joint": -0.26482897748116896,
                "l_upper_arm_roll_joint": 1.3837617139225473,
                "l_elbow_flex_joint": -2.121008899318323,
                "l_forearm_roll_joint": 48.412635641428075,
                "l_wrist_flex_joint": -0.09891710523740382,
                "l_wrist_roll_joint": 6.283008173533879,
                "r_shoulder_pan_joint": -1.7159037142855866,
                "r_shoulder_lift_joint": -0.2564752522693502,
                "r_upper_arm_roll_joint": -1.4633501125737374,
                "r_elbow_flex_joint": -2.121443211452474,
                "r_forearm_roll_joint": -29.649386327092937,
                "r_wrist_flex_joint": -0.09992709160861413,
                "r_wrist_roll_joint": 25.182815307305997,
            }
        )

        kitchen_setup.avoid_all_collisions(0.2)
        kitchen_setup.set_cart_goal(map_T_cart_goal, 'base_footprint', linear_velocity=0.5,
                                    weight=WEIGHT_BELOW_CA)
        kitchen_setup.send_and_check_goal(goal_type=MoveGoal.PLAN_AND_EXECUTE_AND_CUT_OFF_SHAKING)

    def test_bug2020_09_15_10_41_17_dump_wiggling(self, kitchen_setup):
        map_T_odom = tf.pose_to_kdl(convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                                      {
                                                                          "header": {
                                                                              "frame_id": "map",
                                                                              "seq": 0,
                                                                              "stamp": {
                                                                                  "nsecs": 0,
                                                                                  "secs": 0
                                                                              }
                                                                          },
                                                                          "pose": {
                                                                              "orientation": {
                                                                                  "w": 0.7532154668322877,
                                                                                  "x": 0.0013925205455504854,
                                                                                  "y": 0.0020029798643668487,
                                                                                  "z": -0.6577693436781712
                                                                              },
                                                                              "position": {
                                                                                  "x": -0.45177061803897495,
                                                                                  "y": -0.29218384670347713,
                                                                                  "z": 4.1548596924592305e-05
                                                                              }
                                                                          }
                                                                      }
                                                                      ).pose)

        odom_T_base_footprint = PoseStamped()
        odom_T_base_footprint.pose.position = Point(
            -0.10391025295767904,
            1.0221940223228667,
            0
        )
        odom_T_base_footprint.pose.orientation = Quaternion(*quaternion_about_axis(
            2.97736321896431,
            [0, 0, 1]))
        odom_T_base_footprint = tf.pose_to_kdl(odom_T_base_footprint.pose)
        map_T_base_footprint = tf.kdl_to_pose_stamped(map_T_odom * odom_T_base_footprint, u'map')

        kitchen_setup.teleport_base(map_T_base_footprint)

        js = {
            "head_pan_joint": 0.11518460923524085,
            "head_tilt_joint": 0.9514786684689828,
            "l_elbow_flex_joint": -2.121008899318323,
            "l_forearm_roll_joint": 4.430049255743156,
            "l_shoulder_lift_joint": -0.2649981673273779,
            "l_shoulder_pan_joint": 1.9652919379395388,
            "l_upper_arm_roll_joint": 1.3837617139225473,
            "l_wrist_flex_joint": -0.09896061415750401,
            "l_wrist_roll_joint": 1.5548711750357214e-05,
            "r_elbow_flex_joint": -2.0632453854762955,
            "r_forearm_roll_joint": -1.3016527553185637,
            "r_shoulder_lift_joint": -0.30680923151650064,
            "r_shoulder_pan_joint": -1.183809044088452,
            "r_upper_arm_roll_joint": -1.6439096470662296,
            "r_wrist_flex_joint": -1.28493603944117,
            "r_wrist_roll_joint": 4.308453693283392,
            "torso_lift_joint": 0.25,
        }
        kitchen_setup.send_and_check_joint_goal(js)

        kitchen_setup.set_kitchen_js(
            {
                "fridge_area_lower_drawer_main_joint": 0.0,
                "iai_fridge_door_joint": 0.0,
                "kitchen_island_left_lower_drawer_main_joint": 0.0,
                "kitchen_island_left_upper_drawer_main_joint": 0.0,
                "kitchen_island_middle_lower_drawer_main_joint": 0.0,
                "kitchen_island_middle_upper_drawer_main_joint": 0.0,
                "kitchen_island_right_lower_drawer_main_joint": 0.0,
                "kitchen_island_right_upper_drawer_main_joint": 0.0,
                "oven_area_area_left_drawer_main_joint": 0.0,
                "oven_area_area_middle_lower_drawer_main_joint": 0.0,
                "oven_area_area_middle_upper_drawer_main_joint": 0.0,
                "oven_area_area_right_drawer_main_joint": 0.0,
                "oven_area_oven_door_joint": 0.0,
                "oven_area_oven_knob_oven_joint": 0.0,
                "oven_area_oven_knob_stove_1_joint": 0.0,
                "oven_area_oven_knob_stove_2_joint": 0.0,
                "oven_area_oven_knob_stove_3_joint": 0.0,
                "oven_area_oven_knob_stove_4_joint": 0.0,
                "sink_area_dish_washer_door_joint": 0.0,
                "sink_area_dish_washer_main_joint": 0.0,
                "sink_area_dish_washer_tray_main": 0.0,
                "sink_area_left_bottom_drawer_main_joint": 0.0,
                "sink_area_left_middle_drawer_main_joint": 0.48,
                "sink_area_left_upper_drawer_main_joint": 0.0,
                "sink_area_trash_drawer_main_joint": 0.0
            }
        )

        box_pose = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                     {
                                                         "header": {
                                                             "frame_id": "r_wrist_roll_link",
                                                             "seq": 0,
                                                             "stamp": {
                                                                 "nsecs": 0,
                                                                 "secs": 0
                                                             }
                                                         },
                                                         "pose": {
                                                             "orientation": {
                                                                 "w": 0.7050609904138849,
                                                                 "x": 1.981449345260101e-12,
                                                                 "y": -0.7090613364582498,
                                                                 "z": 0.011000951623944002
                                                             },
                                                             "position": {
                                                                 "x": 0.209,
                                                                 "y": -0.059,
                                                                 "z": 0.003
                                                             }
                                                         }
                                                     }
                                                     )
        kitchen_setup.attach_box(u'bowl_1',
                                 size=[.171740325292, 0.171745332082, 0.10578233401],
                                 frame_id='r_wrist_roll_link', pose=box_pose)

        map_T_cart_goal = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                            {"header": {"stamp": {"secs": 1600166439,
                                                                                  "nsecs": 617058.9923858643},
                                                                        "frame_id": "map", "seq": 0}, "pose": {
                                                                "position": {"x": -2.4800000190734863,
                                                                             "y": 0.15999984741210938, "z": 0.0},
                                                                "orientation": {"x": 0.0, "y": 0.0,
                                                                                "z": -0.9870874685064221,
                                                                                "w": 0.16018217602961632}}}
                                                            )

        avoidance_hint = Vector3Stamped()
        avoidance_hint.header.frame_id = 'map'
        avoidance_hint.vector.y = -1
        kitchen_setup.set_json_goal(u'CollisionAvoidanceHint',
                                    link_name=u'base_link',
                                    max_threshold=0.25,
                                    spring_threshold=0.3,
                                    max_linear_velocity=1,
                                    body_b=u'kitchen',
                                    link_b=u'kitchen_island',
                                    weight=WEIGHT_COLLISION_AVOIDANCE,
                                    avoidance_hint=avoidance_hint)
        kitchen_setup.set_json_goal(u'CartesianVelocityLimit',
                                    root_link=u'odom_combined',
                                    tip_link=u'base_footprint',
                                    max_linear_velocity=0.5,
                                    max_angular_velocity=0.2,
                                    )
        kitchen_setup.set_joint_goal({
            "l_shoulder_pan_joint": 1.9728364401190246,
            "l_shoulder_lift_joint": -0.262967889172871,
            "l_upper_arm_roll_joint": 1.3837617139225473,
            "l_elbow_flex_joint": -2.121008899318323,
            "l_forearm_roll_joint": 4.430338491177909,
            "l_wrist_flex_joint": -0.08986724985668793,
            "l_wrist_roll_joint": 0.001103271714241405,
            "r_shoulder_pan_joint": -1.183809044088452,
            "r_shoulder_lift_joint": -0.307232206132023,
            "r_upper_arm_roll_joint": -1.6439096470662296,
            "r_elbow_flex_joint": -2.0632453854762955,
            "r_forearm_roll_joint": -7.5851190340623456,
            "r_wrist_flex_joint": -1.28497954836127,
            "r_wrist_roll_joint": 10.590924211064326
            # "torso_lift_joint": 1,
        })

        kitchen_setup.avoid_all_collisions(0.2)
        # tip_normal = Vector3Stamped()
        # tip_normal.header.frame_id = 'map'
        # tip_normal.vector.z = 1
        # kitchen_setup.align_planes('bowl_1', tip_normal, weight=WEIGHT_ABOVE_CA)
        kitchen_setup.set_cart_goal(map_T_cart_goal, 'base_footprint', linear_velocity=0.5,
                                    weight=WEIGHT_BELOW_CA)
        kitchen_setup.send_and_check_goal()

    def test_bug2020_09_11_11_59_57_dump(self, kitchen_setup):
        map_T_odom = tf.pose_to_kdl(convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                                      {
                                                                          "header": {
                                                                              "frame_id": "map",
                                                                              "seq": 0,
                                                                              "stamp": {
                                                                                  "nsecs": 0,
                                                                                  "secs": 0
                                                                              }
                                                                          },
                                                                          "pose": {
                                                                              "orientation": {
                                                                                  "w": 1.0,
                                                                                  "x": 0.0,
                                                                                  "y": 0.0,
                                                                                  "z": 0.0
                                                                              },
                                                                              "position": {
                                                                                  "x": 0.0,
                                                                                  "y": 0.0,
                                                                                  "z": 0.0
                                                                              }
                                                                          }
                                                                      }
                                                                      ).pose)

        odom_T_base_footprint = PoseStamped()
        odom_T_base_footprint.pose.position = Point(
            0.09598632901906967,
            1.7636791467666626,
            0
        )
        odom_T_base_footprint.pose.orientation = Quaternion(*quaternion_about_axis(
            -2.7521222050783707,
            [0, 0, 1]))
        odom_T_base_footprint = tf.pose_to_kdl(odom_T_base_footprint.pose)
        map_T_base_footprint = tf.kdl_to_pose_stamped(map_T_odom * odom_T_base_footprint, u'map')

        kitchen_setup.teleport_base(map_T_base_footprint)

        js = {
            "head_pan_joint": 0.10143902897834778,
            "head_tilt_joint": 0.6765880584716797,
            "l_elbow_flex_joint": -2.1212997436523438,
            "l_forearm_roll_joint": -1.8530672788619995,
            "l_shoulder_lift_joint": -0.26502490043640137,
            "l_shoulder_pan_joint": 1.9651780128479004,
            "l_upper_arm_roll_joint": 1.3837532997131348,
            "l_wrist_flex_joint": -0.10027802735567093,
            "l_wrist_roll_joint": 9.870521171251312e-05,
            "r_elbow_flex_joint": -2.1213154792785645,
            "r_forearm_roll_joint": -17.083274841308594,
            "r_shoulder_lift_joint": -0.2567230463027954,
            "r_shoulder_pan_joint": -1.7125777006149292,
            "r_upper_arm_roll_joint": -1.4633837938308716,
            "r_wrist_flex_joint": -0.10014520585536957,
            "r_wrist_roll_joint": 12.617490768432617,
            "torso_lift_joint": 0.26358458399772644,
        }
        kitchen_setup.send_and_check_joint_goal(js)

        kitchen_setup.set_kitchen_js(
            {
                "fridge_area_lower_drawer_main_joint": 0.0,
                "iai_fridge_door_joint": 0.0,
                "kitchen_island_left_lower_drawer_main_joint": 0.0,
                "kitchen_island_left_upper_drawer_main_joint": 0.0,
                "kitchen_island_middle_lower_drawer_main_joint": 0.0,
                "kitchen_island_middle_upper_drawer_main_joint": 0.0,
                "kitchen_island_right_lower_drawer_main_joint": 0.0,
                "kitchen_island_right_upper_drawer_main_joint": 0.0,
                "oven_area_area_left_drawer_main_joint": 0.0,
                "oven_area_area_middle_lower_drawer_main_joint": 0.0,
                "oven_area_area_middle_upper_drawer_main_joint": 0.0,
                "oven_area_area_right_drawer_main_joint": 0.0,
                "oven_area_oven_door_joint": 0.0,
                "oven_area_oven_knob_oven_joint": 0.0,
                "oven_area_oven_knob_stove_1_joint": 0.0,
                "oven_area_oven_knob_stove_2_joint": 0.0,
                "oven_area_oven_knob_stove_3_joint": 0.0,
                "oven_area_oven_knob_stove_4_joint": 0.0,
                "sink_area_dish_washer_door_joint": 0.0,
                "sink_area_dish_washer_main_joint": 0.0,
                "sink_area_dish_washer_tray_main": 0.0,
                "sink_area_left_bottom_drawer_main_joint": 0.0,
                "sink_area_left_middle_drawer_main_joint": 0.0,
                "sink_area_left_upper_drawer_main_joint": 0.48,
                "sink_area_trash_drawer_main_joint": 0.0
            }
        )

        SM_TableKnife4_pose = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                                {
                                                                    "header": {
                                                                        "frame_id": "r_wrist_roll_link",
                                                                        "seq": 0,
                                                                        "stamp": {
                                                                            "nsecs": 0,
                                                                            "secs": 0
                                                                        }
                                                                    },
                                                                    "pose": {
                                                                        "orientation": {
                                                                            "w": 0.7071039519219716,
                                                                            "x": -0.002000294064347671,
                                                                            "y": -0.7071039519219521,
                                                                            "z": 0.002000294064347671
                                                                        },
                                                                        "position": {
                                                                            "x": 0.166,
                                                                            "y": 0.003,
                                                                            "z": 0.002
                                                                        }
                                                                    }
                                                                }
                                                                )
        kitchen_setup.attach_box(u'SM_TableKnife4',
                                 size=[0.222074826558, 0.0516345620155, 0.0242430388927],
                                 frame_id='r_wrist_roll_link', pose=SM_TableKnife4_pose)

        SM_BigBowl2_pose = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                             {
                                                                 "header": {
                                                                     "frame_id": "map",
                                                                     "seq": 0,
                                                                     "stamp": {
                                                                         "nsecs": 0,
                                                                         "secs": 0
                                                                     }
                                                                 },
                                                                 "pose": {
                                                                     "orientation": {
                                                                         "w": 0.9442825488612109,
                                                                         "x": 0.008524839235563506,
                                                                         "y": 0.004445144931827071,
                                                                         "z": 0.32899549498240516
                                                                     },
                                                                     "position": {
                                                                         "x": -0.8397645969330706,
                                                                         "y": 1.197540621263799,
                                                                         "z": 0.9046289697156581
                                                                     }
                                                                 }
                                                             }
                                                             )
        kitchen_setup.add_box(u'SM_BigBowl2',
                              size=[0.171740325292, 0.171745332082, 0.10578233401], pose=SM_BigBowl2_pose)

        # goal----------------------------------

        map_T_cart_goal = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                            {"header": {"stamp": {"secs": 1599817878,
                                                                                  "nsecs": 274374.96185302734},
                                                                        "frame_id": "map", "seq": 0}, "pose": {
                                                                "position": {"x": -0.20000028610229492,
                                                                             "y": 2.0399999618530273, "z": 0.0},
                                                                "orientation": {"x": 0.0, "y": 0.0,
                                                                                "z": -0.9210651816262956,
                                                                                "w": 0.3894084374993936}}}
                                                            )

        avoidance_hint = Vector3Stamped()
        avoidance_hint.header.frame_id = 'map'
        avoidance_hint.vector.y = -1
        kitchen_setup.set_json_goal(u'CollisionAvoidanceHint',
                                    link_name=u'base_link',
                                    max_threshold=0.3,
                                    # spring_threshold=0.5,
                                    max_linear_velocity=1,
                                    body_b=u'kitchen',
                                    link_b=u'kitchen_island',
                                    weight=WEIGHT_COLLISION_AVOIDANCE,
                                    avoidance_hint=avoidance_hint)
        kitchen_setup.set_json_goal(u'CartesianVelocityLimit',
                                    root_link=u'odom_combined',
                                    tip_link=u'base_footprint',
                                    max_linear_velocity=0.5,
                                    max_angular_velocity=0.2,
                                    )
        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.avoid_all_collisions(0.2)
        kitchen_setup.set_cart_goal(map_T_cart_goal, 'base_footprint', linear_velocity=0.5,
                                    weight=WEIGHT_BELOW_CA)
        kitchen_setup.send_and_check_goal()

    def test_bug2020_09_09_14_33_36_dump_endless_wiggling(self, kitchen_setup):
        map_T_odom = tf.pose_to_kdl(convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                                      {
                                                                          "header": {
                                                                              "frame_id": "map",
                                                                              "seq": 0,
                                                                              "stamp": {
                                                                                  "nsecs": 0,
                                                                                  "secs": 0
                                                                              }
                                                                          },
                                                                          "pose": {
                                                                              "orientation": {
                                                                                  "w": 0.8038133596161393,
                                                                                  "x": -0.0017914705470469517,
                                                                                  "y": 0.004855839193902281,
                                                                                  "z": -0.5948590541982338
                                                                              },
                                                                              "position": {
                                                                                  "x": -1.2290553421224326,
                                                                                  "y": -0.07699918098148706,
                                                                                  "z": -0.014262627020910567
                                                                              }
                                                                          }
                                                                      }
                                                                      ).pose)

        odom_T_base_footprint = PoseStamped()
        odom_T_base_footprint.pose.position = Point(
            -0.84211314506964,
            -1.0942196618120819,
            0
        )
        odom_T_base_footprint.pose.orientation = Quaternion(*quaternion_about_axis(
            -1.8534894800614323,
            [0, 0, 1]))
        odom_T_base_footprint = tf.pose_to_kdl(odom_T_base_footprint.pose)
        map_T_base_footprint = tf.kdl_to_pose_stamped(map_T_odom * odom_T_base_footprint, u'map')

        kitchen_setup.teleport_base(map_T_base_footprint)

        js = {
            "l_elbow_flex_joint": -2.121008899318323,
            "l_forearm_roll_joint": 60.97852695135053,
            "l_shoulder_lift_joint": -0.2647443825580645,
            "l_shoulder_pan_joint": 1.965291937939539,
            "l_upper_arm_roll_joint": 1.3837617139225473,
            "l_wrist_flex_joint": -0.09987430147959442,
            "l_wrist_roll_joint": -31.415881039502846,
            "r_elbow_flex_joint": -2.120140275050022,
            "r_forearm_roll_joint": -92.48219800774837,
            "r_shoulder_lift_joint": -0.2576595811928125,
            "r_shoulder_pan_joint": -1.7474082288812411,
            "r_upper_arm_roll_joint": -1.4636708222264772,
            "r_wrist_flex_joint": -0.10002450226779036,
            "r_wrist_roll_joint": 69.16652020014351,
            "torso_lift_joint": 0.2575583183617209,
        }
        kitchen_setup.allow_all_collisions()
        kitchen_setup.send_and_check_joint_goal(js)

        kitchen_setup.set_kitchen_js(
            {
                "fridge_area_lower_drawer_main_joint": 0.0,
                "iai_fridge_door_joint": 0.0,
                "kitchen_island_left_lower_drawer_main_joint": 0.0,
                "kitchen_island_left_upper_drawer_main_joint": 0.0,
                "kitchen_island_middle_lower_drawer_main_joint": 0.0,
                "kitchen_island_middle_upper_drawer_main_joint": 0.0,
                "kitchen_island_right_lower_drawer_main_joint": 0.0,
                "kitchen_island_right_upper_drawer_main_joint": 0.0,
                "oven_area_area_left_drawer_main_joint": 0.0,
                "oven_area_area_middle_lower_drawer_main_joint": 0.0,
                "oven_area_area_middle_upper_drawer_main_joint": 0.0,
                "oven_area_area_right_drawer_main_joint": 0.0,
                "oven_area_oven_door_joint": 0.0,
                "oven_area_oven_knob_oven_joint": 0.0,
                "oven_area_oven_knob_stove_1_joint": 0.0,
                "oven_area_oven_knob_stove_2_joint": 0.0,
                "oven_area_oven_knob_stove_3_joint": 0.0,
                "oven_area_oven_knob_stove_4_joint": 0.0,
                "sink_area_dish_washer_door_joint": 0.0,
                "sink_area_dish_washer_main_joint": 0.0,
                "sink_area_dish_washer_tray_main": 0.0,
                "sink_area_left_bottom_drawer_main_joint": 0.0,
                "sink_area_left_middle_drawer_main_joint": 0.0,
                "sink_area_left_upper_drawer_main_joint": 0.0,
                "sink_area_trash_drawer_main_joint": 0.0
            }
        )

        box_pose = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                     {
                                                         "header": {
                                                             "frame_id": "map",
                                                             "seq": 0,
                                                             "stamp": {
                                                                 "nsecs": 0,
                                                                 "secs": 0
                                                             }
                                                         },
                                                         "pose": {
                                                             "orientation": {
                                                                 "w": 0.9894008709219515,
                                                                 "x": 9.813812393299583e-05,
                                                                 "y": 0.00012360265872692514,
                                                                 "z": 0.14520981960657917
                                                             },
                                                             "position": {
                                                                 "x": -3.328302510579427,
                                                                 "y": 0.4969228744506836,
                                                                 "z": 0.82414124806722
                                                             }
                                                         }
                                                     }
                                                     )
        kitchen_setup.add_box(u'bowl_1',
                              size=[0.0858103354772, 0.0858093341192, 0.155754343669], pose=box_pose)

        wrong_odom_T_goal = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                              {
                                                                  "header": {
                                                                      "frame_id": "odom_combined",
                                                                      "seq": 0,
                                                                      "stamp": {
                                                                          "nsecs": 784679174,
                                                                          "secs": 1599662013
                                                                      }
                                                                  },
                                                                  "pose": {
                                                                      "orientation": {
                                                                          "w": 0.7052763626162107,
                                                                          "x": 0.004909433897532277,
                                                                          "y": 0.001082490750537487,
                                                                          "z": -0.7089146376393086
                                                                      },
                                                                      "position": {
                                                                          "x": -1.1672449270105567,
                                                                          "y": -1.8671080893997798,
                                                                          "z": 0.8256542845203053
                                                                      }
                                                                  }
                                                              }
                                                              )
        map_T_cart_goal = tf.kdl_to_pose_stamped(map_T_odom * tf.pose_to_kdl(wrong_odom_T_goal.pose), u'map')
        kitchen_setup.set_json_goal(u'CartesianVelocityLimit',
                                    root_link=u'odom_combined',
                                    tip_link=u'base_footprint',
                                    max_linear_velocity=0.1,
                                    max_angular_velocity=0.2,
                                    )
        # kitchen_setup.avoid_all_collisions(0.2)
        kitchen_setup.allow_collision([
            "r_gripper_l_finger_tip_link",
            "r_gripper_r_finger_tip_link",
            "r_gripper_l_finger_link",
            "r_gripper_r_finger_link",
            "r_gripper_l_finger_tip_frame",
            "r_gripper_palm_link"
        ], 'bowl_1', [])
        kitchen_setup.allow_collision([
            "r_gripper_l_finger_tip_link",
            "r_gripper_r_finger_tip_link",
            "r_gripper_l_finger_link",
            "r_gripper_r_finger_link",
            "r_gripper_l_finger_tip_frame",
            "r_gripper_palm_link"
        ], 'kitchen', [])
        # kitchen_setup.add_json_goal(u'AvoidJointLimits', percentage=40)
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.set_and_check_cart_goal(map_T_cart_goal, 'r_gripper_tool_frame')

    def test_bug2020_09_09_13_36_25_dump(self, kitchen_setup):
        map_T_odom = tf.pose_to_kdl(convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                                      {
                                                                          "header": {
                                                                              "frame_id": "map",
                                                                              "seq": 0,
                                                                              "stamp": {
                                                                                  "nsecs": 0,
                                                                                  "secs": 0
                                                                              }
                                                                          },
                                                                          "pose": {
                                                                              "orientation": {
                                                                                  "w": 0.8779201932099772,
                                                                                  "x": 0.005801928233696417,
                                                                                  "y": 0.0021891835979195693,
                                                                                  "z": -0.47876683203632847
                                                                              },
                                                                              "position": {
                                                                                  "x": -1.2813072440232847,
                                                                                  "y": -0.000900015892780554,
                                                                                  "z": -0.0053088233454366375
                                                                              }
                                                                          }
                                                                      }
                                                                      ).pose)

        odom_T_base_footprint = PoseStamped()
        odom_T_base_footprint.pose.position = Point(
            0.5065161536263582,
            1.2428430519008682,
            0
        )
        odom_T_base_footprint.pose.orientation = Quaternion(*quaternion_about_axis(
            -2.9179043687346833,
            [0, 0, 1]))
        odom_T_base_footprint = tf.pose_to_kdl(odom_T_base_footprint.pose)
        map_T_base_footprint = tf.kdl_to_pose_stamped(map_T_odom * odom_T_base_footprint, u'map')

        kitchen_setup.teleport_base(map_T_base_footprint)

        js = {
            "l_elbow_flex_joint": -2.121008899318323,
            "l_forearm_roll_joint": 60.97875833969833,
            "l_shoulder_lift_joint": -0.26499816732737785,
            "l_shoulder_pan_joint": 1.965291937939539,
            "l_upper_arm_roll_joint": 1.3837617139225473,
            "l_wrist_flex_joint": -0.09896061415750301,
            "l_wrist_roll_joint": -31.415924548422947,
            "r_elbow_flex_joint": -2.121153670029707,
            "r_forearm_roll_joint": -67.34873782238188,
            "r_shoulder_lift_joint": -0.25672903703866357,
            "r_shoulder_pan_joint": -1.712587449591307,
            "r_upper_arm_roll_joint": -1.4633501125737376,
            "r_wrist_flex_joint": -0.09998099334768273,
            "r_wrist_roll_joint": 50.31658512943495,
            "torso_lift_joint": 0.2623074836994323,
        }
        kitchen_setup.send_and_check_joint_goal(js)

        kitchen_setup.set_kitchen_js(
            {
                "fridge_area_lower_drawer_main_joint": 0.0,
                "iai_fridge_door_joint": 0.0,
                "kitchen_island_left_lower_drawer_main_joint": 0.0,
                "kitchen_island_left_upper_drawer_main_joint": 0.48,
                "kitchen_island_middle_lower_drawer_main_joint": 0.0,
                "kitchen_island_middle_upper_drawer_main_joint": 0.0,
                "kitchen_island_right_lower_drawer_main_joint": 0.0,
                "kitchen_island_right_upper_drawer_main_joint": 0.0,
                "oven_area_area_left_drawer_main_joint": 0.0,
                "oven_area_area_middle_lower_drawer_main_joint": 0.0,
                "oven_area_area_middle_upper_drawer_main_joint": 0.0,
                "oven_area_area_right_drawer_main_joint": 0.0,
                "oven_area_oven_door_joint": 0.0,
                "oven_area_oven_knob_oven_joint": 0.0,
                "oven_area_oven_knob_stove_1_joint": 0.0,
                "oven_area_oven_knob_stove_2_joint": 0.0,
                "oven_area_oven_knob_stove_3_joint": 0.0,
                "oven_area_oven_knob_stove_4_joint": 0.0,
                "sink_area_dish_washer_door_joint": 0.0,
                "sink_area_dish_washer_main_joint": 0.0,
                "sink_area_dish_washer_tray_main": 0.0,
                "sink_area_left_bottom_drawer_main_joint": 0.0,
                "sink_area_left_middle_drawer_main_joint": 0.0,
                "sink_area_left_upper_drawer_main_joint": 0.0,
                "sink_area_trash_drawer_main_joint": 0.0
            }
        )

        map_T_cart_goal = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped',
                                                            {"header": {"stamp":
                                                                            {"secs": 0,
                                                                             "nsecs": 43884.0},
                                                                        "frame_id": "map", "seq": 0},
                                                             "pose": {
                                                                 "position":
                                                                     {"x": -0.32000017166137695,
                                                                      "y": 0.31999969482421875, "z": 0.0},
                                                                 "orientation":
                                                                     {"x": 0.0, "y": 0.0,
                                                                      "z": 0.8443331886423444,
                                                                      "w": 0.5358185015068547}}}
                                                            )

        avoidance_hint = Vector3Stamped()
        avoidance_hint.header.frame_id = 'map'
        avoidance_hint.vector.y = -1
        kitchen_setup.set_json_goal(u'CollisionAvoidanceHint',
                                    link_name=u'base_footprint',
                                    max_threshold=0.3,
                                    max_linear_velocity=1,
                                    body_b=u'kitchen',
                                    link_b=u'kitchen_island',
                                    weight=WEIGHT_COLLISION_AVOIDANCE,
                                    avoidance_hint=avoidance_hint)
        kitchen_setup.set_json_goal(u'CartesianVelocityLimit',
                                    root_link=u'odom_combined',
                                    tip_link=u'base_footprint',
                                    max_linear_velocity=0.5,
                                    max_angular_velocity=0.2,
                                    )
        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.avoid_all_collisions(0.2)
        kitchen_setup.set_and_check_cart_goal(map_T_cart_goal, 'base_footprint')

    def test_open_drawer(self, kitchen_setup):
        self.open_drawer(kitchen_setup, kitchen_setup.l_tip, u'iai_kitchen/sink_area_left_middle_drawer_handle',
                         u'sink_area_left_middle_drawer_main_joint')

    def test_handover(self, kitchen_setup):
        js = {
            "l_shoulder_pan_joint": 1.0252138037286773,
            "l_shoulder_lift_joint": - 0.06966848987919201,
            "l_upper_arm_roll_joint": 1.1765832782526544,
            "l_elbow_flex_joint": - 1.9323726623855864,
            "l_forearm_roll_joint": 1.3824994377973336,
            "l_wrist_flex_joint": - 1.8416233909065576,
            "l_wrist_roll_joint": 2.907373693068033,
        }
        kitchen_setup.send_and_check_joint_goal(js)

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
        kitchen_setup.send_and_check_goal()
        kitchen_setup.check_cart_goal(kitchen_setup.r_tip, r_goal)

        kitchen_setup.detach_object('box')
        kitchen_setup.attach_object('box', kitchen_setup.r_tip)

        r_goal2 = PoseStamped()
        r_goal2.header.frame_id = 'box'
        r_goal2.pose.position.x -= -.1
        r_goal2.pose.orientation.w = 1

        kitchen_setup.set_cart_goal(r_goal2, u'box', root_link=kitchen_setup.l_tip,
                                    )
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.send_and_check_goal()
        # kitchen_setup.check_cart_goal(u'box', r_goal2)

    def test_add_box(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        object_name = u'muh'
        p = PoseStamped()
        p.header.frame_id = u'map'
        p.pose.position = Point(1.2, 0, 1.6)
        p.pose.orientation = Quaternion(0.0, 0.0, 0.47942554, 0.87758256)
        zero_pose.add_box(object_name, pose=p)
        m = zero_pose.get_world().get_object(object_name).as_marker_msg()
        compare_poses(m.pose, p.pose)

    def test_add_mesh(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        object_name = u'muh'
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.add_mesh(object_name, path=u'package://giskardpy/test/urdfs/meshes/bowl_21.obj', pose=p)
        # m = zero_pose.get_world().get_object(object_name).as_marker_msg()
        # compare_poses(m.pose, p.pose)
        pass

    def test_add_non_existing_mesh(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        object_name = u'muh'
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.add_mesh(object_name, path=u'package://giskardpy/test/urdfs/meshes/muh.obj', pose=p,
                           expected_error=UpdateWorldResponse.CORRUPT_MESH_ERROR)
        pass

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
        zero_pose.add_mesh(object_name, path=u'package://giskardpy/test/urdfs/meshes/bowl_21.obj', pose=p)
        # m = zero_pose.get_world().get_object(object_name).as_marker_msg()
        # compare_poses(m.pose, p.pose)
        zero_pose.send_and_check_goal()
        pass

    def test_add_box_twice(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        object_name = u'muh'
        p = PoseStamped()
        p.header.frame_id = u'map'
        p.pose.position = Point(1.2, 0, 1.6)
        p.pose.orientation = Quaternion(0.0, 0.0, 0.47942554, 0.87758256)
        zero_pose.add_box(object_name, pose=p)
        m = zero_pose.get_world().get_object(object_name).as_marker_msg()
        compare_poses(m.pose, p.pose)
        zero_pose.add_box(object_name, pose=p, expected_response=UpdateWorldResponse.DUPLICATE_BODY_ERROR)

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
        zero_pose.add_sphere(object_name, pose=p)
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
        zero_pose.add_cylinder(object_name, pose=p)
        zero_pose.remove_object(object_name)

    def test_add_urdf_body(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        kitchen_setup.remove_object(u'kitchen')

    def test_attach_box(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        pocky = u'http://muh#pocky'
        zero_pose.attach_box(pocky, [0.1, 0.02, 0.02], zero_pose.r_tip, [0.05, 0, 0])

    def test_attach_box_as_eef(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        pocky = u'http://muh#pocky'
        zero_pose.attach_box(pocky, [0.1, 0.02, 0.02], zero_pose.r_tip, [0.05, 0, 0], [1, 0, 0, 0])
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.orientation.w = 1
        zero_pose.set_cart_goal(p, pocky, zero_pose.default_root)
        p = tf.transform_pose(zero_pose.default_root, p)
        zero_pose.send_and_check_goal()
        p2 = zero_pose.get_robot().get_fk_pose(zero_pose.default_root, pocky)
        compare_poses(p2.pose, p.pose)
        zero_pose.detach_object(pocky)
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.orientation.w = 1
        p.pose.position.x = -.1
        zero_pose.set_and_check_cart_goal(p, zero_pose.r_tip, zero_pose.default_root)

    def test_attach_remove_box(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        pocky = u'http://muh#pocky'
        zero_pose.attach_box(pocky, [0.1, 0.02, 0.02], zero_pose.r_tip, [0.05, 0, 0])
        zero_pose.detach_object(pocky)
        zero_pose.remove_object(pocky)

    def test_attach_remove_box2(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.send_and_check_joint_goal(gaya_pose)
        pocky = u'http://muh#pocky'
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.orientation.w = 1
        zero_pose.add_box(pocky, pose=p)
        for i in range(10):
            zero_pose.attach_object(pocky, zero_pose.r_tip)
            zero_pose.detach_object(pocky)
        zero_pose.remove_object(pocky)

    def test_remove_attached_box(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        pocky = u'http://muh#pocky'
        zero_pose.attach_box(pocky, [0.1, 0.02, 0.02], zero_pose.r_tip, [0.05, 0, 0])
        zero_pose.remove_object(pocky, expected_response=UpdateWorldResponse.MISSING_BODY_ERROR)

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
        relative_pose = zero_pose.get_robot().get_fk_pose(zero_pose.r_tip, pocky).pose
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
        zero_pose.add_box(object_name, pose=p)
        zero_pose.attach_object(object_name, frame_id=zero_pose.r_tip)
        zero_pose.detach_object(object_name)
        zero_pose.remove_object(object_name)
        zero_pose.add_box(object_name, pose=p)
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
        relative_pose = zero_pose.get_robot().get_fk_pose(zero_pose.r_tip, pocky).pose
        compare_poses(old_p.pose, relative_pose)

        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.position.x = -0.1
        p.pose.orientation.w = 1.0
        zero_pose.set_and_check_cart_goal(p, zero_pose.r_tip, zero_pose.default_root)
        p.header.frame_id = u'map'
        p.pose.position.y = -1
        p.pose.orientation = Quaternion(0, 0, 0.47942554, 0.87758256)
        zero_pose.move_base(p)
        rospy.sleep(.5)

        zero_pose.detach_object(pocky)

    def test_attach_detach_twice(self, zero_pose):
        pocky = u'http://muh#pocky'
        zero_pose.attach_box(pocky, [0.1, 0.02, 0.02], zero_pose.r_tip, [0.05, 0, 0], [1, 0, 0, 0])
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.orientation.w = 1
        zero_pose.set_cart_goal(p, pocky)
        p = tf.transform_pose(zero_pose.default_root, p)
        zero_pose.send_and_check_goal()
        p2 = zero_pose.get_robot().get_fk_pose(zero_pose.default_root, pocky)
        compare_poses(p2.pose, p.pose)

        zero_pose.clear_world()

        old_p = PoseStamped()
        old_p.header.frame_id = zero_pose.r_tip
        old_p.pose.position = Point(0.05, 0, 0)
        old_p.pose.orientation = Quaternion(0., 0., 0.47942554, 0.87758256)
        zero_pose.add_box(pocky, [0.1, 0.02, 0.02], pose=old_p)
        zero_pose.attach_object(pocky, frame_id=zero_pose.r_tip)
        relative_pose = zero_pose.get_robot().get_fk_pose(zero_pose.r_tip, pocky).pose
        compare_poses(old_p.pose, relative_pose)

        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.position.x = -0.1
        p.pose.orientation.w = 1.0
        zero_pose.set_and_check_cart_goal(p, zero_pose.r_tip, zero_pose.default_root)

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

    def test_add_remove_object(self, zero_pose):
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
        zero_pose.add_box(object_name, pose=p)
        zero_pose.remove_object(object_name)
        # FIXME marker does not get removed

    def test_invalid_update_world(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        req = UpdateWorldRequest(42, WorldBody(), True, PoseStamped())
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
        req = UpdateWorldRequest(UpdateWorldRequest.ADD, WorldBody(type=WorldBody.PRIMITIVE_BODY,
                                                                   shape=SolidPrimitive(type=42)), True, PoseStamped())
        assert zero_pose._update_world_srv.call(req).error_codes == UpdateWorldResponse.CORRUPT_SHAPE_ERROR

    def test_unsupported_options(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        wb = WorldBody()
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = str(u'map')
        pose.pose.position = Point()
        pose.pose.orientation = Quaternion(w=1)
        wb.type = WorldBody.URDF_BODY

        req = UpdateWorldRequest(UpdateWorldRequest.ADD, wb, True, pose)
        assert kitchen_setup._update_world_srv.call(req).error_codes == UpdateWorldResponse.UNSUPPORTED_OPTIONS

    def test_infeasible(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        pose = PoseStamped()
        pose.header.frame_id = u'map'
        pose.pose.position = Point(2, 0, 0)
        pose.pose.orientation = Quaternion(w=1)
        kitchen_setup.teleport_base(pose)
        kitchen_setup.send_and_check_goal(expected_error_codes=[MoveResult.HARD_CONSTRAINTS_VIOLATED])

    def test_link_b_set_but_body_b_not(self, box_setup):
        """
        :type box_setup: PR2
        """
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.link_bs = [u'asdf']
        box_setup.set_collision_entries([ce])
        box_setup.send_and_check_goal(expected_error_codes=[MoveResult.WORLD_ERROR])

    def test_unknown_robot_link(self, box_setup):
        """
        :type box_setup: PR2
        """
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.robot_links = [u'asdf']
        box_setup.set_collision_entries([ce])
        box_setup.send_and_check_goal([MoveResult.UNKNOWN_OBJECT])

    def test_unknown_body_b(self, box_setup):
        """
        :type box_setup: PR2
        """
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.body_b = u'asdf'
        box_setup.set_collision_entries([ce])
        box_setup.send_and_check_goal([MoveResult.UNKNOWN_OBJECT])

    def test_unknown_link_b(self, box_setup):
        """
        :type box_setup: PR2
        """
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.body_b = u'box'
        ce.link_bs = [u'asdf']
        box_setup.set_collision_entries([ce])
        box_setup.send_and_check_goal([MoveResult.UNKNOWN_OBJECT])

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
        zero_pose.send_and_check_joint_goal(pocky_pose, expected_error_codes=[MoveResult.HARD_CONSTRAINTS_VIOLATED])

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

        box_setup.send_and_check_goal([MoveResult.UNKNOWN_OBJECT])

    def test_allow_self_collision(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.check_cpi_geq(zero_pose.get_r_gripper_links(), 0.1)
        zero_pose.check_cpi_geq(zero_pose.get_l_gripper_links(), 0.1)

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
        zero_pose.send_and_check_joint_goal(goal_js)

        p = PoseStamped()
        p.header.frame_id = zero_pose.l_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.2
        p.pose.orientation.w = 1
        zero_pose.allow_self_collision()
        zero_pose.set_and_check_cart_goal(p, zero_pose.l_tip, zero_pose.default_root)
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
        zero_pose.send_and_check_joint_goal(goal_js)

        p = PoseStamped()
        p.header.frame_id = zero_pose.l_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.18
        p.pose.position.z = 0.02
        p.pose.orientation.w = 1

        ces = []
        ces.append(CollisionEntry(type=CollisionEntry.ALLOW_COLLISION,
                                  robot_links=zero_pose.get_l_gripper_links(),
                                  body_b=u'pr2',
                                  link_bs=zero_pose.get_r_forearm_links()))
        zero_pose.set_collision_entries(ces)

        zero_pose.set_and_check_cart_goal(p, zero_pose.l_tip, zero_pose.default_root)
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
        zero_pose.send_and_check_joint_goal(goal_js)

        p = PoseStamped()
        p.header.frame_id = zero_pose.l_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.2
        p.pose.orientation.w = 1
        zero_pose.set_cart_goal(p, zero_pose.l_tip, zero_pose.default_root)
        zero_pose.send_goal()
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
        zero_pose.send_and_check_joint_goal(goal_js)

        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.2
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
        zero_pose.send_and_check_joint_goal(goal_js)

        p = PoseStamped()
        p.header.frame_id = zero_pose.l_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.15
        p.pose.orientation.w = 1
        zero_pose.set_cart_goal(p, zero_pose.l_tip, zero_pose.default_root)
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
        box_setup.send_and_check_goal([MoveResult.SUCCESS])
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
        box_setup.send_and_check_goal()
        box_setup.check_cpi_geq([u'base_link'], 0.099)

    def test_avoid_collision2(self, fake_table_setup):
        """
        :type fake_table_setup: PR2
        """
        r_goal = PoseStamped()
        r_goal.header.frame_id = u'map'
        r_goal.pose.position.x = 0.8
        r_goal.pose.position.y = -0.38
        r_goal.pose.position.z = 0.82
        r_goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 2, [0, 1, 0]))
        fake_table_setup.avoid_all_collisions(0.1)
        fake_table_setup.set_and_check_cart_goal(r_goal, fake_table_setup.r_tip)
        fake_table_setup.check_cpi_geq(fake_table_setup.get_l_gripper_links(), 0.1)
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
        box_setup.set_and_check_cart_goal(p, box_setup.r_tip, box_setup.default_root)

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
        pocky_pose_setup.add_box('bl', [0.1, 0.01, 0.2], p)
        p = PoseStamped()
        p.header.frame_id = pocky_pose_setup.r_tip
        p.pose.position.x = 0.15
        p.pose.position.y = -0.04
        p.pose.position.z = 0
        p.pose.orientation.w = 1
        pocky_pose_setup.add_box('br', [0.1, 0.01, 0.2], p)

        p = PoseStamped()
        p.header.frame_id = pocky_pose_setup.r_tip
        p.pose.position = Point(-0.15, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        pocky_pose_setup.set_cart_goal(p, pocky_pose_setup.r_tip, pocky_pose_setup.default_root)

        pocky_pose_setup.send_and_check_goal()
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
        pocky_pose_setup.add_box('b1', [0.01, 0.2, 0.2], p)
        p = PoseStamped()
        p.header.frame_id = pocky_pose_setup.r_tip
        p.pose.position.x = 0.15
        p.pose.position.y = 0.04
        p.pose.position.z = 0
        p.pose.orientation.w = 1
        pocky_pose_setup.add_box('bl', [0.1, 0.01, 0.2], p)
        p = PoseStamped()
        p.header.frame_id = pocky_pose_setup.r_tip
        p.pose.position.x = 0.15
        p.pose.position.y = -0.04
        p.pose.position.z = 0
        p.pose.orientation.w = 1
        pocky_pose_setup.add_box('br', [0.1, 0.01, 0.2], p)

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
        pocky_pose_setup.align_planes('box', x, root_normal=x_map)
        pocky_pose_setup.align_planes('box', y, root_normal=y_map)
        pocky_pose_setup.allow_self_collision()
        # pocky_pose_setup.allow_all_collisions()
        pocky_pose_setup.send_and_check_goal()

    def test_avoid_collision5(self, pocky_pose_setup):
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
        pocky_pose_setup.add_cylinder('bl', [0.2, 0.01], p)
        p = PoseStamped()
        p.header.frame_id = pocky_pose_setup.r_tip
        p.pose.position.x = 0.12
        p.pose.position.y = -0.04
        p.pose.position.z = 0
        p.pose.orientation.w = 1
        pocky_pose_setup.add_cylinder('br', [0.2, 0.01], p)

        pocky_pose_setup.send_and_check_goal(expected_error_codes=[MoveResult.SHAKING])

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
        pocky_pose_setup.add_cylinder('bl', [0.2, 0.01], p)
        p = PoseStamped()
        p.header.frame_id = pocky_pose_setup.r_tip
        p.pose.position.x = 0.12
        p.pose.position.y = -0.04
        p.pose.position.z = 0
        p.pose.orientation.w = 1
        pocky_pose_setup.add_cylinder('br', [0.2, 0.01], p)

        pocky_pose_setup.send_and_check_goal(goal_type=MoveGoal.PLAN_AND_EXECUTE_AND_CUT_OFF_SHAKING,
                                             expected_error_codes=[MoveResult.SHAKING])

    def test_avoid_collision6(self, fake_table_setup):
        """
        :type fake_table_setup: PR2
        """
        js = {
            u'r_shoulder_pan_joint': -0.341482794236,
            u'r_shoulder_lift_joint': 0.0301123643508,
            u'r_upper_arm_roll_joint': -2.67555547662,
            u'r_forearm_roll_joint': -0.472653283346,
            u'r_elbow_flex_joint': -0.149999999999,
            u'r_wrist_flex_joint': -1.40685144215,
            u'r_wrist_roll_joint': 2.87855178783,
            u'odom_x_joint': 0.0708087929675,
            u'odom_y_joint': 0.052896931145,
            u'odom_z_joint': 0.0105784287694,
            u'torso_lift_joint': 0.277729421077,
        }
        # fake_table_setup.allow_all_collisions()
        fake_table_setup.send_and_check_joint_goal(js, weight=WEIGHT_ABOVE_CA)
        fake_table_setup.check_cpi_geq(fake_table_setup.get_l_gripper_links(), 0.048)
        fake_table_setup.check_cpi_leq([u'r_gripper_l_finger_tip_link'], 0.04)
        fake_table_setup.check_cpi_leq([u'r_gripper_r_finger_tip_link'], 0.04)

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
        kitchen_setup.set_and_check_cart_goal(base_pose, u'base_footprint')
        kitchen_setup.check_current_joint_state(gaya_pose)

    def test_avoid_collision9(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        base_pose = PoseStamped()
        base_pose.header.frame_id = u'map'
        base_pose.pose.position.x = 0.8
        base_pose.pose.position.y = 0.9
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 2, [0, 0, 1]))
        kitchen_setup.teleport_base(base_pose)
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0, 0, 1]))
        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.set_and_check_cart_goal(base_pose, u'base_footprint', expected_error_codes=[MoveResult.SHAKING])
        kitchen_setup.check_current_joint_state(gaya_pose)

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
        kitchen_setup.set_and_check_cart_goal(base_pose, 'base_footprint')
        kitchen_setup.check_current_joint_state(gaya_pose)

    def test_go_around_kitchen_island(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
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
                                    link_name=u'base_link',
                                    max_threshold=0.3,
                                    spring_threshold=0.35,
                                    max_linear_velocity=1,
                                    body_b=u'kitchen',
                                    link_b=u'kitchen_island',
                                    weight=WEIGHT_COLLISION_AVOIDANCE,
                                    avoidance_hint=avoidance_hint)
        kitchen_setup.set_joint_goal(gaya_pose)

        kitchen_setup.set_and_check_cart_goal(base_pose, tip, weight=WEIGHT_BELOW_CA, linear_velocity=0.5)

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
                                    link_name=u'base_footprint',
                                    max_threshold=0.25,
                                    spring_threshold=0.3,
                                    max_linear_velocity=1,
                                    body_b=u'kitchen',
                                    link_b=u'kitchen_island',
                                    avoidance_hint=avoidance_hint)
        kitchen_setup.set_joint_goal(gaya_pose)

        kitchen_setup.set_cart_goal(base_pose, tip, weight=WEIGHT_BELOW_CA)
        kitchen_setup.send_and_check_goal()
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

        pocky_pose_setup.send_and_check_goal()
        pocky_pose_setup.check_cpi_geq(pocky_pose_setup.get_l_gripper_links(), 0.048)
        pocky_pose_setup.check_cpi_geq(pocky_pose_setup.get_r_gripper_links(), 0.048)

    def test_avoid_all_collision(self, box_setup):
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

        box_setup.send_and_check_goal()

        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.0)
        box_setup.check_cpi_geq(box_setup.get_r_gripper_links(), 0.0)

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

        box_setup.send_and_check_goal()

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_ALL_COLLISIONS
        collision_entry.min_dist = 0.05
        box_setup.set_collision_entries([collision_entry])

        box_setup.send_and_check_goal()

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
        box_setup.set_and_check_cart_goal(p, box_setup.l_tip, box_setup.default_root)
        box_setup.check_cpi_leq(box_setup.get_l_gripper_links(), 0.0)
        box_setup.check_cpi_geq(box_setup.get_r_gripper_links(), 0.048)

    def test_attached_get_out_of_collision(self, box_setup):
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
        box_setup.set_and_check_cart_goal(p, box_setup.r_tip, box_setup.default_root)
        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.048)
        box_setup.check_cpi_geq([attached_link_name], 0.048)
        box_setup.detach_object(attached_link_name)

    def test_attached_collision2(self, box_setup):
        """
        :type box_setup: PR2
        """
        attached_link_name = u'pocky'
        box_setup.attach_box(attached_link_name, [0.2, 0.04, 0.04], box_setup.r_tip, [0.05, 0, 0])
        box_setup.attach_box(attached_link_name, [0.2, 0.04, 0.04], box_setup.r_tip, [0.05, 0, 0],
                             expected_response=UpdateWorldResponse.DUPLICATE_BODY_ERROR)
        box_setup.send_and_check_goal()
        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.048)
        box_setup.check_cpi_geq([attached_link_name], 0.048)
        box_setup.detach_object(attached_link_name)

    def test_attached_collision3(self, box_setup):
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
        box_setup.set_and_check_cart_goal(p, box_setup.r_tip, box_setup.default_root)
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
        zero_pose.send_goal()

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
        zero_pose.send_goal()

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
        zero_pose.send_goal()

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
        zero_pose.set_and_check_cart_goal(p, zero_pose.r_tip, zero_pose.default_root)

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
        zero_pose.send_goal()

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
        zero_pose.set_and_check_cart_goal(p, zero_pose.l_tip, zero_pose.default_root)
        zero_pose.check_current_joint_state(js_goal)

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
        box_setup.set_and_check_cart_goal(p, box_setup.r_tip, box_setup.default_root)
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
        box_setup.send_goal()
        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.0)

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
        # FIXME visualization bug
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
        zero_pose.send_and_check_joint_goal(js)

        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.r_tip
        r_goal.pose.position.x = 0.4
        r_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[-1, 0, 0, 0],
                                                                      [0, 1, 0, 0],
                                                                      [0, 0, -1, 0],
                                                                      [0, 0, 0, 1]]))
        zero_pose.set_and_check_cart_goal(r_goal, zero_pose.l_tip, u'torso_lift_link')

        zero_pose.attach_box(box1_name, [.2, .04, .04], zero_pose.r_tip, [.1, 0, 0], [0, 0, 0, 1])
        zero_pose.attach_box(box2_name, [.2, .04, .04], zero_pose.l_tip, [.1, 0, 0], [0, 0, 0, 1])

        zero_pose.send_and_check_goal()

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
        kitchen_js = {u'sink_area_left_upper_drawer_main_joint': 0.45}
        kitchen_setup.set_kitchen_js(kitchen_js)

    def open_drawer(self, setup, tool_frame, handle_link, drawer_joint):
        setup.open_r_gripper()
        setup.open_l_gripper()
        tool_frame_goal = PoseStamped()
        tool_frame_goal.header.frame_id = handle_link
        # tool_frame_goal.pose.position.y = -.1
        tool_frame_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[-1, 0, 0, 0],
                                                                               [0, 0, -1, 0],
                                                                               [0, -1, 0, 0],
                                                                               [0, 0, 0, 1]]))
        setup.set_and_check_cart_goal(tool_frame_goal, tool_frame, setup.default_root)

        tool_frame_goal = PoseStamped()
        tool_frame_goal.header.frame_id = tool_frame
        tool_frame_goal.pose.position.x = -.45
        tool_frame_goal.pose.orientation.w = 1
        setup.set_and_check_cart_goal(tool_frame_goal, tool_frame, setup.default_root)

        kitchen_js = {drawer_joint: 0.45}
        setup.set_kitchen_js(kitchen_js)

    def close_drawer(self, setup, tool_frame, handle_link, drawer_joint):
        setup.open_r_gripper()
        setup.open_l_gripper()
        tool_frame_goal = PoseStamped()
        tool_frame_goal.header.frame_id = handle_link
        # tool_frame_goal.pose.position.y = -.1
        tool_frame_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[-1, 0, 0, 0],
                                                                               [0, 0, -1, 0],
                                                                               [0, -1, 0, 0],
                                                                               [0, 0, 0, 1]]))
        setup.set_and_check_cart_goal(tool_frame_goal, tool_frame, setup.default_root)

        kitchen_js = {drawer_joint: 0.}
        setup.set_kitchen_js(kitchen_js)

        tool_frame_goal = PoseStamped()
        tool_frame_goal.header.frame_id = tool_frame
        tool_frame_goal.pose.position.x = .45
        tool_frame_goal.pose.orientation.w = 1
        setup.set_and_check_cart_goal(tool_frame_goal, tool_frame, setup.default_root)

    # def test_milestone_demo(self, kitchen_setup):
    #     spoon_name = u'spoon'
    #     milk_name = u'milk'
    #
    #     # take milk out of fridge
    #     kitchen_setup.set_kitchen_js({u'iai_fridge_door_joint': 1.56})
    #
    #     # spawn milk
    #     milk_pose = PoseStamped()
    #     milk_pose.header.frame_id = u'iai_kitchen/iai_fridge_main_middle_level'
    #     milk_pose.pose.position = Point(0.1, 0, -0.07)
    #     milk_pose.pose.orientation = Quaternion(0, 0, 0, 1)
    #
    #     kitchen_setup.add_box(milk_name, [0.05, 0.05, 0.2], milk_pose)
    #
    #     # take spoon out of drawer
    #     # open drawer
    #     self.open_drawer(kitchen_setup, kitchen_setup.l_tip, u'iai_kitchen/sink_area_left_upper_drawer_handle',
    #                      u'sink_area_left_upper_drawer_main_joint')
    #
    #     # spawn spoon
    #     spoon_pose = PoseStamped()
    #     spoon_pose.header.frame_id = u'map'
    #     spoon_pose.pose.position = Point(0.940, 0.861, 0.745)
    #     spoon_pose.pose.orientation = Quaternion(0, 0, 0, 1)
    #
    #     kitchen_setup.add_box(spoon_name, [0.1, 0.02, 0.02], spoon_pose)
    #
    #     # put gripper above drawer
    #     pick_spoon_pose = PoseStamped()
    #     pick_spoon_pose.header.frame_id = u'base_footprint'
    #     pick_spoon_pose.pose.position = Point(0.567, 0.498, 0.89)
    #     pick_spoon_pose.pose.orientation = Quaternion(0.018, 0.702, 0.004, 0.712)
    #     kitchen_setup.keep_position(kitchen_setup.l_tip)
    #     kitchen_setup.set_and_check_cart_goal(pick_spoon_pose, kitchen_setup.r_tip, kitchen_setup.default_root)
    #
    #     # grasp spoon
    #     kitchen_setup.keep_position(kitchen_setup.l_tip)
    #     kitchen_setup.open_r_gripper()
    #     p = tf.lookup_pose(u'map', kitchen_setup.r_tip)
    #     p.pose.position = spoon_pose.pose.position
    #     kitchen_setup.keep_position(kitchen_setup.l_tip)
    #     kitchen_setup.set_cart_goal(p, kitchen_setup.r_tip, kitchen_setup.default_root)
    #     kitchen_setup.send_and_check_goal()
    #     current_pose = tf.lookup_pose(u'map', kitchen_setup.r_tip)
    #     assert current_pose.pose.position.z < 0.76
    #     kitchen_setup.allow_all_collisions()
    #     kitchen_setup.keep_position(kitchen_setup.l_tip)
    #     kitchen_setup.close_r_gripper()
    #     kitchen_setup.attach_existing(spoon_name, kitchen_setup.r_tip)
    #
    #     spoon_goal = PoseStamped()
    #     spoon_goal.header.frame_id = spoon_name
    #     spoon_goal.pose.position.z = .2
    #     spoon_goal.pose.orientation.w = 1
    #     kitchen_setup.keep_position(kitchen_setup.l_tip)
    #     kitchen_setup.set_and_check_cart_goal(spoon_goal, spoon_name, kitchen_setup.default_root)
    #
    #     # close drawer
    #     self.close_drawer(kitchen_setup, kitchen_setup.l_tip, u'iai_kitchen/sink_area_left_upper_drawer_handle',
    #                       u'sink_area_left_upper_drawer_main_joint')
    #
    #     kitchen_setup.send_and_check_joint_goal(gaya_pose)
    #
    #     # place spoon on kitchen isle
    #     spoon_goal = PoseStamped()
    #     spoon_goal.header.frame_id = u'iai_kitchen/kitchen_island_surface'
    #     spoon_goal.pose.position.y = 0.1
    #     spoon_goal.pose.orientation.w = 1
    #     kitchen_setup.set_cart_goal(spoon_goal, spoon_name, kitchen_setup.default_root)
    #     kitchen_setup.open_l_gripper()
    #     kitchen_setup.detach_object(spoon_name)
    #
    #     kitchen_setup.send_and_check_joint_goal(gaya_pose)
    #
    #     # grasp milk
    #     r_goal = deepcopy(milk_pose)
    #     r_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[-1, 0, 0, 0],
    #                                                                   [0, -1, 0, 0],
    #                                                                   [0, 0, 1, 0],
    #                                                                   [0, 0, 0, 1]]))
    #
    #     # take cereal out of vertical drawer and put it back
    #     # get bowl from left middle drawer left side
    #     # get cup from left middle drawer right side
    #     # put cup bowl and spoon in sink

    def test_ease_fridge(self, kitchen_setup):
        # FIXME less sensitive
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

        kitchen_setup.add_box(milk_name, [0.05, 0.05, 0.2], milk_pose)

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
        kitchen_setup.align_planes(kitchen_setup.l_tip, x, root_normal=x_map)

        # kitchen_setup.allow_collision([], milk_name, [])
        # kitchen_setup.add_json_goal(u'AvoidJointLimits', percentage=15)
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.send_and_check_goal()

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
        kitchen_setup.send_and_check_goal()
        kitchen_setup.send_and_check_joint_goal(gaya_pose)

        # place milk back

        # kitchen_setup.add_json_goal(u'BasePointingForward')
        # milk_goal = PoseStamped()
        # milk_goal.header.frame_id = u'iai_kitchen/kitchen_island_surface'
        # milk_goal.pose.position = Point(.1, -.2, .13)
        # milk_goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0,0,1]))
        kitchen_setup.set_cart_goal(milk_pre_pose, milk_name, kitchen_setup.default_root)
        kitchen_setup.send_and_check_goal()

        kitchen_setup.set_cart_goal(milk_pose, milk_name, kitchen_setup.default_root)
        kitchen_setup.send_and_check_goal()

        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        kitchen_setup.open_l_gripper()

        kitchen_setup.detach_object(milk_name)

        kitchen_setup.send_and_check_joint_goal(gaya_pose)

    def test_ease_cereal(self, kitchen_setup):
        # FIXME shaky af
        cereal_name = u'cereal'
        drawer_frame_id = u'iai_kitchen/oven_area_area_right_drawer_board_3_link'

        # take milk out of fridge
        kitchen_setup.set_kitchen_js({u'oven_area_area_right_drawer_main_joint': 0.48})

        kitchen_setup.set_json_goal(u'BasePointingForward')

        # spawn milk

        cereal_pose = PoseStamped()
        cereal_pose.header.frame_id = drawer_frame_id
        cereal_pose.pose.position = Point(0.123, -0.03, 0.11)
        cereal_pose.pose.orientation = Quaternion(0.0087786, 0.005395, -0.838767, -0.544393)
        kitchen_setup.add_box(cereal_name, [0.1528, 0.0634, 0.22894], cereal_pose)

        drawer_T_box = tf.msg_to_kdl(cereal_pose)

        # grasp milk
        kitchen_setup.open_l_gripper()
        grasp_pose = PoseStamped()
        grasp_pose.header.frame_id = cereal_name
        grasp_pose.pose.position = Point(0.1, 0, 0)
        grasp_pose.pose.orientation = Quaternion(0, 0, 1, 0)
        box_T_r_goal = tf.msg_to_kdl(grasp_pose)
        box_T_r_goal_pre = deepcopy(box_T_r_goal)
        box_T_r_goal_pre.p[0] += 0.1

        grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal_pre, drawer_frame_id)

        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=40)
        kitchen_setup.set_and_check_cart_goal(grasp_pose, tip_link=kitchen_setup.r_tip)

        kitchen_setup.attach_object(cereal_name, kitchen_setup.l_tip)
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
        kitchen_setup.set_cart_goal(grasp_pose, cereal_name, kitchen_setup.default_root)
        kitchen_setup.send_and_check_goal()
        kitchen_setup.send_and_check_joint_goal(gaya_pose)

        # place milk back

        # kitchen_setup.add_json_goal(u'BasePointingForward')
        # milk_goal = PoseStamped()
        # milk_goal.header.frame_id = u'iai_kitchen/kitchen_island_surface'
        # milk_goal.pose.position = Point(.1, -.2, .13)
        # milk_goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0,0,1]))
        kitchen_setup.set_cart_goal(grasp_pose, cereal_name, kitchen_setup.default_root)
        kitchen_setup.send_and_check_goal()

        kitchen_setup.set_cart_goal(cereal_pose, cereal_name, kitchen_setup.default_root)
        kitchen_setup.send_and_check_goal()

        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        kitchen_setup.open_l_gripper()

        kitchen_setup.detach_object(cereal_name)

        kitchen_setup.send_and_check_joint_goal(gaya_pose)

    # def test_nan(self, kitchen_setup):
    #     while True:
    #         kitchen_setup.allow_all_collisions()
    #         kitchen_setup.send_and_check_joint_goal(gaya_pose)
    #         js = {k: 0.0 for k in kitchen_setup.get_world().get_object('kitchen').get_controllable_joints()}
    #         kitchen_setup.set_kitchen_js(js)
    #         self.open_drawer(kitchen_setup, kitchen_setup.l_tip, u'iai_kitchen/sink_area_left_middle_drawer_handle',
    #                          u'sink_area_left_middle_drawer_main_joint')
    #
    # def test_nan2(self, kitchen_setup):
    #     tool_frame_goal = PoseStamped()
    #     tool_frame_goal.header.frame_id = kitchen_setup.l_tip
    #     tool_frame_goal.pose.position.x = -.45
    #     tool_frame_goal.pose.orientation.w = 1
    #     kitchen_setup.set_cart_goal(tool_frame_goal, kitchen_setup.l_tip, kitchen_setup.default_root)
    #     kitchen_setup.send_and_check_goal(execute=False)

    def test_bowl_and_cup(self, kitchen_setup):
        # kernprof -lv py.test -s test/test_integration_pr2.py::TestCollisionAvoidanceGoals::test_bowl_and_cup
        bowl_name = u'bowl'
        cup_name = u'cup'
        percentage = 50

        self.open_drawer(kitchen_setup, kitchen_setup.l_tip, u'iai_kitchen/sink_area_left_middle_drawer_handle',
                         u'sink_area_left_middle_drawer_main_joint')

        # spawn cup
        cup_pose = PoseStamped()
        cup_pose.header.frame_id = u'iai_kitchen/sink_area_left_middle_drawer_main'
        cup_pose.pose.position = Point(0.1, 0.2, -.05)
        cup_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup.add_cylinder(cup_name, [0.07, 0.04], cup_pose)

        # spawn bowl
        bowl_pose = PoseStamped()
        bowl_pose.header.frame_id = u'iai_kitchen/sink_area_left_middle_drawer_main'
        bowl_pose.pose.position = Point(0.1, -0.2, -.05)
        bowl_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup.add_cylinder(bowl_name, [0.05, 0.07], bowl_pose)
        kitchen_setup.send_and_check_joint_goal(gaya_pose)

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
        kitchen_setup.set_and_check_cart_goal(r_goal, kitchen_setup.r_tip, kitchen_setup.default_root)

        l_goal.pose.position.z -= .2
        r_goal.pose.position.z -= .2
        kitchen_setup.allow_collision([CollisionEntry.ALL], bowl_name, [CollisionEntry.ALL])
        kitchen_setup.allow_collision([CollisionEntry.ALL], cup_name, [CollisionEntry.ALL])
        kitchen_setup.set_cart_goal(l_goal, kitchen_setup.l_tip, kitchen_setup.default_root)
        kitchen_setup.set_cart_goal(r_goal, kitchen_setup.r_tip, kitchen_setup.default_root)
        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=percentage)
        kitchen_setup.send_and_check_goal()

        kitchen_setup.attach_object(bowl_name, kitchen_setup.l_tip)
        kitchen_setup.attach_object(cup_name, kitchen_setup.r_tip)

        kitchen_setup.send_and_check_joint_goal(gaya_pose)
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
        kitchen_setup.send_and_check_goal()

        kitchen_setup.detach_object(bowl_name)
        kitchen_setup.detach_object(cup_name)
        kitchen_setup.allow_collision([], cup_name, [])
        kitchen_setup.allow_collision([], bowl_name, [])
        kitchen_setup.send_and_check_joint_goal(gaya_pose)

    def test_ease_grasp_bowl(self, kitchen_setup):
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
        kitchen_setup.send_and_check_joint_goal(js)
        kitchen_setup.set_kitchen_js({u'sink_area_left_middle_drawer_main_joint': 0.45})

        r_goal = PoseStamped()
        r_goal.header.frame_id = kitchen_setup.r_tip
        r_goal.pose.position.x += 0.25
        r_goal.pose.orientation.w = 1

        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=percentage)
        kitchen_setup.set_and_check_cart_goal(r_goal, tip_link=kitchen_setup.r_tip)

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
        spoon_name = u'spoon'
        percentage = 40

        # spawn cup
        cup_pose = PoseStamped()
        cup_pose.header.frame_id = u'iai_kitchen/sink_area_surface'
        cup_pose.pose.position = Point(0.1, -.5, .02)
        cup_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup.add_box(spoon_name, [0.1, 0.02, 0.01], cup_pose)

        # kitchen_setup.send_and_check_joint_goal(gaya_pose)

        # grasp spoon
        l_goal = deepcopy(cup_pose)
        l_goal.pose.position.z += .2
        l_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 0, -1, 0],
                                                                      [0, -1, 0, 0],
                                                                      [-1, 0, 0, 0],
                                                                      [0, 0, 0, 1]]))
        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=percentage)
        kitchen_setup.set_and_check_cart_goal(l_goal, kitchen_setup.l_tip, kitchen_setup.default_root)

        l_goal.pose.position.z -= .2
        # kitchen_setup.allow_collision([CollisionEntry.ALL], spoon_name, [CollisionEntry.ALL])
        kitchen_setup.set_cart_goal(l_goal, kitchen_setup.l_tip, kitchen_setup.default_root)
        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=percentage)
        kitchen_setup.send_and_check_goal()
        kitchen_setup.attach_object(spoon_name, kitchen_setup.l_tip)

        l_goal.pose.position.z += .2
        # kitchen_setup.allow_collision([CollisionEntry.ALL], spoon_name, [CollisionEntry.ALL])
        kitchen_setup.set_cart_goal(l_goal, kitchen_setup.l_tip, kitchen_setup.default_root)
        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=percentage)
        kitchen_setup.send_and_check_goal()

        l_goal.pose.position.z -= .2
        # kitchen_setup.allow_collision([CollisionEntry.ALL], spoon_name, [CollisionEntry.ALL])
        kitchen_setup.set_cart_goal(l_goal, kitchen_setup.l_tip, kitchen_setup.default_root)
        kitchen_setup.set_json_goal(u'AvoidJointLimits', percentage=percentage)
        kitchen_setup.send_and_check_goal()

        kitchen_setup.send_and_check_joint_goal(gaya_pose)

    def test_ease_place_on_new_table(self, kitchen_setup):
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
        kitchen_setup.send_and_check_joint_goal(js)
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

        kitchen_setup.set_and_check_cart_goal(l_goal, tip_link=kitchen_setup.l_tip)

    def test_tray(self, kitchen_setup):
        # FIXME
        tray_name = u'tray'

        tray_pose = PoseStamped()
        tray_pose.header.frame_id = u'iai_kitchen/sink_area_surface'
        tray_pose.pose.position = Point(0.1, -0.4, 0.07)
        tray_pose.pose.orientation.w = 1

        kitchen_setup.add_box(tray_name, [.2, .4, .1], tray_pose)

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
        kitchen_setup.send_and_check_goal()

        kitchen_setup.attach_object(tray_name, kitchen_setup.r_tip)

        kitchen_setup.allow_collision(robot_links=[tray_name],
                                      body_b=kitchen_setup.get_robot().get_name(),
                                      link_bs=kitchen_setup.get_l_gripper_links())
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
        kitchen_setup.move_base(base_goal)

        kitchen_setup.allow_collision(robot_links=[tray_name],
                                      body_b=kitchen_setup.get_robot().get_name(),
                                      link_bs=kitchen_setup.get_l_gripper_links())

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
        kitchen_setup.set_and_check_cart_goal(tray_goal, tray_name, u'base_footprint')
        kitchen_setup.check_cart_goal(kitchen_setup.l_tip, expected_pose)

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
        # m = zero_pose.get_world().get_object(object_name).as_marker_msg()
        # compare_poses(m.pose, p.pose)

        hand_goal = PoseStamped()
        hand_goal.header.frame_id = u'lid'
        hand_goal.pose.position.y = -0.15
        hand_goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi/2, [0,0,1]))
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
        kitchen_setup.align_planes(hand, gripper_axis, root_normal=world_axis)
        # kitchen_setup.allow_all_collisions()

        kitchen_setup.send_and_check_goal()

        kitchen_setup.set_json_goal(u'Open',
                                    tip_link=hand,
                                    object_name=u'kitchen',
                                    object_link_name=handle_name,
                                    goal_joint_state=goal_angle,
                                    # weight=100
                                    )
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.send_and_check_goal()
        kitchen_setup.set_kitchen_js({u'sink_area_dish_washer_door_joint': goal_angle})
        # ----------------------------------------------------------------------------------------
        kitchen_setup.send_and_check_joint_goal(gaya_pose)

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
        kitchen_setup.send_and_check_goal()

        p = tf.lookup_pose(tray_handle_frame_id, hand)
        p.pose.position.x += 0.3

        # p = tf.transform_pose(hand, p)

        # kitchen_setup.add_json_goal(u'CartesianPosition',
        #                             root_link=kitchen_setup.default_root,
        #                             tip_link=hand,
        #                             goal=p)
        kitchen_setup.set_and_check_cart_goal(p, hand)

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
