from __future__ import division

import itertools
from copy import deepcopy

import numpy as np
import pytest
import roslaunch
import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Vector3Stamped, PointStamped
from giskard_msgs.msg import CollisionEntry, MoveActionGoal, MoveResult, WorldBody, MoveGoal
from giskard_msgs.srv import UpdateWorldResponse, UpdateWorldRequest
from numpy import pi
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from tf.transformations import quaternion_from_matrix, quaternion_about_axis

import giskardpy.tfwrapper as tf
from giskardpy import logging, identifier
from giskardpy.constraints import WEIGHT_ABOVE_CA
from giskardpy.identifier import fk_pose
from giskardpy.robot import Robot
from giskardpy.tfwrapper import init as tf_init
from giskardpy.utils import to_joint_state_dict2, publish_marker_vector
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
    rospy.init_node(u'tests')
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
        rospy.signal_shutdown(u'die')
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
    :type giskard: PR2
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
    :type zero_pose: PR2
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
                              tf.lookup_pose(u'map', u'iai_kitchen/world'), u'/kitchen/joint_states')
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
        assert result.error_code == MoveResult.INSOLVABLE

    def test_empty_goal(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.allow_self_collision()
        goal = MoveActionGoal()
        goal.goal.type = MoveGoal.PLAN_AND_EXECUTE
        result = zero_pose.send_goal(goal)
        assert result.error_code == MoveResult.INSOLVABLE

    def test_plan_only(self, zero_pose):
        zero_pose.allow_self_collision()
        zero_pose.set_joint_goal(pocky_pose)
        zero_pose.send_and_check_goal(goal_type=MoveGoal.PLAN_ONLY)
        zero_pose.check_joint_state(default_pose)

    def test_prismatic_joint1(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.allow_self_collision()
        js = {u'torso_lift_joint': 0.1}
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
        zero_pose.set_translation_goal(goal_position, zero_pose.l_tip)
        zero_pose.send_and_check_goal()



    def test_AvoidJointLimits1(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        percentage = 10
        zero_pose.allow_self_collision()
        zero_pose.add_json_goal(u'AvoidJointLimits',
                                percentage=percentage)
        zero_pose.send_and_check_goal()

        joint_non_continuous = [j for j in zero_pose.get_robot().controlled_joints if
                                not zero_pose.get_robot().is_joint_continuous(j)]

        current_joint_state = to_joint_state_dict2(zero_pose.get_current_joint_state())
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
        zero_pose.add_json_goal(u'AvoidJointLimits',
                                percentage=percentage)
        zero_pose.send_and_check_joint_goal(goal_state)

        zero_pose.allow_self_collision()
        zero_pose.add_json_goal(u'AvoidJointLimits',
                                percentage=percentage)
        zero_pose.send_and_check_goal()

        joint_non_continuous = [j for j in zero_pose.get_robot().controlled_joints if
                                not zero_pose.get_robot().is_joint_continuous(j)]

        current_joint_state = to_joint_state_dict2(zero_pose.get_current_joint_state())
        percentage *= 0.99  # if will not reach the exact percentager, because the weight is so low
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
        r_goal.header.frame_id = u'base_footprint'
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

        pocky_pose_setup.wrapper.update_god_map(updates)
        pocky_pose_setup.set_cart_goal(r_goal, u'base_footprint')
        pocky_pose_setup.send_and_check_goal()

        new_pose = tf.lookup_pose(u'map', u'base_footprint')
        compare_poses(old_pose.pose, new_pose.pose)

        assert pocky_pose_setup.get_god_map().unsafe_get_data(identifier.joint_weight + [u'odom_x_joint']) == 1000000
        assert pocky_pose_setup.get_god_map().unsafe_get_data(identifier.joint_weight + [u'torso_lift_joint']) == 0.01
        pocky_pose_setup.set_and_check_cart_goal(r_goal, u'base_footprint')
        assert pocky_pose_setup.get_god_map().unsafe_get_data(identifier.joint_weight + [u'odom_x_joint']) == 0.01
        assert pocky_pose_setup.get_god_map().unsafe_get_data(identifier.joint_weight + [u'torso_lift_joint']) == 0.01

    def test_base_pointing_forward(self, zero_pose):
        """
        :param zero_pose: PR2
        """
        zero_pose.add_json_goal(u'BasePointingForward')
        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.r_tip
        r_goal.pose.position.y = -2
        r_goal.pose.orientation.w = 1
        zero_pose.set_and_check_cart_goal(r_goal, zero_pose.r_tip)

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
        pocky_pose_setup.wrapper.update_god_map(updates)
        pocky_pose_setup.set_cart_goal(r_goal, pocky_pose_setup.r_tip)
        pocky_pose_setup.send_and_check_goal(expected_error_code=MoveResult.INSOLVABLE)
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
        pocky_pose_setup.wrapper.update_god_map(updates)
        pocky_pose_setup.set_cart_goal(r_goal, pocky_pose_setup.r_tip)
        pocky_pose_setup.send_and_check_goal(expected_error_code=MoveResult.INSOLVABLE)
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
        kitchen_setup.wrapper.pointing(tip, goal_point, pointing_axis=pointing_axis)
        kitchen_setup.send_and_check_goal()

        base_goal = PoseStamped()
        base_goal.header.frame_id = u'base_footprint'
        base_goal.pose.position.y = 2
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(1, [0, 0, 1]))
        kitchen_setup.wrapper.pointing(tip, goal_point, pointing_axis=pointing_axis)
        kitchen_setup.move_base(base_goal)

        current_x = Vector3Stamped()
        current_x.header.frame_id = tip
        current_x.vector.x = 1

        expected_x = tf.transform_point(tip, goal_point)
        np.testing.assert_almost_equal(expected_x.point.y, 0, 1)
        np.testing.assert_almost_equal(expected_x.point.z, 0, 1)

        tip = u'head_mount_kinect_rgb_link'
        goal_point = tf.lookup_point(u'map', kitchen_setup.r_tip)
        pointing_axis = Vector3Stamped()
        pointing_axis.header.frame_id = tip
        pointing_axis.vector.x = 1
        kitchen_setup.wrapper.pointing(tip, goal_point, pointing_axis=pointing_axis, root=kitchen_setup.r_tip)

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

        kitchen_setup.set_and_check_cart_goal(r_goal, kitchen_setup.r_tip, u'base_footprint')

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
        goal_state.name = ['r_elbow_flex_joint']
        goal_state.position = [-1.0]
        kwargs = {u'goal_state': goal_state}
        zero_pose.add_json_goal('jointpos', **kwargs)
        zero_pose.send_and_check_goal(expected_error_code=MoveResult.INSOLVABLE)

    def test_python_code_in_constraint_type(self, zero_pose):
        goal_state = JointState()
        goal_state.name = ['r_elbow_flex_joint']
        goal_state.position = [-1.0]
        kwargs = {u'goal_state': goal_state}
        zero_pose.add_json_goal('print("asd")', **kwargs)
        zero_pose.send_and_check_goal(expected_error_code=MoveResult.INSOLVABLE)

    def test_wrong_params1(self, zero_pose):
        goal_state = JointState()
        goal_state.name = 'r_elbow_flex_joint'
        goal_state.position = [-1.0]
        kwargs = {u'goal_state': goal_state}
        zero_pose.add_json_goal('JointPositionList', **kwargs)
        zero_pose.send_and_check_goal(expected_error_code=MoveResult.INSOLVABLE)

    def test_wrong_params2(self, zero_pose):
        goal_state = JointState()
        goal_state.name = [5432]
        goal_state.position = 'test'
        kwargs = {u'goal_state': goal_state}
        zero_pose.add_json_goal('JointPositionList', **kwargs)
        zero_pose.send_and_check_goal(expected_error_code=MoveResult.INSOLVABLE)

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

        kitchen_setup.add_json_goal(u'GraspBar',
                                    root=kitchen_setup.default_root,
                                    tip=kitchen_setup.r_tip,
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

        kitchen_setup.add_json_goal(u'GraspBar',
                                    root=kitchen_setup.default_root,
                                    tip=kitchen_setup.r_tip,
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
        kitchen_setup.align_planes(kitchen_setup.r_tip, x_gripper, root_normal=x_goal, weight=WEIGHT_ABOVE_CA)
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.add_json_goal(u'AvoidJointLimits', percentage=10)
        kitchen_setup.send_and_check_goal()

        kitchen_setup.add_json_goal(u'OpenDoor',
                                    tip=kitchen_setup.r_tip,
                                    object_name=u'kitchen',
                                    handle_link=handle_name,
                                    angle_goal=1.5)
        kitchen_setup.allow_all_collisions()
        kitchen_setup.add_json_goal(u'AvoidJointLimits')
        kitchen_setup.send_and_check_goal()
        kitchen_setup.set_kitchen_js({u'iai_fridge_door_joint': 1.5})

        kitchen_setup.add_json_goal(u'OpenDoor',
                                    tip=kitchen_setup.r_tip,
                                    object_name=u'kitchen',
                                    handle_link=handle_name,
                                    angle_goal=0)
        kitchen_setup.allow_all_collisions()
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

        kitchen_setup.add_json_goal(u'GraspBar',
                                    root=kitchen_setup.default_root,
                                    tip=kitchen_setup.r_tip,
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

        kitchen_setup.add_json_goal(u'Open',
                                    tip=kitchen_setup.r_tip,
                                    object_name=u'kitchen',
                                    handle_link=handle_name)
        kitchen_setup.allow_all_collisions()

        kitchen_setup.send_and_check_goal()
        kitchen_setup.set_kitchen_js({u'iai_fridge_door_joint': np.pi / 2})

        kitchen_setup.add_json_goal(u'Close',
                                    tip=kitchen_setup.r_tip,
                                    object_name=u'kitchen',
                                    handle_link=handle_name)
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

        kitchen_setup.add_json_goal(u'Close',
                                    tip=elbow,
                                    object_name=u'kitchen',
                                    handle_link=handle_name)
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

        kitchen_setup.add_json_goal(u'GraspBar',
                                    root=kitchen_setup.default_root,
                                    tip=kitchen_setup.l_tip,
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

        kitchen_setup.add_json_goal(u'OpenDoor',
                                    tip=kitchen_setup.l_tip,
                                    object_name=u'kitchen',
                                    handle_link=handle_name,
                                    angle_goal=goal_angle)
        kitchen_setup.allow_all_collisions()
        kitchen_setup.send_and_check_goal()
        kitchen_setup.set_kitchen_js({u'oven_area_oven_door_joint': goal_angle})

        kitchen_setup.add_json_goal(u'OpenDoor',
                                    tip=kitchen_setup.l_tip,
                                    object_name=u'kitchen',
                                    handle_link=handle_name,
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

        kitchen_setup.add_json_goal(u'GraspBar',
                                    root=kitchen_setup.default_root,
                                    tip=hand,
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

        kitchen_setup.add_json_goal(u'Open',
                                    tip=hand,
                                    object_name=u'kitchen',
                                    handle_link=handle_name,
                                    goal_joint_state=goal_angle,
                                    # weight=100
                                    )
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.send_and_check_goal()
        kitchen_setup.set_kitchen_js({u'sink_area_dish_washer_door_joint': goal_angle})

        kitchen_setup.add_json_goal(u'Close',
                                    tip=hand,
                                    object_name=u'kitchen',
                                    handle_link=handle_name)
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

        kitchen_setup.add_json_goal(u'GraspBar',
                                    root=kitchen_setup.default_root,
                                    tip=kitchen_setup.r_tip,
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
        x_goal.header.frame_id = handle_frame_id
        x_goal.vector.x = -1

        kitchen_setup.align_planes(kitchen_setup.l_tip,
                                   x_gripper,
                                   root_normal=x_goal)
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.send_and_check_goal()

        kitchen_setup.add_json_goal(u'Open',
                                    tip=kitchen_setup.l_tip,
                                    object_name=u'kitchen',
                                    handle_link=handle_name)
        kitchen_setup.allow_all_collisions()  # makes execution faster
        kitchen_setup.send_and_check_goal()  # send goal to Giskard
        # Update kitchen object
        kitchen_setup.set_kitchen_js({u'sink_area_left_middle_drawer_main_joint': 0.48})

        # Close drawer partially
        kitchen_setup.add_json_goal(u'OpenDrawer',
                                    tip=kitchen_setup.l_tip,
                                    object_name=u'kitchen',
                                    handle_link=handle_name,
                                    distance_goal=0.2)
        kitchen_setup.allow_all_collisions()  # makes execution faster
        kitchen_setup.send_and_check_goal()  # send goal to Giskard
        # Update kitchen object
        kitchen_setup.set_kitchen_js({u'sink_area_left_middle_drawer_main_joint': 0.2})

        kitchen_setup.add_json_goal(u'Close',
                                    tip=kitchen_setup.l_tip,
                                    object_name=u'kitchen',
                                    handle_link=handle_name)
        kitchen_setup.allow_all_collisions()  # makes execution faster
        kitchen_setup.send_and_check_goal()  # send goal to Giskard
        # Update kitchen object
        kitchen_setup.set_kitchen_js({u'sink_area_left_middle_drawer_main_joint': 0.0})

        # TODO: calculate real and desired value and compare

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

        kitchen_setup.set_cart_goal(l_goal, kitchen_setup.l_tip)
        kitchen_setup.set_cart_goal(r_goal, kitchen_setup.r_tip)
        # kitchen_setup.allow_collision([], tray_name, [])
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
        :type box_setup: PR2
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
        pocky_pose_setup.set_and_check_cart_goal(p, pocky_pose_setup.r_tip, pocky_pose_setup.default_root)

        # box_setup.wrapper.avoid_collision()

        # collision_entry = CollisionEntry()
        # collision_entry.type = CollisionEntry.AVOID_COLLISION
        # collision_entry.min_dist = 0.05
        # collision_entry.body_b = u'box'
        # pocky_pose_setup.add_collision_entries([collision_entry])
        #
        # pocky_pose_setup.send_and_check_goal(expected_error_code=MoveResult.INSOLVABLE)

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

        zero_pose.add_waypoint()
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position = Point(0.0, -0.1, -0.1)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_cart_goal(p, zero_pose.r_tip, zero_pose.default_root)

        zero_pose.add_waypoint()
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
        zero_pose.add_waypoint()
        zero_pose.set_joint_goal(pick_up_pose)
        zero_pose.add_waypoint()
        zero_pose.set_joint_goal(gaya_pose)

        zero_pose.send_and_check_goal()
        traj = zero_pose.get_trajectory_msg()
        for i, joint_state in enumerate(traj):
            try:
                zero_pose.compare_joint_state(joint_state, pocky_pose)
                break
            except AssertionError:
                pass
        else:  # if no break
            assert False, u'pocky pose not in trajectory'

        traj = traj[i:]
        for i, joint_state in enumerate(traj):
            try:
                zero_pose.compare_joint_state(joint_state, pick_up_pose)
                break
            except AssertionError:
                pass
        else:  # if no break
            assert False, u'pick_up_pose not in trajectory'

        traj = traj[i:]
        for i, joint_state in enumerate(traj):
            try:
                zero_pose.compare_joint_state(joint_state, gaya_pose)
                break
            except AssertionError:
                pass
        else:  # if no break
            assert False, u'gaya_pose not in trajectory'

        pass

    # TODO test translation and orientation goal in different frame


class TestCollisionAvoidanceGoals(object):
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
                                    tip=kitchen_setup.r_tip,
                                    root=kitchen_setup.l_tip)
        kitchen_setup.send_and_check_goal()
        kitchen_setup.check_cart_goal(kitchen_setup.r_tip, r_goal)

        kitchen_setup.detach_object('box')
        kitchen_setup.attach_existing('box', kitchen_setup.r_tip)

        r_goal2 = PoseStamped()
        r_goal2.header.frame_id = 'box'
        r_goal2.pose.position.x -= -.1
        r_goal2.pose.orientation.w = 1

        kitchen_setup.set_cart_goal(r_goal2, u'box', root=kitchen_setup.l_tip)
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
        p.header.frame_id = u'map'
        p.pose.position = Point(1.2, 0, 1.6)
        p.pose.orientation = Quaternion(0.0, 0.0, 0.47942554, 0.87758256)
        zero_pose.add_mesh(object_name, path=u'urdfs/bowl.obj', pose=p)
        # m = zero_pose.get_world().get_object(object_name).as_marker_msg()
        # compare_poses(m.pose, p.pose)
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
            zero_pose.attach_existing(pocky, zero_pose.r_tip)
            zero_pose.detach_object(pocky)
        zero_pose.remove_object(pocky)

    def test_remove_attached_box(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        pocky = u'http://muh#pocky'
        zero_pose.attach_box(pocky, [0.1, 0.02, 0.02], zero_pose.r_tip, [0.05, 0, 0])
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
        zero_pose.attach_existing(pocky, frame_id=zero_pose.r_tip)
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
        zero_pose.attach_existing(object_name, frame_id=zero_pose.r_tip)
        zero_pose.detach_object(object_name)
        zero_pose.remove_object(object_name)
        zero_pose.add_box(object_name, pose=p)
        assert zero_pose.wrapper.get_attached_objects().object_names == []

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
        zero_pose.attach_existing(pocky, frame_id=zero_pose.r_tip)
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
        zero_pose.attach_existing(pocky, frame_id=zero_pose.r_tip)
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
        assert zero_pose.wrapper.update_world.call(req).error_codes == UpdateWorldResponse.INVALID_OPERATION

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
        assert zero_pose.wrapper.update_world.call(req).error_codes == UpdateWorldResponse.CORRUPT_SHAPE_ERROR

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
        assert kitchen_setup.wrapper.update_world.call(req).error_codes == UpdateWorldResponse.UNSUPPORTED_OPTIONS

    def test_link_b_set_but_body_b_not(self, box_setup):
        """
        :type box_setup: PR2
        """
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.link_bs = [u'asdf']
        box_setup.add_collision_entries([ce])
        box_setup.send_and_check_goal(MoveResult.INSOLVABLE)

    def test_unknown_robot_link(self, box_setup):
        """
        :type box_setup: PR2
        """
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.robot_links = [u'asdf']
        box_setup.add_collision_entries([ce])
        box_setup.send_and_check_goal(MoveResult.UNKNOWN_OBJECT)

    def test_unknown_body_b(self, box_setup):
        """
        :type box_setup: PR2
        """
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.body_b = u'asdf'
        box_setup.add_collision_entries([ce])
        box_setup.send_and_check_goal(MoveResult.UNKNOWN_OBJECT)

    def test_unknown_link_b(self, box_setup):
        """
        :type box_setup: PR2
        """
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.body_b = u'box'
        ce.link_bs = [u'asdf']
        box_setup.add_collision_entries([ce])
        box_setup.send_and_check_goal(MoveResult.UNKNOWN_OBJECT)

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
        zero_pose.send_and_check_joint_goal(pocky_pose)

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
        box_setup.add_collision_entries([collision_entry])

        box_setup.send_and_check_goal(MoveResult.UNKNOWN_OBJECT)

    # def test_interrupt1(self, box_setup):
    #     """
    #     :type box_setup: Context
    #     """
    #     # FIXME
    #     box_setup.add_box()
    #     p = PoseStamped()
    #     p.header.frame_id = box_setup.r_tip
    #     p.pose.position = Point(0.1, 0, 0)
    #     p.pose.orientation = Quaternion(0, 0, 0, 1)
    #     box_setup.set_cart_goal(box_setup.default_root, box_setup.r_tip, p)
    #
    #     collision_entry = CollisionEntry()
    #     collision_entry.type = CollisionEntry.AVOID_ALL_COLLISIONS
    #     collision_entry.min_dist = 0.5
    #     box_setup.set_collision_entries([collision_entry])
    #
    #     box_setup.send_goal(wait=False)
    #     rospy.sleep(.5)
    # box_setup.interrupt()
    # result = self.giskard.get_result()
    # self.assertEqual(result.error_code, MoveResult.INTERRUPTED)

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
        zero_pose.add_collision_entries(ces)

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
        box_setup.add_collision_entries([ce])
        box_setup.send_and_check_goal(MoveResult.SUCCESS)
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

    def test_collision_override2(self, kitchen_setup):
        """
        :type box_setup: PR2
        """
        goal = PoseStamped()
        goal.header.frame_id = u'base_footprint'
        goal.pose.position.x += 1
        goal.pose.orientation.w = 1
        kitchen_setup.set_and_check_cart_goal(goal, u'base_footprint')
        kitchen_setup.check_cpi_geq([u'base_link'], 0.049)

    def test_avoid_collision2(self, fake_table_setup):
        """
        :type box_setup: PR2
        """
        r_goal = PoseStamped()
        r_goal.header.frame_id = u'map'
        r_goal.pose.position.x = 0.8
        r_goal.pose.position.y = -0.38
        r_goal.pose.position.z = 0.82
        r_goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 2, [0, 1, 0]))
        fake_table_setup.set_and_check_cart_goal(r_goal, fake_table_setup.r_tip)
        fake_table_setup.check_cpi_geq(fake_table_setup.get_l_gripper_links(), 0.048)
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
        box_setup.wrapper.set_collision_entries([collision_entry])

        box_setup.allow_self_collision()
        box_setup.set_and_check_cart_goal(p, box_setup.r_tip, box_setup.default_root)

        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.0)
        box_setup.check_cpi_leq(box_setup.get_r_gripper_links(), 0.0)

    def test_avoid_collision3(self, pocky_pose_setup):
        """
        :type box_setup: PR2
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
        :type box_setup: PR2
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
        :type box_setup: PR2
        """
        # fixme
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

        pocky_pose_setup.send_and_check_goal()

    def test_avoid_collision6(self, fake_table_setup):
        """
        :type box_setup: PR2
        """
        # FIXME
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
        fake_table_setup.send_and_check_joint_goal(js)
        fake_table_setup.check_cpi_geq(fake_table_setup.get_l_gripper_links(), 0.048)
        fake_table_setup.check_cpi_leq([u'r_gripper_l_finger_tip_link'], 0.04)
        fake_table_setup.check_cpi_leq([u'r_gripper_r_finger_tip_link'], 0.04)

    def test_avoid_collision7(self, kitchen_setup):
        """
        :type box_setup: PR2
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
        kitchen_setup.check_joint_state(gaya_pose)

    def test_avoid_collision9(self, kitchen_setup):
        """
        :type box_setup: PR2
        """
        base_pose = PoseStamped()
        base_pose.header.frame_id = u'map'
        base_pose.pose.position.x = 0.8
        base_pose.pose.position.y = 0.9
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 2, [0, 0, 1]))
        kitchen_setup.teleport_base(base_pose)
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0, 0, 1]))
        kitchen_setup.set_joint_goal(gaya_pose)
        kitchen_setup.set_and_check_cart_goal(base_pose, u'base_footprint')
        kitchen_setup.check_joint_state(gaya_pose)

    def test_avoid_collision8(self, kitchen_setup):
        """
        :type box_setup: PR2
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
        kitchen_setup.check_joint_state(gaya_pose)

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
        pocky_pose_setup.add_collision_entries([collision_entry])

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
        box_setup.add_collision_entries([collision_entry])

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
        box_setup.add_collision_entries([collision_entry])

        box_setup.send_and_check_goal()

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_ALL_COLLISIONS
        collision_entry.min_dist = 0.05
        box_setup.add_collision_entries([collision_entry])

        box_setup.send_and_check_goal()

        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.0)
        box_setup.check_cpi_geq(box_setup.get_r_gripper_links(), 0.0)

    def test_allow_collision_gripper(self, box_setup):
        """
        :type box_setup: PR2
        """
        ces = box_setup.get_allow_l_gripper(u'box')
        box_setup.add_collision_entries(ces)
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
        :type box_setup: PR2
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
        :type box_setup: PR2
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
        :type box_setup: PR2
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
        zero_pose.check_joint_state(js_goal)

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
        box_setup.add_collision_entries(ces)

        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.y = -0.11
        p.pose.orientation.w = 1
        box_setup.set_and_check_cart_goal(p, box_setup.r_tip, box_setup.default_root)
        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.048)
        box_setup.check_cpi_leq([pocky], 0.0)

    # def test_collision_during_planning1(self, box_setup):
    #     """
    #     :type box_setup: PR2
    #     """
    # FIXME feature not implemented
    #     # FIXME sometimes says endless wiggle detected
    #     p = PoseStamped()
    #     p.header.frame_id = box_setup.r_tip
    #     p.header.stamp = rospy.get_rostime()
    #     p.pose.position = Point(0.1, 0, 0)
    #     p.pose.orientation = Quaternion(0, 0, 0, 1)
    #
    #     collision_entry = CollisionEntry()
    #     collision_entry.type = CollisionEntry.AVOID_ALL_COLLISIONS
    #     collision_entry.min_dist = 1
    #     box_setup.add_collision_entries([collision_entry])
    #     box_setup.set_cart_goal(box_setup.default_root, box_setup.r_tip, p)
    #     box_setup.send_and_check_goal(expected_error_code=MoveResult.PATH_COLLISION)

    # def test_collision_during_planning2(self, box_setup):
    #     """
    #     :type box_setup: PR2
    #     """
    # FIXME feature not implemented
    #     # FIXME sometimes says endless wiggle detected
    #     p = PoseStamped()
    #     p.header.frame_id = box_setup.r_tip
    #     p.header.stamp = rospy.get_rostime()
    #     p.pose.position = Point(0.1, 0, 0)
    #     p.pose.orientation = Quaternion(0, 0, 0, 1)
    #
    #     collision_entry = CollisionEntry()
    #     collision_entry.type = CollisionEntry.AVOID_ALL_COLLISIONS
    #     collision_entry.min_dist = 1
    #     box_setup.add_collision_entries([collision_entry])
    #     box_setup.set_cart_goal(box_setup.default_root, box_setup.r_tip, p)
    #     box_setup.send_and_check_goal(expected_error_code=MoveResult.PATH_COLLISION)
    #
    #     box_setup.set_joint_goal(pocky_pose)
    #     box_setup.allow_all_collisions()
    #     box_setup.send_and_check_goal()

    def test_avoid_collision_gripper(self, box_setup):
        """
        :type box_setup: PR2
        """
        box_setup.allow_all_collisions()
        ces = box_setup.get_l_gripper_collision_entries(u'box', 0.05, CollisionEntry.AVOID_COLLISION)
        box_setup.add_collision_entries(ces)
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
    #     kitchen_setup.wrapper.allow_collision(kitchen_setup.get_l_gripper_links(), u'kitchen',
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
        milk_name = u'milk'

        # take milk out of fridge
        kitchen_setup.set_kitchen_js({u'iai_fridge_door_joint': 1.56})

        # base_goal = PoseStamped()
        # base_goal.header.frame_id = 'map'
        # base_goal.pose.position.x = 0.565
        # base_goal.pose.position.y = -0.5
        # base_goal.pose.orientation.z = -0.51152562713
        # base_goal.pose.orientation.w = 0.85926802151
        # kitchen_setup.teleport_base(base_goal)
        kitchen_setup.add_json_goal(u'BasePointingForward')

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

        kitchen_setup.add_json_goal(u'GraspBar',
                                    root=kitchen_setup.default_root,
                                    tip=kitchen_setup.l_tip,
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
        kitchen_setup.send_and_check_goal()

        kitchen_setup.attach_existing(milk_name, kitchen_setup.l_tip)
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
        bowl_name = u'bowl'
        cup_name = u'cup'

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
        kitchen_setup.set_and_check_cart_goal(r_goal, kitchen_setup.r_tip, kitchen_setup.default_root)

        l_goal.pose.position.z -= .2
        r_goal.pose.position.z -= .2
        kitchen_setup.allow_collision([CollisionEntry.ALL], bowl_name, [CollisionEntry.ALL])
        kitchen_setup.allow_collision([CollisionEntry.ALL], cup_name, [CollisionEntry.ALL])
        kitchen_setup.set_cart_goal(l_goal, kitchen_setup.l_tip, kitchen_setup.default_root)
        kitchen_setup.set_cart_goal(r_goal, kitchen_setup.r_tip, kitchen_setup.default_root)
        kitchen_setup.send_and_check_goal()

        kitchen_setup.attach_existing(bowl_name, kitchen_setup.l_tip)
        kitchen_setup.attach_existing(cup_name, kitchen_setup.r_tip)

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
        kitchen_setup.add_json_goal(u'AvoidJointLimits', percentage=15)
        kitchen_setup.send_and_check_goal()

        kitchen_setup.detach_object(bowl_name)
        kitchen_setup.detach_object(cup_name)
        kitchen_setup.allow_collision([], cup_name, [])
        kitchen_setup.allow_collision([], bowl_name, [])
        kitchen_setup.send_and_check_joint_goal(gaya_pose)

        # fixme
        # self.close_drawer(kitchen_setup, kitchen_setup.l_tip, u'iai_kitchen/sink_area_left_middle_drawer_handle',
        #                   u'sink_area_left_middle_drawer_main_joint')

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

    def test_spoon(self, kitchen_setup):
        spoon_name = u'spoon'

        # spawn cup
        cup_pose = PoseStamped()
        cup_pose.header.frame_id = u'iai_kitchen/sink_area_surface'
        cup_pose.pose.position = Point(0.1, -.5, .02)
        cup_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup.add_box(spoon_name, [0.1, 0.02, 0.01], cup_pose)

        kitchen_setup.send_and_check_joint_goal(gaya_pose)

        # grasp spoon
        l_goal = deepcopy(cup_pose)
        l_goal.pose.position.z += .2
        l_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 0, -1, 0],
                                                                      [0, -1, 0, 0],
                                                                      [-1, 0, 0, 0],
                                                                      [0, 0, 0, 1]]))
        kitchen_setup.set_and_check_cart_goal(l_goal, kitchen_setup.l_tip, kitchen_setup.default_root)

        l_goal.pose.position.z -= .2
        # kitchen_setup.allow_collision([CollisionEntry.ALL], spoon_name, [CollisionEntry.ALL])
        kitchen_setup.set_cart_goal(l_goal, kitchen_setup.l_tip, kitchen_setup.default_root)
        kitchen_setup.send_and_check_goal()
        kitchen_setup.attach_existing(spoon_name, kitchen_setup.l_tip)

        l_goal.pose.position.z += .2
        # kitchen_setup.allow_collision([CollisionEntry.ALL], spoon_name, [CollisionEntry.ALL])
        kitchen_setup.set_cart_goal(l_goal, kitchen_setup.l_tip, kitchen_setup.default_root)
        kitchen_setup.send_and_check_goal()

        l_goal.pose.position.z -= .2
        # kitchen_setup.allow_collision([CollisionEntry.ALL], spoon_name, [CollisionEntry.ALL])
        kitchen_setup.set_cart_goal(l_goal, kitchen_setup.l_tip, kitchen_setup.default_root)
        kitchen_setup.send_and_check_goal()

        kitchen_setup.send_and_check_joint_goal(gaya_pose)
        # base_goal = PoseStamped()
        # base_goal.header.frame_id = u'base_footprint'
        # base_goal.pose.position.x = -.1
        # base_goal.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        # kitchen_setup.teleport_base(base_goal)
        #
        # # place bowl and cup
        # bowl_goal = PoseStamped()
        # bowl_goal.header.frame_id = u'iai_kitchen/kitchen_island_surface'
        # bowl_goal.pose.position = Point(.2, 0, .05)
        # bowl_goal.pose.orientation = Quaternion(0, 0, 0, 1)
        #
        # cup_goal = PoseStamped()
        # cup_goal.header.frame_id = u'iai_kitchen/kitchen_island_surface'
        # cup_goal.pose.position = Point(.15, 0.25, .07)
        # cup_goal.pose.orientation = Quaternion(0, 0, 0, 1)
        #
        # kitchen_setup.set_cart_goal(bowl_goal, bowl_name, kitchen_setup.default_root)
        # kitchen_setup.set_cart_goal(cup_goal, cup_name, kitchen_setup.default_root)
        # kitchen_setup.send_and_check_goal()
        #
        # kitchen_setup.detach_object(bowl_name)
        # kitchen_setup.detach_object(cup_name)
        # kitchen_setup.allow_collision([], cup_name, [])
        # kitchen_setup.allow_collision([], bowl_name, [])
        # kitchen_setup.send_and_check_joint_goal(gaya_pose)

    def test_tray(self, kitchen_setup):
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

        kitchen_setup.attach_existing(tray_name, kitchen_setup.r_tip)

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

    def test_ease_dishwasher(self, kitchen_setup):
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

        kitchen_setup.add_json_goal(u'GraspBar',
                                    root=kitchen_setup.default_root,
                                    tip=hand,
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

        kitchen_setup.add_json_goal(u'Open',
                                    tip=hand,
                                    object_name=u'kitchen',
                                    handle_link=handle_name,
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

        kitchen_setup.add_json_goal(u'GraspBar',
                                    root=kitchen_setup.default_root,
                                    tip=hand,
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
        #                             tip=hand,
        #                             object_name=u'kitchen',
        #                             handle_link=handle_name)
        # kitchen_setup.allow_all_collisions()
        # kitchen_setup.send_and_check_goal()
        # kitchen_setup.set_kitchen_js({u'sink_area_dish_washer_door_joint': 0})


class TestReachability():
    def test_unreachable_goal_0(self, zero_pose):
        js = {}
        js['r_shoulder_lift_joint'] = 10
        zero_pose.set_joint_goal(js)
        zero_pose.check_reachability(expected_error_code=MoveResult.UNREACHABLE)

    def test_unreachable_goal_1(self, zero_pose):
        pose = PoseStamped()
        pose.header.frame_id = zero_pose.r_tip
        pose.pose.position.z = -2
        pose.pose.orientation.w = 1
        zero_pose.set_cart_goal(root=zero_pose.default_root, tip=zero_pose.r_tip, goal_pose=pose)
        zero_pose.check_reachability(expected_error_code=MoveResult.UNREACHABLE)

    def test_unreachable_goal_2(self, zero_pose):
        pose = PoseStamped()
        pose.pose.position.z = 5
        pose.header.frame_id = 'map'
        pose.pose.orientation.w = 1
        zero_pose.set_cart_goal(root='odom_combined', tip='r_gripper_tool_frame', goal_pose=pose)
        zero_pose.check_reachability(expected_error_code=MoveResult.UNREACHABLE)

    def test_unreachable_goal_3(self, zero_pose):
        pose = PoseStamped()
        pose.pose.position.z = 1.2
        pose.header.frame_id = 'map'
        pose.pose.orientation.w = 1
        zero_pose.set_translation_goal(root='odom_combined', tip='r_gripper_tool_frame', goal_pose=pose)
        js = {}
        js['r_shoulder_lift_joint'] = 1.0  # soft lower -0.3536 soft upper 1.2963
        js['r_elbow_flex_joint'] = -0.2  # soft lower -2.1213 soft upper -0.15
        js['torso_lift_joint'] = 0.15  # soft lower 0.0115 soft upper 0.325
        zero_pose.set_joint_goal(js)
        zero_pose.check_reachability(expected_error_code=MoveResult.UNREACHABLE)

    def test_unreachable_goal_4(self, zero_pose):
        pose = PoseStamped()
        pose.pose.position.y = -2.0
        pose.pose.position.z = 1
        pose.header.frame_id = 'map'
        pose.pose.orientation.w = 1
        zero_pose.set_translation_goal(root='odom_combined', tip='r_gripper_tool_frame', goal_pose=pose)
        pose2 = PoseStamped()
        pose2.pose.position.y = 2.0
        pose2.pose.position.z = 1
        pose2.header.frame_id = 'map'
        pose.pose.orientation.w = 1
        zero_pose.set_translation_goal(root='odom_combined', tip='l_gripper_tool_frame', goal_pose=pose2)
        zero_pose.check_reachability(expected_error_code=MoveResult.UNREACHABLE)

    def test_unreachable_goal_5(self, zero_pose):
        pose = PoseStamped()
        pose.pose.position.x = 2
        pose.header.frame_id = 'map'
        pose.pose.orientation.w = 1
        zero_pose.set_cart_goal(root='r_shoulder_lift_link', tip='r_gripper_tool_frame', goal_pose=pose)
        zero_pose.check_reachability(expected_error_code=MoveResult.UNREACHABLE)

    def test_unreachable_goal_6(self, zero_pose):  # TODO torso lift joint xdot has a wrong value
        pose = PoseStamped()
        pose.pose.position.x = 0.0
        pose.pose.position.y = 0.0
        pose.pose.position.z = 0.0
        pose.header.frame_id = 'map'
        pose.pose.orientation.w = 1
        zero_pose.set_translation_goal(root='odom_combined', tip='base_footprint', goal_pose=pose)
        js = {}
        js['odom_x_joint'] = 0.01
        zero_pose.set_joint_goal(js)
        zero_pose.check_reachability(expected_error_code=MoveResult.UNREACHABLE)

    def test_unreachable_goal_7(self, zero_pose):
        js = {}
        js['r_shoulder_lift_joint'] = 0.2
        js['r_elbow_flex_joint'] = -2.5
        js['torso_lift_joint'] = 0.15
        zero_pose.set_joint_goal(js)
        zero_pose.check_reachability(expected_error_code=MoveResult.UNREACHABLE)

    def test_reachable_goal_0(self, zero_pose):
        pose = PoseStamped()
        pose.pose.position.x = 3.0
        pose.pose.position.z = 1.0
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.707
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 0.707
        pose.header.frame_id = 'map'
        zero_pose.set_cart_goal(root='odom_combined', tip='r_gripper_tool_frame', goal_pose=pose)
        zero_pose.check_reachability()
        zero_pose.set_cart_goal(root='odom_combined', tip='r_gripper_tool_frame', goal_pose=pose)
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
        zero_pose.set_cart_goal(root='odom_combined', tip='r_gripper_tool_frame', goal_pose=pose)
        zero_pose.check_reachability()
        zero_pose.set_cart_goal(root='odom_combined', tip='r_gripper_tool_frame', goal_pose=pose)
        zero_pose.send_and_check_goal(goal_type=MoveGoal.PLAN_ONLY)

    def test_reachable_goal_2(self, zero_pose):
        pose = PoseStamped()
        pose.pose.position.x = -1.0
        pose.pose.position.z = 1.0
        pose.header.frame_id = 'map'
        zero_pose.set_cart_goal(root='odom_combined', tip='r_gripper_tool_frame', goal_pose=pose)
        zero_pose.check_reachability()
        zero_pose.set_cart_goal(root='odom_combined', tip='r_gripper_tool_frame', goal_pose=pose)
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
        zero_pose.set_cart_goal(root='odom_combined', tip='r_gripper_tool_frame', goal_pose=pose)
        zero_pose.check_reachability()
        zero_pose.set_cart_goal(root='odom_combined', tip='r_gripper_tool_frame', goal_pose=pose)
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
        zero_pose.set_cart_goal(root='odom_combined', tip='r_gripper_tool_frame', goal_pose=pose)
        js = {}
        js['r_elbow_flex_joint'] = -0.2
        js['torso_lift_joint'] = 0.15
        zero_pose.set_joint_goal(js)
        zero_pose.allow_all_collisions()
        zero_pose.check_reachability()

        zero_pose.set_cart_goal(root='odom_combined', tip='r_gripper_tool_frame', goal_pose=pose)
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
