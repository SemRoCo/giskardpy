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
from std_srvs.srv import Trigger
from tf.transformations import quaternion_from_matrix, quaternion_about_axis, quaternion_from_euler

import giskardpy.utils.tfwrapper as tf
from giskard_msgs.msg import MoveResult, WorldBody, MoveGoal
from giskard_msgs.srv import UpdateWorldResponse, UpdateWorldRequest
from giskardpy import identifier
from giskardpy.configs.pr2 import PR2_Mujoco
from giskardpy.goals.goal import WEIGHT_ABOVE_CA, WEIGHT_BELOW_CA, WEIGHT_COLLISION_AVOIDANCE
from giskardpy.identifier import fk_pose
from giskardpy.model.world import SubWorldTree
from utils_for_tests import compare_poses, compare_points, compare_orientations, publish_marker_vector, \
    JointGoalChecker, GiskardTestWrapper

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

folder_name = 'tmp_data/'


class PR2TestWrapper(GiskardTestWrapper):
    default_pose = {'r_elbow_flex_joint': -0.15,
                    'r_forearm_roll_joint': 0,
                    'r_shoulder_lift_joint': 0,
                    'r_shoulder_pan_joint': 0,
                    'r_upper_arm_roll_joint': 0,
                    'r_wrist_flex_joint': -0.10001,
                    'r_wrist_roll_joint': 0,
                    'l_elbow_flex_joint': -0.15,
                    'l_forearm_roll_joint': 0,
                    'l_shoulder_lift_joint': 0,
                    'l_shoulder_pan_joint': 0,
                    'l_upper_arm_roll_joint': 0,
                    'l_wrist_flex_joint': -0.10001,
                    'l_wrist_roll_joint': 0,
                    'torso_lift_joint': 0.2,
                    'head_pan_joint': 0,
                    'head_tilt_joint': 0}

    better_pose = {'r_shoulder_pan_joint': -1.7125,
                   'r_shoulder_lift_joint': -0.25672,
                   'r_upper_arm_roll_joint': -1.46335,
                   'r_elbow_flex_joint': -2.12,
                   'r_forearm_roll_joint': 1.76632,
                   'r_wrist_flex_joint': -0.10001,
                   'r_wrist_roll_joint': 0.05106,
                   'l_shoulder_pan_joint': 1.9652,
                   'l_shoulder_lift_joint': - 0.26499,
                   'l_upper_arm_roll_joint': 1.3837,
                   'l_elbow_flex_joint': -2.12,
                   'l_forearm_roll_joint': 16.99,
                   'l_wrist_flex_joint': - 0.10001,
                   'l_wrist_roll_joint': 0,
                   'torso_lift_joint': 0.2,

                   'head_pan_joint': 0,
                   'head_tilt_joint': 0,
                   }

    def __init__(self):
        self.r_tip = 'r_gripper_tool_frame'
        self.l_tip = 'l_gripper_tool_frame'
        self.l_gripper_group = 'l_gripper'
        self.r_gripper_group = 'r_gripper'
        # self.r_gripper = rospy.ServiceProxy('r_gripper_simulator/set_joint_states', SetJointState)
        # self.l_gripper = rospy.ServiceProxy('l_gripper_simulator/set_joint_states', SetJointState)
        self.mujoco_reset = rospy.ServiceProxy('pr2/reset', Trigger)
        self.odom_root = 'odom_combined'
        super().__init__(PR2_Mujoco)

    def move_base(self, goal_pose):
        self.set_cart_goal(goal_pose, tip_link='base_footprint', root_link='odom_combined')
        self.plan_and_execute()

    def get_l_gripper_links(self):
        return [str(x) for x in self.world.groups[self.l_gripper_group].link_names_with_collisions]

    def get_r_gripper_links(self):
        return [str(x) for x in self.world.groups[self.r_gripper_group].link_names_with_collisions]

    def get_r_forearm_links(self):
        return ['r_wrist_flex_link', 'r_wrist_roll_link', 'r_forearm_roll_link', 'r_forearm_link',
                'r_forearm_link']

    def open_r_gripper(self):
        return
        sjs = SetJointStateRequest()
        sjs.state.name = ['r_gripper_l_finger_joint', 'r_gripper_r_finger_joint', 'r_gripper_l_finger_tip_joint',
                          'r_gripper_r_finger_tip_joint']
        sjs.state.position = [0.54, 0.54, 0.54, 0.54]
        sjs.state.velocity = [0, 0, 0, 0]
        sjs.state.effort = [0, 0, 0, 0]
        self.r_gripper.call(sjs)

    def close_r_gripper(self):
        return
        sjs = SetJointStateRequest()
        sjs.state.name = ['r_gripper_l_finger_joint', 'r_gripper_r_finger_joint', 'r_gripper_l_finger_tip_joint',
                          'r_gripper_r_finger_tip_joint']
        sjs.state.position = [0, 0, 0, 0]
        sjs.state.velocity = [0, 0, 0, 0]
        sjs.state.effort = [0, 0, 0, 0]
        self.r_gripper.call(sjs)

    def open_l_gripper(self):
        return
        sjs = SetJointStateRequest()
        sjs.state.name = ['l_gripper_l_finger_joint', 'l_gripper_r_finger_joint', 'l_gripper_l_finger_tip_joint',
                          'l_gripper_r_finger_tip_joint']
        sjs.state.position = [0.54, 0.54, 0.54, 0.54]
        sjs.state.velocity = [0, 0, 0, 0]
        sjs.state.effort = [0, 0, 0, 0]
        self.l_gripper.call(sjs)

    def close_l_gripper(self):
        return
        sjs = SetJointStateRequest()
        sjs.state.name = ['l_gripper_l_finger_joint', 'l_gripper_r_finger_joint', 'l_gripper_l_finger_tip_joint',
                          'l_gripper_r_finger_tip_joint']
        sjs.state.position = [0, 0, 0, 0]
        sjs.state.velocity = [0, 0, 0, 0]
        sjs.state.effort = [0, 0, 0, 0]
        self.l_gripper.call(sjs)

    def reset(self):
        self.mujoco_reset()
        self.open_l_gripper()
        self.open_r_gripper()
        self.clear_world()
        self.reset_base()
        self.register_group('l_gripper',
                            parent_group_name=self.god_map.unsafe_get_data(identifier.robot_group_name),
                            root_link_name='l_wrist_roll_link')
        self.register_group('r_gripper',
                            parent_group_name=self.god_map.unsafe_get_data(identifier.robot_group_name),
                            root_link_name='r_wrist_roll_link')


@pytest.fixture(scope='module')
def giskard(request, ros):
    c = PR2TestWrapper()
    request.addfinalizer(c.tear_down)
    return c


@pytest.fixture()
def pocky_pose_setup(resetted_giskard: PR2TestWrapper) -> PR2TestWrapper:
    resetted_giskard.set_joint_goal(pocky_pose)
    resetted_giskard.allow_all_collisions()
    resetted_giskard.plan_and_execute()
    return resetted_giskard


@pytest.fixture()
def box_setup(pocky_pose_setup: PR2TestWrapper) -> PR2TestWrapper:
    p = PoseStamped()
    p.header.frame_id = 'map'
    p.pose.position.x = 1.2
    p.pose.position.y = 0
    p.pose.position.z = 0.5
    p.pose.orientation.w = 1
    pocky_pose_setup.add_box(name='box', size=(1, 1, 1), pose=p)
    return pocky_pose_setup


@pytest.fixture()
def fake_table_setup(pocky_pose_setup: PR2TestWrapper) -> PR2TestWrapper:
    p = PoseStamped()
    p.header.frame_id = 'map'
    p.pose.position.x = 1.2
    p.pose.position.y = 0
    p.pose.position.z = 0.3
    p.pose.orientation.w = 1
    pocky_pose_setup.add_box(name='box', size=(1, 1, 1), pose=p)
    return pocky_pose_setup


class TestFk(object):
    def test_fk(self, zero_pose: PR2TestWrapper):
        for root, tip in itertools.product(zero_pose.robot.link_names, repeat=2):
            try:
                fk1 = zero_pose.god_map.get_data(fk_pose + [(root, tip)])
            except Exception as e:
                fk1 = zero_pose.god_map.get_data(fk_pose + [(root, tip)])
                pass
            fk2 = tf.lookup_pose(str(root), str(tip))
            compare_poses(fk1.pose, fk2.pose)

    def test_fk_attached(self, zero_pose: PR2TestWrapper):
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

    def test_fk_world(self, kitchen_setup: PR2TestWrapper):
        kitchen: SubWorldTree = kitchen_setup.world.groups['kitchen']
        robot: SubWorldTree = kitchen_setup.robot
        kitchen_links = list(kitchen.link_names)
        robot_links = list(robot.link_names)
        for i in range(25):
            if i % 2 == 0:
                root = kitchen_links[i]
                tip = robot_links[i]
            else:
                tip = kitchen_links[i]
                root = robot_links[i]
            fk1 = kitchen_setup.god_map.get_data(fk_pose + [(root, tip)])
            if i % 2 == 0:
                root = f'iai_kitchen/{root}'
            else:
                tip = f'iai_kitchen/{tip}'
            fk2 = tf.lookup_pose(str(root), str(tip))
            print(f'{root} {tip}')
            try:
                compare_poses(fk1.pose, fk2.pose)
            except Exception as e:
                pass
                raise


class TestJointGoals(object):
    def test_joint_movement1(self, zero_pose: PR2TestWrapper):
        zero_pose.allow_all_collisions()
        zero_pose.set_joint_goal(pocky_pose)
        zero_pose.plan_and_execute()

    def test_partial_joint_state_goal1(self, zero_pose: PR2TestWrapper):
        zero_pose.allow_self_collision()
        js = dict(list(pocky_pose.items())[:3])
        zero_pose.set_joint_goal(js)
        zero_pose.plan_and_execute()

    def test_continuous_joint1(self, zero_pose: PR2TestWrapper):
        zero_pose.allow_self_collision()
        # zero_pose.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
        js = {'r_wrist_roll_joint': -pi,
              'l_wrist_roll_joint': -2.1 * pi, }
        zero_pose.set_joint_goal(js)
        zero_pose.plan_and_execute()

    def test_prismatic_joint1(self, zero_pose: PR2TestWrapper):
        zero_pose.allow_self_collision()
        js = {'torso_lift_joint': 0.1}
        zero_pose.set_joint_goal(js)
        zero_pose.plan_and_execute()

    def test_hard_joint_limits(self, zero_pose: PR2TestWrapper):
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

    def test_SetPredictionHorizon(self, zero_pose: PR2TestWrapper):
        zero_pose.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
        zero_pose.set_joint_goal(zero_pose.better_pose)
        zero_pose.plan_and_execute()
        zero_pose.set_joint_goal(zero_pose.default_pose)
        zero_pose.plan_and_execute()

    def test_JointPositionRange(self, zero_pose: PR2TestWrapper):
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

    def test_CollisionAvoidanceHint(self, kitchen_setup: PR2TestWrapper):
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

    def test_CartesianPosition(self, zero_pose: PR2TestWrapper):
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

    def test_CartesianPose(self, zero_pose: PR2TestWrapper):
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

    def test_JointPositionRevolute(self, zero_pose: PR2TestWrapper):
        joint = 'r_shoulder_lift_joint'
        joint_goal = 1
        zero_pose.allow_all_collisions()
        zero_pose.set_json_goal('JointPositionRevolute',
                                joint_name=joint,
                                goal=joint_goal,
                                max_velocity=0.5)
        zero_pose.plan_and_execute()
        np.testing.assert_almost_equal(zero_pose.robot.state[joint].position, joint_goal, decimal=3)

    def test_JointVelocityRevolute(self, zero_pose: PR2TestWrapper):
        joint = 'r_shoulder_lift_joint'
        joint_goal = 1
        zero_pose.allow_all_collisions()
        zero_pose.set_json_goal('JointVelocityRevolute',
                                joint_name=joint,
                                max_velocity=0.5,
                                hard=True)
        zero_pose.set_json_goal('JointPositionRevolute',
                                joint_name=joint,
                                goal=joint_goal,
                                max_velocity=0.5)
        zero_pose.plan_and_execute()
        np.testing.assert_almost_equal(zero_pose.robot.state[joint].position, joint_goal, decimal=3)

    def test_JointPositionContinuous(self, zero_pose: PR2TestWrapper):
        joint = 'odom_z_joint'
        joint_goal = 4
        zero_pose.allow_all_collisions()
        zero_pose.set_json_goal('JointPositionContinuous',
                                joint_name=joint,
                                goal=joint_goal,
                                max_velocity=1)
        zero_pose.plan_and_execute()
        np.testing.assert_almost_equal(zero_pose.robot.state[joint].position, -2.283, decimal=2)

    def test_JointPosition_kitchen(self, kitchen_setup: PR2TestWrapper):
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

    def test_CartesianOrientation(self, zero_pose: PR2TestWrapper):
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

    def test_CartesianPoseStraight1(self, zero_pose: PR2TestWrapper):
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

    def test_CartesianPoseStraight2(self, better_pose: PR2TestWrapper):
        better_pose.close_l_gripper()
        goal_position = PoseStamped()
        goal_position.header.frame_id = 'base_link'
        goal_position.pose.position.x = 0.8
        goal_position.pose.position.y = 0.5
        goal_position.pose.position.z = 1
        goal_position.pose.orientation.w = 1

        start_pose = tf.lookup_pose('map', better_pose.l_tip)
        map_T_goal_position = tf.transform_pose('map', goal_position)

        object_pose = PoseStamped()
        object_pose.header.frame_id = 'map'
        object_pose.pose.position.x = (start_pose.pose.position.x + map_T_goal_position.pose.position.x) / 2.
        object_pose.pose.position.y = (start_pose.pose.position.y + map_T_goal_position.pose.position.y) / 2.
        object_pose.pose.position.z = (start_pose.pose.position.z + map_T_goal_position.pose.position.z) / 2.
        object_pose.pose.position.z += 0.08
        object_pose.pose.orientation.w = 1

        better_pose.add_sphere('sphere', 0.05, pose=object_pose)

        publish_marker_vector(start_pose.pose.position, map_T_goal_position.pose.position)

        goal = deepcopy(object_pose)
        goal.pose.position.x -= 0.1
        goal.pose.position.y += 0.4
        better_pose.set_straight_cart_goal(goal, better_pose.l_tip)
        better_pose.plan_and_execute()

        goal = deepcopy(object_pose)
        goal.pose.position.z -= 0.4
        better_pose.set_straight_cart_goal(goal, better_pose.l_tip)
        better_pose.plan_and_execute()

        goal = deepcopy(object_pose)
        goal.pose.position.y -= 0.4
        goal.pose.position.x -= 0.2
        better_pose.set_straight_cart_goal(goal, better_pose.l_tip)
        better_pose.plan_and_execute()

        goal = deepcopy(object_pose)
        goal.pose.position.x -= 0.4
        better_pose.set_straight_cart_goal(goal, better_pose.l_tip)
        better_pose.plan_and_execute()

    def test_CartesianVelocityLimit(self, zero_pose: PR2TestWrapper):
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

    def test_AvoidJointLimits1(self, zero_pose: PR2TestWrapper):
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

    def test_AvoidJointLimits2(self, zero_pose: PR2TestWrapper):
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

    def test_OverwriteWeights1(self, pocky_pose_setup: PR2TestWrapper):
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

    def test_pointing(self, kitchen_setup: PR2TestWrapper):
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

    def test_open_fridge(self, kitchen_setup: PR2TestWrapper):
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

    def test_open_drawer(self, kitchen_setup: PR2TestWrapper):
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

    def test_open_close_dishwasher(self, kitchen_setup: PR2TestWrapper):
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

    def test_align_planes1(self, zero_pose: PR2TestWrapper):
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

    def test_wrong_constraint_type(self, zero_pose: PR2TestWrapper):
        goal_state = JointState()
        goal_state.name = ['r_elbow_flex_joint']
        goal_state.position = [-1.0]
        kwargs = {'goal_state': goal_state}
        zero_pose.set_json_goal('jointpos', **kwargs)
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.UNKNOWN_CONSTRAINT])

    def test_python_code_in_constraint_type(self, zero_pose: PR2TestWrapper):
        goal_state = JointState()
        goal_state.name = ['r_elbow_flex_joint']
        goal_state.position = [-1.0]
        kwargs = {'goal_state': goal_state}
        zero_pose.set_json_goal('print("asd")', **kwargs)
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.UNKNOWN_CONSTRAINT])

    def test_wrong_params1(self, zero_pose: PR2TestWrapper):
        goal_state = JointState()
        goal_state.name = 'r_elbow_flex_joint'
        goal_state.position = [-1.0]
        kwargs = {'goal_state': goal_state}
        zero_pose.set_json_goal('JointPositionList', **kwargs)
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.CONSTRAINT_INITIALIZATION_ERROR])

    def test_wrong_params2(self, zero_pose: PR2TestWrapper):
        goal_state = JointState()
        goal_state.name = [5432]
        goal_state.position = 'test'
        kwargs = {'goal_state': goal_state}
        zero_pose.set_json_goal('JointPositionList', **kwargs)
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.CONSTRAINT_INITIALIZATION_ERROR])

    def test_align_planes2(self, zero_pose: PR2TestWrapper):
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

    def test_align_planes3(self, zero_pose: PR2TestWrapper):
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

    def test_align_planes4(self, kitchen_setup: PR2TestWrapper):
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

    def test_grasp_fridge_handle(self, kitchen_setup: PR2TestWrapper):
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

    def test_close_fridge_with_elbow(self, kitchen_setup: PR2TestWrapper):
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

    def test_open_close_oven(self, kitchen_setup: PR2TestWrapper):
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

    def test_grasp_dishwasher_handle(self, kitchen_setup: PR2TestWrapper):
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


class TestCartGoals(object):

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
            for i, point in enumerate(tj):
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
            kitchen_setup.allow_all_collisions()
            kitchen_setup.set_cart_goal(base_pose,
                                        'base_footprint',
                                        linear_velocity=0.5,
                                        angular_velocity=0.5)
            kitchen_setup.plan_and_execute()

            base_pose = PoseStamped()
            base_pose.header.frame_id = 'map'
            base_pose.pose.position.x = -2.0
            base_pose.pose.position.y = 2.0
            base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
            goal_c = base_pose

            # kitchen_setup.set_json_goal(u'SetPredictionHorizon', prediction_horizon=1)
            # kitchen_setup.allow_all_collisions()
            kitchen_setup.set_json_goal(u'CartesianPathCarrot',
                                        root_link=kitchen_setup.default_root,
                                        tip_link=tip_link,
                                        goals=poses,
                                        goal=goal_c,
                                        predict_f=10.0)
            try:
                kitchen_setup.plan_and_execute()
            except Exception:
                pass
        # kitchen_setup.send_goal()
        # kitchen_setup.check_cart_goal(tip_link, goal_c)
        # zero_pose.check_cart_goal(zero_pose.l_tip, l_goal)

    def test_move_base(self, zero_pose: PR2TestWrapper):
        map_T_odom = PoseStamped()
        map_T_odom.pose.position.x = 1
        map_T_odom.pose.position.y = 1
        map_T_odom.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 3, [0, 0, 1]))
        zero_pose.set_localization(map_T_odom)

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 1
        base_goal.pose.position.y = -1
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(-pi / 4, [0, 0, 1]))
        zero_pose.set_cart_goal(base_goal, 'base_footprint')
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

    def test_move_base1(self, zero_pose: PR2TestWrapper):
        map_T_odom = PoseStamped()
        map_T_odom.header.frame_id = 'map'
        map_T_odom.pose.position.y = 2
        map_T_odom.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 3, [0, 0, 1]))
        zero_pose.teleport_base(map_T_odom)

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 1
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        zero_pose.set_straight_cart_goal(base_goal, 'base_footprint')
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

    def test_move_base2(self, zero_pose: PR2TestWrapper):
        map_T_odom = PoseStamped()
        map_T_odom.pose.position.x = 1
        map_T_odom.pose.position.y = 1
        map_T_odom.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 3, [0, 0, 1]))
        zero_pose.set_localization(map_T_odom)

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = -1
        base_goal.pose.position.y = -1
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        zero_pose.set_straight_cart_goal(base_goal, 'base_footprint')
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

    def test_move_base3(self, zero_pose: PR2TestWrapper):
        map_T_odom = PoseStamped()
        map_T_odom.pose.position.x = 1
        map_T_odom.pose.position.y = 1
        map_T_odom.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 3, [0, 0, 1]))
        zero_pose.set_localization(map_T_odom)

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = -1
        base_goal.pose.position.y = -1
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 3, [0, 0, 1]))
        zero_pose.set_straight_cart_goal(base_goal, 'base_footprint')
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

    def test_move_base4(self, zero_pose: PR2TestWrapper):
        map_T_odom = PoseStamped()
        map_T_odom.header.frame_id = 'map'
        map_T_odom.pose.position.x = 2
        map_T_odom.pose.position.y = 0
        map_T_odom.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 3, [0, 0, 1]))
        zero_pose.teleport_base(map_T_odom)

        base_goal = PointStamped()
        base_goal.header.frame_id = 'map'
        base_goal.point.x = -1
        base_goal.point.y = 2
        zero_pose.set_json_goal('CartesianPositionStraight',
                                root_link=zero_pose.default_root,
                                tip_link='base_footprint',
                                goal_point=base_goal)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

    def test_base_driving1a(self, zero_pose):
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.orientation = Quaternion(*quaternion_about_axis(1 / 2, [0, 0, 1]))
        zero_pose.set_straight_cart_goal(p, 'base_footprint')
        zero_pose.plan_and_execute()

        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position.x = -2
        p.pose.position.y = 5
        p.pose.orientation = Quaternion(*quaternion_about_axis(-1, [0, 0, 1]))
        # zero_pose.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
        zero_pose.allow_collision()
        zero_pose.set_straight_cart_goal(p, 'base_footprint')
        zero_pose.plan_and_execute()

    def test_base_driving1b(self, zero_pose):
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.orientation = Quaternion(*quaternion_about_axis(1 / 2, [0, 0, 1]))
        zero_pose.set_straight_cart_goal(p, 'base_footprint')
        zero_pose.plan_and_execute()

        p = PoseStamped()
        p.header.frame_id = 'base_footprint'
        p.pose.position.x = -2
        p.pose.position.y = 5
        p.pose.orientation = Quaternion(*quaternion_about_axis(-1, [0, 0, 1]))
        # zero_pose.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
        zero_pose.allow_collision()
        zero_pose.set_straight_cart_goal(p, 'base_footprint')
        zero_pose.plan_and_execute()

    def test_base_driving2(self, zero_pose):
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.orientation.w = 1
        zero_pose.set_straight_cart_goal(p, 'base_footprint')
        zero_pose.plan_and_execute()

        p = PoseStamped()
        p.header.frame_id = 'base_footprint'
        p.pose.position.x = -2
        p.pose.position.y = 5
        p.pose.orientation.w = 1
        # zero_pose.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
        zero_pose.allow_collision()
        zero_pose.set_straight_cart_goal(p, 'base_footprint')
        zero_pose.plan_and_execute()

    def test_base_driving3(self, zero_pose: PR2TestWrapper):
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.orientation.w = 1
        zero_pose.set_straight_cart_goal(p, 'base_footprint')
        zero_pose.plan_and_execute()

        base_goal = PointStamped()
        base_goal.header.frame_id = 'map'
        base_goal.point.x = -2
        base_goal.point.y = 5
        zero_pose.set_json_goal('CartesianPositionStraight',
                                root_link=zero_pose.default_root,
                                tip_link='base_footprint',
                                goal_point=base_goal)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

    def test_rotate_gripper(self, zero_pose: PR2TestWrapper):
        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.r_tip
        r_goal.pose.orientation = Quaternion(*quaternion_about_axis(pi, [1, 0, 0]))
        zero_pose.set_cart_goal(r_goal, zero_pose.r_tip)
        zero_pose.plan_and_execute()

    def test_keep_position1(self, zero_pose: PR2TestWrapper):
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

    def test_keep_position2(self, zero_pose: PR2TestWrapper):
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

    def test_keep_position3(self, zero_pose: PR2TestWrapper):
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

    def test_cart_goal_1eef(self, zero_pose: PR2TestWrapper):
        p = PoseStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = zero_pose.r_tip
        p.pose.position = Point(-0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.allow_self_collision()
        zero_pose.set_cart_goal(p, zero_pose.r_tip, 'base_footprint')
        zero_pose.plan_and_execute()

    def test_cart_goal_1eef2(self, zero_pose: PR2TestWrapper):
        # zero_pose.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
        p = PoseStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = 'base_footprint'
        p.pose.position = Point(0.599, -0.009, 0.983)
        p.pose.orientation = Quaternion(0.524, -0.495, 0.487, -0.494)
        zero_pose.allow_all_collisions()
        zero_pose.set_cart_goal(p, zero_pose.l_tip, 'torso_lift_link')
        zero_pose.plan_and_execute()

    def test_cart_goal_1eef3(self, zero_pose: PR2TestWrapper):
        self.test_cart_goal_1eef(zero_pose)
        self.test_cart_goal_1eef2(zero_pose)

    def test_cart_goal_1eef4(self, zero_pose: PR2TestWrapper):
        p = PoseStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = 'map'
        p.pose.position = Point(2., 0, 1.)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.allow_all_collisions()
        zero_pose.set_cart_goal(p, zero_pose.r_tip, zero_pose.default_root)
        zero_pose.plan_and_execute()

    def test_cart_goal_orientation_singularity(self, zero_pose: PR2TestWrapper):
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

    def test_cart_goal_2eef2(self, zero_pose: PR2TestWrapper):
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

    def test_cart_goal_left_right_chain(self, zero_pose: PR2TestWrapper):
        r_goal = tf.lookup_pose(zero_pose.l_tip, zero_pose.r_tip)
        r_goal.pose.position.x -= 0.1
        zero_pose.allow_all_collisions()
        zero_pose.set_cart_goal(r_goal, zero_pose.r_tip, zero_pose.l_tip)
        zero_pose.plan_and_execute()

    def test_wiggle1(self, kitchen_setup: PR2TestWrapper):
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

    def test_wiggle2(self, zero_pose: PR2TestWrapper):
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

    def test_wiggle3(self, zero_pose: PR2TestWrapper):
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

    def test_root_link_not_equal_chain_root(self, zero_pose: PR2TestWrapper):
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
    def test_interrupt1(self, zero_pose: PR2TestWrapper):
        p = PoseStamped()
        p.header.frame_id = 'base_footprint'
        p.pose.position = Point(2, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_cart_goal(p, 'base_footprint')
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.PREEMPTED], stop_after=1)

    def test_interrupt2(self, zero_pose: PR2TestWrapper):
        p = PoseStamped()
        p.header.frame_id = 'base_footprint'
        p.pose.position = Point(2, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_cart_goal(p, 'base_footprint')
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.PREEMPTED], stop_after=6)

    def test_undefined_type(self, zero_pose: PR2TestWrapper):
        zero_pose.allow_all_collisions()
        zero_pose.send_goal(goal_type=MoveGoal.UNDEFINED,
                            expected_error_codes=[MoveResult.INVALID_GOAL])

    def test_empty_goal(self, zero_pose: PR2TestWrapper):
        zero_pose.cmd_seq = []
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.INVALID_GOAL])

    def test_plan_only(self, zero_pose: PR2TestWrapper):
        zero_pose.allow_self_collision()
        zero_pose.set_joint_goal(pocky_pose, check=False)
        zero_pose.add_goal_check(JointGoalChecker(zero_pose.god_map, zero_pose.default_pose))
        zero_pose.send_goal(goal_type=MoveGoal.PLAN_ONLY)


class TestWayPoints(object):
    def test_interrupt_way_points1(self, zero_pose: PR2TestWrapper):
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

    def test_waypoints2(self, zero_pose: PR2TestWrapper):
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

    def test_waypoints_with_fail(self, zero_pose: PR2TestWrapper):
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

    def test_waypoints_with_fail1(self, zero_pose: PR2TestWrapper):
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

    def test_waypoints_with_fail2(self, zero_pose: PR2TestWrapper):
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

    def test_waypoints_with_fail3(self, zero_pose: PR2TestWrapper):
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

    def test_skip_failures1(self, zero_pose: PR2TestWrapper):
        zero_pose.set_json_goal('muh')
        zero_pose.send_goal(expected_error_codes=[MoveResult.UNKNOWN_CONSTRAINT, ],
                            goal_type=MoveGoal.PLAN_AND_EXECUTE_AND_SKIP_FAILURES)

    def test_skip_failures2(self, zero_pose: PR2TestWrapper):
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


class TestShaking(object):
    def test_wiggle_prismatic_joint_neglectable_shaking(self, kitchen_setup: PR2TestWrapper):
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

    def test_wiggle_revolute_joint_neglectable_shaking(self, kitchen_setup: PR2TestWrapper):
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

    def test_wiggle_continuous_joint_neglectable_shaking(self, kitchen_setup: PR2TestWrapper):
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

    def test_wiggle_revolute_joint_shaking(self, kitchen_setup: PR2TestWrapper):
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

    def test_wiggle_prismatic_joint_shaking(self, kitchen_setup: PR2TestWrapper):
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

    def test_wiggle_continuous_joint_shaking(self, kitchen_setup: PR2TestWrapper):
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

    def test_only_revolute_joint_shaking(self, kitchen_setup: PR2TestWrapper):
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

    def test_only_revolute_joint_neglectable_shaking(self, kitchen_setup: PR2TestWrapper):
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

    def test_dye_group(self, kitchen_setup: PR2TestWrapper):
        kitchen_setup.dye_group(kitchen_setup.get_robot_name(), (1, 0, 0, 1))
        kitchen_setup.dye_group('kitchen', (0, 1, 0, 1))
        kitchen_setup.dye_group(kitchen_setup.r_gripper_group, (0, 0, 1, 1))
        kitchen_setup.set_joint_goal(kitchen_setup.default_pose)
        kitchen_setup.plan_and_execute()

    def test_clear_world(self, zero_pose: PR2TestWrapper):
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

    def test_attach_remove_box(self, better_pose: PR2TestWrapper):
        pocky = 'http://muh#pocky'
        p = PoseStamped()
        p.header.frame_id = better_pose.r_tip
        p.pose.orientation.w = 1
        better_pose.add_box(pocky, size=(1, 1, 1), pose=p)
        for i in range(3):
            better_pose.update_parent_link_of_group(name=pocky, parent_link=better_pose.r_tip)
            better_pose.detach_group(pocky)
        better_pose.remove_group(pocky)

    def test_reattach_box(self, zero_pose: PR2TestWrapper):
        pocky = 'http://muh#pocky'
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.position = Point(0.05, 0, 0)
        p.pose.orientation = Quaternion(0., 0., 0.47942554, 0.87758256)
        zero_pose.add_box(pocky, (0.1, 0.02, 0.02), pose=p)
        zero_pose.update_parent_link_of_group(pocky, parent_link=zero_pose.r_tip)
        relative_pose = zero_pose.robot.compute_fk_pose(zero_pose.r_tip, pocky).pose
        compare_poses(p.pose, relative_pose)

    def test_add_box_twice(self, zero_pose: PR2TestWrapper):
        object_name = 'muh'
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position = Point(1.2, 0, 1.6)
        p.pose.orientation = Quaternion(0.0, 0.0, 0.47942554, 0.87758256)
        zero_pose.add_box(object_name, size=(1, 1, 1), pose=p)
        zero_pose.add_box(object_name, size=(1, 1, 1), pose=p,
                          expected_error_code=UpdateWorldResponse.DUPLICATE_GROUP_ERROR)

    def test_add_remove_sphere(self, zero_pose: PR2TestWrapper):
        object_name = 'muh'
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position.x = 1.2
        p.pose.position.y = 0
        p.pose.position.z = 1.6
        p.pose.orientation.w = 1
        zero_pose.add_sphere(object_name, radius=1, pose=p)
        zero_pose.remove_group(object_name)

    def test_add_remove_cylinder(self, zero_pose: PR2TestWrapper):
        object_name = 'muh'
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position.x = 0.5
        p.pose.position.y = 0
        p.pose.position.z = 0
        p.pose.orientation.w = 1
        zero_pose.add_cylinder(object_name, height=1, radius=1, pose=p)
        zero_pose.remove_group(object_name)

    def test_add_urdf_body(self, kitchen_setup: PR2TestWrapper):
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

    def test_add_mesh(self, zero_pose: PR2TestWrapper):
        object_name = 'muh'
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.add_mesh(object_name, mesh='package://giskardpy/test/urdfs/meshes/bowl_21.obj', pose=p)

    def test_add_non_existing_mesh(self, zero_pose: PR2TestWrapper):
        object_name = 'muh'
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.add_mesh(object_name, mesh='package://giskardpy/test/urdfs/meshes/muh.obj', pose=p,
                           expected_error_code=UpdateWorldResponse.CORRUPT_MESH_ERROR)

    def test_add_attach_detach_remove_add(self, zero_pose: PR2TestWrapper):
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

    def test_attach_to_kitchen(self, kitchen_setup: PR2TestWrapper):
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

    def test_update_group_pose1(self, zero_pose: PR2TestWrapper):
        group_name = 'muh'
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position = Point(1.2, 0, 1.6)
        p.pose.orientation = Quaternion(0.0, 0.0, 0.47942554, 0.87758256)
        zero_pose.add_box(group_name, size=(1, 1, 1), pose=p)
        p.pose.position = Point(1, 0, 0)
        zero_pose.update_group_pose('asdf', p, expected_error_code=UpdateWorldResponse.UNKNOWN_GROUP_ERROR)
        zero_pose.update_group_pose(group_name, p)

    def test_update_group_pose2(self, zero_pose: PR2TestWrapper):
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

    def test_attach_existing_box2(self, zero_pose: PR2TestWrapper):
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

    def test_attach_to_nonexistant_robot_link(self, zero_pose: PR2TestWrapper):
        pocky = 'http://muh#pocky'
        p = PoseStamped()
        zero_pose.add_box(name=pocky,
                          size=(0.1, 0.02, 0.02),
                          pose=p,
                          parent_link='muh',
                          expected_error_code=UpdateWorldResponse.UNKNOWN_LINK_ERROR)

    def test_reattach_unknown_object(self, zero_pose: PR2TestWrapper):
        zero_pose.update_parent_link_of_group('muh',
                                              parent_link='',
                                              parent_link_group='',
                                              expected_response=UpdateWorldResponse.UNKNOWN_GROUP_ERROR)

    def test_add_remove_box(self, zero_pose: PR2TestWrapper):
        object_name = 'muh'
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position.x = 1.2
        p.pose.position.y = 0
        p.pose.position.z = 1.6
        p.pose.orientation.w = 1
        zero_pose.add_box(object_name, size=(1, 1, 1), pose=p)
        zero_pose.remove_group(object_name)

    def test_invalid_update_world(self, zero_pose: PR2TestWrapper):
        req = UpdateWorldRequest()
        req.timeout = 500
        req.body = WorldBody()
        req.pose = PoseStamped()
        req.parent_link = zero_pose.r_tip
        req.operation = 42
        assert zero_pose._update_world_srv.call(req).error_codes == UpdateWorldResponse.INVALID_OPERATION

    def test_remove_unkown_group(self, zero_pose: PR2TestWrapper):
        zero_pose.remove_group('muh', expected_response=UpdateWorldResponse.UNKNOWN_GROUP_ERROR)

    def test_corrupt_shape_error(self, zero_pose: PR2TestWrapper):
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

    def test_tf_error(self, zero_pose: PR2TestWrapper):
        req = UpdateWorldRequest()
        req.body = WorldBody(type=WorldBody.PRIMITIVE_BODY,
                             shape=SolidPrimitive(type=1))
        req.pose = PoseStamped()
        req.parent_link = 'base_link'
        req.operation = UpdateWorldRequest.ADD
        assert zero_pose._update_world_srv.call(req).error_codes == UpdateWorldResponse.TF_ERROR

    def test_unsupported_options(self, kitchen_setup: PR2TestWrapper):
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


class TestCollisionAvoidanceGoals:

    def test_cram_reset(self, kitchen_setup: PR2TestWrapper):
        js = {
            'torso_lift_joint': 0.011505660600960255,
            'r_upper_arm_roll_joint': 1.4073297904815263e-07,
            'r_shoulder_pan_joint': 1.6493133898620727e-06,
            'r_shoulder_lift_joint': 1.1900528988917358e-05,
            'r_forearm_roll_joint': -1.299561915857339e-07,
            'r_elbow_flex_joint': -0.14999248087406158,
            'r_wrist_flex_joint': -0.10000340640544891,
            'r_wrist_roll_joint': -2.0980905901524238e-10,
            'l_upper_arm_roll_joint': 1.763919925679147e-07,
            'l_shoulder_pan_joint': -1.997101435335935e-07,
            'l_shoulder_lift_joint': 1.5421014722960535e-06,
            'l_forearm_roll_joint': -2.038720481323253e-08,
            'l_elbow_flex_joint': -0.14998672902584076,
            'l_wrist_flex_joint': -0.1000063493847847,
            'l_wrist_roll_joint': 6.953541742404923e-08,
            'head_pan_joint': -5.877892590433476e-07,
            'head_tilt_joint': 5.255556243355386e-05,
        }
        kitchen_setup.set_joint_goal(js)
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_limit_cartesian_velocity_goal(root_link='odom_combined',
                                                        tip_link='l_wrist_roll_link',
                                                        max_linear_velocity=0.1,
                                                        max_angular_velocity=0.5,
                                                        hard=False)
        kitchen_setup.set_limit_cartesian_velocity_goal(root_link='odom_combined',
                                                        tip_link='r_wrist_roll_link',
                                                        max_linear_velocity=0.1,
                                                        max_angular_velocity=0.5,
                                                        hard=False)
        cart_goal = PoseStamped()
        cart_goal.header.frame_id = 'base_footprint'
        cart_goal.pose.orientation.w = 1
        kitchen_setup.set_cart_goal(goal_pose=cart_goal,
                                    tip_link='base_footprint',
                                    root_link='odom_combined',
                                    linear_velocity=0.4,
                                    weight=2500)
        js1 = {'l_shoulder_pan_joint': 1.9652919379395388,
               'l_shoulder_lift_joint': -0.26499816732737785,
               'l_upper_arm_roll_joint': 1.3837617139225473,
               'l_elbow_flex_joint': -2.1224566064321584,
               'l_forearm_roll_joint': 16.99646118944817,
               'l_wrist_flex_joint': -0.07350789589924167,
               'l_wrist_roll_joint': 0.0}
        kitchen_setup.set_joint_goal(js1)
        js2 = {
            'r_shoulder_pan_joint': -1.712587449591307,
            'r_shoulder_lift_joint': -0.2567290370386635,
            'r_upper_arm_roll_joint': -1.4633501125737374,
            'r_elbow_flex_joint': -2.1221670650093913,
            'r_forearm_roll_joint': 1.7663253481913623,
            'r_wrist_flex_joint': -0.07942669250968948,
            'r_wrist_roll_joint': 0.05106258161229582
        }
        kitchen_setup.set_joint_goal(js2)
        kitchen_setup.avoid_all_collisions(0.1)
        kitchen_setup.plan_and_execute()

    def test_handover(self, kitchen_setup: PR2TestWrapper):
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

    def test_only_collision_avoidance(self, zero_pose: PR2TestWrapper):
        zero_pose.plan_and_execute()

    def test_mesh_collision_avoidance(self, zero_pose: PR2TestWrapper):
        zero_pose.close_r_gripper()
        object_name = 'muh'
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.position = Point(0.01, 0, 0)
        p.pose.orientation = Quaternion(*quaternion_about_axis(-np.pi / 2, [0, 1, 0]))
        zero_pose.add_mesh(object_name, mesh='package://giskardpy/test/urdfs/meshes/bowl_21.obj', pose=p)
        zero_pose.plan_and_execute()

    def test_attach_box_as_eef(self, zero_pose: PR2TestWrapper):
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

    def test_infeasible(self, kitchen_setup: PR2TestWrapper):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position = Point(2, 0, 0)
        pose.pose.orientation = Quaternion(w=1)
        kitchen_setup.teleport_base(pose)
        kitchen_setup.plan_and_execute(expected_error_codes=[MoveResult.HARD_CONSTRAINTS_VIOLATED])

    def test_unknown_group1(self, box_setup: PR2TestWrapper):
        box_setup.avoid_collision(min_distance=0.05, group1='muh')
        box_setup.plan_and_execute([MoveResult.UNKNOWN_GROUP])

    def test_unknown_group2(self, box_setup: PR2TestWrapper):
        box_setup.avoid_collision(group2='muh')
        box_setup.plan_and_execute([MoveResult.UNKNOWN_GROUP])

    def test_base_link_in_collision(self, zero_pose: PR2TestWrapper):
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

    def test_allow_self_collision(self, zero_pose: PR2TestWrapper):
        zero_pose.check_cpi_geq(zero_pose.get_r_gripper_links(), 0.05)
        zero_pose.check_cpi_geq(zero_pose.get_l_gripper_links(), 0.05)

    def test_allow_self_collision2(self, zero_pose: PR2TestWrapper):
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

    def test_allow_self_collision3(self, zero_pose: PR2TestWrapper):
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

    def test_avoid_self_collision(self, zero_pose: PR2TestWrapper):
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

    def test_avoid_self_collision2(self, zero_pose: PR2TestWrapper):
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

    def test_avoid_self_collision3(self, zero_pose: PR2TestWrapper):
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

    def test_get_out_of_self_collision(self, zero_pose: PR2TestWrapper):
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

    def test_avoid_collision(self, box_setup: PR2TestWrapper):
        box_setup.avoid_collision(min_distance=0.05, group1=box_setup.get_robot_name())
        box_setup.avoid_collision(min_distance=0.15, group1=box_setup.l_gripper_group, group2='box')
        box_setup.avoid_collision(min_distance=0.10, group1=box_setup.r_gripper_group, group2='box')
        box_setup.allow_self_collision()
        box_setup.plan_and_execute()
        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.148)
        box_setup.check_cpi_geq(box_setup.get_r_gripper_links(), 0.088)

    def test_collision_override(self, box_setup: PR2TestWrapper):
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

    def test_avoid_collision2(self, fake_table_setup: PR2TestWrapper):
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

    def test_allow_collision(self, box_setup: PR2TestWrapper):
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

    def test_avoid_collision3(self, pocky_pose_setup: PR2TestWrapper):
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

    def test_avoid_collision4(self, pocky_pose_setup: PR2TestWrapper):
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

    def test_avoid_collision_two_sticks(self, pocky_pose_setup: PR2TestWrapper):
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

    def test_avoid_collision5_cut_off(self, pocky_pose_setup: PR2TestWrapper):
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

    # def test_avoid_collision6(self, fake_table_setup: PR2TestWrapper):
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

    def test_avoid_collision7(self, kitchen_setup: PR2TestWrapper):
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

    def test_avoid_collision_at_kitchen_corner(self, kitchen_setup: PR2TestWrapper):
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

    def test_avoid_collision8(self, kitchen_setup: PR2TestWrapper):
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

    def test_avoid_collision_drive_under_drawer(self, kitchen_setup: PR2TestWrapper):
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

    def test_avoid_collision_with_far_object(self, pocky_pose_setup: PR2TestWrapper):
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

    def test_avoid_collision_touch(self, box_setup: PR2TestWrapper):
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

    def test_get_out_of_collision(self, box_setup: PR2TestWrapper):
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

    def test_allow_collision_gripper(self, box_setup: PR2TestWrapper):
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

    def test_attached_get_below_soft_threshold(self, box_setup: PR2TestWrapper):
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

    def test_attached_get_out_of_collision_below(self, box_setup: PR2TestWrapper):
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

    def test_attached_get_out_of_collision_and_stay_in_hard_threshold(self, box_setup: PR2TestWrapper):
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

    def test_attached_get_out_of_collision_stay_in(self, box_setup: PR2TestWrapper):
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

    def test_attached_get_out_of_collision_passive(self, box_setup: PR2TestWrapper):
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

    def test_attached_collision2(self, box_setup: PR2TestWrapper):
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

    def test_attached_self_collision(self, zero_pose: PR2TestWrapper):
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

    def test_attached_self_collision2(self, zero_pose: PR2TestWrapper):
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

    def test_attached_self_collision3(self, zero_pose: PR2TestWrapper):
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

    def test_attached_collision_allow(self, box_setup: PR2TestWrapper):
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

    def test_avoid_collision_gripper(self, box_setup: PR2TestWrapper):
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

    # def test_end_state_collision(self, box_setup: PR2TestWrapper):
    #     # TODO endstate impossible as long as we check for path collision?
    #     pass

    # def test_filled_vel_values(self, box_setup: PR2TestWrapper):
    #     pass
    #
    # def test_undefined_goal(self, box_setup: PR2TestWrapper):
    #     pass

    # TODO test plan only

    def test_attached_two_items(self, zero_pose: PR2TestWrapper):
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

    # def test_pick_and_place(self, kitchen_setup: PR2TestWrapper):
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

    # def test_hand_in_kitchen(self, kitchen_setup: PR2TestWrapper):
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

    def test_set_kitchen_joint_state(self, kitchen_setup: PR2TestWrapper):
        kitchen_js = {'sink_area_left_upper_drawer_main_joint': 0.45}
        kitchen_setup.set_kitchen_js(kitchen_js)

    def test_ease_fridge(self, kitchen_setup: PR2TestWrapper):
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

    def test_ease_cereal(self, kitchen_setup: PR2TestWrapper):
        # FIXME
        cereal_name = 'cereal'
        drawer_frame_id = 'iai_kitchen/oven_area_area_right_drawer_board_3_link'

        # take milk out of fridge
        kitchen_setup.set_kitchen_js({'oven_area_area_right_drawer_main_joint': 0.48})

        # spawn milk

        cereal_pose = PoseStamped()
        cereal_pose.header.frame_id = drawer_frame_id
        cereal_pose.pose.position = Point(0.123, -0.03, 0.11)
        cereal_pose.pose.orientation = Quaternion(0.0087786, 0.005395, -0.838767, -0.544393)
        kitchen_setup.add_box(cereal_name, (0.1528, 0.0634, 0.22894), pose=cereal_pose)

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

        grasp_pose = tf.kdl_to_pose_stamped(drawer_T_box * box_T_r_goal_pre, drawer_frame_id)

        kitchen_setup.set_json_goal('AvoidJointLimits', percentage=40)
        kitchen_setup.set_cart_goal(grasp_pose, tip_link=kitchen_setup.r_tip)
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
        kitchen_setup.set_cart_goal(grasp_pose, cereal_name, kitchen_setup.default_root)
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_joint_goal(kitchen_setup.better_pose)
        kitchen_setup.plan_and_execute()

        # place milk back

        # kitchen_setup.add_json_goal('BasePointingForward')
        # milk_goal = PoseStamped()
        # milk_goal.header.frame_id = 'iai_kitchen/kitchen_island_surface'
        # milk_goal.pose.position = Point(.1, -.2, .13)
        # milk_goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0,0,1]))
        kitchen_setup.set_cart_goal(grasp_pose, cereal_name, kitchen_setup.default_root)
        kitchen_setup.plan_and_execute()

        kitchen_setup.set_cart_goal(cereal_pose, cereal_name, kitchen_setup.default_root)
        kitchen_setup.plan_and_execute()

        # kitchen_setup.keep_position(kitchen_setup.r_tip)
        kitchen_setup.open_l_gripper()

        kitchen_setup.detach_group(cereal_name)

        kitchen_setup.set_joint_goal(kitchen_setup.better_pose)
        kitchen_setup.plan_and_execute()

    def test_bowl_and_cup(self, kitchen_setup: PR2TestWrapper):
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
        kitchen_setup.move_base(base_goal)

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

    def test_ease_grasp_bowl(self, kitchen_setup: PR2TestWrapper):
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

    # def test_avoid_self_collision2(self, kitchen_setup: PR2TestWrapper):
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

    def test_ease_spoon(self, kitchen_setup: PR2TestWrapper):
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

    def test_ease_place_on_new_table(self, kitchen_setup: PR2TestWrapper):
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

    def test_tray(self, kitchen_setup: PR2TestWrapper):
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

    def test_iis(self, kitchen_setup: PR2TestWrapper):
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

    def test_ease_dishwasher(self, kitchen_setup: PR2TestWrapper):
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
        # kitchen_setup.set_kitchen_js({'sink_area_dish_washer_door_joint': 0})


class TestInfoServices(object):
    def test_get_object_info(self, zero_pose: PR2TestWrapper):
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
# pytest.main(['-s', __file__ + '::TestWayPoints::test_waypoints2'])
# pytest.main(['-s', __file__ + '::TestCartGoals::test_keep_position3'])
