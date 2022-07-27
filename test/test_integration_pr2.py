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
from tf.transformations import quaternion_from_matrix, quaternion_about_axis

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
        kitchen_setup.set_joint_goal(kitchen_setup.default_pose)
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
