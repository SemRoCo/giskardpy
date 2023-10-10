from __future__ import division

from itertools import combinations

import urdf_parser_py.urdf as up
from copy import deepcopy
from typing import Optional

import numpy as np
import pytest
import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Vector3Stamped, PointStamped, QuaternionStamped, Pose
from numpy import pi
from shape_msgs.msg import SolidPrimitive
from tf.transformations import quaternion_from_matrix, quaternion_about_axis

import giskardpy.utils.tfwrapper as tf
from giskard_msgs.msg import MoveResult, WorldBody, MoveGoal
from giskard_msgs.srv import UpdateWorldResponse, UpdateWorldRequest
from giskardpy import identifier
from giskardpy.configs.behavior_tree_config import StandAloneBTConfig
from giskardpy.configs.giskard import Giskard
from giskardpy.configs.iai_robots.pr2 import PR2CollisionAvoidance, PR2StandaloneInterface, WorldWithPR2Config
from giskardpy.configs.qp_controller_config import QPControllerConfig, SupportedQPSolver
from giskardpy.model.better_pybullet_syncer import BetterPyBulletSyncer
from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer
from giskardpy.model.utils import make_world_body_box, hacky_urdf_parser_fix
from giskardpy.model.world import WorldTree
from giskardpy.my_types import PrefixName
from giskardpy.goals.goal import WEIGHT_ABOVE_CA, WEIGHT_BELOW_CA, WEIGHT_COLLISION_AVOIDANCE
from giskardpy.python_interface import GiskardWrapper
from giskardpy.utils.utils import launch_launchfile, suppress_stderr, resolve_ros_iris
from giskardpy.utils.math import compare_points
from utils_for_tests import compare_poses, publish_marker_vector, \
    GiskardTestWrapper, pr2_urdf

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
              'l_gripper_l_finger_joint': 0.55,
              'r_gripper_l_finger_joint': 0.55
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
    'l_gripper_l_finger_joint': 0.55,
    'r_gripper_l_finger_joint': 0.55
}


class PR2TestWrapper(GiskardTestWrapper):
    default_pose = {
        'r_elbow_flex_joint': -0.15,
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
        'head_tilt_joint': 0,
        'l_gripper_l_finger_joint': 0.55,
        'r_gripper_l_finger_joint': 0.55
    }

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
                   'l_gripper_l_finger_joint': 0.55,
                   'r_gripper_l_finger_joint': 0.55,
                   'head_pan_joint': 0,
                   'head_tilt_joint': 0,
                   }

    def __init__(self, giskard: Optional[Giskard] = None):
        self.r_tip = 'r_gripper_tool_frame'
        self.l_tip = 'l_gripper_tool_frame'
        self.l_gripper_group = 'l_gripper'
        self.r_gripper_group = 'r_gripper'
        # self.r_gripper = rospy.ServiceProxy('r_gripper_simulator/set_joint_states', SetJointState)
        # self.l_gripper = rospy.ServiceProxy('l_gripper_simulator/set_joint_states', SetJointState)
        self.odom_root = 'odom_combined'
        drive_joint_name = 'brumbrum'
        if giskard is None:
            giskard = Giskard(world_config=WorldWithPR2Config(drive_joint_name=drive_joint_name),
                              robot_interface_config=PR2StandaloneInterface(drive_joint_name=drive_joint_name),
                              collision_avoidance_config=PR2CollisionAvoidance(drive_joint_name=drive_joint_name),
                              behavior_tree_config=StandAloneBTConfig())
        super().__init__(giskard)
        self.robot = self.world.groups[self.robot_name]

    def teleport_base(self, goal_pose, group_name: Optional[str] = None):
        self.set_seed_odometry(base_pose=goal_pose, group_name=group_name)
        self.allow_all_collisions()
        self.plan_and_execute()

    def move_base(self, goal_pose):
        # self.set_move_base_goal(goal_pose=goal_pose)
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

    def close_r_gripper(self):
        return

    def open_l_gripper(self):
        return

    def close_l_gripper(self):
        return

    def reset_base(self):
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.orientation.w = 1
        if self.is_standalone():
            self.teleport_base(p)
        else:
            self.move_base(p)

    def set_localization(self, map_T_odom: PoseStamped):
        map_T_odom.pose.position.z = 0
        self.set_seed_odometry(map_T_odom)
        self.plan_and_execute()
        # self.wait_heartbeats(15)
        # p2 = self.world.compute_fk_pose(self.world.root_link_name, self.odom_root)
        # compare_poses(p2.pose, map_T_odom.pose)

    def reset(self):
        self.clear_world()
        self.open_l_gripper()
        self.open_r_gripper()
        self.reset_base()
        self.register_group('l_gripper',
                            root_link_group_name=self.robot_name,
                            root_link_name='l_wrist_roll_link')
        self.register_group('r_gripper',
                            root_link_group_name=self.robot_name,
                            root_link_name='r_wrist_roll_link')

        # self.register_group('fl_l',
        #                     root_link_group_name=self.robot_name,
        #                     root_link_name='fl_caster_l_wheel_link')
        # self.dye_group('fl_l', rgba=(1, 0, 0, 1))
        #
        # self.register_group('fr_l',
        #                     root_link_group_name=self.robot_name,
        #                     root_link_name='fr_caster_l_wheel_link')
        # self.dye_group('fr_l', rgba=(1, 0, 0, 1))
        #
        # self.register_group('bl_l',
        #                     root_link_group_name=self.robot_name,
        #                     root_link_name='bl_caster_l_wheel_link')
        # self.dye_group('bl_l', rgba=(1, 0, 0, 1))
        #
        # self.register_group('br_l',
        #                     root_link_group_name=self.robot_name,
        #                     root_link_name='br_caster_l_wheel_link')
        # self.dye_group('br_l', rgba=(1, 0, 0, 1))


@pytest.fixture(scope='module')
def giskard(request, ros):
    launch_launchfile('package://iai_pr2_description/launch/upload_pr2_calibrated_with_ft2.launch')
    # launch_launchfile('package://iai_pr2_description/launch/upload_pr2_cableguide.launch')
    c = PR2TestWrapper()
    # c = PR2TestWrapperMujoco()
    request.addfinalizer(c.tear_down)
    return c


@pytest.fixture()
def pocky_pose_setup(resetted_giskard: PR2TestWrapper) -> PR2TestWrapper:
    if resetted_giskard.is_standalone():
        resetted_giskard.set_seed_configuration(pocky_pose)
        resetted_giskard.allow_all_collisions()
    else:
        resetted_giskard.allow_all_collisions()
        resetted_giskard.set_joint_goal(pocky_pose)
    resetted_giskard.plan_and_execute()
    return resetted_giskard


@pytest.fixture()
def world_setup(zero_pose: PR2TestWrapper) -> WorldTree:
    zero_pose.stop_ticking()
    return zero_pose.world


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


# class TestFk(object):
#     def test_fk(self, zero_pose: PR2TestWrapper):
#         for root, tip in itertools.product(zero_pose.robot().link_names, repeat=2):
#             try:
#                 fk1 = zero_pose.god_map.get_data(fk_pose + [(root, tip)])
#             except Exception as e:
#                 fk1 = zero_pose.god_map.get_data(fk_pose + [(root, tip)])
#                 pass
#             fk2 = tf.lookup_pose(str(root), str(tip))
#             compare_poses(fk1.pose, fk2.pose)
#
#     def test_fk_attached(self, zero_pose: PR2TestWrapper):
#         pocky = 'box'
#         p = PoseStamped()
#         p.header.frame_id = zero_pose.r_tip
#         p.pose.position.x = 0.05
#         p.pose.orientation.x = 1
#         zero_pose.add_box(pocky, size=(0.1, 0.02, 0.02), parent_link=zero_pose.r_tip, pose=p)
#         for root, tip in itertools.product(zero_pose.robot.link_names, [pocky]):
#             fk1 = zero_pose.god_map.get_data(fk_pose + [(root, tip)])
#             fk2 = tf.lookup_pose(str(root), str(tip))
#             compare_poses(fk1.pose, fk2.pose)
#
#     def test_fk_world(self, kitchen_setup: PR2TestWrapper):
#         kitchen: SubWorldTree = kitchen_setup.world.groups['kitchen']
#         robot: SubWorldTree = kitchen_setup.robot
#         kitchen_links = list(kitchen.link_names)
#         robot_links = list(robot.link_names)
#         for i in range(25):
#             if i % 2 == 0:
#                 root = kitchen_links[i]
#                 tip = robot_links[i]
#             else:
#                 tip = kitchen_links[i]
#                 root = robot_links[i]
#             fk1 = kitchen_setup.god_map.get_data(fk_pose + [(root, tip)])
#             if i % 2 == 0:
#                 root = f'iai_kitchen/{root}'
#             else:
#                 tip = f'iai_kitchen/{tip}'
#             fk2 = tf.lookup_pose(str(root), str(tip))
#             print(f'{root} {tip}')
#             try:
#                 compare_poses(fk1.pose, fk2.pose)
#             except Exception as e:
#                 pass
#                 raise


class TestJointGoals:
    def test_joint_goal(self, zero_pose: PR2TestWrapper):
        js = {
            'torso_lift_joint': 0.2999225173357618,
            'head_pan_joint': 0.041880780651479044,
            'head_tilt_joint': -0.37,
            'r_upper_arm_roll_joint': -0.9487714747527726,
            'r_shoulder_pan_joint': -1.0047307505973626,
            'r_shoulder_lift_joint': 0.48736790658811985,
            'r_forearm_roll_joint': -14.895833882874182,
            'r_elbow_flex_joint': -1.392377908925028,
            'r_wrist_flex_joint': -0.4548695149411013,
            'r_wrist_roll_joint': 0.11426798984097819,
            'l_upper_arm_roll_joint': 1.7383062350263658,
            'l_shoulder_pan_joint': 1.8799810286792007,
            'l_shoulder_lift_joint': 0.011627231224188975,
            'l_forearm_roll_joint': 312.67276414458695,
            'l_elbow_flex_joint': -2.0300928925694675,
            'l_wrist_flex_joint': -0.10014623223021513,
            'l_wrist_roll_joint': -6.062015047706399,
        }
        zero_pose.set_joint_goal(js)
        zero_pose.allow_all_collisions()
        zero_pose.set_json_goal('EnableVelocityTrajectoryTracking', enabled=True)
        zero_pose.plan_and_execute()

    def test_gripper_goal(self, zero_pose: PR2TestWrapper):
        js = {
            'r_gripper_l_finger_joint': 0.55
        }
        zero_pose.set_joint_goal(js)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

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

    def test_prismatic_joint2(self, kitchen_setup: PR2TestWrapper):
        kitchen_setup.allow_self_collision(kitchen_setup.robot_name)
        js = {'torso_lift_joint': 0.1}
        kitchen_setup.set_joint_goal(js)
        kitchen_setup.plan_and_execute()

    def test_hard_joint_limits(self, zero_pose: PR2TestWrapper):
        zero_pose.allow_self_collision()
        r_elbow_flex_joint = zero_pose.world.search_for_joint_name('r_elbow_flex_joint')
        torso_lift_joint = zero_pose.world.search_for_joint_name('torso_lift_joint')
        head_pan_joint = zero_pose.world.search_for_joint_name('head_pan_joint')
        r_elbow_flex_joint_limits = zero_pose.world.get_joint_position_limits(r_elbow_flex_joint)
        torso_lift_joint_limits = zero_pose.world.get_joint_position_limits(torso_lift_joint)
        head_pan_joint_limits = zero_pose.world.get_joint_position_limits(head_pan_joint)

        goal_js = {'r_elbow_flex_joint': r_elbow_flex_joint_limits[0] - 0.2,
                   'torso_lift_joint': torso_lift_joint_limits[0] - 0.2,
                   'head_pan_joint': head_pan_joint_limits[0] - 0.2}
        zero_pose.set_joint_goal(goal_js, check=False)
        zero_pose.plan_and_execute()
        js = {'torso_lift_joint': 0.32}
        zero_pose.set_joint_goal(js)
        zero_pose.plan_and_execute()

        goal_js = {'r_elbow_flex_joint': r_elbow_flex_joint_limits[1] + 0.2,
                   'torso_lift_joint': torso_lift_joint_limits[1] + 0.2,
                   'head_pan_joint': head_pan_joint_limits[1] + 0.2}

        zero_pose.set_joint_goal(goal_js, check=False)
        zero_pose.plan_and_execute()


class TestConstraints:
    # TODO write buggy constraints that test sanity checks

    def test_add_debug_expr(self, zero_pose: PR2TestWrapper):
        zero_pose.set_json_goal(constraint_type='DebugGoal')
        zero_pose.plan_and_execute()

    def test_SetSeedConfiguration(self, zero_pose: PR2TestWrapper):
        zero_pose.set_seed_configuration(seed_configuration=zero_pose.better_pose)
        zero_pose.set_joint_goal(zero_pose.default_pose)
        zero_pose.plan()

    def test_drive_into_apartment(self, apartment_setup: PR2TestWrapper):
        base_pose = PoseStamped()
        base_pose.header.frame_id = 'base_footprint'
        base_pose.pose.position.x = 0.4
        base_pose.pose.position.y = -2
        base_pose.pose.orientation.w = 1
        # apartment_setup.allow_all_collisions()
        apartment_setup.set_cart_goal(goal_pose=base_pose,
                                      tip_link='base_footprint',
                                      root_link=apartment_setup.default_root,
                                      check=False)
        apartment_setup.plan_and_execute()

    def test_VelocityLimitUnreachableException(self, zero_pose: PR2TestWrapper):
        zero_pose.set_prediction_horizon(prediction_horizon=7)
        zero_pose.set_joint_goal(zero_pose.better_pose)
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.QP_SOLVER_ERROR])

    def test_SetPredictionHorizon11(self, zero_pose: PR2TestWrapper):
        default_prediction_horizon = zero_pose.god_map.get_data(identifier.prediction_horizon)
        zero_pose.set_prediction_horizon(prediction_horizon=11)
        zero_pose.set_joint_goal(zero_pose.better_pose)
        zero_pose.plan_and_execute()
        assert zero_pose.god_map.get_data(identifier.prediction_horizon) == 11
        zero_pose.set_joint_goal(zero_pose.default_pose)
        zero_pose.plan_and_execute()
        assert zero_pose.god_map.get_data(identifier.prediction_horizon) == default_prediction_horizon

    def test_SetMaxTrajLength(self, zero_pose: PR2TestWrapper):
        new_length = 4
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 10
        base_goal.pose.orientation.w = 1
        zero_pose.set_max_traj_length(new_length)
        zero_pose.set_cart_goal(base_goal, tip_link='base_footprint', root_link='map')
        result = zero_pose.plan_and_execute(expected_error_codes=[MoveResult.PLANNING_ERROR])
        dt = zero_pose.god_map.get_data(identifier.sample_period)
        np.testing.assert_almost_equal(len(result.trajectory.points) * dt, new_length + dt * 2)

        zero_pose.set_cart_goal(base_goal, tip_link='base_footprint', root_link='map')
        result = zero_pose.plan_and_execute(expected_error_codes=[MoveResult.PLANNING_ERROR])
        dt = zero_pose.god_map.get_data(identifier.sample_period)
        assert len(result.trajectory.points) * dt > new_length + 1

    def test_CollisionAvoidanceHint(self, kitchen_setup: PR2TestWrapper):
        tip = 'base_footprint'
        base_pose = PoseStamped()
        base_pose.header.frame_id = 'map'
        base_pose.pose.position.x = 0
        base_pose.pose.position.y = 1.5
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0, 0, 1]))
        kitchen_setup.teleport_base(goal_pose=base_pose)
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

        expected = zero_pose.transform_msg('map', p)

        zero_pose.allow_all_collisions()
        zero_pose.set_translation_goal(root_link=zero_pose.default_root,
                                       tip_link=tip,
                                       goal_point=p)
        zero_pose.plan_and_execute()

    def test_CartesianPosition1(self, zero_pose: PR2TestWrapper):
        pocky = 'box'
        pocky_ps = PoseStamped()
        pocky_ps.header.frame_id = zero_pose.l_tip
        pocky_ps.pose.position.x = 0.05
        pocky_ps.pose.orientation.w = 1
        zero_pose.add_box(name=pocky,
                          size=(0.1, 0.02, 0.02),
                          parent_link=zero_pose.l_tip,
                          pose=pocky_ps)

        tip = zero_pose.r_tip
        p = PointStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = tip
        p.point = Point(0.0, 0.5, 0.0)

        zero_pose.allow_all_collisions()
        zero_pose.set_translation_goal(root_link=tip,
                                       root_group=zero_pose.robot_name,
                                       tip_link=pocky,
                                       tip_group='box',
                                       goal_point=p)
        zero_pose.plan_and_execute()

    def test_CartesianPose(self, zero_pose: PR2TestWrapper):
        tip = zero_pose.r_tip
        p = PoseStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = tip
        p.pose.position = Point(-0.4, -0.2, -0.3)
        p.pose.orientation = Quaternion(0, 0, 1, 0)

        expected = zero_pose.transform_msg('map', p)

        zero_pose.allow_all_collisions()
        zero_pose.set_json_goal('CartesianPose',
                                root_link=zero_pose.default_root,
                                root_group=None,
                                tip_link=tip,
                                tip_group=zero_pose.robot_name,
                                goal_pose=p)
        zero_pose.plan_and_execute()
        new_pose = zero_pose.world.compute_fk_pose('map', tip)
        compare_points(expected.pose.position, new_pose.pose.position)

    def test_JointPositionRevolute(self, zero_pose: PR2TestWrapper):
        joint = zero_pose.world.search_for_joint_name('r_shoulder_lift_joint')
        joint_goal = 1
        zero_pose.allow_all_collisions()
        zero_pose.set_json_goal('JointPositionRevolute',
                                joint_name=joint,
                                goal=joint_goal,
                                max_velocity=0.23)
        zero_pose.plan_and_execute()
        np.testing.assert_almost_equal(zero_pose.world.state[joint].position, joint_goal, decimal=3)

    def test_JointVelocityRevolute(self, zero_pose: PR2TestWrapper):
        joint = zero_pose.world.search_for_joint_name('r_shoulder_lift_joint')
        joint_goal = 1
        zero_pose.allow_all_collisions()
        zero_pose.set_json_goal('JointVelocityRevolute',
                                joint_name=joint,
                                max_velocity=0.4,
                                hard=True)
        zero_pose.set_json_goal('JointPositionRevolute',
                                joint_name=joint,
                                goal=joint_goal)
        zero_pose.plan_and_execute()
        np.testing.assert_almost_equal(zero_pose.world.state[joint].position, joint_goal, decimal=3)

    def test_JointPositionContinuous(self, zero_pose: PR2TestWrapper):
        joint = 'r_wrist_roll_joint'
        joint_goal = 4
        zero_pose.allow_all_collisions()
        zero_pose.set_json_goal('JointPositionContinuous',
                                joint_name=joint,
                                goal=joint_goal,
                                max_velocity=1)
        zero_pose.plan_and_execute()
        joint = zero_pose.world.search_for_joint_name(joint)
        np.testing.assert_almost_equal(zero_pose.world.state[joint].position, -2.283, decimal=2)

    def test_JointPosition_kitchen(self, kitchen_setup: PR2TestWrapper):
        joint_name1 = 'iai_fridge_door_joint'
        joint_name2 = 'sink_area_left_upper_drawer_main_joint'
        group_name = 'iai_kitchen'
        joint_goal = 0.4
        # kitchen_setup.allow_all_collisions()
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
            kitchen_setup.god_map.get_data(identifier.trajectory).get_last()[
                PrefixName(joint_name1, group_name)].position,
            joint_goal, decimal=2)
        np.testing.assert_almost_equal(
            kitchen_setup.god_map.get_data(identifier.trajectory).get_last()[
                PrefixName(joint_name2, group_name)].position,
            joint_goal, decimal=2)

    def test_CartesianOrientation(self, zero_pose: PR2TestWrapper):
        tip = 'base_footprint'
        root = 'odom_combined'
        q = QuaternionStamped()
        q.header.frame_id = tip
        q.quaternion = Quaternion(*quaternion_about_axis(4, [0, 0, 1]))

        expected = zero_pose.transform_msg('map', q)

        zero_pose.allow_all_collisions()
        zero_pose.set_rotation_goal(root_link=root,
                                    root_group=None,
                                    tip_link=tip,
                                    tip_group=zero_pose.robot_name,
                                    goal_orientation=q,
                                    max_velocity=0.15)
        zero_pose.plan_and_execute()

    def test_CartesianPoseStraight1(self, zero_pose: PR2TestWrapper):
        zero_pose.close_l_gripper()
        goal_position = PoseStamped()
        goal_position.header.frame_id = 'base_link'
        goal_position.pose.position.x = 0.3
        goal_position.pose.position.y = 0.5
        goal_position.pose.position.z = 1
        goal_position.pose.orientation.w = 1

        start_pose = zero_pose.world.compute_fk_pose('map', zero_pose.l_tip)
        map_T_goal_position = zero_pose.transform_msg('map', goal_position)

        object_pose = PoseStamped()
        object_pose.header.frame_id = 'map'
        object_pose.pose.position.x = (start_pose.pose.position.x + map_T_goal_position.pose.position.x) / 2.
        object_pose.pose.position.y = (start_pose.pose.position.y + map_T_goal_position.pose.position.y) / 2.
        object_pose.pose.position.z = (start_pose.pose.position.z + map_T_goal_position.pose.position.z) / 2.
        object_pose.pose.position.z += 0.08
        object_pose.pose.orientation.w = 1

        zero_pose.add_sphere('sphere', 0.05, pose=object_pose)

        publish_marker_vector(start_pose.pose.position, map_T_goal_position.pose.position)
        zero_pose.allow_self_collision(zero_pose.robot_name)
        goal_position_p = deepcopy(goal_position)
        goal_position_p.header.frame_id = 'base_link'
        zero_pose.set_straight_cart_goal(goal_pose=goal_position_p, tip_link=zero_pose.l_tip,
                                         root_link=zero_pose.default_root)
        zero_pose.plan_and_execute()

    def test_CartesianPoseStraight2(self, better_pose: PR2TestWrapper):
        better_pose.close_l_gripper()
        goal_position = PoseStamped()
        goal_position.header.frame_id = 'base_link'
        goal_position.pose.position.x = 0.8
        goal_position.pose.position.y = 0.5
        goal_position.pose.position.z = 1
        goal_position.pose.orientation.w = 1

        start_pose = better_pose.world.compute_fk_pose('map', better_pose.l_tip)
        map_T_goal_position = better_pose.transform_msg('map', goal_position)

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
        better_pose.set_straight_cart_goal(goal_pose=goal, tip_link=better_pose.l_tip,
                                           root_link=better_pose.default_root)
        better_pose.plan_and_execute()

        goal = deepcopy(object_pose)
        goal.pose.position.z -= 0.4
        better_pose.set_straight_cart_goal(goal_pose=goal, tip_link=better_pose.l_tip,
                                           root_link=better_pose.default_root)
        better_pose.plan_and_execute()

        goal = deepcopy(object_pose)
        goal.pose.position.y -= 0.4
        goal.pose.position.x -= 0.2
        better_pose.set_straight_cart_goal(goal_pose=goal, tip_link=better_pose.l_tip,
                                           root_link=better_pose.default_root)
        better_pose.plan_and_execute()

        goal = deepcopy(object_pose)
        goal.pose.position.x -= 0.4
        better_pose.set_straight_cart_goal(goal_pose=goal, tip_link=better_pose.l_tip,
                                           root_link=better_pose.default_root)
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
        zero_pose.set_avoid_joint_limits_goal(percentage=percentage)
        zero_pose.plan_and_execute()

        joint_non_continuous = [j for j in zero_pose.robot.controlled_joints if
                                not zero_pose.world.is_joint_continuous(j) and
                                (zero_pose.world.is_joint_prismatic(j) or zero_pose.world.is_joint_revolute(j))]

        current_joint_state = zero_pose.world.state.to_position_dict()
        percentage *= 0.95  # it will not reach the exact percentage, because the weight is so low
        for joint in joint_non_continuous:
            position = current_joint_state[joint]
            lower_limit, upper_limit = zero_pose.world.get_joint_position_limits(joint)
            joint_range = upper_limit - lower_limit
            center = (upper_limit + lower_limit) / 2.
            upper_limit2 = center + joint_range / 2. * (1 - percentage / 100.)
            lower_limit2 = center - joint_range / 2. * (1 - percentage / 100.)
            assert upper_limit2 >= position >= lower_limit2

    def test_AvoidJointLimits2(self, zero_pose: PR2TestWrapper):
        percentage = 10
        joint_non_continuous = [j for j in zero_pose.robot.controlled_joints if
                                not zero_pose.world.is_joint_continuous(j) and
                                (zero_pose.world.is_joint_prismatic(j) or zero_pose.world.is_joint_revolute(j))]
        goal_state = {j: zero_pose.world.get_joint_position_limits(j)[1] for j in joint_non_continuous}
        zero_pose.set_json_goal('AvoidJointLimits',
                                percentage=percentage)
        zero_pose.set_joint_goal(goal_state, check=False)
        zero_pose.allow_self_collision()
        zero_pose.plan_and_execute()

        zero_pose.set_json_goal('AvoidJointLimits',
                                percentage=percentage)
        zero_pose.allow_self_collision()
        zero_pose.plan_and_execute()

        current_joint_state = zero_pose.world.state.to_position_dict()
        percentage *= 0.9  # it will not reach the exact percentage, because the weight is so low
        for joint in joint_non_continuous:
            position = current_joint_state[joint]
            lower_limit, upper_limit = zero_pose.world.get_joint_position_limits(joint)
            joint_range = upper_limit - lower_limit
            center = (upper_limit + lower_limit) / 2.
            upper_limit2 = center + joint_range / 2. * (1 - percentage / 100.)
            lower_limit2 = center - joint_range / 2. * (1 - percentage / 100.)
            assert upper_limit2 >= position >= lower_limit2

    # def test_OverwriteWeights1(self, pocky_pose_setup: PR2TestWrapper):
    #     # FIXME
    #     # joint_velocity_weight = identifier.joint_weights + ['velocity', 'override']
    #     # old_torso_value = pocky_pose_setup.world.joints['torso_lift_joint'].free_variable.quadratic_weights
    #     # old_odom_x_value = pocky_pose_setup.world.joints['odom_x_joint'].free_variable.quadratic_weights
    #
    #     r_goal = PoseStamped()
    #     r_goal.header.frame_id = pocky_pose_setup.r_tip
    #     r_goal.pose.orientation.w = 1
    #     r_goal.pose.position.x += 0.1
    #     updates = {
    #         1: {
    #             'odom_x_joint': 1000000,
    #             'odom_y_joint': 1000000,
    #             'odom_z_joint': 1000000
    #         },
    #     }
    #
    #     old_pose = tf.lookup_pose('map', 'base_footprint')
    #
    #     pocky_pose_setup.set_overwrite_joint_weights_goal(updates)
    #     pocky_pose_setup.set_cart_goal(r_goal, pocky_pose_setup.r_tip, check=False)
    #     pocky_pose_setup.plan_and_execute()
    #
    #     new_pose = tf.lookup_pose('map', 'base_footprint')
    #     compare_poses(new_pose.pose, old_pose.pose)
    #
    #     assert pocky_pose_setup.world._joints['odom_x_joint'].free_variable.quadratic_weights[1] == 1000000
    #     assert not isinstance(pocky_pose_setup.world._joints['torso_lift_joint'].free_variable.quadratic_weights[1],
    #                           int)
    #
    #     updates = {
    #         1: {
    #             'odom_x_joint': 0.0001,
    #             'odom_y_joint': 0.0001,
    #             'odom_z_joint': 0.0001,
    #         },
    #     }
    #     # old_pose = tf.lookup_pose('map', 'base_footprint')
    #     # old_pose.pose.position.x += 0.1
    #     pocky_pose_setup.set_overwrite_joint_weights_goal(updates)
    #     pocky_pose_setup.set_cart_goal(r_goal, pocky_pose_setup.r_tip)
    #     pocky_pose_setup.plan_and_execute()
    #
    #     new_pose = tf.lookup_pose('map', 'base_footprint')
    #
    #     # compare_poses(old_pose.pose, new_pose.pose)
    #     assert new_pose.pose.position.x >= 0.03
    #     assert pocky_pose_setup.world._joints['odom_x_joint'].free_variable.quadratic_weights[1] == 0.0001
    #     assert not isinstance(pocky_pose_setup.world._joints['torso_lift_joint'].free_variable.quadratic_weights[1],
    #                           float)
    #     pocky_pose_setup.plan_and_execute()
    #     assert not isinstance(pocky_pose_setup.world._joints['odom_x_joint'].free_variable.quadratic_weights[1],
    #                           float)
    #     assert not isinstance(pocky_pose_setup.world._joints['torso_lift_joint'].free_variable.quadratic_weights[1],
    #                           float)

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
        kitchen_setup.set_pointing_goal(tip_link=tip, goal_point=goal_point, root_link=kitchen_setup.default_root,
                                        pointing_axis=pointing_axis)
        kitchen_setup.plan_and_execute()

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'pr2/base_footprint'
        base_goal.pose.position.y = 2
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(1, [0, 0, 1]))
        kitchen_setup.set_pointing_goal(tip_link=tip, goal_point=goal_point, pointing_axis=pointing_axis,
                                        root_link=kitchen_setup.default_root)
        gaya_pose2 = deepcopy(kitchen_setup.better_pose)
        del gaya_pose2['head_pan_joint']
        del gaya_pose2['head_tilt_joint']
        kitchen_setup.set_joint_goal(gaya_pose2)
        kitchen_setup.move_base(base_goal)

        current_x = Vector3Stamped()
        current_x.header.frame_id = tip
        current_x.vector.x = 1

        expected_x = kitchen_setup.transform_msg(tip, goal_point)
        np.testing.assert_almost_equal(expected_x.point.y, 0, 1)
        np.testing.assert_almost_equal(expected_x.point.z, 0, 1)

        rospy.loginfo("Starting looking")
        tip = 'head_mount_kinect_rgb_link'
        goal_point = kitchen_setup.world.compute_fk_point('map', kitchen_setup.r_tip)
        goal_point.header.stamp = rospy.Time()
        pointing_axis = Vector3Stamped()
        pointing_axis.header.frame_id = tip
        pointing_axis.vector.x = 1
        kitchen_setup.set_pointing_goal(tip_link=tip, goal_point=goal_point, pointing_axis=pointing_axis,
                                        root_link=kitchen_setup.r_tip)

        rospy.loginfo("Starting pointing")
        r_goal = PoseStamped()
        r_goal.header.frame_id = kitchen_setup.r_tip
        r_goal.pose.position.x -= 0.3
        r_goal.pose.position.z += 0.6
        r_goal.pose.orientation.w = 1
        r_goal = kitchen_setup.transform_msg(kitchen_setup.default_root, r_goal)
        r_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 0, -1, 0],
                                                                      [0, 1, 0, 0],
                                                                      [1, 0, 0, 0],
                                                                      [0, 0, 0, 1]]))
        r_goal.header.frame_id = kitchen_setup.r_tip
        kitchen_setup.set_cart_goal(goal_pose=r_goal,
                                    tip_link=kitchen_setup.r_tip,
                                    root_link='base_footprint',
                                    weight=WEIGHT_BELOW_CA,
                                    check=False)
        kitchen_setup.plan_and_execute()

    # def test_pointing_bug(self, zero_pose: PR2TestWrapper):
    #     initial_joint_state = {
    #         'torso_lift_joint': 0.31261531343064947,
    #         'head_pan_joint': -2.8762399155129605,
    #         'head_tilt_joint': 1.227067553622289,
    #         'r_upper_arm_roll_joint': -1.4298359538624308,
    #         'r_shoulder_pan_joint': -0.03837121868433646,
    #         'r_shoulder_lift_joint': -0.2777931728916727,
    #         'r_forearm_roll_joint': -35.932852605836715,
    #         'r_elbow_flex_joint': -2.1155076122857492,
    #         'r_wrist_flex_joint': -0.10505779734036036,
    #         'r_wrist_roll_joint': -12.515290560123026,
    #         'l_upper_arm_roll_joint': 1.3837617139225475,
    #         'l_shoulder_pan_joint': 1.965374844556896,
    #         'l_shoulder_lift_joint': -0.2649135724042734,
    #         'l_forearm_roll_joint': 117.52740957656653,
    #         'l_elbow_flex_joint': -2.1157971537085163,
    #         'l_wrist_flex_joint': -0.10313747048706379,
    #         'l_wrist_roll_joint': 6.28332367137161,
    #     }
    #     zero_pose.set_seed_configuration(initial_joint_state)
    #     initial_base_pose = PoseStamped()
    #     initial_base_pose.header.frame_id = 'map'
    #     initial_base_pose.pose.position = Point(1.576, 2.535, -0.000)
    #     initial_base_pose.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.000)
    #     zero_pose.set_seed_odometry(initial_base_pose)
    #     zero_pose.plan_and_execute()
    #
    #     pointing_axis = Vector3Stamped()
    #     pointing_axis.header.frame_id = 'narrow_stereo_optical_frame'
    #     pointing_axis.vector.z = 1
    #
    #     goal_point = PointStamped()
    #     goal_point.header.frame_id = 'map'
    #     goal_point.point = Point(2.0, 2.6, 1.0)
    #     zero_pose.set_pointing_goal(goal_point=goal_point,
    #                                     pointing_axis=pointing_axis,
    #                                     tip_link='narrow_stereo_optical_frame',
    #                                     root_link='base_footprint')
    #     zero_pose.plan_and_execute()

    # def test_open_fridge(self, kitchen_setup: PR2TestWrapper):
    #     handle_frame_id = 'iai_kitchen/iai_fridge_door_handle'
    #     handle_name = 'iai_fridge_door_handle'
    #
    #     base_goal = PoseStamped()
    #     base_goal.header.frame_id = 'map'
    #     base_goal.pose.position = Point(0.3, -0.5, 0)
    #     base_goal.pose.orientation.w = 1
    #     kitchen_setup.teleport_base(base_goal)
    #
    #     bar_axis = Vector3Stamped()
    #     bar_axis.header.frame_id = handle_frame_id
    #     bar_axis.vector.z = 1
    #
    #     bar_center = PointStamped()
    #     bar_center.header.frame_id = handle_frame_id
    #
    #     tip_grasp_axis = Vector3Stamped()
    #     tip_grasp_axis.header.frame_id = kitchen_setup.r_tip
    #     tip_grasp_axis.vector.z = 1
    #
    #     kitchen_setup.set_json_goal('GraspBar',
    #                                 root_link=kitchen_setup.default_root,
    #                                 tip_link=kitchen_setup.r_tip,
    #                                 tip_grasp_axis=tip_grasp_axis,
    #                                 bar_center=bar_center,
    #                                 bar_axis=bar_axis,
    #                                 bar_length=.4)
    #     x_gripper = Vector3Stamped()
    #     x_gripper.header.frame_id = kitchen_setup.r_tip
    #     x_gripper.vector.x = 1
    #
    #     x_goal = Vector3Stamped()
    #     x_goal.header.frame_id = handle_frame_id
    #     x_goal.vector.x = -1
    #     kitchen_setup.set_align_planes_goal(tip_link=kitchen_setup.r_tip, tip_normal=x_gripper,
    #                                         goal_normal=x_goal)
    #     kitchen_setup.allow_all_collisions()
    #     # kitchen_setup.add_json_goal('AvoidJointLimits', percentage=10)
    #     kitchen_setup.plan_and_execute()
    #
    #     kitchen_setup.set_json_goal('Open',
    #                                 tip_link=kitchen_setup.r_tip,
    #                                 environment_link=handle_name,
    #                                 goal_joint_state=1.5)
    #     kitchen_setup.set_json_goal('AvoidJointLimits', percentage=40)
    #     kitchen_setup.allow_all_collisions()
    #     # kitchen_setup.add_json_goal('AvoidJointLimits')
    #     kitchen_setup.plan_and_execute()
    #     kitchen_setup.set_kitchen_js({'iai_fridge_door_joint': 1.5})
    #
    #     kitchen_setup.set_json_goal('Open',
    #                                 tip_link=kitchen_setup.r_tip,
    #                                 environment_link=handle_name,
    #                                 goal_joint_state=0)
    #     kitchen_setup.allow_all_collisions()
    #     kitchen_setup.set_json_goal('AvoidJointLimits', percentage=40)
    #     kitchen_setup.plan_and_execute()
    #     kitchen_setup.set_kitchen_js({'iai_fridge_door_joint': 0})
    #
    #     # kitchen_setup.plan_and_execute()
    #
    #     kitchen_setup.set_joint_goal(kitchen_setup.better_pose)
    #     kitchen_setup.allow_all_collisions()
    #     kitchen_setup.plan_and_execute()

    def test_open_drawer(self, kitchen_setup: PR2TestWrapper):
        handle_frame_id = 'iai_kitchen/sink_area_left_middle_drawer_handle'
        handle_name = 'sink_area_left_middle_drawer_handle'
        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = handle_frame_id
        bar_axis.vector.y = 1

        bar_center = PointStamped()
        bar_center.header.frame_id = handle_frame_id

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = str(PrefixName(kitchen_setup.l_tip, 'pr2'))
        tip_grasp_axis.vector.z = 1

        kitchen_setup.set_json_goal('GraspBar',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=kitchen_setup.l_tip,
                                    tip_grasp_axis=tip_grasp_axis,
                                    bar_center=bar_center,
                                    bar_axis=bar_axis,
                                    bar_length=0.4)
        x_gripper = Vector3Stamped()
        x_gripper.header.frame_id = str(PrefixName(kitchen_setup.l_tip, 'pr2'))
        x_gripper.vector.x = 1

        x_goal = Vector3Stamped()
        x_goal.header.frame_id = handle_frame_id
        x_goal.vector.x = -1

        kitchen_setup.set_align_planes_goal(tip_link=kitchen_setup.l_tip,
                                            tip_normal=x_gripper,
                                            goal_normal=x_goal, check=False)
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

    def test_open_close_dishwasher(self, kitchen_setup: PR2TestWrapper):
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.orientation.w = 1
        p.pose.position.x = 0.5
        p.pose.position.y = 0.2
        kitchen_setup.teleport_base(p)

        hand = kitchen_setup.r_tip

        goal_angle = np.pi / 4
        handle_frame_id = 'sink_area_dish_washer_door_handle'
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
        kitchen_setup.set_align_planes_goal(tip_link=hand,
                                            tip_normal=x_gripper,
                                            goal_normal=x_goal)
        # kitchen_setup.allow_all_collisions()

        kitchen_setup.plan_and_execute()

        kitchen_setup.set_json_goal('Open',
                                    tip_link=hand,
                                    environment_link=handle_name,
                                    goal_joint_state=goal_angle)
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.allow_collision(group1=kitchen_setup.kitchen_name, group2=kitchen_setup.r_gripper_group)
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_kitchen_js({'sink_area_dish_washer_door_joint': goal_angle})

        kitchen_setup.set_json_goal('Open',
                                    tip_link=hand,
                                    environment_link=handle_name,
                                    goal_joint_state=0)
        kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_kitchen_js({'sink_area_dish_washer_door_joint': 0})

    def test_align_planes1(self, zero_pose: PR2TestWrapper):
        x_gripper = Vector3Stamped()
        x_gripper.header.frame_id = str(PrefixName(zero_pose.r_tip, zero_pose.robot_name))
        x_gripper.vector.x = 1
        y_gripper = Vector3Stamped()
        y_gripper.header.frame_id = str(PrefixName(zero_pose.r_tip, zero_pose.robot_name))
        y_gripper.vector.y = 1

        x_goal = Vector3Stamped()
        x_goal.header.frame_id = 'map'
        x_goal.vector.x = 1
        y_goal = Vector3Stamped()
        y_goal.header.frame_id = 'map'
        y_goal.vector.z = 1
        zero_pose.set_align_planes_goal(tip_link=zero_pose.r_tip, tip_normal=x_gripper, goal_normal=x_goal)
        zero_pose.set_align_planes_goal(tip_link=zero_pose.r_tip, tip_normal=y_gripper, goal_normal=y_goal)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

    def test_wrong_constraint_type(self, zero_pose: PR2TestWrapper):
        goal_state = {'r_elbow_flex_joint': -1.0}
        kwargs = {'goal_state': goal_state}
        zero_pose.set_json_goal('jointpos', **kwargs)
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.UNKNOWN_CONSTRAINT])

    def test_python_code_in_constraint_type(self, zero_pose: PR2TestWrapper):
        goal_state = {'r_elbow_flex_joint': -1.0}
        kwargs = {'goal_state': goal_state}
        zero_pose.set_json_goal('print("muh")', **kwargs)
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.UNKNOWN_CONSTRAINT])

    def test_wrong_params1(self, zero_pose: PR2TestWrapper):
        goal_state = {5432: 'muh'}
        kwargs = {'goal_state': goal_state}
        zero_pose.set_json_goal('JointPositionList', **kwargs)
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.CONSTRAINT_INITIALIZATION_ERROR])

    def test_wrong_params2(self, zero_pose: PR2TestWrapper):
        goal_state = {'r_elbow_flex_joint': 'muh'}
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
        zero_pose.set_align_planes_goal(tip_link=zero_pose.r_tip, tip_normal=x_gripper, goal_normal=x_goal)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

    def test_align_planes3(self, zero_pose: PR2TestWrapper):
        eef_vector = Vector3Stamped()
        eef_vector.header.frame_id = 'base_footprint'
        eef_vector.vector.y = 1

        goal_vector = Vector3Stamped()
        goal_vector.header.frame_id = 'map'
        goal_vector.vector.x = 1
        goal_vector.vector = tf.normalize(goal_vector.vector)
        zero_pose.set_align_planes_goal(tip_link='base_footprint', tip_normal=eef_vector, goal_normal=goal_vector)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

    def test_align_planes4(self, kitchen_setup: PR2TestWrapper):
        elbow = 'r_elbow_flex_link'
        handle_frame_id = 'iai_fridge_door_handle'

        tip_axis = Vector3Stamped()
        tip_axis.header.frame_id = elbow
        tip_axis.vector.x = 1

        env_axis = Vector3Stamped()
        env_axis.header.frame_id = handle_frame_id
        env_axis.vector.z = 1
        kitchen_setup.set_align_planes_goal(tip_link=elbow, tip_normal=tip_axis, goal_normal=env_axis,
                                            weight=WEIGHT_ABOVE_CA)
        kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()

    def test_grasp_fridge_handle(self, kitchen_setup: PR2TestWrapper):
        handle_name = 'iai_fridge_door_handle'
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

        x_gripper = Vector3Stamped()
        x_gripper.header.frame_id = kitchen_setup.r_tip
        x_gripper.vector.x = 1

        x_goal = Vector3Stamped()
        x_goal.header.frame_id = 'iai_fridge_door_handle'
        x_goal.vector.x = -1
        kitchen_setup.set_align_planes_goal(tip_link=kitchen_setup.r_tip, tip_normal=x_gripper, goal_normal=x_goal)
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()

    def test_close_fridge_with_elbow(self, kitchen_setup: PR2TestWrapper):
        base_pose = PoseStamped()
        base_pose.header.frame_id = 'map'
        base_pose.pose.position.y = -1.5
        base_pose.pose.orientation.w = 1
        kitchen_setup.teleport_base(base_pose)

        handle_frame_id = 'iai_fridge_door_handle'
        handle_name = 'iai_fridge_door_handle'

        kitchen_setup.set_kitchen_js({'iai_fridge_door_joint': np.pi / 2})

        elbow = 'r_elbow_flex_link'

        tip_axis = Vector3Stamped()
        tip_axis.header.frame_id = elbow
        tip_axis.vector.x = 1

        env_axis = Vector3Stamped()
        env_axis.header.frame_id = handle_frame_id
        env_axis.vector.z = 1
        kitchen_setup.set_align_planes_goal(tip_link=elbow, tip_normal=tip_axis, goal_normal=env_axis,
                                            weight=WEIGHT_ABOVE_CA)
        kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()
        elbow_point = PointStamped()
        elbow_point.header.frame_id = handle_frame_id
        elbow_point.point.x += 0.1
        kitchen_setup.set_translation_goal(elbow_point, elbow)
        kitchen_setup.set_align_planes_goal(tip_link=elbow, tip_normal=tip_axis, goal_normal=env_axis,
                                            weight=WEIGHT_ABOVE_CA)
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
        kitchen_setup.set_align_planes_goal(tip_link=kitchen_setup.l_tip, tip_normal=x_gripper,
                                            goal_normal=x_goal)
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
        tip_grasp_axis.header.frame_id = str(PrefixName(kitchen_setup.r_tip, kitchen_setup.robot_name))
        tip_grasp_axis.vector.z = 1

        kitchen_setup.set_grasp_bar_goal(root_link=kitchen_setup.default_root,
                                         tip_link=kitchen_setup.r_tip,
                                         tip_grasp_axis=tip_grasp_axis,
                                         bar_center=bar_center,
                                         bar_axis=bar_axis,
                                         bar_length=.3)
        kitchen_setup.register_group('handle', 'iai_kitchen', 'sink_area_dish_washer_door_handle')
        kitchen_setup.allow_collision(kitchen_setup.robot_name, 'handle')
        kitchen_setup.plan_and_execute()


class TestMoveBaseGoals:
    def test_left_1m(self, zero_pose: PR2TestWrapper):
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.y = 1
        base_goal.pose.orientation.w = 1
        zero_pose.move_base(base_goal)

    def test_left_1cm(self, zero_pose: PR2TestWrapper):
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.y = 0.01
        base_goal.pose.orientation.w = 1
        zero_pose.move_base(base_goal)

    def test_forward_left_rotate(self, zero_pose: PR2TestWrapper):
        map_T_odom = PoseStamped()
        map_T_odom.header.frame_id = 'map'
        map_T_odom.pose.position.x = 1
        map_T_odom.pose.position.y = 1
        map_T_odom.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 3, [0, 0, 1]))
        zero_pose.set_localization(map_T_odom)

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 1
        base_goal.pose.position.y = -1
        # base_goal.pose.orientation.w = 1
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(-pi / 4, [0, 0, 1]))
        zero_pose.allow_all_collisions()
        zero_pose.move_base(base_goal)

    def test_circle(self, zero_pose: PR2TestWrapper):
        center = PointStamped()
        center.header.frame_id = zero_pose.default_root
        zero_pose.set_json_goal(constraint_type='Circle',
                                center=center,
                                radius=0.5,
                                tip_link='base_footprint',
                                scale=0.1)
        # zero_pose.set_json_goal('PR2CasterConstraints')
        zero_pose.set_max_traj_length(new_length=60)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

    def test_stay_put(self, zero_pose: PR2TestWrapper):
        base_goal = PoseStamped()
        base_goal.header.frame_id = zero_pose.default_root
        # base_goal.pose.position.y = 0.05
        base_goal.pose.orientation.w = 1
        # zero_pose.set_json_goal('PR2CasterConstraints')
        zero_pose.set_joint_goal(zero_pose.better_pose)
        zero_pose.move_base(base_goal)

    def test_forward_1m(self, zero_pose: PR2TestWrapper):
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 1
        base_goal.pose.orientation.w = 1
        zero_pose.allow_all_collisions()
        zero_pose.move_base(base_goal)

    def test_wave(self, zero_pose: PR2TestWrapper):
        center = PointStamped()
        center.header.frame_id = zero_pose.default_root
        zero_pose.allow_all_collisions()
        zero_pose.set_json_goal(constraint_type='Wave',
                                center=center,
                                radius=0.05,
                                tip_link='base_footprint',
                                scale=2)
        zero_pose.set_joint_goal(zero_pose.better_pose, check=False)
        zero_pose.plan_and_execute()

    def test_forward_1cm(self, zero_pose: PR2TestWrapper):
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 0.01
        base_goal.pose.orientation.w = 1
        zero_pose.move_base(base_goal)

    def test_forward_left_1m_1m(self, zero_pose: PR2TestWrapper):
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 1
        base_goal.pose.position.y = 1
        base_goal.pose.orientation.w = 1
        zero_pose.move_base(base_goal)

    def test_forward_left_1cm_1cm(self, zero_pose: PR2TestWrapper):
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 0.01
        base_goal.pose.position.y = 0.01
        base_goal.pose.orientation.w = 1
        zero_pose.move_base(base_goal)

    def test_forward_right_and_rotate(self, zero_pose: PR2TestWrapper):
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 1
        base_goal.pose.position.y = -1
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(-pi / 4, [0, 0, 1]))
        zero_pose.move_base(base_goal)

    def test_forward_then_left(self, zero_pose: PR2TestWrapper):
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 1
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(-pi / 4, [0, 0, 1]))
        zero_pose.move_base(base_goal)
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 0
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(pi / 3, [0, 0, 1]))
        zero_pose.move_base(base_goal)

    def test_rotate_pi_half(self, zero_pose: PR2TestWrapper):
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(-pi / 2, [0, 0, 1]))
        zero_pose.allow_all_collisions()
        zero_pose.move_base(base_goal)

    def test_rotate_pi(self, zero_pose: PR2TestWrapper):
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        zero_pose.move_base(base_goal)

    def test_rotate_0_001_rad(self, zero_pose: PR2TestWrapper):
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(0.001, [0, 0, 1]))
        zero_pose.move_base(base_goal)


class TestCartGoals:
    def test_two_eef_goal(self, zero_pose: PR2TestWrapper):
        r_goal = PoseStamped()
        r_goal.header.frame_id = 'r_gripper_tool_frame'
        r_goal.pose.position = Point(-0.2, -0.2, 0.2)
        r_goal.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_cart_goal(root_link='map', tip_link='r_gripper_tool_frame', goal_pose=r_goal)

        l_goal = PoseStamped()
        l_goal.header.frame_id = 'l_gripper_tool_frame'
        l_goal.pose.position = Point(0.2, 0.2, 0.2)
        l_goal.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_cart_goal(root_link='map', tip_link='l_gripper_tool_frame', goal_pose=l_goal)

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
        expected_pose = zero_pose.world.compute_fk_pose(zero_pose.default_root, zero_pose.r_tip)
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
        zero_pose.allow_all_collisions()
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
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

    def test_cart_goal_left_right_chain(self, zero_pose: PR2TestWrapper):
        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.l_tip
        r_goal.pose.position.x = 0.2
        r_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[-1, 0, 0, 0],
                                                                      [0, -1, 0, 0],
                                                                      [0, 0, 1, 0],
                                                                      [0, 0, 0, 1]]))
        zero_pose.allow_all_collisions()
        zero_pose.set_cart_goal(goal_pose=r_goal,
                                tip_link=zero_pose.r_tip,
                                root_link=zero_pose.l_tip)
        zero_pose.plan_and_execute()

    # def test_wiggle1(self, kitchen_setup: PR2TestWrapper):
    #     # FIXME
    #     tray_pose = PoseStamped()
    #     tray_pose.header.frame_id = 'sink_area_surface'
    #     tray_pose.pose.position = Point(0.1, -0.4, 0.07)
    #     tray_pose.pose.orientation.w = 1
    #
    #     l_goal = deepcopy(tray_pose)
    #     l_goal.pose.position.y -= 0.18
    #     l_goal.pose.position.z += 0.05
    #     l_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[0, -1, 0, 0],
    #                                                                   [1, 0, 0, 0],
    #                                                                   [0, 0, 1, 0],
    #                                                                   [0, 0, 0, 1]]))
    #
    #     r_goal = deepcopy(tray_pose)
    #     r_goal.pose.position.y += 0.18
    #     r_goal.pose.position.z += 0.05
    #     r_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 1, 0, 0],
    #                                                                   [-1, 0, 0, 0],
    #                                                                   [0, 0, 1, 0],
    #                                                                   [0, 0, 0, 1]]))
    #
    #     kitchen_setup.set_cart_goal(l_goal, kitchen_setup.l_tip, weight=WEIGHT_BELOW_CA)
    #     kitchen_setup.set_cart_goal(r_goal, kitchen_setup.r_tip, weight=WEIGHT_BELOW_CA)
    #     # kitchen_setup.allow_collision([], tray_name, [])
    #     # kitchen_setup.allow_all_collisions()
    #     kitchen_setup.set_limit_cartesian_velocity_goal(tip_link='base_footprint',
    #                                                     root_link=kitchen_setup.default_root,
    #                                                     max_linear_velocity=0.1,
    #                                                     max_angular_velocity=0.2)
    #     kitchen_setup.plan_and_execute()

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


class TestWayPoints:
    def test_waypoints2(self, zero_pose: PR2TestWrapper):
        zero_pose.set_joint_goal(pocky_pose, check=False)
        zero_pose.allow_all_collisions()
        zero_pose.add_cmd()
        zero_pose.set_joint_goal(pick_up_pose, check=False)
        zero_pose.allow_all_collisions()
        zero_pose.add_cmd()
        zero_pose.set_joint_goal(zero_pose.better_pose, check=False)
        zero_pose.allow_all_collisions()

        traj = zero_pose.plan_and_execute().trajectory
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
                                   goal_type=MoveGoal.PLAN_AND_EXECUTE_AND_SKIP_FAILURES).trajectory

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
                                   goal_type=MoveGoal.PLAN_AND_EXECUTE_AND_SKIP_FAILURES).trajectory

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
                                   goal_type=MoveGoal.PLAN_AND_EXECUTE_AND_SKIP_FAILURES).trajectory

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
                                   goal_type=MoveGoal.PLAN_AND_EXECUTE).trajectory

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
                                   goal_type=MoveGoal.PLAN_AND_EXECUTE_AND_SKIP_FAILURES).trajectory

        for i, p in enumerate(traj.points):
            js = {joint_name: position for joint_name, position in zip(traj.joint_names, p.positions)}
            try:
                zero_pose.compare_joint_state(js, pocky_pose)
                break
            except AssertionError:
                pass
        else:  # if no break
            assert False, 'pocky pose not in trajectory'


# class TestShaking(object):
#     def test_wiggle_prismatic_joint_neglectable_shaking(self, kitchen_setup: PR2TestWrapper):
#         # FIXME
#         sample_period = kitchen_setup.god_map.get_data(identifier.sample_period)
#         frequency_range = kitchen_setup.god_map.get_data(identifier.frequency_range)
#         amplitude_threshold = kitchen_setup.god_map.get_data(identifier.amplitude_threshold)
#         max_detectable_freq = int(1 / (2 * sample_period))
#         min_wiggle_frequency = int(frequency_range * max_detectable_freq)
#         while np.fmod(min_wiggle_frequency, 5.0) != 0.0:
#             min_wiggle_frequency += 1
#         distance_between_frequencies = 5
#
#         for i, t in enumerate([('torso_lift_joint', 0.05), ('odom_x_joint', 0.5)]):  # max vel: 0.015 and 0.5
#             for f in range(min_wiggle_frequency, max_detectable_freq, distance_between_frequencies):
#                 target_freq = float(f)
#                 joint = t[0]
#                 goal = t[1]
#                 kitchen_setup.set_json_goal('JointPositionPrismatic',
#                                             joint_name=joint,
#                                             goal=0.0,
#                                             )
#                 kitchen_setup.plan_and_execute()
#                 kitchen_setup.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
#                 kitchen_setup.set_json_goal('ShakyJointPositionRevoluteOrPrismatic',
#                                             joint_name=joint,
#                                             noise_amplitude=amplitude_threshold - 0.05,
#                                             goal=goal,
#                                             frequency=target_freq
#                                             )
#                 kitchen_setup.plan_and_execute()
#
#     def test_wiggle_revolute_joint_neglectable_shaking(self, kitchen_setup: PR2TestWrapper):
#         # FIXME
#         sample_period = kitchen_setup.god_map.get_data(identifier.sample_period)
#         frequency_range = kitchen_setup.god_map.get_data(identifier.frequency_range)
#         amplitude_threshold = kitchen_setup.god_map.get_data(identifier.amplitude_threshold)
#         max_detectable_freq = int(1 / (2 * sample_period))
#         min_wiggle_frequency = int(frequency_range * max_detectable_freq)
#         while np.fmod(min_wiggle_frequency, 5.0) != 0.0:
#             min_wiggle_frequency += 1
#         distance_between_frequencies = 5
#
#         for i, joint in enumerate(['r_wrist_flex_joint', 'head_pan_joint']):  # max vel: 1.0 and 0.5
#             for f in range(min_wiggle_frequency, max_detectable_freq, distance_between_frequencies):
#                 target_freq = float(f)
#                 kitchen_setup.set_json_goal('JointPositionRevolute',
#                                             joint_name=joint,
#                                             goal=0.0,
#                                             )
#                 kitchen_setup.plan_and_execute()
#                 kitchen_setup.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
#                 kitchen_setup.set_json_goal('ShakyJointPositionRevoluteOrPrismatic',
#                                             joint_name=joint,
#                                             noise_amplitude=amplitude_threshold - 0.05,
#                                             goal=-1.0,
#                                             frequency=target_freq
#                                             )
#                 kitchen_setup.plan_and_execute()
#
#     def test_wiggle_continuous_joint_neglectable_shaking(self, kitchen_setup: PR2TestWrapper):
#         sample_period = kitchen_setup.god_map.get_data(identifier.sample_period)
#         frequency_range = kitchen_setup.god_map.get_data(identifier.frequency_range)
#         amplitude_threshold = kitchen_setup.god_map.get_data(identifier.amplitude_threshold)
#         max_detectable_freq = int(1 / (2 * sample_period))
#         min_wiggle_frequency = int(frequency_range * max_detectable_freq)
#         while np.fmod(min_wiggle_frequency, 5.0) != 0.0:
#             min_wiggle_frequency += 1
#         distance_between_frequencies = 5
#
#         for continuous_joint in ['l_wrist_roll_joint', 'r_forearm_roll_joint']:  # max vel. of 1.0 and 1.0
#             for f in range(min_wiggle_frequency, max_detectable_freq, distance_between_frequencies):
#                 kitchen_setup.set_json_goal('JointPositionContinuous',
#                                             joint_name=continuous_joint,
#                                             goal=5.0,
#                                             )
#                 kitchen_setup.plan_and_execute()
#                 target_freq = float(f)
#                 kitchen_setup.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
#                 kitchen_setup.set_json_goal('ShakyJointPositionContinuous',
#                                             joint_name=continuous_joint,
#                                             goal=-5.0,
#                                             noise_amplitude=amplitude_threshold - 0.05,
#                                             frequency=target_freq
#                                             )
#                 kitchen_setup.plan_and_execute()
#
#     def test_wiggle_revolute_joint_shaking(self, kitchen_setup: PR2TestWrapper):
#         sample_period = kitchen_setup.god_map.get_data(identifier.sample_period)
#         frequency_range = kitchen_setup.god_map.get_data(identifier.frequency_range)
#         max_detectable_freq = int(1 / (2 * sample_period))
#         min_wiggle_frequency = int(frequency_range * max_detectable_freq)
#         while np.fmod(min_wiggle_frequency, 5.0) != 0.0:
#             min_wiggle_frequency += 1
#         distance_between_frequencies = 5
#
#         for joint in ['head_pan_joint', 'r_wrist_flex_joint']:  # max vel: 1.0 and 0.5
#             for f in range(min_wiggle_frequency, max_detectable_freq, distance_between_frequencies):
#                 kitchen_setup.set_json_goal('JointPositionRevolute',
#                                             joint_name=joint,
#                                             goal=0.5,
#                                             )
#                 kitchen_setup.plan_and_execute()
#                 target_freq = float(f)
#                 kitchen_setup.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
#                 kitchen_setup.set_json_goal('ShakyJointPositionRevoluteOrPrismatic',
#                                             joint_name=joint,
#                                             goal=0.0,
#                                             frequency=target_freq
#                                             )
#                 r = kitchen_setup.plan_and_execute(expected_error_codes=[MoveResult.SHAKING])
#                 assert len(r.error_codes) != 0
#                 error_code = r.error_codes[0]
#                 assert error_code == MoveResult.SHAKING
#                 error_message = r.error_messages[0]
#                 freqs_str = re.findall("[0-9]+\.[0-9]+ hertz", error_message)
#                 assert any(map(lambda f_str: float(f_str[:-6]) == target_freq, freqs_str))
#
#     def test_wiggle_prismatic_joint_shaking(self, kitchen_setup: PR2TestWrapper):
#         sample_period = kitchen_setup.god_map.get_data(identifier.sample_period)
#         frequency_range = kitchen_setup.god_map.get_data(identifier.frequency_range)
#         max_detectable_freq = int(1 / (2 * sample_period))
#         min_wiggle_frequency = int(frequency_range * max_detectable_freq)
#         while np.fmod(min_wiggle_frequency, 5.0) != 0.0:
#             min_wiggle_frequency += 1
#         distance_between_frequencies = 5
#
#         for joint in ['odom_x_joint']:  # , 'torso_lift_joint']: # max vel: 0.015 and 0.5
#             for f in range(min_wiggle_frequency, max_detectable_freq, distance_between_frequencies):
#                 kitchen_setup.set_json_goal('JointPositionPrismatic',
#                                             joint_name=joint,
#                                             goal=0.02,
#                                             )
#                 kitchen_setup.plan_and_execute()
#                 target_freq = float(f)
#                 kitchen_setup.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
#                 kitchen_setup.set_json_goal('ShakyJointPositionRevoluteOrPrismatic',
#                                             joint_name=joint,
#                                             goal=0.0,
#                                             frequency=target_freq
#                                             )
#                 r = kitchen_setup.plan_and_execute(expected_error_codes=[MoveResult.SHAKING])
#                 assert len(r.error_codes) != 0
#                 error_code = r.error_codes[0]
#                 assert error_code == MoveResult.SHAKING
#                 error_message = r.error_messages[0]
#                 freqs_str = re.findall("[0-9]+\.[0-9]+ hertz", error_message)
#                 assert any(map(lambda f_str: float(f_str[:-6]) == target_freq, freqs_str))
#
#     def test_wiggle_continuous_joint_shaking(self, kitchen_setup: PR2TestWrapper):
#         sample_period = kitchen_setup.god_map.get_data(identifier.sample_period)
#         frequency_range = kitchen_setup.god_map.get_data(identifier.frequency_range)
#         max_detectable_freq = int(1 / (2 * sample_period))
#         min_wiggle_frequency = int(frequency_range * max_detectable_freq)
#         while np.fmod(min_wiggle_frequency, 5.0) != 0.0:
#             min_wiggle_frequency += 1
#         distance_between_frequencies = 5
#
#         for continuous_joint in ['l_wrist_roll_joint', 'r_forearm_roll_joint']:  # max vel. of 1.0 and 1.0
#             for f in range(min_wiggle_frequency, max_detectable_freq, distance_between_frequencies):
#                 kitchen_setup.set_json_goal('JointPositionContinuous',
#                                             joint_name=continuous_joint,
#                                             goal=5.0,
#                                             )
#                 kitchen_setup.plan_and_execute()
#                 target_freq = float(f)
#                 kitchen_setup.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
#                 kitchen_setup.set_json_goal('ShakyJointPositionContinuous',
#                                             joint_name=continuous_joint,
#                                             goal=-5.0,
#                                             frequency=target_freq
#                                             )
#                 r = kitchen_setup.plan_and_execute(expected_error_codes=[MoveResult.SHAKING])
#                 assert len(r.error_codes) != 0
#                 error_code = r.error_codes[0]
#                 assert error_code == MoveResult.SHAKING
#                 error_message = r.error_messages[0]
#                 freqs_str = re.findall("[0-9]+\.[0-9]+ hertz", error_message)
#                 assert any(map(lambda f_str: float(f_str[:-6]) == target_freq, freqs_str))
#
#     def test_only_revolute_joint_shaking(self, kitchen_setup: PR2TestWrapper):
#         sample_period = kitchen_setup.god_map.get_data(identifier.sample_period)
#         frequency_range = kitchen_setup.god_map.get_data(identifier.frequency_range)
#         amplitude_threshold = kitchen_setup.god_map.get_data(identifier.amplitude_threshold)
#         max_detectable_freq = int(1 / (2 * sample_period))
#         min_wiggle_frequency = int(frequency_range * max_detectable_freq)
#         while np.fmod(min_wiggle_frequency, 5.0) != 0.0:
#             min_wiggle_frequency += 1
#         distance_between_frequencies = 5
#
#         for revolute_joint in ['r_wrist_flex_joint', 'head_pan_joint']:  # max vel. of 1.0 and 1.0
#             for f in range(min_wiggle_frequency, max_detectable_freq, distance_between_frequencies):
#                 target_freq = float(f)
#
#                 if f == min_wiggle_frequency:
#                     kitchen_setup.set_json_goal('JointPositionRevolute',
#                                                 joint_name=revolute_joint,
#                                                 goal=0.0,
#                                                 )
#                     kitchen_setup.plan_and_execute()
#
#                 kitchen_setup.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
#                 kitchen_setup.set_json_goal('ShakyJointPositionRevoluteOrPrismatic',
#                                             joint_name=revolute_joint,
#                                             goal=0.0,
#                                             noise_amplitude=amplitude_threshold + 0.02,
#                                             frequency=target_freq
#                                             )
#                 r = kitchen_setup.plan_and_execute(expected_error_codes=[MoveResult.SHAKING])
#                 assert len(r.error_codes) != 0
#                 error_code = r.error_codes[0]
#                 assert error_code == MoveResult.SHAKING
#                 error_message = r.error_messages[0]
#                 freqs_str = re.findall("[0-9]+\.[0-9]+ hertz", error_message)
#                 assert any(map(lambda f_str: float(f_str[:-6]) == target_freq, freqs_str))
#
#     def test_only_revolute_joint_neglectable_shaking(self, kitchen_setup: PR2TestWrapper):
#         # FIXME
#         sample_period = kitchen_setup.god_map.get_data(identifier.sample_period)
#         frequency_range = kitchen_setup.god_map.get_data(identifier.frequency_range)
#         amplitude_threshold = kitchen_setup.god_map.get_data(identifier.amplitude_threshold)
#         max_detectable_freq = int(1 / (2 * sample_period))
#         min_wiggle_frequency = int(frequency_range * max_detectable_freq)
#         while np.fmod(min_wiggle_frequency, 5.0) != 0.0:
#             min_wiggle_frequency += 1
#         distance_between_frequencies = 5
#
#         for revolute_joint in ['r_wrist_flex_joint', 'head_pan_joint']:  # max vel. of 1.0 and 0.5
#             for f in range(min_wiggle_frequency, max_detectable_freq, distance_between_frequencies):
#                 target_freq = float(f)
#                 if f == min_wiggle_frequency:
#                     kitchen_setup.set_json_goal('JointPositionRevolute',
#                                                 joint_name=revolute_joint,
#                                                 goal=0.0,
#                                                 )
#                     kitchen_setup.plan_and_execute()
#                 kitchen_setup.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
#                 kitchen_setup.set_json_goal('ShakyJointPositionRevoluteOrPrismatic',
#                                             joint_name=revolute_joint,
#                                             goal=0.0,
#                                             noise_amplitude=amplitude_threshold - 0.02,
#                                             frequency=target_freq
#                                             )
#                 r = kitchen_setup.plan_and_execute()
#                 if any(map(lambda c: c == MoveResult.SHAKING, r.error_codes)):
#                     error_message = r.error_messages[0]
#                     freqs_str = re.findall("[0-9]+\.[0-9]+ hertz", error_message)
#                     assert all(map(lambda f_str: float(f_str[:-6]) != target_freq, freqs_str))
#                 else:
#                     assert True


class TestWorldManipulation:

    def test_save_graph_pdf(self, kitchen_setup):
        kitchen_setup.world.save_graph_pdf()

    def test_dye_group(self, kitchen_setup: PR2TestWrapper):
        old_color = kitchen_setup.world.groups[kitchen_setup.robot_name].get_link('base_link').collisions[0].color
        kitchen_setup.dye_group(kitchen_setup.robot_name, (1, 0, 0, 1))
        kitchen_setup.world.groups[kitchen_setup.robot_name].get_link('base_link')
        color_robot = kitchen_setup.world.groups[kitchen_setup.robot_name].get_link('base_link').collisions[0].color
        assert color_robot.r == 1
        assert color_robot.g == 0
        assert color_robot.b == 0
        assert color_robot.a == 1
        kitchen_setup.dye_group('iai_kitchen', (0, 1, 0, 1))
        color_kitchen = kitchen_setup.world.groups['iai_kitchen'].get_link('iai_kitchen/sink_area_sink').collisions[
            0].color
        assert color_robot.r == 1
        assert color_robot.g == 0
        assert color_robot.b == 0
        assert color_robot.a == 1
        assert color_kitchen.r == 0
        assert color_kitchen.g == 1
        assert color_kitchen.b == 0
        assert color_kitchen.a == 1
        kitchen_setup.dye_group(kitchen_setup.r_gripper_group, (0, 0, 1, 1))
        color_hand = kitchen_setup.world.groups[kitchen_setup.robot_name].get_link('r_gripper_palm_link').collisions[
            0].color
        assert color_robot.r == 1
        assert color_robot.g == 0
        assert color_robot.b == 0
        assert color_robot.a == 1
        assert color_kitchen.r == 0
        assert color_kitchen.g == 1
        assert color_kitchen.b == 0
        assert color_kitchen.a == 1
        assert color_hand.r == 0
        assert color_hand.g == 0
        assert color_hand.b == 1
        assert color_hand.a == 1
        kitchen_setup.set_joint_goal(kitchen_setup.default_pose)
        kitchen_setup.plan_and_execute()
        assert color_robot.r == 1
        assert color_robot.g == 0
        assert color_robot.b == 0
        assert color_robot.a == 1
        assert color_kitchen.r == 0
        assert color_kitchen.g == 1
        assert color_kitchen.b == 0
        assert color_kitchen.a == 1
        assert color_hand.r == 0
        assert color_hand.g == 0
        assert color_hand.b == 1
        assert color_hand.a == 1
        kitchen_setup.clear_world()
        color_robot = kitchen_setup.world.groups[kitchen_setup.robot_name].get_link('base_link').collisions[0].color
        assert color_robot.r == old_color.r
        assert color_robot.g == old_color.g
        assert color_robot.b == old_color.b
        assert color_robot.a == old_color.a

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
        zero_pose.set_joint_goal(zero_pose.better_pose)
        zero_pose.plan_and_execute()

    def test_attach_remove_box(self, better_pose: PR2TestWrapper):
        pocky = 'http:muh#pocky'
        p = PoseStamped()
        p.header.frame_id = better_pose.r_tip
        p.pose.orientation.w = 1
        better_pose.add_box(pocky, size=(1, 1, 1), pose=p)
        for i in range(3):
            better_pose.update_parent_link_of_group(name=pocky, parent_link=better_pose.r_tip)
            better_pose.detach_group(pocky)
        better_pose.remove_group(pocky)

    def test_reattach_box(self, zero_pose: PR2TestWrapper):
        pocky = 'http:muh#pocky'
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
        object_name = kitchen_setup.kitchen_name
        kitchen_setup.set_kitchen_js({'sink_area_left_middle_drawer_main_joint': 0.1})
        kitchen_setup.clear_world()
        try:
            GiskardWrapper.set_object_joint_state(kitchen_setup, object_name, {})
        except KeyError:
            pass
        else:
            raise 'expected error'
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position.x = 1
        p.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0, 0, 1]))
        if kitchen_setup.is_standalone():
            js_topic = ''
            set_js_topic = ''
        else:
            js_topic = '/kitchen/joint_states'
            set_js_topic = '/kitchen/cram_joint_states'
        kitchen_setup.add_urdf(name=object_name,
                               urdf=rospy.get_param('kitchen_description'),
                               pose=p,
                               js_topic=js_topic,
                               set_js_topic=set_js_topic)
        joint_state = kitchen_setup.get_group_info(object_name).joint_state
        for i, joint_name in enumerate(joint_state.name):
            actual = joint_state.position[i]
            assert actual == 0, f'Joint {joint_name} is at {actual} instead of 0'
        kitchen_setup.set_kitchen_js({'sink_area_left_middle_drawer_main_joint': 0.1})
        kitchen_setup.remove_group(object_name)
        try:
            GiskardWrapper.set_object_joint_state(kitchen_setup, object_name, {})
        except KeyError:
            pass
        else:
            raise 'expected error'
        kitchen_setup.add_urdf(name=object_name,
                               urdf=rospy.get_param('kitchen_description'),
                               pose=p,
                               js_topic=js_topic,
                               set_js_topic=set_js_topic)
        joint_state = kitchen_setup.get_group_info(object_name).joint_state
        for i, joint_name in enumerate(joint_state.name):
            actual = joint_state.position[i]
            assert actual == 0, f'Joint {joint_name} is at {actual} instead of 0'

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
        timeout = 1
        object_name = 'muh'
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position = Point(1.2, 0, 1.6)
        p.pose.orientation = Quaternion(0.0, 0.0, 0.47942554, 0.87758256)
        zero_pose.add_box(object_name, size=(1, 1, 1), pose=p, timeout=timeout)
        zero_pose.update_parent_link_of_group(object_name, parent_link=zero_pose.r_tip, timeout=timeout)
        zero_pose.detach_group(object_name, timeout=timeout)
        zero_pose.remove_group(object_name, timeout=timeout)
        zero_pose.add_box(object_name, size=(1, 1, 1), pose=p, timeout=timeout)

    def test_attach_to_kitchen(self, kitchen_setup: PR2TestWrapper):
        object_name = 'muh'
        drawer_joint = 'sink_area_left_middle_drawer_main_joint'

        cup_pose = PoseStamped()
        cup_pose.header.frame_id = 'sink_area_left_middle_drawer_main'
        cup_pose.pose.position = Point(0.1, 0.2, -.05)
        cup_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup.add_cylinder(object_name, height=0.07, radius=0.04, pose=cup_pose,
                                   parent_link_group='iai_kitchen', parent_link='sink_area_left_middle_drawer_main')
        kitchen_setup.set_kitchen_js({drawer_joint: 0.48})
        kitchen_setup.plan_and_execute()
        kitchen_setup.detach_group(object_name)
        kitchen_setup.set_kitchen_js({drawer_joint: 0})
        kitchen_setup.plan_and_execute()

    def test_single_joint_urdf(self, zero_pose: PR2TestWrapper):
        object_name = 'spoon'
        path = resolve_ros_iris('package://giskardpy/test/spoon/urdf/spoon.urdf')
        with open(path, 'r')as f:
            urdf_str = hacky_urdf_parser_fix(f.read())
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position.x = 1
        pose.pose.orientation.w = 1
        zero_pose.add_urdf(name=object_name, urdf=urdf_str, pose=pose, parent_link='map')
        pose.pose.position.x = 1.5
        zero_pose.update_group_pose(group_name=object_name, new_pose=pose)

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
        pocky = 'http:muh#pocky'
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


class TestSelfCollisionAvoidance:

    def test_cable_guide_collision(self, zero_pose: PR2TestWrapper):
        js = {
            'head_pan_joint': 2.84,
            'head_tilt_joint': 1.
        }
        zero_pose.set_joint_goal(js)
        zero_pose.plan_and_execute()

    def test_attached_self_collision_avoid_stick(self, zero_pose: PR2TestWrapper):
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

    def test_allow_self_collision_in_arm(self, zero_pose: PR2TestWrapper):
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
        zero_pose.set_cart_goal(goal_pose=p, tip_link=zero_pose.l_tip, root_link='base_footprint')
        zero_pose.plan_and_execute()
        zero_pose.check_cpi_leq(zero_pose.get_l_gripper_links(), 0.01)
        zero_pose.check_cpi_leq(['r_forearm_link'], 0.01)
        zero_pose.check_cpi_geq(zero_pose.get_r_gripper_links(), 0.05)

    def test_avoid_self_collision_with_r_arm(self, zero_pose: PR2TestWrapper):
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
        zero_pose.check_cpi_geq(zero_pose.get_l_gripper_links(), 0.047)

    def test_avoid_self_collision_with_l_arm(self, zero_pose: PR2TestWrapper):
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
        zero_pose.set_cart_goal(goal_pose=p, tip_link=zero_pose.r_tip, root_link='base_footprint')
        zero_pose.send_goal()
        zero_pose.check_cpi_geq(zero_pose.get_r_gripper_links(), 0.048)

    def test_avoid_self_collision_specific_link(self, zero_pose: PR2TestWrapper):
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
        zero_pose.set_cart_goal(goal_pose=p, tip_link=zero_pose.r_tip, root_link='base_footprint')
        zero_pose.allow_all_collisions()
        zero_pose.register_group('forearm', root_link_name='l_forearm_link', root_link_group_name='pr2')
        zero_pose.register_group('forearm_roll', root_link_name='l_forearm_roll_link', root_link_group_name='pr2')
        zero_pose.avoid_collision(group1='forearm_roll', group2=zero_pose.r_gripper_group)
        zero_pose.allow_collision(group1='forearm', group2=zero_pose.r_gripper_group)
        zero_pose.plan_and_execute()
        zero_pose.check_cpi_geq(zero_pose.get_r_gripper_links(), 0.048)

    def test_avoid_self_collision_move_away(self, zero_pose: PR2TestWrapper):
        goal_js = {
            'r_shoulder_pan_joint': -0.07,
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
        zero_pose.set_cart_goal(goal_pose=p, tip_link=zero_pose.r_tip, root_link='base_footprint')
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
        zero_pose.send_goal(expected_error_codes=[MoveResult.SELF_COLLISION_VIOLATED])


class TestCollisionAvoidanceGoals:

    def test_handover(self, kitchen_setup: PR2TestWrapper):
        js = {
            'l_shoulder_pan_joint': 1.0252138037286773,
            'l_shoulder_lift_joint': - 0.06966848987919201,
            'l_upper_arm_roll_joint': 1.1765832782526544,
            'l_elbow_flex_joint': - 1.9323726623855864,
            'l_forearm_roll_joint': 1.3824994377973336,
            'l_wrist_flex_joint': - 1.8416233909065576,
            'l_wrist_roll_joint': 2.907373693068033,
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
        kitchen_setup.allow_collision(group1=kitchen_setup.robot_name, group2='box')
        kitchen_setup.plan_and_execute()

        kitchen_setup.update_parent_link_of_group('box', kitchen_setup.r_tip)

        r_goal2 = PoseStamped()
        r_goal2.header.frame_id = 'box'
        r_goal2.pose.position.x -= -.1
        r_goal2.pose.orientation.w = 1

        kitchen_setup.set_cart_goal(r_goal2, 'box', root_link=kitchen_setup.l_tip)
        kitchen_setup.allow_self_collision()
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
        pocky = 'muh#pocky'
        box_pose = PoseStamped()
        box_pose.header.frame_id = zero_pose.r_tip
        box_pose.pose.position = Point(0.05, 0, 0, )
        box_pose.pose.orientation = Quaternion(1, 0, 0, 0)
        zero_pose.add_box(name=pocky, size=(0.1, 0.02, 0.02), pose=box_pose, parent_link=zero_pose.r_tip,
                          parent_link_group=zero_pose.robot_name)
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.orientation.w = 1
        zero_pose.set_cart_goal(p, pocky, zero_pose.default_root)
        p = zero_pose.transform_msg(zero_pose.default_root, p)
        zero_pose.plan_and_execute()
        p2 = zero_pose.world.compute_fk_pose(zero_pose.default_root, pocky)
        compare_poses(p2.pose, p.pose)
        zero_pose.detach_group(pocky)
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.orientation.w = 1
        p.pose.position.x = -.1
        zero_pose.set_cart_goal(p, zero_pose.r_tip, zero_pose.default_root)
        zero_pose.plan_and_execute()

    def test_hard_constraints_violated(self, kitchen_setup: PR2TestWrapper):
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

    def test_avoid_collision_with_box(self, box_setup: PR2TestWrapper):
        box_setup.avoid_collision(min_distance=0.05, group1=box_setup.robot_name)
        box_setup.avoid_collision(min_distance=0.15, group1=box_setup.l_gripper_group, group2='box')
        box_setup.avoid_collision(min_distance=0.10, group1=box_setup.r_gripper_group, group2='box')
        box_setup.allow_self_collision()
        base_goal = PoseStamped()
        base_goal.header.frame_id = box_setup.default_root
        base_goal.pose.orientation.w = 1
        box_setup.move_base(base_goal)
        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.148)
        box_setup.check_cpi_geq(box_setup.get_r_gripper_links(), 0.088)

    def test_avoid_collision_drive_into_box(self, box_setup: PR2TestWrapper):
        base_goal = PoseStamped()
        base_goal.header.frame_id = box_setup.default_root
        base_goal.pose.position.x = 0.25
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0, 0, 1]))
        box_setup.teleport_base(base_goal)
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'base_footprint'
        base_goal.pose.position.x = -1
        base_goal.pose.orientation.w = 1
        box_setup.allow_self_collision()
        box_setup.set_cart_goal(goal_pose=base_goal, tip_link='base_footprint', root_link='map', weight=WEIGHT_BELOW_CA,
                                check=False)
        box_setup.plan_and_execute()
        box_setup.check_cpi_geq(['base_link'], 0.09)

    def test_avoid_collision_lower_soft_threshold(self, box_setup: PR2TestWrapper):
        base_goal = PoseStamped()
        base_goal.header.frame_id = box_setup.default_root
        base_goal.pose.position.x = 0.35
        base_goal.pose.orientation.z = 1
        box_setup.teleport_base(base_goal)
        box_setup.avoid_collision(min_distance=0.05, group1=box_setup.robot_name)
        box_setup.allow_self_collision()
        box_setup.plan_and_execute()
        box_setup.check_cpi_geq(['base_link'], 0.048)
        box_setup.check_cpi_leq(['base_link'], 0.07)

    def test_collision_override(self, box_setup: PR2TestWrapper):
        p = PoseStamped()
        p.header.frame_id = box_setup.default_root
        p.pose.position.x += 0.5
        p.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0, 0, 1]))
        box_setup.teleport_base(p)
        box_setup.allow_self_collision()
        box_setup.avoid_collision(min_distance=0.25, group1=box_setup.robot_name, group2='box')
        box_setup.plan_and_execute()
        box_setup.check_cpi_geq(['base_link'], distance_threshold=0.25, check_self=False)

    def test_ignore_all_collisions_of_links(self, box_setup: PR2TestWrapper):
        p = PoseStamped()
        p.header.frame_id = box_setup.default_root
        p.pose.position.x += 0.5
        p.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0, 0, 1]))
        box_setup.teleport_base(p)
        box_setup.check_cpi_geq(['bl_caster_l_wheel_link', 'bl_caster_r_wheel_link',
                                 'fl_caster_l_wheel_link', 'fl_caster_r_wheel_link',
                                 'br_caster_l_wheel_link', 'br_caster_r_wheel_link',
                                 'fr_caster_l_wheel_link', 'fr_caster_r_wheel_link'],
                                distance_threshold=0.25,
                                check_self=False)

    def test_avoid_collision_go_around_corner(self, fake_table_setup: PR2TestWrapper):
        r_goal = PoseStamped()
        r_goal.header.frame_id = 'map'
        r_goal.pose.position.x = 0.8
        r_goal.pose.position.y = -0.38
        r_goal.pose.position.z = 0.84
        r_goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 2, [0, 1, 0]))
        fake_table_setup.avoid_all_collisions(0.1)
        fake_table_setup.set_cart_goal(r_goal, fake_table_setup.r_tip)
        fake_table_setup.plan_and_execute()
        fake_table_setup.check_cpi_geq(fake_table_setup.get_l_gripper_links(), 0.05)
        fake_table_setup.check_cpi_leq(['r_gripper_l_finger_tip_link'], 0.04)
        fake_table_setup.check_cpi_leq(['r_gripper_r_finger_tip_link'], 0.04)

    def test_allow_collision_drive_into_box(self, box_setup: PR2TestWrapper):
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

    def test_avoid_collision_box_between_boxes(self, pocky_pose_setup: PR2TestWrapper):
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
        pocky_pose_setup.check_cpi_geq(['box'], 0.048)

    def test_avoid_collision_box_between_3_boxes(self, pocky_pose_setup: PR2TestWrapper):
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
        pocky_pose_setup.set_align_planes_goal(tip_link='box', tip_normal=x, goal_normal=x_map)
        pocky_pose_setup.set_align_planes_goal(tip_link='box', tip_normal=y, goal_normal=y_map)
        pocky_pose_setup.allow_self_collision()

        pocky_pose_setup.plan_and_execute()
        assert ('box', 'bl') not in pocky_pose_setup.collision_scene.self_collision_matrix
        pocky_pose_setup.check_cpi_geq(pocky_pose_setup.get_group_info('r_gripper').links, 0.04)

    def test_avoid_collision_box_between_cylinders(self, pocky_pose_setup: PR2TestWrapper):
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

        pocky_pose_setup.plan_and_execute()

    def test_avoid_collision_at_kitchen_corner(self, kitchen_setup: PR2TestWrapper):
        base_pose = PoseStamped()
        base_pose.header.stamp = rospy.get_rostime()
        base_pose.header.frame_id = 'map'
        base_pose.pose.position.x = 0.75
        base_pose.pose.position.y = 0.9
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 2, [0, 0, 1]))
        kitchen_setup.teleport_base(base_pose)
        base_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0, 0, 1]))
        kitchen_setup.set_joint_goal(kitchen_setup.better_pose, weight=WEIGHT_ABOVE_CA, check=False)
        kitchen_setup.set_cart_goal(goal_pose=base_pose, tip_link='base_footprint', root_link='map', check=False)
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
        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = -0.15
        p.pose.orientation.w = 1
        box_setup.set_cart_goal(goal_pose=p, tip_link=box_setup.r_tip, root_link=box_setup.default_root)
        box_setup.plan_and_execute()
        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.048)
        box_setup.check_cpi_geq([attached_link_name], 0.048)

        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.1
        p.pose.orientation.w = 1
        box_setup.set_cart_goal(goal_pose=p, tip_link=box_setup.r_tip,
                                root_link=box_setup.default_root, check=False)
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
        # TODO this creates shaking with a box
        attached_link_name = 'pocky'
        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.pose.position.x = 0.05
        p.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 0, 1, 0],
                                                                 [0, 1, 0, 0],
                                                                 [-1, 0, 0, 0],
                                                                 [0, 0, 0, 1]]))
        box_setup.add_cylinder(attached_link_name,
                               # size=(0.2, 0.04, 0.04),
                               height=0.2,
                               radius=0.04,
                               parent_link=box_setup.r_tip,
                               pose=p)
        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = -0.09
        p.pose.orientation.w = 1
        box_setup.set_cart_goal(p, box_setup.r_tip, box_setup.default_root)
        box_setup.plan_and_execute()
        box_setup.check_cpi_geq([attached_link_name], -0.003)

        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.08
        p.pose.orientation.w = 1
        box_setup.set_cart_goal(p, box_setup.r_tip, box_setup.default_root, check=False)
        box_setup.plan_and_execute()
        box_setup.check_cpi_geq([attached_link_name], -0.005)
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
        box_setup.check_cpi_geq([attached_link_name], 0.048)
        box_setup.detach_group(attached_link_name)

    def test_attached_collision_with_box(self, box_setup: PR2TestWrapper):
        attached_link_name = 'pocky'
        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.pose.position.x = 0.01
        p.pose.orientation.w = 1
        box_setup.add_box(name=attached_link_name,
                          size=(0.2, 0.04, 0.04),
                          parent_link=box_setup.r_tip,
                          pose=p)
        box_setup.plan_and_execute()
        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.048)
        box_setup.check_cpi_geq([attached_link_name], 0.048)
        box_setup.detach_group(attached_link_name)

    def test_attached_collision_allow(self, box_setup: PR2TestWrapper):
        pocky = 'http:muh#pocky'
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

    def test_get_milk_out_of_fridge(self, kitchen_setup: PR2TestWrapper):
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

        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = 'map'
        bar_axis.vector.z = 1

        bar_center = PointStamped()
        bar_center.header.frame_id = milk_pose.header.frame_id
        bar_center.point = deepcopy(milk_pose.pose.position)

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = kitchen_setup.l_tip
        tip_grasp_axis.vector.z = 1
        kitchen_setup.set_grasp_bar_goal(bar_center=bar_center,
                                         bar_axis=bar_axis,
                                         bar_length=0.12,
                                         tip_link=kitchen_setup.l_tip,
                                         tip_grasp_axis=tip_grasp_axis,
                                         root_link=kitchen_setup.default_root)

        x = Vector3Stamped()
        x.header.frame_id = kitchen_setup.l_tip
        x.vector.x = 1
        x_map = Vector3Stamped()
        x_map.header.frame_id = 'iai_kitchen/iai_fridge_door'
        x_map.vector.x = 1
        kitchen_setup.set_align_planes_goal(tip_link=kitchen_setup.l_tip,
                                            tip_normal=x,
                                            goal_normal=x_map)

        kitchen_setup.plan_and_execute()

        kitchen_setup.update_parent_link_of_group(milk_name, kitchen_setup.l_tip)
        kitchen_setup.close_l_gripper()

        # Remove Milk
        kitchen_setup.set_cart_goal(milk_pre_pose, milk_name, kitchen_setup.default_root)
        kitchen_setup.plan_and_execute()
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'base_footprint'
        base_goal.pose.orientation.w = 1
        kitchen_setup.set_joint_goal(kitchen_setup.better_pose, check=False)
        kitchen_setup.move_base(base_goal)

        # place milk back
        kitchen_setup.set_cart_goal(milk_pre_pose, milk_name, kitchen_setup.default_root)
        kitchen_setup.plan_and_execute()

        kitchen_setup.set_cart_goal(milk_pose, milk_name, kitchen_setup.default_root)
        kitchen_setup.plan_and_execute()

        kitchen_setup.open_l_gripper()

        kitchen_setup.detach_group(milk_name)

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
        cup_pose.header.stamp = rospy.get_rostime() + rospy.Duration(0.5)
        cup_pose.pose.position = Point(0.1, 0.2, -.05)
        cup_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup.add_cylinder(name=cup_name, height=0.07, radius=0.04, pose=cup_pose,
                                   parent_link='sink_area_left_middle_drawer_main')

        # spawn bowl
        bowl_pose = PoseStamped()
        bowl_pose.header.frame_id = 'iai_kitchen/sink_area_left_middle_drawer_main'
        bowl_pose.pose.position = Point(0.1, -0.2, -.05)
        bowl_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup.add_cylinder(name=bowl_name, height=0.05, radius=0.07, pose=bowl_pose,
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

        kitchen_setup.set_grasp_bar_goal(bar_center=bar_center,
                                         bar_axis=bar_axis,
                                         bar_length=0.4,
                                         tip_link=kitchen_setup.l_tip,
                                         tip_grasp_axis=tip_grasp_axis,
                                         root_link=kitchen_setup.default_root)
        x_gripper = Vector3Stamped()
        x_gripper.header.frame_id = kitchen_setup.l_tip
        x_gripper.vector.x = 1

        x_goal = Vector3Stamped()
        x_goal.header.frame_id = drawer_handle
        x_goal.vector.x = -1

        kitchen_setup.set_align_planes_goal(tip_link=kitchen_setup.l_tip,
                                            tip_normal=x_gripper,
                                            root_link=kitchen_setup.default_root,
                                            goal_normal=x_goal)
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.plan_and_execute()

        # open drawer
        kitchen_setup.set_open_container_goal(tip_link=kitchen_setup.l_tip,
                                              environment_link=drawer_handle)
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_kitchen_js({drawer_joint: 0.48})

        kitchen_setup.set_joint_goal(kitchen_setup.better_pose)
        base_pose = PoseStamped()
        base_pose.header.frame_id = 'map'
        base_pose.pose.position.y = 1
        base_pose.pose.position.x = .1
        base_pose.pose.orientation.w = 1
        kitchen_setup.move_base(base_pose)

        # grasp bowl
        l_goal = deepcopy(bowl_pose)
        l_goal.header.frame_id = 'iai_kitchen/sink_area_left_middle_drawer_main'
        l_goal.pose.position.z += .2
        l_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 1, 0, 0],
                                                                      [0, 0, -1, 0],
                                                                      [-1, 0, 0, 0],
                                                                      [0, 0, 0, 1]]))
        kitchen_setup.set_cart_goal(goal_pose=l_goal,
                                    tip_link=kitchen_setup.l_tip,
                                    root_link=kitchen_setup.default_root)
        kitchen_setup.allow_collision(kitchen_setup.l_gripper_group, bowl_name)

        # grasp cup
        r_goal = deepcopy(cup_pose)
        r_goal.header.frame_id = 'iai_kitchen/sink_area_left_middle_drawer_main'
        r_goal.pose.position.z += .2
        r_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 1, 0, 0],
                                                                      [0, 0, -1, 0],
                                                                      [-1, 0, 0, 0],
                                                                      [0, 0, 0, 1]]))
        kitchen_setup.set_avoid_joint_limits_goal(percentage=percentage)
        kitchen_setup.set_cart_goal(goal_pose=r_goal,
                                    tip_link=kitchen_setup.r_tip,
                                    root_link=kitchen_setup.default_root)
        kitchen_setup.plan_and_execute()

        l_goal.pose.position.z -= .2
        r_goal.pose.position.z -= .2
        kitchen_setup.set_cart_goal(goal_pose=l_goal,
                                    tip_link=kitchen_setup.l_tip,
                                    root_link=kitchen_setup.default_root)
        kitchen_setup.set_cart_goal(goal_pose=r_goal,
                                    tip_link=kitchen_setup.r_tip,
                                    root_link=kitchen_setup.default_root)
        kitchen_setup.set_avoid_joint_limits_goal(percentage=percentage)
        kitchen_setup.avoid_all_collisions(0.05)
        kitchen_setup.allow_collision(group1=kitchen_setup.robot_name, group2=bowl_name)
        kitchen_setup.allow_collision(group1=kitchen_setup.robot_name, group2=cup_name)
        kitchen_setup.plan_and_execute()

        kitchen_setup.update_parent_link_of_group(name=bowl_name, parent_link=kitchen_setup.l_tip)
        kitchen_setup.update_parent_link_of_group(name=cup_name, parent_link=kitchen_setup.r_tip)

        kitchen_setup.set_joint_goal(kitchen_setup.better_pose)
        kitchen_setup.plan_and_execute()
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'base_footprint'
        base_goal.pose.position.x = -.1
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        kitchen_setup.move_base(base_goal)

        # place bowl and cup
        bowl_goal = PoseStamped()
        bowl_goal.header.frame_id = 'kitchen_island_surface'
        bowl_goal.pose.position = Point(.2, 0, .05)
        bowl_goal.pose.orientation = Quaternion(0, 0, 0, 1)

        cup_goal = PoseStamped()
        cup_goal.header.frame_id = 'kitchen_island_surface'
        cup_goal.pose.position = Point(.15, 0.25, .07)
        cup_goal.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup.set_cart_goal(goal_pose=bowl_goal, tip_link=bowl_name, root_link=kitchen_setup.default_root)
        kitchen_setup.set_cart_goal(goal_pose=cup_goal, tip_link=cup_name, root_link=kitchen_setup.default_root)
        kitchen_setup.set_avoid_joint_limits_goal(percentage=percentage)
        kitchen_setup.avoid_all_collisions(0.05)
        kitchen_setup.plan_and_execute()

        kitchen_setup.detach_group(name=bowl_name)
        kitchen_setup.detach_group(name=cup_name)
        kitchen_setup.allow_collision(group1=kitchen_setup.robot_name, group2=cup_name)
        kitchen_setup.allow_collision(group1=kitchen_setup.robot_name, group2=bowl_name)
        kitchen_setup.set_joint_goal(kitchen_setup.better_pose)
        kitchen_setup.plan_and_execute()

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

    def test_tray(self, kitchen_setup: PR2TestWrapper):
        tray_name = 'tray'
        percentage = 50

        tray_pose = PoseStamped()
        tray_pose.header.frame_id = 'iai_kitchen/sink_area_surface'
        tray_pose.pose.position = Point(0.2, -0.4, 0.07)
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
        kitchen_setup.allow_collision(kitchen_setup.robot_name, tray_name)
        kitchen_setup.set_avoid_joint_limits_goal(percentage=percentage)
        # grasp tray
        kitchen_setup.plan_and_execute()

        kitchen_setup.update_parent_link_of_group(tray_name, kitchen_setup.r_tip)

        r_goal = PoseStamped()
        r_goal.header.frame_id = kitchen_setup.l_tip
        r_goal.pose.orientation.w = 1
        kitchen_setup.set_cart_goal(r_goal, kitchen_setup.l_tip, tray_name)

        tray_goal = kitchen_setup.world.compute_fk_pose('base_footprint', tray_name)
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
        kitchen_setup.set_avoid_joint_limits_goal(percentage=percentage)
        kitchen_setup.allow_collision(group1=tray_name,
                                      group2=kitchen_setup.l_gripper_group)
        # kitchen_setup.allow_self_collision()
        # drive back
        kitchen_setup.move_base(base_goal)

        r_goal = PoseStamped()
        r_goal.header.frame_id = kitchen_setup.l_tip
        r_goal.pose.orientation.w = 1
        kitchen_setup.set_cart_goal(r_goal, kitchen_setup.l_tip, tray_name)

        expected_pose = kitchen_setup.world.compute_fk_pose(tray_name, kitchen_setup.l_tip)
        expected_pose.header.stamp = rospy.Time()

        tray_goal = PoseStamped()
        tray_goal.header.frame_id = tray_name
        tray_goal.pose.position.z = .1
        tray_goal.pose.position.x = .1
        tray_goal.pose.orientation = Quaternion(*quaternion_about_axis(-1, [0, 1, 0]))
        kitchen_setup.set_avoid_joint_limits_goal(percentage=percentage)
        kitchen_setup.allow_collision(group1=tray_name,
                                      group2=kitchen_setup.l_gripper_group)
        kitchen_setup.set_cart_goal(tray_goal, tray_name, 'base_footprint')
        kitchen_setup.plan_and_execute()

    # TODO FIXME attaching and detach of urdf objects that listen to joint states

    # def test_iis(self, kitchen_setup: PR2TestWrapper):
    #     # rosrun tf static_transform_publisher 0 - 0.2 0.93 1.5707963267948966 0 0 iai_kitchen/table_area_main lid 10
    #     # rosrun tf static_transform_publisher 0 - 0.15 0 0 0 0 lid goal 10
    #     # kitchen_setup.set_joint_goal(pocky_pose)
    #     # kitchen_setup.send_and_check_goal()
    #     object_name = 'lid'
    #     pot_pose = PoseStamped()
    #     pot_pose.header.frame_id = 'lid'
    #     pot_pose.pose.position.z = -0.22
    #     # pot_pose.pose.orientation.w = 1
    #     pot_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 2, [0, 0, 1]))
    #     kitchen_setup.add_mesh(object_name,
    #                            mesh='package://cad_models/kitchen/cooking-vessels/cookingpot.dae',
    #                            pose=pot_pose)
    #
    #     base_pose = PoseStamped()
    #     base_pose.header.frame_id = 'iai_kitchen/table_area_main'
    #     base_pose.pose.position.y = -1.1
    #     base_pose.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 2, [0, 0, 1]))
    #     kitchen_setup.teleport_base(base_pose)
    #     # m = zero_pose.world.get_object(object_name).as_marker_msg()
    #     # compare_poses(m.pose, p.pose)
    #
    #     hand_goal = PoseStamped()
    #     hand_goal.header.frame_id = 'lid'
    #     hand_goal.pose.position.y = -0.15
    #     hand_goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 2, [0, 0, 1]))
    #     # kitchen_setup.allow_all_collisions()
    #     # kitchen_setup.avoid_collision([], 'kitchen', ['table_area_main'], 0.05)
    #     kitchen_setup.set_cart_goal(hand_goal, 'r_gripper_tool_frame')
    #     kitchen_setup.send_goal(goal_type=MoveGoal.PLAN_ONLY)
    #     kitchen_setup.set_cart_goal(hand_goal, 'r_gripper_tool_frame')
    #     kitchen_setup.send_goal()
    #
    #     hand_goal = PoseStamped()
    #     hand_goal.header.frame_id = 'r_gripper_tool_frame'
    #     hand_goal.pose.position.x = 0.15
    #     hand_goal.pose.orientation.w = 1
    #     # kitchen_setup.allow_all_collisions()
    #     # kitchen_setup.avoid_collision([], 'kitchen', ['table_area_main'], 0.05)
    #     kitchen_setup.set_cart_goal(hand_goal, 'r_gripper_tool_frame')
    #     kitchen_setup.allow_all_collisions()
    #     kitchen_setup.send_goal(goal_type=MoveGoal.PLAN_ONLY)
    #     kitchen_setup.set_cart_goal(hand_goal, 'r_gripper_tool_frame')
    #     kitchen_setup.allow_all_collisions()
    #     kitchen_setup.send_goal()
    #
    #     # kitchen_setup.add_cylinder('pot', size=[0.2,0.2], pose=pot_pose)


class TestInfoServices:
    def test_get_object_info(self, zero_pose: PR2TestWrapper):
        result = zero_pose.get_group_info('pr2')
        expected = {'brumbrum',
                    'head_pan_joint',
                    'head_tilt_joint',
                    'l_elbow_flex_joint',
                    'l_forearm_roll_joint',
                    'l_shoulder_lift_joint',
                    'l_shoulder_pan_joint',
                    'l_upper_arm_roll_joint',
                    'l_wrist_flex_joint',
                    'l_wrist_roll_joint',
                    'r_elbow_flex_joint',
                    'r_forearm_roll_joint',
                    'r_shoulder_lift_joint',
                    'r_shoulder_pan_joint',
                    'r_upper_arm_roll_joint',
                    'r_wrist_flex_joint',
                    'r_wrist_roll_joint',
                    'torso_lift_joint'}
        assert set(result.controlled_joints) == expected


class TestWorld:
    def test_compute_self_collision_matrix(self, world_setup: WorldTree):
        disabled_links = {world_setup.search_for_link_name('br_caster_l_wheel_link'),
                          world_setup.search_for_link_name('fr_caster_l_wheel_link')}
        reference_collision_scene = BetterPyBulletSyncer()
        reference_reasons, reference_disabled_links = reference_collision_scene.load_self_collision_matrix_from_srdf(
            'package://giskardpy/test/data/pr2_test.srdf', 'pr2')
        collision_scene: CollisionWorldSynchronizer = world_setup.god_map.get_data(identifier.collision_scene)
        actual_reasons = collision_scene.compute_self_collision_matrix('pr2',
                                                                       number_of_tries_never=500)
        assert actual_reasons == reference_reasons
        assert reference_disabled_links == disabled_links

    def test_compute_chain_reduced_to_controlled_joints(self, world_setup: WorldTree):
        r_gripper_tool_frame = world_setup.search_for_link_name('r_gripper_tool_frame')
        l_gripper_tool_frame = world_setup.search_for_link_name('l_gripper_tool_frame')
        link_a, link_b = world_setup.compute_chain_reduced_to_controlled_joints(r_gripper_tool_frame,
                                                                                l_gripper_tool_frame)
        assert link_a == world_setup.search_for_link_name('r_wrist_roll_link')
        assert link_b == world_setup.search_for_link_name('l_wrist_roll_link')

    def test_add_box(self, world_setup: WorldTree):
        box1_name = 'box1'
        box2_name = 'box2'
        box = make_world_body_box()
        pose = Pose()
        pose.orientation.w = 1
        world_setup.add_world_body(group_name=box1_name,
                                   msg=box,
                                   pose=pose,
                                   parent_link_name=world_setup.root_link_name)
        world_setup.add_world_body(group_name=box2_name,
                                   msg=box,
                                   pose=pose,
                                   parent_link_name=PrefixName('r_gripper_tool_frame', 'pr2'))
        assert box1_name in world_setup.groups
        assert world_setup.groups['pr2'].search_for_link_name(box2_name) == 'box2/box2'
        assert world_setup.groups['pr2'].search_for_link_name('box2/box2') == 'box2/box2'
        assert world_setup.search_for_link_name(box2_name) == 'box2/box2'
        assert world_setup.search_for_link_name(box1_name) == 'box1/box1'

    def test_attach_box(self, world_setup: WorldTree):
        box_name = 'boxy'
        box = make_world_body_box()
        pose = Pose()
        pose.orientation.w = 1
        world_setup.add_world_body(group_name=box_name,
                                   msg=box,
                                   pose=pose,
                                   parent_link_name=world_setup.root_link_name)
        new_parent_link_name = world_setup.search_for_link_name('r_gripper_tool_frame')
        old_fk = world_setup.compute_fk_pose(world_setup.root_link_name, box_name)

        world_setup.move_group(box_name, new_parent_link_name)

        new_fk = world_setup.compute_fk_pose(world_setup.root_link_name, box_name)
        assert world_setup.search_for_link_name(box_name) in world_setup.groups[
            world_setup.robot_names[0]].link_names_as_set
        assert world_setup.get_parent_link_of_link(world_setup.search_for_link_name(box_name)) == new_parent_link_name
        compare_poses(old_fk.pose, new_fk.pose)

        assert box_name in world_setup.groups[world_setup.robot_names[0]].groups
        assert world_setup.robot_names[0] not in world_setup.groups[world_setup.robot_names[0]].groups
        assert box_name not in world_setup.minimal_group_names

    def test_group_pr2_hand(self, world_setup: WorldTree):
        world_setup.register_group('r_hand', world_setup.search_for_link_name('r_wrist_roll_link'))
        assert set(world_setup.groups['r_hand'].joint_names) == {
            world_setup.search_for_joint_name('r_gripper_palm_joint'),
            world_setup.search_for_joint_name('r_gripper_led_joint'),
            world_setup.search_for_joint_name(
                'r_gripper_motor_accelerometer_joint'),
            world_setup.search_for_joint_name('r_gripper_tool_joint'),
            world_setup.search_for_joint_name(
                'r_gripper_motor_slider_joint'),
            world_setup.search_for_joint_name('r_gripper_l_finger_joint'),
            world_setup.search_for_joint_name('r_gripper_r_finger_joint'),
            world_setup.search_for_joint_name(
                'r_gripper_motor_screw_joint'),
            world_setup.search_for_joint_name(
                'r_gripper_l_finger_tip_joint'),
            world_setup.search_for_joint_name(
                'r_gripper_r_finger_tip_joint'),
            world_setup.search_for_joint_name('r_gripper_joint')}
        assert set(world_setup.groups['r_hand'].link_names_as_set) == {
            world_setup.search_for_link_name('r_wrist_roll_link'),
            world_setup.search_for_link_name('r_gripper_palm_link'),
            world_setup.search_for_link_name('r_gripper_led_frame'),
            world_setup.search_for_link_name(
                'r_gripper_motor_accelerometer_link'),
            world_setup.search_for_link_name(
                'r_gripper_tool_frame'),
            world_setup.search_for_link_name(
                'r_gripper_motor_slider_link'),
            world_setup.search_for_link_name(
                'r_gripper_motor_screw_link'),
            world_setup.search_for_link_name(
                'r_gripper_l_finger_link'),
            world_setup.search_for_link_name(
                'r_gripper_l_finger_tip_link'),
            world_setup.search_for_link_name(
                'r_gripper_r_finger_link'),
            world_setup.search_for_link_name(
                'r_gripper_r_finger_tip_link'),
            world_setup.search_for_link_name(
                'r_gripper_l_finger_tip_frame')}

    def test_get_chain(self, world_setup: WorldTree):
        with suppress_stderr():
            urdf = pr2_urdf()
            parsed_urdf = up.URDF.from_xml_string(hacky_urdf_parser_fix(urdf))

        root_link = 'base_footprint'
        tip_link = 'r_gripper_tool_frame'
        real = world_setup.compute_chain(root_link_name=world_setup.search_for_link_name(root_link),
                                         tip_link_name=world_setup.search_for_link_name(tip_link),
                                         add_joints=True,
                                         add_links=True,
                                         add_fixed_joints=True,
                                         add_non_controlled_joints=True)
        expected = parsed_urdf.get_chain(root_link, tip_link, True, True, True)
        assert {x.short_name for x in real} == set(expected)

    def test_get_chain2(self, world_setup: WorldTree):
        root_link = world_setup.search_for_link_name('l_gripper_tool_frame')
        tip_link = world_setup.search_for_link_name('r_gripper_tool_frame')
        try:
            world_setup.compute_chain(root_link, tip_link, True, True, True, True)
            assert False
        except ValueError:
            pass

    def test_get_chain_group(self, world_setup: WorldTree):
        root_link = world_setup.search_for_link_name('r_wrist_roll_link')
        tip_link = world_setup.search_for_link_name('r_gripper_r_finger_tip_link')
        world_setup.register_group('r_hand', root_link)
        real = world_setup.compute_chain(root_link, tip_link, True, True, True, True)
        assert real == ['pr2/r_wrist_roll_link',
                        'pr2/r_gripper_palm_joint',
                        'pr2/r_gripper_palm_link',
                        'pr2/r_gripper_r_finger_joint',
                        'pr2/r_gripper_r_finger_link',
                        'pr2/r_gripper_r_finger_tip_joint',
                        'pr2/r_gripper_r_finger_tip_link']

    def test_get_chain_group2(self, world_setup: WorldTree):
        root_link = world_setup.search_for_link_name('r_gripper_l_finger_tip_link')
        tip_link = world_setup.search_for_link_name('r_gripper_r_finger_tip_link')
        world_setup.register_group('r_hand', world_setup.search_for_link_name('r_wrist_roll_link'))
        try:
            real = world_setup.compute_chain(root_link, tip_link, True, True, True, True)
            assert False
        except ValueError:
            pass

    def test_get_split_chain(self, world_setup: WorldTree):
        root_link = world_setup.search_for_link_name('l_gripper_r_finger_tip_link')
        tip_link = world_setup.search_for_link_name('l_gripper_l_finger_tip_link')
        chain1, connection, chain2 = world_setup.compute_split_chain(root_link, tip_link, True, True, True, True)
        chain1 = [n.short_name for n in chain1]
        connection = [n.short_name for n in connection]
        chain2 = [n.short_name for n in chain2]
        assert chain1 == ['l_gripper_r_finger_tip_link', 'l_gripper_r_finger_tip_joint', 'l_gripper_r_finger_link',
                          'l_gripper_r_finger_joint']
        assert connection == ['l_gripper_palm_link']
        assert chain2 == ['l_gripper_l_finger_joint', 'l_gripper_l_finger_link', 'l_gripper_l_finger_tip_joint',
                          'l_gripper_l_finger_tip_link']

    def test_get_split_chain_group(self, world_setup: WorldTree):
        root_link = world_setup.search_for_link_name('r_gripper_l_finger_tip_link')
        tip_link = world_setup.search_for_link_name('r_gripper_r_finger_tip_link')
        world_setup.register_group('r_hand', world_setup.search_for_link_name('r_wrist_roll_link'))
        chain1, connection, chain2 = world_setup.compute_split_chain(root_link, tip_link,
                                                                     True, True, True, True)
        assert chain1 == ['pr2/r_gripper_l_finger_tip_link',
                          'pr2/r_gripper_l_finger_tip_joint',
                          'pr2/r_gripper_l_finger_link',
                          'pr2/r_gripper_l_finger_joint']
        assert connection == ['pr2/r_gripper_palm_link']
        assert chain2 == ['pr2/r_gripper_r_finger_joint',
                          'pr2/r_gripper_r_finger_link',
                          'pr2/r_gripper_r_finger_tip_joint',
                          'pr2/r_gripper_r_finger_tip_link']

    def test_get_joint_limits2(self, world_setup: WorldTree):
        lower_limit, upper_limit = world_setup.get_joint_position_limits(
            world_setup.search_for_joint_name('l_shoulder_pan_joint'))
        assert lower_limit == -0.564601836603
        assert upper_limit == 2.1353981634

    def test_search_branch(self, world_setup: WorldTree):
        result = world_setup.search_branch(world_setup.search_for_link_name('odom_combined'),
                                           stop_at_joint_when=lambda _: False,
                                           stop_at_link_when=lambda _: False)
        assert result == ([], [])
        result = world_setup.search_branch(world_setup.search_for_link_name('odom_combined'),
                                           stop_at_joint_when=world_setup.is_joint_controlled,
                                           stop_at_link_when=lambda _: False,
                                           collect_link_when=world_setup.has_link_collisions)
        assert result == ([], [])
        result = world_setup.search_branch(world_setup.search_for_link_name('base_footprint'),
                                           stop_at_joint_when=world_setup.is_joint_controlled,
                                           collect_link_when=world_setup.has_link_collisions)
        assert set(result[0]) == {'pr2/base_bellow_link',
                                  'pr2/fl_caster_l_wheel_link',
                                  'pr2/fl_caster_r_wheel_link',
                                  'pr2/fl_caster_rotation_link',
                                  'pr2/fr_caster_l_wheel_link',
                                  'pr2/fr_caster_r_wheel_link',
                                  'pr2/fr_caster_rotation_link',
                                  'pr2/bl_caster_l_wheel_link',
                                  'pr2/bl_caster_r_wheel_link',
                                  'pr2/bl_caster_rotation_link',
                                  'pr2/br_caster_l_wheel_link',
                                  'pr2/br_caster_r_wheel_link',
                                  'pr2/br_caster_rotation_link',
                                  'pr2/base_link'}
        result = world_setup.search_branch(world_setup.search_for_link_name('l_elbow_flex_link'),
                                           collect_joint_when=world_setup.is_joint_fixed)
        assert set(result[0]) == set()
        assert set(result[1]) == {'pr2/l_force_torque_adapter_joint',
                                  'pr2/l_force_torque_joint',
                                  'pr2/l_forearm_cam_frame_joint',
                                  'pr2/l_forearm_cam_optical_frame_joint',
                                  'pr2/l_forearm_joint',
                                  'pr2/l_gripper_led_joint',
                                  'pr2/l_gripper_motor_accelerometer_joint',
                                  'pr2/l_gripper_palm_joint',
                                  'pr2/l_gripper_tool_joint'}
        links, joints = world_setup.search_branch(world_setup.search_for_link_name('r_wrist_roll_link'),
                                                  stop_at_joint_when=world_setup.is_joint_controlled,
                                                  collect_link_when=world_setup.has_link_collisions,
                                                  collect_joint_when=lambda _: True)
        assert set(links) == {'pr2/r_gripper_l_finger_tip_link',
                              'pr2/r_gripper_l_finger_link',
                              'pr2/r_gripper_r_finger_tip_link',
                              'pr2/r_gripper_r_finger_link',
                              'pr2/r_gripper_palm_link',
                              'pr2/r_wrist_roll_link'}
        assert set(joints) == {'pr2/r_gripper_palm_joint',
                               'pr2/r_gripper_led_joint',
                               'pr2/r_gripper_motor_accelerometer_joint',
                               'pr2/r_gripper_tool_joint',
                               'pr2/r_gripper_motor_slider_joint',
                               'pr2/r_gripper_motor_screw_joint',
                               'pr2/r_gripper_l_finger_joint',
                               'pr2/r_gripper_l_finger_tip_joint',
                               'pr2/r_gripper_r_finger_joint',
                               'pr2/r_gripper_r_finger_tip_joint',
                               'pr2/r_gripper_joint'}
        links, joints = world_setup.search_branch(world_setup.search_for_link_name('br_caster_l_wheel_link'),
                                                  collect_link_when=lambda _: True,
                                                  collect_joint_when=lambda _: True)
        assert links == ['pr2/br_caster_l_wheel_link']
        assert joints == []

    # def test_get_siblings_with_collisions(self, world_setup: WorldTree):
    #     # FIXME
    #     result = world_setup.get_siblings_with_collisions(world_setup.search_for_joint_name('brumbrum'))
    #     assert result == []
    #     result = world_setup.get_siblings_with_collisions(world_setup.search_for_joint_name('l_elbow_flex_joint'))
    #     assert set(result) == {'pr2/l_upper_arm_roll_link', 'pr2/l_upper_arm_link'}
    #     result = world_setup.get_siblings_with_collisions(world_setup.search_for_joint_name('r_wrist_roll_joint'))
    #     assert result == ['pr2/r_wrist_flex_link']
    #     result = world_setup.get_siblings_with_collisions(world_setup.search_for_joint_name('br_caster_l_wheel_joint'))
    #     assert set(result) == {'pr2/base_bellow_link',
    #                            'pr2/fl_caster_l_wheel_link',
    #                            'pr2/fl_caster_r_wheel_link',
    #                            'pr2/fl_caster_rotation_link',
    #                            'pr2/fr_caster_l_wheel_link',
    #                            'pr2/fr_caster_r_wheel_link',
    #                            'pr2/fr_caster_rotation_link',
    #                            'pr2/bl_caster_l_wheel_link',
    #                            'pr2/bl_caster_r_wheel_link',
    #                            'pr2/bl_caster_rotation_link',
    #                            'pr2/br_caster_r_wheel_link',
    #                            'pr2/br_caster_rotation_link',
    #                            'pr2/base_link'}

    def test_get_controlled_parent_joint_of_link(self, world_setup: WorldTree):
        with pytest.raises(KeyError) as e_info:
            world_setup.get_controlled_parent_joint_of_link(world_setup.search_for_link_name('odom_combined'))
        assert world_setup.get_controlled_parent_joint_of_link(
            world_setup.search_for_link_name('base_footprint')) == 'pr2/brumbrum'

    def test_get_parent_joint_of_joint(self, world_setup: WorldTree):
        # TODO shouldn't this return a not found error?
        with pytest.raises(KeyError) as e_info:
            world_setup.get_controlled_parent_joint_of_joint('pr2/brumbrum')
        with pytest.raises(KeyError) as e_info:
            world_setup.search_for_parent_joint(world_setup.search_for_joint_name('r_wrist_roll_joint'),
                                                stop_when=lambda x: False)
        assert world_setup.get_controlled_parent_joint_of_joint(
            world_setup.search_for_joint_name('r_torso_lift_side_plate_joint')) == 'pr2/torso_lift_joint'
        assert world_setup.get_controlled_parent_joint_of_joint(
            world_setup.search_for_joint_name('torso_lift_joint')) == 'pr2/brumbrum'

    def test_possible_collision_combinations(self, world_setup: WorldTree):
        result = world_setup.groups[world_setup.robot_names[0]].possible_collision_combinations()
        reference = {world_setup.sort_links(link_a, link_b) for link_a, link_b in
                     combinations(world_setup.groups[world_setup.robot_names[0]].link_names_with_collisions, 2) if
                     not world_setup.are_linked(link_a, link_b)}
        assert result == reference

    def test_compute_chain_reduced_to_controlled_joints2(self, world_setup: WorldTree):
        link_a, link_b = world_setup.compute_chain_reduced_to_controlled_joints(
            world_setup.search_for_link_name('l_upper_arm_link'),
            world_setup.search_for_link_name('r_upper_arm_link'))
        assert link_a == 'pr2/l_upper_arm_roll_link'
        assert link_b == 'pr2/r_upper_arm_roll_link'

    def test_compute_chain_reduced_to_controlled_joints3(self, world_setup: WorldTree):
        with pytest.raises(KeyError):
            world_setup.compute_chain_reduced_to_controlled_joints(
                world_setup.search_for_link_name('l_wrist_roll_link'),
                world_setup.search_for_link_name('l_gripper_r_finger_link'))


class TestBenchmark:
    qp_solvers = [
        SupportedQPSolver.qpSWIFT,
        SupportedQPSolver.gurobi,
        SupportedQPSolver.qpalm
    ]

    def test_joint_goal_torso_lift_joint(self, zero_pose: PR2TestWrapper):
        horizons = [1, 7, 9, 21, 31, 41, 51]
        # horizons = [1]
        for qp_solver in self.qp_solvers:
            for h in horizons:
                js = {'torso_lift_joint': 1}
                zero_pose.set_prediction_horizon(h)
                zero_pose.set_json_goal('SetQPSolver', qp_solver_id=qp_solver)
                zero_pose.set_joint_goal(js, check=False)
                zero_pose.allow_all_collisions()
                zero_pose.plan_and_execute()

                zero_pose.set_seed_configuration(zero_pose.default_pose)
                zero_pose.allow_all_collisions()
                zero_pose.reset_base()

    def test_joint_goal2(self, zero_pose: PR2TestWrapper):
        horizons = [1, 7, 9, 21, 31, 41]
        # horizons = [1, 7, 9, 21]
        # horizons = [9]
        for qp_solver in self.qp_solvers:
            for h in horizons:
                zero_pose.set_prediction_horizon(h)
                zero_pose.set_json_goal('SetQPSolver', qp_solver_id=qp_solver)
                zero_pose.set_joint_goal(zero_pose.better_pose, check=False)
                zero_pose.allow_all_collisions()
                zero_pose.plan_and_execute()

                zero_pose.set_seed_configuration(zero_pose.default_pose)
                zero_pose.allow_all_collisions()
                zero_pose.reset_base()

    def test_cart_goal_2eef2(self, zero_pose: PR2TestWrapper):
        horizons = [1, 7, 9, 11, 13, 21]
        # horizons = [9]
        for qp_solver in self.qp_solvers:
            for h in horizons:
                zero_pose.set_prediction_horizon(h)
                zero_pose.set_json_goal('SetQPSolver', qp_solver_id=qp_solver)
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
                zero_pose.plan_and_execute()

                zero_pose.set_seed_configuration(zero_pose.default_pose)
                zero_pose.allow_all_collisions()
                zero_pose.reset_base()

    def test_avoid_collision_go_around_corner(self, fake_table_setup: PR2TestWrapper):
        horizons = [1, 7, 9, 13, 21, 31]
        # horizons = [7]
        for qp_solver in self.qp_solvers:
            for h in horizons:
                fake_table_setup.set_prediction_horizon(h)
                fake_table_setup.set_json_goal('SetQPSolver', qp_solver_id=qp_solver)
                r_goal = PoseStamped()
                r_goal.header.frame_id = 'map'
                r_goal.pose.position.x = 0.8
                r_goal.pose.position.y = -0.38
                r_goal.pose.position.z = 0.84
                r_goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 2, [0, 1, 0]))
                fake_table_setup.avoid_all_collisions(0.1)
                fake_table_setup.set_cart_goal(r_goal, fake_table_setup.r_tip, check=False)
                fake_table_setup.plan_and_execute()

                fake_table_setup.set_seed_configuration(pocky_pose)
                fake_table_setup.allow_all_collisions()
                fake_table_setup.reset_base()

# kernprof -lv py.test -s test/test_integration_pr2.py
# time: [1-9][1-9]*.[1-9]* s
# import pytest
# pytest.main(['-s', __file__ + '::TestJointGoals::test_joint_goal2'])
# pytest.main(['-s', __file__ + '::TestConstraints::test_open_dishwasher_apartment'])
# pytest.main(['-s', __file__ + '::TestCollisionAvoidanceGoals::test_bowl_and_cup'])
# pytest.main(['-s', __file__ + '::TestCollisionAvoidanceGoals::test_avoid_collision_go_around_corner'])
# pytest.main(['-s', __file__ + '::TestCollisionAvoidanceGoals::test_avoid_collision_box_between_boxes'])
# pytest.main(['-s', __file__ + '::TestCollisionAvoidanceGoals::test_avoid_self_collision'])
# pytest.main(['-s', __file__ + '::TestCollisionAvoidanceGoals::test_avoid_collision_at_kitchen_corner'])
# pytest.main(['-s', __file__ + '::TestWayPoints::test_waypoints2'])
# pytest.main(['-s', __file__ + '::TestCartGoals::test_cart_goal_2eef2'])
# pytest.main(['-s', __file__ + '::TestCartGoals::test_keep_position3'])
# pytest.main(['-s', __file__ + '::TestWorld::test_compute_self_collision_matrix'])
