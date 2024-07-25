from __future__ import division

from itertools import combinations

import urdf_parser_py.urdf as up
from copy import deepcopy
from typing import Optional

import numpy as np
import pytest
import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Vector3Stamped, PointStamped, QuaternionStamped, Pose, \
    Vector3
from numpy import pi
from shape_msgs.msg import SolidPrimitive
from tf.transformations import quaternion_from_matrix, quaternion_about_axis

import giskardpy.utils.tfwrapper as tf
from giskard_msgs.msg import WorldBody, CollisionEntry, WorldGoal, GiskardError
from giskardpy.configs.behavior_tree_config import StandAloneBTConfig
from giskardpy.configs.giskard import Giskard
from giskardpy.configs.other_robots.kevin import KevinCollisionAvoidanceConfig, KevinStandaloneInterface
from giskardpy.configs.world_config import WorldWithDiffDriveRobot
from giskardpy.configs.qp_controller_config import SupportedQPSolver, QPControllerConfig
from giskardpy.goals.cartesian_goals import RelativePositionSequence
from giskardpy.goals.caster import Circle, Wave
from giskardpy.goals.collision_avoidance import CollisionAvoidanceHint
from giskardpy.goals.goals_tests import DebugGoal
from giskardpy.goals.joint_goals import JointVelocityLimit
from giskardpy.goals.set_prediction_horizon import SetQPSolver
from giskardpy.goals.tracebot import InsertCylinder
from giskardpy.god_map import god_map
from giskardpy.model.better_pybullet_syncer import BetterPyBulletSyncer
from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer
from giskardpy.model.utils import make_world_body_box, hacky_urdf_parser_fix
from giskardpy.model.world import WorldTree
from giskardpy.data_types import PrefixName
from giskardpy.tasks.task import WEIGHT_BELOW_CA, WEIGHT_ABOVE_CA, WEIGHT_COLLISION_AVOIDANCE
from giskardpy.python_interface.old_python_interface import OldGiskardWrapper
from giskardpy.utils.utils import launch_launchfile, suppress_stderr, resolve_ros_iris
from giskardpy.utils.math import compare_points
from utils_for_tests import compare_poses, publish_marker_vector, \
    GiskardTestWrapper, pr2_urdf
from giskardpy.goals.manipulability_goals import MaxManipulability


class KevinTestWrapper(GiskardTestWrapper):
    default_pose = {}
    better_pose = {}

    def __init__(self, giskard: Optional[Giskard] = None):
        if giskard is None:
            giskard = Giskard(world_config=WorldWithDiffDriveRobot(),
                              collision_avoidance_config=KevinCollisionAvoidanceConfig(),
                              robot_interface_config=KevinStandaloneInterface(),
                              behavior_tree_config=StandAloneBTConfig(simulation_max_hz=20, publish_tf=True))
        super().__init__(giskard)


@pytest.fixture(scope='module')
def giskard(request, ros):
    launch_launchfile('package://kevin_robot_description/launch/kevin_robot_state.launch')
    c = KevinTestWrapper()
    request.addfinalizer(c.tear_down)
    return c


class TestJointGoals:
    def test_joint_goal(self, zero_pose: KevinTestWrapper):
        zero_pose.set_joint_goal({'robot_arm_gripper_joint': 0.066})  # open
        # zero_pose.set_joint_goal({'robot_arm_gripper_joint': 0.035}) #close
        zero_pose.allow_all_collisions()
        zero_pose.execute()


class TestCartGoals:
    def test_grasping(self, zero_pose: KevinTestWrapper):
        box_pose = PoseStamped()
        box_pose.header.frame_id = 'map'
        box_pose.pose.orientation.w = 1
        box_pose.pose.position = Point(2, 1, 0.8)
        zero_pose.add_box('box', (0.1, 0.05, 0.1), box_pose, 'map')

        goal_pose = box_pose
        goal_pose.pose.position.x -= 0.2
        zero_pose.motion_goals.add_cartesian_pose(goal_pose, 'robot_arm_tool_link', 'map')
        zero_pose.set_joint_goal({'robot_arm_gripper_joint': 0.066})
        zero_pose.execute()

        goal_pose.pose.position.x += 0.1
        zero_pose.motion_goals.add_cartesian_pose(goal_pose, 'robot_arm_tool_link', 'map')
        zero_pose.set_joint_goal({'robot_arm_gripper_joint': 0.066})

        zero_pose.execute()

    def test_scenario(self, zero_pose: KevinTestWrapper):
        # table_pose = PoseStamped()
        # table_pose.header.frame_id = 'map'
        # table_pose.pose.orientation.w = 1
        # table_pose.pose.position = Point(2, 0, 0.8)
        # zero_pose.add_box('table', (0.6, 1.2, 0.01), table_pose, 'map')
        #
        sbs_pose = PoseStamped()
        sbs_pose.header.frame_id = 'map'
        sbs_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[1, 0, 0, 0],
                                                                        [0, 0, -1, 0],
                                                                        [0, 1, 0, 0],
                                                                        [0, 0, 0, 1]]))
        sbs_pose.pose.position = Point(1.8, 0, 0.81)
        zero_pose.add_mesh_to_world(name='sbs', mesh='package://giskardpy/test/data/Falcon6erSBS.STL', pose=sbs_pose,
                                    scale=(0.001, 0.001, 0.001))

        cold_pose = PoseStamped()
        cold_pose.header.frame_id = 'map'
        cold_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[1, 0, 0, 0],
                                                                         [0, 0, -1, 0],
                                                                         [0, 1, 0, 0],
                                                                         [0, 0, 0, 1]]))
        cold_pose.pose.position = Point(1.8, 0.5, 0.81)
        zero_pose.add_mesh_to_world(name='coldplate', mesh='package://giskardpy/test/data/coldplate.stl',
                                    pose=cold_pose,
                                    scale=(0.001, 0.001, 0.001))
        #
        # zero_pose.set_joint_goal({'robot_arm_gripper_joint': 0.066})
        #
        # base_goal = PoseStamped()
        # base_goal.header.frame_id = 'map'
        # base_goal.pose.position = Point(1, 0, 0)
        # base_goal.pose.orientation.w = 1
        # zero_pose.set_diff_drive_base_goal(goal_pose=base_goal, tip_link='robot_base_footprint', root_link='map')
        #
        # hotel_pose = PoseStamped()
        # hotel_pose.header.frame_id = 'robot_hotel_nest_17_link'
        # hotel_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[-1, 0, 0, 0],
        #                                                                   [0, -1, 0, 0],
        #                                                                   [0, 0, 1, 0],
        #                                                                   [0, 0, 0, 1]]))
        # hotel_pose.pose.position.x = 0.4
        # zero_pose.motion_goals.add_cartesian_pose(hotel_pose, 'robot_arm_tool_link', 'robot_arm_base_link')
        #
        # zero_pose.allow_all_collisions()
        # zero_pose.execute()
        #
        # goal_pose = sbs_pose
        # goal_pose.pose.orientation.w = 1
        # goal_pose.pose.orientation.x = 0
        # goal_pose.pose.orientation.y = 0
        # goal_pose.pose.orientation.z = 0
        # goal_pose.pose.position.y -= 0.043
        # goal_pose.pose.position.z += 0.2
        # zero_pose.motion_goals.add_cartesian_pose(goal_pose, 'robot_arm_tool_link', 'map')
        # zero_pose.allow_all_collisions()
        # zero_pose.execute()
        # goal_pose.pose.position.z -= 0.19
        # zero_pose.motion_goals.add_cartesian_pose(goal_pose, 'robot_arm_tool_link', 'map')
        # zero_pose.allow_all_collisions()
        # zero_pose.execute()
        #
        # zero_pose.set_joint_goal({'robot_arm_gripper_joint': 0.049})  # close
        # zero_pose.allow_all_collisions()
        # zero_pose.execute()
        #
        # zero_pose.update_parent_link_of_group(name='sbs', parent_link='robot_arm_tool_link')
        #
        # goal_pose.pose.position.z += 0.1
        # zero_pose.motion_goals.add_cartesian_pose(goal_pose, 'robot_arm_tool_link', 'map')
        # zero_pose.allow_all_collisions()
        # zero_pose.execute()
        #
        # hotel_pose = PoseStamped()
        # hotel_pose.header.frame_id = 'robot_hotel_nest_17_link'
        # hotel_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[-1, 0, 0, 0],
        #                                                                   [0, -1, 0, 0],
        #                                                                   [0, 0, 1, 0],
        #                                                                   [0, 0, 0, 1]]))
        # hotel_pose.pose.position.x = 0.4
        # zero_pose.motion_goals.add_cartesian_pose(hotel_pose, 'robot_arm_tool_link', 'robot_arm_base_link')
        # zero_pose.allow_all_collisions()
        # zero_pose.execute()
        #
        # base_goal = PoseStamped()
        # base_goal.header.frame_id = 'map'
        # base_goal.pose.position = Point(0.8, 0.4, 0)
        # base_goal.pose.orientation.w = 1
        # zero_pose.set_diff_drive_base_goal(goal_pose=base_goal, tip_link='robot_base_footprint', root_link='map')
        # zero_pose.allow_all_collisions()
        # zero_pose.execute()
        #
        # goal_pose2 = cold_pose
        # goal_pose2.pose.orientation.w = 1
        # goal_pose2.pose.orientation.x = 0
        # goal_pose2.pose.orientation.y = 0
        # goal_pose2.pose.orientation.z = 0
        # goal_pose2.pose.position.x -= 0.05
        # goal_pose2.pose.position.z += 0.2
        # zero_pose.motion_goals.add_cartesian_pose(goal_pose2, 'robot_arm_tool_link', 'map')
        # zero_pose.allow_all_collisions()
        # zero_pose.execute()
        #
        # goal_pose2.pose.position.x -= 0.01
        # goal_pose2.pose.position.z -= 0.12
        # zero_pose.motion_goals.add_cartesian_pose(goal_pose2, 'robot_arm_tool_link', 'map')
        # zero_pose.allow_all_collisions()
        # zero_pose.execute()
        #
        # zero_pose.update_parent_link_of_group(name='sbs', parent_link='map')
        # zero_pose.set_joint_goal({'robot_arm_gripper_joint': 0.066})
        # goal_pose2.pose.position.z += 0.2
        # zero_pose.motion_goals.add_cartesian_pose(goal_pose2, 'robot_arm_tool_link', 'map')
        # zero_pose.allow_all_collisions()
        # zero_pose.execute()