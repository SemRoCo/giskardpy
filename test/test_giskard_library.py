import gc
import time
import traceback
from collections import defaultdict
from datetime import datetime
from itertools import combinations, chain
from typing import Dict, List, Callable, Tuple, Optional

import giskardpy.casadi_wrapper as cas
import numpy as np
import pandas as pd
import pytest
import urdf_parser_py.urdf as up
from giskardpy.data_types.data_types import PrefixName, Derivatives, ColorRGBA
from giskardpy.data_types.exceptions import EmptyProblemException
from giskardpy.god_map import god_map
from giskardpy.model.better_pybullet_syncer import BetterPyBulletSyncer
from giskardpy.model.collision_avoidance_config import DefaultCollisionAvoidanceConfig
from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer
from giskardpy.model.joints import OmniDrive, PrismaticJoint
from giskardpy.model.links import Link, BoxGeometry
from giskardpy.model.trajectory import Trajectory
from giskardpy.model.utils import hacky_urdf_parser_fix
from giskardpy.model.world import WorldTree
from giskardpy.model.world_config import EmptyWorld, WorldWithOmniDriveRobot
from giskardpy.motion_statechart.monitors.cartesian_monitors import PoseReached
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from giskardpy.qp.constraint import EqualityConstraint, InequalityConstraint, DerivativeInequalityConstraint
from giskardpy.qp.qp_controller import QPController
from giskardpy.qp.qp_controller import QPFormulation
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.qp.qp_solver_ids import SupportedQPSolver
from giskardpy.symbol_manager import symbol_manager
from giskardpy.user_interface import GiskardWrapper
from giskardpy.utils.utils import suppress_stderr
from giskardpy.model.collision_avoidance_config import CollisionAvoidanceConfig
from giskardpy.model.collision_world_syncer import CollisionCheckerLib
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose, CartesianPosition, \
    CartesianPositionVelocityTarget
from giskardpy.motion_statechart.tasks.joint_tasks import JointVelocity
from giskardpy.motion_statechart.tasks.task import WEIGHT_BELOW_CA, WEIGHT_COLLISION_AVOIDANCE
from giskardpy.qp.constraint import DerivativeEqualityConstraint
from giskardpy.utils.math import limit
from utils_for_tests import pr2_urdf


class PR2CollisionAvoidance(CollisionAvoidanceConfig):
    def __init__(self, drive_joint_name: str = 'brumbrum',
                 collision_checker: CollisionCheckerLib = CollisionCheckerLib.bpb):
        super().__init__(collision_checker=collision_checker)
        self.drive_joint_name = drive_joint_name

    def setup(self):
        self.load_self_collision_matrix('self_collision_matrices/iai/pr2.srdf')
        self.set_default_external_collision_avoidance(soft_threshold=0.1,
                                                      hard_threshold=0.0)
        for joint_name in ['r_wrist_roll_joint', 'l_wrist_roll_joint']:
            self.overwrite_external_collision_avoidance(joint_name,
                                                        number_of_repeller=4,
                                                        soft_threshold=0.05,
                                                        hard_threshold=0.0,
                                                        max_velocity=0.2)
        for joint_name in ['r_wrist_flex_joint', 'l_wrist_flex_joint']:
            self.overwrite_external_collision_avoidance(joint_name,
                                                        number_of_repeller=2,
                                                        soft_threshold=0.05,
                                                        hard_threshold=0.0,
                                                        max_velocity=0.2)
        for joint_name in ['r_elbow_flex_joint', 'l_elbow_flex_joint']:
            self.overwrite_external_collision_avoidance(joint_name,
                                                        soft_threshold=0.05,
                                                        hard_threshold=0.0)
        for joint_name in ['r_forearm_roll_joint', 'l_forearm_roll_joint']:
            self.overwrite_external_collision_avoidance(joint_name,
                                                        soft_threshold=0.025,
                                                        hard_threshold=0.0)
        self.fix_joints_for_collision_avoidance([
            'r_gripper_l_finger_joint',
            'l_gripper_l_finger_joint'
        ])
        self.overwrite_external_collision_avoidance(self.drive_joint_name,
                                                    number_of_repeller=2,
                                                    soft_threshold=0.2,
                                                    hard_threshold=0.1)


try:
    import rospy
    from giskardpy_ros1.ros1.ros_msg_visualization import ROSMsgVisualization

    rospy.init_node('tests')
    vis = ROSMsgVisualization(tf_frame='map')
    rospy.sleep(1)
except ImportError as e:
    pass

try:
    from giskardpy_ros.ros2 import rospy

    rospy.init_node('giskard')
except ImportError as e:
    pass


def visualize():
    god_map.world.notify_state_change()
    god_map.collision_scene.sync()
    # vis.publish_markers()


@pytest.fixture()
def empty_world() -> WorldTree:
    config = EmptyWorld()
    config.setup()
    return config.world


@pytest.fixture()
def fixed_box_world() -> WorldTree:
    class WorldWithFixedBox(EmptyWorld):
        box_name = PrefixName('box')
        joint_name = PrefixName('box_joint')

        def setup(self) -> None:
            super().setup()
            with self.world.modify_world():
                box = Link(self.box_name)
                box_geometry = BoxGeometry(1, 1, 1, color=ColorRGBA(1, 0, 0, 1))
                box.collisions.append(box_geometry)
                box.visuals.append(box_geometry)
                self.world.add_link(box)
                self.world.add_fixed_joint(parent_link=self.world.root_link, child_link=box, joint_name=self.joint_name)

    config = WorldWithFixedBox()
    collision_avoidance = DefaultCollisionAvoidanceConfig()
    config.setup()
    collision_avoidance.setup()
    return config.world


@pytest.fixture()
def box_world_prismatic() -> WorldTree:
    class WorldWithPrismaticBox(EmptyWorld):
        box_name = PrefixName('box')
        joint_name = PrefixName('box_joint')

        def setup(self) -> None:
            super().setup()
            with self.world.modify_world():
                box = Link(self.box_name)
                box_geometry = BoxGeometry(1, 1, 1, color=ColorRGBA(1, 0, 0, 1))
                box.collisions.append(box_geometry)
                box.visuals.append(box_geometry)
                self.world.add_link(box)
                joint = PrismaticJoint(name=self.joint_name,
                                       free_variable_name=self.joint_name,
                                       parent_link_name=self.world.root_link_name,
                                       child_link_name=self.box_name,
                                       axis=(1, 0, 0),
                                       lower_limits={Derivatives.position: -1,
                                                     Derivatives.velocity: -1,
                                                     Derivatives.acceleration: -np.inf,
                                                     Derivatives.jerk: -30},
                                       upper_limits={Derivatives.position: 1,
                                                     Derivatives.velocity: 1,
                                                     Derivatives.acceleration: np.inf,
                                                     Derivatives.jerk: 30})
                self.world.add_joint(joint)

    config = WorldWithPrismaticBox()
    collision_avoidance = DefaultCollisionAvoidanceConfig()
    config.setup()
    collision_avoidance.setup()
    assert config.joint_name in config.world.joints
    assert config.box_name in config.world.links
    return config.world


@pytest.fixture()
def box_world():
    class WorldWithOmniBox(EmptyWorld):
        box_name = PrefixName('box')
        joint_name = PrefixName('box_joint')

        def setup(self) -> None:
            super().setup()
            with self.world.modify_world():
                box = Link(self.box_name)
                box_geometry = BoxGeometry(1, 1, 1, color=ColorRGBA(1, 0, 0, 1))
                box.collisions.append(box_geometry)
                box.visuals.append(box_geometry)
                self.world.add_link(box)
                joint = OmniDrive(name=self.joint_name,
                                  parent_link_name=self.world.root_link_name,
                                  child_link_name=self.box_name,
                                  translation_limits={Derivatives.velocity: 1,
                                                      Derivatives.acceleration: np.inf,
                                                      Derivatives.jerk: 30},
                                  rotation_limits={Derivatives.velocity: 1,
                                                   Derivatives.acceleration: np.inf,
                                                   Derivatives.jerk: 30})
                self.world.add_joint(joint)

    config = WorldWithOmniBox()
    collision_avoidance = DefaultCollisionAvoidanceConfig()
    config.setup()
    collision_avoidance.setup()
    assert config.joint_name in config.world.joints
    assert config.box_name in config.world.links
    return config.world


@pytest.fixture()
def simple_two_arm_world() -> WorldTree:
    config = WorldWithOmniDriveRobot()
    urdf = open('urdfs/simple_two_arm_robot.urdf', 'r').read()
    config.setup(urdf, 'muh')
    config.world.register_controlled_joints(config.world.movable_joint_names)
    collision_avoidance = DefaultCollisionAvoidanceConfig()
    collision_avoidance.setup()
    return config.world


@pytest.fixture()
def pr2_world() -> WorldTree:
    urdf = open('urdfs/pr2.urdf', 'r').read()
    config = WorldWithOmniDriveRobot(urdf=urdf)
    with config.world.modify_world():
        config.setup()
    config.world.register_controlled_joints(config.world.movable_joint_names)
    return config.world


@pytest.fixture()
def giskard_pr2() -> GiskardWrapper:
    urdf = open('urdfs/pr2.urdf', 'r').read()
    giskard = GiskardWrapper(world_config=WorldWithOmniDriveRobot(urdf=urdf),
                             collision_avoidance_config=PR2CollisionAvoidance(),
                             qp_controller_config=QPControllerConfig())
    return giskard


import yaml


class TestWorld:
    def test_empty_world(self, empty_world: WorldTree):
        assert len(empty_world.joints) == 0

    def test_fixed_box_world(self, fixed_box_world: WorldTree):
        assert len(fixed_box_world.joints) == 1
        assert len(fixed_box_world.links) == 2
        visualize()

    def test_simple_two_arm_robot(self, simple_two_arm_world: WorldTree):
        simple_two_arm_world.state[PrefixName('prismatic_joint', 'muh')].position = 0.4
        simple_two_arm_world.state[PrefixName('r_joint_1', 'muh')].position = 0.2
        simple_two_arm_world.state[PrefixName('r_joint_2', 'muh')].position = 0.2
        simple_two_arm_world.state[PrefixName('r_joint_3', 'muh')].position = 0.2
        simple_two_arm_world.state[PrefixName('l_joint_1', 'muh')].position = 0.2
        simple_two_arm_world.state[PrefixName('l_joint_2', 'muh')].position = 0.2
        simple_two_arm_world.state[PrefixName('l_joint_3', 'muh')].position = 0.2
        visualize()

    def test_compute_fk(self, box_world_prismatic: WorldTree):
        joint_name = box_world_prismatic.joint_names[0]
        box_name = box_world_prismatic.link_names[-1]

        box_world_prismatic.state[joint_name].position = 1
        box_world_prismatic.notify_state_change()
        fk = box_world_prismatic.compute_fk_point(root_link=box_world_prismatic.root_link_name,
                                                  tip_link=box_name).to_np()
        assert fk[0] == 1
        visualize()

    def test_cart_goal(self, box_world: WorldTree):
        joint_name = box_world.joint_names[0]
        box_name = box_world.link_names[-1]
        dt = 0.05
        goal = [1, 1, 0]

        cart_goal = CartesianPose(root_link=box_world.root_link_name,
                                  tip_link=box_name,
                                  goal_pose=cas.TransMatrix.from_xyz_rpy(x=goal[0], y=goal[1], z=goal[2],
                                                                         reference_frame=box_world.root_link_name))

        god_map.motion_statechart_manager.add_motion_goal(cart_goal)
        god_map.motion_statechart_manager.init_task_state()

        eq, neq, neqd, lin_weight, quad_weight = god_map.motion_statechart_manager.get_constraints_from_goals()
        controller = QPController(mpc_dt=dt)
        controller.init(free_variables=list(box_world.free_variables.values()),
                        equality_constraints=eq)
        controller.compile()
        traj = []
        for i in range(100):
            parameters = controller.get_parameter_names()
            substitutions = symbol_manager.resolve_symbols(parameters)
            next_cmd = controller.get_cmd(substitutions)
            box_world.update_state(next_cmd, dt, Derivatives.jerk)
            box_world.notify_state_change()
            traj.append(box_world.state[joint_name].position)
            visualize()
        fk = box_world.compute_fk_point(root_link=box_world.root_link_name, tip_link=box_name).to_np()
        np.testing.assert_almost_equal(fk[0], goal[0], decimal=3)
        np.testing.assert_almost_equal(fk[1], goal[1], decimal=3)

    def test_cart_goal_abs_sequence(self, box_world: WorldTree):
        joint_name = box_world.joint_names[0]
        box_name = box_world.link_names[-1]
        dt = 0.05
        goal1 = cas.TransMatrix.from_xyz_rpy(x=1, reference_frame=box_world.root_link_name)
        goal2 = cas.TransMatrix.from_xyz_rpy(y=1, reference_frame=box_name)

        cart_monitor = PoseReached(name='goal1 reached',
                                   root_link=box_world.root_link_name,
                                   tip_link=box_name,
                                   goal_pose=goal1)
        god_map.motion_statechart_manager.add_monitor(cart_monitor)

        cart_goal1 = CartesianPose(name='g1',
                                   root_link=box_world.root_link_name,
                                   tip_link=box_name,
                                   goal_pose=goal1,
                                   end_condition=cart_monitor.get_observation_state_expression())
        cart_goal2 = CartesianPose(name='g2',
                                   root_link=box_world.root_link_name,
                                   tip_link=box_name,
                                   goal_pose=goal2,
                                   absolute=True,
                                   start_condition=cart_monitor.get_observation_state_expression())

        god_map.motion_statechart_manager.add_motion_goal(cart_goal1)
        god_map.motion_statechart_manager.add_motion_goal(cart_goal2)

        god_map.motion_statechart_manager.compile_node_state_updaters()
        god_map.motion_statechart_manager.init_task_state()

        eq, neq, neqd, lin_weight, quad_weight = god_map.motion_statechart_manager.get_constraints_from_goals()
        controller = QPController(mpc_dt=dt)
        controller.init(free_variables=list(box_world.free_variables.values()),
                        equality_constraints=eq)
        controller.compile()
        traj = []
        god_map.time = 0
        god_map.control_cycle_counter = 0
        for i in range(200):
            parameters = controller.get_parameter_names()
            substitutions = symbol_manager.resolve_symbols(parameters)
            next_cmd = controller.get_cmd(substitutions)

            box_world.update_state(next_cmd, dt, Derivatives.jerk)
            box_world.notify_state_change()

            god_map.motion_statechart_manager.evaluate_node_states()
            traj.append(box_world.state[joint_name].position)
            god_map.time += controller.mpc_dt
            god_map.control_cycle_counter += 1
        fk = box_world.compute_fk_point(root_link=box_world.root_link_name, tip_link=box_name).to_np()
        np.testing.assert_almost_equal(fk[0], goal2.to_position().to_np()[0], decimal=3)
        np.testing.assert_almost_equal(fk[1], goal2.to_position().to_np()[1], decimal=3)

    def test_cart_goal_rel_sequence(self, box_world: WorldTree):
        joint_name = box_world.joint_names[0]
        box_name = box_world.link_names[-1]
        dt = 0.05
        goal1 = cas.TransMatrix.from_xyz_rpy(x=1, reference_frame=box_world.root_link_name)
        goal2 = cas.TransMatrix.from_xyz_rpy(y=1, reference_frame=box_name)

        cart_monitor = PoseReached(name='goal1 reached',
                                   root_link=box_world.root_link_name,
                                   tip_link=box_name,
                                   goal_pose=goal1)
        god_map.motion_statechart_manager.add_monitor(cart_monitor)

        cart_goal1 = CartesianPose(name='g1',
                                   root_link=box_world.root_link_name,
                                   tip_link=box_name,
                                   goal_pose=goal1,
                                   end_condition=cart_monitor.get_observation_state_expression())
        cart_goal2 = CartesianPose(name='g2',
                                   root_link=box_world.root_link_name,
                                   tip_link=box_name,
                                   goal_pose=goal2,
                                   start_condition=cart_monitor.get_observation_state_expression())

        god_map.motion_statechart_manager.add_motion_goal(cart_goal1)
        god_map.motion_statechart_manager.add_motion_goal(cart_goal2)

        god_map.motion_statechart_manager.compile_node_state_updaters()
        god_map.motion_statechart_manager.init_task_state()

        eq, neq, neqd, lin_weight, quad_weight = god_map.motion_statechart_manager.get_constraints_from_goals()
        controller = QPController(mpc_dt=dt)
        controller.init(free_variables=list(box_world.free_variables.values()),
                        equality_constraints=eq)
        controller.compile()
        traj = []
        god_map.time = 0
        god_map.control_cycle_counter = 0
        for i in range(200):
            parameters = controller.get_parameter_names()
            substitutions = symbol_manager.resolve_symbols(parameters)
            next_cmd = controller.get_cmd(substitutions)
            box_world.update_state(next_cmd, dt, Derivatives.jerk)
            box_world.notify_state_change()
            god_map.motion_statechart_manager.evaluate_node_states()
            traj.append((box_world.state[box_world.joints[joint_name].x_name].position,
                         box_world.state[box_world.joints[joint_name].y_name].position))
            god_map.time += controller.mpc_dt
            god_map.control_cycle_counter += 1
        fk = box_world.compute_fk_point(root_link=box_world.root_link_name, tip_link=box_name).to_np()
        np.testing.assert_almost_equal(fk[0], goal1.to_position().to_np()[0], decimal=2)
        np.testing.assert_almost_equal(fk[1], goal2.to_position().to_np()[1], decimal=2)

    def test_compute_self_collision_matrix(self, pr2_world: WorldTree):
        disabled_links = {pr2_world.search_for_link_name('br_caster_l_wheel_link'),
                          pr2_world.search_for_link_name('fr_caster_l_wheel_link')}
        reference_collision_scene = BetterPyBulletSyncer()
        reference_collision_scene.load_self_collision_matrix_from_srdf(
            'package://giskardpy/test/data/pr2_test.srdf', 'pr2')
        reference_reasons = reference_collision_scene.self_collision_matrix
        reference_disabled_links = reference_collision_scene.disabled_links
        collision_scene: CollisionWorldSynchronizer = god_map.collision_scene
        actual_reasons = collision_scene.compute_self_collision_matrix('pr2',
                                                                       number_of_tries_never=500)
        assert actual_reasons == reference_reasons
        assert reference_disabled_links == disabled_links

    def test_compute_chain_reduced_to_controlled_joints(self, simple_two_arm_world: WorldTree):
        r_gripper_tool_frame = simple_two_arm_world.search_for_link_name('r_eef')
        l_gripper_tool_frame = simple_two_arm_world.search_for_link_name('l_eef')
        link_a, link_b = simple_two_arm_world.compute_chain_reduced_to_controlled_joints(r_gripper_tool_frame,
                                                                                         l_gripper_tool_frame)
        assert link_a == simple_two_arm_world.search_for_link_name('r_eef')
        assert link_b == simple_two_arm_world.search_for_link_name('l_link_3')

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
        result = world_setup._search_branch(world_setup.search_for_link_name('odom_combined'),
                                            stop_at_joint_when=lambda _: False,
                                            stop_at_link_when=lambda _: False)
        assert result == ([], [])
        result = world_setup._search_branch(world_setup.search_for_link_name('odom_combined'),
                                            stop_at_joint_when=world_setup.is_joint_controlled,
                                            stop_at_link_when=lambda _: False,
                                            collect_link_when=world_setup.has_link_collisions)
        assert result == ([], [])
        result = world_setup._search_branch(world_setup.search_for_link_name('base_footprint'),
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
        result = world_setup._search_branch(world_setup.search_for_link_name('l_elbow_flex_link'),
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
        links, joints = world_setup._search_branch(world_setup.search_for_link_name('r_wrist_roll_link'),
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
        links, joints = world_setup._search_branch(world_setup.search_for_link_name('br_caster_l_wheel_link'),
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
            world_setup.get_controlled_parent_joint_of_joint(PrefixName('brumbrum', 'pr2'))
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


class TestController:
    def test_joint_goal(self, giskard_pr2: GiskardWrapper):
        init = 'init'
        g1 = 'g1'
        g2 = 'g2'
        giskard_pr2.monitors.add_set_seed_configuration(seed_configuration={'r_wrist_roll_joint': 2},
                                                        name=init)
        giskard_pr2.motion_goals.add_joint_position({'r_wrist_roll_joint': -1}, name=g1,
                                                    start_condition=init,
                                                    end_condition=g1)
        giskard_pr2.motion_goals.add_joint_position({'r_wrist_roll_joint': 1}, name=g2,
                                                    start_condition=g1)
        giskard_pr2.monitors.add_end_motion(start_condition=g2)
        giskard_pr2.execute()

    def test_cart_goal(self, giskard_pr2: GiskardWrapper):
        init = 'init'
        g1 = 'g1'
        g2 = 'g2'
        init_goal1 = cas.TransMatrix(reference_frame=PrefixName('map'))
        init_goal1.x = -0.5

        base_goal1 = cas.TransMatrix(reference_frame=PrefixName('map'))
        base_goal1.x = 1.0

        base_goal2 = cas.TransMatrix(reference_frame=PrefixName('map'))
        base_goal2.x = -1.0

        giskard_pr2.monitors.add_set_seed_odometry(base_pose=init_goal1, name=init)
        giskard_pr2.motion_goals.add_cartesian_pose(goal_pose=base_goal1, name=g1,
                                                    root_link='map',
                                                    tip_link='base_footprint',
                                                    start_condition=init,
                                                    end_condition=g1)
        giskard_pr2.motion_goals.add_cartesian_pose(goal_pose=base_goal2, name=g2,
                                                    root_link='map',
                                                    tip_link='base_footprint',
                                                    start_condition=g1)
        giskard_pr2.monitors.add_end_motion(start_condition=g2)
        giskard_pr2.execute(sim_time=20)

# import pytest
# pytest.main(['-s', __file__ + '::TestController::test_joint_goal_pr2'])
