import gc
import threading
import traceback
from collections import defaultdict
from datetime import datetime
from itertools import combinations, chain
from typing import Dict, List, Callable, Tuple, Optional

from data_types.data_types import JointStates
from hypothesis import strategies as st, settings, HealthCheck, example, reproduce_failure

import numpy as np
import pytest
import urdf_parser_py.urdf as up
import pandas as pd
import time
import giskardpy.casadi_wrapper as cas
from giskardpy.data_types.exceptions import EmptyProblemException
from giskardpy.data_types.data_types import PrefixName, Derivatives, ColorRGBA
from giskardpy.god_map import god_map
from giskardpy.model.better_pybullet_syncer import BetterPyBulletSyncer
from giskardpy.model.collision_avoidance_config import DefaultCollisionAvoidanceConfig
from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer
from giskardpy.model.joints import OmniDrive, PrismaticJoint
from giskardpy.model.links import Link, BoxGeometry
from giskardpy.model.utils import hacky_urdf_parser_fix
from giskardpy.model.world import WorldTree
from giskardpy.model.world_config import EmptyWorld, WorldWithOmniDriveRobot
from giskardpy.motion_graph.monitors.cartesian_monitors import PoseReached
from giskardpy.motion_graph.tasks.joint_tasks import JointPositionList
from giskardpy.qp.qp_controller import QPController
from giskardpy.symbol_manager import symbol_manager
from giskardpy.utils.utils import suppress_stderr
from giskardpy.model.collision_avoidance_config import DisableCollisionAvoidanceConfig
from giskardpy.model.trajectory import Trajectory
from giskardpy.qp.constraint import EqualityConstraint, InequalityConstraint, DerivativeInequalityConstraint
from giskardpy.qp.qp_controller import QPFormulation
from giskardpy.qp.qp_solver_ids import SupportedQPSolver
from hypothesis import given
from motion_graph.tasks.cartesian_tasks import CartesianPoseAsTask
from motion_graph.tasks.joint_tasks import JointVelocityLimit, JointVelocity
from motion_graph.tasks.task import WEIGHT_ABOVE_CA
from qp.constraint import DerivativeEqualityConstraint
from sympy.strategies.branch import condition
from tensorflow.python.ops.gen_math_ops import accumulate_nv2
from utils.math import limit
from utils_for_tests import pr2_urdf

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
    collision_avoidance = DisableCollisionAvoidanceConfig()
    collision_avoidance.setup()
    return config.world


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

        god_map.motion_graph_manager.add_motion_goal(cart_goal)
        god_map.motion_graph_manager.init_task_state()

        eq, neq, neqd, lin_weight, quad_weight = god_map.motion_graph_manager.get_constraints_from_goals()
        controller = QPController(sample_period=dt)
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
        god_map.motion_graph_manager.add_monitor(cart_monitor)

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

        god_map.motion_graph_manager.add_motion_goal(cart_goal1)
        god_map.motion_graph_manager.add_motion_goal(cart_goal2)

        god_map.motion_graph_manager.compile_node_state_updaters()
        god_map.motion_graph_manager.init_task_state()

        eq, neq, neqd, lin_weight, quad_weight = god_map.motion_graph_manager.get_constraints_from_goals()
        controller = QPController(sample_period=dt)
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

            god_map.motion_graph_manager.evaluate_node_states()
            traj.append(box_world.state[joint_name].position)
            god_map.time += controller.sample_period
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
        god_map.motion_graph_manager.add_monitor(cart_monitor)

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

        god_map.motion_graph_manager.add_motion_goal(cart_goal1)
        god_map.motion_graph_manager.add_motion_goal(cart_goal2)

        god_map.motion_graph_manager.compile_node_state_updaters()
        god_map.motion_graph_manager.init_task_state()

        eq, neq, neqd, lin_weight, quad_weight = god_map.motion_graph_manager.get_constraints_from_goals()
        controller = QPController(sample_period=dt)
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
            god_map.motion_graph_manager.evaluate_node_states()
            traj.append((box_world.state[box_world.joints[joint_name].x_name].position,
                         box_world.state[box_world.joints[joint_name].y_name].position))
            god_map.time += controller.sample_period
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


class Simulator:
    goal_names: List[str]

    def __init__(self, world: WorldTree, control_dt: float, mpc_dt: float, h: int, solver: SupportedQPSolver,
                 jerk_limit: float,
                 alpha: float, graph_styles: Optional[List[Tuple[str, str]]],
                 qp_formulation: QPFormulation):
        world.update_default_weights({Derivatives.velocity: 0.01,
                                      Derivatives.acceleration: 0.0,
                                      Derivatives.jerk: 0.0})
        world.update_default_limits({
            Derivatives.velocity: 1,
            Derivatives.acceleration: np.inf,
            Derivatives.jerk: jerk_limit
        })
        god_map.tmp_folder = '.'
        self.reset()
        self.world = world
        self.control_dt = control_dt
        self.mpc_dt = mpc_dt
        self.h = h
        self.alpha = alpha
        self.solver = solver
        self.qp_formulation = qp_formulation
        self.max_derivative = Derivatives.jerk
        if graph_styles is None:
            self.graph_styles = [
                ['-', '#996900'],
                [':', 'blue'],
                [':', 'blue'],
                ['--', 'black'],
            ]
        else:
            self.graph_styles = graph_styles
        god_map.qp_controller = QPController(sample_period=self.mpc_dt,
                                             solver_id=self.solver,
                                             prediction_horizon=self.h,
                                             max_derivative=self.max_derivative,
                                             qp_formulation=self.qp_formulation,
                                             alpha=self.alpha,
                                             verbose=False)

    def add_cart_goal(self, root_link: PrefixName, tip_link: PrefixName, x_goal: float):
        goal_name = PrefixName(f'x_goal')
        v = self.world.add_virtual_free_variable(goal_name)
        self.world.state[v.name].position = x_goal
        self.cart_goal = cas.TransMatrix.from_xyz_rpy(x=v.get_symbol(Derivatives.position),
                                                      reference_frame=root_link,
                                                      child_frame=tip_link)
        task = CartesianPoseAsTask(name='cart g1', root_link=root_link, tip_link=tip_link,
                                   goal_pose=self.cart_goal, absolute=True)

        god_map.motion_graph_manager.add_task(task)

    def add_joint_goal(self, joint_names: List[PrefixName], goal: float, name: str = 'g1'):
        self.goal_names.append(name)
        self.joint_goal = {}
        goal_name = PrefixName(name)
        v = self.world.add_virtual_free_variable(goal_name)
        self.world.state[v.name].position = goal
        for joint_name in joint_names:
            self.joint_goal[joint_name] = v.get_symbol(Derivatives.position)
        joint_task = JointPositionList(name=name, goal_state=self.joint_goal,
                                       # weight=WEIGHT_ABOVE_CA
                                       )

        god_map.motion_graph_manager.add_task(joint_task)

    def add_joint_vel_goal(self, joint_names: List[PrefixName], goal: float):
        self.joint_goal = {}
        goal_name = PrefixName(f'joint_goal')
        v = self.world.add_virtual_free_variable(goal_name)
        self.world.state[v.name].position = goal
        for joint_name in joint_names:
            self.joint_goal[joint_name] = v.get_symbol(Derivatives.position)
        joint_task = JointVelocity(name='g1', joint_names=joint_names, vel_goal=v.get_symbol(Derivatives.position))
        # joint_task = JointVelocityLimit(name='g1', joint_names=joint_names, max_velocity=goal)

        god_map.motion_graph_manager.add_task(joint_task)

    def compile(self):
        god_map.motion_graph_manager.initialize_states()

        eq, neq, eqd, neqd, lin_weight, quad_weight = god_map.motion_graph_manager.get_constraints_from_tasks()
        god_map.qp_controller.init(free_variables=self.get_active_free_symbols(eq, neq, eqd, neqd),
                                   equality_constraints=eq,
                                   inequality_constraints=neq,
                                   eq_derivative_constraints=eqd,
                                   derivative_constraints=neqd)
        god_map.qp_controller.compile()
        god_map.debug_expression_manager.compile_debug_expressions()
        self.traj = Trajectory()

    def get_active_free_symbols(self,
                                eq_constraints: List[EqualityConstraint],
                                neq_constraints: List[InequalityConstraint],
                                eq_derivative_constraints: List[DerivativeEqualityConstraint],
                                derivative_constraints: List[DerivativeInequalityConstraint]):
        symbols = set()
        for c in chain(eq_constraints, neq_constraints, eq_derivative_constraints, derivative_constraints):
            symbols.update(str(s) for s in cas.free_symbols(c.expression))
        free_variables = list(sorted([v for v in god_map.world.free_variables.values() if v.position_name in symbols],
                                     key=lambda x: x.position_name))
        if len(free_variables) == 0:
            raise EmptyProblemException('Goal parsing resulted in no free variables.')
        god_map.free_variables = free_variables
        return free_variables

    def reset(self):
        self.goal_names = []
        god_map.time = 0
        god_map.control_cycle_counter = 0
        god_map.motion_graph_manager.reset()
        god_map.debug_expression_manager.reset()
        for joint_state in god_map.world.state.values():
            joint_state.position = 0
            joint_state.velocity = 0
            joint_state.acceleration = 0
            joint_state.jerk = 0

    def step(self):
        parameters = god_map.qp_controller.get_parameter_names()
        total_time_start = time.time()
        substitutions = symbol_manager.resolve_symbols(parameters)
        parameter_time = time.time() - total_time_start

        qp_time_start = time.time()
        next_cmd = god_map.qp_controller.get_cmd(substitutions)
        qp_time = time.time() - qp_time_start
        god_map.debug_expression_manager.eval_debug_expressions()

        self.world.update_state(next_cmd, self.control_dt, god_map.qp_controller.max_derivative)

        self.world.notify_state_change()
        total_time = time.time() - total_time_start
        self.traj.set(god_map.control_cycle_counter, self.world.state)
        god_map.time += self.control_dt
        god_map.control_cycle_counter += 1
        return total_time, parameter_time, qp_time

    def run(self, goal_function: Callable[[str, float], float], sim_time: float,
            pos_noise: float, vel_noise: float, acc_noise: float,
            plot_legend: bool = True):
        np.random.seed(69)
        total_times, parameter_times, qp_times = [], [], []
        try:
            while god_map.time < sim_time:
                total_time, parameter_time, qp_time = self.step()
                self.apply_noise(pos_noise, vel_noise, acc_noise)

                total_times.append(total_time)
                parameter_times.append(parameter_time)
                qp_times.append(qp_time)
                for goal_name in self.goal_names:
                    self.update_joint_goal(goal_name, goal_function(goal_name, god_map.time))

        except Exception as e:
            traceback.print_exc()
            print(e)
            assert False
        finally:
            self.plot_traj(plot_legend)
        avg = np.average(qp_times)
        print(f'avg time {avg} or {1 / avg}hz')

    def apply_noise(self, pos: float, vel: float, acc: float, joint_names: List[str] = None):
        if joint_names is None:
            joint_names = self.world.movable_joint_names
        for joint_name in joint_names:
            pos_noise = np.random.normal(0, pos, 1)[0]
            curr_pose = self.world.state[joint_name].position
            try:
                lb, ub = self.world.joints[joint_name].get_limit_expressions(0)
                pos_noise = limit(pos_noise, min(0, lb - curr_pose), max(0, ub - curr_pose))
            except (KeyError, AttributeError) as e:
                pass
            self.world.state[joint_name].position += pos_noise
            self.world.state[joint_name].velocity += np.random.normal(0, vel, 1)[0]
            self.world.state[joint_name].acceleration += np.random.normal(0, acc, 1)[0]

    def update_joint_goal(self, goal_name: str, new_value: float):
        self.world.state[goal_name].position = new_value

    def update_cart_goal(self, new_value: float):
        self.world.state['x_goal'].position = new_value

    def plot_traj(self, plot_legend: bool = True):
        self.traj.plot_trajectory('test', sample_period=self.control_dt, filter_0_vel=True,
                                  hspace=0.5, height_per_derivative=4)
        color_map = defaultdict(lambda: self.graph_styles[len(color_map)])
        god_map.debug_expression_manager.raw_traj_to_traj(control_dt=self.control_dt).plot_trajectory(
            'debug', sample_period=self.control_dt, filter_0_vel=False,
            hspace=1, height_per_derivative=3,
            color_map=color_map,
            plot0_lines=False,
            legend=plot_legend,
            sort=False,
        )


def joint_thing(jerk_limit: float, sim_time: float, control_dt: float, mpc_dt: float, h: int,
                world: WorldTree, goal_function: Callable[[float], float], pos_noise: float, vel_noise: float,
                acc_noise: float, joint_names: Tuple[str] = ('pr2/r_wrist_roll_joint',), plot_legend: bool = True,
                graph_styles=None):

    simulator = Simulator(world=world,
                          control_dt=control_dt,
                          mpc_dt=mpc_dt,
                          h=h,
                          solver=SupportedQPSolver.gurobi,
                          alpha=0.1,
                          graph_styles=graph_styles,
                          jerk_limit=jerk_limit,
                          qp_formulation=QPFormulation.explicit_no_acc)
    simulator.reset()
    simulator.add_joint_goal(
        # joint_names=pr2_world.movable_joint_names[:1],
        joint_names=joint_names,
        # joint_names=pr2_world.movable_joint_names[14:15],
        # joint_names=pr2_world.movable_joint_names[:35],
        # joint_names=pr2_world.movable_joint_names[27:31],
        # joint_names=pr2_world.movable_joint_names[:-1],
        # joint_names=[
        # pr2_world.movable_joint_names[12], # torso
        # pr2_world.movable_joint_names[27], # r_gripper_r_finger
        # pr2_world.movable_joint_names[13] # head pan
        # pr2_world.movable_joint_names[14] # head pan
        # ],
        goal=goal_function(0))
    simulator.compile()

    simulator.run(goal_function, sim_time, pos_noise, vel_noise, acc_noise,
                  plot_legend=plot_legend)
    # print(f'min pos {min(simulator.traj.to_dict()[0][pr2_world.movable_joint_names[0]])}')
    # print(f'max vel {min(simulator.traj.to_dict()[1][pr2_world.movable_joint_names[0]])}')


class GoalSwapper:
    def __init__(self, goal: float, swap_time: float, noise: float, offset: float = 0):
        self.goal = goal
        self.next_swap = swap_time
        self.swap_time = swap_time
        self.noise = noise
        self.offset = offset

    def __call__(self, time: float):
        if time > self.next_swap:
            self.goal = - self.goal
            self.next_swap += self.swap_time
        return self.goal + self.offset + np.random.normal(0, self.noise, 1)[0]


class GoalSequence:
    def __init__(self, goals: List[Tuple[float, float]], noise: float):
        self.goals = goals
        self.noise = noise
        self.goal = 0
        self.i = 0

    def __call__(self, time: float):
        while self.i < len(self.goals) - 1 and time >= self.goals[self.i + 1][0]:
            self.i += 1
        self.goal = self.goals[self.i][1]
        return self.goal + np.random.normal(0, self.noise, 1)[0]


class GoalSin:
    def __init__(self, goal: float, magnitude: float, phase: float, noise: float):
        self.goal = goal
        self.magnitude = magnitude
        self.phase = phase
        self.noise = noise

    def __call__(self, time: float):
        self.goal = np.cos(god_map.time * np.pi / self.phase) * self.magnitude
        return self.goal + np.random.normal(0, self.noise, 1)[0]


class TestController:
    def test_joint_goal(self, box_world_prismatic: WorldTree):
        sim_time = 5
        joint_name = box_world_prismatic.joint_names[0]
        goal = 2
        box_world_prismatic.update_default_weights({Derivatives.velocity: 0.01,
                                                    Derivatives.acceleration: 0,
                                                    Derivatives.jerk: 0.0})
        box_world_prismatic.state[joint_name].position = 2
        simulator = Simulator(world=box_world_prismatic,
                              control_dt=0.05,
                              mpc_dt=0.05,
                              h=9,
                              solver=SupportedQPSolver.qpSWIFT,
                              alpha=0.1,
                              qp_formulation=QPFormulation.explicit)
        simulator.add_joint_goal(joint_names=[joint_name], goal=goal)
        simulator.compile()

        np.random.seed(69)
        swap_distance = 2
        next_swap = swap_distance
        try:
            while god_map.time < sim_time:
                # box_world_prismatic.state[goal_name].position = np.random.rand() * 2 -1
                simulator.step()
                if god_map.time > next_swap:
                    # goal *= -1
                    simulator.update_joint_goal(goal)
                    next_swap += swap_distance

        except Exception as e:
            traceback.print_exc()
            print(e)
        simulator.plot_traj()
        # fk = box_world_prismatic.compute_fk_point(root_link=box_world_prismatic.root_link_name,
        #                                           tip_link=box_name).to_np()
        # np.testing.assert_almost_equal(fk[0], box_world_prismatic.state[v.name].position, decimal=3)
        vel_profile = simulator.traj.to_dict()[Derivatives.velocity][joint_name]
        vel_limit = box_world_prismatic.joints[joint_name].free_variables[0].get_upper_limit(Derivatives.velocity)
        print(f'max violation {np.max(np.abs(vel_profile) - vel_limit)}')
        violated = np.all(np.abs(vel_profile) < vel_limit + 1e-4)
        assert violated, f'violation {np.max(np.abs(vel_profile) - vel_limit)}'

    def test_joint_goal_pr2_dt_vs_jerk(self, pr2_world: WorldTree):
        jerk_limit = 2500
        sim_time = 4
        noise = 0.25
        pos_noise = 0
        vel_noise = 0
        acc_noise = 0
        graph_styles = [
            ['--', 'black'],
            ['-', '#993000'],
        ]
        joint_thing(jerk_limit=jerk_limit, sim_time=sim_time,
                    control_dt=0.05, mpc_dt=0.05, h=5, world=pr2_world,
                    goal_function=GoalSwapper(1, 1.5, noise),
                    graph_styles=graph_styles, plot_legend=True,
                    pos_noise=pos_noise, vel_noise=vel_noise, acc_noise=acc_noise)
        joint_thing(jerk_limit=jerk_limit, sim_time=sim_time,
                    control_dt=0.01, mpc_dt=0.01, h=5, world=pr2_world,
                    goal_function=GoalSwapper(1, 1.5, noise),
                    graph_styles=graph_styles, plot_legend=False,
                    pos_noise=pos_noise, vel_noise=vel_noise, acc_noise=acc_noise)

    def test_joint_goal_pr2_implicit_limit(self, pr2_world: WorldTree):
        jerk_limit = 100
        sim_time = 4
        noise = 0.
        pos_noise = 0
        vel_noise = 0
        acc_noise = 0
        joint_thing(jerk_limit=jerk_limit, sim_time=sim_time,
                    control_dt=0.05, mpc_dt=0.05, h=5, world=pr2_world,
                    goal_function=GoalSwapper(1, 1.5, noise, 0),
                    plot_legend=False, pos_noise=pos_noise, vel_noise=vel_noise, acc_noise=acc_noise)
        joint_thing(jerk_limit=jerk_limit, sim_time=sim_time,
                    control_dt=0.05, mpc_dt=0.05, h=5, world=pr2_world,
                    goal_function=GoalSwapper(1, 1.5, noise, 0),
                    plot_legend=False, pos_noise=pos_noise, vel_noise=vel_noise, acc_noise=acc_noise)

    def test_joint_goal_pr2_dt_vs_dt_actual(self, pr2_world: WorldTree):
        jerk_limit = 100
        sim_time = 4
        noise = 0.
        pos_noise = 0
        vel_noise = 0
        acc_noise = 0
        joint_thing(jerk_limit=jerk_limit, sim_time=sim_time,
                    control_dt=0.01, mpc_dt=0.05, h=9, world=pr2_world,
                    goal_function=GoalSwapper(1, 1.5, noise),
                    plot_legend=True, pos_noise=pos_noise, vel_noise=vel_noise, acc_noise=acc_noise)
        joint_thing(jerk_limit=jerk_limit, sim_time=sim_time,
                    control_dt=0.05, mpc_dt=0.05, h=9, world=pr2_world,
                    goal_function=GoalSwapper(1, 1.5, noise),
                    plot_legend=False, pos_noise=pos_noise, vel_noise=vel_noise, acc_noise=acc_noise)
        joint_thing(jerk_limit=jerk_limit, sim_time=sim_time + 0.05,
                    control_dt=0.075, mpc_dt=0.05, h=5, world=pr2_world,
                    goal_function=GoalSwapper(1, 1.5, noise),
                    plot_legend=False, pos_noise=pos_noise, vel_noise=vel_noise, acc_noise=acc_noise)

    def test_joint_goal_pr2_pos_limits(self, pr2_world: WorldTree):
        jerk_limit = 1111
        sim_time = 6
        noise = 0.00
        control_dt = 0.01
        pos_noise = 0.0005
        vel_noise = 0
        acc_noise = 0
        graph_styles = [
            [':', 'black'],
            ['!shade above', 'red'],
            ['!shade above', 'red'],
            ['!shade below', 'red'],
            ['!shade above', 'red'],
            ['!shade below', 'red'],
            ['-', '#003399'],
            ['-', '#003399'],
            ['-', '#003399'],
            ['-', '#003399'],
        ]
        goals = [
            [0, -0.1],
            # [0.8, -0.15]
        ]
        x = np.linspace(0.8, sim_time, 1000)
        x2 = np.linspace(0.75, 1.75, 1000)
        y = np.cos((x) * np.pi * x2) * 0.1 - 0.2
        goals.extend(list(zip(x.tolist(), y.tolist())))
        joint_thing(jerk_limit=jerk_limit, sim_time=sim_time,
                    control_dt=control_dt, mpc_dt=control_dt, h=7, world=pr2_world,
                    goal_function=GoalSequence(goals, noise),
                    joint_names=('pr2/r_elbow_flex_joint',),
                    plot_legend=True, pos_noise=pos_noise, vel_noise=vel_noise, acc_noise=acc_noise,
                    graph_styles=graph_styles)
        # simulator.traj.to_dict()[0][joint_names[0]][61:]

    def test_joint_goal_pr2_fight(self, pr2_world: WorldTree):
        jerk_limit = 1111
        sim_time = 6
        h = 7
        noise = 0.00
        control_dt = 0.01
        pos_noise = 0.0005
        vel_noise = 0
        acc_noise = 0
        graph_styles = [
            [':', 'black'],
            # ['!shade above', 'red'],
            # ['!shade above', 'red'],
            # ['!shade below', 'red'],
            # ['!shade above', 'red'],
            # ['!shade below', 'red'],
            ['-', '#003399'],
            ['-', '#003399'],
            ['-', '#003399'],
            ['-', '#003399'],
            [':', 'black'],
            # ['!shade above', 'red'],
            # ['!shade above', 'red'],
            # ['!shade below', 'red'],
            # ['!shade above', 'red'],
            # ['!shade below', 'red'],
            ['-', '#003399'],
            ['-', '#003399'],
            ['-', '#003399'],
            ['-', '#003399'],
            [':', 'black'],
            # ['!shade above', 'red'],
            # ['!shade above', 'red'],
            # ['!shade below', 'red'],
            # ['!shade above', 'red'],
            # ['!shade below', 'red'],
            ['-', '#003399'],
            ['-', '#003399'],
            ['-', '#003399'],
            ['-', '#003399'],
        ]
        goals = [
            [0, -0.1],
            # [0.8, -0.15]
        ]
        x = np.linspace(0.8, sim_time, 1000)
        x2 = np.linspace(0.75, 1.75, 1000)
        y = np.cos((x) * np.pi * x2) * 0.1 - 0.2
        goals.extend(list(zip(x.tolist(), y.tolist())))

        simulator = Simulator(world=pr2_world,
                              control_dt=control_dt,
                              mpc_dt=control_dt,
                              h=h,
                              solver=SupportedQPSolver.gurobi,
                              alpha=0.1,
                              graph_styles=graph_styles,
                              jerk_limit=jerk_limit,
                              qp_formulation=QPFormulation.explicit_no_acc)
        simulator.reset()
        simulator.add_joint_goal(
            # joint_names=pr2_world.movable_joint_names[:1],
            joint_names=('pr2/r_wrist_roll_joint',),
            # joint_names=pr2_world.movable_joint_names[14:15],
            # joint_names=pr2_world.movable_joint_names[:35],
            # joint_names=pr2_world.movable_joint_names[27:31],
            # joint_names=pr2_world.movable_joint_names[:-1],
            # joint_names=[
            # pr2_world.movable_joint_names[12], # torso
            # pr2_world.movable_joint_names[27], # r_gripper_r_finger
            # pr2_world.movable_joint_names[13] # head pan
            # pr2_world.movable_joint_names[14] # head pan
            # ],
            name='g1',
            goal=1)
        simulator.add_joint_goal(
            # joint_names=pr2_world.movable_joint_names[:1],
            joint_names=('pr2/r_wrist_roll_joint',),
            # joint_names=pr2_world.movable_joint_names[14:15],
            # joint_names=pr2_world.movable_joint_names[:35],
            # joint_names=pr2_world.movable_joint_names[27:31],
            # joint_names=pr2_world.movable_joint_names[:-1],
            # joint_names=[
            # pr2_world.movable_joint_names[12], # torso
            # pr2_world.movable_joint_names[27], # r_gripper_r_finger
            # pr2_world.movable_joint_names[13] # head pan
            # pr2_world.movable_joint_names[14] # head pan
            # ],
            name='g2',
            goal=-1)
        simulator.compile()
        def goal_function(name, time):
            if name == 'g1':
                return 1
            return -1
        simulator.run(goal_function, sim_time, pos_noise, vel_noise, acc_noise,
                      plot_legend=False)

    def test_joint_goal_pr2_tune_guide(self, pr2_world: WorldTree):
        sim_time = 4
        noise = 0.05
        pos_noise = 0.0
        vel_noise = 0
        acc_noise = 0
        phase = 1.5
        graph_styles = [
            ['--', 'black'],
            # ['!shade above', 'red'],
            # ['!shade above', 'red'],
            # ['!shade below', 'red'],
            ['-', '#993000'],
        ]
        joint_thing(jerk_limit=5000, sim_time=sim_time,
                    control_dt=0.01, mpc_dt=0.01, h=4, world=pr2_world,
                    goal_function=GoalSin(1, magnitude=0.5, phase=phase, noise=noise),
                    pos_noise=pos_noise, vel_noise=vel_noise, acc_noise=acc_noise,
                    graph_styles=graph_styles,
                    plot_legend=False)
        joint_thing(jerk_limit=40000, sim_time=sim_time,
                    control_dt=0.01, mpc_dt=0.01, h=4, world=pr2_world,
                    goal_function=GoalSin(1, magnitude=0.5, phase=phase, noise=noise),
                    pos_noise=pos_noise, vel_noise=vel_noise, acc_noise=acc_noise,
                    graph_styles=graph_styles,
                    plot_legend=False)
        joint_thing(jerk_limit=1000, sim_time=sim_time,
                    control_dt=0.01, mpc_dt=0.01, h=10, world=pr2_world,
                    goal_function=GoalSin(1, magnitude=0.5, phase=phase, noise=noise),
                    pos_noise=pos_noise, vel_noise=vel_noise, acc_noise=acc_noise,
                    graph_styles=graph_styles,
                    plot_legend=False)
        # joint_thing(jerk_limit=500, sim_time=sim_time,
        #             control_dt=0.01, mpc_dt=0.01, h=10, world=pr2_world,
        #             goal_function=GoalSin(1, magnitude=0.5, phase=phase, noise=noise),
        #             pos_noise=pos_noise, vel_noise=vel_noise, acc_noise=acc_noise,
        #             plot_legend=False)

    def test_joint_vel_goal_pr2(self, pr2_world: WorldTree):
        sim_time = 4
        goal = 0.0
        pr2_world.update_default_weights({Derivatives.velocity: 0.01,
                                          Derivatives.acceleration: 0.0,
                                          Derivatives.jerk: 0.0})
        pr2_world.update_default_limits({
            Derivatives.velocity: 1,
            Derivatives.acceleration: np.inf,
            Derivatives.jerk: 100
        })
        simulator = Simulator(world=pr2_world,
                              control_dt=0.05,
                              mpc_dt=0.05,
                              h=9,
                              solver=SupportedQPSolver.qpSWIFT,
                              alpha=.1,
                              qp_formulation=QPFormulation.implicit)
        joint_names = pr2_world.movable_joint_names[:1]
        # joint_names=pr2_world.movable_joint_names[14:15],
        # joint_names=pr2_world.movable_joint_names[:35],
        # joint_names=pr2_world.movable_joint_names[27:31],
        # joint_names=pr2_world.movable_joint_names[:-1],
        # joint_names=[
        # pr2_world.movable_joint_names[27],
        # pr2_world.movable_joint_names[13]
        # ],
        simulator.add_joint_vel_goal(goal=goal, joint_names=joint_names)
        simulator.compile()

        np.random.seed(69)
        swap_distance = 2
        next_swap = swap_distance
        try:
            while god_map.time < sim_time:
                simulator.step()
                simulator.apply_noise(0, 0.1, 0, joint_names)
                goal = np.sin(god_map.time * 8) * 0.5
                simulator.update_joint_goal(goal)
                # goal = np.random.rand() * 2 -1
                # simulator.update_joint_goal(goal)
                # if god_map.time > next_swap:
                #     goal *= -1
                #     simulator.update_joint_goal(goal)
                #     next_swap += swap_distance

        except Exception as e:
            traceback.print_exc()
            print(e)
            assert False
        finally:
            simulator.plot_traj()
        # print(f'min pos {min(simulator.traj.to_dict()[0][pr2_world.movable_joint_names[0]])}')
        # print(f'max vel {min(simulator.traj.to_dict()[1][pr2_world.movable_joint_names[0]])}')

    # @settings(print_blob=True, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    # @given(
    #     pos=st.floats(min_value=-3.5, max_value=3.5),  # Set appropriate min/max limits
    #     vel=st.floats(min_value=-20.0, max_value=20.0),
    #     acc=st.floats(min_value=-100.0, max_value=100.0),
    #     goal=st.floats(min_value=-4.0, max_value=4.0)
    # )
    # @example(pos=0., vel=20.0, acc=100, goal=0)  # replace with actual failing values
    def test_pr2_pos_limits(self, pr2_world: WorldTree):
        # new 316
        # pos_values = [3, 2.857, 2.8, 0, -3]
        # vel_values = [20, 1, 0, 1 -20]
        # acc_values = [100, 1, 0, -100]
        # goal_values = [2.8]
        joint = 'pr2/r_elbow_flex_joint'
        pos_values = [-0.10264586748458397]
        vel_values = [-0.3688338790328299]
        acc_values = [27.329946331793636]
        goal_values = [-0.1]
        graph_styles = [
            [':', 'black'],
            ['!shade above', 'red'],
            ['!shade above', 'red'],
            ['!shade below', 'red'],
            ['!shade above', 'red'],
            ['!shade below', 'red'],
            ['-', '#003399'],
        ]
        jerk_violations = 0
        for pos in pos_values:
            for vel in vel_values:
                for acc in acc_values:
                    for goal in goal_values:
                        print(f'testing {pos} {vel} {acc} {goal}')
                        # joint = pr2_world.movable_joint_names[13]
                        # joint = pr2_world.movable_joint_names[14]
                        sim_time = 4
                        pr2_world.update_default_weights({Derivatives.velocity: 0.01,
                                                          Derivatives.acceleration: 0.0,
                                                          Derivatives.jerk: 0.0})
                        pr2_world.update_default_limits({
                            Derivatives.velocity: 1,
                            Derivatives.acceleration: np.inf,
                            Derivatives.jerk: 1111
                        })
                        simulator = Simulator(world=pr2_world,
                                              control_dt=0.01,
                                              mpc_dt=0.01,
                                              h=7,
                                              solver=SupportedQPSolver.gurobi,
                                              alpha=.1,
                                              graph_styles=graph_styles,
                                              qp_formulation=QPFormulation.explicit_no_acc)
                        pr2_world.state[joint][0] = pos
                        pr2_world.state[joint][1] = vel
                        pr2_world.state[joint][2] = acc
                        simulator.add_joint_goal(
                            # joint_names=pr2_world.movable_joint_names[:1],
                            # joint_names=pr2_world.movable_joint_names[14:15],
                            # joint_names=pr2_world.movable_joint_names[:35],
                            # joint_names=pr2_world.movable_joint_names[27:31],
                            # joint_names=pr2_world.movable_joint_names[:-1],
                            joint_names=[
                                # pr2_world.movable_joint_names[27],
                                # pr2_world.movable_joint_names[13]
                                joint
                            ],
                            goal=goal)
                        simulator.compile()

                        np.random.seed(69)
                        swap_distance = 2
                        next_swap = swap_distance
                        try:
                            while god_map.time < sim_time:
                                simulator.step()
                                # goal = np.random.rand() * 2 -1
                                # simulator.update_joint_goal(goal)
                                # if god_map.time > next_swap:
                                #     goal *= -1
                                #     simulator.update_joint_goal(goal)
                                #     next_swap += swap_distance

                        except Exception as e:
                            traceback.print_exc()
                            print(e)
                            assert False
                        finally:
                            simulator.plot_traj()
                        pos_ub, vel, _, jerk = list(pr2_world.joints[joint].free_variable.get_upper_limits(3).values())
                        pos_lb, _, _, _ = list(pr2_world.joints[joint].free_variable.get_lower_limits(3).values())
                        # print(f'min pos {min(simulator.traj.to_dict()[0][joint])}')
                        eps = 1e-2
                        traj = simulator.traj.to_dict(filter_0_vel=False, normalize_position=False)
                        assert traj[0][joint][-1] <= pos_ub + eps
                        assert traj[0][joint][-1] >= pos_lb - eps
                        assert np.all(np.abs(traj[1][joint]) <= vel + eps)
                        if np.any(np.abs(traj[Derivatives.jerk][joint]) > jerk + eps):
                            jerk_violations += 1
                        print(f'num jerk violations {jerk_violations}')
                        # assert np.all(np.abs(simulator.traj.to_dict()[3][joint]) <= jerk)
                        # assert np.all(simulator.traj.to_dict()[3][joint] >= vel)
                        # print(f'max vel {min(simulator.traj.to_dict()[1][joint])}')

    def test_cart_goal_pr2(self, pr2_world: WorldTree):
        sim_time = 4
        goal = -0.9
        pr2_world.update_default_weights({Derivatives.velocity: 0.01,
                                          Derivatives.acceleration: 0.0,
                                          Derivatives.jerk: 0.0})
        pr2_world.update_default_limits({
            Derivatives.velocity: 1,
            Derivatives.acceleration: np.inf,
            Derivatives.jerk: 100
        })
        chain_a, l, chain_b = pr2_world.compute_split_chain(
            root_link_name=PrefixName('l_gripper_r_finger_tip_link', 'pr2'),
            tip_link_name=PrefixName('r_gripper_r_finger_tip_link', 'pr2'),
            add_joints=False, add_links=True, add_fixed_joints=False,
            add_non_controlled_joints=False)
        left_right_kin_chain = chain_a + l + chain_b
        simulator = Simulator(world=pr2_world,
                              control_dt=0.05,
                              mpc_dt=0.05,
                              h=9,
                              solver=SupportedQPSolver.qpSWIFT,
                              qp_formulation=QPFormulation.implicit)
        length = 50
        tip_link = left_right_kin_chain[0]
        for i in range(1, len(left_right_kin_chain)):
            root_link = left_right_kin_chain[i]
            a, b, c = pr2_world.compute_split_chain(tip_link, root_link, add_joints=True, add_links=False,
                                                    add_fixed_joints=False, add_non_controlled_joints=True)
            actual_length = len(a + b + c)
            if actual_length == length:
                break
        print(actual_length)
        simulator.add_cart_goal(root_link=tip_link,
                                tip_link=root_link,
                                x_goal=goal)
        simulator.compile()

        np.random.seed(69)
        swap_distance = 1.7
        next_swap = swap_distance
        try:
            while god_map.time < sim_time:
                simulator.step()
                goal = np.random.rand() * 2 - 1
                simulator.update_cart_goal(goal)
                # if god_map.time > next_swap:
                #     goal *= -1
                #     simulator.update_cart_goal(goal)
                #     next_swap += swap_distance
            h_density = god_map.qp_controller.qp_solver.H_density
            a_density = god_map.qp_controller.qp_solver.A_density
            e_density = god_map.qp_controller.qp_solver.E_density
            print(h_density, a_density, e_density)

        except Exception as e:
            traceback.print_exc()
            print(e)
            assert False
        finally:
            simulator.plot_traj()

    def test_pr2_joint_benchmark(self, pr2_world: WorldTree):
        # i = 4
        # solvers = list(SupportedQPSolver)[i:i+1]
        solvers = [
            # SupportedQPSolver.qpSWIFT,
            # SupportedQPSolver.qpalm,
            # SupportedQPSolver.gurobi,
            SupportedQPSolver.piqp,
            # SupportedQPSolver.clarabel,
            # SupportedQPSolver.cvxopt,
            # SupportedQPSolver.osqp,
            # SupportedQPSolver.proxsuite,
            # SupportedQPSolver.daqp,
        ]
        formulations = [
            # ControllerMode.explicit,
            # ControllerMode.explicit_no_acc,
            QPFormulation.implicit,
            # ControllerMode.no_mpc
        ]
        max_number_of_joints_total = len(pr2_world.movable_joint_names) - 1  # the last joint doesn't work
        sim_time = 4
        goal = -0.9
        mpc_dt = 0.05
        control_dt = 0.05
        alpha = .1
        horizion = 9
        pr2_world.update_default_weights({
            Derivatives.velocity: 0.01,
            Derivatives.acceleration: 0,
            Derivatives.jerk: 0.0
        })

        # List to store results
        date_str = datetime.now().strftime('%Yy-%mm-%dd--%Hh-%Mm-%Ss')
        for solver in solvers:
            results = []
            print(f'solver {solver.name}')
            for formulation in formulations:
                print(f'formulation {formulation.name}')
                num_fails = 0
                for num_joints in range(1, max_number_of_joints_total, 2):
                    gc.collect()
                    print(f'#joints {num_joints}')
                    simulator = Simulator(
                        world=pr2_world,
                        control_dt=control_dt,
                        mpc_dt=mpc_dt,
                        h=horizion,
                        solver=solver,
                        alpha=alpha,
                        qp_formulation=formulation
                    )
                    simulator.add_joint_goal(
                        joint_names=pr2_world.movable_joint_names[0:num_joints],
                        goal=goal)
                    simulator.compile()

                    np.random.seed(69)
                    swap_distance = 2
                    next_swap = swap_distance

                    # Variables to collect statistics
                    total_times, parameter_times, qp_times = [], [], []

                    try:
                        while god_map.time < sim_time:
                            total_time, parameter_time, qp_time = simulator.step()

                            total_times.append(total_time)
                            parameter_times.append(parameter_time)
                            qp_times.append(qp_time)

                            if god_map.time > next_swap:
                                goal *= -1
                                simulator.update_joint_goal(goal)
                                next_swap += swap_distance
                    except Exception as e:
                        # traceback.print_exc()
                        print(e)
                        num_fails += 1
                    h_density = god_map.qp_controller.qp_solver.H_density
                    a_density = god_map.qp_controller.qp_solver.A_density
                    e_density = god_map.qp_controller.qp_solver.E_density
                    total_density = god_map.qp_controller.qp_solver.total_density
                    # Append results to the list
                    print(f'number of fails {num_fails}')
                    results.append({
                        'Solver': solver.name,
                        'Formulation': formulation.name,
                        'Num_Joints': num_joints,
                        'Total_Time_Sum': np.sum(total_times) if total_times else 100,
                        'Total_Time_Avg': np.average(total_times) if total_times else 100,
                        'Total_Time_Std': np.std(total_times) if total_times else 100,
                        'Total_Time_Min': np.min(total_times) if total_times else 100,
                        'Total_Time_Max': np.max(total_times) if total_times else 100,
                        'Total_Time_Median': np.median(total_times) if total_times else 100,
                        'Parameter_Time_Sum': np.sum(parameter_times) if total_times else 100,
                        'Parameter_Time_Avg': np.average(parameter_times) if total_times else 100,
                        'Parameter_Time_Median': np.median(parameter_times) if total_times else 100,
                        'Parameter_Time_Std': np.std(parameter_times) if total_times else 100,
                        'Parameter_Time_Min': np.min(parameter_times) if total_times else 100,
                        'Parameter_Time_Max': np.max(parameter_times) if total_times else 100,
                        'QP_Time_Sum': np.sum(qp_times) if total_times else 100,
                        'QP_Time_Avg': np.average(qp_times) if total_times else 100,
                        'QP_Time_Median': np.median(qp_times) if total_times else 100,
                        'QP_Time_Std': np.std(qp_times) if total_times else 100,
                        'QP_Time_Min': np.min(qp_times) if total_times else 100,
                        'QP_Time_Max': np.max(qp_times) if total_times else 100,
                        'Num_Fails': num_fails,
                        'Control dt': control_dt,
                        'horizon': horizion,
                        'MPC dt': mpc_dt,
                        'Num_Variables': god_map.qp_controller.num_free_variables,
                        'Num_Inequality_Constraints': god_map.qp_controller.num_ineq_constraints,
                        'Num_Equality_Constraints': god_map.qp_controller.num_eq_constraints,
                        'H_Density': h_density,
                        'A_Density': a_density,
                        'E_Density': e_density,
                        'Combined_Density': total_density
                    })

            # Convert results to a Pandas DataFrame
            df = pd.DataFrame(results)
            # df.to_csv(f'benchmark_results_{date_str}.csv', index=False)
            file_name = f'benchmark_joint_results_{solver.name}_{date_str}.pkl'
            df.to_pickle(file_name)
            print(f'saved {file_name}')

    def test_pr2_cart_benchmark(self, pr2_world: WorldTree):
        # i = 4
        # solvers = list(SupportedQPSolver)[i:i+1]
        solvers = [
            # SupportedQPSolver.qpSWIFT,
            # SupportedQPSolver.qpalm,
            # SupportedQPSolver.gurobi,
            # SupportedQPSolver.piqp,
            # SupportedQPSolver.clarabel,
            # SupportedQPSolver.osqp,
            # SupportedQPSolver.cvxopt,
            # SupportedQPSolver.proxsuite,
            SupportedQPSolver.daqp,
        ]
        formulations = [
            # ControllerMode.explicit,
            QPFormulation.explicit_no_acc,
            QPFormulation.implicit,
            QPFormulation.no_mpc
        ]
        max_chain_length = 18
        sim_time = 4
        goal = -0.9
        mpc_dt = 0.05
        control_dt = 0.05
        horizion = 9
        pr2_world.update_default_weights({
            Derivatives.velocity: 0.01,
            Derivatives.acceleration: 0,
            Derivatives.jerk: 0.0
        })

        chain_a, l, chain_b = pr2_world.compute_split_chain(
            root_link_name=PrefixName('l_gripper_r_finger_tip_link', 'pr2'),
            tip_link_name=PrefixName('r_gripper_r_finger_tip_link', 'pr2'),
            add_joints=False, add_links=True, add_fixed_joints=False,
            add_non_controlled_joints=False)
        left_right_kin_chain = chain_a + l + chain_b

        # List to store results
        date_str = datetime.now().strftime('%Yy-%mm-%dd--%Hh-%Mm-%Ss')
        for solver in solvers:
            print(f'solver {solver.name}')
            results = []
            for formulation in formulations:
                print(f'formulation {formulation.name}')
                num_fails = 0
                for num_joints in range(1, max_chain_length):
                    gc.collect()
                    print(f'#joints {num_joints}')
                    simulator = Simulator(
                        world=pr2_world,
                        control_dt=control_dt,
                        mpc_dt=mpc_dt,
                        h=horizion,
                        solver=solver,
                        qp_formulation=formulation
                    )
                    tip_link = left_right_kin_chain[0]
                    for i in range(1, len(left_right_kin_chain)):
                        root_link = left_right_kin_chain[i]
                        a, b, c = pr2_world.compute_split_chain(tip_link, root_link, add_joints=True, add_links=False,
                                                                add_fixed_joints=False, add_non_controlled_joints=True)
                        actual_num_joints = len(a + b + c)
                        if actual_num_joints == num_joints:
                            break
                    simulator.add_cart_goal(root_link=tip_link,
                                            tip_link=root_link,
                                            x_goal=goal)
                    simulator.compile()

                    np.random.seed(69)
                    swap_distance = 1.7
                    next_swap = swap_distance

                    # Variables to collect statistics
                    total_times, parameter_times, qp_times = [], [], []

                    try:
                        while god_map.time < sim_time:
                            total_time, parameter_time, qp_time = simulator.step()

                            total_times.append(total_time)
                            parameter_times.append(parameter_time)
                            qp_times.append(qp_time)

                            if god_map.time > next_swap:
                                goal *= -1
                                simulator.update_joint_goal(goal)
                                next_swap += swap_distance
                    except Exception as e:
                        # traceback.print_exc()
                        print(e)
                        num_fails += 1
                        next_swap += swap_distance
                    h_density = god_map.qp_controller.qp_solver.H_density
                    a_density = god_map.qp_controller.qp_solver.A_density
                    e_density = god_map.qp_controller.qp_solver.E_density
                    total_density = god_map.qp_controller.qp_solver.total_density
                    # Append results to the list
                    print(f'number of fails {num_fails}')
                    results.append({
                        'Solver': solver.name,
                        'Formulation': formulation.name,
                        'Num_Joints': num_joints,
                        'Total_Time_Sum': np.sum(total_times) if total_times else 100,
                        'Total_Time_Avg': np.average(total_times) if total_times else 100,
                        'Total_Time_Std': np.std(total_times) if total_times else 100,
                        'Total_Time_Min': np.min(total_times) if total_times else 100,
                        'Total_Time_Max': np.max(total_times) if total_times else 100,
                        'Total_Time_Median': np.median(total_times) if total_times else 100,
                        'Parameter_Time_Sum': np.sum(parameter_times) if total_times else 100,
                        'Parameter_Time_Avg': np.average(parameter_times) if total_times else 100,
                        'Parameter_Time_Median': np.median(parameter_times) if total_times else 100,
                        'Parameter_Time_Std': np.std(parameter_times) if total_times else 100,
                        'Parameter_Time_Min': np.min(parameter_times) if total_times else 100,
                        'Parameter_Time_Max': np.max(parameter_times) if total_times else 100,
                        'QP_Time_Sum': np.sum(qp_times) if total_times else 100,
                        'QP_Time_Avg': np.average(qp_times) if total_times else 100,
                        'QP_Time_Median': np.median(qp_times) if total_times else 100,
                        'QP_Time_Std': np.std(qp_times) if total_times else 100,
                        'QP_Time_Min': np.min(qp_times) if total_times else 100,
                        'QP_Time_Max': np.max(qp_times) if total_times else 100,
                        'Num_Fails': num_fails,
                        'Control dt': control_dt,
                        'horizon': horizion,
                        'MPC dt': mpc_dt,
                        'Num_Variables': god_map.qp_controller.num_free_variables,
                        'Num_Inequality_Constraints': god_map.qp_controller.num_ineq_constraints,
                        'Num_Equality_Constraints': god_map.qp_controller.num_eq_constraints,
                        'H_Density': h_density,
                        'A_Density': a_density,
                        'E_Density': e_density,
                        'Combined_Density': total_density
                    })

            # Convert results to a Pandas DataFrame
            df = pd.DataFrame(results)
            # df.to_csv(f'benchmark_results_{date_str}.csv', index=False)
            file_name = f'benchmark_cart_results_{solver.name}_{date_str}.pkl'
            df.to_pickle(file_name)
            print(f'saved {file_name}')

# import pytest
# pytest.main(['-s', __file__ + '::TestController::test_joint_goal_pr2'])
