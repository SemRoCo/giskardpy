import gc
import threading
import traceback
from datetime import datetime
from itertools import combinations, chain
from typing import Dict, List

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
from giskardpy.qp.qp_controller import ControllerMode
from giskardpy.qp.qp_solver_ids import SupportedQPSolver
from motion_graph.tasks.cartesian_tasks import CartesianPoseAsTask
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
    def __init__(self, world: WorldTree, control_dt: float, mpc_dt: float, h: int, solver: SupportedQPSolver,
                 qp_formulation: ControllerMode):
        god_map.tmp_folder = '.'
        self.reset()
        self.world = world
        self.control_dt = control_dt
        self.mpc_dt = mpc_dt
        self.h = h
        self.solver = solver
        self.qp_formulation = qp_formulation
        self.max_derivative = Derivatives.jerk

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

    def add_joint_goal(self, joint_names: List[PrefixName], goal: float):
        self.joint_goal = {}
        goal_name = PrefixName(f'joint_goal')
        v = self.world.add_virtual_free_variable(goal_name)
        self.world.state[v.name].position = goal
        for joint_name in joint_names:
            self.joint_goal[joint_name] = v.get_symbol(Derivatives.position)
        joint_task = JointPositionList(name='g1', goal_state=self.joint_goal)

        god_map.motion_graph_manager.add_task(joint_task)

    def compile(self):
        god_map.motion_graph_manager.initialize_states()

        eq, neq, neqd, lin_weight, quad_weight = god_map.motion_graph_manager.get_constraints_from_tasks()
        god_map.qp_controller = QPController(sample_period=self.mpc_dt,
                                             solver_id=self.solver,
                                             prediction_horizon=self.h,
                                             max_derivative=self.max_derivative,
                                             qp_formulation=self.qp_formulation,
                                             verbose=False)
        god_map.qp_controller.init(free_variables=self.get_active_free_symbols(eq, neq, neqd),
                                   equality_constraints=eq)
        god_map.qp_controller.compile()
        self.traj = Trajectory()

    def get_active_free_symbols(self,
                                eq_constraints: List[EqualityConstraint],
                                neq_constraints: List[InequalityConstraint],
                                derivative_constraints: List[DerivativeInequalityConstraint]):
        symbols = set()
        for c in chain(eq_constraints, neq_constraints, derivative_constraints):
            symbols.update(str(s) for s in cas.free_symbols(c.expression))
        free_variables = list(sorted([v for v in god_map.world.free_variables.values() if v.position_name in symbols],
                                     key=lambda x: x.position_name))
        if len(free_variables) == 0:
            raise EmptyProblemException('Goal parsing resulted in no free variables.')
        god_map.free_variables = free_variables
        return free_variables

    def reset(self):
        god_map.time = 0
        god_map.control_cycle_counter = 0
        god_map.motion_graph_manager.reset()

    def step(self):
        parameters = god_map.qp_controller.get_parameter_names()
        total_time_start = time.time()
        substitutions = symbol_manager.resolve_symbols(parameters)
        parameter_time = time.time() - total_time_start

        qp_time_start = time.time()
        next_cmd = god_map.qp_controller.get_cmd(substitutions)
        qp_time = time.time() - qp_time_start

        self.world.update_state(next_cmd, self.control_dt, god_map.qp_controller.max_derivative)

        self.world.notify_state_change()
        total_time = time.time() - total_time_start
        self.traj.set(god_map.control_cycle_counter, self.world.state)
        god_map.time += self.control_dt
        god_map.control_cycle_counter += 1
        return total_time, parameter_time, qp_time

    def update_joint_goal(self, new_value: float):
        self.world.state['joint_goal'].position = new_value

    def update_cart_goal(self, new_value: float):
        self.world.state['x_goal'].position = new_value

    def plot_traj(self):
        self.traj.plot_trajectory('test', sample_period=self.control_dt, filter_0_vel=False)


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
                              qp_formulation=ControllerMode.explicit,
                              joint_names=[joint_name],
                              goal=goal)

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

    def test_joint_goal_pr2(self, pr2_world: WorldTree):
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
        simulator = Simulator(world=pr2_world,
                              control_dt=0.02,
                              mpc_dt=0.05,
                              h=9,
                              solver=SupportedQPSolver.qpSWIFT,
                              qp_formulation=ControllerMode.explicit)
        simulator.add_joint_goal(
            joint_names=pr2_world.movable_joint_names[:1],
            # joint_names=pr2_world.movable_joint_names[14:15],
            # joint_names=pr2_world.movable_joint_names[:35],
            # joint_names=pr2_world.movable_joint_names[27:31],
            # joint_names=pr2_world.movable_joint_names[:-1],
            # joint_names=[
            #     pr2_world.movable_joint_names[27],
            #     pr2_world.movable_joint_names[31]
            # ],
            goal=goal)
        simulator.compile()

        np.random.seed(69)
        swap_distance = 2
        next_swap = swap_distance
        try:
            while god_map.time < sim_time:
                simulator.step()
                goal = np.random.rand() * 2 -1
                simulator.update_joint_goal(goal)
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

    def test_cart_goal_pr2(self, pr2_world: WorldTree):
        sim_time = 4
        goal = -0.9
        pr2_world.update_default_weights({Derivatives.velocity: 0.01,
                                          Derivatives.acceleration: 0.0,
                                          Derivatives.jerk: 0.0})
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
                              solver=SupportedQPSolver.proxsuite,
                              qp_formulation=ControllerMode.no_mpc)
        length = 10
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
                if god_map.time > next_swap:
                    goal *= -1
                    simulator.update_cart_goal(goal)
                    next_swap += swap_distance
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
            # SupportedQPSolver.piqp,
            # SupportedQPSolver.clarabel,
            # SupportedQPSolver.cvxopt,
            # SupportedQPSolver.osqp,
            # SupportedQPSolver.proxsuite,
            SupportedQPSolver.daqp,
        ]
        formulations = [
            # ControllerMode.explicit,
            ControllerMode.explicit_no_acc,
            ControllerMode.implicit,
            ControllerMode.no_mpc
        ]
        max_number_of_joints_total = len(pr2_world.movable_joint_names) - 1  # the last joint doesn't work
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
            ControllerMode.explicit_no_acc,
            ControllerMode.implicit,
            ControllerMode.no_mpc
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
