import numpy as np
import pytest

from giskardpy.data_types.data_types import PrefixName, Derivatives
from giskardpy.goals.cartesian_goals import CartesianPose
from giskardpy.goals.joint_goals import JointPositionList
from giskardpy.goals.motion_goal_manager import MotionGoalManager
from giskardpy.god_map import god_map
from giskardpy.model.joints import OmniDrive, PrismaticJoint
from giskardpy.model.links import Link, BoxGeometry
from giskardpy.model.world import WorldTree
from giskardpy.monitors.cartesian_monitors import PoseReached
from giskardpy.monitors.monitor_manager import MonitorManager
from giskardpy.qp.qp_controller import QPController
from giskardpy.symbol_manager import symbol_manager
import giskardpy.casadi_wrapper as cas


@pytest.fixture(scope='module')
def create_world():
    return god_map.world


@pytest.fixture()
def empty_world(create_world: WorldTree):
    create_world.clear()
    return create_world


@pytest.fixture()
def box_world_prismatic(empty_world: WorldTree):
    box_name = PrefixName('box')
    root_link_name = PrefixName('map')
    joint_name = PrefixName('box_joint')

    with empty_world.modify_world() as world:
        root_link = Link(root_link_name)
        world.add_link(root_link)

        box = Link(box_name)
        box_geometry = BoxGeometry(1, 1, 1)
        box.collisions.append(box_geometry)
        box.visuals.append(box_geometry)
        world.add_link(box)

        joint = PrismaticJoint(name=joint_name,
                               free_variable_name=joint_name,
                               parent_link_name=root_link_name,
                               child_link_name=box_name,
                               axis=(1, 0, 0),
                               lower_limits={Derivatives.position: -10,
                                             Derivatives.velocity: -1,
                                             Derivatives.acceleration: -np.inf,
                                             Derivatives.jerk: -30},
                               upper_limits={Derivatives.position: 10,
                                             Derivatives.velocity: 1,
                                             Derivatives.acceleration: np.inf,
                                             Derivatives.jerk: 30})
        world.add_joint(joint)
    assert joint_name in empty_world.joints
    assert root_link_name in empty_world.root_link_name
    assert box_name in empty_world.links
    return empty_world


@pytest.fixture()
def box_world(empty_world: WorldTree):
    box_name = PrefixName('box')
    root_link_name = PrefixName('map')
    joint_name = PrefixName('box_joint')

    with empty_world.modify_world() as world:
        root_link = Link(root_link_name)
        world.add_link(root_link)

        box = Link(box_name)
        box_geometry = BoxGeometry(1, 1, 1)
        box.collisions.append(box_geometry)
        box.visuals.append(box_geometry)
        world.add_link(box)

        joint = OmniDrive(name=joint_name,
                          parent_link_name=root_link_name,
                          child_link_name=box_name,
                          translation_limits={Derivatives.velocity: 1,
                                              Derivatives.acceleration: np.inf,
                                              Derivatives.jerk: 30},
                          rotation_limits={Derivatives.velocity: 1,
                                           Derivatives.acceleration: np.inf,
                                           Derivatives.jerk: 30})
        world.add_joint(joint)
    assert joint_name in empty_world.joints
    assert root_link_name in empty_world.root_link_name
    assert box_name in empty_world.links
    return empty_world


class TestWorld:
    def test_compute_fk(self, box_world_prismatic: WorldTree):
        joint_name = box_world_prismatic.joint_names[0]
        box_name = box_world_prismatic.link_names[-1]

        box_world_prismatic.state[joint_name].position = 1
        box_world_prismatic.notify_state_change()
        fk = box_world_prismatic.compute_fk_point(root=box_world_prismatic.root_link_name, tip=box_name).to_np()
        assert fk[0] == 1

    def test_joint_goal(self, box_world_prismatic: WorldTree):
        joint_name = box_world_prismatic.joint_names[0]
        box_name = box_world_prismatic.link_names[-1]
        dt = 0.05
        goal = 2

        joint_goal = JointPositionList(goal_state={joint_name: goal})

        god_map.motion_goal_manager.add_motion_goal(joint_goal)
        god_map.motion_goal_manager.init_task_state()

        eq, neq, neqd, lin_weight, quad_weight = god_map.motion_goal_manager.get_constraints_from_goals()
        controller = QPController(sample_period=dt)
        controller.init(free_variables=list(box_world_prismatic.free_variables.values()),
                        equality_constraints=eq)
        controller.compile()
        traj = []
        for i in range(100):
            parameters = controller.get_parameter_names()
            substitutions = symbol_manager.resolve_symbols(parameters)
            next_cmd = controller.get_cmd(substitutions)
            box_world_prismatic.update_state(next_cmd, dt, Derivatives.jerk)
            box_world_prismatic.notify_state_change()
            traj.append(box_world_prismatic.state[joint_name].position)
            if box_world_prismatic.state[joint_name].position >= goal - 1e-3:
                break
        fk = box_world_prismatic.compute_fk_point(root=box_world_prismatic.root_link_name, tip=box_name).to_np()
        np.testing.assert_almost_equal(fk[0], goal, decimal=3)

    def test_cart_goal(self, box_world: WorldTree):
        joint_name = box_world.joint_names[0]
        box_name = box_world.link_names[-1]
        dt = 0.05
        goal = [1, 1, 0]

        cart_goal = CartesianPose(root_link=box_world.root_link_name,
                                  tip_link=box_name,
                                  goal_pose=cas.TransMatrix.from_xyz_rpy(x=goal[0], y=goal[1], z=goal[2],
                                                                         reference_frame=box_world.root_link_name))

        god_map.motion_goal_manager.add_motion_goal(cart_goal)
        god_map.motion_goal_manager.init_task_state()

        eq, neq, neqd, lin_weight, quad_weight = god_map.motion_goal_manager.get_constraints_from_goals()
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
        fk = box_world.compute_fk_point(root=box_world.root_link_name, tip=box_name).to_np()
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
        god_map.monitor_manager.add_monitor(cart_monitor)

        cart_goal1 = CartesianPose(name='g1',
                                   root_link=box_world.root_link_name,
                                   tip_link=box_name,
                                   goal_pose=goal1,
                                   end_condition=cart_monitor.get_state_expression())
        cart_goal2 = CartesianPose(name='g2',
                                   root_link=box_world.root_link_name,
                                   tip_link=box_name,
                                   goal_pose=goal2,
                                   absolute=True,
                                   start_condition=cart_monitor.get_state_expression())


        god_map.motion_goal_manager.add_motion_goal(cart_goal1)
        god_map.motion_goal_manager.add_motion_goal(cart_goal2)

        god_map.monitor_manager.compile_monitors()
        god_map.motion_goal_manager.init_task_state()

        eq, neq, neqd, lin_weight, quad_weight = god_map.motion_goal_manager.get_constraints_from_goals()
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
            god_map.monitor_manager.evaluate_monitors()
            traj.append(box_world.state[joint_name].position)
            god_map.time += controller.sample_period
            god_map.control_cycle_counter += 1
        fk = box_world.compute_fk_point(root=box_world.root_link_name, tip=box_name).to_np()
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
        god_map.monitor_manager.add_monitor(cart_monitor)

        cart_goal1 = CartesianPose(name='g1',
                                   root_link=box_world.root_link_name,
                                   tip_link=box_name,
                                   goal_pose=goal1,
                                   end_condition=cart_monitor.get_state_expression())
        cart_goal2 = CartesianPose(name='g2',
                                   root_link=box_world.root_link_name,
                                   tip_link=box_name,
                                   goal_pose=goal2,
                                   start_condition=cart_monitor.get_state_expression())


        god_map.motion_goal_manager.add_motion_goal(cart_goal1)
        god_map.motion_goal_manager.add_motion_goal(cart_goal2)

        god_map.monitor_manager.compile_monitors()
        god_map.motion_goal_manager.init_task_state()

        eq, neq, neqd, lin_weight, quad_weight = god_map.motion_goal_manager.get_constraints_from_goals()
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
            god_map.monitor_manager.evaluate_monitors()
            traj.append((box_world.state[box_world.joints[joint_name].x_name].position,
                         box_world.state[box_world.joints[joint_name].y_name].position))
            god_map.time += controller.sample_period
            god_map.control_cycle_counter += 1
        fk = box_world.compute_fk_point(root=box_world.root_link_name, tip=box_name).to_np()
        np.testing.assert_almost_equal(fk[0], goal1.to_position().to_np()[0], decimal=2)
        np.testing.assert_almost_equal(fk[1], goal2.to_position().to_np()[1], decimal=2)
