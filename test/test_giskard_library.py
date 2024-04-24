import numpy as np
import pytest

from giskardpy.data_types.data_types import PrefixName, Derivatives
from giskardpy.goals import motion_goal_manager
from giskardpy.goals.joint_goals import JointPositionList
from giskardpy.goals.motion_goal_manager import MotionGoalManager
from giskardpy.god_map import god_map
from giskardpy.model.joints import OmniDrive, PrismaticJoint
from giskardpy.model.links import Link, BoxGeometry
from giskardpy.model.world import WorldTree
from giskardpy.monitors.monitor_manager import MonitorManager
from giskardpy.qp.qp_controller import QPProblemBuilder
from giskardpy.symbol_manager import symbol_manager


@pytest.fixture(scope='module')
def create_world():
    return WorldTree()


@pytest.fixture()
def empty_world(create_world: WorldTree):
    create_world.clear()
    return create_world


@pytest.fixture()
def box_world(empty_world: WorldTree):
    box_name = PrefixName('box')
    root_link_name = PrefixName('map')
    joint_name = PrefixName('box_joint')

    world = WorldTree()
    with world.modify_world():
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
    assert joint_name in world.joints
    assert root_link_name in world.root_link_name
    assert box_name in world.links
    return world


class TestWorld:
    def test_compute_fk(self, box_world: WorldTree):
        joint_name = box_world.joint_names[0]
        box_name = box_world.link_names[-1]

        box_world.state[joint_name].position = 1
        box_world.notify_state_change()
        fk = box_world.compute_fk_point(root=box_world.root_link_name, tip=box_name).to_np()
        assert fk[0] == 1

    def test_goal(self, box_world: WorldTree):
        joint_name = box_world.joint_names[0]
        box_name = box_world.link_names[-1]
        dt = 0.05
        goal = 2

        joint_goal = JointPositionList(goal_state={joint_name: goal})

        monitor_manager = MonitorManager()
        god_map.monitor_manager = monitor_manager

        motion_goal_manager = MotionGoalManager()
        god_map.motion_goal_manager = motion_goal_manager
        motion_goal_manager.add_motion_goal(joint_goal)
        motion_goal_manager.init_task_state()

        eq, neq, neqd, lin_weight, quad_weight = motion_goal_manager.get_constraints_from_goals()
        controller = QPProblemBuilder(sample_period=dt,
                                      free_variables=list(box_world.free_variables.values()),
                                      equality_constraints=list(eq.values()))
        for i in range(100):
            parameters = controller.get_parameter_names()
            substitutions = symbol_manager.resolve_symbols(parameters)
            next_cmd = controller.get_cmd(substitutions)
            box_world.update_state(next_cmd, dt, Derivatives.jerk)
            box_world.notify_state_change()
            if box_world.state[joint_name].position >= goal-1e-3:
                break
        fk = box_world.compute_fk_point(root=box_world.root_link_name, tip=box_name).to_np()
        np.testing.assert_almost_equal(fk[0], goal, decimal=3)
