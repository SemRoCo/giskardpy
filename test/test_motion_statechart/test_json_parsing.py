import json

import numpy as np
import pytest

from giskardpy.executor import Executor
from giskardpy.model.collision_matrix_manager import (
    CollisionRequest,
    CollisionAvoidanceTypes,
)
from giskardpy.motion_statechart.context import BuildContext
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from giskardpy.motion_statechart.exceptions import (
    NodeNotFoundError,
)
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import (
    TrinaryCondition,
    EndMotion,
    CancelMotion,
)
from giskardpy.motion_statechart.motion_statechart import (
    MotionStatechart,
    LifeCycleState,
    ObservationState,
)
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from giskardpy.motion_statechart.test_nodes.test_nodes import (
    ConstTrueNode,
    TestNestedGoal,
)
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    KinematicStructureEntityKwargsTracker,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.spatial_types import Vector3, TransformationMatrix
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.spatial_types.spatial_types import (
    trinary_logic_and,
    trinary_logic_not,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import Body


def test_TrueMonitor():
    node = ConstTrueNode()
    json_data = node.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)
    node_copy = ConstTrueNode.from_json(new_json_data)
    assert node_copy.name == node.name


def test_CollisionRequest(pr2_world: World):
    robot = pr2_world.get_semantic_annotations_by_type(AbstractRobot)[0]
    collision_request = CollisionRequest(
        type_=CollisionAvoidanceTypes.AVOID_COLLISION,
        distance=0.2,
        body_group1=robot.bodies,
        body_group2=robot.bodies,
    )
    json_data = collision_request.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)
    tracker = KinematicStructureEntityKwargsTracker.from_world(pr2_world)
    kwargs = tracker.create_kwargs()
    collision_request_copy = CollisionRequest.from_json(new_json_data, **kwargs)
    assert collision_request_copy.type_ == collision_request.type_
    assert collision_request_copy.distance == collision_request.distance
    assert collision_request_copy.body_group1 == collision_request.body_group1
    assert collision_request_copy.body_group2 == collision_request.body_group2


def test_trinary_transition():
    msc = MotionStatechart()
    node1 = ConstTrueNode()
    node2 = ConstTrueNode()
    node3 = ConstTrueNode()
    node4 = ConstTrueNode()
    msc.add_node(node1)
    msc.add_node(node2)
    msc.add_node(node3)
    msc.add_node(node4)

    node1.start_condition = trinary_logic_and(
        node2.observation_variable,
        trinary_logic_and(
            node3.observation_variable, trinary_logic_not(node4.observation_variable)
        ),
    )
    condition = node1._start_condition
    json_data = condition.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)
    condition_copy = TrinaryCondition.from_json(new_json_data, motion_statechart=msc)
    assert condition_copy == condition


def test_to_json_joint_position_list(mini_world):
    connection = mini_world.connections[0]
    node = JointPositionList(
        goal_state=JointState({connection: 0.5}),
        threshold=0.5,
    )
    json_data = node.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)
    node_copy = JointPositionList.from_json(new_json_data, world=mini_world)
    assert node_copy.name == node.name
    assert node_copy.threshold == node.threshold
    assert node_copy.goal_state == node.goal_state


def test_start_condition(mini_world):
    msc = MotionStatechart()
    node1 = ConstTrueNode()
    msc.add_node(node1)
    node2 = ConstTrueNode()
    msc.add_node(node2)
    node3 = ConstTrueNode()
    msc.add_node(node3)
    end = ConstTrueNode()
    msc.add_node(end)

    node1.end_condition = node1.observation_variable
    node2.start_condition = node1.observation_variable
    node2.pause_condition = node3.observation_variable
    end.start_condition = trinary_logic_and(
        node2.observation_variable, node3.observation_variable
    )

    json_data = msc.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)
    msc_copy = MotionStatechart.from_json(new_json_data, world=mini_world)

    kin_sim = Executor(world=mini_world)
    kin_sim.compile(motion_statechart=msc_copy)
    for index, node in enumerate(msc.nodes):
        assert node.name == msc_copy.nodes[index].name
    assert len(msc.edges) == len(msc_copy.edges)
    for index, edge in enumerate(msc.edges):
        assert edge == msc_copy.edges[index]


def test_executing_json_parsed_statechart():
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("root"))
        tip = Body(name=PrefixedName("tip"))
        tip2 = Body(name=PrefixedName("tip2"))
        ul = DerivativeMap()
        ul.velocity = 1
        ll = DerivativeMap()
        ll.velocity = -1
        dof = DegreeOfFreedom(
            name=PrefixedName("dof", "a"), lower_limits=ll, upper_limits=ul
        )
        world.add_degree_of_freedom(dof)
        root_C_tip = RevoluteConnection(
            parent=root, child=tip, axis=Vector3.Z(), dof_name=dof.name
        )
        world.add_connection(root_C_tip)

        dof = DegreeOfFreedom(
            name=PrefixedName("dof", "b"), lower_limits=ll, upper_limits=ul
        )
        world.add_degree_of_freedom(dof)
        root_C_tip2 = RevoluteConnection(
            parent=root, child=tip2, axis=Vector3.Z(), dof_name=dof.name
        )
        world.add_connection(root_C_tip2)

    msc = MotionStatechart()

    task1 = JointPositionList(goal_state=JointState({root_C_tip: 0.5}))
    always_true = ConstTrueNode()
    msc.add_node(always_true)
    msc.add_node(task1)
    end = EndMotion()
    msc.add_node(end)

    task1.start_condition = always_true.observation_variable
    end.start_condition = trinary_logic_and(
        task1.observation_variable, always_true.observation_variable
    )

    json_data = msc.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)
    msc_copy = MotionStatechart.from_json(new_json_data, world=world)

    kin_sim = Executor(
        world=world,
        controller_config=QPControllerConfig.create_default_with_50hz(),
    )
    kin_sim.compile(motion_statechart=msc_copy)

    task1_copy = msc_copy.get_node_by_index(task1.index)
    end_copy = msc_copy.get_node_by_index(end.index)
    assert task1_copy.observation_state == ObservationStateValues.UNKNOWN
    assert end_copy.observation_state == ObservationStateValues.UNKNOWN
    assert task1_copy.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end_copy.life_cycle_state == LifeCycleValues.NOT_STARTED
    msc_copy.draw("muh.pdf")
    kin_sim.tick_until_end()
    msc_copy.draw("muh.pdf")
    assert task1_copy.observation_state == ObservationStateValues.TRUE
    assert end_copy.observation_state == ObservationStateValues.TRUE
    assert task1_copy.life_cycle_state == LifeCycleValues.RUNNING
    assert end_copy.life_cycle_state == LifeCycleValues.RUNNING

    life_cycle_json = msc_copy.life_cycle_state.to_json()
    json_str = json.dumps(life_cycle_json)
    life_cycle_json_copy = json.loads(json_str)
    life_cycle_copy = LifeCycleState.from_json(
        life_cycle_json_copy, motion_statechart=msc_copy
    )
    assert life_cycle_copy == msc_copy.life_cycle_state

    observation_json = msc_copy.observation_state.to_json()
    json_str = json.dumps(observation_json)
    observation_json_copy = json.loads(json_str)
    observation_copy = ObservationState.from_json(
        observation_json_copy, motion_statechart=msc_copy
    )
    assert observation_copy == msc_copy.observation_state


def test_cart_goal_simple(pr2_world: World):
    tip = pr2_world.get_kinematic_structure_entity_by_name("base_footprint")
    root = pr2_world.get_kinematic_structure_entity_by_name("odom_combined")
    tip_goal = TransformationMatrix.from_xyz_quaternion(pos_x=-0.2, reference_frame=tip)

    msc = MotionStatechart()
    cart_goal = CartesianPose(
        root_link=root,
        tip_link=tip,
        goal_pose=tip_goal,
    )
    msc.add_node(cart_goal)
    end = EndMotion()
    msc.add_node(end)
    end.start_condition = cart_goal.observation_variable

    json_data = msc.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)

    tracker = KinematicStructureEntityKwargsTracker.from_world(pr2_world)
    kwargs = tracker.create_kwargs()
    msc_copy = MotionStatechart.from_json(new_json_data, **kwargs)

    kin_sim = Executor(
        world=pr2_world,
        controller_config=QPControllerConfig.create_default_with_50hz(),
    )

    kin_sim.compile(motion_statechart=msc_copy)
    kin_sim.tick_until_end()

    fk = pr2_world.compute_forward_kinematics_np(root, tip)
    assert np.allclose(fk, tip_goal.to_np(), atol=cart_goal.threshold)


def test_compressed_copy_can_be_plotted(pr2_world: World):
    tip = pr2_world.get_kinematic_structure_entity_by_name("base_footprint")
    root = pr2_world.get_kinematic_structure_entity_by_name("odom_combined")
    tip_goal = TransformationMatrix.from_xyz_quaternion(pos_x=-0.2, reference_frame=tip)

    msc = MotionStatechart()
    cart_goal = CartesianPose(
        root_link=root,
        tip_link=tip,
        goal_pose=tip_goal,
    )
    msc.add_node(cart_goal)
    end = EndMotion()
    msc.add_node(end)
    end.start_condition = cart_goal.observation_variable

    msc._expand_goals(BuildContext.empty())
    json_data = msc.create_structure_copy().to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)

    msc_copy = MotionStatechart.from_json(new_json_data)
    msc_copy.draw("muh.pdf")


def test_nested_goals():
    msc = MotionStatechart()
    msc.add_node(
        sequence := Sequence(
            [
                ConstTrueNode(),
                TestNestedGoal(),
            ]
        )
    )
    msc.add_node(EndMotion.when_true(sequence))

    msc._expand_goals(BuildContext.empty())
    json_data = msc.create_structure_copy().to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)

    msc_copy = MotionStatechart.from_json(new_json_data)
    msc_copy._add_transitions()
    msc_copy.draw("muh.pdf")

    for node in msc.nodes:
        node_copy = msc_copy.get_node_by_index(node.index)
        assert node.index == node_copy.index
        if node.parent_node_index is not None:
            assert node.parent_node.unique_name == node_copy.parent_node.unique_name
        else:
            assert node_copy.parent_node_index is None


def test_cancel_motion():
    msc = MotionStatechart()
    msc.add_node(node := ConstTrueNode())
    msc.add_node(CancelMotion.when_true(node, exception=NodeNotFoundError(name="muh")))

    json_data = msc.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)
    msc_copy = MotionStatechart.from_json(new_json_data)

    kin_sim = Executor(
        world=World(),
    )

    kin_sim.compile(motion_statechart=msc_copy)

    with pytest.raises(Exception):
        kin_sim.tick_until_end()
