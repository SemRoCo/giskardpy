import json
from dataclasses import dataclass

import numpy as np
import pytest
import time

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.data_types.exceptions import InvalidGoalException, NoQPControllerConfigException
from giskardpy.executor import Executor
from giskardpy.model.collision_matrix_manager import CollisionRequest
from giskardpy.model.collision_world_syncer import CollisionCheckerLib
from giskardpy.motion_statechart.binding_policy import GoalBindingPolicy
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from giskardpy.motion_statechart.exceptions import (
    NotInMotionStatechartError,
    InvalidSelfReferenceInStartCondition,
    InvalidVariableInCondition,
    NodeAlreadyInMotionStatechartError,
    NodeAlreadyHasParentGoalError,
    DuplicateNodeInGoalError,
    InvalidConditionError,
)
from giskardpy.motion_statechart.goals.collision_avoidance import (
    CollisionAvoidance,
)
from giskardpy.motion_statechart.goals.open_close import Open, Close
from giskardpy.motion_statechart.goals.templates import Sequence, Parallel
from giskardpy.motion_statechart.graph_node import (
    EndMotion,
    CancelMotion,
)
from giskardpy.motion_statechart.graph_node import ThreadPayloadMonitor
from giskardpy.motion_statechart.monitors.joint_monitors import JointPositionReached
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import (
    SetSeedConfiguration,
    SetOdometry,
)
from giskardpy.motion_statechart.monitors.payload_monitors import (
    Print,
    Pulse,
    CountSeconds,
    CountTicks,
)
from giskardpy.motion_statechart.motion_statechart import (
    MotionStatechart,
)
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianOrientation,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from giskardpy.motion_statechart.tasks.pointing import Pointing
from giskardpy.motion_statechart.test_nodes.test_nodes import (
    ChangeStateOnEvents,
    ConstTrueNode,
    TestGoal,
    TestNestedGoal,
    ConstFalseNode,
)
from giskardpy.qp.exceptions import HardConstraintsViolatedException
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.utils.math import angle_between_vector
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    KinematicStructureEntityKwargsTracker,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.factories import (
    DoorFactory,
    SemanticPositionDescription,
    HandleFactory,
    HorizontalSemanticDirection,
    VerticalSemanticDirection,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle
from semantic_digital_twin.spatial_types import TransformationMatrix, Vector3
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.spatial_types.spatial_types import (
    trinary_logic_and,
    trinary_logic_not,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
    ActiveConnection1DOF,
    FixedConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.geometry import Cylinder
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


def test_condition_to_str():
    msc = MotionStatechart()
    node1 = ConstTrueNode()
    msc.add_node(node1)
    node2 = ConstTrueNode()
    msc.add_node(node2)
    node3 = ConstTrueNode()
    msc.add_node(node3)
    end = EndMotion()
    msc.add_node(end)

    end.start_condition = cas.trinary_logic_and(
        node1.observation_variable,
        cas.trinary_logic_or(
            node2.observation_variable,
            cas.trinary_logic_not(node3.observation_variable),
        ),
    )
    a = str(end._start_condition)
    assert a == '("ConstTrueNode#0" and ("ConstTrueNode#1" or not "ConstTrueNode#2"))'


def test_motion_statechart_to_dot():
    msc = MotionStatechart()
    node1 = ConstTrueNode()
    msc.add_node(node1)
    node2 = ConstTrueNode()
    msc.add_node(node2)
    end = EndMotion()
    msc.add_node(end)
    node1.end_condition = node2.observation_variable
    end.start_condition = cas.trinary_logic_and(
        node1.observation_variable, node2.observation_variable
    )
    msc.draw("muh.pdf")


@pytest.mark.skip(reason="not implemented yet")
def test_self_start_condition():
    pass


@pytest.mark.skip(reason="not implemented yet")
def test_all_conditions_with_goals():
    pass


@pytest.mark.skip(reason="not implemented yet")
def test_all_conditions_with_nodes():
    pass


@pytest.mark.skip(reason="not implemented yet")
def test_transition_hooks():
    pass


@pytest.mark.skip(reason="not implemented yet")
def test_optionality_of_qp_controller_in_compile():
    pass


@pytest.mark.skip(reason="not implemented yet")
def test_state_deletion():
    pass


def test_motion_statechart():
    msc = MotionStatechart()

    node1 = ConstTrueNode()
    msc.add_node(node1)
    node2 = ConstTrueNode()
    msc.add_node(node2)
    node3 = ConstTrueNode()
    msc.add_node(node3)
    end = EndMotion()
    msc.add_node(end)

    node1.start_condition = cas.trinary_logic_or(
        node3.observation_variable, node2.observation_variable
    )
    end.start_condition = node1.observation_variable

    kin_sim = Executor(world=World())
    kin_sim.compile(motion_statechart=msc)

    assert len(msc.nodes) == 4
    assert len(msc.edges) == 3
    kin_sim.tick_until_end()

    assert len(msc.history) == 5
    # %% node1
    assert msc.history.get_life_cycle_history_of_node(node1) == [
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
    ]
    assert msc.history.get_observation_history_of_node(node1) == [
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.TRUE,
        ObservationStateValues.TRUE,
    ]
    # %% node2
    assert msc.history.get_life_cycle_history_of_node(node2) == [
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
    ]
    assert msc.history.get_observation_history_of_node(node2) == [
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.TRUE,
        ObservationStateValues.TRUE,
        ObservationStateValues.TRUE,
    ]
    # %% node3
    assert msc.history.get_life_cycle_history_of_node(node3) == [
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
    ]
    assert msc.history.get_observation_history_of_node(node3) == [
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.TRUE,
        ObservationStateValues.TRUE,
        ObservationStateValues.TRUE,
    ]
    # %% end
    assert msc.history.get_life_cycle_history_of_node(end) == [
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
    ]
    assert msc.history.get_observation_history_of_node(end) == [
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.TRUE,
    ]


def test_sequence_goal():
    msc = MotionStatechart()
    node = Sequence(
        nodes=[
            ConstTrueNode(),
            ConstTrueNode(),
            ConstTrueNode(),
            ConstTrueNode(),
        ]
    )
    msc.add_node(node)
    msc.add_node(EndMotion.when_true(node))

    kin_sim = Executor(world=World())
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()
    msc.draw("muh.pdf")
    assert kin_sim.control_cycles == 7
    assert msc.nodes[0].life_cycle_state == LifeCycleValues.RUNNING
    assert msc.nodes[1].life_cycle_state == LifeCycleValues.RUNNING
    assert msc.nodes[2].life_cycle_state == LifeCycleValues.DONE
    assert msc.nodes[3].life_cycle_state == LifeCycleValues.DONE
    assert msc.nodes[4].life_cycle_state == LifeCycleValues.DONE
    assert msc.nodes[5].life_cycle_state == LifeCycleValues.DONE


def test_print():
    msc = MotionStatechart()
    print_node1 = Print(name="cow", message="muh")
    msc.add_node(print_node1)
    print_node2 = Print(name="cow2", message="muh")
    msc.add_node(print_node2)

    node1 = ConstTrueNode()
    msc.add_node(node1)
    end = EndMotion()
    msc.add_node(end)

    node1.start_condition = print_node1.observation_variable
    print_node2.start_condition = node1.observation_variable
    end.start_condition = print_node2.observation_variable

    kin_sim = Executor(world=World())
    kin_sim.compile(motion_statechart=msc)

    assert len(msc.nodes) == 4
    assert len(msc.edges) == 3

    assert print_node1.observation_state == ObservationStateValues.UNKNOWN
    assert node1.observation_state == ObservationStateValues.UNKNOWN
    assert print_node2.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN

    assert print_node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert print_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    kin_sim.tick()
    assert print_node1.observation_state == ObservationStateValues.UNKNOWN
    assert node1.observation_state == ObservationStateValues.UNKNOWN
    assert print_node2.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN

    assert print_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert print_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    kin_sim.tick()
    assert print_node1.observation_state == ObservationStateValues.TRUE
    assert node1.observation_state == ObservationStateValues.UNKNOWN
    assert print_node2.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN

    assert print_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert print_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    kin_sim.tick()
    assert print_node1.observation_state == ObservationStateValues.TRUE
    assert node1.observation_state == ObservationStateValues.TRUE
    assert print_node2.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN

    assert print_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert print_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    kin_sim.tick()
    assert print_node1.observation_state == ObservationStateValues.TRUE
    assert node1.observation_state == ObservationStateValues.TRUE
    assert print_node2.observation_state == ObservationStateValues.TRUE
    assert end.observation_state == ObservationStateValues.UNKNOWN

    assert print_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert print_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert not msc.is_end_motion()

    kin_sim.tick()
    assert print_node1.observation_state == ObservationStateValues.TRUE
    assert node1.observation_state == ObservationStateValues.TRUE
    assert print_node2.observation_state == ObservationStateValues.TRUE
    assert end.observation_state == ObservationStateValues.TRUE

    assert print_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert print_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert msc.is_end_motion()


def test_cancel_motion():
    msc = MotionStatechart()
    node1 = ConstTrueNode()
    msc.add_node(node1)
    cancel = CancelMotion(exception=Exception("test"))
    msc.add_node(cancel)
    cancel.start_condition = node1.observation_variable

    kin_sim = Executor(world=World())
    kin_sim.compile(motion_statechart=msc)

    kin_sim.tick()  # first tick, cancel motion node1 turns true
    kin_sim.tick()  # second tick, cancel goes into running
    with pytest.raises(Exception):
        kin_sim.tick()  # third tick, cancel goes true and triggers
    msc.draw("muh.pdf")


def test_draw_with_invisible_node():
    msc = MotionStatechart()
    msc.add_nodes(
        [
            sequence := Sequence(
                nodes=[s1n1 := ConstTrueNode(), s1n2 := ConstTrueNode()]
            ),
            sequence2 := Sequence(
                nodes=[s2n1 := ConstTrueNode(), s2n2 := ConstTrueNode()]
            ),
        ]
    )
    msc.add_node(EndMotion.when_all_true(msc.nodes))

    sequence.plot_specs.visible = False
    s1n2.plot_specs.visible = False
    s2n2.plot_specs.visible = False

    kin_sim = Executor(world=World())
    kin_sim.compile(motion_statechart=msc)
    msc.draw("muh.pdf")


def test_joint_goal():
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
            parent=root, child=tip, axis=cas.Vector3.Z(), dof_name=dof.name
        )
        world.add_connection(root_C_tip)

        dof = DegreeOfFreedom(
            name=PrefixedName("dof", "b"), lower_limits=ll, upper_limits=ul
        )
        world.add_degree_of_freedom(dof)
        root_C_tip2 = RevoluteConnection(
            parent=root, child=tip2, axis=cas.Vector3.Z(), dof_name=dof.name
        )
        world.add_connection(root_C_tip2)

    msc = MotionStatechart()

    task1 = JointPositionList(goal_state=JointState({root_C_tip: 1}))
    always_true = ConstTrueNode()
    msc.add_node(always_true)
    msc.add_node(task1)
    end = EndMotion()
    msc.add_node(end)

    task1.start_condition = always_true.observation_variable
    end.start_condition = cas.trinary_logic_and(
        task1.observation_variable, always_true.observation_variable
    )

    kin_sim = Executor(
        world=world,
        controller_config=QPControllerConfig.create_default_with_50hz(),
    )
    kin_sim.compile(motion_statechart=msc)

    assert task1.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN
    assert task1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    msc.draw("muh.pdf")
    kin_sim.tick_until_end()
    msc.draw("muh.pdf")
    assert len(msc.history) == 6
    assert (
        msc.history.get_observation_history_of_node(task1)[-1]
        == ObservationStateValues.TRUE
    )
    assert (
        msc.history.get_observation_history_of_node(end)[-1]
        == ObservationStateValues.TRUE
    )
    assert (
        msc.history.get_life_cycle_history_of_node(task1)[-1] == LifeCycleValues.RUNNING
    )
    assert (
        msc.history.get_life_cycle_history_of_node(end)[-1] == LifeCycleValues.RUNNING
    )


def test_reset():
    msc = MotionStatechart()
    node1 = ConstTrueNode()
    msc.add_node(node1)
    node2 = ConstTrueNode()
    msc.add_node(node2)
    node3 = ConstTrueNode()
    msc.add_node(node3)
    end = EndMotion()
    msc.add_node(end)
    node1.reset_condition = node2.observation_variable
    node2.start_condition = node1.observation_variable
    node3.start_condition = node2.observation_variable
    node2.end_condition = node2.observation_variable
    end.start_condition = cas.trinary_logic_and(
        node1.observation_variable,
        node2.observation_variable,
        node3.observation_variable,
    )

    kin_sim = Executor(world=World())
    kin_sim.compile(motion_statechart=msc)
    msc.draw("muh.pdf")

    kin_sim.tick()
    assert node1.observation_state == ObservationStateValues.UNKNOWN
    assert node2.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    kin_sim.tick()
    assert node1.observation_state == ObservationStateValues.TRUE
    assert node2.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node2.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    kin_sim.tick()
    assert node1.observation_state == ObservationStateValues.TRUE
    assert node2.observation_state == ObservationStateValues.TRUE
    assert end.observation_state == ObservationStateValues.UNKNOWN
    assert node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert node2.life_cycle_state == LifeCycleValues.DONE
    assert node3.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    kin_sim.tick()
    assert node1.observation_state == ObservationStateValues.UNKNOWN
    assert node2.observation_state == ObservationStateValues.TRUE
    assert node3.observation_state == ObservationStateValues.TRUE
    assert end.observation_state == ObservationStateValues.UNKNOWN
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node2.life_cycle_state == LifeCycleValues.DONE
    assert node3.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    kin_sim.tick()
    assert node1.observation_state == ObservationStateValues.TRUE
    assert node2.observation_state == ObservationStateValues.TRUE
    assert node3.observation_state == ObservationStateValues.TRUE
    assert end.observation_state == ObservationStateValues.UNKNOWN
    assert node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert node2.life_cycle_state == LifeCycleValues.DONE
    assert node3.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert not msc.is_end_motion()

    kin_sim.tick()
    assert node1.observation_state == ObservationStateValues.UNKNOWN
    assert node2.observation_state == ObservationStateValues.TRUE
    assert node3.observation_state == ObservationStateValues.TRUE
    assert end.observation_state == ObservationStateValues.TRUE
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node2.life_cycle_state == LifeCycleValues.DONE
    assert node3.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert msc.is_end_motion()


def test_nested_goals():
    msc = MotionStatechart()

    node1 = ConstTrueNode()
    msc.add_node(node1)

    outer = TestNestedGoal()
    msc.add_node(outer)
    outer.start_condition = node1.observation_variable

    end = EndMotion()
    msc.add_node(end)
    end.start_condition = outer.observation_variable

    json_data = msc.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)
    msc_copy = MotionStatechart.from_json(new_json_data)

    for node in msc.nodes:
        assert node.index == msc_copy.get_node_by_index(node.index).index

    kin_sim = Executor(world=World())
    node1 = msc_copy.get_nodes_by_type(ConstTrueNode)[0]
    outer = msc_copy.get_nodes_by_type(TestNestedGoal)[0]
    end = msc_copy.get_nodes_by_type(EndMotion)[0]
    kin_sim.compile(motion_statechart=msc_copy)
    msc_copy.draw("muh.pdf")
    assert node1.observation_state == ObservationStateValues.UNKNOWN
    assert outer.inner.sub_node1.observation_state == ObservationStateValues.UNKNOWN
    assert outer.inner.sub_node2.observation_state == ObservationStateValues.UNKNOWN
    assert outer.inner.observation_state == ObservationStateValues.UNKNOWN
    assert outer.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN

    assert node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert outer.inner.sub_node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert outer.inner.sub_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert outer.inner.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert outer.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc_copy.is_end_motion()

    kin_sim.tick()
    msc_copy.draw("muh.pdf")
    assert node1.observation_state == ObservationStateValues.UNKNOWN
    assert outer.inner.sub_node1.observation_state == ObservationStateValues.UNKNOWN
    assert outer.inner.sub_node2.observation_state == ObservationStateValues.UNKNOWN
    assert outer.inner.observation_state == ObservationStateValues.UNKNOWN
    assert outer.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN

    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.inner.sub_node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert outer.inner.sub_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert outer.inner.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert outer.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc_copy.is_end_motion()

    kin_sim.tick()
    msc_copy.draw("muh.pdf")
    assert node1.observation_state == ObservationStateValues.TRUE
    assert outer.inner.sub_node1.observation_state == ObservationStateValues.UNKNOWN
    assert outer.inner.sub_node2.observation_state == ObservationStateValues.UNKNOWN
    assert outer.inner.observation_state == ObservationStateValues.UNKNOWN
    assert outer.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN

    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.inner.sub_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.inner.sub_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert outer.inner.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc_copy.is_end_motion()

    kin_sim.tick()
    msc_copy.draw("muh.pdf")
    assert node1.observation_state == ObservationStateValues.TRUE
    assert outer.inner.sub_node1.observation_state == ObservationStateValues.TRUE
    assert outer.inner.sub_node2.observation_state == ObservationStateValues.UNKNOWN
    assert outer.inner.observation_state == ObservationStateValues.UNKNOWN
    assert outer.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN

    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.inner.sub_node1.life_cycle_state == LifeCycleValues.DONE
    assert outer.inner.sub_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.inner.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc_copy.is_end_motion()

    kin_sim.tick()
    msc_copy.draw("muh.pdf")
    assert node1.observation_state == ObservationStateValues.TRUE
    assert outer.inner.sub_node1.observation_state == ObservationStateValues.TRUE
    assert outer.inner.sub_node2.observation_state == ObservationStateValues.TRUE
    assert outer.inner.observation_state == ObservationStateValues.UNKNOWN
    assert outer.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN

    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.inner.sub_node1.life_cycle_state == LifeCycleValues.DONE
    assert outer.inner.sub_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.inner.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc_copy.is_end_motion()

    kin_sim.tick()
    msc_copy.draw("muh.pdf")
    assert node1.observation_state == ObservationStateValues.TRUE
    assert outer.inner.sub_node1.observation_state == ObservationStateValues.TRUE
    assert outer.inner.sub_node2.observation_state == ObservationStateValues.TRUE
    assert outer.inner.observation_state == ObservationStateValues.TRUE
    assert outer.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN

    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.inner.sub_node1.life_cycle_state == LifeCycleValues.DONE
    assert outer.inner.sub_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.inner.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc_copy.is_end_motion()

    kin_sim.tick()
    msc_copy.draw("muh.pdf")
    assert node1.observation_state == ObservationStateValues.TRUE
    assert outer.inner.sub_node1.observation_state == ObservationStateValues.TRUE
    assert outer.inner.sub_node2.observation_state == ObservationStateValues.TRUE
    assert outer.inner.observation_state == ObservationStateValues.TRUE
    assert outer.observation_state == ObservationStateValues.TRUE
    assert end.observation_state == ObservationStateValues.UNKNOWN

    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.inner.sub_node1.life_cycle_state == LifeCycleValues.DONE
    assert outer.inner.sub_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.inner.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert not msc_copy.is_end_motion()

    kin_sim.tick()
    msc_copy.draw("muh.pdf")
    assert node1.observation_state == ObservationStateValues.TRUE
    assert outer.inner.sub_node1.observation_state == ObservationStateValues.TRUE
    assert outer.inner.sub_node2.observation_state == ObservationStateValues.TRUE
    assert outer.inner.observation_state == ObservationStateValues.TRUE
    assert outer.observation_state == ObservationStateValues.TRUE
    assert end.observation_state == ObservationStateValues.TRUE

    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.inner.sub_node1.life_cycle_state == LifeCycleValues.DONE
    assert outer.inner.sub_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.inner.life_cycle_state == LifeCycleValues.RUNNING
    assert outer.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert msc_copy.is_end_motion()


@dataclass(eq=False, repr=False)
class _TestThreadMonitor(ThreadPayloadMonitor):
    delay: float = 0.05
    return_value: float = ObservationStateValues.TRUE

    def _compute_observation(self):
        time.sleep(self.delay)
        return self.return_value


def test_thread_payload_monitor_non_blocking_and_caching():
    msc = MotionStatechart()
    mon = _TestThreadMonitor(
        delay=0.05,
        return_value=ObservationStateValues.TRUE,
    )
    msc.add_node(mon)
    # First call should be non-blocking and return Unknown until worker completes at least once
    start = time.perf_counter()
    val0 = mon.compute_observation()
    elapsed = time.perf_counter() - start
    assert elapsed < mon.delay / 4.0
    assert val0 == ObservationStateValues.UNKNOWN
    # Wait for worker to finish and cache
    time.sleep(mon.delay * 2)
    val1 = mon.compute_observation()
    assert val1 == ObservationStateValues.TRUE


@pytest.mark.skip(reason="Not working yet")
def test_thread_payload_monitor_integration():
    msc = MotionStatechart()
    mon = _TestThreadMonitor(
        delay=0.03,
        return_value=ObservationStateValues.TRUE,
    )
    msc.add_node(mon)
    end = EndMotion()
    msc.add_node(end)
    end.start_condition = mon.observation_variable

    kin_sim = Executor(world=World())

    kin_sim.compile(motion_statechart=msc)

    # tick 1: monitor not started yet becomes RUNNING; end not started
    kin_sim.tick()
    assert mon.observation_state == ObservationStateValues.UNKNOWN
    assert mon.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED

    # tick 2: compute_observation is triggered asynchronously; still Unknown immediately
    kin_sim.tick()
    assert mon.observation_state == ObservationStateValues.UNKNOWN
    assert mon.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED

    # allow background to finish and propagate on next tick
    time.sleep(mon.delay * 2)
    kin_sim.tick()
    assert mon.observation_state == ObservationStateValues.TRUE
    assert end.life_cycle_state == LifeCycleValues.RUNNING

    # next tick the EndMotion should turn true
    kin_sim.tick()
    assert end.observation_state == ObservationStateValues.TRUE


def test_goal():
    msc = MotionStatechart()

    node1 = ConstTrueNode()
    msc.add_node(node1)

    goal = TestGoal()
    msc.add_node(goal)

    goal.start_condition = node1.observation_variable

    end = EndMotion()
    msc.add_node(end)
    end.start_condition = goal.observation_variable

    kin_sim = Executor(world=World())

    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()
    assert len(msc.history) == 7
    # %% goal
    assert msc.history.get_life_cycle_history_of_node(goal) == [
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
    ]
    assert msc.history.get_observation_history_of_node(goal) == [
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.TRUE,
        ObservationStateValues.TRUE,
    ]
    # %% node1
    assert msc.history.get_life_cycle_history_of_node(node1) == [
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
    ]
    assert msc.history.get_observation_history_of_node(node1) == [
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.TRUE,
        ObservationStateValues.TRUE,
        ObservationStateValues.TRUE,
        ObservationStateValues.TRUE,
        ObservationStateValues.TRUE,
    ]
    # %% sub_node1
    assert msc.history.get_life_cycle_history_of_node(goal.sub_node1) == [
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.RUNNING,
        LifeCycleValues.DONE,
        LifeCycleValues.DONE,
        LifeCycleValues.DONE,
        LifeCycleValues.DONE,
    ]
    assert msc.history.get_observation_history_of_node(goal.sub_node1) == [
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.TRUE,
        ObservationStateValues.TRUE,
        ObservationStateValues.TRUE,
        ObservationStateValues.TRUE,
    ]
    # %% sub_node2
    assert msc.history.get_life_cycle_history_of_node(goal.sub_node2) == [
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
    ]
    assert msc.history.get_observation_history_of_node(goal.sub_node2) == [
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.TRUE,
        ObservationStateValues.TRUE,
        ObservationStateValues.TRUE,
    ]
    # %% sub_node2
    assert msc.history.get_life_cycle_history_of_node(end) == [
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.RUNNING,
        LifeCycleValues.RUNNING,
    ]
    assert msc.history.get_observation_history_of_node(end) == [
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.UNKNOWN,
        ObservationStateValues.TRUE,
    ]
    msc.draw("muh.pdf")


def test_set_seed_configuration(pr2_world):
    msc = MotionStatechart()
    goal = 0.1

    connection: ActiveConnection1DOF = pr2_world.get_connection_by_name(
        "torso_lift_joint"
    )

    node1 = SetSeedConfiguration(seed_configuration=JointState({connection: goal}))
    end = EndMotion()
    msc.add_node(node1)
    msc.add_node(end)
    node1.end_condition = node1.observation_variable
    end.start_condition = node1.observation_variable

    kin_sim = Executor(world=pr2_world)
    kin_sim.compile(motion_statechart=msc)

    kin_sim.tick_until_end()
    assert node1.observation_state == ObservationStateValues.TRUE
    assert node1.life_cycle_state == LifeCycleValues.DONE
    assert end.observation_state == ObservationStateValues.TRUE
    assert end.life_cycle_state == LifeCycleValues.RUNNING

    assert np.isclose(connection.position, goal)


def test_set_seed_odometry(pr2_world):
    msc = MotionStatechart()

    goal = TransformationMatrix.from_xyz_rpy(
        x=1, y=-1, z=1, roll=1, pitch=1, yaw=1, reference_frame=pr2_world.root
    )
    expected = TransformationMatrix.from_xyz_rpy(
        x=1, y=-1, yaw=1, reference_frame=pr2_world.root
    )

    node1 = SetOdometry(
        base_pose=goal,
    )
    end = EndMotion()
    msc.add_node(node1)
    msc.add_node(end)
    node1.end_condition = node1.observation_variable
    end.start_condition = node1.observation_variable

    kin_sim = Executor(world=pr2_world)
    kin_sim.compile(motion_statechart=msc)

    kin_sim.tick_until_end()
    assert node1.observation_state == ObservationStateValues.TRUE
    assert node1.life_cycle_state == LifeCycleValues.DONE
    assert end.observation_state == ObservationStateValues.TRUE
    assert end.life_cycle_state == LifeCycleValues.RUNNING

    assert np.allclose(
        expected.to_np(),
        pr2_world.compute_forward_kinematics_np(
            pr2_world.root, node1.odom_connection.child
        ),
    )


def test_continuous_joint(pr2_world):
    msc = MotionStatechart()
    joint_goal = JointPositionList(
        goal_state=JointState.from_str_dict(
            {
                "r_wrist_roll_joint": -np.pi,
                "l_wrist_roll_joint": -2.1 * np.pi,
            },
            world=pr2_world,
        ),
    )
    msc.add_node(joint_goal)
    end = EndMotion()
    msc.add_node(end)
    end.start_condition = joint_goal.observation_variable

    kin_sim = Executor(
        world=pr2_world,
        controller_config=QPControllerConfig.create_default_with_50hz(),
    )
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()


def test_revolute_joint(pr2_world):
    msc = MotionStatechart()
    joint_goal = JointPositionList(
        goal_state=JointState.from_str_dict(
            {
                "head_pan_joint": 0.041880780651479044,
                "head_tilt_joint": -0.37,
            },
            world=pr2_world,
        ),
    )
    msc.add_node(joint_goal)
    end = EndMotion()
    msc.add_node(end)
    end.start_condition = joint_goal.observation_variable

    kin_sim = Executor(
        world=pr2_world,
        controller_config=QPControllerConfig.create_default_with_50hz(),
    )
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()


def test_cart_goal_1eef(pr2_world: World):
    tip = pr2_world.get_kinematic_structure_entity_by_name("r_gripper_tool_frame")
    root = pr2_world.get_kinematic_structure_entity_by_name("base_footprint")
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

    kin_sim = Executor(
        world=pr2_world,
        controller_config=QPControllerConfig.create_default_with_50hz(),
    )
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()


def test_cart_goal_sequence_at_build(pr2_world: World):
    tip = pr2_world.get_kinematic_structure_entity_by_name("base_footprint")
    root = pr2_world.get_kinematic_structure_entity_by_name("odom_combined")

    tip_goal1 = TransformationMatrix.from_xyz_quaternion(
        pos_x=-0.2, reference_frame=tip
    )
    tip_goal2 = TransformationMatrix.from_xyz_quaternion(pos_x=0.2, reference_frame=tip)

    msc = MotionStatechart()
    cart_goal1 = CartesianPose(
        root_link=root,
        tip_link=tip,
        goal_pose=tip_goal1,
    )
    msc.add_node(cart_goal1)

    cart_goal2 = CartesianPose(
        root_link=root,
        tip_link=tip,
        goal_pose=tip_goal2,
        binding_policy=GoalBindingPolicy.Bind_at_build,
    )
    msc.add_node(cart_goal2)

    cart_goal1.end_condition = cart_goal1.observation_variable
    cart_goal2.start_condition = cart_goal1.observation_variable

    end = EndMotion()
    msc.add_node(end)
    end.start_condition = cas.trinary_logic_and(
        cart_goal1.observation_variable, cart_goal2.observation_variable
    )

    kin_sim = Executor(
        world=pr2_world,
        controller_config=QPControllerConfig.create_default_with_50hz(),
    )

    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()

    fk = pr2_world.compute_forward_kinematics_np(root, tip)
    assert np.allclose(fk, tip_goal2.to_np(), atol=cart_goal2.threshold)


def test_cart_goal_sequence_on_start(pr2_world: World):
    tip = pr2_world.get_kinematic_structure_entity_by_name("base_footprint")
    root = pr2_world.get_kinematic_structure_entity_by_name("odom_combined")

    tip_goal1 = TransformationMatrix.from_xyz_quaternion(
        pos_x=-0.2, reference_frame=tip
    )
    tip_goal2 = TransformationMatrix.from_xyz_quaternion(pos_x=0.2, reference_frame=tip)

    msc = MotionStatechart()
    cart_goal1 = CartesianPose(
        root_link=root,
        tip_link=tip,
        goal_pose=tip_goal1,
    )
    msc.add_node(cart_goal1)

    cart_goal2 = CartesianPose(
        root_link=root,
        tip_link=tip,
        goal_pose=tip_goal2,
    )
    msc.add_node(cart_goal2)

    cart_goal1.end_condition = cart_goal1.observation_variable
    cart_goal2.start_condition = cart_goal1.observation_variable

    end = EndMotion()
    msc.add_node(end)
    end.start_condition = cas.trinary_logic_and(
        cart_goal1.observation_variable, cart_goal2.observation_variable
    )

    kin_sim = Executor(
        world=pr2_world,
        controller_config=QPControllerConfig.create_default_with_50hz(),
    )
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()

    fk = pr2_world.compute_forward_kinematics_np(root, tip)
    expected = np.eye(4)
    assert np.allclose(fk, expected, atol=cart_goal2.threshold)


def test_CartesianOrientation(pr2_world: World):
    tip = pr2_world.get_kinematic_structure_entity_by_name("base_footprint")
    root = pr2_world.get_kinematic_structure_entity_by_name("odom_combined")

    tip_goal = cas.RotationMatrix.from_axis_angle(
        cas.Vector3.Z(), 4.0, reference_frame=tip
    )

    msc = MotionStatechart()
    cart_goal = CartesianOrientation(
        root_link=root,
        tip_link=tip,
        goal_orientation=tip_goal,
    )
    msc.add_node(cart_goal)
    end = EndMotion()
    msc.add_node(end)
    end.start_condition = cart_goal.observation_variable

    kin_sim = Executor(
        world=pr2_world,
        controller_config=QPControllerConfig.create_default_with_50hz(),
    )
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()

    fk = pr2_world.compute_forward_kinematics_np(root, tip)
    assert np.allclose(fk, tip_goal.to_np(), atol=cart_goal.threshold)


def test_pointing(pr2_world: World):
    tip = pr2_world.get_kinematic_structure_entity_by_name("r_gripper_tool_frame")
    root = pr2_world.get_kinematic_structure_entity_by_name("odom_combined")

    msc = MotionStatechart()

    goal_point = cas.Point3(2, 0, 0, reference_frame=root)
    pointing_axis = cas.Vector3.X(reference_frame=tip)

    pointing = Pointing(
        root_link=root,
        tip_link=tip,
        goal_point=goal_point,
        pointing_axis=pointing_axis,
    )
    msc.add_node(pointing)
    end = EndMotion()
    msc.add_node(end)
    end.start_condition = pointing.observation_variable

    kin_sim = Executor(
        world=pr2_world,
        controller_config=QPControllerConfig.create_default_with_50hz(),
    )
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()


def test_align_planes(pr2_world: World):
    tip = pr2_world.get_kinematic_structure_entity_by_name("r_gripper_tool_frame")
    root = pr2_world.get_kinematic_structure_entity_by_name("odom_combined")

    msc = MotionStatechart()

    goal_normal = cas.Vector3.X(reference_frame=root)
    tip_normal = cas.Vector3.Y(reference_frame=tip)

    align_planes = AlignPlanes(
        root_link=root, tip_link=tip, goal_normal=goal_normal, tip_normal=tip_normal
    )
    msc.add_node(align_planes)

    end = EndMotion()
    msc.add_node(end)
    end.start_condition = align_planes.observation_variable

    kin_sim = Executor(
        world=pr2_world,
        controller_config=QPControllerConfig.create_default_with_50hz(),
    )
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()

    # Check if the angle between normal vectors is below the threshold
    root_V_goal_normal = pr2_world.transform(
        target_frame=root, spatial_object=goal_normal
    )
    root_V_goal_normal.scale(1)
    root_V_tip_normal = pr2_world.transform(
        target_frame=root, spatial_object=tip_normal
    )
    root_V_tip_normal.scale(1)
    v_tip = root_V_tip_normal.to_np()[:3]
    v_goal = root_V_goal_normal.to_np()[:3]

    eps = 1e-9
    assert np.linalg.norm(v_goal) > eps, "goal normal became zero-length"
    assert np.linalg.norm(v_tip) > eps, "tip normal became zero-length"

    angle = angle_between_vector(v_tip, v_goal)

    assert (
        angle <= align_planes.threshold
    ), f"AlignPlanes failed: final angle {angle:.6f} rad > threshold {align_planes.threshold:.6f} rad"


def test_transition_triggers():
    msc = MotionStatechart()

    changer = ChangeStateOnEvents()
    msc.add_node(changer)

    node1 = Pulse()
    msc.add_node(node1)

    node2 = Pulse()
    msc.add_node(node2)
    node2.start_condition = node1.observation_variable

    node3 = Pulse()
    msc.add_node(node3)
    node3.start_condition = cas.trinary_logic_and(
        cas.trinary_logic_not(node1.observation_variable),
        cas.trinary_logic_not(node2.observation_variable),
    )

    node4 = Pulse()
    msc.add_node(node4)
    node4.start_condition = node3.observation_variable

    changer.start_condition = node1.observation_variable
    changer.pause_condition = node2.observation_variable
    changer.end_condition = node3.observation_variable
    changer.reset_condition = node4.observation_variable

    kin_sim = Executor(world=World())
    kin_sim.compile(motion_statechart=msc)

    assert changer.state is None

    kin_sim.tick()
    msc.draw("muh.pdf")
    kin_sim.tick()
    msc.draw("muh.pdf")
    assert changer.life_cycle_state == LifeCycleValues.RUNNING
    assert changer.state == "on_start"

    kin_sim.tick()
    msc.draw("muh.pdf")
    assert changer.life_cycle_state == LifeCycleValues.PAUSED
    assert changer.state == "on_pause"

    kin_sim.tick()
    msc.draw("muh.pdf")
    assert changer.life_cycle_state == LifeCycleValues.RUNNING
    assert changer.state == "on_unpause"

    kin_sim.tick()
    msc.draw("muh.pdf")
    assert changer.life_cycle_state == LifeCycleValues.DONE
    assert changer.state == "on_end"

    kin_sim.tick()
    msc.draw("muh.pdf")
    assert changer.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert changer.state == "on_reset"


def test_not_not_in_motion_statechart():
    node = ConstTrueNode()
    with pytest.raises(NotInMotionStatechartError):
        muh = node.observation_variable
    with pytest.raises(NotInMotionStatechartError):
        muh = node.life_cycle_variable
    with pytest.raises(NotInMotionStatechartError):
        node.start_condition = node.observation_variable
    with pytest.raises(NotInMotionStatechartError):
        node.pause_condition = node.observation_variable
    with pytest.raises(NotInMotionStatechartError):
        node.end_condition = node.observation_variable
    with pytest.raises(NotInMotionStatechartError):
        node.reset_condition = node.observation_variable


def test_counting():
    msc = MotionStatechart()
    seconds = 3
    counter = CountSeconds(seconds=seconds)
    msc.add_node(counter)

    node1 = Pulse()
    msc.add_node(node1)
    node1.start_condition = counter.observation_variable

    end = EndMotion()
    msc.add_node(end)

    counter.reset_condition = node1.observation_variable

    end.start_condition = trinary_logic_and(
        counter.observation_variable, trinary_logic_not(node1.observation_variable)
    )

    kin_sim = Executor(
        world=World(),
    )
    kin_sim.compile(motion_statechart=msc)

    current_time = time.time()

    kin_sim.tick_until_end(1_000_000)
    msc.draw("muh.pdf")

    actual = time.time() - current_time
    assert np.isclose(actual, seconds * 2, rtol=0.01)


def test_count_ticks():
    msc = MotionStatechart()
    msc.add_node(counter := CountTicks(ticks=3))
    msc.add_node(EndMotion.when_true(counter))
    kin_sim = Executor(world=World())
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()
    # ending tacks 2 ticks, one to turn EndMotion to Running and one more to turn it to true
    assert kin_sim.control_cycles == 3 + 2


def test_InvalidConditionError():
    msc = MotionStatechart()
    node = ConstTrueNode()
    msc.add_node(node)
    with pytest.raises(InvalidConditionError):
        node.end_condition = node


class TestEndMotion:
    def test_end_motion_when_all_done1(self):
        msc = MotionStatechart()
        msc.add_nodes(
            [
                ConstTrueNode(),
                ConstTrueNode(),
            ]
        )
        end = EndMotion.when_all_true(msc.nodes)
        msc.add_node(end)

        kin_sim = Executor(
            world=World(),
        )
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()
        msc.draw("muh.pdf")
        assert end.life_cycle_state == LifeCycleValues.RUNNING

    def test_end_motion_when_all_done2(self):
        msc = MotionStatechart()
        msc.add_nodes(
            [
                ConstTrueNode(),
                ConstFalseNode(),
            ]
        )
        end = EndMotion.when_all_true(msc.nodes)
        msc.add_node(end)

        kin_sim = Executor(
            world=World(),
        )
        kin_sim.compile(motion_statechart=msc)
        with pytest.raises(TimeoutError):
            kin_sim.tick_until_end()
        msc.draw("muh.pdf")
        assert end.life_cycle_state == LifeCycleValues.NOT_STARTED

    def test_end_motion_when_any_done1(self):
        msc = MotionStatechart()
        msc.add_nodes(
            [
                ConstTrueNode(),
                ConstFalseNode(),
            ]
        )
        end = EndMotion.when_any_true(msc.nodes)
        msc.add_node(end)

        kin_sim = Executor(
            world=World(),
        )
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()
        msc.draw("muh.pdf")
        assert end.life_cycle_state == LifeCycleValues.RUNNING

    def test_end_motion_when_any_done2(self):
        msc = MotionStatechart()
        msc.add_nodes(
            [
                ConstFalseNode(),
                ConstFalseNode(),
            ]
        )
        end = EndMotion.when_any_true(msc.nodes)
        msc.add_node(end)

        kin_sim = Executor(
            world=World(),
        )
        kin_sim.compile(motion_statechart=msc)
        with pytest.raises(TimeoutError):
            kin_sim.tick_until_end()
        msc.draw("muh.pdf")
        assert end.life_cycle_state == LifeCycleValues.NOT_STARTED


class TestParallel:
    def test_parallel(self):
        msc = MotionStatechart()
        msc.add_nodes(
            [
                parallel := Parallel([CountTicks(ticks=3), CountTicks(ticks=5)]),
            ]
        )
        msc.add_node(EndMotion.when_true(parallel))

        kin_sim = Executor(
            world=World(),
        )
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()
        # 5 (longest ticker) + 1 (for parallel to turn True) + 2 (for end to trigger)
        assert kin_sim.control_cycles == 8

    def test_parallel_with_tasks(self, pr2_world: World):
        map = pr2_world.root
        r_tip = pr2_world.get_kinematic_structure_entity_by_name("r_gripper_tool_frame")
        msc = MotionStatechart()
        msc.add_node(
            parallel := Parallel(
                [
                    AlignPlanes(
                        root_link=map,
                        tip_link=r_tip,
                        tip_normal=Vector3.X(reference_frame=r_tip),
                        goal_normal=Vector3.X(reference_frame=map),
                    ),
                    AlignPlanes(
                        root_link=map,
                        tip_link=r_tip,
                        tip_normal=Vector3.Y(reference_frame=r_tip),
                        goal_normal=Vector3.Z(reference_frame=map),
                    ),
                ]
            )
        )
        msc.add_node(EndMotion.when_true(parallel))

        kin_sim = Executor(
            world=pr2_world,
            controller_config=QPControllerConfig.create_default_with_50hz(),
        )
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()


class TestOpenClose:
    def test_open(self, pr2_world):
        factory = DoorFactory(
            name=PrefixedName("door"),
            handle_factory=HandleFactory(name=PrefixedName("handle")),
            semantic_position=SemanticPositionDescription(
                horizontal_direction_chain=[
                    HorizontalSemanticDirection.RIGHT,
                    HorizontalSemanticDirection.FULLY_CENTER,
                ],
                vertical_direction_chain=[VerticalSemanticDirection.FULLY_CENTER],
            ),
        )
        door_world = factory.create()
        with pr2_world.modify_world():
            lower_limits = DerivativeMap()
            lower_limits.position = -np.pi / 2
            lower_limits.velocity = -1
            upper_limits = DerivativeMap()
            upper_limits.position = np.pi / 2
            upper_limits.velocity = 1
            dof = DegreeOfFreedom(
                lower_limits=lower_limits,
                upper_limits=upper_limits,
                name=PrefixedName("hinge"),
            )
            pr2_world.add_degree_of_freedom(dof)
            root_T_door = RevoluteConnection(
                dof_name=dof.name,
                parent=pr2_world.root,
                child=door_world.root,
                axis=-cas.Vector3.Z(),
                parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(
                    x=1.5, z=1, yaw=np.pi, reference_frame=pr2_world.root
                ),
            )
            pr2_world.merge_world(door_world, root_connection=root_T_door)

        r_tip = pr2_world.get_body_by_name("r_gripper_tool_frame")
        handle = pr2_world.get_semantic_annotations_by_type(Handle)[0].body
        open_goal = 1
        close_goal = -1

        msc = MotionStatechart()
        msc.add_nodes(
            [
                Sequence(
                    [
                        CartesianPose(
                            root_link=pr2_world.root,
                            tip_link=r_tip,
                            goal_pose=TransformationMatrix.from_xyz_rpy(
                                yaw=np.pi, reference_frame=handle
                            ),
                        ),
                        Parallel(
                            [
                                Open(
                                    tip_link=r_tip,
                                    environment_link=handle,
                                    goal_joint_state=open_goal,
                                ),
                                opened := JointPositionReached(
                                    connection=root_T_door,
                                    position=open_goal,
                                    name="opened",
                                ),
                            ]
                        ),
                        Parallel(
                            [
                                Close(
                                    tip_link=r_tip,
                                    environment_link=handle,
                                    goal_joint_state=close_goal,
                                ),
                                closed := JointPositionReached(
                                    connection=root_T_door,
                                    position=close_goal,
                                    name="closed",
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        )
        msc.add_node(EndMotion.when_true(msc.nodes[0]))

        kin_sim = Executor(
            world=pr2_world,
            controller_config=QPControllerConfig.create_default_with_50hz(),
        )
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()
        msc.draw("muh.pdf")

        assert opened.observation_state == ObservationStateValues.TRUE
        assert closed.observation_state == ObservationStateValues.TRUE


class TestCollisionAvoidance:
    def test_collision_avoidance(self, box_bot_world: World):
        tip = box_bot_world.get_kinematic_structure_entity_by_name("bot")

        msc = MotionStatechart()
        msc.add_nodes(
            [
                CartesianPose(
                    root_link=box_bot_world.root,
                    tip_link=tip,
                    goal_pose=TransformationMatrix.from_xyz_rpy(
                        x=1, reference_frame=box_bot_world.root
                    ),
                ),
                CollisionAvoidance(
                    collision_entries=[CollisionRequest.avoid_all_collision()],
                ),
                local_min := LocalMinimumReached(),
            ]
        )
        msc.add_node(EndMotion.when_true(local_min))

        json_data = msc.to_json()
        json_str = json.dumps(json_data)
        new_json_data = json.loads(json_str)

        tracker = KinematicStructureEntityKwargsTracker.from_world(box_bot_world)
        kwargs = tracker.create_kwargs()
        msc_copy = MotionStatechart.from_json(new_json_data, **kwargs)

        kin_sim = Executor(
            world=box_bot_world,
            controller_config=QPControllerConfig.create_default_with_50hz(),
            collision_checker=CollisionCheckerLib.bpb,
        )
        kin_sim.compile(motion_statechart=msc_copy)

        msc_copy.draw("muh.pdf")
        kin_sim.tick_until_end(500)
        kin_sim.collision_scene.check_collisions()
        contact_distance = (
            kin_sim.collision_scene.closest_points.external_collisions[tip]
            .data[0]
            .contact_distance
        )
        assert contact_distance > 0.049

    def test_hard_constraints_violated(self, box_bot_world: World):
        root = box_bot_world.root
        with box_bot_world.modify_world():
            env2 = Body(
                name=PrefixedName("environment2"),
                collision=ShapeCollection(shapes=[Cylinder(width=0.5, height=0.1)]),
            )
            env_connection = FixedConnection(
                parent=root,
                child=env2,
                parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(0.75),
            )
            box_bot_world.add_connection(env_connection)

            env3 = Body(
                name=PrefixedName("environment3"),
                collision=ShapeCollection(shapes=[Cylinder(width=0.5, height=0.1)]),
            )
            env_connection = FixedConnection(
                parent=root,
                child=env3,
                parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(1.25),
            )
            box_bot_world.add_connection(env_connection)
            env4 = Body(
                name=PrefixedName("environment4"),
                collision=ShapeCollection(shapes=[Cylinder(width=0.5, height=0.1)]),
            )
            env_connection = FixedConnection(
                parent=root,
                child=env4,
                parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(
                    x=1, y=-0.25
                ),
            )
            box_bot_world.add_connection(env_connection)
            env5 = Body(
                name=PrefixedName("environment5"),
                collision=ShapeCollection(shapes=[Cylinder(width=0.5, height=0.1)]),
            )
            env_connection = FixedConnection(
                parent=root,
                child=env5,
                parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(
                    x=1, y=0.25
                ),
            )
            box_bot_world.add_connection(env_connection)

        tip = box_bot_world.get_kinematic_structure_entity_by_name("bot")

        msc = MotionStatechart()
        msc.add_node(
            Sequence(
                [
                    SetOdometry(
                        base_pose=TransformationMatrix.from_xyz_rpy(
                            x=1, reference_frame=box_bot_world.root
                        )
                    ),
                    Parallel(
                        [
                            CartesianPose(
                                root_link=box_bot_world.root,
                                tip_link=tip,
                                goal_pose=TransformationMatrix.from_xyz_rpy(
                                    x=1, reference_frame=box_bot_world.root
                                ),
                            ),
                            CollisionAvoidance(
                                collision_entries=[
                                    CollisionRequest.avoid_all_collision()
                                ],
                            ),
                        ]
                    ),
                ]
            )
        )
        msc.add_node(local_min := LocalMinimumReached())
        msc.add_node(EndMotion.when_true(local_min))

        json_data = msc.to_json()
        json_str = json.dumps(json_data)
        new_json_data = json.loads(json_str)

        tracker = KinematicStructureEntityKwargsTracker.from_world(box_bot_world)
        kwargs = tracker.create_kwargs()
        msc_copy = MotionStatechart.from_json(new_json_data, **kwargs)

        kin_sim = Executor(
            world=box_bot_world,
            controller_config=QPControllerConfig.create_default_with_50hz(),
            collision_checker=CollisionCheckerLib.bpb,
        )
        kin_sim.compile(motion_statechart=msc_copy)

        msc_copy.draw("muh.pdf")
        with pytest.raises(HardConstraintsViolatedException):
            kin_sim.tick_until_end()

def test_goal_cannot_have_endmotion_add_node():
    msc = MotionStatechart()
    goal = TestGoal()
    msc.add_node(goal)
    with pytest.raises(InvalidGoalException):
        goal.add_node(EndMotion())

def test_start_condition_cannot_reference_self():
    msc = MotionStatechart()
    goal = TestGoal()
    msc.add_node(goal)
    with pytest.raises(InvalidSelfReferenceInStartCondition):
        goal.start_condition = goal.observation_variable > 0

def test_start_condition_requires_observation_variables():
    msc = MotionStatechart()
    n1 = ConstTrueNode()
    n2 = ConstTrueNode()
    n3 = ConstTrueNode()
    msc.add_node(n1)
    msc.add_node(n2)
    msc.add_node(n3)

    # Valid: only observation variables
    n3.start_condition = cas.trinary_logic_and(
        n1.observation_variable, n2.observation_variable
    )

    # Invalid: includes a LifeCycleVariable
    with pytest.raises(InvalidVariableInCondition) as excinfo:
        n3.start_condition = cas.trinary_logic_and(
            n1.observation_variable, n1.life_cycle_variable
        )
    err = excinfo.value
    assert err.condition_type == "start"
    # Sanity: the offending variable name is the lifecycle variable's name
    assert str(err.variable_name) == str(n1.life_cycle_variable.name)


def test_pause_condition_requires_observation_variables():
    msc = MotionStatechart()
    n1 = ConstTrueNode()
    n2 = ConstTrueNode()
    msc.add_node(n1)
    msc.add_node(n2)

    # Valid: only observation variables
    n2.pause_condition = cas.trinary_logic_or(
        n1.observation_variable, cas.trinary_logic_not(n2.observation_variable)
    )

    # Invalid: includes a LifeCycleVariable
    with pytest.raises(InvalidVariableInCondition) as excinfo:
        n2.pause_condition = cas.trinary_logic_and(
            n1.observation_variable, n1.life_cycle_variable
        )
    err = excinfo.value
    assert err.condition_type == "pause"
    assert str(err.variable_name) == str(n1.life_cycle_variable.name)


def test_end_condition_requires_observation_variables():
    msc = MotionStatechart()
    n1 = ConstTrueNode()
    n2 = ConstTrueNode()
    msc.add_node(n1)
    msc.add_node(n2)

    # Valid: only observation variables
    n2.end_condition = cas.trinary_logic_and(
        n1.observation_variable, n2.observation_variable
    )

    # Invalid: includes a LifeCycleVariable
    with pytest.raises(InvalidVariableInCondition) as excinfo:
        n2.end_condition = cas.trinary_logic_and(
            n1.observation_variable, n1.life_cycle_variable
        )
    err = excinfo.value
    assert err.condition_type == "end"
    assert str(err.variable_name) == str(n1.life_cycle_variable.name)

def test_node_cannot_be_in_two_motion_statecharts():
    msc1 = MotionStatechart()
    msc2 = MotionStatechart()
    node = ConstTrueNode()

    msc1.add_node(node)
    with pytest.raises(NodeAlreadyInMotionStatechartError):
        msc2.add_node(node)


def test_node_cannot_be_in_two_goals():
    msc = MotionStatechart()
    g1, g2 = TestGoal(), TestGoal()
    n = ConstTrueNode()

    msc.add_node(g1)
    msc.add_node(g2)

    g1.add_node(n)
    with pytest.raises(NodeAlreadyHasParentGoalError):
        g2.add_node(n)


def test_duplicate_node_in_goal_is_rejected():
    msc = MotionStatechart()
    g = TestGoal()
    n = ConstTrueNode()

    msc.add_node(g)
    g.add_node(n)
    with pytest.raises(DuplicateNodeInGoalError):
        g.add_node(n)

def test_compile_without_qp_controller_when_no_constraints():
    msc = MotionStatechart()
    msc.add_node(ConstTrueNode())  # observation only, no constraints
    exe = Executor(world=World(), controller_config=None)

    #should compile fine and not create a qp_controller
    exe.compile(msc)
    assert exe.qp_controller is None

def test_compile_raises_when_constraints_present_but_no_qp_config(pr2_world):
    msc = MotionStatechart()
    joint_goal = JointPositionList(
        goal_state=JointState.from_str_dict(
            {
                "head_pan_joint": 0.041880780651479044,
                "head_tilt_joint": -0.37,
            },
            world=pr2_world,
        ),
    )
    msc.add_node(joint_goal)

    exe = Executor(world=pr2_world, controller_config=None)

    # should raise because constraints exist but no controller config was provided
    with pytest.raises(NoQPControllerConfigException):
        exe.compile(msc)
