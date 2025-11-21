import json
import time
from dataclasses import dataclass

import numpy as np
import pytest

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.data_types.exceptions import InvalidGoalException
from giskardpy.executor import Executor
from giskardpy.model.collision_matrix_manager import CollisionRequest
from giskardpy.model.collision_world_syncer import CollisionCheckerLib
from giskardpy.motion_statechart.binding_policy import GoalBindingPolicy
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from giskardpy.motion_statechart.exceptions import NotInMotionStatechartError, InvalidSelfReferenceInStartCondition
from giskardpy.motion_statechart.goals.collision_avoidance import (
    CollisionAvoidance,
)
from giskardpy.motion_statechart.graph_node import (
    EndMotion,
    CancelMotion,
)
from giskardpy.motion_statechart.graph_node import ThreadPayloadMonitor
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import (
    SetSeedConfiguration,
    SetOdometry,
)
from giskardpy.motion_statechart.monitors.payload_monitors import (
    Print,
    Pulse,
    CountSeconds,
)
from giskardpy.motion_statechart.motion_statechart import (
    MotionStatechart,
)
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
)
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    KinematicStructureEntityKwargsTracker,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import TransformationMatrix
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.spatial_types.spatial_types import (
    trinary_logic_and,
    trinary_logic_not,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
    ActiveConnection1DOF,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
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
def test_arrange_in_sequence():
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


def test_collision_avoidance(box_bot_world):
    msc = MotionStatechart()

    root = box_bot_world.root
    tip = box_bot_world.get_kinematic_structure_entity_by_name("bot")

    target_pose = TransformationMatrix.from_xyz_quaternion(
        1, reference_frame=box_bot_world.root
    )
    cart_goal = CartesianPose(
        root_link=root,
        tip_link=tip,
        goal_pose=target_pose,
    )
    msc.add_node(cart_goal)

    collision_avoidance = CollisionAvoidance(
        collision_entries=[CollisionRequest.avoid_all_collision()],
    )
    msc.add_node(collision_avoidance)

    local_min = LocalMinimumReached()
    msc.add_node(local_min)

    end = EndMotion()
    msc.add_node(end)
    end.start_condition = local_min.observation_variable

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
    kin_sim.compile(motion_statechart=msc)

    msc_copy.draw("muh.pdf")
    kin_sim.tick_until_end(500)
    kin_sim.collision_scene.check_collisions()
    contact_distance = (
        kin_sim.collision_scene.closest_points.external_collisions[tip]
        .data[0]
        .contact_distance
    )
    assert contact_distance > 0.049


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