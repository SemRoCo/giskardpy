from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Any

import numpy as np
import rustworkx as rx
from krrood.adapters.json_serializer import SubclassJSONSerializer
from typing_extensions import List, MutableMapping, ClassVar, Self, Type

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.motion_statechart.context import BuildContext, ExecutionContext
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from giskardpy.motion_statechart.exceptions import NodeNotFoundError
from giskardpy.motion_statechart.exceptions import (
    NodeAlreadyInMotionStatechartError,
)
from giskardpy.motion_statechart.graph_node import (
    MotionStatechartNode,
    TrinaryCondition,
    Goal,
    EndMotion,
    CancelMotion,
    GenericMotionStatechartNode,
    ObservationVariable,
    LifeCycleVariable,
)
from giskardpy.motion_statechart.graph_node import Task
from giskardpy.motion_statechart.plotters.graphviz import MotionStatechartGraphviz
from giskardpy.qp.constraint_collection import ConstraintCollection
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName


@dataclass(repr=False, eq=False)
class State(MutableMapping[MotionStatechartNode, float], SubclassJSONSerializer):
    motion_statechart: MotionStatechart
    default_value: ClassVar[float] = field(init=False)
    data: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    def grow(self) -> None:
        self.data = np.append(self.data, self.default_value)

    def life_cycle_symbols(self) -> List[LifeCycleVariable]:
        return [node.life_cycle_variable for node in self.motion_statechart.nodes]

    def observation_symbols(self) -> List[ObservationVariable]:
        return [node.observation_variable for node in self.motion_statechart.nodes]

    def __getitem__(self, node: MotionStatechartNode) -> float:
        return float(self.data[node.index])

    def __setitem__(self, node: MotionStatechartNode, value: float) -> None:
        self.data[node.index] = value

    def __delitem__(self, node: MotionStatechartNode) -> None:
        self.data = np.delete(self.data, node.index)

    def __iter__(self) -> iter:
        return iter(self.data)

    def __len__(self) -> int:
        return self.data.shape[0]

    def keys(self) -> List[MotionStatechartNode]:
        return self.motion_statechart.nodes

    def items(self) -> List[tuple[MotionStatechartNode, float]]:
        return [(node, self[node]) for node in self.motion_statechart.nodes]

    def values(self) -> List[float]:
        return [self[node] for node in self.keys()]

    def __contains__(self, node: MotionStatechartNode) -> bool:
        return node in self.motion_statechart.nodes

    def __deepcopy__(self, memo) -> Self:
        """
        Create a deep copy of the WorldState.
        """
        return self.__class__(
            motion_statechart=self.motion_statechart,
            data=self.data.copy(),
        )

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "data": self.data.tolist()}

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        motion_statechart = kwargs["motion_statechart"]
        return cls(
            motion_statechart=motion_statechart,
            data=np.array(data["data"], dtype=np.float64),
        )

    def __str__(self) -> str:
        return str({str(symbol.name): value for symbol, value in self.items()})

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: Self) -> bool:
        return str(self) == str(other)


@dataclass(repr=False, eq=False)
class LifeCycleState(State):

    default_value: ClassVar[float] = LifeCycleValues.NOT_STARTED
    _compiled_updater: cas.CompiledFunction = field(init=False)

    def compile(self):
        state_updater = []
        for node in self.motion_statechart.nodes:
            state_symbol = node.life_cycle_variable

            not_started_transitions = cas.if_else(
                condition=node.start_condition == cas.TrinaryTrue,
                if_result=cas.Expression(LifeCycleValues.RUNNING),
                else_result=cas.Expression(LifeCycleValues.NOT_STARTED),
            )
            running_transitions = cas.if_cases(
                cases=[
                    (
                        node.reset_condition == cas.TrinaryTrue,
                        cas.Expression(LifeCycleValues.NOT_STARTED),
                    ),
                    (
                        node.end_condition == cas.TrinaryTrue,
                        cas.Expression(LifeCycleValues.DONE),
                    ),
                    (
                        node.pause_condition == cas.TrinaryTrue,
                        cas.Expression(LifeCycleValues.PAUSED),
                    ),
                ],
                else_result=cas.Expression(LifeCycleValues.RUNNING),
            )
            pause_transitions = cas.if_cases(
                cases=[
                    (
                        node.reset_condition == cas.TrinaryTrue,
                        cas.Expression(LifeCycleValues.NOT_STARTED),
                    ),
                    (
                        node.end_condition == cas.TrinaryTrue,
                        cas.Expression(LifeCycleValues.DONE),
                    ),
                    (
                        node.pause_condition == cas.TrinaryFalse,
                        cas.Expression(LifeCycleValues.RUNNING),
                    ),
                ],
                else_result=cas.Expression(LifeCycleValues.PAUSED),
            )
            ended_transitions = cas.if_else(
                condition=node.reset_condition == cas.TrinaryTrue,
                if_result=cas.Expression(LifeCycleValues.NOT_STARTED),
                else_result=cas.Expression(LifeCycleValues.DONE),
            )

            state_machine = cas.if_eq_cases(
                a=state_symbol,
                b_result_cases=[
                    (LifeCycleValues.NOT_STARTED, not_started_transitions),
                    (LifeCycleValues.RUNNING, running_transitions),
                    (LifeCycleValues.PAUSED, pause_transitions),
                    (LifeCycleValues.DONE, ended_transitions),
                ],
                else_result=cas.Expression(state_symbol),
            )
            state_updater.append(state_machine)
        state_updater = cas.Expression(state_updater)
        self._compiled_updater = state_updater.compile(
            parameters=[self.observation_symbols(), self.life_cycle_symbols()],
            sparse=False,
        )

    def __getitem__(self, node: MotionStatechartNode) -> LifeCycleValues:
        return LifeCycleValues(super().__getitem__(node))

    def update_state(self, observation_state: np.ndarray):
        self.data = self._compiled_updater(observation_state, self.data)

    def __str__(self) -> str:
        return str(
            {
                str(symbol.name): LifeCycleValues(value).name
                for symbol, value in self.items()
            }
        )


@dataclass(repr=False, eq=False)
class ObservationState(State):
    default_value: ClassVar[ObservationStateValues] = ObservationStateValues.UNKNOWN

    _compiled_updater: cas.CompiledFunction = field(init=False)

    def compile(self, context: BuildContext):
        observation_state_updater = []
        for node in self.motion_statechart.nodes:
            state_f = cas.if_eq_cases(
                a=node.life_cycle_variable,
                b_result_cases=[
                    (
                        int(LifeCycleValues.RUNNING),
                        node._observation_expression,
                    ),
                    (
                        int(LifeCycleValues.NOT_STARTED),
                        cas.TrinaryUnknown,
                    ),
                ],
                else_result=cas.Expression(node.observation_variable),
            )
            observation_state_updater.append(state_f)
        self._compiled_updater = cas.Expression(observation_state_updater).compile(
            parameters=[
                self.observation_symbols(),
                self.life_cycle_symbols(),
                context.world.state.get_variables(),
                context.collision_scene.get_external_collision_symbol(),
                context.collision_scene.get_self_collision_symbol(),
                context.auxiliary_variable_manager.variables,
            ],
            sparse=False,
        )

    def update_state(
        self,
        life_cycle_state: np.ndarray,
        world_state: np.ndarray,
        external_collision_data: np.ndarray,
        self_collision_data: np.ndarray,
        auxiliar_variables: np.ndarray,
    ):
        self.data = self._compiled_updater(
            self.data,
            life_cycle_state,
            world_state,
            external_collision_data,
            self_collision_data,
            auxiliar_variables,
        )


@dataclass(repr=False, eq=False)
class StateHistoryItem:
    control_cycle: int
    life_cycle_state: LifeCycleState
    observation_state: ObservationState

    def __post_init__(self):
        self.life_cycle_state = deepcopy(self.life_cycle_state)
        self.observation_state = deepcopy(self.observation_state)

    def __eq__(self, other: StateHistoryItem) -> bool:
        has_life_cycle_changed = np.any(
            other.life_cycle_state.data != self.life_cycle_state.data
        )
        has_observation_changed = np.any(
            other.observation_state.data != self.observation_state.data
        )
        return not has_life_cycle_changed and not has_observation_changed

    def __repr__(self) -> str:
        merged = {
            node.name: f"{self.observation_state[node]} | {life_cycle.name}"
            for node, life_cycle in self.life_cycle_state.items()
        }
        return str(merged)


@dataclass
class StateHistory:
    history: List[StateHistoryItem] = field(default_factory=list)

    def append(self, next_item: StateHistoryItem):
        if len(self.history) != 0:
            if next_item == self.history[-1]:
                return
        self.history.append(next_item)

    def get_life_cycle_history_of_node(
        self, node: MotionStatechartNode
    ) -> list[LifeCycleValues]:
        return [history_item.life_cycle_state[node] for history_item in self.history]

    def get_observation_history_of_node(
        self, node: MotionStatechartNode
    ) -> list[LifeCycleValues]:
        return [history_item.observation_state[node] for history_item in self.history]

    def __len__(self) -> int:
        return len(self.history)


@dataclass
class MotionStatechart(SubclassJSONSerializer):
    """
    Represents a motion statechart.
    A motion statechart is a directed graph of nodes and edges.
    Nodes have two states: observation state and life cycle state.
    Life cycle states indicate the current state in the life cycle of the node:
        - NOT_STARTED: the node has not started yet.
        - RUNNING: the node is running.
        - PAUSED: the node is paused.
        - DONE: the node has ended.
    Out of these 4 states, nodes are only "active" if they are in the RUNNING state.
    Observation states indicate the current observation of the node:
        - TrinaryFalse: the thing the node is observing is not True.
        - TrinaryUnknown: the node has not yet made an observation or it cannot determine its truth value yet.
        - TrinaryTrue: the thing the node is observing is True.
    Nodes are connected with edges, or transitions.
    There are 4 types of transitions:
        - start condition: If True, the node transitions from NOT_STARTED to RUNNING.
        - pause condition: If True, the node transitions from RUNNING to PAUSED.
                           If False, the node transitions from PAUSED to RUNNING.
        - end condition: If True, the node transitions from RUNNING or PAUSED to DONE.
        - reset condition: If True, the node transitions from any state to NOT_STARTED.
    If multiple conditions are met, the following order is used:
        1. reset condition
        2. end condition
        3. pause condition
        4. start condition
    How to use this class:
        1. initialized with a world
        2. add nodes.
        3. set the transition conditions of nodes
        4. compile the motion statechart.
        5. call tick() to update the observation state and life cycle state.
            tick() will raise an exception if the cancel motion condition is met.
        6. call is_end_motion() to check if the motion is done.
    """

    rx_graph: rx.PyDiGraph[MotionStatechartNode] = field(
        default_factory=lambda: rx.PyDAG(multigraph=True), init=False, repr=False
    )
    """
    The underlying graph of the motion statechart.
    """

    observation_state: ObservationState = field(init=False)
    """
    Combined representation of the observation state of the motion statechart, to enable an efficient tick().
    """

    life_cycle_state: LifeCycleState = field(init=False)
    """
    Combined representation of the life cycle state of the motion statechart, to enable an efficient tick().
    """

    control_cycle_counter: int = field(default=0, init=False)
    """
    Increases by 1 each time tick() is called.
    """
    history: StateHistory = field(default_factory=StateHistory, init=False)
    """
    The history of how the state of the motion statechart changed over time.
    """

    def __post_init__(self):
        self.life_cycle_state = LifeCycleState(self)
        self.observation_state = ObservationState(self)

    def create_structure_copy(self) -> MotionStatechart:
        """
        Creates a copy of the motion statechart, where all nodes are MotionStatechartNodes or Goals.
        This is useful if only the structure of the motion statechart is needed, for example, for visualization.
        """
        motion_statechart_copy = MotionStatechart()
        self._create_structure_copy(self.top_level_nodes, motion_statechart_copy)
        return motion_statechart_copy

    def _create_structure_copy(
        self, nodes: List[MotionStatechartNode], destination: MotionStatechart | Goal
    ):
        """
        Creates a structure copy of a node and adds it to the destination.
        """
        for node in nodes:
            match node:
                case Goal():
                    node_copy = Goal(name=node.name)
                case Task():
                    node_copy = Task(name=node.name)
                case _:
                    node_copy = MotionStatechartNode(name=node.name)
            destination.add_node(node_copy)
            node_copy.index = node.index
            node_copy.start_condition = node.start_condition
            node_copy.pause_condition = node.pause_condition
            node_copy.end_condition = node.end_condition
            node_copy.reset_condition = node.reset_condition
            if isinstance(node, Goal):
                self._create_structure_copy(node.nodes, node_copy)

    @property
    def nodes(self) -> List[MotionStatechartNode]:
        return list(self.rx_graph.nodes())

    @property
    def top_level_nodes(self) -> List[MotionStatechartNode]:
        """
        :return: All nodes that don't belong to a Goal.
        """
        return [node for node in self.nodes if node.parent_node is None]

    @property
    def edges(self) -> List[TrinaryCondition]:
        """
        The edges of the underlying graph.
        .. warning:: This may return duplicate edges if a transition uses multiple nodes.
        """
        return self.rx_graph.edges()

    @property
    def unique_edges(self) -> List[TrinaryCondition]:
        """
        :return: The edges of the motion statechart, without duplicates.
        """
        return list(set(self.edges))

    def add_node(self, node: MotionStatechartNode):
        """
        Adds a node to the motion statechart and finalizes the initialization of the node.
        """
        # Prevent adding a node that already belongs to a different motion statechart
        if node._motion_statechart is not None and node._motion_statechart is not self:
            raise NodeAlreadyInMotionStatechartError(
                node_name=node.unique_name if node.index is not None else node.name,
                current_msc=str(node._motion_statechart),
                target_msc=str(self),
            )
        if node._motion_statechart is self:
            return

        node.motion_statechart = self
        node.index = self.rx_graph.add_node(node)
        node._post_add_to_motion_statechart()
        self.life_cycle_state.grow()
        self.observation_state.grow()

    def get_node_by_index(self, index: int) -> MotionStatechartNode:
        return self.rx_graph.get_node_data(index)

    def _add_transitions(self):
        for node in self.nodes:
            self._create_edge_for_condition(node, node._start_condition)
            self._create_edge_for_condition(node, node._pause_condition)
            self._create_edge_for_condition(node, node._end_condition)
            self._create_edge_for_condition(node, node._reset_condition)

    def _create_edge_for_condition(
        self, owner: MotionStatechartNode, condition: TrinaryCondition
    ):
        for parent_node in condition.node_dependencies:
            self.rx_graph.add_edge(owner.index, parent_node.index, condition)

    def _build_nodes(self, context: BuildContext):
        for node in self.nodes:
            self._build_and_apply_artifacts(node, context=context)

    def _build_and_apply_artifacts(
        self, node: MotionStatechartNode, context: BuildContext
    ):
        if isinstance(node, Goal):
            node.build(context=context)
            for child_node in node.nodes:
                self._build_and_apply_artifacts(child_node, context=context)
        artifacts = node.build(context=context)
        node._constraint_collection = artifacts.constraints
        node._constraint_collection.link_to_motion_statechart_node(node)
        # if no observation is set, use the symbol for its observation variable to copy the state from last tick,
        # in case `on_tick` doesn't overwrite it.
        node._observation_expression = (
            artifacts.observation or node.observation_variable
        )
        node._debug_expressions = artifacts.debug_expressions

    def _apply_goal_conditions_to_their_children(self):
        for goal in self.get_nodes_by_type(Goal):
            goal._apply_goal_conditions_to_children()

    def compile(self, context: BuildContext):
        """
        Compiles all components of the motion statechart given the provided context.
        This method must be called before tick().

        :param context: The build context required to execute the compilation process.
        """
        self._expand_goals(context=context)
        self._apply_goal_conditions_to_their_children()
        self._build_nodes(context=context)
        self._add_transitions()
        self.observation_state.compile(context=context)
        self.life_cycle_state.compile()
        self.history.append(
            next_item=StateHistoryItem(
                control_cycle=self.control_cycle_counter,
                life_cycle_state=self.life_cycle_state,
                observation_state=self.observation_state,
            )
        )

    def _expand_goals(self, context: BuildContext):
        """
        Triggers the expansion of all goals in the motion statechart and add its children to the motion statechart.
        """
        for goal in self.get_nodes_by_type(Goal):
            self._expand_goal(goal, context=context)

    def _expand_goal(self, goal: Goal, context: BuildContext):
        goal.expand(context)
        for child_node in goal.nodes:
            if isinstance(child_node, Goal):
                self._expand_goal(child_node, context=context)

    def combine_constraint_collections_of_nodes(self) -> ConstraintCollection:
        combined_constraint_collection = ConstraintCollection()
        for node in self.nodes:
            combined_constraint_collection.merge(
                name_prefix=node.unique_name, other=node._constraint_collection
            )
        return combined_constraint_collection

    def _update_observation_state(self, context: BuildContext):
        self.observation_state.update_state(
            life_cycle_state=self.life_cycle_state.data,
            world_state=context.world.state.data,
            external_collision_data=context.collision_scene.get_external_collision_data(),
            self_collision_data=context.collision_scene.get_self_collision_data(),
            auxiliar_variables=context.auxiliary_variable_manager.resolve_auxiliary_variables(),
        )
        for node in self.nodes:
            if self.life_cycle_state[node] == LifeCycleValues.RUNNING:
                observation_overwrite = node.on_tick(
                    context=context.to_execution_context()
                )
                if observation_overwrite is not None:
                    self.observation_state[node] = observation_overwrite

    def _update_life_cycle_state(self, context: ExecutionContext):
        previous = self.life_cycle_state.data.copy()
        self.life_cycle_state.update_state(self.observation_state.data)
        self._trigger_life_cycle_callbacks(
            previous, self.life_cycle_state.data, context
        )

    def _trigger_life_cycle_callbacks(
        self,
        previous_state: np.ndarray,
        current_state: np.ndarray,
        context: ExecutionContext,
    ) -> None:
        for node in self.nodes:
            prev = LifeCycleValues(int(previous_state[node.index]))
            curr = LifeCycleValues(int(current_state[node.index]))

            if prev == curr:
                continue

            match (prev, curr):
                case (_, LifeCycleValues.NOT_STARTED):
                    node.on_reset(context=context)
                case (LifeCycleValues.NOT_STARTED, LifeCycleValues.RUNNING):
                    node.on_start(context=context)
                case (LifeCycleValues.RUNNING, LifeCycleValues.PAUSED):
                    node.on_pause(context=context)
                case (LifeCycleValues.PAUSED, LifeCycleValues.RUNNING):
                    node.on_unpause(context=context)
                case (
                    (LifeCycleValues.RUNNING | LifeCycleValues.PAUSED),
                    LifeCycleValues.DONE,
                ):
                    node.on_end(context=context)
                case _:
                    pass

    def tick(self, context: BuildContext):
        """
        Executes a single tick of the motion statechart.
        First the observation state is updated, then the life cycle state is updated.
        :param context: The context required to execute the tick.
        """
        self.control_cycle_counter += 1
        self._update_observation_state(context)
        self._update_life_cycle_state(context.to_execution_context())
        self._raise_if_cancel_motion()
        self.history.append(
            next_item=StateHistoryItem(
                control_cycle=self.control_cycle_counter,
                life_cycle_state=self.life_cycle_state,
                observation_state=self.observation_state,
            )
        )

    def get_nodes_by_type(
        self, node_type: Type[GenericMotionStatechartNode]
    ) -> List[GenericMotionStatechartNode]:
        return [node for node in self.nodes if isinstance(node, node_type)]

    def is_end_motion(self) -> bool:
        """
        :return: True if the motion is done, meaning at least one EndMotion is in observation state True, False otherwise.
        """
        return any(
            self.observation_state[node] == ObservationStateValues.TRUE
            for node in self.get_nodes_by_type(EndMotion)
        )

    def _raise_if_cancel_motion(self):
        for node in self.get_nodes_by_type(CancelMotion):
            if self.observation_state[node] == ObservationStateValues.TRUE:
                raise node.exception

    def draw(self, file_name: str):
        """
        Uses graphviz to draw the motion statechart and safe it at `file_name`.
        """
        MotionStatechartGraphviz(self).to_dot_graph_pdf(file_name=file_name)

    def to_json(self) -> Dict[str, Any]:
        self._add_transitions()
        result = super().to_json()
        result["nodes"] = [
            node.to_json() for node in sorted(self.nodes, key=lambda n: n.index)
        ]
        result["unique_edges"] = [edge.to_json() for edge in self.unique_edges]
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        motion_statechart = cls()
        for json_data in data["nodes"]:
            node = MotionStatechartNode.from_json(json_data, **kwargs)
            motion_statechart.add_node(node)
        for json_data in data["unique_edges"]:
            transition = TrinaryCondition.from_json(
                json_data, motion_statechart=motion_statechart, **kwargs
            )
            transition.owner._set_transition(transition)
        for node in motion_statechart.nodes:
            if node.parent_node is not None:
                node.parent_node.nodes.append(node)
        return motion_statechart
