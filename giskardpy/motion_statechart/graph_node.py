from __future__ import annotations

import ast
import re
import threading
from abc import ABC
from dataclasses import field, dataclass, fields
from enum import Enum

from krrood.adapters.json_serializer import SubclassJSONSerializer
from typing_extensions import (
    Dict,
    Any,
    Self,
    Optional,
    TYPE_CHECKING,
    List,
    TypeVar,
)

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.data_types.exceptions import InvalidGoalException
from giskardpy.motion_statechart.context import BuildContext, ExecutionContext
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
    TransitionKind,
)
from giskardpy.motion_statechart.exceptions import (
    NotInMotionStatechartError,
    InvalidSelfReferenceInStartCondition,
    InvalidVariableInCondition,
    NodeAlreadyHasParentGoalError,
    DuplicateNodeInGoalError,
    NodeAlreadyInMotionStatechartError,
)
from giskardpy.motion_statechart.plotters.plot_specs import NodePlotSpec
from giskardpy.qp.constraint_collection import ConstraintCollection
from giskardpy.utils.utils import string_shortener
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.geometry import Color

if TYPE_CHECKING:
    from giskardpy.motion_statechart.motion_statechart import (
        MotionStatechart,
    )


@dataclass(eq=False, repr=False)
class TrinaryCondition(SubclassJSONSerializer):
    """
    Represents a trinary condition used to define transitions in a motion statechart model.

    This class serves as a representation of a logical trinary condition with three possible states: true, false, and
    unknown. It is used as part of a motion statechart system to define transitions between nodes. The condition is
    evaluated using a logical expression and connects nodes via parent-child relationships. It includes methods to
    create predefined trinary values, update the expression of the condition, and format the condition for display.
    """

    kind: TransitionKind
    """
    The type of transition associated with this condition.
    """
    expression: cas.Expression = cas.TrinaryUnknown
    """
    The logical trinary condition to be evaluated.
    """

    owner: Optional[MotionStatechartNode] = field(default=None)
    """
    The node this transition belongs to.
    """

    def __hash__(self) -> int:
        return hash((str(self), self.kind))

    def __eq__(self, other):
        return hash(self) == hash(other)

    @classmethod
    def create_true(
        cls, kind: TransitionKind, owner: Optional[MotionStatechartNode] = None
    ) -> Self:
        return cls(expression=cas.TrinaryTrue, kind=kind, owner=owner)

    @classmethod
    def create_false(
        cls, kind: TransitionKind, owner: Optional[MotionStatechartNode] = None
    ) -> Self:
        return cls(expression=cas.TrinaryFalse, kind=kind, owner=owner)

    @classmethod
    def create_unknown(
        cls, kind: TransitionKind, owner: Optional[MotionStatechartNode] = None
    ) -> Self:
        return cls(
            expression=cas.TrinaryUnknown,
            kind=kind,
            owner=owner,
        )

    def update_expression(
        self, new_expression: cas.Expression, child: MotionStatechartNode
    ) -> None:
        self.expression = new_expression
        self._child = child

    @property
    def node_dependencies(self) -> List[MotionStatechartNode]:
        """
        List of parent nodes involved in the condition, derived from the free symbols in the expression.
        """
        return [
            x.motion_statechart_node
            for x in self.expression.free_variables()
            if isinstance(x, ObservationVariable)
        ]

    def __str__(self):
        """
        Replaces the state symbols with motion statechart node names and formats it nicely.
        """
        free_symbols = self.expression.free_variables()
        if not free_symbols:
            str_representation = str(cas.is_const_binary_true(self.expression))
        else:
            str_representation = cas.trinary_logic_to_str(self.expression)
        str_representation = re.sub(
            r'"([^"]*?)/observation"', r'"\1"', str_representation
        )
        return str_representation

    def __repr__(self):
        return str(self)

    def to_json(self) -> Dict[str, Any]:
        json_data = super().to_json()
        json_data["kind"] = self.kind.name
        json_data["expression"] = str(self)
        json_data["owner"] = self.owner.index if self.owner else None
        return json_data

    @classmethod
    def create_from_trinary_logic_str(
        cls,
        kind: TransitionKind,
        trinary_logic_str: str,
        observation_variables: List[ObservationVariable],
        owner: Optional[MotionStatechartNode] = None,
    ):
        tree = ast.parse(trinary_logic_str, mode="eval")
        return cls(
            kind=kind,
            expression=cls._parse_ast_expression(tree.body, observation_variables),
            owner=owner,
        )

    @staticmethod
    def _parse_ast_expression(
        node: ast.expr, observation_variables: List[ObservationVariable]
    ) -> cas.Expression:
        match node:
            case ast.BoolOp(op=ast.And()):
                return TrinaryCondition._parse_ast_and(node, observation_variables)
            case ast.BoolOp(op=ast.Or()):
                return TrinaryCondition._parse_ast_or(node, observation_variables)
            case ast.UnaryOp():
                return TrinaryCondition._parse_ast_not(node, observation_variables)
            case ast.Constant(value=str(val)):
                variable_name = PrefixedName("observation", val)
                for v in observation_variables:
                    if variable_name == v.name:
                        return v
                raise KeyError(f"unknown observation variable: {val!r}")
            case ast.Constant(value=True):
                return cas.TrinaryTrue
            case ast.Constant(value=False):
                return cas.TrinaryFalse
            case _:
                raise TypeError(f"failed to parse {type(node).__name__}")

    @staticmethod
    def _parse_ast_and(node, observation_variables: List[ObservationVariable]):
        return cas.trinary_logic_and(
            *[
                TrinaryCondition._parse_ast_expression(x, observation_variables)
                for x in node.values
            ]
        )

    @staticmethod
    def _parse_ast_or(node, observation_variables: List[ObservationVariable]):
        return cas.trinary_logic_or(
            *[
                TrinaryCondition._parse_ast_expression(x, observation_variables)
                for x in node.values
            ]
        )

    @staticmethod
    def _parse_ast_not(node, observation_variables: List[ObservationVariable]):
        if isinstance(node.op, ast.Not):
            return cas.trinary_logic_not(
                TrinaryCondition._parse_ast_expression(
                    node.operand, observation_variables
                )
            )

    @classmethod
    def _from_json(
        cls, data: Dict[str, Any], motion_statechart: MotionStatechart, **kwargs
    ) -> Self:
        return cls.create_from_trinary_logic_str(
            kind=TransitionKind[data["kind"]],
            trinary_logic_str=data["expression"],
            observation_variables=motion_statechart.observation_state.observation_symbols(),
            owner=motion_statechart.get_node_by_index(data["owner"]),
        )


@dataclass(repr=False, eq=False)
class ObservationVariable(cas.FloatVariable):
    """
    A symbol representing the observation state of a node.
    """

    name: PrefixedName = field(kw_only=True)
    motion_statechart_node: MotionStatechartNode
    """
    The node this variable is the observation state of.
    """

    def resolve(self) -> float:
        return self.motion_statechart_node.observation_state


@dataclass(repr=False, eq=False)
class LifeCycleVariable(cas.FloatVariable):
    """
    A symbol representing the life cycle state of a node.
    """

    name: PrefixedName = field(kw_only=True)
    motion_statechart_node: MotionStatechartNode
    """
    The node this variable is the life cycle state of.
    """

    def resolve(self) -> LifeCycleValues:
        return self.motion_statechart_node.life_cycle_state


@dataclass
class DebugExpression:
    """
    Symbolic expressions used for debugging only.
    Allows you to keep track of any expression and evaluate them later in debug mode.
    """

    name: str
    """
    Name used for this expression in some debugging tools.
    """

    expression: (
        cas.Expression
        | cas.Point3
        | cas.Vector3
        | cas.Quaternion
        | cas.RotationMatrix
        | cas.TransformationMatrix
    )

    color: Color = field(default_factory=lambda: Color(1, 0, 0, 1))
    """
    The color used when this expression is rendered in visualization tools.
    """


@dataclass
class NodeArtifacts:
    """
    Represents the artifacts produced by the `build` method of a node.
    It makes explicit what artifacts are produced by a node.
    """

    constraints: ConstraintCollection = field(default_factory=ConstraintCollection)
    """
    A collection of constraints that describe a motion task. 
    """
    observation: Optional[cas.Expression] = field(default=None)
    """
    A symbolic expression that describes the observation state of this node.
    Instead of setting this attribute directly, you may also implement the `on_tick` method of a node.
    The advantage of using observation is that you can reuse the expressions used in constraints.
    .. warning:: the result of `on_tick` takes precedence over the observation expression.
    """
    debug_expressions: List[DebugExpression] = field(default_factory=list)
    """
    A list of symbolic expressions used for debugging only.
    While in debug mode, you can call .evaluate() on them to get their current value.
    """


@dataclass(repr=False, eq=False)
class MotionStatechartNode(SubclassJSONSerializer):
    name: str = field(default=None, kw_only=True)
    """
    A name for the node within a motion statechart.
    The name is not unique, use `.unique_name`, if you need a unique identifier.
    """

    _motion_statechart: MotionStatechart = field(init=False, default=None)
    """
    Back reference to the motion statechart that owns this node.
    """
    index: Optional[int] = field(default=None, init=False)
    """
    The index of this node in the motion statechart.
    """

    parent_node_index: Optional[int] = field(default=None, init=False)
    """
    The index of the parent node in the motion statechart, if None, it is on the top layer of a motion statechart.
    """

    _life_cycle_variable: LifeCycleVariable = field(init=False, default=None)
    """
    A variable referring to the life cycle state of this node.
    """
    _observation_variable: ObservationVariable = field(init=False, default=None)
    """
    A variable referring to the observation state of this node.
    """

    _constraint_collection: ConstraintCollection = field(init=False)
    """The parameter is set after build() using its NodeArtifacts."""
    _observation_expression: cas.Expression = field(init=False)
    """The parameter is set after build() using its NodeArtifacts."""
    _debug_expressions: List[DebugExpression] = field(default_factory=list, init=False)
    """The parameter is set after build() using its NodeArtifacts."""

    _start_condition: TrinaryCondition = field(init=False, default=None)
    """
    Decides when this node transitions from life cycle state NOT_STARTED to RUNNING.
    """
    _pause_condition: TrinaryCondition = field(init=False, default=None)
    """
    Decides when this node transitions from RUNNING to PAUSED or back.
    """
    _end_condition: TrinaryCondition = field(init=False, default=None)
    """
    Decides when this node transitions from RUNNING or PAUSED to DONE.
    """
    _reset_condition: TrinaryCondition = field(init=False, default=None)
    """
    Decides when this transitions to NOT_STARTED.
    """

    plot_specs: NodePlotSpec = field(
        default_factory=NodePlotSpec.create_monitor_style, kw_only=True, init=False
    )

    def __post_init__(self):
        if self.name is None:
            self.name = self.__class__.__name__

    def _post_add_to_motion_statechart(self):
        """
        Called after this node is added to a motion statechart.
        Finalizes the initialization parts that require the motion statechart to be set.
        """
        self._observation_variable = ObservationVariable(
            name=PrefixedName("observation", self.unique_name),
            motion_statechart_node=self,
        )
        self._life_cycle_variable = LifeCycleVariable(
            name=PrefixedName("life_cycle", self.unique_name),
            motion_statechart_node=self,
        )
        self._start_condition = TrinaryCondition.create_true(
            kind=TransitionKind.START, owner=self
        )
        self._pause_condition = TrinaryCondition.create_false(
            kind=TransitionKind.PAUSE, owner=self
        )
        self._end_condition = TrinaryCondition.create_false(
            kind=TransitionKind.END, owner=self
        )
        self._reset_condition = TrinaryCondition.create_false(
            kind=TransitionKind.RESET, owner=self
        )

    @property
    def parent_node(self) -> Optional[MotionStatechartNode]:
        """
        :return: Reference to the parent node of this node.
        """
        if self.parent_node_index is None:
            return None
        return self._motion_statechart.get_node_by_index(self.parent_node_index)

    @parent_node.setter
    def parent_node(self, parent_node: Optional[MotionStatechartNode]) -> None:
        if parent_node is None:
            self.parent_node_index = None
        else:
            self.parent_node_index = parent_node.index

    def _set_transition(self, transition: TrinaryCondition) -> None:
        """
        Sets the transition condition for this node, depending on its kind.
        Used in json parsing.
        """
        match transition.kind:
            case TransitionKind.START:
                self._start_condition = transition
            case TransitionKind.PAUSE:
                self._pause_condition = transition
            case TransitionKind.END:
                self._end_condition = transition
            case TransitionKind.RESET:
                self._reset_condition = transition
            case _:
                raise ValueError(f"Unknown transition kind: {transition.kind}")

    @property
    def life_cycle_variable(self) -> LifeCycleVariable:
        """
        The variable representing the life cycle state of this node.
        :return:
        """
        if self._life_cycle_variable is None:
            raise NotInMotionStatechartError(self.name)
        return self._life_cycle_variable

    @property
    def observation_variable(self) -> ObservationVariable:
        if self._observation_variable is None:
            raise NotInMotionStatechartError(self.name)
        return self._observation_variable

    @property
    def motion_statechart(self) -> MotionStatechart:
        if self._motion_statechart is None:
            raise NotInMotionStatechartError(self.name)
        return self._motion_statechart

    @motion_statechart.setter
    def motion_statechart(self, motion_statechart: MotionStatechart) -> None:
        self._motion_statechart = motion_statechart

    def build(self, context: BuildContext) -> NodeArtifacts:
        """
        Describe this node by returning its constraints and the observation expression.
        Called exactly once during motion statechart compilation.
        .. warning:: Don't create other nodes within this function.
        """
        return NodeArtifacts(
            constraints=ConstraintCollection(),
        )

    def on_tick(self, context: ExecutionContext) -> Optional[ObservationStateValues]:
        """
        Triggered when the node is ticked.
        .. warning:: Only happens while the node is in state RUNNING.
        .. warning:: The result of this method takes precedence over the observation expression created in build().
        :return: An optional observation state overwrite
        """

    def on_start(self, context: ExecutionContext):
        """
        Triggered when the node transitions from NOT_STARTED to RUNNING.
        """

    def on_pause(self, context: ExecutionContext):
        """
        Triggered when the node transitions from RUNNING to PAUSED.
        """

    def on_unpause(self, context: ExecutionContext):
        """
        Triggered when the node transitions from PAUSED to RUNNING.
        """

    def on_end(self, context: ExecutionContext):
        """
        Triggered when the node transitions from RUNNING to DONE.
        """

    def on_reset(self, context: ExecutionContext):
        """
        Triggered when the node transitions from any state to NOT_STARTED.
        """

    def __hash__(self):
        return hash(self.name)

    @property
    def life_cycle_state(self) -> LifeCycleValues:
        """
        :return: The current life cycle state of this node.
        """
        return LifeCycleValues(self.motion_statechart.life_cycle_state[self])

    @property
    def observation_state(self) -> float:
        """
        :return: The current observation state of this node.
        """
        return self.motion_statechart.observation_state[self]

    @property
    def start_condition(self) -> cas.Expression:
        return self._start_condition.expression

    @start_condition.setter
    def start_condition(self, expression: cas.Expression) -> None:
        if self._start_condition is None:
            raise NotInMotionStatechartError(self.name)
        for var in expression.free_variables():
            if isinstance(var, ObservationVariable) and var.motion_statechart_node is self:
                raise InvalidSelfReferenceInStartCondition(self.name)
        self._check_condition_for_observation_variable(expression, "start")
        self._start_condition.update_expression(expression, self)

    @property
    def pause_condition(self) -> cas.Expression:
        return self._pause_condition.expression

    @pause_condition.setter
    def pause_condition(self, expression: cas.Expression) -> None:
        if self._pause_condition is None:
            raise NotInMotionStatechartError(self.name)
        self._check_condition_for_observation_variable(expression, "pause")
        self._pause_condition.update_expression(expression, self)

    @property
    def end_condition(self) -> cas.Expression:
        return self._end_condition.expression

    @end_condition.setter
    def end_condition(self, expression: cas.Expression) -> None:
        if self._end_condition is None:
            raise NotInMotionStatechartError(self.name)
        self._check_condition_for_observation_variable(expression, "end")
        self._end_condition.update_expression(expression, self)

    @property
    def reset_condition(self) -> cas.Expression:
        return self._reset_condition.expression

    @reset_condition.setter
    def reset_condition(self, expression: cas.Expression) -> None:
        if self._reset_condition is None:
            raise NotInMotionStatechartError(self.name)
        self._reset_condition.update_expression(expression, self)

    def _check_condition_for_observation_variable(self, expression: cas.Expression, condition_type: str):
        for var in expression.free_variables():
            if not isinstance(var, ObservationVariable):
                raise InvalidVariableInCondition(self.name, var.name, condition_type)

    def to_json(self) -> Dict[str, Any]:
        json_data = super().to_json()
        for field_ in fields(self):
            if not field_.name.startswith("_") and field_.init:
                value = getattr(self, field_.name)
                json_data[field_.name] = self._attribute_to_json(value)
        if self.parent_node_index is not None:
            json_data["parent_node_index"] = self.parent_node.index
        return json_data

    def _attribute_to_json(self, value: Any) -> Any:
        if isinstance(value, dict):
            return self._dict_to_json(value)
        if isinstance(value, list):
            return self._list_to_json(value)
        if isinstance(value, SubclassJSONSerializer):
            return value.to_json()
        return value

    def _list_to_json(self, list_attr: list) -> list:
        return [self._attribute_to_json(value) for value in list_attr]

    def _dict_to_json(self, dict_attr: Dict[Any, Any]) -> Dict[str, Any]:
        result = {}
        for key, value in dict_attr.items():
            json_value = self._attribute_to_json(value)
            result[key] = json_value
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        node_kwargs = {}
        for field_name, field_data in data.items():
            if field_name == "type":
                continue
            if isinstance(field_data, dict) and "type" in field_data:
                field_data = SubclassJSONSerializer.from_json(field_data, **kwargs)
            if isinstance(field_data, list):
                field_data = [
                    SubclassJSONSerializer.from_json(element_data, **kwargs)
                    for element_data in field_data
                ]
            if isinstance(field_data, dict):
                raise NotImplementedError(
                    "dict parameters of MotionStatechartNode are not supported yet. Use a list instead."
                )
            node_kwargs[field_name] = field_data
        parent_node_name = node_kwargs.pop("parent_node_index", None)
        result = cls(**node_kwargs)
        result.parent_node_name = parent_node_name
        return result

    def formatted_name(self, quoted: bool = False) -> str:
        formatted_name = string_shortener(
            original_str=str(self.name), max_lines=4, max_line_length=25
        )
        result = (
            f"{formatted_name}\n"
            f"----start_condition----\n"
            f"{str(self._start_condition)}\n"
            f"----pause_condition----\n"
            f"{str(self._pause_condition)}\n"
            f"----end_condition----\n"
            f"{str(self._end_condition)}\n"
            f"----reset_condition----\n"
            f"{str(self._reset_condition)}"
        )
        if quoted:
            return '"' + result + '"'
        return result

    @property
    def unique_name(self) -> str:
        return f"{self.name}#{self.index}"

    def __repr__(self) -> str:
        return self.unique_name


GenericMotionStatechartNode = TypeVar(
    "GenericMotionStatechartNode", bound=MotionStatechartNode
)


@dataclass(eq=False, repr=False)
class Task(MotionStatechartNode):
    """
    Tasks are MotionStatechartNodes that add motion constraints.
    """

    plot_specs: NodePlotSpec = field(
        default_factory=NodePlotSpec.create_task_style, kw_only=True, init=False
    )


@dataclass(eq=False, repr=False)
class Goal(MotionStatechartNode):
    nodes: List[MotionStatechartNode] = field(default_factory=list, init=False)
    plot_specs: NodePlotSpec = field(
        default_factory=NodePlotSpec.create_goal_style, kw_only=True, init=False
    )

    def expand(self, context: BuildContext) -> None:
        """
        Instantiate child nodes, add them to this goal, and wire their life cycle transition conditions.
        ..warning:: Nodes have not been built yet.
        """

    def add_node(self, node: MotionStatechartNode) -> None:
        """
        Adds a node to this goal and the motion statechart this goal belongs to.
        Should be used in expand().
        """
        if isinstance(node, EndMotion):
            raise InvalidGoalException(
                "EndMotion cannot be added as a child of a Goal. Place EndMotion at the MotionStatechart top level"
            )
        # Prevent duplicate insertion into the same goal
        if node in self.nodes:
            raise DuplicateNodeInGoalError(
                node_name=node.unique_name if node.index is not None else node.name,
                goal_name=self.unique_name if self.index is not None else self.name,
            )

        # Prevent cross-goal reuse of the same node instance
        if node.parent_node is not None and node.parent_node is not self:
            raise NodeAlreadyHasParentGoalError(
                node_name=node.unique_name if node.index is not None else node.name,
                current_parent=node.parent_node.unique_name,
                target_parent=self.unique_name if self.index is not None else self.name,
            )

        # Ensure node’s motion statechart matches the goal’s MSC (or is unassigned)
        if (
            node._motion_statechart is not None
            and node._motion_statechart is not self.motion_statechart
        ):
            raise NodeAlreadyInMotionStatechartError(
                node_name=node.unique_name if node.index is not None else node.name,
                current_msc=str(node._motion_statechart),
                target_msc=str(self.motion_statechart),
            )

        self.nodes.append(node)
        node.parent_node = self
        self.motion_statechart.add_node(node)

    def _apply_goal_conditions_to_children(self) -> None:
        """
        This method is called after expand() to link the conditions of this goal to its children.
        """
        for node in self.nodes:
            self._apply_start_condition_to_node(node)
            self._apply_pause_condition_to_node(node)
            self._apply_end_condition_to_node(node)
            self._apply_reset_condition_to_node(node)
            if isinstance(node, Goal):
                node._apply_goal_conditions_to_children()

    def _apply_start_condition_to_node(self, node: MotionStatechartNode) -> None:
        """
        Links the start condition of this goal to the start condition of the node.
        Ensures that the node can only be started when this goal is started.
        """
        if cas.is_const_trinary_true(node.start_condition):
            node.start_condition = self.start_condition
            return
        node.start_condition = cas.trinary_logic_and(
            node.start_condition, self.start_condition
        )

    def _apply_pause_condition_to_node(self, node: MotionStatechartNode) -> None:
        """
        Links the pause condition of this goal to the pause condition of the node.
        Ensures that the node is always paused when the goal is paused.
        """
        if cas.is_const_trinary_false(node.pause_condition):
            node.pause_condition = self.pause_condition
        elif not cas.is_const_trinary_false(node.pause_condition):
            node.pause_condition = cas.trinary_logic_or(
                node.pause_condition, self.pause_condition
            )

    def _apply_end_condition_to_node(self, node: MotionStatechartNode) -> None:
        """
        Links the end condition of this goal to the end condition of the node.
        Ensures that the node is automatically ended when the goal is ended.
        """
        if cas.is_const_trinary_false(node.end_condition):
            node.end_condition = self.end_condition
        elif not cas.is_const_trinary_false(self.end_condition):
            node.end_condition = cas.trinary_logic_or(
                node.end_condition, self.end_condition
            )

    def _apply_reset_condition_to_node(self, node: MotionStatechartNode):
        """
        Links the reset condition of this goal to the reset condition of the node.
        Ensures that the node is reset, when the goal is reset.
        """
        if cas.is_const_trinary_false(node.reset_condition):
            node.reset_condition = self.reset_condition
        elif not cas.is_const_trinary_false(node.pause_condition):
            node.reset_condition = cas.trinary_logic_or(
                node.reset_condition, self.reset_condition
            )


@dataclass(eq=False, repr=False)
class ThreadPayloadMonitor(MotionStatechartNode, ABC):
    """
    Payload monitor that evaluates _compute_observation in a background thread.

    - compute_observation triggers an async evaluation and immediately returns.
    - Until the first successful completion, returns TrinaryUnknown.
    - Afterwards, returns the last successfully computed value.
    """

    # Internal threading primitives
    _request_event: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    _stop_event: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    _thread: threading.Thread = field(init=False, repr=False)

    # Cache of last successful result from _compute_observation
    _has_result: bool = field(default=False, init=False, repr=False)
    _last_result: float = field(
        default=float(cas.TrinaryUnknown.to_np()), init=False, repr=False
    )

    def __post_init__(self):
        super().__post_init__()
        # Start a daemon worker thread that computes observations when requested
        self._thread = threading.Thread(
            target=self._worker_loop,
            name=f"{self.__class__.__name__}-worker",
            daemon=True,
        )
        self._thread.start()

    def compute_observation(
        self,
    ) -> float:
        # Signal the worker to compute a fresh value if it is not already signaled.
        self._request_event.set()
        # Return the last known result (initialized to Unknown until first success)
        return self._last_result

    def _worker_loop(self):
        while not self._stop_event.is_set():
            # Wait until a request is made (wake periodically to check for stop)
            triggered = self._request_event.wait(timeout=0.1)
            if not triggered:
                continue
            # Clear early to allow new requests while we compute
            self._request_event.clear()
            try:
                result = self._compute_observation()
                # Accept only valid trinary values (floats expected)
                self._last_result = result
                self._has_result = True
            except Exception:
                # On failure, keep previous result and mark as having no new value
                pass


@dataclass(eq=False, repr=False)
class EndMotion(MotionStatechartNode):

    plot_specs: NodePlotSpec = field(
        default_factory=NodePlotSpec.create_end_style, kw_only=True, init=False
    )

    def build(self, context: BuildContext) -> NodeArtifacts:
        return NodeArtifacts(observation=cas.TrinaryTrue)


@dataclass(eq=False, repr=False)
class CancelMotion(MotionStatechartNode):
    exception: Exception = field(kw_only=True)
    observation_expression: cas.Expression = field(
        default_factory=lambda: cas.TrinaryTrue, init=False
    )

    plot_specs: NodePlotSpec = field(
        default_factory=NodePlotSpec.create_cancel_style, kw_only=True, init=False
    )

    def build(self, context: BuildContext) -> NodeArtifacts:
        return NodeArtifacts(observation=cas.TrinaryTrue)

    def on_tick(self, context: ExecutionContext) -> Optional[float]:
        raise self.exception
