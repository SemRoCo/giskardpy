from dataclasses import dataclass

from krrood.adapters.json_serializer import JSON_TYPE_NAME, JSONSerializableTypeRegistry
from krrood.utils import get_full_class_name
from typing_extensions import Any, Dict


class MotionStatechartError(Exception):
    pass


class GoalInitalizationException(MotionStatechartError):
    pass


class EmptyMotionStatechartError(MotionStatechartError):
    def __init__(self):
        super().__init__("MotionStatechart is empty.")


@dataclass
class NodeNotFoundError(MotionStatechartError):
    name: str

    def __post_init__(self):
        super().__init__(f"Node '{self.name}' not found in MotionStatechart.")


@dataclass
class NotInMotionStatechartError(MotionStatechartError):
    name: str

    def __post_init__(self):
        super().__init__(
            f"Operation can't be performed because node '{self.name}' does not belong to a MotionStatechart."
        )


@dataclass
class InvalidConditionError(MotionStatechartError):
    expression: Any

    def __post_init__(self):
        super().__init__(
            f"Invalid condition: {self.expression}. Did you forget '.observation_variable'?"
        )


def serialize_exception(obj: Exception) -> Dict[str, Any]:

    return {
        JSON_TYPE_NAME: get_full_class_name(type(obj)),
        "value": str(obj),
    }


def deserialize_exception(data: Dict[str, Any]) -> Exception:

    return Exception(data["value"])


JSONSerializableTypeRegistry().register(
    Exception, serialize_exception, deserialize_exception
)


@dataclass
class InvalidSelfReferenceInStartCondition(MotionStatechartError):
    node_name: str

    def __post_init__(self):
        super().__init__(
            f"Start condition of node '{self.node_name}' must not reference the node itself."
        )


@dataclass
class InvalidVariableInCondition(MotionStatechartError):
    node_name: str
    variable_name: str
    condition_type: str

    def __post_init__(self):
        super().__init__(
            f"Variable {self.variable_name} in {self.condition_type}condition of node '{self.node_name}' is not an observation variable."
        )


@dataclass
class NodeAlreadyInMotionStatechartError(MotionStatechartError):
    node_name: str
    current_msc: str
    target_msc: str

    def __post_init__(self):
        super().__init__(
            f"Node '{self.node_name}' already belongs to MotionStatechart '{self.current_msc}' and cannot be added to '{self.target_msc}'."
        )


@dataclass
class NodeAlreadyHasParentGoalError(MotionStatechartError):
    node_name: str
    current_parent: str
    target_parent: str

    def __post_init__(self):
        super().__init__(
            f"Node '{self.node_name}' already has parent goal '{self.current_parent}' and cannot be added to '{self.target_parent}'."
        )


@dataclass
class DuplicateNodeInGoalError(MotionStatechartError):
    node_name: str
    goal_name: str

    def __post_init__(self):
        super().__init__(
            f"Node '{self.node_name}' is already part of goal '{self.goal_name}'."
        )