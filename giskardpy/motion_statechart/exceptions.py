from dataclasses import dataclass

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName


class MotionStatechartError(Exception):
    pass


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
class InvalidSelfReferenceInStartCondition(MotionStatechartError):
    node_name: str

    def __post_init__(self):
        super().__init__(
            f"Start condition of node '{self.node_name}' must not reference the node itself."
        )


@dataclass
class InvalidVariableInCondition(MotionStatechartError):
    node_name: str
    variable_name: PrefixedName
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