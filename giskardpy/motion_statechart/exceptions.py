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