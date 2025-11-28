from dataclasses import field, dataclass

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.motion_statechart.context import BuildContext
from giskardpy.motion_statechart.graph_node import MotionStatechartNode, NodeArtifacts
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
    ActiveConnection1DOF,
)


@dataclass(eq=False, repr=False)
class JointPositionReached(MotionStatechartNode):
    connection: ActiveConnection1DOF = field(kw_only=True)
    position: float = field(kw_only=True)
    threshold: float = field(default=0.01, kw_only=True)

    def build(self, context: BuildContext) -> NodeArtifacts:
        current = self.connection.dof.variables.position
        if (
            isinstance(self.connection, RevoluteConnection)
            and not self.connection.dof.has_position_limits()
        ):
            error = cas.shortest_angular_distance(current, self.position)
        else:
            error = self.position - current
        return NodeArtifacts(observation=cas.abs(error) < self.threshold)
