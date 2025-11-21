from dataclasses import dataclass, field

import semantic_digital_twin.spatial_types.spatial_types as cas

from giskardpy.motion_statechart.context import BuildContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.graph_node import Task, NodeArtifacts, DebugExpression
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import KinematicStructureEntity

@dataclass
class AlignPlanes(Task):
    """
    Will orient the tip plane to align with the goal plane.
    The planes are represented as normal vectors.
    """
    root_link: KinematicStructureEntity = field(kw_only=True)
    """root link of the kinematic chain."""
    tip_link: KinematicStructureEntity = field(kw_only=True)
    """tip link of the kinematic chain."""

    goal_normal: cas.Vector3 = field(kw_only=True)
    """normal vector of the goal plane."""
    tip_normal: cas.Vector3 = field(kw_only=True)
    """normal vector of the tip plane."""

    threshold: float = field(default = 0.01, kw_only=True)
    reference_velocity: float = field(default = 0.5, kw_only=True)
    weight: float = field(default=DefaultWeights.WEIGHT_ABOVE_CA, kw_only=True)

    def build(self, context: BuildContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        tip_V_tip_normal = context.world.transform(
             target_frame=self.tip_link, spatial_object=self.tip_normal
        )
        tip_V_tip_normal.scale(1)
        root_V_root_normal = context.world.transform(
            target_frame=self.root_link, spatial_object=self.goal_normal
        )
        root_V_root_normal.scale(1)

        root_R_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_rotation_matrix()
        root_V_tip_normal = root_R_tip @ tip_V_tip_normal
        root_V_tip_normal.scale(1)

        root_V_tip_normal.vis_frame = self.tip_link
        current_dbg = DebugExpression(
            name=f"{self.name}/current_normal",
            expression=root_V_tip_normal,
            color=Color(1, 0, 0, 1),
        )
        goal_dbg = DebugExpression(
            name=f"{self.name}/goal_normal",
            expression=root_V_root_normal,
            color=Color(0, 0, 1, 1),
        )
        artifacts.debug_expressions.append(current_dbg)
        artifacts.debug_expressions.append(goal_dbg)

        artifacts.constraints.add_vector_goal_constraints(
            frame_V_current=root_V_tip_normal,
            frame_V_goal=root_V_root_normal,
            reference_velocity=self.reference_velocity,
            weight=self.weight,
        )
        artifacts.observation = (
            root_V_tip_normal.angle_between(root_V_root_normal) <= self.threshold
        )

        return artifacts