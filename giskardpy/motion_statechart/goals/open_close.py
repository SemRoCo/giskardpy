from __future__ import division

from dataclasses import dataclass, field
from typing import Optional

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.motion_statechart.context import BuildContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.graph_node import Goal, NodeArtifacts
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from semantic_digital_twin.spatial_types.spatial_types import trinary_logic_and
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass(eq=False, repr=False)
class Open(Goal):
    """
    Open a container in an environment.
    Only works with the environment was added as urdf.
    Assumes that a handle has already been grasped.
    Can only handle containers with 1 dof, e.g. drawers or doors.
    """

    tip_link: KinematicStructureEntity = field(kw_only=True)
    """end effector that is grasping the handle"""

    environment_link: KinematicStructureEntity = field(kw_only=True)
    """name of the handle that was grasped"""

    goal_joint_state: Optional[float] = field(default=None, kw_only=True)
    """goal state for the container. default is maximum joint state."""

    weight: float = field(default=DefaultWeights.WEIGHT_ABOVE_CA, kw_only=True)

    def expand(self, context: BuildContext) -> None:
        self.connection = self.environment_link.get_first_parent_connection_of_type(
            ActiveConnection1DOF
        )

        max_position = self.connection.dof.upper_limits.position
        if self.goal_joint_state is None:
            self.goal_joint_state = max_position
        else:
            self.goal_joint_state = min(max_position, self.goal_joint_state)

        self.add_nodes(
            [
                JointPositionList(
                    name="hinge goal",
                    goal_state=JointState({self.connection: self.goal_joint_state}),
                    weight=self.weight,
                ),
                CartesianPose(
                    name="hold handle",
                    root_link=self.environment_link,
                    tip_link=self.tip_link,
                    goal_pose=cas.TransformationMatrix(
                        reference_frame=self.tip_link, child_frame=self.tip_link
                    ),
                    weight=self.weight,
                ),
            ]
        )

    def build(self, context: BuildContext) -> NodeArtifacts:
        return NodeArtifacts(
            observation=trinary_logic_and(
                *[node.observation_variable for node in self.nodes]
            )
        )


@dataclass(eq=False, repr=False)
class Close(Open):
    """
    Open a container in an environment.
    Only works with the environment was added as urdf.
    Assumes that a handle has already been grasped.
    Can only handle containers with 1 dof, e.g. drawers or doors.
    """

    tip_link: KinematicStructureEntity = field(kw_only=True)
    """end effector that is grasping the handle"""

    environment_link: KinematicStructureEntity = field(kw_only=True)
    """name of the handle that was grasped"""

    goal_joint_state: Optional[float] = field(default=None, kw_only=True)
    """goal state for the container. default is maximum joint state."""

    weight: float = field(default=DefaultWeights.WEIGHT_ABOVE_CA, kw_only=True)

    def expand(self, context: BuildContext) -> None:
        self.connection = self.environment_link.get_first_parent_connection_of_type(
            ActiveConnection1DOF
        )

        min_position = self.connection.dof.lower_limits.position
        if self.goal_joint_state is None:
            self.goal_joint_state = min_position
        else:
            self.goal_joint_state = max(min_position, self.goal_joint_state)

        self.add_nodes(
            [
                JointPositionList(
                    name="hinge goal",
                    goal_state=JointState({self.connection: self.goal_joint_state}),
                    weight=self.weight,
                ),
                CartesianPose(
                    name="hold handle",
                    root_link=self.environment_link,
                    tip_link=self.tip_link,
                    goal_pose=cas.TransformationMatrix(
                        reference_frame=self.tip_link, child_frame=self.tip_link
                    ),
                    weight=self.weight,
                ),
            ]
        )

    def build(self, context: BuildContext) -> NodeArtifacts:
        return NodeArtifacts(
            observation=trinary_logic_and(
                *[node.observation_variable for node in self.nodes]
            )
        )
