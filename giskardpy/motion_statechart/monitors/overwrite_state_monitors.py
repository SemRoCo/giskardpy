from __future__ import division

from dataclasses import field, dataclass
from typing import Optional, Type, Tuple

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.motion_statechart.exceptions import GoalInitalizationException
from giskardpy.motion_statechart.context import BuildContext, ExecutionContext
from giskardpy.motion_statechart.graph_node import (
    MotionStatechartNode,
    NodeArtifacts,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointState
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
)
from semantic_digital_twin.world_description.world_entity import Connection


@dataclass(eq=False, repr=False)
class SetSeedConfiguration(MotionStatechartNode):
    """
    Overwrite the configuration of the world to allow starting the planning from a different state.
    CAUTION! don't use this to overwrite the robot's state outside standalone mode!
    :param seed_configuration: maps joint name to float
    :param group_name: if joint names are not unique, it will search in this group for matches.
    """

    seed_configuration: JointState = field(kw_only=True)

    def build(self, context: BuildContext) -> NodeArtifacts:
        return NodeArtifacts(observation=cas.TrinaryTrue)

    def on_start(self, context: ExecutionContext):
        # TODO does notify state change too often
        for connection, value in self.seed_configuration.items():
            connection.position = value


@dataclass(eq=False, repr=False)
class SetOdometry(MotionStatechartNode):
    base_pose: cas.TransformationMatrix = field(kw_only=True)
    _odom_joints: Tuple[Type[Connection], ...] = field(default=(OmniDrive,), init=False)
    odom_connection: Optional[OmniDrive] = field(default=None, kw_only=True)

    def build(self, context: BuildContext) -> NodeArtifacts:
        if self.odom_connection is None:
            drive_connections = context.world.get_connections_by_type(self._odom_joints)
            if len(drive_connections) == 0:
                raise GoalInitalizationException("No drive joints in world")
            elif len(drive_connections) == 1:
                self.odom_connection = drive_connections[0]
            else:
                raise GoalInitalizationException(
                    "Multiple drive joint found in world, please set 'group_name'"
                )
        return NodeArtifacts(observation=cas.TrinaryTrue)

    def on_start(self, context: ExecutionContext):
        parent_T_pose_ref = cas.TransformationMatrix(
            context.world.compute_forward_kinematics_np(
                self.odom_connection.parent, self.base_pose.reference_frame
            )
        )
        parent_T_pose = parent_T_pose_ref @ self.base_pose

        self.odom_connection.origin = parent_T_pose
