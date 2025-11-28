from dataclasses import field, dataclass, InitVar
from typing import Optional, Dict, List, Tuple, Union, Any

from krrood.adapters.json_serializer import SubclassJSONSerializer
from typing_extensions import Self

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.motion_statechart.exceptions import GoalInitalizationException
from giskardpy.motion_statechart.context import BuildContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.graph_node import NodeArtifacts
from giskardpy.motion_statechart.graph_node import Task
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.derivatives import Derivatives
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
    ActiveConnection,
    PrismaticConnection,
    ActiveConnection1DOF,
)


@dataclass
class JointState(SubclassJSONSerializer):
    mapping: InitVar[Dict[ActiveConnection1DOF, float]]

    _connections: List[ActiveConnection1DOF] = field(init=False, default_factory=list)
    _target_values: List[float] = field(init=False, default_factory=list)

    def __post_init__(self, mapping: Dict[ActiveConnection1DOF, float]):
        for connection, target in mapping.items():
            self._connections.append(connection)
            self._target_values.append(target)

    def __len__(self):
        return len(self._connections)

    def items(self):
        return zip(self._connections, self._target_values)

    @classmethod
    def from_str_dict(cls, mapping: Dict[str, float], world: World):
        mapping = {
            world.get_connection_by_name(connection_name): target
            for connection_name, target in mapping.items()
        }
        return cls(mapping=mapping)

    @classmethod
    def from_lists(cls, connections: List[ActiveConnection1DOF], targets: List[float]):
        return cls(mapping=dict(zip(connections, targets)))

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "_connections": [
                connection.name.to_json() for connection in self._connections
            ],
            "_target_values": self._target_values,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        world: World = kwargs["world"]
        connections = [
            world.get_connection_by_name(PrefixedName.from_json(name, **kwargs))
            for name in data["_connections"]
        ]
        target_values = data["_target_values"]
        return cls.from_lists(connections, target_values)


@dataclass(eq=False, repr=False)
class JointPositionList(Task):
    goal_state: JointState = field(kw_only=True)
    threshold: float = field(default=0.01, kw_only=True)
    weight: float = field(default=DefaultWeights.WEIGHT_BELOW_CA, kw_only=True)
    max_velocity: float = field(default=1.0, kw_only=True)

    def build(self, context: BuildContext) -> NodeArtifacts:
        if len(self.goal_state) == 0:
            raise GoalInitalizationException(f"Can't initialize {self} with no joints.")

        artifacts = NodeArtifacts()

        errors = []
        for connection, target in self.goal_state.items():
            current = connection.dof.variables.position
            target = self.apply_limits_to_target(target, connection)
            velocity = self.apply_limits_to_velocity(self.max_velocity, connection)
            if (
                isinstance(connection, RevoluteConnection)
                and not connection.dof.has_position_limits()
            ):
                error = cas.shortest_angular_distance(current, target)
            else:
                error = target - current
            artifacts.constraints.add_equality_constraint(
                name=str(connection.name),
                reference_velocity=velocity,
                equality_bound=error,
                weight=self.weight,
                task_expression=current,
            )
            errors.append(cas.abs(error) < self.threshold)
        artifacts.observation = cas.logic_all(cas.Expression(errors))
        return artifacts

    def apply_limits_to_target(
        self, target: float, connection: ActiveConnection1DOF
    ) -> cas.Expression:
        ul_pos = connection.dof.upper_limits.position
        ll_pos = connection.dof.lower_limits.position
        if ll_pos is not None:
            target = cas.limit(target, ll_pos, ul_pos)
        return target

    def apply_limits_to_velocity(
        self, velocity: float, connection: ActiveConnection1DOF
    ) -> cas.Expression:
        ul_vel = connection.dof.upper_limits.velocity
        ll_vel = connection.dof.lower_limits.velocity
        return cas.limit(velocity, ll_vel, ul_vel)


@dataclass
class MirrorJointPosition(Task):
    mapping: Dict[Union[PrefixedName, str], str] = field(default_factory=lambda: dict)
    threshold: float = 0.01
    weight: Optional[float] = None
    max_velocity: Optional[float] = None

    def __post_init__(self):
        if self.weight is None:
            self.weight = DefaultWeights.WEIGHT_BELOW_CA
        if self.max_velocity is None:
            self.max_velocity = 1.0
        self.current_positions = []
        self.goal_positions = []
        self.velocity_limits = []
        self.connections = []
        goal_state = {}
        for joint_name, target_joint_name in self.mapping.items():
            connection = context.world.get_connection_by_name(joint_name)
            self.connections.append(connection)
            target_connection = context.world.get_connection_by_name(target_joint_name)

            ll_vel = connection.dof.lower_limits
            ul_vel = connection.dof.upper_limits
            velocity_limit = cas.limit(self.max_velocity, ll_vel, ul_vel)
            self.current_positions.append(connection.position)
            self.goal_positions.append(target_connection.position)
            self.velocity_limits.append(velocity_limit)
            goal_state[joint_name.name] = self.goal_positions[-1]

        for connection, current, goal, velocity_limit in zip(
            self.connections,
            self.current_positions,
            self.goal_positions,
            self.velocity_limits,
        ):
            if (
                isinstance(connection, RevoluteConnection)
                and not connection.dof.has_position_limits()
            ):
                error = cas.shortest_angular_distance(current, goal)
            else:
                error = goal - current

            self.add_equality_constraint(
                name=f"{self.name}/{connection.name}",
                reference_velocity=velocity_limit,
                equality_bound=0,
                weight=self.weight,
                task_expression=error,
            )
        joint_monitor = JointGoalReached(
            goal_state=goal_state, threshold=self.threshold
        )
        self.observation_expression = joint_monitor.observation_expression


@dataclass
class JointPositionLimitList(Task):
    lower_upper_limits: Dict[Union[PrefixedName, str], Tuple[float, float]] = field(
        kw_only=True
    )
    weight: float = DefaultWeights.WEIGHT_BELOW_CA
    max_velocity: float = 1

    def __post_init__(self):
        """
        Calls JointPosition for a list of joints.
        :param goal_state: maps joint_name to goal position
        :param group_name: if joint_name is not unique, search in this group for matches.
        :param weight:
        :param max_velocity: will be applied to all joints, you should group joint types, e.g., prismatic joints
        :param hard: turns this into a hard constraint.
        """
        self.current_positions = []
        self.lower_limits = []
        self.upper_limits = []
        self.velocity_limits = []
        self.connections = []
        self.joint_names = list(sorted(self.lower_upper_limits.keys()))
        if len(self.lower_upper_limits) == 0:
            raise GoalInitalizationException(f"Can't initialize {self} with no joints.")

        for joint_name, (lower_limit, upper_limit) in self.lower_upper_limits.items():
            connection: ActiveConnection1DOF = context.world.get_connection_by_name(
                joint_name
            )
            self.connections.append(connection)

            ll_pos = connection.dof.lower_limits.position
            ul_pos = connection.dof.upper_limits.position

            if ll_pos is not None:
                lower_limit = min(ul_pos, max(ll_pos, lower_limit))
                upper_limit = min(ul_pos, max(ll_pos, upper_limit))

            ll_vel = connection.dof.lower_limits.velocity
            ul_vel = connection.dof.upper_limits.velocity

            velocity_limit = min(ul_vel, max(ll_vel, self.max_velocity))

            self.current_positions.append(connection.position)
            self.lower_limits.append(lower_limit)
            self.upper_limits.append(upper_limit)
            self.velocity_limits.append(velocity_limit)

        for connection, current, lower_limit, upper_limit, velocity_limit in zip(
            self.connections,
            self.current_positions,
            self.lower_limits,
            self.upper_limits,
            self.velocity_limits,
        ):
            if (
                isinstance(connection, RevoluteConnection)
                and not connection.dof.has_position_limits()
            ):
                lower_error = cas.shortest_angular_distance(current, lower_limit)
                upper_error = cas.shortest_angular_distance(current, upper_limit)
            else:
                lower_error = lower_limit - current
                upper_error = upper_limit - current

            self.add_inequality_constraint(
                name=f"{self.name}/{connection.name}",
                reference_velocity=velocity_limit,
                lower_error=lower_error,
                upper_error=upper_error,
                weight=self.weight,
                task_expression=current,
            )


@dataclass
class JustinTorsoLimit(Task):
    connection: ActiveConnection = field(kw_only=True)
    lower_limit: Optional[float] = None
    upper_limit: Optional[float] = None
    weight: float = DefaultWeights.WEIGHT_BELOW_CA
    max_velocity: float = 1

    def __post_init__(self):
        joint = self.connection

        current = joint.q3

        if isinstance(self.connection, RevoluteConnection) or isinstance(
            self.connection, PrismaticConnection
        ):
            lower_error = cas.shortest_angular_distance(current, self.lower_limit)
            upper_error = cas.shortest_angular_distance(current, self.upper_limit)
        else:
            lower_error = self.lower_limit - current
            upper_error = self.upper_limit - current

        context.add_debug_expression("torso 4 joint", current)
        context.add_debug_expression(
            "torso 2 joint", joint.q1.get_symbol(Derivatives.position)
        )
        context.add_debug_expression(
            "torso 3 joint", joint.q2.get_symbol(Derivatives.position)
        )
        context.add_debug_expression("lower_limit", self.lower_limit)
        context.add_debug_expression("upper_limit", self.upper_limit)

        self.add_inequality_constraint(
            name=self.name,
            reference_velocity=1,
            lower_error=lower_error,
            upper_error=upper_error,
            weight=self.weight,
            task_expression=current,
        )


@dataclass
class JointVelocityLimit(Task):
    joints: List[ActiveConnection1DOF] = field(kw_only=True)
    weight: float = DefaultWeights.WEIGHT_BELOW_CA
    max_velocity: float = 1
    hard: bool = False

    def __post_init__(self):
        """
        Limits the joint velocity of a revolute joint.
        :param joint_name:
        :param group_name: if joint_name is not unique, will search in this group for matches.
        :param weight:
        :param max_velocity: rad/s
        :param hard: turn this into a hard constraint.
        """
        for joint in self.joints:
            current_joint = joint.joint_position_expression
            try:
                limit_expr = joint.dof.upper_limits.velocity
                max_velocity = cas.min(self.max_velocity, limit_expr)
            except IndexError:
                max_velocity = self.max_velocity
            if self.hard:
                self.add_velocity_constraint(
                    lower_velocity_limit=-max_velocity,
                    upper_velocity_limit=max_velocity,
                    weight=self.weight,
                    task_expression=current_joint,
                    velocity_limit=max_velocity,
                    lower_slack_limit=0,
                    upper_slack_limit=0,
                )
            else:
                self.add_velocity_constraint(
                    lower_velocity_limit=-max_velocity,
                    upper_velocity_limit=max_velocity,
                    weight=self.weight,
                    task_expression=current_joint,
                    velocity_limit=max_velocity,
                    name=joint.name.name,
                )


@dataclass
class JointVelocity(Task):
    connections: List[ActiveConnection1DOF] = field(kw_only=True)
    vel_goal: float = field(kw_only=True)
    weight: float = DefaultWeights.WEIGHT_BELOW_CA
    max_velocity: float = 1
    hard: bool = False

    def __post_init__(self):
        """
        Limits the joint velocity of a revolute joint.
        :param connection:
        :param group_name: if connection is not unique, will search in this group for matches.
        :param weight:
        :param max_velocity: rad/s
        :param hard: turn this into a hard constraint.
        """
        for connection in self.connections:
            current_joint = connection.dof.variables.position
            try:
                limit_expr = connection.dof.upper_limits.velocity
                max_velocity = cas.min(self.max_velocity, limit_expr)
            except IndexError:
                max_velocity = self.max_velocity
            self.add_velocity_eq_constraint(
                velocity_goal=self.vel_goal,
                weight=self.weight,
                task_expression=current_joint,
                velocity_limit=max_velocity,
                name=str(connection.name),
            )


@dataclass
class AvoidJointLimits(Task):
    """
    Calls AvoidSingleJointLimits for each joint in joint_list
    """

    percentage: float = 15

    connection_list: Optional[List[ActiveConnection1DOF]] = None
    """list of joints for which AvoidSingleJointLimits will be called"""

    weight: float = DefaultWeights.WEIGHT_BELOW_CA

    def __post_init__(self):
        if self.connection_list is None:
            self.connection_list = context.world.controlled_connections
        for connection in self.connection_list:
            if isinstance(connection, (RevoluteConnection, PrismaticConnection)):
                if not connection.dof.has_position_limits():
                    continue
                weight = self.weight
                connection_symbol = connection.dof.variables.position
                percentage = self.percentage / 100.0
                lower_limit = connection.dof.lower_limits.position
                upper_limit = connection.dof.upper_limits.position
                max_velocity = cas.min(100, connection.dof.upper_limits.velocity)

                joint_range = upper_limit - lower_limit
                center = (upper_limit + lower_limit) / 2.0

                max_error = joint_range / 2.0 * percentage

                upper_goal = center + joint_range / 2.0 * (1 - percentage)
                lower_goal = center - joint_range / 2.0 * (1 - percentage)

                upper_err = upper_goal - connection_symbol
                lower_err = lower_goal - connection_symbol

                error = cas.max(
                    cas.abs(cas.min(upper_err, 0)), cas.abs(cas.max(lower_err, 0))
                )
                weight = weight * (error / max_error)

                self.add_inequality_constraint(
                    reference_velocity=max_velocity,
                    name=str(connection.name),
                    lower_error=lower_err,
                    upper_error=upper_err,
                    weight=weight,
                    task_expression=connection_symbol,
                )
