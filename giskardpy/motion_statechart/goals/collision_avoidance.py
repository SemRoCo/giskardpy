from collections import defaultdict
from dataclasses import field, dataclass
from typing import Dict, Optional, List, Tuple

from line_profiler import profile

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.data_types.exceptions import GiskardException
from giskardpy.middleware import get_middleware
from giskardpy.model.collision_matrix_manager import CollisionRequest
from giskardpy.motion_statechart.context import BuildContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.graph_node import (
    Goal,
    MotionStatechartNode,
    NodeArtifacts,
)
from giskardpy.motion_statechart.graph_node import (
    Task,
)
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)


@dataclass(eq=False, repr=False)
class ExternalCollisionDistanceMonitor(MotionStatechartNode):
    tip: Body = field(kw_only=True)
    idx: int = field(default=0, kw_only=True)

    def build(self, context: BuildContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        artifacts.observation = (
            context.collision_scene.external_contact_distance_symbol(self.tip, self.idx)
            > 50
        )

        return artifacts


@dataclass(eq=False, repr=False)
class ExternalCollisionAvoidanceTask(Task):
    """
    Moves root_T_tip @ tip_P_contact in root_T_contact_normal direction until the distance is larger than buffer_zone.
    Limits the slack variables to prevent the tip from coming closer than violated_distance.
    Can result in insolvable QPs if multiple of these constraints are violated.
    """

    connection: ActiveConnection = field(kw_only=True)
    max_velocity: float = field(default=0.2, kw_only=True)
    idx: int = field(default=0, kw_only=True)
    max_avoided_bodies: int = field(default=1, kw_only=True)

    @property
    def tip(self) -> KinematicStructureEntity:
        return self.connection.child

    def create_weight(self, context: BuildContext) -> cas.Expression:
        """
        Creates a weight for this task which is scaled by the number of external collisions.
        :return:
        """
        number_of_external_collisions = (
            context.collision_scene.external_number_of_collisions_symbol(self.tip)
        )
        weight = cas.Expression(
            data=DefaultWeights.WEIGHT_COLLISION_AVOIDANCE
        ).safe_division(cas.min(number_of_external_collisions, self.max_avoided_bodies))
        return weight

    def create_buffer_zone_expression(
        self, context: BuildContext
    ) -> Tuple[cas.Expression, cas.Expression]:
        """
        Creates an expression that is equal to the buffer zone distance of the body that is currently
        closest to the main body.
        """
        direct_children = context.world.get_direct_child_bodies_with_collision(
            self.connection
        )

        buffer_zone_distance = max(
            b.get_collision_config().buffer_zone_distance
            for b in direct_children
            if b.get_collision_config().buffer_zone_distance is not None
        )
        violated_distance = max(
            b.get_collision_config().violated_distance for b in direct_children
        )

        actual_link_b_hash = context.collision_scene.external_link_b_hash_symbol(
            self.tip, self.idx
        )
        b_result_cases = []
        for body in context.world.bodies_with_enabled_collision:
            if body.get_collision_config().buffer_zone_distance is None:
                continue
            if body.get_collision_config().disabled:
                continue
            if body.get_collision_config().buffer_zone_distance > buffer_zone_distance:
                b_result_cases.append(
                    (body.__hash__(), body.get_collision_config().buffer_zone_distance)
                )
        if len(b_result_cases) > 0:
            buffer_zone_expr = cas.if_eq_cases(
                a=actual_link_b_hash,
                b_result_cases=b_result_cases,
                else_result=cas.Expression(buffer_zone_distance),
            )
        else:
            buffer_zone_expr = buffer_zone_distance

        return buffer_zone_expr, cas.Expression(violated_distance)

    def compute_control_horizon(
        self, qp_controller_config: QPControllerConfig
    ) -> float:
        control_horizon = qp_controller_config.prediction_horizon - (
            qp_controller_config.max_derivative - 1
        )
        return max(1, control_horizon)

    def create_upper_slack(
        self,
        context: BuildContext,
        lower_limit: cas.Expression,
        buffer_zone_expr: cas.Expression,
        violated_distance: cas.Expression,
        distance_expression: cas.Expression,
    ) -> cas.Expression:
        qp_limits_for_lba = (
            self.max_velocity
            * context.qp_controller_config.mpc_dt
            * self.compute_control_horizon(context.qp_controller_config)
        )

        hard_threshold = cas.min(violated_distance, buffer_zone_expr / 2)

        lower_limit_limited = cas.limit(
            lower_limit, -qp_limits_for_lba, qp_limits_for_lba
        )

        upper_slack = cas.if_greater(
            distance_expression,
            hard_threshold,
            lower_limit_limited + cas.max(0, distance_expression - hard_threshold),
            lower_limit_limited - 1e-4,
        )
        # undo factor in A
        upper_slack /= context.qp_controller_config.mpc_dt

        upper_slack = cas.if_greater(
            distance_expression,
            50,  # assuming that distance of unchecked closest points is 100
            cas.Expression(1e4),
            cas.max(0, upper_slack),
        )
        return upper_slack

    def build(self, context: BuildContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        root_V_contact_normal = context.collision_scene.external_map_V_n_symbol(
            self.tip, self.idx
        )
        tip_P_contact = context.collision_scene.external_new_a_P_pa_symbol(
            self.tip, self.idx
        )
        distance_expression = context.collision_scene.external_contact_distance_symbol(
            self.tip, self.idx
        )

        buffer_zone_expr, violated_distance = self.create_buffer_zone_expression(
            context
        )

        map_T_a = context.world.compose_forward_kinematics_expression(
            context.world.root, self.tip
        )

        map_V_pa = cas.Vector3.from_iterable(map_T_a @ tip_P_contact)

        # the position distance is not accurate, but the derivative is still correct
        dist = root_V_contact_normal @ map_V_pa

        lower_limit = buffer_zone_expr - distance_expression

        artifacts.constraints.add_inequality_constraint(
            name=self.name,
            reference_velocity=self.max_velocity,
            lower_error=lower_limit,
            upper_error=float("inf"),
            weight=self.create_weight(context),
            task_expression=dist,
            lower_slack_limit=-float("inf"),
            upper_slack_limit=self.create_upper_slack(
                context=context,
                lower_limit=lower_limit,
                buffer_zone_expr=buffer_zone_expr,
                violated_distance=violated_distance,
                distance_expression=cas.Expression(distance_expression),
            ),
        )

        return artifacts


@dataclass(eq=False, repr=False)
class ExternalCollisionAvoidance(Goal):
    """
    Avoidance collision between all direct children of a connection and the environment.
    """

    connection: ActiveConnection = field(kw_only=True)
    max_velocity: float = field(default=0.2, kw_only=True)
    idx: int = field(default=0, kw_only=True)
    max_avoided_bodies: int = field(default=1, kw_only=True)

    # %% init false
    buffer_zone_distance: float = field(init=False)
    violated_distance: float = field(init=False)
    _main_body: Body = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.connection.child, Body):
            self._main_body = self.connection.child
        else:
            raise GiskardException(
                "Child of connection must be of type body for collision avoidance."
            )

    def expand(self, context: BuildContext) -> None:
        distance_monitor = ExternalCollisionDistanceMonitor(
            name=PrefixedName("collision distance", str(self.name)),
            tip=self._main_body,
            idx=self.idx,
        )

        self.add_node(distance_monitor)

        task = ExternalCollisionAvoidanceTask(
            name=PrefixedName(f"task", str(self.name)),
            connection=self.connection,
            max_velocity=self.max_velocity,
            max_avoided_bodies=self.max_avoided_bodies,
            idx=self.idx,
        )
        self.add_node(task)

        task.pause_condition = distance_monitor.observation_variable

    def build(self, context: BuildContext) -> NodeArtifacts:
        context.collision_scene.monitor_link_for_external(self._main_body, self.idx)
        return NodeArtifacts()


@dataclass(eq=False, repr=False)
class SelfCollisionDistanceMonitor(MotionStatechartNode):
    body_a: Body = field(kw_only=True)
    body_b: Body = field(kw_only=True)
    idx: int = field(default=0, kw_only=True)

    def build(self, context: BuildContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        artifacts.observation = (
            context.collision_scene.self_contact_distance_symbol(
                self.body_a, self.body_b, self.idx
            )
            > 50
        )

        return artifacts


@dataclass(eq=False, repr=False)
class SelfCollisionAvoidanceTask(Task):
    body_a: Body = field(kw_only=True)
    body_b: Body = field(kw_only=True)
    max_velocity: float = field(default=0.2, kw_only=True)
    idx: int = field(default=0, kw_only=True)
    max_avoided_bodies: int = field(default=1, kw_only=True)
    buffer_zone_distance: float = field(kw_only=True)

    def build(self, context: BuildContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        self.root = context.world.root
        self.control_horizon = context.qp_controller_config.prediction_horizon - (
            context.qp_controller_config.max_derivative - 1
        )
        self.control_horizon = max(1, self.control_horizon)
        # buffer_zone_distance = max(self.body_a.collision_config.buffer_zone_distance,
        #                            self.body_b.collision_config.buffer_zone_distance)
        violated_distance = max(
            self.body_a.get_collision_config().violated_distance,
            self.body_b.get_collision_config().violated_distance,
        )
        violated_distance = cas.min(violated_distance, self.buffer_zone_distance / 2)
        actual_distance = context.collision_scene.self_contact_distance_symbol(
            self.body_a, self.body_b, self.idx
        )
        number_of_self_collisions = (
            context.collision_scene.self_number_of_collisions_symbol(
                self.body_a, self.body_b
            )
        )
        sample_period = context.qp_controller_config.mpc_dt

        b_T_a = context.world._forward_kinematic_manager.compose_expression(
            self.body_b, self.body_a
        )
        b_P_pb = context.collision_scene.self_new_b_P_pb_symbol(
            self.body_a, self.body_b, self.idx
        )
        pb_T_b = cas.TransformationMatrix.from_point_rotation_matrix(
            point=b_P_pb
        ).inverse()
        a_P_pa = context.collision_scene.self_new_a_P_pa_symbol(
            self.body_a, self.body_b, self.idx
        )

        pb_V_n = context.collision_scene.self_new_b_V_n_symbol(
            self.body_a, self.body_b, self.idx
        )

        pb_V_pa = cas.Vector3.from_iterable(pb_T_b @ b_T_a @ a_P_pa)

        dist = pb_V_n @ pb_V_pa

        qp_limits_for_lba = self.max_velocity * sample_period * self.control_horizon

        lower_limit = self.buffer_zone_distance - actual_distance

        lower_limit_limited = cas.limit(
            lower_limit, -qp_limits_for_lba, qp_limits_for_lba
        )

        upper_slack = cas.if_greater(
            actual_distance,
            violated_distance,
            lower_limit_limited + cas.max(0, actual_distance - violated_distance),
            lower_limit_limited,
        )

        # undo factor in A
        upper_slack /= sample_period

        upper_slack = cas.if_greater(
            actual_distance,
            50,  # assuming that distance of unchecked closest points is 100
            cas.Expression(1e4),
            cas.max(0, upper_slack),
        )

        weight = cas.Expression(
            data=DefaultWeights.WEIGHT_COLLISION_AVOIDANCE
        ).safe_division(cas.min(number_of_self_collisions, self.max_avoided_bodies))

        artifacts.constraints.add_inequality_constraint(
            name=self.name,
            reference_velocity=self.max_velocity,
            lower_error=lower_limit,
            upper_error=float("inf"),
            weight=weight,
            task_expression=dist,
            lower_slack_limit=-float("inf"),
            upper_slack_limit=upper_slack,
        )

        return artifacts


@dataclass(eq=False, repr=False)
class SelfCollisionAvoidance(Goal):
    body_a: Body = field(kw_only=True)
    body_b: Body = field(kw_only=True)
    max_velocity: float = field(default=0.2, kw_only=True)
    idx: int = field(default=0, kw_only=True)
    max_avoided_bodies: int = field(default=1, kw_only=True)
    buffer_zone_distance: float = field(kw_only=True)

    def expand(self, context: BuildContext) -> None:
        distance_monitor = SelfCollisionDistanceMonitor(
            name=PrefixedName("collision distance", str(self.name)),
            body_a=self.body_a,
            body_b=self.body_b,
            idx=self.idx,
        )
        self.add_node(distance_monitor)

        task = SelfCollisionAvoidanceTask(
            name=PrefixedName(f"task", str(self.name)),
            body_a=self.body_a,
            body_b=self.body_b,
            max_velocity=self.max_velocity,
            idx=self.idx,
            max_avoided_bodies=self.max_avoided_bodies,
            buffer_zone_distance=self.buffer_zone_distance,
        )
        self.add_node(task)

        task.pause_condition = distance_monitor.observation_variable

    def build(self, context: BuildContext) -> NodeArtifacts:
        context.collision_scene.monitor_link_for_self(
            self.body_a, self.body_b, self.idx
        )
        return NodeArtifacts()


# use cases
# avoid all
# allow all
# avoid all then allow something
# avoid only something


@dataclass(eq=False, repr=False)
class CollisionAvoidance(Goal):
    collision_entries: List[CollisionRequest] = field(default_factory=list)

    def expand(self, context: BuildContext) -> None:
        context.collision_scene.matrix_manager.parse_collision_requests(
            self.collision_entries
        )
        self.collision_entries = (
            context.collision_scene.matrix_manager.collision_requests
        )
        if (
            not self.collision_entries
            or not self.collision_entries[-1].is_allow_all_collision()
        ):
            self.add_external_collision_avoidance_constraints(context)
        if (
            not self.collision_entries
            or not self.collision_entries[-1].is_allow_all_collision()
        ):
            self.add_self_collision_avoidance_constraints(context)
        collision_matrix = (
            context.collision_scene.matrix_manager.compute_collision_matrix()
        )
        context.collision_scene.set_collision_matrix(collision_matrix)

    @profile
    def add_external_collision_avoidance_constraints(self, context: BuildContext):
        robot: AbstractRobot
        # thresholds = god_map.collision_scene.matrix_manager.external_thresholds
        for robot in context.world.get_semantic_annotations_by_type(AbstractRobot):
            if robot.drive:
                connection_list = robot.controlled_connections.union({robot.drive})
            else:
                connection_list = robot.controlled_connections
            for connection in connection_list:
                if connection.frozen_for_collision_avoidance:
                    continue
                bodies = context.world.get_direct_child_bodies_with_collision(
                    connection
                )
                if not bodies:
                    continue
                max_avoided_bodies = 0
                for body in bodies:
                    max_avoided_bodies = max(
                        max_avoided_bodies,
                        body.get_collision_config().max_avoided_bodies,
                    )
                for idx in range(max_avoided_bodies):
                    self.add_node(
                        node := ExternalCollisionAvoidance(
                            name=PrefixedName(
                                f"{connection.name}/{idx}", str(self.name)
                            ),
                            connection=connection,
                            idx=idx,
                            max_avoided_bodies=max_avoided_bodies,
                        )
                    )
                    node.plot_specs.visible = False

        # get_middleware().loginfo(f'Adding {num_constrains} external collision avoidance constraints.')

    @profile
    def add_self_collision_avoidance_constraints(self, context: BuildContext):
        counter: Dict[Tuple[Body, Body], float] = defaultdict(float)
        num_constr = 0
        robot: AbstractRobot
        # collect bodies from the same connection to the main body pair
        for robot in context.world.get_semantic_annotations_by_type(AbstractRobot):
            for body_a_original in robot.bodies_with_enabled_collision:
                for body_b_original in robot.bodies_with_enabled_collision:
                    if (
                        (
                            body_a_original,
                            body_b_original,
                        )
                        in context.world._collision_pair_manager.disabled_collision_pairs
                        or (
                            body_b_original,
                            body_a_original,
                        )
                        in context.world._collision_pair_manager.disabled_collision_pairs
                    ):
                        continue
                    body_a, body_b = (
                        context.world.compute_chain_reduced_to_controlled_connections(
                            body_a_original, body_b_original
                        )
                    )
                    if body_b.name < body_a.name:
                        body_a, body_b = body_b, body_a
                    counter[body_a, body_b] = max(
                        [
                            counter[body_a, body_b],
                            body_a_original.get_collision_config().buffer_zone_distance
                            or 0,
                            body_b_original.get_collision_config().buffer_zone_distance
                            or 0,
                        ]
                    )

        for link_a, link_b in counter:
            # num_of_constraints = min(1, counter[link_a, link_b])
            # for i in range(num_of_constraints):
            #     number_of_repeller = min(link_a.collision_config.max_avoided_bodies,
            #                              link_b.collision_config.max_avoided_bodies)
            ca_goal = SelfCollisionAvoidance(
                body_a=link_a,
                body_b=link_b,
                name=PrefixedName(f"{link_a.name}/{link_b.name}", str(self.name)),
                idx=0,
                max_avoided_bodies=1,
                buffer_zone_distance=counter[link_a, link_b],
            )
            ca_goal.plot_specs.visible = False
            self.add_node(ca_goal)
            num_constr += 1
        get_middleware().loginfo(
            f"Adding {num_constr} self collision avoidance constraints."
        )
