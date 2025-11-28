from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing_extensions import List, Optional, Union, TYPE_CHECKING, Set

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.data_types.exceptions import (
    DuplicateNameException,
)
from giskardpy.motion_statechart.data_types import LifeCycleValues, DefaultWeights
from giskardpy.motion_statechart.exceptions import GoalInitalizationException
from giskardpy.qp.constraint import (
    EqualityConstraint,
    InequalityConstraint,
    DerivativeInequalityConstraint,
    DerivativeEqualityConstraint,
    BaseConstraint,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.derivatives import Derivatives

if TYPE_CHECKING:
    from giskardpy.motion_statechart.graph_node import MotionStatechartNode

Large_Number = 1e4


@dataclass
class ConstraintCollection:
    constraints: List[BaseConstraint] = field(default_factory=list, init=False)

    @property
    def eq_constraints(self) -> List[EqualityConstraint]:
        return [c for c in self.constraints if isinstance(c, EqualityConstraint)]

    @property
    def neq_constraints(self) -> List[InequalityConstraint]:
        return [c for c in self.constraints if isinstance(c, InequalityConstraint)]

    @property
    def derivative_constraints(self) -> List[DerivativeInequalityConstraint]:
        return [
            c for c in self.constraints if isinstance(c, DerivativeInequalityConstraint)
        ]

    @property
    def eq_derivative_constraints(self) -> List[DerivativeEqualityConstraint]:
        return [
            c for c in self.constraints if isinstance(c, DerivativeEqualityConstraint)
        ]

    def merge(self, name_prefix: str, other: ConstraintCollection):
        for constraint in other.constraints:
            constraint.name = f"{name_prefix}/{constraint.name}"
        self.constraints.extend(other.constraints)
        self._are_names_unique()

    def _are_names_unique(self):
        names = set()
        for c in self.constraints:
            if c.name in names:
                raise DuplicateNameException(
                    f"Constraint named {c.name} already exists."
                )
            names.add(c.name)

    def get_all_float_variable_names(self) -> Set[PrefixedName]:
        return {v.name for c in self.constraints for v in c.expression.free_variables()}

    def link_to_motion_statechart_node(self, node: MotionStatechartNode):
        for constraint in self.constraints:
            is_running = cas.if_eq(
                node.life_cycle_variable,
                LifeCycleValues.RUNNING,
                if_result=cas.Expression(1),
                else_result=cas.Expression(0),
            )
            constraint.quadratic_weight *= is_running

    def add_equality_constraint(
        self,
        task_expression: cas.SymbolicScalar,
        equality_bound: cas.ScalarData,
        weight: cas.ScalarData,
        reference_velocity: cas.ScalarData,
        name: Optional[str] = None,
        lower_slack_limit: cas.ScalarData = -Large_Number,
        upper_slack_limit: cas.ScalarData = Large_Number,
    ):
        """
        Add a task constraint to the motion problem. This should be used for most constraints.
        It will not strictly stick to the reference velocity, but requires only a single constraint in the final
        optimization problem and is therefore faster.
        :param reference_velocity: used by Giskard to limit the error and normalize the weight, will not be strictly
                                    enforced.
        :param task_expression: defines the task function
        :param equality_bound: goal for the derivative of task_expression
        :param weight:
        :param name: give this constraint a name, required if you add more than one in the same goal
        :param lower_slack_limit: how much the lower error can be violated, don't use unless you know what you are doing
        :param upper_slack_limit: how much the upper error can be violated, don't use unless you know what you are doing
        """
        if task_expression.shape != (1, 1):
            raise GoalInitalizationException(
                f"expression must have shape (1, 1), has {task_expression.shape}"
            )
        name = name or f"{len(self.constraints)}"
        lower_slack_limit = (
            lower_slack_limit if lower_slack_limit is not None else -float("inf")
        )
        upper_slack_limit = (
            upper_slack_limit if upper_slack_limit is not None else float("inf")
        )
        constraint = EqualityConstraint(
            name=name,
            expression=task_expression,
            bound=equality_bound,
            normalization_factor=reference_velocity,
            quadratic_weight=weight,
            lower_slack_limit=lower_slack_limit,
            upper_slack_limit=upper_slack_limit,
            linear_weight=0,
        )
        if constraint.name in self.constraints:
            raise DuplicateNameException(
                f"Constraint named {constraint.name} already exists."
            )
        self.constraints.append(constraint)

    def add_inequality_constraint(
        self,
        reference_velocity: cas.ScalarData,
        lower_error: cas.ScalarData,
        upper_error: cas.ScalarData,
        weight: cas.ScalarData,
        task_expression: cas.SymbolicScalar,
        name: Optional[str] = None,
        lower_slack_limit: cas.ScalarData = -Large_Number,
        upper_slack_limit: cas.ScalarData = Large_Number,
    ):
        """
        Add a task constraint to the motion problem. This should be used for most constraints.
        It will not strictly stick to the reference velocity, but requires only a single constraint in the final
        optimization problem and is therefore faster.
        :param reference_velocity: used by Giskard to limit the error and normalize the weight, will not be strictly
                                    enforced.
        :param lower_error: lower bound for the error of expression
        :param upper_error: upper bound for the error of expression
        :param weight:
        :param task_expression: defines the task function
        :param name: give this constraint a name, required if you add more than one in the same goal
        :param lower_slack_limit: how much the lower error can be violated, don't use unless you know what you are doing
        :param upper_slack_limit: how much the upper error can be violated, don't use unless you know what you are doing
        """
        if task_expression.shape != (1, 1):
            raise GoalInitalizationException(
                f"expression must have shape (1,1), has {task_expression.shape}"
            )
        name = name or ""
        lower_slack_limit = (
            lower_slack_limit if lower_slack_limit is not None else -float("inf")
        )
        upper_slack_limit = (
            upper_slack_limit if upper_slack_limit is not None else float("inf")
        )
        constraint = InequalityConstraint(
            name=name,
            expression=task_expression,
            lower_error=lower_error,
            upper_error=upper_error,
            normalization_factor=reference_velocity,
            quadratic_weight=weight,
            lower_slack_limit=lower_slack_limit,
            upper_slack_limit=upper_slack_limit,
            linear_weight=0,
        )
        if name in self.constraints:
            raise DuplicateNameException(
                f"A constraint with name '{name}' already exists. "
                f"You need to set a name, if you add multiple constraints."
            )
        self.constraints.append(constraint)

    def add_point_goal_constraints(
        self,
        frame_P_current: cas.Point3,
        frame_P_goal: cas.Point3,
        reference_velocity: cas.ScalarData,
        weight: cas.ScalarData,
        name: Optional[str] = None,
    ):
        """
        Adds three constraints to move frame_P_current to frame_P_goal.
        Make sure that both points are expressed relative to the same frame!
        :param frame_P_current: a vector describing a 3D point
        :param frame_P_goal: a vector describing a 3D point
        :param reference_velocity: m/s
        :param weight:
        :param name:
        """
        frame_V_error = frame_P_goal - frame_P_current
        for i in range(3):
            self.add_equality_constraint(
                task_expression=frame_P_current[i],
                equality_bound=frame_V_error[i],
                weight=weight,
                reference_velocity=reference_velocity,
                name=f"{name}/{i}",
            )

    def add_position_constraint(
        self,
        expr_current: cas.SymbolicScalar,
        expr_goal: cas.ScalarData,
        reference_velocity: cas.ScalarData,
        weight: cas.ScalarData = DefaultWeights.WEIGHT_BELOW_CA,
        name: Optional[str] = None,
    ):
        """
        A wrapper around add_constraint. Will add a constraint that tries to move expr_current to expr_goal.
        """
        error = expr_goal - expr_current
        self.add_equality_constraint(
            reference_velocity=reference_velocity,
            equality_bound=error,
            weight=weight,
            task_expression=expr_current,
            name=name,
        )

    def add_position_range_constraint(
        self,
        expr_current: cas.SymbolicScalar,
        expr_min: cas.ScalarData,
        expr_max: cas.ScalarData,
        reference_velocity: cas.ScalarData,
        weight: cas.ScalarData = DefaultWeights.WEIGHT_BELOW_CA,
        name: Optional[str] = None,
    ):
        """
        A wrapper around add_constraint. Will add a constraint that tries to move expr_current to expr_goal.
        """
        error_min = expr_min - expr_current
        error_max = expr_max - expr_current
        self.add_inequality_constraint(
            reference_velocity=reference_velocity,
            lower_error=error_min,
            upper_error=error_max,
            weight=weight,
            task_expression=expr_current,
            name=name,
        )

    def add_vector_goal_constraints(
        self,
        frame_V_current: cas.Vector3,
        frame_V_goal: cas.Vector3,
        reference_velocity: cas.ScalarData,
        weight: cas.ScalarData = DefaultWeights.WEIGHT_BELOW_CA,
        name: Optional[str] = None,
    ):
        """
        Adds constraints to align frame_V_current with frame_V_goal. Make sure that both vectors are expressed
        relative to the same frame and are normalized to a length of 1.
        :param frame_V_current: a vector describing a 3D vector
        :param frame_V_goal: a vector describing a 3D vector
        :param reference_velocity: rad/s
        :param weight:
        :param name:
        """
        angle = cas.safe_acos(frame_V_current.dot(frame_V_goal))
        # avoid singularity by staying away from pi
        angle_limited = cas.min(cas.max(angle, -reference_velocity), reference_velocity)
        angle_limited = angle_limited.safe_division(angle)
        root_V_goal_normal_intermediate = frame_V_current.slerp(
            frame_V_goal, angle_limited
        )

        error = root_V_goal_normal_intermediate - frame_V_current
        for i in range(3):
            self.add_equality_constraint(
                task_expression=frame_V_current[i],
                equality_bound=error[i],
                reference_velocity=reference_velocity,
                weight=weight,
                name=f"{name}/{i}",
            )

    def add_rotation_goal_constraints(
        self,
        frame_R_current: cas.RotationMatrix,
        frame_R_goal: cas.RotationMatrix,
        reference_velocity: cas.ScalarData,
        weight: cas.ScalarData,
        name: Optional[str] = None,
    ):
        """
        Adds constraints to move frame_R_current to frame_R_goal. Make sure that both are expressed relative to the same
        frame.
        :param frame_R_current: current rotation as rotation matrix
        :param frame_R_goal: goal rotation as rotation matrix
        :param reference_velocity: rad/s
        :param weight:
        :param name:
        """
        # avoid singularity
        # the sign determines in which direction the robot moves when in singularity.
        # -0.0001 preserves the old behavior from before this goal was refactored
        hack = cas.RotationMatrix.from_axis_angle(cas.Vector3.Z(), -0.0001)
        frame_R_current = frame_R_current.dot(hack)
        q_actual = frame_R_current.to_quaternion()
        q_goal = frame_R_goal.to_quaternion()
        q_goal = cas.if_less(q_goal.dot(q_actual), 0, -q_goal, q_goal)
        q_error = q_actual.diff(q_goal)

        # w is redundant
        for i in range(3):
            self.add_equality_constraint(
                task_expression=q_error[i],
                equality_bound=-q_error[i],
                weight=weight,
                reference_velocity=reference_velocity,
                name=f"{name}/{i}",
            )

    def add_velocity_constraint(
        self,
        lower_velocity_limit: Union[cas.ScalarData, List[cas.ScalarData]],
        upper_velocity_limit: Union[cas.ScalarData, List[cas.ScalarData]],
        weight: cas.ScalarData,
        task_expression: cas.SymbolicScalar,
        velocity_limit: cas.ScalarData,
        name: Optional[str] = None,
        lower_slack_limit: Union[cas.ScalarData, List[cas.ScalarData]] = -Large_Number,
        upper_slack_limit: Union[cas.ScalarData, List[cas.ScalarData]] = Large_Number,
    ):
        """
        Add a velocity constraint. Internally, this will be converted into multiple constraints, to ensure that the
        velocity stays within the given bounds.
        :param lower_velocity_limit:
        :param upper_velocity_limit:
        :param weight:
        :param task_expression:
        :param velocity_limit: Used for normalizing the expression, like reference_velocity, must be positive
        :param name:
        :param lower_slack_limit:
        :param upper_slack_limit:
        """
        name = name or ""
        constraint = DerivativeInequalityConstraint(
            name=name,
            derivative=Derivatives.velocity,
            expression=task_expression,
            lower_limit=lower_velocity_limit,
            upper_limit=upper_velocity_limit,
            quadratic_weight=weight,
            normalization_factor=velocity_limit,
            lower_slack_limit=lower_slack_limit,
            upper_slack_limit=upper_slack_limit,
            linear_weight=0,
        )
        if constraint.name in self.constraints:
            raise KeyError(f"a constraint with name '{name}' already exists")
        self.constraints.append(constraint)

    def add_velocity_eq_constraint(
        self,
        velocity_goal: Union[cas.ScalarData, List[cas.ScalarData]],
        weight: cas.ScalarData,
        task_expression: cas.SymbolicScalar,
        velocity_limit: cas.ScalarData,
        name: Optional[str] = None,
        lower_slack_limit: Union[cas.ScalarData, List[cas.ScalarData]] = -Large_Number,
        upper_slack_limit: Union[cas.ScalarData, List[cas.ScalarData]] = Large_Number,
    ):
        """
        Add a velocity constraint. Internally, this will be converted into multiple constraints, to ensure that the
        velocity stays within the given bounds.
        :param velocity_goal:
        :param weight:
        :param task_expression:
        :param velocity_limit: Used for normalizing the expression, like reference_velocity, must be positive
        :param name:
        :param lower_slack_limit:
        :param upper_slack_limit:
        """
        name = name or ""
        constraint = DerivativeEqualityConstraint(
            name=name,
            derivative=Derivatives.velocity,
            expression=task_expression,
            bound=velocity_goal,
            quadratic_weight=weight,
            normalization_factor=velocity_limit,
            lower_slack_limit=lower_slack_limit,
            upper_slack_limit=upper_slack_limit,
            linear_weight=0,
        )
        if constraint.name in self.constraints:
            raise KeyError(f"a constraint with name '{name}' already exists")
        self.constraints.append(constraint)

    def add_velocity_eq_constraint_vector(
        self,
        velocity_goals: Union[
            cas.Expression, cas.Vector3, cas.Point3, List[cas.ScalarData]
        ],
        reference_velocities: Union[
            cas.Expression, cas.Vector3, cas.Point3, List[cas.ScalarData]
        ],
        weights: Union[cas.Expression, cas.Vector3, cas.Point3, List[cas.ScalarData]],
        task_expression: Union[
            cas.Expression, cas.Vector3, cas.Point3, List[cas.SymbolicScalar]
        ],
        names: List[str],
    ):
        for i in range(len(velocity_goals)):
            name_suffix = names[i] if names else None
            self.add_velocity_eq_constraint(
                velocity_goal=velocity_goals[i],
                weight=weights[i],
                velocity_limit=reference_velocities[i],
                task_expression=task_expression[i],
                name=name_suffix,
                lower_slack_limit=-np.inf,
                upper_slack_limit=np.inf,
            )

    def add_translational_velocity_limit(
        self,
        frame_P_current: cas.Point3,
        max_velocity: cas.ScalarData,
        weight: cas.ScalarData,
        max_violation: cas.ScalarData = np.inf,
        name: Optional[str] = None,
    ):
        """
        Adds constraints to limit the translational velocity of frame_P_current. Be aware that the velocity is relative
        to frame.
        :param frame_P_current: a vector describing a 3D point
        :param max_velocity:
        :param weight:
        :param max_violation: m/s
        :param name:
        """
        trans_error = frame_P_current.norm()
        trans_error = cas.if_eq_zero(trans_error, cas.Expression(0.01), trans_error)
        self.add_velocity_constraint(
            upper_velocity_limit=max_velocity,
            lower_velocity_limit=-max_velocity,
            weight=weight,
            task_expression=trans_error,
            lower_slack_limit=-max_violation,
            upper_slack_limit=max_violation,
            velocity_limit=max_velocity,
            name=name,
        )

    def add_rotational_velocity_limit(
        self,
        frame_R_current: cas.RotationMatrix,
        max_velocity: cas.ScalarData,
        weight: cas.ScalarData,
        max_violation: cas.ScalarData = Large_Number,
        name: Optional[str] = None,
    ):
        """
        Add velocity constraints to limit the velocity of frame_R_current. Be aware that the velocity is relative to
        frame.
        :param frame_R_current: Rotation matrix describing the current rotation.
        :param max_velocity: rad/s
        :param weight:
        :param max_violation:
        :param name:
        """
        root_Q_tipCurrent = frame_R_current.to_quaternion()
        angle_error = root_Q_tipCurrent.to_axis_angle()[1]
        self.add_velocity_constraint(
            upper_velocity_limit=max_velocity,
            lower_velocity_limit=-max_velocity,
            weight=weight,
            task_expression=angle_error,
            lower_slack_limit=-max_violation,
            upper_slack_limit=max_violation,
            name=name,
            velocity_limit=max_velocity,
        )
