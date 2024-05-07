from enum import IntEnum
from typing import Optional, List, Union, Dict, Callable, Iterable, overload, DefaultDict

import numpy as np

import giskard_msgs.msg as giskard_msgs
import giskardpy.casadi_wrapper as cas
from giskardpy.exceptions import GiskardException, GoalInitalizationException, DuplicateNameException
from giskardpy.god_map import god_map
from giskardpy.monitors.monitors import ExpressionMonitor, Monitor
from giskardpy.data_types import Derivatives, PrefixName, TaskState
from giskardpy.qp.constraint import EqualityConstraint, InequalityConstraint, DerivativeInequalityConstraint, Constraint
from giskardpy.symbol_manager import symbol_manager
from giskardpy.utils.decorators import memoize
from giskardpy.utils.utils import string_shortener
from giskardpy.qp.weight_gain import QuadraticWeightGain, LinearWeightGain
from giskardpy.qp.free_variable import FreeVariable

WEIGHT_MAX = giskard_msgs.Weights.WEIGHT_MAX
WEIGHT_ABOVE_CA = giskard_msgs.Weights.WEIGHT_ABOVE_CA
WEIGHT_COLLISION_AVOIDANCE = giskard_msgs.Weights.WEIGHT_COLLISION_AVOIDANCE
WEIGHT_BELOW_CA = giskard_msgs.Weights.WEIGHT_BELOW_CA
WEIGHT_MIN = giskard_msgs.Weights.WEIGHT_MIN


class Task:
    """
    Tasks are a set of constraints with the same predicates.
    """
    eq_constraints: Dict[PrefixName, EqualityConstraint]
    neq_constraints: Dict[PrefixName, InequalityConstraint]
    derivative_constraints: Dict[PrefixName, DerivativeInequalityConstraint]
    _start_condition: cas.Expression
    _hold_condition: cas.Expression
    _end_condition: cas.Expression
    _name: str
    _parent_goal_name: str
    _id: int
    plot: bool

    def __init__(self, parent_goal_name: str, name: Optional[str] = None):
        self.plot = True
        if name is None:
            self._name = str(self.__class__.__name__)
        else:
            self._name = name
        self._parent_goal_name = parent_goal_name
        self.eq_constraints = {}
        self.neq_constraints = {}
        self.derivative_constraints = {}
        self._start_condition = cas.TrueSymbol
        self._hold_condition = cas.FalseSymbol
        self._end_condition = cas.FalseSymbol
        self.manip_constraints = {}
        self.quadratic_gains = []
        self.linear_weight_gains = []
        self._id = -1

    def to_ros_msg(self) -> giskard_msgs.MotionGoal:
        msg = giskard_msgs.MotionGoal()
        msg.name = str(self.name)
        msg.motion_goal_class = self.__class__.__name__
        msg.start_condition = god_map.monitor_manager.format_condition(self.start_condition, new_line=' ')
        msg.hold_condition = god_map.monitor_manager.format_condition(self.hold_condition, new_line=' ')
        msg.end_condition = god_map.monitor_manager.format_condition(self.end_condition, new_line=' ')
        return msg

    @property
    def start_condition(self) -> cas.Expression:
        return self._start_condition

    @start_condition.setter
    def start_condition(self, value: cas.Expression) -> None:
        for monitor_state_expr in value.free_symbols():
            if not god_map.monitor_manager.is_monitor_registered(monitor_state_expr):
                raise GiskardException(f'No monitor found for this state expr: "{monitor_state_expr}".')
        self._start_condition = value

    @property
    def hold_condition(self) -> cas.Expression:
        return self._hold_condition

    @hold_condition.setter
    def hold_condition(self, value: cas.Expression) -> None:
        for monitor_state_expr in value.free_symbols():
            if not god_map.monitor_manager.is_monitor_registered(monitor_state_expr):
                raise GiskardException(f'No monitor found for this state expr: "{monitor_state_expr}".')
        self._hold_condition = value

    @property
    def end_condition(self) -> cas.Expression:
        return self._end_condition

    @end_condition.setter
    def end_condition(self, value: cas.Expression) -> None:
        for monitor_state_expr in value.free_symbols():
            if not god_map.monitor_manager.is_monitor_registered(monitor_state_expr):
                raise GiskardException(f'No monitor found for this state expr: "{monitor_state_expr}".')
        self._end_condition = value

    @property
    def id(self) -> int:
        assert self._id >= 0, f'id of {self.name} is not set.'
        return self._id

    def set_id(self, new_id: int) -> None:
        self._id = new_id

    @property
    def name(self) -> PrefixName:
        return PrefixName(self._name, self._parent_goal_name)

    def __str__(self):
        return self.name

    def formatted_name(self, quoted: bool = False) -> str:
        formatted_name = string_shortener(original_str=str(self.name),
                                          max_lines=4,
                                          max_line_length=25)
        result = (f'{formatted_name}\n'
                  f'----start_condition----\n'
                  f'{god_map.monitor_manager.format_condition(self.start_condition)}\n'
                  f'----hold_condition----\n'
                  f'{god_map.monitor_manager.format_condition(self.hold_condition)}\n'
                  f'----end_condition----\n'
                  f'{god_map.monitor_manager.format_condition(self.end_condition)}')
        if quoted:
            return '"' + result + '"'
        return result

    def get_eq_constraints(self) -> List[EqualityConstraint]:
        return self._apply_monitors_to_constraints(self.eq_constraints.values())

    def get_neq_constraints(self) -> List[InequalityConstraint]:
        return self._apply_monitors_to_constraints(self.neq_constraints.values())

    def get_derivative_constraints(self) -> List[DerivativeInequalityConstraint]:
        return self._apply_monitors_to_constraints(self.derivative_constraints.values())

    def get_quadratic_gains(self) -> List[QuadraticWeightGain]:
        return self.quadratic_gains

    def get_linear_gains(self) -> List[LinearWeightGain]:
        return self.linear_weight_gains

    def get_state_expression(self) -> cas.Symbol:
        return symbol_manager.get_symbol(f'god_map.motion_goal_manager.task_state[{self.id}]')

    @overload
    def _apply_monitors_to_constraints(self, constraints: Iterable[EqualityConstraint]) \
            -> List[Union[EqualityConstraint]]:
        ...

    @overload
    def _apply_monitors_to_constraints(self, constraints: Iterable[InequalityConstraint]) \
            -> List[Union[InequalityConstraint]]:
        ...

    @overload
    def _apply_monitors_to_constraints(self, constraints: Iterable[DerivativeInequalityConstraint]) \
            -> List[Union[DerivativeInequalityConstraint]]:
        ...

    def _apply_monitors_to_constraints(self, constraints):
        output_constraints = []
        for constraint in constraints:
            is_running = cas.if_eq(self.get_state_expression(), int(TaskState.running),
                                   if_result=1,
                                   else_result=0)
            constraint.quadratic_weight *= is_running
            output_constraints.append(constraint)
        return output_constraints

    def add_quadratic_weight_gain(self, name: str, gains: List[DefaultDict[Derivatives, Dict[FreeVariable, float]]]):
        q_gain = QuadraticWeightGain(name=name,
                                     gains=gains)
        self.quadratic_gains.append(q_gain)

    def add_linear_weight_gain(self, name: str, gains: List[DefaultDict[Derivatives, Dict[FreeVariable, float]]]):
        q_gain = LinearWeightGain(name=name,
                                  gains=gains)
        self.linear_weight_gains.append(q_gain)

    def add_equality_constraint(self,
                                reference_velocity: cas.symbol_expr_float,
                                equality_bound: cas.symbol_expr_float,
                                weight: cas.symbol_expr_float,
                                task_expression: cas.symbol_expr,
                                name: str = None,
                                lower_slack_limit: Optional[cas.symbol_expr_float] = None,
                                upper_slack_limit: Optional[cas.symbol_expr_float] = None,
                                control_horizon: Optional[int] = None):
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
            raise GoalInitalizationException(f'expression must have shape (1, 1), has {task_expression.shape}')
        name = name or f'{len(self.eq_constraints)}'
        lower_slack_limit = lower_slack_limit if lower_slack_limit is not None else -float('inf')
        upper_slack_limit = upper_slack_limit if upper_slack_limit is not None else float('inf')
        constraint = EqualityConstraint(name=name,
                                        parent_task_name=self.name,
                                        expression=task_expression,
                                        derivative_goal=equality_bound,
                                        velocity_limit=reference_velocity,
                                        quadratic_weight=weight,
                                        lower_slack_limit=lower_slack_limit,
                                        upper_slack_limit=upper_slack_limit,
                                        control_horizon=control_horizon)
        if constraint.name in self.eq_constraints:
            raise DuplicateNameException(f'Constraint named {constraint.name} already exists.')
        self.eq_constraints[constraint.name] = constraint

    def add_inequality_constraint(self,
                                  reference_velocity: cas.symbol_expr_float,
                                  lower_error: cas.symbol_expr_float,
                                  upper_error: cas.symbol_expr_float,
                                  weight: cas.symbol_expr_float,
                                  task_expression: cas.symbol_expr,
                                  name: Optional[str] = None,
                                  lower_slack_limit: Optional[cas.symbol_expr_float] = None,
                                  upper_slack_limit: Optional[cas.symbol_expr_float] = None,
                                  control_horizon: Optional[int] = None):
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
            raise GoalInitalizationException(f'expression must have shape (1,1), has {task_expression.shape}')
        name = name or ''
        lower_slack_limit = lower_slack_limit if lower_slack_limit is not None else -float('inf')
        upper_slack_limit = upper_slack_limit if upper_slack_limit is not None else float('inf')
        constraint = InequalityConstraint(name=name,
                                          parent_task_name=self.name,
                                          expression=task_expression,
                                          lower_error=lower_error,
                                          upper_error=upper_error,
                                          velocity_limit=reference_velocity,
                                          quadratic_weight=weight,
                                          lower_slack_limit=lower_slack_limit,
                                          upper_slack_limit=upper_slack_limit,
                                          control_horizon=control_horizon)
        if name in self.neq_constraints:
            raise DuplicateNameException(f'A constraint with name \'{name}\' already exists. '
                                         f'You need to set a name, if you add multiple constraints.')
        self.neq_constraints[constraint.name] = constraint

    def add_inequality_constraint_vector(self,
                                         reference_velocities: Union[
                                             cas.Expression, cas.Vector3, cas.Point3, List[cas.symbol_expr_float]],
                                         lower_errors: Union[
                                             cas.Expression, cas.Vector3, cas.Point3, List[cas.symbol_expr_float]],
                                         upper_errors: Union[
                                             cas.Expression, cas.Vector3, cas.Point3, List[cas.symbol_expr_float]],
                                         weights: Union[
                                             cas.Expression, cas.Vector3, cas.Point3, List[cas.symbol_expr_float]],
                                         task_expression: Union[
                                             cas.Expression, cas.Vector3, cas.Point3, List[cas.symbol_expr]],
                                         names: List[str],
                                         lower_slack_limits: Optional[List[cas.symbol_expr_float]] = None,
                                         upper_slack_limits: Optional[List[cas.symbol_expr_float]] = None):
        """
        Calls add_constraint for a list of expressions.
        """
        if len(lower_errors) != len(upper_errors) \
                or len(lower_errors) != len(task_expression) \
                or len(lower_errors) != len(reference_velocities) \
                or len(lower_errors) != len(weights) \
                or (names is not None and len(lower_errors) != len(names)) \
                or (lower_slack_limits is not None and len(lower_errors) != len(lower_slack_limits)) \
                or (upper_slack_limits is not None and len(lower_errors) != len(upper_slack_limits)):
            raise GoalInitalizationException('All parameters must have the same length.')
        for i in range(len(lower_errors)):
            name_suffix = names[i] if names else None
            lower_slack_limit = lower_slack_limits[i] if lower_slack_limits else None
            upper_slack_limit = upper_slack_limits[i] if upper_slack_limits else None
            self.add_inequality_constraint(reference_velocity=reference_velocities[i],
                                           lower_error=lower_errors[i],
                                           upper_error=upper_errors[i],
                                           weight=weights[i],
                                           task_expression=task_expression[i],
                                           name=name_suffix,
                                           lower_slack_limit=lower_slack_limit,
                                           upper_slack_limit=upper_slack_limit)

    def add_equality_constraint_vector(self,
                                       reference_velocities: Union[
                                           cas.Expression, cas.Vector3, cas.Point3, List[cas.symbol_expr_float]],
                                       equality_bounds: Union[
                                           cas.Expression, cas.Vector3, cas.Point3, List[cas.symbol_expr_float]],
                                       weights: Union[
                                           cas.Expression, cas.Vector3, cas.Point3, List[cas.symbol_expr_float]],
                                       task_expression: Union[
                                           cas.Expression, cas.Vector3, cas.Point3, List[cas.symbol_expr]],
                                       names: List[str],
                                       lower_slack_limits: Optional[List[cas.symbol_expr_float]] = None,
                                       upper_slack_limits: Optional[List[cas.symbol_expr_float]] = None):
        """
        Calls add_constraint for a list of expressions.
        """
        for i in range(len(equality_bounds)):
            name_suffix = names[i] if names else None
            lower_slack_limit = lower_slack_limits[i] if lower_slack_limits else None
            upper_slack_limit = upper_slack_limits[i] if upper_slack_limits else None
            self.add_equality_constraint(reference_velocity=reference_velocities[i],
                                         equality_bound=equality_bounds[i],
                                         weight=weights[i],
                                         task_expression=task_expression[i],
                                         name=name_suffix,
                                         lower_slack_limit=lower_slack_limit,
                                         upper_slack_limit=upper_slack_limit)

    def add_point_goal_constraints(self,
                                   frame_P_current: cas.Point3,
                                   frame_P_goal: cas.Point3,
                                   reference_velocity: cas.symbol_expr_float,
                                   weight: cas.symbol_expr_float,
                                   name: str = ''):
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
        self.add_equality_constraint_vector(reference_velocities=[reference_velocity] * 3,
                                            equality_bounds=frame_V_error[:3],
                                            weights=[weight] * 3,
                                            task_expression=frame_P_current[:3],
                                            names=[f'{name}/x',
                                                   f'{name}/y',
                                                   f'{name}/z'])

    def add_position_constraint(self,
                                expr_current: Union[cas.symbol_expr, float],
                                expr_goal: Union[cas.symbol_expr_float, float],
                                reference_velocity: Union[cas.symbol_expr_float, float],
                                weight: Union[cas.symbol_expr_float, float] = WEIGHT_BELOW_CA,
                                name: str = ''):
        """
        A wrapper around add_constraint. Will add a constraint that tries to move expr_current to expr_goal.
        """
        error = expr_goal - expr_current
        self.add_equality_constraint(reference_velocity=reference_velocity,
                                     equality_bound=error,
                                     weight=weight,
                                     task_expression=expr_current,
                                     name=name)

    def add_vector_goal_constraints(self,
                                    frame_V_current: cas.Vector3,
                                    frame_V_goal: cas.Vector3,
                                    reference_velocity: cas.symbol_expr_float,
                                    weight: cas.symbol_expr_float = WEIGHT_BELOW_CA,
                                    name: str = ''):
        """
        Adds constraints to align frame_V_current with frame_V_goal. Make sure that both vectors are expressed
        relative to the same frame and are normalized to a length of 1.
        :param frame_V_current: a vector describing a 3D vector
        :param frame_V_goal: a vector describing a 3D vector
        :param reference_velocity: rad/s
        :param weight:
        :param name:
        """
        angle = cas.save_acos(frame_V_current.dot(frame_V_goal))
        # avoid singularity by staying away from pi
        angle_limited = cas.min(cas.max(angle, -reference_velocity), reference_velocity)
        angle_limited = cas.save_division(angle_limited, angle)
        root_V_goal_normal_intermediate = cas.slerp(frame_V_current, frame_V_goal, angle_limited)

        error = root_V_goal_normal_intermediate - frame_V_current

        self.add_equality_constraint_vector(reference_velocities=[reference_velocity] * 3,
                                            equality_bounds=error[:3],
                                            weights=[weight] * 3,
                                            task_expression=frame_V_current[:3],
                                            names=[f'{name}/trans/x',
                                                   f'{name}/trans/y',
                                                   f'{name}/trans/z'])

    def add_rotation_goal_constraints(self,
                                      frame_R_current: cas.RotationMatrix,
                                      frame_R_goal: cas.RotationMatrix,
                                      current_R_frame_eval: cas.RotationMatrix,
                                      reference_velocity: Union[cas.Symbol, float],
                                      weight: Union[cas.Symbol, float],
                                      name: str = ''):
        """
        Adds constraints to move frame_R_current to frame_R_goal. Make sure that both are expressed relative to the same
        frame.
        :param frame_R_current: current rotation as rotation matrix
        :param frame_R_goal: goal rotation as rotation matrix
        :param current_R_frame_eval: an expression that computes the reverse of frame_R_current.
                                        Use self.get_fk_evaluated for this.
        :param reference_velocity: rad/s
        :param weight:
        :param name:
        """
        hack = cas.RotationMatrix.from_axis_angle(cas.Vector3((0, 0, 1)), 0.0001)
        frame_R_current = frame_R_current.dot(hack)  # hack to avoid singularity
        tip_Q_tipCurrent = current_R_frame_eval.dot(frame_R_current).to_quaternion()
        tip_R_goal = current_R_frame_eval.dot(frame_R_goal)

        tip_Q_goal = tip_R_goal.to_quaternion()

        tip_Q_goal = cas.if_greater_zero(-tip_Q_goal[3], -tip_Q_goal, tip_Q_goal)  # flip to get shortest path

        expr = tip_Q_tipCurrent
        # w is not needed because its derivative is always 0 for identity quaternions
        self.add_equality_constraint_vector(reference_velocities=[reference_velocity] * 3,
                                            equality_bounds=tip_Q_goal[:3],
                                            weights=[weight] * 3,
                                            task_expression=expr[:3],
                                            names=[f'{name}/rot/x',
                                                   f'{name}/rot/y',
                                                   f'{name}/rot/z'])

    def add_velocity_constraint(self,
                                lower_velocity_limit: Union[cas.symbol_expr_float, List[cas.symbol_expr_float]],
                                upper_velocity_limit: Union[cas.symbol_expr_float, List[cas.symbol_expr_float]],
                                weight: cas.symbol_expr_float,
                                task_expression: cas.symbol_expr,
                                velocity_limit: cas.symbol_expr_float,
                                name: Optional[str] = None,
                                control_horizon: Optional[cas.symbol_expr_float] = None,
                                lower_slack_limit: Union[cas.symbol_expr_float, List[cas.symbol_expr_float]] = -1e4,
                                upper_slack_limit: Union[cas.symbol_expr_float, List[cas.symbol_expr_float]] = 1e4,
                                horizon_function: Optional[Callable[[float, int], float]] = None):
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
        :param horizon_function: A function that can takes 'weight' and the id within the horizon as input and computes
                                    a new weight. Can be used to give points towards the end of the horizon a different
                                    weight
        """
        name = name or ''
        constraint = DerivativeInequalityConstraint(name=name,
                                                    parent_task_name=self.name,
                                                    derivative=Derivatives.velocity,
                                                    expression=task_expression,
                                                    lower_limit=lower_velocity_limit,
                                                    upper_limit=upper_velocity_limit,
                                                    quadratic_weight=weight,
                                                    normalization_factor=velocity_limit,
                                                    lower_slack_limit=lower_slack_limit,
                                                    upper_slack_limit=upper_slack_limit,
                                                    control_horizon=control_horizon,
                                                    horizon_function=horizon_function)
        if constraint.name in self.derivative_constraints:
            raise KeyError(f'a constraint with name \'{name}\' already exists')
        self.derivative_constraints[constraint.name] = constraint

    def add_acceleration_constraint(self,
                                    lower_acceleration_limit: Union[cas.symbol_expr_float, List[cas.symbol_expr_float]],
                                    upper_acceleration_limit: Union[cas.symbol_expr_float, List[cas.symbol_expr_float]],
                                    weight: cas.symbol_expr_float,
                                    task_expression: cas.symbol_expr,
                                    acceleration_limit: cas.symbol_expr_float,
                                    name: Optional[str] = None,
                                    lower_slack_limit: Union[cas.symbol_expr_float, List[cas.symbol_expr_float]] = -1e4,
                                    upper_slack_limit: Union[cas.symbol_expr_float, List[cas.symbol_expr_float]] = 1e4,
                                    horizon_function: Optional[Callable[[float, int], float]] = None):
        """
        Add a acceleration constraint. Internally, this will be converted into multiple constraints, to ensure that the
        acceleration stays within the given bounds.
        :param lower_acceleration_limit:
        :param upper_acceleration_limit:
        :param weight:
        :param task_expression:
        :param acceleration_limit:
        :param name:
        :param lower_slack_limit:
        :param upper_slack_limit:
        :param horizon_function: A function that can takes 'weight' and the id within the horizon as input and computes
                                    a new weight. Can be used to give points towards the end of the horizon a different
                                    weight
        """
        name = name if name else ''
        constraint = DerivativeInequalityConstraint(name=name,
                                                    parent_task_name=self.name,
                                                    derivative=Derivatives.acceleration,
                                                    expression=task_expression,
                                                    lower_limit=lower_acceleration_limit,
                                                    upper_limit=upper_acceleration_limit,
                                                    quadratic_weight=weight,
                                                    normalization_factor=acceleration_limit,
                                                    lower_slack_limit=lower_slack_limit,
                                                    upper_slack_limit=upper_slack_limit,
                                                    horizon_function=horizon_function)
        if name in self.derivative_constraints:
            raise DuplicateNameException(f'a constraint with name \'{name}\' already exists')
        self.derivative_constraints[constraint.name] = constraint

    def add_jerk_constraint(self,
                            lower_jerk_limit: Union[cas.symbol_expr_float, List[cas.symbol_expr_float]],
                            upper_jerk_limit: Union[cas.symbol_expr_float, List[cas.symbol_expr_float]],
                            weight: cas.symbol_expr_float,
                            task_expression: cas.symbol_expr,
                            acceleration_limit: cas.symbol_expr_float,
                            name: Optional[str] = None,
                            lower_slack_limit: Union[cas.symbol_expr_float, List[cas.symbol_expr_float]] = -1e4,
                            upper_slack_limit: Union[cas.symbol_expr_float, List[cas.symbol_expr_float]] = 1e4,
                            horizon_function: Optional[Callable[[float, int], float]] = None):
        name = name if name else ''
        constraint = DerivativeInequalityConstraint(name=name,
                                                    parent_task_name=self.name,
                                                    derivative=Derivatives.jerk,
                                                    expression=task_expression,
                                                    lower_limit=lower_jerk_limit,
                                                    upper_limit=upper_jerk_limit,
                                                    quadratic_weight=weight,
                                                    normalization_factor=acceleration_limit,
                                                    lower_slack_limit=lower_slack_limit,
                                                    upper_slack_limit=upper_slack_limit,
                                                    horizon_function=horizon_function)
        if name in self.derivative_constraints:
            raise KeyError(f'a constraint with name \'{name}\' already exists')
        self.derivative_constraints[constraint.name] = constraint

    def add_translational_velocity_limit(self,
                                         frame_P_current: cas.Point3,
                                         max_velocity: cas.symbol_expr_float,
                                         weight: cas.symbol_expr_float,
                                         max_violation: cas.symbol_expr_float = 1e4,
                                         name=''):
        """
        Adds constraints to limit the translational velocity of frame_P_current. Be aware that the velocity is relative
        to frame.
        :param frame_P_current: a vector describing a 3D point
        :param max_velocity:
        :param weight:
        :param max_violation: m/s
        :param name:
        """
        trans_error = cas.norm(frame_P_current)
        trans_error = cas.if_eq_zero(trans_error, 0.01, trans_error)
        god_map.debug_expression_manager.add_debug_expression('trans_error', trans_error)
        self.add_velocity_constraint(upper_velocity_limit=max_velocity,
                                     lower_velocity_limit=-max_velocity,
                                     weight=weight,
                                     task_expression=trans_error,
                                     lower_slack_limit=-max_violation,
                                     upper_slack_limit=max_violation,
                                     velocity_limit=max_velocity,
                                     name=f'{name}/vel')

    def add_rotational_velocity_limit(self,
                                      frame_R_current: cas.RotationMatrix,
                                      max_velocity: Union[cas.Symbol, float],
                                      weight: Union[cas.Symbol, float],
                                      max_violation: Union[cas.Symbol, float] = 1e4,
                                      name: str = ''):
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
        self.add_velocity_constraint(upper_velocity_limit=max_velocity,
                                     lower_velocity_limit=-max_velocity,
                                     weight=weight,
                                     task_expression=angle_error,
                                     lower_slack_limit=-max_violation,
                                     upper_slack_limit=max_violation,
                                     name=f'{name}/q/vel',
                                     velocity_limit=max_velocity)
