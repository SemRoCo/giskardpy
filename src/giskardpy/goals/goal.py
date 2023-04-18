from __future__ import annotations

import abc
from abc import ABC
from collections import OrderedDict
from typing import Optional, Tuple, Dict, List, Union, Callable

import giskardpy.identifier as identifier
import giskardpy.utils.tfwrapper as tf
from giskard_msgs.msg import Constraint as Constraint_msg
from giskardpy import casadi_wrapper as w
from giskardpy.casadi_wrapper import symbol_expr_float
from giskardpy.exceptions import ConstraintInitalizationException, GiskardException, UnknownGroupException
from giskardpy.god_map import GodMap
from giskardpy.model.joints import OneDofJoint
from giskardpy.model.world import WorldTree
from giskardpy.my_types import my_string, transformable_message, PrefixName, Derivatives
from giskardpy.qp.constraint import InequalityConstraint, EqualityConstraint, DerivativeInequalityConstraint

WEIGHT_MAX = Constraint_msg.WEIGHT_MAX
WEIGHT_ABOVE_CA = Constraint_msg.WEIGHT_ABOVE_CA
WEIGHT_COLLISION_AVOIDANCE = Constraint_msg.WEIGHT_COLLISION_AVOIDANCE
WEIGHT_BELOW_CA = Constraint_msg.WEIGHT_BELOW_CA
WEIGHT_MIN = Constraint_msg.WEIGHT_MIN


class Goal(ABC):

    @abc.abstractmethod
    def __init__(self):
        """
        This is where you specify goal parameters and save them as self attributes.
        """
        self.god_map = GodMap()
        self._test_mode = self.god_map.get_data(identifier.test_mode)
        self._sub_goals: List[Goal] = []
        self.world = self.god_map.get_data(identifier.world)  # type: WorldTree

    @property
    def prediction_horizon(self) -> int:
        return self.god_map.get_data(identifier.prediction_horizon)

    @abc.abstractmethod
    def make_constraints(self):
        """
        This is where you create your constraints using casadi_wrapper.
        Use self.add_constraint.
        """

    @abc.abstractmethod
    def __str__(self) -> str:
        """
        Make sure the returns a unique identifier, in case multiple goals of the same type are added.
        Usage of 'self.__class__.__name__' is recommended.
        """
        return str(self.__class__.__name__)

    def add_collision_check(self, link_a: PrefixName, link_b: PrefixName, distance: float):
        """
        Tell Giskard to check this collision, even if it got disabled through other means such as allow_all_collisions.
        :param link_a:
        :param link_b:
        :param distance: distance threshold for the collision check. Only distances smaller than this value can be
                            detected.
        """
        if self.world.link_order(link_a, link_b):
            key = (link_a, link_b)
        else:
            key = (link_b, link_a)
        if self.world.are_linked(link_a, link_b):
            return
        try:
            added_checks = self.god_map.get_data(identifier.added_collision_checks)
        except KeyError:
            added_checks = {}
            self.god_map.set_data(identifier.added_collision_checks, added_checks)
        if key in added_checks:
            added_checks[key] = max(added_checks[key], distance)
        else:
            added_checks[key] = distance

    def _save_self_on_god_map(self):
        try:
            self.god_map.get_data(self._get_identifier())
            raise ConstraintInitalizationException(f'Constraint named {str(self)} already exists.')
        except KeyError:
            self.god_map.set_data(self._get_identifier(), self)

    def _get_identifier(self):
        try:
            return identifier.goals + [str(self)]
        except AttributeError as e:
            raise AttributeError(
                f'You have to ensure that str(self) is possible before calling parents __init__: {e}')

    def traj_time_in_seconds(self) -> w.Expression:
        t = self.god_map.to_expr(identifier.time)
        return t * self.get_sampling_period_symbol()

    def transform_msg(self, target_frame: my_string, msg: transformable_message, tf_timeout: float = 1) \
            -> transformable_message:
        """
        First tries to transform the message using the worlds internal kinematic tree.
        If it fails, it uses tf as a backup.
        :param target_frame:
        :param msg:
        :param tf_timeout: for how long Giskard should wait for tf.
        :return: message relative to target frame
        """
        try:
            try:
                msg.header.frame_id = self.world.search_for_link_name(msg.header.frame_id)
            except UnknownGroupException:
                pass
            return self.world.transform_msg(target_frame, msg)
        except KeyError:
            return tf.transform_msg(target_frame, msg, timeout=tf_timeout)

    def get_joint_position_symbol(self, joint_name: PrefixName) -> Union[w.Symbol, float]:
        """
        returns a symbol that refers to the given joint
        """
        if not self.world.has_joint(joint_name):
            raise KeyError(f'World doesn\'t have joint named: {joint_name}.')
        joint = self.world.joints[joint_name]
        if isinstance(joint, OneDofJoint):
            return joint.get_symbol(Derivatives.position)
        raise TypeError(f'get_joint_position_symbol is only supported for OneDofJoint, not {type(joint)}')

    @property
    def sample_period(self) -> float:
        return self.god_map.get_data(identifier.sample_period)

    def get_sampling_period_symbol(self) -> Union[w.Symbol, float]:
        return self.god_map.to_symbol(identifier.sample_period)

    def get_fk(self, root: PrefixName, tip: PrefixName) -> w.TransMatrix:
        """
        Return the homogeneous transformation matrix root_T_tip as a function that is dependent on the joint state.
        """
        result: w.TransMatrix = self.world.compose_fk_expression(root, tip)
        result.reference_frame = root
        result.child_frame = tip
        return result

    def get_fk_evaluated(self, root: PrefixName, tip: PrefixName) -> w.TransMatrix:
        """
        Return the homogeneous transformation matrix root_T_tip. This Matrix refers to the evaluated current transform.
        This means that the derivative towards the joint symbols will be 0.
        """
        result: w.TransMatrix = self.god_map.list_to_frame(identifier.fk_np + [(root, tip)])
        result.reference_frame = root
        result.child_frame = tip
        return result

    def get_parameter_as_symbolic_expression(self, name: str) -> Union[Union[w.Symbol, float], w.Expression]:
        """
        :param name: name of a class attribute, e.g. self.muh
        :return: a symbol (or matrix of symbols) that refers to self.muh
        """
        if not hasattr(self, name):
            raise AttributeError(f'{self.__class__.__name__} doesn\'t have attribute {name}')
        return self.god_map.to_expr(self._get_identifier() + [name])

    def get_expr_velocity(self, expr: w.Expression) -> w.Expression:
        """
        Creates an expressions that computes the total derivative of expr
        """
        return w.total_derivative(expr,
                                  self.joint_position_symbols,
                                  self.joint_velocity_symbols)

    @property
    def joint_position_symbols(self) -> List[Union[w.Symbol, float]]:
        position_symbols = []
        for joint in self.world.controlled_joints:
            position_symbols.extend(self.world.joints[joint].free_variables)
        return [x.get_symbol(Derivatives.position) for x in position_symbols]

    @property
    def joint_velocity_symbols(self) -> List[Union[w.Symbol, float]]:
        velocity_symbols = []
        for joint in self.world.controlled_joints:
            velocity_symbols.extend(self.world._joints[joint].free_variable_list)
        return [x.get_symbol(Derivatives.velocity) for x in velocity_symbols]

    @property
    def joint_acceleration_symbols(self) -> List[Union[w.Symbol, float]]:
        acceleration_symbols = []
        for joint in self.world.controlled_joints:
            acceleration_symbols.extend(self.world.joints[joint].free_variables)
        return [x.get_symbol(Derivatives.acceleration) for x in acceleration_symbols]

    def get_fk_velocity(self, root: PrefixName, tip: PrefixName) -> w.Expression:
        r_T_t = self.get_fk(root, tip)
        r_R_t = r_T_t.to_rotation()
        axis, angle = r_R_t.to_axis_angle()
        r_R_t_axis_angle = axis * angle
        r_P_t = r_T_t.to_position()
        fk = w.Expression([r_P_t[0],
                           r_P_t[1],
                           r_P_t[2],
                           r_R_t_axis_angle[0],
                           r_R_t_axis_angle[1],
                           r_R_t_axis_angle[2]])
        return self.get_expr_velocity(fk)

    @profile
    def get_constraints(self) -> Tuple[Dict[str, EqualityConstraint],
                                       Dict[str, InequalityConstraint],
                                       Dict[str, DerivativeInequalityConstraint],
                                       Dict[str, Union[w.Symbol, float]]]:
        self._equality_constraints = OrderedDict()
        self._inequality_constraints = OrderedDict()
        self._derivative_constraints = OrderedDict()
        self._debug_expressions = OrderedDict()
        self.make_constraints()
        for sub_goal in self._sub_goals:
            sub_goal._save_self_on_god_map()
            equality_constraints, inequality_constraints, derivative_constraints, debug_expressions = \
                sub_goal.get_constraints()
            # TODO check for duplicates
            self._equality_constraints.update(_prepend_prefix(self.__class__.__name__, equality_constraints))
            self._inequality_constraints.update(_prepend_prefix(self.__class__.__name__, inequality_constraints))
            self._derivative_constraints.update(_prepend_prefix(self.__class__.__name__, derivative_constraints))
            self._debug_expressions.update(_prepend_prefix(self.__class__.__name__, debug_expressions))
        return self._equality_constraints, self._inequality_constraints, self._derivative_constraints, \
               self._debug_expressions

    def add_constraints_of_goal(self, goal: Goal):
        self._sub_goals.append(goal)

    def add_velocity_constraint(self,
                                lower_velocity_limit: Union[w.symbol_expr_float, List[w.symbol_expr_float]],
                                upper_velocity_limit: Union[w.symbol_expr_float, List[w.symbol_expr_float]],
                                weight: w.symbol_expr_float,
                                task_expression: w.symbol_expr,
                                velocity_limit: w.symbol_expr_float,
                                name_suffix: Optional[str] = None,
                                control_horizon: Optional[w.symbol_expr_float] = None,
                                lower_slack_limit: Union[w.symbol_expr_float, List[w.symbol_expr_float]] = -1e4,
                                upper_slack_limit: Union[w.symbol_expr_float, List[w.symbol_expr_float]] = 1e4,
                                horizon_function: Optional[Callable[[float, int], float]] = None):
        """
        Add a velocity constraint. Internally, this will be converted into multiple constraints, to ensure that the
        velocity stays within the given bounds.
        :param lower_velocity_limit:
        :param upper_velocity_limit:
        :param weight:
        :param task_expression:
        :param velocity_limit:
        :param name_suffix:
        :param lower_slack_limit:
        :param upper_slack_limit:
        :param horizon_function: A function that can takes 'weight' and the id within the horizon as input and computes
                                    a new weight. Can be used to give points towards the end of the horizon a different
                                    weight
        """
        name_suffix = name_suffix if name_suffix else ''
        name = str(self) + name_suffix
        if name in self._derivative_constraints:
            raise KeyError(f'a constraint with name \'{name}\' already exists')
        self._derivative_constraints[name] = DerivativeInequalityConstraint(name=name,
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

    def add_acceleration_constraint(self,
                                    lower_acceleration_limit: Union[w.symbol_expr_float, List[w.symbol_expr_float]],
                                    upper_acceleration_limit: Union[w.symbol_expr_float, List[w.symbol_expr_float]],
                                    weight: w.symbol_expr_float,
                                    task_expression: w.symbol_expr,
                                    acceleration_limit: w.symbol_expr_float,
                                    name_suffix: Optional[str] = None,
                                    lower_slack_limit: Union[w.symbol_expr_float, List[w.symbol_expr_float]] = -1e4,
                                    upper_slack_limit: Union[w.symbol_expr_float, List[w.symbol_expr_float]] = 1e4,
                                    horizon_function: Optional[Callable[[float, int], float]] = None):
        """
        Add a acceleration constraint. Internally, this will be converted into multiple constraints, to ensure that the
        acceleration stays within the given bounds.
        :param lower_acceleration_limit:
        :param upper_acceleration_limit:
        :param weight:
        :param task_expression:
        :param acceleration_limit:
        :param name_suffix:
        :param lower_slack_limit:
        :param upper_slack_limit:
        :param horizon_function: A function that can takes 'weight' and the id within the horizon as input and computes
                                    a new weight. Can be used to give points towards the end of the horizon a different
                                    weight
        """
        name_suffix = name_suffix if name_suffix else ''
        name = str(self) + name_suffix
        if name in self._derivative_constraints:
            raise KeyError(f'a constraint with name \'{name}\' already exists')
        self._derivative_constraints[name] = DerivativeInequalityConstraint(name=name,
                                                                            derivative=Derivatives.acceleration,
                                                                            expression=task_expression,
                                                                            lower_limit=lower_acceleration_limit,
                                                                            upper_limit=upper_acceleration_limit,
                                                                            quadratic_weight=weight,
                                                                            normalization_factor=acceleration_limit,
                                                                            lower_slack_limit=lower_slack_limit,
                                                                            upper_slack_limit=upper_slack_limit,
                                                                            horizon_function=horizon_function)

    def add_jerk_constraint(self,
                            lower_jerk_limit: Union[w.symbol_expr_float, List[w.symbol_expr_float]],
                            upper_jerk_limit: Union[w.symbol_expr_float, List[w.symbol_expr_float]],
                            weight: w.symbol_expr_float,
                            task_expression: w.symbol_expr,
                            acceleration_limit: w.symbol_expr_float,
                            name_suffix: Optional[str] = None,
                            lower_slack_limit: Union[w.symbol_expr_float, List[w.symbol_expr_float]] = -1e4,
                            upper_slack_limit: Union[w.symbol_expr_float, List[w.symbol_expr_float]] = 1e4,
                            horizon_function: Optional[Callable[[float, int], float]] = None):
        name_suffix = name_suffix if name_suffix else ''
        name = str(self) + name_suffix
        if name in self._derivative_constraints:
            raise KeyError(f'a constraint with name \'{name}\' already exists')
        self._derivative_constraints[name] = DerivativeInequalityConstraint(name=name,
                                                                            derivative=Derivatives.jerk,
                                                                            expression=task_expression,
                                                                            lower_limit=lower_jerk_limit,
                                                                            upper_limit=upper_jerk_limit,
                                                                            quadratic_weight=weight,
                                                                            normalization_factor=acceleration_limit,
                                                                            lower_slack_limit=lower_slack_limit,
                                                                            upper_slack_limit=upper_slack_limit,
                                                                            horizon_function=horizon_function)

    def add_inequality_constraint(self,
                                  reference_velocity: w.symbol_expr_float,
                                  lower_error: symbol_expr_float,
                                  upper_error: symbol_expr_float,
                                  weight: symbol_expr_float,
                                  task_expression: w.symbol_expr,
                                  name: Optional[str] = None,
                                  lower_slack_limit: Optional[w.symbol_expr_float] = None,
                                  upper_slack_limit: Optional[w.symbol_expr_float] = None,
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
            raise GiskardException(f'expression must have shape (1,1), has {task_expression.shape}')
        name = name if name else ''
        name = str(self) + name
        if name in self._inequality_constraints:
            raise KeyError(f'A constraint with name \'{name}\' already exists. '
                           f'You need to set a name, if you add multiple constraints.')
        lower_slack_limit = lower_slack_limit if lower_slack_limit is not None else -float('inf')
        upper_slack_limit = upper_slack_limit if upper_slack_limit is not None else float('inf')
        self._inequality_constraints[name] = InequalityConstraint(name=name,
                                                                  expression=task_expression,
                                                                  lower_error=lower_error,
                                                                  upper_error=upper_error,
                                                                  velocity_limit=reference_velocity,
                                                                  quadratic_weight=weight,
                                                                  lower_slack_limit=lower_slack_limit,
                                                                  upper_slack_limit=upper_slack_limit,
                                                                  control_horizon=control_horizon)

    def add_equality_constraint(self,
                                reference_velocity: w.symbol_expr_float,
                                equality_bound: w.symbol_expr_float,
                                weight: w.symbol_expr_float,
                                task_expression: w.symbol_expr,
                                name: Optional[str] = None,
                                lower_slack_limit: Optional[w.symbol_expr_float] = None,
                                upper_slack_limit: Optional[w.symbol_expr_float] = None,
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
            raise GiskardException(f'expression must have shape (1,1), has {task_expression.shape}')
        name = name if name else ''
        name = str(self) + name
        if name in self._inequality_constraints:
            raise KeyError(f'A constraint with name \'{name}\' already exists. '
                           f'You need to set a name, if you add multiple constraints.')
        lower_slack_limit = lower_slack_limit if lower_slack_limit is not None else -float('inf')
        upper_slack_limit = upper_slack_limit if upper_slack_limit is not None else float('inf')
        self._equality_constraints[name] = EqualityConstraint(name=name,
                                                              expression=task_expression,
                                                              derivative_goal=equality_bound,
                                                              velocity_limit=reference_velocity,
                                                              quadratic_weight=weight,
                                                              lower_slack_limit=lower_slack_limit,
                                                              upper_slack_limit=upper_slack_limit,
                                                              control_horizon=control_horizon)

    def add_inequality_constraint_vector(self,
                                         reference_velocities: Union[
                                             w.Expression, w.Vector3, w.Point3, List[w.symbol_expr_float]],
                                         lower_errors: Union[
                                             w.Expression, w.Vector3, w.Point3, List[w.symbol_expr_float]],
                                         upper_errors: Union[
                                             w.Expression, w.Vector3, w.Point3, List[w.symbol_expr_float]],
                                         weights: Union[w.Expression, w.Vector3, w.Point3, List[w.symbol_expr_float]],
                                         task_expression: Union[w.Expression, w.Vector3, w.Point3, List[w.symbol_expr]],
                                         names: List[str],
                                         lower_slack_limits: Optional[List[w.symbol_expr_float]] = None,
                                         upper_slack_limits: Optional[List[w.symbol_expr_float]] = None):
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
            raise ConstraintInitalizationException('All parameters must have the same length.')
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
                                           w.Expression, w.Vector3, w.Point3, List[w.symbol_expr_float]],
                                       equality_bounds: Union[
                                           w.Expression, w.Vector3, w.Point3, List[w.symbol_expr_float]],
                                       weights: Union[w.Expression, w.Vector3, w.Point3, List[w.symbol_expr_float]],
                                       task_expression: Union[w.Expression, w.Vector3, w.Point3, List[w.symbol_expr]],
                                       names: List[str],
                                       lower_slack_limits: Optional[List[w.symbol_expr_float]] = None,
                                       upper_slack_limits: Optional[List[w.symbol_expr_float]] = None):
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

    def add_debug_expr(self, name: str, expr: w.all_expressions_float):
        """
        Add any expression for debug purposes. They will be evaluated as well and can be plotted by activating
        the debug plotter in this Giskard config.
        :param name:
        :param expr:
        """
        name = f'{self}/{name}'
        if not isinstance(expr, w.Symbol_):
            expr = w.Expression(expr)
        self._debug_expressions[name] = expr

    def add_position_constraint(self,
                                expr_current: Union[w.Symbol, float],
                                expr_goal: Union[w.Symbol, float],
                                reference_velocity: Union[w.Symbol, float],
                                weight: Union[w.Symbol, float] = WEIGHT_BELOW_CA,
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

    def add_point_goal_constraints(self,
                                   frame_P_current: w.Point3,
                                   frame_P_goal: w.Point3,
                                   reference_velocity: w.symbol_expr_float,
                                   weight: w.symbol_expr_float,
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

    def add_translational_velocity_limit(self,
                                         frame_P_current: w.Point3,
                                         max_velocity: w.symbol_expr_float,
                                         weight: w.symbol_expr_float,
                                         max_violation: w.symbol_expr_float = 1e4,
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
        trans_error = w.norm(frame_P_current)
        trans_error = w.if_eq_zero(trans_error, 0.01, trans_error)
        self.add_velocity_constraint(upper_velocity_limit=max_velocity,
                                     lower_velocity_limit=-max_velocity,
                                     weight=weight,
                                     task_expression=trans_error,
                                     lower_slack_limit=-max_violation,
                                     upper_slack_limit=max_violation,
                                     velocity_limit=max_velocity,
                                     name_suffix=f'{name}/vel')

    def add_vector_goal_constraints(self,
                                    frame_V_current: w.Vector3,
                                    frame_V_goal: w.Vector3,
                                    reference_velocity: w.symbol_expr_float,
                                    weight: w.symbol_expr_float = WEIGHT_BELOW_CA,
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
        angle = w.save_acos(frame_V_current.dot(frame_V_goal))
        # avoid singularity by staying away from pi
        angle_limited = w.min(w.max(angle, -reference_velocity), reference_velocity)
        angle_limited = w.save_division(angle_limited, angle)
        root_V_goal_normal_intermediate = w.slerp(frame_V_current, frame_V_goal, angle_limited)

        error = root_V_goal_normal_intermediate - frame_V_current

        self.add_equality_constraint_vector(reference_velocities=[reference_velocity] * 3,
                                            equality_bounds=error[:3],
                                            weights=[weight] * 3,
                                            task_expression=frame_V_current[:3],
                                            names=[f'{name}/trans/x',
                                                   f'{name}/trans/y',
                                                   f'{name}/trans/z'])

    def add_rotation_goal_constraints(self,
                                      frame_R_current: w.RotationMatrix,
                                      frame_R_goal: w.RotationMatrix,
                                      current_R_frame_eval: w.RotationMatrix,
                                      reference_velocity: Union[w.Symbol, float],
                                      weight: Union[w.Symbol, float],
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
        hack = w.RotationMatrix.from_axis_angle(w.Vector3((0, 0, 1)), 0.0001)
        frame_R_current = frame_R_current.dot(hack)  # hack to avoid singularity
        tip_Q_tipCurrent = current_R_frame_eval.dot(frame_R_current).to_quaternion()
        tip_R_goal = current_R_frame_eval.dot(frame_R_goal)

        tip_Q_goal = tip_R_goal.to_quaternion()

        tip_Q_goal = w.if_greater_zero(-tip_Q_goal[3], -tip_Q_goal, tip_Q_goal)  # flip to get shortest path

        expr = tip_Q_tipCurrent
        # w is not needed because its derivative is always 0 for identity quaternions
        self.add_equality_constraint_vector(reference_velocities=[reference_velocity] * 3,
                                            equality_bounds=tip_Q_goal[:3],
                                            weights=[weight] * 3,
                                            task_expression=expr[:3],
                                            names=[f'{name}/rot/x',
                                                   f'{name}/rot/y',
                                                   f'{name}/rot/z'])

    def add_rotational_velocity_limit(self,
                                      frame_R_current: w.RotationMatrix,
                                      max_velocity: Union[w.Symbol, float],
                                      weight: Union[w.Symbol, float],
                                      max_violation: Union[w.Symbol, float] = 1e4,
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
                                     name_suffix=f'{name}/q/vel',
                                     velocity_limit=max_velocity)


def _prepend_prefix(prefix, d):
    new_dict = OrderedDict()
    for key, value in d.items():
        new_key = f'{prefix}/{key}'
        try:
            value.name = f'{prefix}/{value.name}'
        except AttributeError:
            pass
        new_dict[new_key] = value
    return new_dict


class NonMotionGoal(Goal):
    """
    Inherit from this goal, if the goal does not add any constraints.
    """

    def make_constraints(self):
        pass
