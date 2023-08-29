from __future__ import annotations

import abc
from abc import ABC
from collections import OrderedDict
from typing import Optional, Tuple, Dict, List, Union, Callable, TYPE_CHECKING

from giskardpy.goals.monitors.monitors import Monitor
from giskardpy.goals.tasks.task import Task, WEIGHT_BELOW_CA
from giskardpy.god_map_user import GodMapWorshipper

if TYPE_CHECKING:
    from giskardpy.tree.control_modes import ControlModes

import giskardpy.identifier as identifier
import giskardpy.utils.tfwrapper as tf
from giskardpy import casadi_wrapper as w
from giskardpy.exceptions import ConstraintInitalizationException, UnknownGroupException
from giskardpy.model.joints import OneDofJoint
from giskardpy.my_types import my_string, transformable_message, PrefixName, Derivatives
from giskardpy.qp.constraint import InequalityConstraint, EqualityConstraint, DerivativeInequalityConstraint


class Goal(GodMapWorshipper, ABC):
    _sub_goals: List[Goal]
    tasks: List[Task]

    @abc.abstractmethod
    def __init__(self):
        """
        This is where you specify goal parameters and save them as self attributes.
        """
        self._sub_goals = []
        self.tasks = []

    def clean_up(self):
        pass

    def is_done(self):
        return None

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
        key = link_a, link_b
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
        if self.god_map.get_data(identifier.control_mode) == ControlModes.close_loop:
            return t
        else:
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
        for task in self.tasks:
            for constraint in task.get_eq_constraints():
                name = f'{str(self)}/{task.name}/{constraint.name}'
                constraint.name = name
                self._equality_constraints[name] = constraint
            # self._equality_constraints.update(_prepend_prefix(self.__class__.__name__, equality_constraints))

        for sub_goal in self._sub_goals:
            sub_goal._save_self_on_god_map()
            equality_constraints, inequality_constraints, derivative_constraints, debug_expressions = \
                sub_goal.get_constraints()

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

    def add_task(self, task: Task):
        self.tasks.append(task)

    def add_monitor(self, monitor: Monitor):
        self.monitor_manager.add_monitor(monitor)

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
