from __future__ import annotations

import abc
from abc import ABC
from collections import OrderedDict
from typing import Optional, Tuple, Dict, List, Union, Callable, TYPE_CHECKING

from giskardpy.goals.monitors.monitors import Monitor
from giskardpy.goals.tasks.task import Task, WEIGHT_BELOW_CA
from giskardpy.god_map import god_map
from giskardpy.symbol_manager import symbol_manager

if TYPE_CHECKING:
    from giskardpy.tree.control_modes import ControlModes

import giskardpy.utils.tfwrapper as tf
from giskardpy import casadi_wrapper as w
from giskardpy.exceptions import ConstraintInitalizationException, UnknownGroupException
from giskardpy.model.joints import OneDofJoint
from giskardpy.my_types import my_string, transformable_message, PrefixName, Derivatives
from giskardpy.qp.constraint import InequalityConstraint, EqualityConstraint, DerivativeInequalityConstraint


class Goal(ABC):
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

    @abc.abstractmethod
    def __str__(self) -> str:
        """
        Make sure the returns a unique str, in case multiple goals of the same type are added.
        Usage of 'self.__class__.__name__' is recommended.
        """
        return str(self.__class__.__name__)

    def traj_time_in_seconds(self) -> w.Expression:
        t = symbol_manager.time
        if god_map.is_closed_loop():
            return t
        else:
            return t * god_map.qp_controller_config.sample_period

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
                msg.header.frame_id = god_map.world.search_for_link_name(msg.header.frame_id)
            except UnknownGroupException:
                pass
            return god_map.world.transform_msg(target_frame, msg)
        except KeyError:
            return tf.transform_msg(target_frame, msg, timeout=tf_timeout)

    def get_joint_position_symbol(self, joint_name: PrefixName) -> Union[w.Symbol, float]:
        """
        returns a symbol that refers to the given joint
        """
        if not god_map.world.has_joint(joint_name):
            raise KeyError(f'World doesn\'t have joint named: {joint_name}.')
        joint = god_map.world.joints[joint_name]
        if isinstance(joint, OneDofJoint):
            return joint.get_symbol(Derivatives.position)
        raise TypeError(f'get_joint_position_symbol is only supported for OneDofJoint, not {type(joint)}')

    def connect_to_end(self, monitor: Monitor):
        for task in self.tasks:
            task.add_to_end_monitor(monitor)

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
        for joint in god_map.world.controlled_joints:
            position_symbols.extend(god_map.world.joints[joint].free_variables)
        return [x.get_symbol(Derivatives.position) for x in position_symbols]

    @property
    def joint_velocity_symbols(self) -> List[Union[w.Symbol, float]]:
        velocity_symbols = []
        for joint in god_map.world.controlled_joints:
            velocity_symbols.extend(god_map.world.joints[joint].free_variable_list)
        return [x.get_symbol(Derivatives.velocity) for x in velocity_symbols]

    @property
    def joint_acceleration_symbols(self) -> List[Union[w.Symbol, float]]:
        acceleration_symbols = []
        for joint in god_map.world.controlled_joints:
            acceleration_symbols.extend(god_map.world.joints[joint].free_variables)
        return [x.get_symbol(Derivatives.acceleration) for x in acceleration_symbols]

    @profile
    def get_constraints(self) -> Tuple[Dict[str, EqualityConstraint],
    Dict[str, InequalityConstraint],
    Dict[str, DerivativeInequalityConstraint],
    Dict[str, Union[w.Symbol, float]]]:
        self._equality_constraints = OrderedDict()
        self._inequality_constraints = OrderedDict()
        self._derivative_constraints = OrderedDict()
        self._debug_expressions = OrderedDict()
        if not isinstance(self, NonMotionGoal) and not self.tasks:
            raise ConstraintInitalizationException(f'Goal {str(self)} has no tasks.')
        for task in self.tasks:
            for constraint in task.get_eq_constraints():
                name = f'{str(self)}/{task.name}/{constraint.name}'
                constraint.name = name
                self._equality_constraints[name] = constraint
            for constraint in task.get_neq_constraints():
                name = f'{str(self)}/{task.name}/{constraint.name}'
                constraint.name = name
                self._inequality_constraints[name] = constraint
            for constraint in task.get_derivative_constraints():
                name = f'{str(self)}/{task.name}/{constraint.name}'
                constraint.name = name
                self._derivative_constraints[name] = constraint

        for sub_goal in self._sub_goals:
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

    def add_task(self, task: Task):
        self.tasks.append(task)

    def add_tasks(self, tasks: List[Task]):
        for task in tasks:
            self.add_task(task)

    def add_monitor(self, monitor: Monitor):
        god_map.monitor_manager.add_monitor(monitor)


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
