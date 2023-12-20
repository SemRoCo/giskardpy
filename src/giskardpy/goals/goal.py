from __future__ import annotations

import abc
from abc import ABC
from collections import OrderedDict
from typing import Optional, Tuple, Dict, List, Union, Callable, TYPE_CHECKING, overload

from geometry_msgs.msg import PoseStamped, PointStamped, QuaternionStamped, Vector3Stamped

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
from giskardpy.qp.constraint import InequalityConstraint, EqualityConstraint, DerivativeInequalityConstraint, \
    ManipulabilityConstraint


class Goal(ABC):
    tasks: List[Task]
    name: str

    @abc.abstractmethod
    def __init__(self,
                 name: str,
                 start_monitors: Optional[List[Monitor]] = None,
                 hold_monitors: Optional[List[Monitor]] = None,
                 end_monitors: Optional[List[Monitor]] = None):
        """
        This is where you specify goal parameters and save them as self attributes.
        """
        self.tasks = []
        self.name = name

    def clean_up(self):
        pass

    def is_done(self):
        return None

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    def has_tasks(self) -> bool:
        return len(self.tasks) > 0

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

    def connect_start_monitors_to_all_tasks(self, monitors: List[Monitor]):
        for monitor in monitors:
            for task in self.tasks:
                task.add_start_monitors_monitor(monitor)

    def connect_hold_monitors_to_all_tasks(self, monitors: List[Monitor]):
        for monitor in monitors:
            for task in self.tasks:
                task.add_hold_monitors_monitor(monitor)

    def connect_end_monitors_to_all_tasks(self, monitors: List[Monitor]):
        for monitor in monitors:
            for task in self.tasks:
                task.add_end_monitors_monitor(monitor)

    def connect_monitors_to_all_tasks(self, start_monitors: List[Monitor], hold_monitors: List[Monitor], end_monitors: List[Monitor]):
        self.connect_start_monitors_to_all_tasks(start_monitors)
        self.connect_hold_monitors_to_all_tasks(hold_monitors)
        self.connect_end_monitors_to_all_tasks(end_monitors)

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
                                       Dict[str, Union[w.Symbol, float]],
                                       Dict[str, ManipulabilityConstraint]]:
        self._equality_constraints = OrderedDict()
        self._inequality_constraints = OrderedDict()
        self._derivative_constraints = OrderedDict()
        self._debug_expressions = OrderedDict()
        self._manip_constraints = OrderedDict()
        
        self._task_sanity_check()
        
        for task in self.tasks:
            for constraint in task.get_eq_constraints():
                name = f'{task.name}/{constraint.name}'
                constraint.name = name
                self._equality_constraints[constraint.name] = constraint
            for constraint in task.get_neq_constraints():
                name = f'{task.name}/{constraint.name}'
                constraint.name = name
                self._inequality_constraints[constraint.name] = constraint
            for constraint in task.get_derivative_constraints():
                name = f'{task.name}/{constraint.name}'
                constraint.name = name
                self._derivative_constraints[constraint.name] = constraint
            for constraint in task.get_manipulability_constraint():
                name = f'{task.name}/{constraint.name}'
                constraint.name = name
                self._manip_constraints[constraint.name] = constraint

        return self._equality_constraints, self._inequality_constraints, self._derivative_constraints, \
               self._manip_constraints, self._debug_expressions

    def _task_sanity_check(self):
        if not self.has_tasks():
            raise ConstraintInitalizationException(f'Goal {str(self)} has no tasks.')

    def add_constraints_of_goal(self, goal: Goal):
        for task in goal.tasks:
            if not [t for t in self.tasks if t.name == task.name]:
                self.tasks.append(task)
            else:
                raise ConstraintInitalizationException(f'Constraint with name {task.name} already exists.')

    def add_task(self, task: Task):
        if task.name != '':
            task.name = f'{self.name}/{task.name}'
        else:
            task.name = self.name
        self.tasks.append(task)

    def add_tasks(self, tasks: List[Task]):
        for task in tasks:
            self.add_task(task)

    def add_monitor(self, monitor: Monitor):
        god_map.monitor_manager.add_monitor(monitor)


class NonMotionGoal(Goal):
    """
    Inherit from this goal, if the goal does not add any constraints.
    """

    def make_constraints(self):
        pass

    def _task_sanity_check(self):
        pass
