from __future__ import annotations

import abc
from abc import ABC
from typing import List, Union, Optional

from giskardpy.motion_graph.graph_node import MotionStatechartNode
from giskardpy.motion_graph.monitors.monitors import Monitor, Monitor
from giskardpy.god_map import god_map
from giskardpy.motion_graph.tasks.task import Task
from giskardpy.symbol_manager import symbol_manager
from giskardpy.utils.utils import string_shortener
from giskardpy.data_types.exceptions import GoalInitalizationException
from giskardpy.model.joints import OneDofJoint
from giskardpy.data_types.data_types import PrefixName, Derivatives
import giskardpy.casadi_wrapper as cas


class Goal(MotionStatechartNode):
    tasks: List[Task]
    monitors: List[Monitor]
    goals: List[Goal]

    def __init__(self, *, name: Optional[str] = None, plot: bool = True):
        super().__init__(name=name, plot=plot)
        self.tasks = []
        self.monitors = []
        self.goals = []

    def get_observation_state_expression(self):
        return symbol_manager.get_symbol(f'god_map'
                                         f'.motion_graph_manager'
                                         f'.goal_state'
                                         f'.get_observation_state(\'{self.name}\')')

    def get_life_cycle_state_expression(self):
        return symbol_manager.get_symbol(f'god_map'
                                         f'.motion_graph_manager'
                                         f'.goal_state'
                                         f'.get_life_cycle_state(\'{self.name}\')')

    def has_tasks(self) -> bool:
        return len(self.tasks) > 0

    def arrange_in_sequence(self, nodes: List[MotionStatechartNode]) -> None:
        first_node = nodes[0]
        first_node.end_condition = first_node.get_observation_state_expression()
        for node in nodes[1:]:
            node.start_condition = first_node.get_observation_state_expression()
            node.end_condition = node.get_observation_state_expression()
            first_node = node

    def get_joint_position_symbol(self, joint_name: PrefixName) -> Union[cas.Symbol, float]:
        """
        returns a symbol that refers to the given joint
        """
        if not god_map.world.has_joint(joint_name):
            raise KeyError(f'World doesn\'t have joint named: {joint_name}.')
        joint = god_map.world.joints[joint_name]
        if isinstance(joint, OneDofJoint):
            return joint.get_symbol(Derivatives.position)
        raise TypeError(f'get_joint_position_symbol is only supported for OneDofJoint, not {type(joint)}')

    def connect_start_condition_to_all_tasks(self, condition: cas.Expression) -> None:
        for task in self.tasks:
            if cas.is_true_symbol(task.start_condition):
                task.start_condition = condition
            else:
                task.start_condition = cas.logic_and(task.start_condition, condition)

    def connect_pause_condition_to_all_tasks(self, condition: cas.Expression) -> None:
        for task in self.tasks:
            if cas.is_false_symbol(task.pause_condition):
                task.pause_condition = condition
            else:
                task.pause_condition = cas.logic_or(task.pause_condition, condition)

    def connect_end_condition_to_all_tasks(self, condition: cas.Expression) -> None:
        for task in self.tasks:
            if cas.is_false_symbol(task.end_condition):
                task.end_condition = condition
            elif not cas.is_false_symbol(condition):
                task.end_condition = cas.logic_and(task.end_condition, condition)

    def connect_monitors_to_all_tasks(self,
                                      start_condition: cas.Expression,
                                      pause_condition: cas.Expression,
                                      end_condition: cas.Expression):
        self.connect_start_condition_to_all_tasks(start_condition)
        self.connect_pause_condition_to_all_tasks(pause_condition)
        self.connect_end_condition_to_all_tasks(end_condition)

    @property
    def ref_str(self) -> str:
        """
        A string referring to self on the god_map. Used with symbol manager.
        """
        return f'god_map.motion_graph_manager.goal_state.get_node(\'{self.name}\')'

    def __add__(self, other: str) -> str:
        if isinstance(other, str):
            return self.ref_str + other
        raise NotImplementedError('Goal can only be added with a string.')

    # def get_expr_velocity(self, expr: cas.Expression) -> cas.Expression:
    #     """
    #     Creates an expressions that computes the total derivative of expr
    #     """
    #     return cas.total_derivative(expr,
    #                                 self.joint_position_symbols,
    #                                 self.joint_velocity_symbols)
    #
    # @property
    # def joint_position_symbols(self) -> List[Union[cas.Symbol, float]]:
    #     position_symbols = []
    #     for joint in god_map.world.controlled_joints:
    #         position_symbols.extend(god_map.world.joints[joint].free_variables)
    #     return [x.get_symbol(Derivatives.position) for x in position_symbols]
    #
    # @property
    # def joint_velocity_symbols(self) -> List[Union[cas.Symbol, float]]:
    #     velocity_symbols = []
    #     for joint in god_map.world.controlled_joints:
    #         velocity_symbols.extend(god_map.world.joints[joint].free_variable_list)
    #     return [x.get_symbol(Derivatives.velocity) for x in velocity_symbols]
    #
    # @property
    # def joint_acceleration_symbols(self) -> List[Union[cas.Symbol, float]]:
    #     acceleration_symbols = []
    #     for joint in god_map.world.controlled_joints:
    #         acceleration_symbols.extend(god_map.world.joints[joint].free_variables)
    #     return [x.get_symbol(Derivatives.acceleration) for x in acceleration_symbols]

    def _task_sanity_check(self):
        if not self.has_tasks():
            raise GoalInitalizationException(f'Goal {str(self)} has no tasks.')

    def add_constraints_of_goal(self, goal: Goal):
        for task in goal.tasks:
            if not [t for t in self.tasks if t.name == task.name]:
                self.tasks.append(task)
            else:
                raise GoalInitalizationException(f'Constraint with name {task.name} already exists.')

    def add_task(self, task: Task) -> None:
        self.tasks.append(task)

    def add_monitor(self, monitor: Monitor) -> None:
        self.monitors.append(monitor)

    def add_goal(self, goal: Goal) -> None:
        self.goals.append(goal)
