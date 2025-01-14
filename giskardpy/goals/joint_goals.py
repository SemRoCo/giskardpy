from __future__ import division

from typing import Dict, Optional, List, Tuple

from giskardpy import casadi_wrapper as cas
from giskardpy.data_types.data_types import Derivatives
from giskardpy.data_types.exceptions import GoalInitalizationException
from giskardpy.goals.goal import Goal
from giskardpy.god_map import god_map
from giskardpy.model.joints import OneDofJoint
from giskardpy.symbol_manager import symbol_manager
from giskardpy.motion_graph.tasks.task import WEIGHT_BELOW_CA





class AvoidJointLimits(Goal):
    def __init__(self,
                 percentage: float = 15,
                 joint_list: Optional[List[str]] = None,
                 group_name: Optional[str] = None,
                 weight: float = WEIGHT_BELOW_CA,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.BinaryTrue,
                 pause_condition: cas.Expression = cas.BinaryFalse,
                 end_condition: cas.Expression = cas.BinaryFalse
                 ):
        """
        Calls AvoidSingleJointLimits for each joint in joint_list
        :param percentage:
        :param joint_list: list of joints for which AvoidSingleJointLimits will be called
        :param weight:
        """
        self.joint_list = joint_list
        if name is None:
            name = f'{self.__class__.__name__}/{self.joint_list}'
        super().__init__(name)
        self.weight = weight
        self.percentage = percentage
        if joint_list is not None:
            joint_list = [god_map.world.search_for_joint_name(joint_name, group_name) for joint_name in joint_list]
        else:
            if group_name is None:
                joint_list = god_map.world.controlled_joints
            else:
                joint_list = god_map.world.groups[group_name].controlled_joints
        task = self.create_and_add_task('avoid joint limits')
        for joint_name in joint_list:
            if god_map.world.is_joint_prismatic(joint_name) or god_map.world.is_joint_revolute(joint_name):
                weight = self.weight
                joint_symbol = self.get_joint_position_symbol(joint_name)
                percentage = self.percentage / 100.
                lower_limit, upper_limit = god_map.world.get_joint_position_limits(joint_name)
                max_velocity = 100
                max_velocity = cas.min(max_velocity,
                                       god_map.world.get_joint_velocity_limits(joint_name)[1])

                joint_range = upper_limit - lower_limit
                center = (upper_limit + lower_limit) / 2.

                max_error = joint_range / 2. * percentage

                upper_goal = center + joint_range / 2. * (1 - percentage)
                lower_goal = center - joint_range / 2. * (1 - percentage)

                upper_err = upper_goal - joint_symbol
                lower_err = lower_goal - joint_symbol

                error = cas.max(cas.abs(cas.min(upper_err, 0)), cas.abs(cas.max(lower_err, 0)))
                weight = weight * (error / max_error)

                task.add_inequality_constraint(reference_velocity=max_velocity,
                                               name=str(joint_name),
                                               lower_error=lower_err,
                                               upper_error=upper_err,
                                               weight=weight,
                                               task_expression=joint_symbol)
        self.connect_monitors_to_all_tasks(start_condition, pause_condition, end_condition)


class JointPositionList(Goal):
    def __init__(self,
                 goal_state: Dict[str, float],
                 group_name: Optional[str] = None,
                 weight: float = WEIGHT_BELOW_CA,
                 max_velocity: float = 1,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.BinaryTrue,
                 pause_condition: cas.Expression = cas.BinaryFalse,
                 end_condition: cas.Expression = cas.BinaryFalse):
        """
        Calls JointPosition for a list of joints.
        :param goal_state: maps joint_name to goal position
        :param group_name: if joint_name is not unique, search in this group for matches.
        :param weight:
        :param max_velocity: will be applied to all joints, you should group joint types, e.g., prismatic joints
        :param hard: turns this into a hard constraint.
        """
        self.current_positions = []
        self.goal_positions = []
        self.velocity_limits = []
        self.names = []
        self.joint_names = list(sorted(goal_state.keys()))
        if name is None:
            name = f'{self.__class__.__name__} {self.joint_names}'
        super().__init__(name)
        self.max_velocity = max_velocity
        self.weight = weight
        if len(goal_state) == 0:
            raise GoalInitalizationException(f'Can\'t initialize {self} with no joints.')
        for joint_name, goal_position in goal_state.items():
            joint_name = god_map.world.search_for_joint_name(joint_name, group_name)

            ll_pos, ul_pos = god_map.world.compute_joint_limits(joint_name, Derivatives.position)
            if ll_pos is not None:
                goal_position = min(ul_pos, max(ll_pos, goal_position))

            ll_vel, ul_vel = god_map.world.compute_joint_limits(joint_name, Derivatives.velocity)
            velocity_limit = min(ul_vel, max(ll_vel, max_velocity))

            joint: OneDofJoint = god_map.world.joints[joint_name]
            self.names.append(str(joint_name))
            self.current_positions.append(joint.get_symbol(Derivatives.position))
            self.goal_positions.append(goal_position)
            self.velocity_limits.append(velocity_limit)

        task = self.create_and_add_task('joint goal')
        for name, current, goal, velocity_limit in zip(self.names, self.current_positions,
                                                       self.goal_positions, self.velocity_limits):
            if god_map.world.is_joint_continuous(name):
                error = cas.shortest_angular_distance(current, goal)
            else:
                error = goal - current

            task.add_equality_constraint(name=name,
                                         reference_velocity=velocity_limit,
                                         equality_bound=error,
                                         weight=self.weight,
                                         task_expression=current)

        self.connect_monitors_to_all_tasks(start_condition, pause_condition, end_condition)


class JointSignWave(Goal):
    def __init__(self, name: str, joint_name: str,
                 frequency: float,
                 amp_percentage: float,
                 start_condition: cas.Expression = cas.BinaryTrue,
                 pause_condition: cas.Expression = cas.BinaryFalse,
                 end_condition: cas.Expression = cas.BinaryFalse):
        super().__init__(name=name, start_condition=start_condition, pause_condition=pause_condition,
                         end_condition=end_condition)
        joint_name = god_map.world.search_for_joint_name(joint_name)
        t = self.create_and_add_task('task')
        joint_symbol = self.get_joint_position_symbol(joint_name)
        min_, max_ = god_map.world.compute_joint_limits(joint_name, Derivatives.position)
        _, max_vel = god_map.world.compute_joint_limits(joint_name, Derivatives.velocity)
        center = (max_ + min_) / 2
        goal_position = center + cas.sin(symbol_manager.time * 2 * cas.pi * (frequency)) * (
                max_ - center) * amp_percentage
        t.add_position_constraint(expr_current=joint_symbol,
                                  expr_goal=goal_position,
                                  reference_velocity=max_vel,
                                  weight=WEIGHT_BELOW_CA)
        god_map.debug_expression_manager.add_debug_expression('goal', goal_position)
        self.connect_monitors_to_all_tasks(start_condition, pause_condition, end_condition)


class UnlimitedJointGoal(Goal):
    def __init__(self, name: str, joint_name: str, goal_position: float,
                 start_condition: cas.Expression = cas.BinaryTrue,
                 pause_condition: cas.Expression = cas.BinaryFalse, end_condition: cas.Expression = cas.BinaryFalse):
        super().__init__(name=name, start_condition=start_condition, pause_condition=pause_condition,
                         end_condition=end_condition)
        joint_name = god_map.world.search_for_joint_name(joint_name)
        t = self.create_and_add_task('task')
        joint_symbol = self.get_joint_position_symbol(joint_name)
        t.add_position_constraint(expr_current=joint_symbol,
                                  expr_goal=goal_position,
                                  reference_velocity=2,
                                  weight=WEIGHT_BELOW_CA)
        self.connect_monitors_to_all_tasks(start_condition, pause_condition, end_condition)
