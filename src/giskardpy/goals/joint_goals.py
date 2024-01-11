from __future__ import division

from typing import Dict, Optional, List

from geometry_msgs.msg import PoseStamped

from giskardpy import casadi_wrapper as cas
from giskardpy.monitors.monitors import ExpressionMonitor
from giskardpy.god_map import god_map
from giskardpy.symbol_manager import symbol_manager
from giskardpy.exceptions import MotionBuildingException, GoalInitalizationException
from giskardpy.goals.goal import Goal, NonMotionGoal
from giskardpy.tasks.task import WEIGHT_BELOW_CA, Task
from giskardpy.model.joints import OmniDrive, DiffDrive, OmniDrivePR22, OneDofJoint
from giskardpy.data_types import PrefixName, Derivatives
from giskardpy.utils.expression_definition_utils import transform_msg
from giskardpy.utils.math import axis_angle_from_quaternion


class SetSeedConfiguration(NonMotionGoal):
    def __init__(self,
                 seed_configuration: Dict[str, float],
                 group_name: Optional[str] = None,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.TrueSymbol,
                 end_condition: cas.Expression = cas.TrueSymbol):
        """
        Overwrite the configuration of the world to allow starting the planning from a different state.
        Can only be used in plan only mode.
        :param seed_configuration: maps joint name to float
        :param group_name: if joint names are not unique, it will search in this group for matches.
        """
        self.seed_configuration = seed_configuration
        if name is None:
            name = f'{str(self.__class__.__name__)}/{list(self.seed_configuration.keys())}'
        super().__init__(name)
        if group_name is not None:
            seed_configuration = {PrefixName(joint_name, group_name): v for joint_name, v in seed_configuration.items()}
        if god_map.is_goal_msg_type_execute() and not god_map.is_standalone():
            raise GoalInitalizationException(f'It is not allowed to combine {str(self)} with plan and execute.')
        for joint_name, initial_joint_value in seed_configuration.items():
            joint_name = god_map.world.search_for_joint_name(joint_name, group_name)
            if joint_name not in god_map.world.state:
                raise KeyError(f'World has no joint \'{joint_name}\'.')
            god_map.world.state[joint_name].position = initial_joint_value
        god_map.world.notify_state_change()
        self.connect_monitors_to_all_tasks(start_condition, hold_condition, end_condition)


class SetOdometry(NonMotionGoal):
    def __init__(self,
                 group_name: str,
                 base_pose: PoseStamped,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.TrueSymbol,
                 end_condition: cas.Expression = cas.TrueSymbol):
        self.group_name = group_name
        if name is None:
            name = f'{self.__class__.__name__}/{self.group_name}'
        super().__init__(name)
        if god_map.is_goal_msg_type_execute() and not god_map.is_standalone():
            raise GoalInitalizationException(f'It is not allowed to combine {str(self)} with plan and execute.')
        brumbrum_joint_name = god_map.world.groups[group_name].root_link.child_joint_names[0]
        brumbrum_joint = god_map.world.joints[brumbrum_joint_name]
        if not isinstance(brumbrum_joint, (OmniDrive, DiffDrive, OmniDrivePR22)):
            raise GoalInitalizationException(f'Group {group_name} has no odometry joint.')
        base_pose = transform_msg(brumbrum_joint.parent_link_name, base_pose).pose
        god_map.world.state[brumbrum_joint.x.name].position = base_pose.position.x
        god_map.world.state[brumbrum_joint.y.name].position = base_pose.position.y
        axis, angle = axis_angle_from_quaternion(base_pose.orientation.x,
                                                 base_pose.orientation.y,
                                                 base_pose.orientation.z,
                                                 base_pose.orientation.w)
        if axis[-1] < 0:
            angle = -angle
        if isinstance(brumbrum_joint, OmniDrivePR22):
            god_map.world.state[brumbrum_joint.yaw1_vel.name].position = 0
            # god_map.get_world().state[brumbrum_joint.yaw2_name].position = angle
            god_map.world.state[brumbrum_joint.yaw.name].position = angle
        else:
            god_map.world.state[brumbrum_joint.yaw.name].position = angle
        god_map.world.notify_state_change()
        self.connect_monitors_to_all_tasks(start_condition, hold_condition, end_condition)


class JointVelocityLimit(Goal):
    def __init__(self,
                 joint_names: List[str],
                 group_name: Optional[str] = None,
                 weight: float = WEIGHT_BELOW_CA,
                 max_velocity: float = 1,
                 hard: bool = False,
                 name: Optional[str] = None,
                 start_monitors: Optional[List[ExpressionMonitor]] = None,
                 hold_monitors: Optional[List[ExpressionMonitor]] = None,
                 end_monitors: Optional[List[ExpressionMonitor]] = None):
        """
        Limits the joint velocity of a revolute joint.
        :param joint_name:
        :param group_name: if joint_name is not unique, will search in this group for matches.
        :param weight:
        :param max_velocity: rad/s
        :param hard: turn this into a hard constraint.
        """
        self.weight = weight
        self.max_velocity = max_velocity
        self.hard = hard
        self.joint_names = joint_names
        if name is None:
            name = f'{self.__class__.__name__}/{self.joint_names}'
        super().__init__(name)

        task = self.create_and_add_task('joint vel limit')
        for joint_name in self.joint_names:
            joint_name = god_map.world.search_for_joint_name(joint_name, group_name)
            joint: OneDofJoint = god_map.world.joints[joint_name]
            current_joint = joint.get_symbol(Derivatives.position)
            try:
                limit_expr = joint.get_limit_expressions(Derivatives.velocity)[1]
                max_velocity = cas.min(self.max_velocity, limit_expr)
            except IndexError:
                max_velocity = self.max_velocity
            if self.hard:
                task.add_velocity_constraint(lower_velocity_limit=-max_velocity,
                                             upper_velocity_limit=max_velocity,
                                             weight=self.weight,
                                             task_expression=current_joint,
                                             velocity_limit=max_velocity,
                                             lower_slack_limit=0,
                                             upper_slack_limit=0)
            else:
                task.add_velocity_constraint(lower_velocity_limit=-max_velocity,
                                             upper_velocity_limit=max_velocity,
                                             weight=self.weight,
                                             task_expression=current_joint,
                                             velocity_limit=max_velocity)
        self.connect_monitors_to_all_tasks(start_monitors, hold_monitors, end_monitors)


class AvoidJointLimits(Goal):
    def __init__(self,
                 percentage: float = 15,
                 joint_list: Optional[List[str]] = None,
                 group_name: Optional[str] = None,
                 weight: float = WEIGHT_BELOW_CA,
                 name: Optional[str] = None,
                 start_monitors: Optional[List[ExpressionMonitor]] = None,
                 hold_monitors: Optional[List[ExpressionMonitor]] = None,
                 end_monitors: Optional[List[ExpressionMonitor]] = None
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
        self.connect_monitors_to_all_tasks(start_monitors, hold_monitors, end_monitors)


class JointPositionList(Goal):
    def __init__(self,
                 goal_state: Dict[str, float],
                 group_name: Optional[str] = None,
                 weight: float = WEIGHT_BELOW_CA,
                 max_velocity: float = 1,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.TrueSymbol,
                 end_condition: cas.Expression = cas.TrueSymbol):
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

        self.connect_monitors_to_all_tasks(start_condition, hold_condition, end_condition)
