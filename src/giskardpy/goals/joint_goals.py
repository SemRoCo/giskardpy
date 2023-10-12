from __future__ import division

from typing import Dict, Optional, List

from geometry_msgs.msg import PoseStamped

from giskardpy import casadi_wrapper as cas
from giskardpy.god_map import god_map
from giskardpy.symbol_manager import symbol_manager
from giskardpy.exceptions import ConstraintException, ConstraintInitalizationException
from giskardpy.goals.goal import Goal, NonMotionGoal
from giskardpy.goals.tasks.task import WEIGHT_BELOW_CA, WEIGHT_ABOVE_CA, WEIGHT_COLLISION_AVOIDANCE, Task
from giskardpy.model.joints import OmniDrive, DiffDrive, OmniDrivePR22, OneDofJoint
from giskardpy.my_types import PrefixName, Derivatives
from giskardpy.utils.math import axis_angle_from_quaternion


class SetSeedConfiguration(NonMotionGoal):
    def __init__(self,
                 seed_configuration: Dict[str, float],
                 group_name: Optional[str] = None):
        """
        Overwrite the configuration of the world to allow starting the planning from a different state.
        Can only be used in plan only mode.
        :param seed_configuration: maps joint name to float
        :param group_name: if joint names are not unique, it will search in this group for matches.
        """
        self.seed_configuration = seed_configuration
        super().__init__()
        if group_name is not None:
            seed_configuration = {PrefixName(joint_name, group_name): v for joint_name, v in seed_configuration.items()}
        if god_map.is_goal_msg_type_execute() and not god_map.is_standalone():
            raise ConstraintInitalizationException(f'It is not allowed to combine {str(self)} with plan and execute.')
        for joint_name, initial_joint_value in seed_configuration.items():
            joint_name = god_map.world.search_for_joint_name(joint_name, group_name)
            if joint_name not in god_map.world.state:
                raise KeyError(f'World has no joint \'{joint_name}\'.')
            god_map.world.state[joint_name].position = initial_joint_value
        god_map.world.notify_state_change()

    def __str__(self) -> str:
        return f'{str(self.__class__.__name__)}/{list(self.seed_configuration.keys())}'


class SetOdometry(NonMotionGoal):
    def __init__(self, group_name: str, base_pose: PoseStamped):
        super().__init__()
        self.group_name = group_name
        if god_map.is_goal_msg_type_execute() and not god_map.is_standalone():
            raise ConstraintInitalizationException(f'It is not allowed to combine {str(self)} with plan and execute.')
        brumbrum_joint_name = god_map.world.groups[group_name].root_link.child_joint_names[0]
        brumbrum_joint = god_map.world.joints[brumbrum_joint_name]
        if not isinstance(brumbrum_joint, (OmniDrive, DiffDrive, OmniDrivePR22)):
            raise ConstraintInitalizationException(f'Group {group_name} has no odometry joint.')
        base_pose = self.transform_msg(brumbrum_joint.parent_link_name, base_pose).pose
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

    def __str__(self) -> str:
        return f'{str(self.__class__.__name__)}/{self.group_name}'


class JointVelocityLimit(Goal):
    def __init__(self,
                 joint_names: List[str],
                 group_name: Optional[str] = None,
                 weight: float = WEIGHT_BELOW_CA,
                 max_velocity: float = 1,
                 hard: bool = False):
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
        super().__init__()

        task = Task(name='joint vel limit')
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
        self.add_task(task)

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.joint_names}'


class ShakyJointPositionRevoluteOrPrismatic(Goal):
    def __init__(self, joint_name, goal, frequency, group_name: str = None, noise_amplitude=1.0, weight=WEIGHT_BELOW_CA,
                 max_velocity=1):
        """
        This goal will move a revolute or prismatic joint to the goal position and shake the joint with the given
        frequency.
        :param joint_name: str
        :param goal: float
        :param frequency: float
        :param noise_amplitude: float
        :param weight: float, default WEIGHT_BELOW_CA
        :param max_velocity: float, rad/s, default 3451, meaning the urdf/config limits are active
        """
        super().__init__()
        self.joint_name = god_map.world.search_for_joint_name(joint_name, group_name)
        if not god_map.world.is_joint_revolute(self.joint_name) and not god_map.world.is_joint_prismatic(joint_name):
            raise ConstraintException(
                f'{self.__class__.__name__} called with non revolute/prismatic joint {joint_name}')

        self.goal = goal
        self.frequency = frequency
        self.noise_amplitude = noise_amplitude
        self.weight = weight
        self.max_velocity = max_velocity

    def make_constraints(self):
        current_joint = self.get_joint_position_symbol(self.joint_name)
        frequency = self.frequency
        noise_amplitude = self.noise_amplitude
        joint_goal = self.goal
        weight = self.weight

        time = symbol_manager.time
        time_in_secs = god_map.qp_controller_config.sample_period * time

        max_velocity = cas.min(self.max_velocity,
                               god_map.world.get_joint_velocity_limits(self.joint_name)[1])

        fun_params = frequency * 2.0 * cas.pi * time_in_secs
        err = (joint_goal - current_joint) + noise_amplitude * max_velocity * cas.sin(fun_params)
        capped_err = cas.limit(err, -noise_amplitude * max_velocity, noise_amplitude * max_velocity)

        self.add_equality_constraint(equality_bound=capped_err,
                                     reference_velocity=max_velocity,
                                     weight=weight,
                                     task_expression=current_joint)

    def __str__(self):
        s = super(ShakyJointPositionRevoluteOrPrismatic, self).__str__()
        return f'{s}/{self.joint_name}'


class ShakyJointPositionContinuous(Goal):
    def __init__(self, joint_name, goal, frequency, group_name: str = None, noise_amplitude=10, weight=WEIGHT_BELOW_CA,
                 max_velocity=1):
        """
        This goal will move a continuous joint to the goal position and shake the joint with the given frequency.
        :param joint_name: str
        :param goal: float
        :param frequency: float
        :param noise_amplitude: float
        :param weight: float, default WEIGHT_BELOW_CA
        :param max_velocity: float, rad/s, default 3451, meaning the urdf/config limits are active
        """
        self.goal = goal
        self.frequency = frequency
        self.noise_amplitude = noise_amplitude
        self.weight = weight
        self.max_velocity = max_velocity
        super().__init__()
        self.joint_name = god_map.world.search_for_joint_name(joint_name, group_name)
        if not god_map.world.is_joint_continuous(self.joint_name):
            raise ConstraintException(f'{self.__class__.__name__} called with non continuous joint {joint_name}')

    def make_constraints(self):
        current_joint = self.get_joint_position_symbol(self.joint_name)
        frequency = self.frequency
        noise_amplitude = self.noise_amplitude
        joint_goal = self.goal
        weight = self.weight

        time = symbol_manager.time
        time_in_secs = god_map.qp_controller_config.sample_period * time

        max_velocity = cas.min(self.max_velocity,
                               god_map.world.get_joint_velocity_limits(self.joint_name)[1])

        fun_params = frequency * 2.0 * cas.pi * time_in_secs
        err = cas.shortest_angular_distance(current_joint, joint_goal) + noise_amplitude * max_velocity * cas.sin(
            fun_params)

        capped_err = cas.limit(err, -noise_amplitude * max_velocity, noise_amplitude * max_velocity)

        self.add_equality_constraint(equality_bound=capped_err,
                                     reference_velocity=max_velocity,
                                     weight=weight,
                                     task_expression=current_joint)

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.joint_name}'


class AvoidJointLimits(Goal):
    def __init__(self,
                 percentage: float = 15,
                 joint_list: Optional[List[str]] = None,
                 group_name: Optional[str] = None,
                 weight: float = WEIGHT_BELOW_CA):
        """
        Calls AvoidSingleJointLimits for each joint in joint_list
        :param percentage:
        :param joint_list: list of joints for which AvoidSingleJointLimits will be called
        :param weight:
        """
        self.joint_list = joint_list
        super().__init__()
        self.weight = weight
        self.percentage = percentage
        if joint_list is not None:
            joint_list = [god_map.world.search_for_joint_name(joint_name, group_name) for joint_name in joint_list]
        else:
            if group_name is None:
                joint_list = god_map.world.controlled_joints
            else:
                joint_list = god_map.world.groups[group_name].controlled_joints
        task = Task('avoid joint limits')
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
        self.add_task(task)

    def __str__(self) -> str:
        return f'{super().__str__()}/{self.joint_list}'


class JointPositionList(Goal):
    def __init__(self,
                 goal_state: Dict[str, float],
                 group_name: Optional[str] = None,
                 weight: float = WEIGHT_BELOW_CA,
                 max_velocity: float = 1):
        """
        Calls JointPosition for a list of joints.
        :param goal_state: maps joint_name to goal position
        :param group_name: if joint_name is not unique, search in this group for matches.
        :param weight:
        :param max_velocity: will be applied to all joints, you should group joint types, e.g., prismatic joints
        :param hard: turns this into a hard constraint.
        """
        super().__init__()
        self.current_positions = []
        self.goal_positions = []
        self.velocity_limits = []
        self.names = []
        self.joint_names = list(goal_state.keys())
        self.max_velocity = max_velocity
        self.weight = weight
        if len(goal_state) == 0:
            raise ConstraintInitalizationException(f'Can\'t initialize {self} with no joints.')
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

        task = Task(name='joint goal')
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

        self.add_task(task)

    def __str__(self):
        s = super().__str__()
        return f'{s} {self.joint_names}'
