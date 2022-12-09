from __future__ import division

from typing import Dict, Optional, List

from geometry_msgs.msg import PoseStamped

from giskardpy import casadi_wrapper as w, identifier
from giskardpy.configs.default_giskard import ControlModes
from giskardpy.exceptions import ConstraintException, ConstraintInitalizationException
from giskardpy.goals.goal import Goal, WEIGHT_BELOW_CA, NonMotionGoal
from giskardpy.model.joints import OmniDrive, DiffDrive
from giskardpy.my_types import PrefixName
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
        if self.god_map.get_data(identifier.execute) \
                and self.god_map.get_data(identifier.control_mode) != ControlModes.stand_alone:
            raise ConstraintInitalizationException(f'It is not allowed to combine {str(self)} with plan and execute.')
        for joint_name, initial_joint_value in seed_configuration.items():
            joint_name = self.world.get_joint_name(joint_name, group_name)
            if joint_name not in self.world.state:
                raise KeyError(f'World has no joint \'{joint_name}\'.')
            self.world.state[joint_name].position = initial_joint_value
        self.world.notify_state_change()

    def __str__(self) -> str:
        return f'{str(self.__class__.__name__)}/{list(self.seed_configuration.keys())}'


class SetOdometry(NonMotionGoal):
    def __init__(self, group_name: str, base_pose: PoseStamped):
        super().__init__()
        self.group_name = group_name
        if self.god_map.get_data(identifier.execute) \
                and self.god_map.get_data(identifier.control_mode) != ControlModes.stand_alone:
            raise ConstraintInitalizationException(f'It is not allowed to combine {str(self)} with plan and execute.')
        brumbrum_joint_name = self.world.groups[group_name].root_link.parent_joint_name
        brumbrum_joint = self.world._joints[brumbrum_joint_name]
        if not isinstance(brumbrum_joint, (OmniDrive, DiffDrive)):
            raise ConstraintInitalizationException(f'Group {group_name} has no odometry joint.')
        base_pose = self.transform_msg(brumbrum_joint.parent_link_name, base_pose).pose
        self.world.state[brumbrum_joint.x_name].position = base_pose.position.x
        self.world.state[brumbrum_joint.y_name].position = base_pose.position.y
        axis, angle = axis_angle_from_quaternion(base_pose.orientation.x,
                                                 base_pose.orientation.y,
                                                 base_pose.orientation.z,
                                                 base_pose.orientation.w)
        if axis[-1] < 0:
            angle = -angle
        self.world.state[brumbrum_joint.yaw_name].position = angle
        self.world.notify_state_change()

    def __str__(self) -> str:
        return f'{str(self.__class__.__name__)}/{self.group_name}'


class JointPositionContinuous(Goal):

    def __init__(self,
                 joint_name: str,
                 goal: float,
                 group_name: str = None,
                 weight: float = WEIGHT_BELOW_CA,
                 max_velocity: float = 1,
                 hard: bool = False):
        """
        Use JointPosition or JointPositionList instead.
        This goal will move a continuous joint to a goal position.
        :param joint_name:
        :param goal: goal position
        :param group_name: if joint_name is not unique, search in this group for matches.
        :param weight:
        :param max_velocity: rad/s
        :param hard: turn this into a hard constraint.
        """
        self.joint_goal = goal
        self.weight = weight
        self.max_velocity = max_velocity
        self.hard = hard
        super().__init__()
        self.joint_name = self.world.get_joint_name(joint_name, group_name)
        if not self.world.is_joint_continuous(self.joint_name):
            raise ConstraintException(f'{self.__class__.__name__} called with non continuous joint {joint_name}')

    def make_constraints(self):
        current_joint = self.get_joint_position_symbol(self.joint_name)
        max_velocity = w.min(self.max_velocity,
                             self.world.get_joint_velocity_limits(self.joint_name)[1])

        error = w.shortest_angular_distance(current_joint, self.joint_goal)

        if self.hard:
            self.add_constraint(reference_velocity=max_velocity,
                                lower_error=error,
                                upper_error=error,
                                weight=self.weight,
                                task_expression=current_joint,
                                lower_slack_limit=0,
                                upper_slack_limit=0)
        else:
            self.add_constraint(reference_velocity=max_velocity,
                                lower_error=error,
                                upper_error=error,
                                weight=self.weight,
                                task_expression=current_joint)

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.joint_name}'


class JointPositionPrismatic(Goal):
    def __init__(self,
                 joint_name: str,
                 goal: float,
                 group_name: str = None,
                 weight: float = WEIGHT_BELOW_CA,
                 max_velocity: float = 1,
                 hard: bool = False):
        """
        Use JointPosition or JointPositionList instead.
        Moves a prismatic joint to a goal position.
        :param joint_name:
        :param goal:
        :param group_name: if joint_name is not unique, will search in this group for matches
        :param weight:
        :param max_velocity: m/s
        :param hard: turn this into a hard constraint
        """
        self.goal = goal
        self.weight = weight
        self.max_velocity = max_velocity
        self.hard = hard
        super().__init__()
        self.joint_name = self.world.get_joint_name(joint_name, group_name)
        if not self.world.is_joint_prismatic(self.joint_name):
            raise ConstraintException(f'{self.__class__.__name__} called with non prismatic joint {joint_name}')

    def make_constraints(self):
        current_joint = self.get_joint_position_symbol(self.joint_name)

        try:
            limit_expr = self.world.get_joint_velocity_limits(self.joint_name)[1]
            max_velocity = w.min(self.max_velocity,
                                 limit_expr)
        except IndexError:
            max_velocity = self.max_velocity

        error = self.goal - current_joint

        if self.hard:
            self.add_constraint(reference_velocity=max_velocity,
                                lower_error=error,
                                upper_error=error,
                                weight=self.weight,
                                task_expression=current_joint,
                                upper_slack_limit=0,
                                lower_slack_limit=0)
        else:
            self.add_constraint(reference_velocity=max_velocity,
                                lower_error=error,
                                upper_error=error,
                                weight=self.weight,
                                task_expression=current_joint)

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.joint_name}'


class JointVelocityRevolute(Goal):
    def __init__(self,
                 joint_name: str,
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
        super().__init__()
        self.joint_name = self.world.get_joint_name(joint_name, group_name)
        if not self.world.is_joint_revolute(self.joint_name):
            raise ConstraintException(f'{self.__class__.__name__} called with non revolute joint {joint_name}')

    def make_constraints(self):
        current_joint = self.get_joint_position_symbol(self.joint_name)

        try:
            limit_expr = self.world.get_joint_velocity_limits(self.joint_name)[1]
            max_velocity = w.min(self.max_velocity,
                                 limit_expr)
        except IndexError:
            max_velocity = self.max_velocity

        if self.hard:
            self.add_velocity_constraint(lower_velocity_limit=-max_velocity,
                                         upper_velocity_limit=max_velocity,
                                         weight=self.weight,
                                         task_expression=current_joint,
                                         velocity_limit=max_velocity,
                                         lower_slack_limit=0,
                                         upper_slack_limit=0)
        else:
            self.add_velocity_constraint(lower_velocity_limit=-max_velocity,
                                         upper_velocity_limit=max_velocity,
                                         weight=self.weight,
                                         task_expression=current_joint,
                                         velocity_limit=max_velocity)

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.joint_name}'


class JointPositionRevolute(Goal):
    def __init__(self,
                 joint_name: str,
                 goal: float,
                 group_name: str = None,
                 weight: float = WEIGHT_BELOW_CA,
                 max_velocity: float = 1,
                 hard: bool = False):
        """
        Use JointPosition or JointPositionList instead.
        Moves a revolute joint to a goal pose.
        :param joint_name:
        :param goal:
        :param group_name: if joint_name is not unique, will search in this group for matches.
        :param weight:
        :param max_velocity: rad/s
        :param hard: turn this into a hard constraint.
        """
        self.goal = goal
        self.weight = weight
        self.max_velocity = max_velocity
        self.hard = hard
        super().__init__()
        self.joint_name = self.world.get_joint_name(joint_name, group_name)
        if not self.world.is_joint_revolute(self.joint_name):
            raise ConstraintException(f'{self.__class__.__name__} called with non revolute joint {joint_name}')

    @profile
    def make_constraints(self):
        current_joint = self.get_joint_position_symbol(self.joint_name)

        joint_goal = self.goal
        weight = self.weight

        max_velocity = w.min(self.max_velocity,
                             self.world.get_joint_velocity_limits(self.joint_name)[1])

        error = joint_goal - current_joint
        if self.hard:
            self.add_constraint(reference_velocity=max_velocity,
                                lower_error=error,
                                upper_error=error,
                                weight=weight,
                                task_expression=current_joint,
                                upper_slack_limit=0,
                                lower_slack_limit=0)
        else:
            self.add_constraint(reference_velocity=max_velocity,
                                lower_error=error,
                                upper_error=error,
                                weight=weight,
                                task_expression=current_joint)

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.joint_name}'


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
        self.joint_name = self.world.get_joint_name(joint_name, group_name)
        if not self.world.is_joint_revolute(self.joint_name) and not self.world.is_joint_prismatic(joint_name):
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

        time = self.god_map.to_symbol(identifier.time)
        time_in_secs = self.sample_period * time

        max_velocity = w.min(self.max_velocity,
                             self.world.get_joint_velocity_limits(self.joint_name)[1])

        fun_params = frequency * 2.0 * w.pi * time_in_secs
        err = (joint_goal - current_joint) + noise_amplitude * max_velocity * w.sin(fun_params)
        capped_err = w.limit(err, -noise_amplitude * max_velocity, noise_amplitude * max_velocity)

        self.add_constraint(lower_error=capped_err,
                            upper_error=capped_err,
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
        self.joint_name = self.world.get_joint_name(joint_name, group_name)
        if not self.world.is_joint_continuous(self.joint_name):
            raise ConstraintException(f'{self.__class__.__name__} called with non continuous joint {joint_name}')

    def make_constraints(self):
        current_joint = self.get_joint_position_symbol(self.joint_name)
        frequency = self.frequency
        noise_amplitude = self.noise_amplitude
        joint_goal = self.goal
        weight = self.weight

        time = self.god_map.to_symbol(identifier.time)
        time_in_secs = self.sample_period * time

        max_velocity = w.min(self.max_velocity,
                             self.world.get_joint_velocity_limits(self.joint_name)[1])

        fun_params = frequency * 2.0 * w.pi * time_in_secs
        err = w.shortest_angular_distance(current_joint, joint_goal) + noise_amplitude * max_velocity * w.sin(
            fun_params)

        capped_err = w.limit(err, -noise_amplitude * max_velocity, noise_amplitude * max_velocity)

        self.add_constraint(lower_error=capped_err,
                            upper_error=capped_err,
                            reference_velocity=max_velocity,
                            weight=weight,
                            task_expression=current_joint)

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.joint_name}'


class AvoidSingleJointLimits(Goal):
    def __init__(self,
                 joint_name,
                 group_name: Optional[str] = None,
                 weight: float = 0.1,
                 max_linear_velocity: float = 100,
                 percentage: float = 5):
        """
        This goal will push revolute joints away from their position limits
        :param joint_name:
        :param group_name: if joint_name is not unique, will search in this group for matches.
        :param weight:
        :param max_linear_velocity: m/s for prismatic joints, rad/s for revolute joints
        :param percentage: default 15, if limits are 0-100, the constraint will push into the 15-85 range
        """
        self.weight = weight
        self.max_velocity = max_linear_velocity
        self.percentage = percentage
        super().__init__()
        self.joint_name = self.world.get_joint_name(joint_name, group_name)
        if not self.world.is_joint_revolute(self.joint_name) and not self.world.is_joint_prismatic(self.joint_name):
            raise ConstraintException(
                f'{self.__class__.__name__} called with non prismatic or revolute joint {joint_name}')

    def make_constraints(self):
        weight = self.weight
        joint_symbol = self.get_joint_position_symbol(self.joint_name)
        percentage = self.percentage / 100.
        lower_limit, upper_limit = self.world.get_joint_position_limits(self.joint_name)
        max_velocity = self.max_velocity
        max_velocity = w.min(max_velocity,
                             self.world.get_joint_velocity_limits(self.joint_name)[1])

        joint_range = upper_limit - lower_limit
        center = (upper_limit + lower_limit) / 2.

        max_error = joint_range / 2. * percentage

        upper_goal = center + joint_range / 2. * (1 - percentage)
        lower_goal = center - joint_range / 2. * (1 - percentage)

        upper_err = upper_goal - joint_symbol
        lower_err = lower_goal - joint_symbol

        error = w.max(w.abs(w.min(upper_err, 0)), w.abs(w.max(lower_err, 0)))
        weight = weight * (error / max_error)

        self.add_constraint(reference_velocity=max_velocity,
                            lower_error=lower_err,
                            upper_error=upper_err,
                            weight=weight,
                            task_expression=joint_symbol)

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
        if joint_list is not None:
            for joint_name in joint_list:
                joint_name = self.world.get_joint_name(joint_name, group_name)
                if self.world.is_joint_prismatic(joint_name) or self.world.is_joint_revolute(joint_name):
                    self.add_constraints_of_goal(AvoidSingleJointLimits(joint_name=joint_name.short_name,
                                                                        group_name=group_name,
                                                                        percentage=percentage,
                                                                        weight=weight))
        else:
            joint_list = self.god_map.get_data(identifier.controlled_joints)
            for joint_name in joint_list:
                try:
                    group_name = self.world.get_group_of_joint(joint_name).name
                except KeyError:
                    child_link = self.world._joints[joint_name].child_link_name
                    group_name = self.world._get_group_name_containing_link(child_link)
                if self.world.is_joint_prismatic(joint_name) or self.world.is_joint_revolute(joint_name):
                    self.add_constraints_of_goal(AvoidSingleJointLimits(joint_name=joint_name.short_name,
                                                                        group_name=group_name,
                                                                        percentage=percentage,
                                                                        weight=weight))

    def make_constraints(self):
        pass

    def __str__(self) -> str:
        return f'{super().__str__()}/{self.joint_list}'


class JointPositionList(Goal):
    def __init__(self,
                 goal_state: Dict[str, float],
                 group_name: Optional[str] = None,
                 weight: Optional[float] = None,
                 max_velocity: Optional[float] = None,
                 hard: bool = False):
        """
        Calls JointPosition for a list of joints.
        :param goal_state: maps joint_name to goal position
        :param group_name: if joint_name is not unique, search in this group for matches.
        :param weight:
        :param max_velocity: will be applied to all joints, you should group joint types, e.g., prismatic joints
        :param hard: turns this into a hard constraint.
        """
        super().__init__()
        self.joint_names = list(goal_state.keys())
        if len(goal_state) == 0:
            raise ConstraintInitalizationException(f'Can\'t initialize {self} with no joints.')
        for joint_name, goal_position in goal_state.items():
            params = {'joint_name': joint_name,
                      'group_name': group_name,
                      'goal': goal_position}
            if weight is not None:
                params['weight'] = weight
            if max_velocity is not None:
                params['max_velocity'] = max_velocity
            params['hard'] = hard
            self.add_constraints_of_goal(JointPosition(**params))

    def make_constraints(self):
        pass

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.joint_names}'


class JointPosition(Goal):
    def __init__(self,
                 joint_name: str,
                 goal: float,
                 group_name: Optional[str] = None,
                 weight: float = WEIGHT_BELOW_CA,
                 max_velocity: float = 100,
                 hard: bool = False):
        """
        Moves joint_name to goal.
        :param joint_name:
        :param goal:
        :param group_name: if joint_name is not unique, search in this group for matches.
        :param weight:
        :param max_velocity: m/s for prismatic joints, rad/s for revolute or continuous joints, limited by urdf
        """
        super().__init__()
        self.joint_name = self.world.get_joint_name(joint_name, group_name)
        if self.world.is_joint_continuous(self.joint_name):
            C = JointPositionContinuous
        elif self.world.is_joint_revolute(self.joint_name):
            C = JointPositionRevolute
        elif self.world.is_joint_prismatic(self.joint_name):
            C = JointPositionPrismatic
        else:
            raise ConstraintInitalizationException(f'\'{joint_name}\' has to be continuous, revolute or prismatic')
        self.add_constraints_of_goal(C(joint_name=joint_name,
                                       group_name=group_name,
                                       goal=goal,
                                       weight=weight,
                                       max_velocity=max_velocity,
                                       hard=hard))

    def make_constraints(self):
        pass

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.joint_name}'


class JointPositionRange(Goal):
    def __init__(self,
                 joint_name: str,
                 upper_limit: float,
                 lower_limit: float,
                 group_name: Optional[str] = None,
                 hard: bool = False):
        """
        Sets artificial joint limits.
        :param joint_name:
        :param upper_limit:
        :param lower_limit:
        :param group_name: if joint_name is not unique, search in this group for matches.
        :param hard: turn this into a hard constraint
        """
        super().__init__()
        self.joint_name = self.world.get_joint_name(joint_name, group_name)
        if self.world.is_joint_continuous(self.joint_name):
            raise NotImplementedError(f'Can\'t limit range of continues joint \'{self.joint_name}\'.')
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.hard = hard
        if self.hard:
            current_position = self.world.state[self.joint_name].position
            if current_position > self.upper_limit + 2e-3 or current_position < self.lower_limit - 2e-3:
                raise ConstraintInitalizationException(f'{self.joint_name} out of set limits. '
                                                       '{self.lower_limit} <= {current_position} <= {self.upper_limit} '
                                                       'is not true.')

    def make_constraints(self):
        joint_position = self.get_joint_position_symbol(self.joint_name)
        if self.hard:
            self.add_constraint(reference_velocity=self.world.get_joint_velocity_limits(self.joint_name)[1],
                                lower_error=self.lower_limit - joint_position,
                                upper_error=self.upper_limit - joint_position,
                                weight=WEIGHT_BELOW_CA,
                                task_expression=joint_position,
                                lower_slack_limit=0,
                                upper_slack_limit=0)
        else:
            self.add_constraint(reference_velocity=self.world.get_joint_velocity_limits(self.joint_name)[1],
                                lower_error=self.lower_limit - joint_position,
                                upper_error=self.upper_limit - joint_position,
                                weight=WEIGHT_BELOW_CA,
                                task_expression=joint_position)

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.joint_name}'
