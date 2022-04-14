from __future__ import division

from typing import Union, Dict

from sensor_msgs.msg import JointState

from giskardpy import casadi_wrapper as w, identifier
from giskardpy.data_types import PrefixName
from giskardpy.exceptions import ConstraintException, ConstraintInitalizationException
from giskardpy.goals.goal import Goal, WEIGHT_BELOW_CA


class JointPositionContinuous(Goal):

    def __init__(self, joint_name: str, group_name: str, goal: float, weight: float =WEIGHT_BELOW_CA,
                 max_velocity: float = 1, hard: bool = False, **kwargs):
        """
        This goal will move a continuous joint to the goal position
        :param joint_name: str
        :param goal: float
        :param weight: float, default WEIGHT_BELOW_CA
        :param max_velocity: float, rad/s, default 1423, meaning the urdf/config limits are active
        """
        super(JointPositionContinuous, self).__init__(**kwargs)
        prefix = self.world.groups[group_name].prefix
        self.joint_name = PrefixName(joint_name, prefix)
        if not self.world.is_joint_continuous(self.joint_name):
            raise ConstraintException(f'{self.__class__.__name__} called with non continuous joint {self.joint_name}')
        self.joint_goal = goal
        self.weight = weight
        self.max_velocity = max_velocity
        self.hard = hard

    def make_constraints(self):
        """
        example:
        name='JointPosition'
        parameter_value_pair='{
            "joint_name": "torso_lift_joint", #required
            "goal_position": 0, #required
            "weight": 1, #optional
            "max_velocity": 1 #optional -- rad/s or m/s depending on joint; can not go higher than urdf limit
        }'
        :return:
        """
        current_joint = self.get_joint_position_symbol(self.joint_name)
        max_velocity = w.min(self.get_parameter_as_symbolic_expression('max_velocity'),
                             self.world.joint_limit_expr(self.joint_name, 1)[1])

        error = w.shortest_angular_distance(current_joint, self.joint_goal)

        if self.hard:
            self.add_constraint(reference_velocity=max_velocity,
                                lower_error=error,
                                upper_error=error,
                                weight=self.weight,
                                expression=current_joint,
                                lower_slack_limit=0,
                                upper_slack_limit=0)
        else:
            self.add_constraint(reference_velocity=max_velocity,
                                lower_error=error,
                                upper_error=error,
                                weight=self.weight,
                                expression=current_joint)

    def __str__(self):
        s = super(JointPositionContinuous, self).__str__()
        return f'{s}/{self.joint_name}'


class JointPositionPrismatic(Goal):
    def __init__(self, joint_name: str, group_name: str, goal: float, weight: float = WEIGHT_BELOW_CA,
                 max_velocity: float = 1, hard: bool = False, prefix=None, **kwargs):
        """
        This goal will move a prismatic joint to the goal position
        :param weight: default WEIGHT_BELOW_CA
        :param max_velocity: m/s, default 4535, meaning the urdf/config limits are active
        """
        super(JointPositionPrismatic, self).__init__(**kwargs)
        prefix = self.world.groups[group_name].prefix
        self.joint_name = PrefixName(joint_name, prefix)
        if not self.world.is_joint_prismatic(self.joint_name):
            raise ConstraintException(f'{self.__class__.__name__} called with non prismatic joint {self.joint_name}')
        self.goal = goal
        self.weight = weight
        self.max_velocity = max_velocity
        self.hard = hard

    def make_constraints(self):
        """
        example:
        name='JointPosition'
        parameter_value_pair='{
            "joint_name": "torso_lift_joint", #required
            "goal_position": 0, #required
            "weight": 1, #optional
            "gain": 10, #optional -- error is multiplied with this value
            "max_speed": 1 #optional -- rad/s or m/s depending on joint; can not go higher than urdf limit
        }'
        :return:
        """
        current_joint = self.get_joint_position_symbol(self.joint_name)

        joint_goal = self.get_parameter_as_symbolic_expression('goal')
        weight = self.get_parameter_as_symbolic_expression('weight')

        try:
            # if self.world.is_joint_mimic(self.joint_name):
            #     mimed_joint_name = self.world.joints[self.joint_name].mimed_joint_name
            #     mimed_joint_symbol = self.get_joint_position_symbol(mimed_joint_name)
            #     mimied_limit = self.world.joint_limit_expr(self.joint_name, 1)[1]
            #     limit_expr = w.substitute(current_joint, mimed_joint_symbol, mimied_limit)
            # else:
            limit_expr = self.world.joint_limit_expr(self.joint_name, 1)[1]
            max_velocity = w.min(self.get_parameter_as_symbolic_expression('max_velocity'),
                                 limit_expr)
        except IndexError:
            max_velocity = self.get_parameter_as_symbolic_expression('max_velocity')

        error = joint_goal - current_joint

        if self.hard:
            self.add_constraint(reference_velocity=max_velocity,
                                lower_error=error,
                                upper_error=error,
                                weight=weight,
                                expression=current_joint,
                                upper_slack_limit=0,
                                lower_slack_limit=0)
        else:
            self.add_constraint(reference_velocity=max_velocity,
                                lower_error=error,
                                upper_error=error,
                                weight=weight,
                                expression=current_joint)

    def __str__(self):
        s = super(JointPositionPrismatic, self).__str__()
        return f'{s}/{self.joint_name}'


class JointPositionRevolute(Goal):

    def __init__(self, joint_name: str, group_name: str, goal: float, weight: float = WEIGHT_BELOW_CA,
                 max_velocity: float = 1, hard: bool = False, **kwargs):
        """
        This goal will move a revolute joint to the goal position
        :param joint_name: str
        :param goal: float
        :param weight: float, default WEIGHT_BELOW_CA
        :param max_velocity: float, rad/s, default 3451, meaning the urdf/config limits are active
        """
        super(JointPositionRevolute, self).__init__(**kwargs)
        prefix = self.world.groups[group_name].prefix
        self.joint_name = PrefixName(joint_name, prefix)
        if not self.world.is_joint_revolute(self.joint_name):
            raise ConstraintException(f'{self.__class__.__name__} called with non revolute joint {self.joint_name}')
        self.goal = goal
        self.weight = weight
        self.max_velocity = max_velocity
        self.hard = hard

    def make_constraints(self):
        """
        example:
        name='JointPosition'
        parameter_value_pair='{
            "joint_name": "torso_lift_joint", #required
            "goal_position": 0, #required
            "weight": 1, #optional
            "gain": 10, #optional -- error is multiplied with this value
            "max_speed": 1 #optional -- rad/s or m/s depending on joint; can not go higher than urdf limit
        }'
        :return:
        """
        current_joint = self.get_joint_position_symbol(self.joint_name)

        joint_goal = self.get_parameter_as_symbolic_expression('goal')
        weight = self.get_parameter_as_symbolic_expression('weight')

        max_velocity = w.min(self.get_parameter_as_symbolic_expression('max_velocity'),
                             self.world.joint_limit_expr(self.joint_name, 1)[1])

        error = joint_goal - current_joint
        self.add_debug_expr('error', error)
        if self.hard:
            self.add_constraint(reference_velocity=max_velocity,
                                lower_error=error,
                                upper_error=error,
                                weight=weight,
                                expression=current_joint,
                                upper_slack_limit=0,
                                lower_slack_limit=0)
        else:
            self.add_constraint(reference_velocity=max_velocity,
                                lower_error=error,
                                upper_error=error,
                                weight=weight,
                                expression=current_joint)

    def __str__(self):
        s = super(JointPositionRevolute, self).__str__()
        return f'{s}/{self.joint_name}'


class ShakyJointPositionRevoluteOrPrismatic(Goal):
    def __init__(self, joint_name, group_name, goal, frequency, noise_amplitude=1.0, weight=WEIGHT_BELOW_CA,
                 max_velocity=1, **kwargs):
        """
        This goal will move a revolute or prismatic joint to the goal position and shake the joint with the given frequency.
        :param joint_name: str
        :param goal: float
        :param frequency: float
        :param noise_amplitude: float
        :param weight: float, default WEIGHT_BELOW_CA
        :param max_velocity: float, rad/s, default 3451, meaning the urdf/config limits are active
        """
        super(ShakyJointPositionRevoluteOrPrismatic, self).__init__(**kwargs)
        prefix = self.world.groups[group_name].prefix
        self.joint_name = PrefixName(joint_name, prefix)
        if not self.world.is_joint_revolute(joint_name) and not self.world.is_joint_prismatic(joint_name):
            raise ConstraintException(f'{self.__class__.__name__} called with non revolute/prismatic joint {joint_name}')

        self.goal = goal
        self.frequency = frequency
        self.noise_amplitude = noise_amplitude
        self.weight = weight
        self.max_velocity = max_velocity

    def make_constraints(self):
        """
        example:
        name='ShakyJointPositionRevoluteOrPrismatic'
        parameter_value_pair='{
            "joint_name": "r_wrist_flex_joint", #required
            "goal_position": -1.0, #required
            "frequency": 5.0, #required
            "weight": 1, #optional
            "max_velocity": 1 #optional -- rad/s or m/s depending on joint; can not go higher than urdf limit
        }'
        :return:
        """
        current_joint = self.get_joint_position_symbol(self.joint_name)
        frequency = self.get_parameter_as_symbolic_expression('frequency')
        noise_amplitude = self.get_parameter_as_symbolic_expression('noise_amplitude')
        joint_goal = self.get_parameter_as_symbolic_expression('goal')
        weight = self.get_parameter_as_symbolic_expression('weight')

        time = self.god_map.to_symbol(identifier.time)
        time_in_secs = self.get_sampling_period_symbol() * time

        max_velocity = w.min(self.get_parameter_as_symbolic_expression('max_velocity'),
                             self.world.joint_limit_expr(self.joint_name, 1)[1])

        fun_params = frequency * 2.0 * w.pi * time_in_secs
        err = (joint_goal - current_joint) + noise_amplitude * max_velocity * w.sin(fun_params)
        capped_err = w.limit(err, -noise_amplitude * max_velocity, noise_amplitude * max_velocity)

        self.add_constraint(lower_error=capped_err,
                            upper_error=capped_err,
                            reference_velocity=max_velocity,
                            weight=weight,
                            expression=current_joint)

    def __str__(self):
        s = super(ShakyJointPositionRevoluteOrPrismatic, self).__str__()
        return f'{s}/{self.joint_name}'


class ShakyJointPositionContinuous(Goal):
    def __init__(self, joint_name, group_name, goal, frequency, noise_amplitude=10, weight=WEIGHT_BELOW_CA,
                 max_velocity=1, **kwargs):
        """
        This goal will move a continuous joint to the goal position and shake the joint with the given frequency.
        :param joint_name: str
        :param goal: float
        :param frequency: float
        :param noise_amplitude: float
        :param weight: float, default WEIGHT_BELOW_CA
        :param max_velocity: float, rad/s, default 3451, meaning the urdf/config limits are active
        """
        super(ShakyJointPositionContinuous, self).__init__(**kwargs)
        prefix = self.world.groups[group_name].prefix
        self.joint_name = PrefixName(joint_name, prefix)
        self.goal = goal
        self.frequency = frequency
        self.noise_amplitude = noise_amplitude
        self.weight = weight
        self.max_velocity = max_velocity
        if not self.world.is_joint_continuous(joint_name):
            raise ConstraintException(f'{self.__class__.__name__} called with non continuous joint {joint_name}')

    def make_constraints(self):
        """
        example:
        name='JointPosition'
        parameter_value_pair='{
            "joint_name": "l_wrist_roll_joint", #required
            "goal_position": -5.0, #required
            "frequency": 5.0, #required
            "weight": 1, #optional
            "max_velocity": 1 #optional -- rad/s or m/s depending on joint; can not go higher than urdf limit
        }'
        :return:
        """
        current_joint = self.get_joint_position_symbol(self.joint_name)
        frequency = self.get_parameter_as_symbolic_expression('frequency')
        noise_amplitude = self.get_parameter_as_symbolic_expression('noise_amplitude')
        joint_goal = self.get_parameter_as_symbolic_expression('goal')
        weight = self.get_parameter_as_symbolic_expression('weight')

        time = self.god_map.to_symbol(identifier.time)
        time_in_secs = self.get_sampling_period_symbol() * time

        max_velocity = w.min(self.get_parameter_as_symbolic_expression('max_velocity'),
                             self.world.joint_limit_expr(self.joint_name, 1)[1])

        fun_params = frequency * 2.0 * w.pi * time_in_secs
        err = w.shortest_angular_distance(current_joint, joint_goal) + noise_amplitude * max_velocity * w.sin(
            fun_params)

        capped_err = w.limit(err, -noise_amplitude * max_velocity, noise_amplitude * max_velocity)

        self.add_constraint(lower_error=capped_err,
                            upper_error=capped_err,
                            reference_velocity=max_velocity,
                            weight=weight,
                            expression=current_joint)

    def __str__(self):
        s = super(ShakyJointPositionContinuous, self).__str__()
        return f'{s}/{self.joint_name}'


class AvoidJointLimitsRevolute(Goal):
    def __init__(self, joint_name, group_name, weight=0.1, max_linear_velocity=100, percentage=5, **kwargs):
        """
        This goal will push revolute joints away from their position limits
        :param joint_name: str
        :param weight: float, default WEIGHT_BELOW_CA
        :param max_linear_velocity: float, default 1e9, meaning the urdf/config limit will kick in
        :param percentage: float, default 15, if limits are 0-100, the constraint will push into the 15-85 range
        """
        super(AvoidJointLimitsRevolute, self).__init__(**kwargs)
        prefix = self.world.groups[group_name].prefix
        self.joint_name = PrefixName(joint_name, prefix)
        self.weight = weight
        self.max_velocity = max_linear_velocity
        self.percentage = percentage
        if not self.world.is_joint_revolute(joint_name):
            raise ConstraintException(f'{self.__class__.__name__} called with non prismatic joint {joint_name}')

    def make_constraints(self):
        weight = self.get_parameter_as_symbolic_expression('weight')
        joint_symbol = self.get_joint_position_symbol(self.joint_name)
        percentage = self.get_parameter_as_symbolic_expression('percentage') / 100.
        lower_limit, upper_limit = self.world.get_joint_position_limits(self.joint_name)
        max_velocity = self.get_parameter_as_symbolic_expression('max_velocity')
        max_velocity = w.min(max_velocity,
                             self.world.joint_limit_expr(self.joint_name, 1)[1])

        joint_range = upper_limit - lower_limit
        center = (upper_limit + lower_limit) / 2.

        max_error = joint_range / 2. * percentage

        upper_goal = center + joint_range / 2. * (1 - percentage)
        lower_goal = center - joint_range / 2. * (1 - percentage)

        upper_err = upper_goal - joint_symbol
        lower_err = lower_goal - joint_symbol

        # upper_err_capped = self.limit_velocity(upper_err, max_velocity)
        # lower_err_capped = self.limit_velocity(lower_err, max_velocity)

        error = w.max(w.abs(w.min(upper_err, 0)), w.abs(w.max(lower_err, 0)))
        weight = weight * (error / max_error)

        self.add_constraint(reference_velocity=max_velocity,
                            lower_error=lower_err,
                            upper_error=upper_err,
                            weight=weight,
                            expression=joint_symbol)

    def __str__(self):
        s = super(AvoidJointLimitsRevolute, self).__str__()
        return f'{s}/{self.joint_name}'


class AvoidJointLimitsPrismatic(Goal):
    def __init__(self, joint_name, group_name, weight=0.1, max_angular_velocity=100, percentage=5, **kwargs):
        """
        This goal will push prismatic joints away from their position limits
        :param joint_name: str
        :param weight: float, default WEIGHT_BELOW_CA
        :param max_angular_velocity: float, default 1e9, meaning the urdf/config limit will kick in
        :param percentage: float, default 15, if limits are 0-100, the constraint will push into the 15-85 range
        """
        super(AvoidJointLimitsPrismatic, self).__init__(**kwargs)
        prefix = self.world.groups[group_name].prefix
        self.joint_name = PrefixName(joint_name, prefix)
        self.weight = weight
        self.max_velocity = max_angular_velocity
        self.percentage = percentage
        if not self.world.is_joint_prismatic(joint_name):
            raise ConstraintException(f'{self.__class__.__name__} called with non prismatic joint {joint_name}')

    def make_constraints(self):
        weight = self.get_parameter_as_symbolic_expression('weight')
        joint_symbol = self.get_joint_position_symbol(self.joint_name)
        percentage = self.get_parameter_as_symbolic_expression('percentage') / 100.
        lower_limit, upper_limit = self.world.get_joint_position_limits(self.joint_name)
        max_velocity = self.get_parameter_as_symbolic_expression('max_velocity')
        max_velocity = w.min(max_velocity,
                             self.world.joint_limit_expr(self.joint_name, 1)[1])

        joint_range = upper_limit - lower_limit
        center = (upper_limit + lower_limit) / 2.

        max_error = joint_range / 2. * percentage

        upper_goal = center + joint_range / 2. * (1 - percentage)
        lower_goal = center - joint_range / 2. * (1 - percentage)

        upper_err = upper_goal - joint_symbol
        lower_err = lower_goal - joint_symbol

        # upper_err_capped = self.limit_velocity(upper_err, max_velocity)
        # lower_err_capped = self.limit_velocity(lower_err, max_velocity)

        error = w.max(w.abs(w.min(upper_err, 0)), w.abs(w.max(lower_err, 0)))
        weight = weight * (error / max_error)

        self.add_constraint(reference_velocity=max_velocity,
                            lower_error=lower_err,
                            upper_error=upper_err,
                            weight=weight,
                            expression=joint_symbol)

    def __str__(self):
        s = super(AvoidJointLimitsPrismatic, self).__str__()
        return f'{s}/{self.joint_name}'


class JointPositionList(Goal):
    def __init__(self, goal_state: Union[Dict[str, float], JointState], group_name: str, weight: float = None,
                 max_velocity: float = None, hard: bool = False, **kwargs):
        """
        This goal takes a joint state and adds the other JointPosition goals depending on their type
        :param weight: default is the default of the added joint goals
        :param max_velocity: default is the default of the added joint goals
        """
        super(JointPositionList, self).__init__(**kwargs)
        if len(goal_state.name) == 0:
            raise ConstraintInitalizationException(f'Can\'t initialize {self} with no joints.')
        prefix = self.world.groups[group_name].prefix
        for i, joint_name in enumerate(goal_state.name):
            if not self.world.has_joint(PrefixName(joint_name, prefix)):
                raise KeyError(f'unknown joint \'{joint_name}\'')
            goal_position = goal_state.position[i]
            params = kwargs
            params.update({'joint_name': joint_name,
                           'group_name': group_name,
                           'goal': goal_position})
            if weight is not None:
                params['weight'] = weight
            if max_velocity is not None:
                params['max_velocity'] = max_velocity
            params['hard'] = hard
            self.add_constraints_of_goal(JointPosition(**params))


class JointPosition(Goal):
    def __init__(self, joint_name: str, group_name: str, goal: float, weight: float = WEIGHT_BELOW_CA,
                 max_velocity: float = 100, **kwargs):
        super(JointPosition, self).__init__(**kwargs)
        prefix = self.world.groups[group_name].prefix
        full_joint_name = PrefixName(joint_name, prefix)
        if self.world.is_joint_continuous(full_joint_name):
            C = JointPositionContinuous
        elif self.world.is_joint_revolute(full_joint_name):
            C = JointPositionRevolute
        elif self.world.is_joint_prismatic(full_joint_name):
            C = JointPositionPrismatic
        else:
            raise ConstraintInitalizationException(f'\'{joint_name}\' has to be continuous, revolute or prismatic')
        self.add_constraints_of_goal(C(joint_name=joint_name,
                                       group_name=group_name,
                                       goal=goal,
                                       weight=weight,
                                       max_velocity=max_velocity,
                                       **kwargs))


class AvoidJointLimits(Goal):
    def __init__(self, percentage=15, weight=WEIGHT_BELOW_CA, **kwargs):
        """
        This goal will push joints away from their position limits
        :param percentage: float, default 15, if limits are 0-100, the constraint will push into the 15-85 range
        :param weight: float, default WEIGHT_BELOW_CA
        """
        super(AvoidJointLimits, self).__init__(**kwargs)
        for joint_name in self.god_map.get_data(identifier.controlled_joints):
            group_name = self.world.get_group_of_joint(joint_name).name
            if self.world.is_joint_revolute(joint_name):
                self.add_constraints_of_goal(AvoidJointLimitsRevolute(joint_name=joint_name.short_name,
                                                                      group_name=group_name,
                                                                      percentage=percentage,
                                                                      weight=weight, **kwargs))
            elif self.world.is_joint_prismatic(joint_name):
                self.add_constraints_of_goal(AvoidJointLimitsPrismatic(joint_name=joint_name.short_name,
                                                                       group_name=group_name,
                                                                       percentage=percentage,
                                                                       weight=weight, **kwargs))


class JointPositionRange(Goal):
    def __init__(self, joint_name, group_name, upper_limit, lower_limit, hard=False, **kwargs):
        super(JointPositionRange, self).__init__(**kwargs)
        prefix = self.world.groups[group_name].prefix
        self.joint_name = PrefixName(joint_name, prefix)
        if self.world.is_joint_continuous(joint_name):
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
            self.add_constraint(reference_velocity=self.world.joint_limit_expr(self.joint_name, 1)[1],
                                lower_error=self.lower_limit - joint_position,
                                upper_error=self.upper_limit - joint_position,
                                weight=WEIGHT_BELOW_CA,
                                expression=joint_position,
                                lower_slack_limit=0,
                                upper_slack_limit=0)
        else:
            self.add_constraint(reference_velocity=self.world.joint_limit_expr(self.joint_name, 1)[1],
                                lower_error=self.lower_limit - joint_position,
                                upper_error=self.upper_limit - joint_position,
                                weight=WEIGHT_BELOW_CA,
                                expression=joint_position)

    def __str__(self):
        s = super(JointPositionRange, self).__str__()
        return f'{s}/{self.joint_name}'
