from __future__ import division

from giskardpy import casadi_wrapper as w, identifier
from giskardpy.exceptions import ConstraintException, ConstraintInitalizationException
from giskardpy.goals.goal import Goal, WEIGHT_BELOW_CA


class JointPositionContinuous(Goal):

    def __init__(self, joint_name, goal, weight=WEIGHT_BELOW_CA, max_velocity=100, **kwargs):
        """
        This goal will move a continuous joint to the goal position
        :param joint_name: str
        :param goal: float
        :param weight: float, default WEIGHT_BELOW_CA
        :param max_velocity: float, rad/s, default 1423, meaning the urdf/config limits are active
        """
        self.joint_name = joint_name
        self.joint_goal = goal
        self.weight = weight
        self.max_velocity = max_velocity
        super(JointPositionContinuous, self).__init__(**kwargs)

        if not self.robot.is_joint_continuous(joint_name):
            raise ConstraintException(u'{} called with non continuous joint {}'.format(self.__class__.__name__,
                                                                                       joint_name))

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
        max_velocity = w.min(self.get_parameter_as_symbolic_expression(u'max_velocity'),
                             self.robot.joint_limit_expr(self.joint_name, 1)[0])

        error = w.shortest_angular_distance(current_joint, self.joint_goal)

        self.add_constraint(reference_velocity=max_velocity,
                            lower_error=error,
                            upper_error=error,
                            weight=self.weight,
                            expression=current_joint)
        if self.joint_name == 'odom_z_joint':
            self.add_debug_expr('error', error)

    def __str__(self):
        s = super(JointPositionContinuous, self).__str__()
        return u'{}/{}'.format(s, self.joint_name)


class JointPositionPrismatic(Goal):

    def __init__(self, joint_name, goal, weight=WEIGHT_BELOW_CA, max_velocity=100, **kwargs):
        """
        This goal will move a prismatic joint to the goal position
        :param joint_name: str
        :param goal: float
        :param weight: float, default WEIGHT_BELOW_CA
        :param max_velocity: float, m/s, default 4535, meaning the urdf/config limits are active
        """
        self.joint_name = joint_name
        self.goal = goal
        self.weight = weight
        self.max_velocity = max_velocity
        super(JointPositionPrismatic, self).__init__(**kwargs)
        if not self.robot.is_joint_prismatic(joint_name):
            raise ConstraintException(u'{} called with non prismatic joint {}'.format(self.__class__.__name__,
                                                                                      joint_name))

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

        joint_goal = self.get_parameter_as_symbolic_expression(u'goal')
        weight = self.get_parameter_as_symbolic_expression(u'weight')

        max_velocity = w.min(self.get_parameter_as_symbolic_expression(u'max_velocity'),
                             self.robot.joint_limit_expr(self.joint_name, 1)[0])

        error = joint_goal - current_joint

        self.add_constraint(reference_velocity=max_velocity,
                            lower_error=error,
                            upper_error=error,
                            weight=weight,
                            expression=current_joint)

    def __str__(self):
        s = super(JointPositionPrismatic, self).__str__()
        return u'{}/{}'.format(s, self.joint_name)


class JointPositionRevolute(Goal):

    def __init__(self, joint_name, goal, weight=WEIGHT_BELOW_CA, max_velocity=100, **kwargs):
        """
        This goal will move a revolute joint to the goal position
        :param joint_name: str
        :param goal: float
        :param weight: float, default WEIGHT_BELOW_CA
        :param max_velocity: float, rad/s, default 3451, meaning the urdf/config limits are active
        """
        self.joint_name = joint_name
        self.goal = goal
        self.weight = weight
        self.max_velocity = max_velocity
        super(JointPositionRevolute, self).__init__(**kwargs)
        if not self.robot.is_joint_revolute(joint_name):
            raise ConstraintException(u'{} called with non revolute joint {}'.format(self.__class__.__name__,
                                                                                     joint_name))

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

        joint_goal = self.get_parameter_as_symbolic_expression(u'goal')
        weight = self.get_parameter_as_symbolic_expression(u'weight')

        max_velocity = w.min(self.get_parameter_as_symbolic_expression(u'max_velocity'),
                             self.robot.joint_limit_expr(self.joint_name, 1)[0])

        error = joint_goal - current_joint

        self.add_constraint(reference_velocity=max_velocity,
                            lower_error=error,
                            upper_error=error,
                            weight=weight,
                            expression=current_joint)

    def __str__(self):
        s = super(JointPositionRevolute, self).__str__()
        return u'{}/{}'.format(s, self.joint_name)


class ShakyJointPositionRevoluteOrPrismatic(Goal):
    def __init__(self, joint_name, goal, frequency, noise_amplitude=1.0, weight=WEIGHT_BELOW_CA,
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
        self.joint_name = joint_name
        super(ShakyJointPositionRevoluteOrPrismatic, self).__init__(**kwargs)
        if not self.robot.is_joint_revolute(joint_name) and not self.robot.is_joint_prismatic(joint_name):
            raise ConstraintException(u'{} called with non revolute/prismatic joint {}'.format(self.__class__.__name__,
                                                                                               joint_name))

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
        frequency = self.get_parameter_as_symbolic_expression(u'frequency')
        noise_amplitude = self.get_parameter_as_symbolic_expression(u'noise_amplitude')
        joint_goal = self.get_parameter_as_symbolic_expression(u'goal')
        weight = self.get_parameter_as_symbolic_expression(u'weight')

        time = self.get_god_map().to_symbol(identifier.time)
        time_in_secs = self.get_sampling_period_symbol() * time

        max_velocity = w.min(self.get_parameter_as_symbolic_expression(u'max_velocity'),
                             self.robot.joint_limit_expr(self.joint_name, 1)[0])

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
        return u'{}/{}'.format(s, self.joint_name)


class ShakyJointPositionContinuous(Goal):
    def __init__(self, joint_name, goal, frequency, noise_amplitude=10, weight=WEIGHT_BELOW_CA,
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
        self.joint_name = joint_name
        self.goal = goal
        self.frequency = frequency
        self.noise_amplitude = noise_amplitude
        self.weight = weight
        self.max_velocity = max_velocity
        super(ShakyJointPositionContinuous, self).__init__(**kwargs)
        if not self.robot.is_joint_continuous(joint_name):
            raise ConstraintException(u'{} called with non continuous joint {}'.format(self.__class__.__name__,
                                                                                       joint_name))

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
        frequency = self.get_parameter_as_symbolic_expression(u'frequency')
        noise_amplitude = self.get_parameter_as_symbolic_expression(u'noise_amplitude')
        joint_goal = self.get_parameter_as_symbolic_expression(u'goal')
        weight = self.get_parameter_as_symbolic_expression(u'weight')

        time = self.get_god_map().to_symbol(identifier.time)
        time_in_secs = self.get_sampling_period_symbol() * time

        max_velocity = w.min(self.get_parameter_as_symbolic_expression(u'max_velocity'),
                             self.robot.joint_limit_expr(self.joint_name, 1)[0])

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
        return u'{}/{}'.format(s, self.joint_name)


class AvoidJointLimitsRevolute(Goal):
    def __init__(self, joint_name, weight=0.1, max_linear_velocity=100, percentage=5, **kwargs):
        """
        This goal will push revolute joints away from their position limits
        :param joint_name: str
        :param weight: float, default WEIGHT_BELOW_CA
        :param max_linear_velocity: float, default 1e9, meaning the urdf/config limit will kick in
        :param percentage: float, default 15, if limits are 0-100, the constraint will push into the 15-85 range
        """
        self.joint_name = joint_name
        self.weight = weight
        self.max_velocity = max_linear_velocity
        self.percentage = percentage
        super(AvoidJointLimitsRevolute, self).__init__(**kwargs)
        if not self.robot.is_joint_revolute(joint_name):
            raise ConstraintException(u'{} called with non prismatic joint {}'.format(self.__class__.__name__,
                                                                                      joint_name))

    def make_constraints(self):
        weight = self.get_parameter_as_symbolic_expression('weight')
        joint_symbol = self.get_joint_position_symbol(self.joint_name)
        percentage = self.get_parameter_as_symbolic_expression('percentage') / 100.
        lower_limit, upper_limit = self.robot.get_joint_position_limits(self.joint_name)
        max_velocity = self.get_parameter_as_symbolic_expression('max_velocity')
        max_velocity = w.min(max_velocity,
                             self.robot.joint_limit_expr(self.joint_name, 1)[0])

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
        return u'{}/{}'.format(s, self.joint_name)


class AvoidJointLimitsPrismatic(Goal):
    def __init__(self, joint_name, weight=0.1, max_angular_velocity=100, percentage=5, **kwargs):
        """
        This goal will push prismatic joints away from their position limits
        :param joint_name: str
        :param weight: float, default WEIGHT_BELOW_CA
        :param max_angular_velocity: float, default 1e9, meaning the urdf/config limit will kick in
        :param percentage: float, default 15, if limits are 0-100, the constraint will push into the 15-85 range
        """
        self.joint_name = joint_name
        self.weight = weight
        self.max_velocity = max_angular_velocity
        self.percentage = percentage
        super(AvoidJointLimitsPrismatic, self).__init__(**kwargs)
        if not self.robot.is_joint_prismatic(joint_name):
            raise ConstraintException(u'{} called with non prismatic joint {}'.format(self.__class__.__name__,
                                                                                      joint_name))

    def make_constraints(self):
        weight = self.get_parameter_as_symbolic_expression('weight')
        joint_symbol = self.get_joint_position_symbol(self.joint_name)
        percentage = self.get_parameter_as_symbolic_expression('percentage') / 100.
        lower_limit, upper_limit = self.robot.get_joint_position_limits(self.joint_name)
        max_velocity = self.get_parameter_as_symbolic_expression('max_velocity')
        max_velocity = w.min(max_velocity,
                             self.robot.joint_limit_expr(self.joint_name, 1)[0])

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
        return u'{}/{}'.format(s, self.joint_name)


class JointPositionList(Goal):
    def __init__(self, goal_state, weight=None, max_velocity=None, **kwargs):
        """
        This goal takes a joint state and adds the other JointPosition goals depending on their type
        :param goal_state: JointState as json
        :param weight: float, default is the default of the added joint goals
        :param max_velocity: float, default is the default of the added joint goals
        """
        super(JointPositionList, self).__init__(**kwargs)
        for i, joint_name in enumerate(goal_state.name):
            if not self.robot.has_joint(joint_name):
                raise KeyError(u'unknown joint "{}"'.format(joint_name))
            goal_position = goal_state.position[i]
            params = kwargs
            params.update({u'joint_name': joint_name,
                           u'goal': goal_position})
            if weight is not None:
                params[u'weight'] = weight
            if max_velocity is not None:
                params[u'max_velocity'] = max_velocity
            if self.robot.is_joint_continuous(joint_name):
                self.add_constraints_of_goal(JointPositionContinuous(**params))
            elif self.robot.is_joint_revolute(joint_name):
                self.add_constraints_of_goal(JointPositionRevolute(**params))
            elif self.robot.is_joint_prismatic(joint_name):
                self.add_constraints_of_goal(JointPositionPrismatic(**params))


class AvoidJointLimits(Goal):
    def __init__(self, percentage=15, weight=WEIGHT_BELOW_CA, **kwargs):
        """
        This goal will push joints away from their position limits
        :param percentage: float, default 15, if limits are 0-100, the constraint will push into the 15-85 range
        :param weight: float, default WEIGHT_BELOW_CA
        """
        super(AvoidJointLimits, self).__init__(**kwargs)
        for joint_name in self.god_map.get_data(identifier.controlled_joints):
            if self.robot.is_joint_revolute(joint_name):
                self.add_constraints_of_goal(AvoidJointLimitsRevolute(joint_name=joint_name,
                                                                      percentage=percentage,
                                                                      weight=weight, **kwargs))
            elif self.robot.is_joint_prismatic(joint_name):
                self.add_constraints_of_goal(AvoidJointLimitsPrismatic(joint_name=joint_name,
                                                                       percentage=percentage,
                                                                       weight=weight, **kwargs))


class JointPositionRange(Goal):

    def __init__(self, joint_name, upper_limit, lower_limit, **kwargs):
        self.joint_name = joint_name
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        super(JointPositionRange, self).__init__(**kwargs)
        current_position = self.robot.state[self.joint_name].position
        if current_position > self.upper_limit + 2e-3 or current_position < self.lower_limit - 2e-3:
            raise ConstraintInitalizationException(u'{} out of set limits. '
                                                   u'{} <= {} <= {} is not true.'.format(self.joint_name,
                                                                                         self.lower_limit,
                                                                                         current_position,
                                                                                         self.upper_limit))

    def make_constraints(self):
        joint_position = self.get_joint_position_symbol(self.joint_name)
        self.add_constraint(reference_velocity=self.robot.joint_limit_expr(self.joint_name, 1)[0],
                            lower_error=self.lower_limit - joint_position,
                            upper_error=self.upper_limit - joint_position,
                            weight=WEIGHT_BELOW_CA,
                            expression=joint_position,
                            lower_slack_limit=0,
                            upper_slack_limit=0)

    def __str__(self):
        s = super(JointPositionRange, self).__str__()
        return u'{}/{}'.format(s, self.joint_name)
