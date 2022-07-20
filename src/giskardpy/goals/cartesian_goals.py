from __future__ import division

from tf2_py import LookupException

import giskardpy.utils.tfwrapper as tf
from giskardpy import casadi_wrapper as w, identifier
from giskardpy.exceptions import ConstraintException
from giskardpy.goals.goal import Goal, WEIGHT_ABOVE_CA


class CartesianPosition(Goal):
    def __init__(self, root_link, tip_link, goal, reference_velocity=None, max_velocity=0.2, weight=WEIGHT_ABOVE_CA,
                 **kwargs):
        """
        This goal will use the kinematic chain between root and tip link to achieve a goal position for tip link.
        :param root_link: root link of kinematic chain
        :type root_link: str
        :param tip_link: tip link of kinematic chain
        :type tip_link: str
        :param goal: the goal, orientation part will be ignored
        :type goal: PoseStamped
        :param max_velocity: m/s
        :type max_velocity: float
        :param reference_velocity: m/s
        :type reference_velocity: float
        :param weight: default WEIGHT_ABOVE_CA
        :type weight: float
        """
        super(CartesianPosition, self).__init__(**kwargs)
        if reference_velocity is None:
            reference_velocity = max_velocity
        self.root_link = root_link
        self.tip_link = tip_link
        self.goal_pose = self.transform_msg(self.root_link, goal)
        self.reference_velocity = reference_velocity
        self.max_velocity = max_velocity
        self.weight = weight
        if self.max_velocity is not None:
            self.add_constraints_of_goal(TranslationVelocityLimit(root_link=root_link,
                                                                  tip_link=tip_link,
                                                                  weight=weight,
                                                                  max_velocity=max_velocity,
                                                                  hard=False,
                                                                  **kwargs))

    def make_constraints(self):
        r_P_g = w.position_of(self.get_parameter_as_symbolic_expression(u'goal_pose'))
        r_P_c = w.position_of(self.get_fk(self.root_link, self.tip_link))
        self.add_point_goal_constraints(frame_P_goal=r_P_g,
                                        frame_P_current=r_P_c,
                                        reference_velocity=self.reference_velocity,
                                        weight=self.weight)

    def __str__(self):
        s = super(CartesianPosition, self).__str__()
        return u'{}/{}/{}'.format(s, self.root_link, self.tip_link)


class ShakyCartesianPosition(Goal):
    def __init__(self, root_link, tip_link, frequency, shaking_axis='z', noise_amplitude=0.2, weight=WEIGHT_ABOVE_CA,
                 max_velocity=0.2, **kwargs):
        """
        This goal will move a revolute or prismatic joint to the goal position and shake the joint with the given frequency.
        :param joint_name: str
        :param goal: float
        :param frequency: float
        :param noise_amplitude: float
        :param weight: float, default WEIGHT_BELOW_CA
        :param max_velocity: float, rad/s, default 3451, meaning the urdf/config limits are active
        """
        self.root_link = root_link
        self.tip_link = tip_link
        super(ShakyCartesianPosition, self).__init__(**kwargs)
        if not self.world.are_linked(root_link, tip_link):
            raise ConstraintException(u'{} called with non linked links {} and {} '.format(self.__class__.__name__,
                                                                                           root_link, tip_link))

        self.frequency = frequency
        self.noise_amplitude = noise_amplitude
        self.weight = weight
        self.max_velocity = max_velocity
        if shaking_axis == 'x':
            self.shaking_index = 0
        elif shaking_axis == 'y':
            self.shaking_index = 1
        elif shaking_axis == 'z':
            self.shaking_index = 2
        else:
            raise Exception('no')

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
        frequency = self.get_parameter_as_symbolic_expression(u'frequency')
        noise_amplitude = self.get_parameter_as_symbolic_expression(u'noise_amplitude')
        weight = self.get_parameter_as_symbolic_expression(u'weight')
        r_P_c = w.position_of(self.get_fk(self.root_link, self.tip_link))

        time = self.god_map.to_symbol(identifier.time)
        time_in_secs = self.get_sampling_period_symbol() * time

        fun_params = frequency * 2.0 * w.pi * time_in_secs
        err = noise_amplitude * self.max_velocity * w.sin(fun_params)
        capped_err = w.limit(err, -noise_amplitude * self.max_velocity, noise_amplitude * self.max_velocity)

        self.add_constraint_vector(reference_velocities=[self.max_velocity] * 1,
                                   lower_errors=capped_err,
                                   upper_errors=capped_err,
                                   weights=[weight] * 1,
                                   expressions=r_P_c[self.shaking_index],
                                   name_suffixes=['/z'])

    def __str__(self):
        s = super(ShakyCartesianPosition, self).__str__()
        return u'{}/{}'.format(s, self.tip_link)


class CartesianOrientation(Goal):
    def __init__(self, root_link, tip_link, goal, reference_velocity=None, max_velocity=0.5, weight=WEIGHT_ABOVE_CA,
                 **kwargs):
        super(CartesianOrientation, self).__init__(**kwargs)
        if reference_velocity is None:
            reference_velocity = max_velocity
        self.root_link = root_link
        self.tip_link = tip_link
        self.goal_pose = self.transform_msg(self.root_link, goal)
        self.reference_velocity = reference_velocity
        self.max_velocity = max_velocity
        self.weight = weight
        # if self.max_velocity is not None:
        #     self.add_constraints_of_goal(RotationVelocityLimit(root_link=root_link,
        #                                                        tip_link=tip_link,
        #                                                        weight=weight,
        #                                                        max_velocity=max_velocity,
        #                                                        hard=False,
        #                                                        **kwargs))

    def make_constraints(self):
        r_R_g = w.rotation_of(self.get_parameter_as_symbolic_expression('goal_pose'))
        r_R_c = self.get_fk(self.root_link, self.tip_link)
        c_R_r_eval = self.get_fk_evaluated(self.tip_link, self.root_link)
        self.add_rotation_goal_constraints(frame_R_current=r_R_c,
                                           frame_R_goal=r_R_g,
                                           current_R_frame_eval=c_R_r_eval,
                                           reference_velocity=self.reference_velocity,
                                           weight=self.weight)

    def __str__(self):
        s = super(CartesianOrientation, self).__str__()
        return u'{}/{}/{}'.format(s, self.root_link, self.tip_link)


class CartesianPositionStraight(Goal):
    def __init__(self, root_link, tip_link, goal, reference_velocity=None, max_velocity=0.2,
                 weight=WEIGHT_ABOVE_CA, **kwargs):
        if reference_velocity is None:
            reference_velocity = max_velocity
        self.reference_velocity = reference_velocity
        self.max_velocity = max_velocity
        self.weight = weight
        self.root_link = root_link
        self.tip_link = tip_link
        self.goal_pose = self.transform_msg(self.root_link, goal)
        super(CartesianPositionStraight, self).__init__(**kwargs)

        self.start = self.world.compute_fk_pose(self.root_link, self.tip_link)

    def make_constraints(self):
        root_P_goal = w.position_of(self.get_parameter_as_symbolic_expression('goal_pose'))
        root_P_tip = w.position_of(self.get_fk(self.root_link, self.tip_link))
        root_V_start = w.position_of(self.get_parameter_as_symbolic_expression('start'))

        # Constraint to go to goal pos
        self.add_point_goal_constraints(frame_P_current=root_P_tip,
                                        frame_P_goal=root_P_goal,
                                        reference_velocity=self.reference_velocity,
                                        weight=self.weight,
                                        name_suffix=u'goal')

        dist, nearest = w.distance_point_to_line_segment(root_P_tip,
                                                         root_V_start,
                                                         root_P_goal)
        # Constraint to stick to the line
        self.add_point_goal_constraints(frame_P_goal=nearest,
                                        frame_P_current=root_P_tip,
                                        reference_velocity=self.reference_velocity,
                                        name_suffix=u'line',
                                        weight=self.weight * 2)

        if self.max_velocity is not None:
            self.add_translational_velocity_limit(frame_P_current=root_P_tip,
                                                  max_velocity=self.max_velocity,
                                                  weight=self.weight)


class CartesianPose(Goal):
    def __init__(self, root_link, tip_link, goal, max_linear_velocity=0.1,
                 max_angular_velocity=0.5, weight=WEIGHT_ABOVE_CA, **kwargs):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose
        :param root_link: str, name of the root link of the kin chain
        :param tip_link: str, name of the tip link of the kin chain
        :param goal: PoseStamped as json
        :param max_linear_velocity: float, m/s, default 0.1
        :param max_angular_velocity: float, rad/s, default 0.5
        :param weight: float, default WEIGHT_ABOVE_CA
        """
        super(CartesianPose, self).__init__(**kwargs)
        self.add_constraints_of_goal(CartesianPosition(root_link=root_link,
                                                       tip_link=tip_link,
                                                       goal=goal,
                                                       max_velocity=max_linear_velocity,
                                                       weight=weight,
                                                       **kwargs))
        self.add_constraints_of_goal(CartesianOrientation(root_link=root_link,
                                                          tip_link=tip_link,
                                                          goal=goal,
                                                          max_velocity=max_angular_velocity,
                                                          weight=weight,
                                                          **kwargs))


class CartesianPoseStraight(Goal):
    def __init__(self, root_link, tip_link, goal, translation_max_velocity=0.1, rotation_max_velocity=0.5,
                 weight=WEIGHT_ABOVE_CA, **kwargs):
        super(CartesianPoseStraight, self).__init__(**kwargs)
        self.add_constraints_of_goal(CartesianPositionStraight(root_link=root_link,
                                                               tip_link=tip_link,
                                                               goal=goal,
                                                               max_velocity=translation_max_velocity,
                                                               weight=weight,
                                                               **kwargs))
        self.add_constraints_of_goal(CartesianOrientation(root_link=root_link,
                                                          tip_link=tip_link,
                                                          goal=goal,
                                                          max_velocity=rotation_max_velocity,
                                                          weight=weight,
                                                          **kwargs))


class CartesianPreGrasp(Goal):
    def __init__(self, root_link, tip_link, grasping_goal, grasping_object, goal, goal_position=None, dist=0, max_linear_velocity=0.1,
                 max_angular_velocity=0.5, weight=WEIGHT_ABOVE_CA, **kwargs):
        super(CartesianPreGrasp, self).__init__(**kwargs)
        self.root_link = root_link
        self.tip_link = tip_link
        self.goal = goal
        self.grasping_goal = grasping_goal
        self.grasping_object = grasping_object
        self.goal_position = goal_position
        self.dist = dist
        self.add_constraints_of_goal(CartesianPosition(root_link=root_link,
                                                       tip_link=tip_link,
                                                       goal=goal,
                                                       max_velocity=max_linear_velocity,
                                                       weight=weight,
                                                       **kwargs))
        self.add_constraints_of_goal(CartesianOrientation(root_link=root_link,
                                                          tip_link=tip_link,
                                                          goal=goal,
                                                          max_velocity=max_angular_velocity,
                                                          weight=weight,
                                                          **kwargs))


class TranslationVelocityLimit(Goal):
    def __init__(self, root_link, tip_link, weight=WEIGHT_ABOVE_CA, max_velocity=0.1, hard=True, **kwargs):
        """
        This goal will limit the cartesian velocity of the tip link relative to root link
        :param root_link: str, root link of the kin chain
        :param tip_link: str, tip link of the kin chain
        :param weight: float, default WEIGHT_ABOVE_CA
        :param max_linear_velocity: float, m/s, default 0.1
        :param max_angular_velocity: float, rad/s, default 0.5
        :param hard: bool, default True, will turn this into a hard constraint, that will always be satisfied, can could
                                make some goal combination infeasible
        """
        self.root_link = root_link
        self.tip_link = tip_link
        self.hard = hard
        self.weight = weight
        self.max_velocity = max_velocity
        super(TranslationVelocityLimit, self).__init__(**kwargs)

    def make_constraints(self):
        r_P_c = w.position_of(self.get_fk(self.root_link, self.tip_link))
        if self.hard:
            self.add_translational_velocity_limit(frame_P_current=r_P_c,
                                                  max_velocity=self.max_velocity,
                                                  weight=self.weight)
        else:
            self.add_translational_velocity_limit(frame_P_current=r_P_c,
                                                  max_velocity=self.max_velocity,
                                                  weight=self.weight,
                                                  max_violation=0)

    def __str__(self):
        s = super(TranslationVelocityLimit, self).__str__()
        return u'{}/{}/{}'.format(s, self.root_link, self.tip_link)


class RotationVelocityLimit(Goal):
    def __init__(self, root_link, tip_link, weight=WEIGHT_ABOVE_CA, max_velocity=0.5, hard=True, **kwargs):
        """
        This goal will limit the cartesian velocity of the tip link relative to root link
        :param root_link: str, root link of the kin chain
        :param tip_link: str, tip link of the kin chain
        :param weight: float, default WEIGHT_ABOVE_CA
        :param max_linear_velocity: float, m/s, default 0.1
        :param max_angular_velocity: float, rad/s, default 0.5
        :param hard: bool, default True, will turn this into a hard constraint, that will always be satisfied, can could
                                make some goal combination infeasible
        """
        self.root_link = root_link
        self.tip_link = tip_link
        self.hard = hard

        self.weight = weight
        self.max_velocity = max_velocity
        super(RotationVelocityLimit, self).__init__(**kwargs)

    def make_constraints(self):
        r_R_c = w.rotation_of(self.get_fk(self.root_link, self.tip_link))
        if self.hard:
            self.add_rotational_velocity_limit(frame_R_current=r_R_c,
                                               max_velocity=self.max_velocity,
                                               weight=self.weight)
        else:
            self.add_rotational_velocity_limit(frame_R_current=r_R_c,
                                               max_velocity=self.max_velocity,
                                               weight=self.weight,
                                               max_violation=0)

    def __str__(self):
        s = super(RotationVelocityLimit, self).__str__()
        return u'{}/{}/{}'.format(s, self.root_link, self.tip_link)


class CartesianVelocityLimit(Goal):
    def __init__(self, root_link, tip_link, max_linear_velocity=0.1, max_angular_velocity=0.5, weight=WEIGHT_ABOVE_CA,
                 hard=False, **kwargs):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose
        :param root_link: str, name of the root link of the kin chain
        :param tip_link: str, name of the tip link of the kin chain
        :param goal: PoseStamped as json
        :param max_linear_velocity: float, m/s, default 0.1
        :param max_angular_velocity: float, rad/s, default 0.5
        :param weight: float, default WEIGHT_ABOVE_CA
        """
        super(CartesianVelocityLimit, self).__init__(**kwargs)
        self.add_constraints_of_goal(TranslationVelocityLimit(root_link=root_link,
                                                              tip_link=tip_link,
                                                              max_velocity=max_linear_velocity,
                                                              weight=weight,
                                                              hard=hard,
                                                              **kwargs))
        self.add_constraints_of_goal(RotationVelocityLimit(root_link=root_link,
                                                           tip_link=tip_link,
                                                           max_velocity=max_angular_velocity,
                                                           weight=weight,
                                                           hard=hard,
                                                           **kwargs))
