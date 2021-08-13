from __future__ import division

import giskardpy.utils.tfwrapper as tf
from giskardpy import casadi_wrapper as w
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
        if reference_velocity is None:
            reference_velocity = max_velocity
        self.root_link = root_link
        self.tip_link = tip_link
        self.goal_pose = tf.transform_pose(self.root_link, goal)
        self.reference_velocity = reference_velocity
        self.max_velocity = max_velocity
        self.weight = weight
        super(CartesianPosition, self).__init__(**kwargs)
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


class CartesianOrientation(Goal):
    def __init__(self, root_link, tip_link, goal, reference_velocity=None, max_velocity=0.5, weight=WEIGHT_ABOVE_CA,
                 **kwargs):
        if reference_velocity is None:
            reference_velocity = max_velocity
        self.root_link = root_link
        self.tip_link = tip_link
        self.goal_pose = tf.transform_pose(self.root_link, goal)
        self.reference_velocity = reference_velocity
        self.max_velocity = max_velocity
        self.weight = weight
        super(CartesianOrientation, self).__init__(**kwargs)

    def make_constraints(self):
        r_R_g = w.rotation_of(self.get_parameter_as_symbolic_expression('goal_pose'))
        r_R_c = self.get_fk(self.root_link, self.tip_link)
        c_R_r_eval = self.get_fk_evaluated(self.tip_link, self.root_link)
        self.add_rotation_goal_constraints(frame_R_current=r_R_c,
                                           frame_R_goal=r_R_g,
                                           current_R_frame_eval=c_R_r_eval,
                                           reference_velocity=self.reference_velocity,
                                           weight=self.weight)
        if self.max_velocity is not None:
            self.add_rotational_velocity_limit(frame_R_current=r_R_c,
                                               max_velocity=self.max_velocity,
                                               weight=self.weight)


class CartesianPositionStraight(Goal):
    def __init__(self, root_link, tip_link, goal, reference_velocity=None, max_velocity=0.2,
                 weight=WEIGHT_ABOVE_CA, **kwargs):
        if reference_velocity is None:
            reference_velocity = max_velocity
        super(CartesianPositionStraight, self).__init__(root_link=root_link,
                                                        tip_link=tip_link,
                                                        goal=goal,
                                                        reference_velocity=reference_velocity,
                                                        max_velocity=max_velocity,
                                                        weight=weight,
                                                        **kwargs)

        start = tf.lookup_pose(self.root, self.tip)

        self.start = start

    def make_constraints(self):
        """
        example:
        name='CartesianPositionStraight'
        parameter_value_pair='{
            "root": "base_footprint", #required
            "tip": "r_gripper_tool_frame", #required
            "goal_position": {"header":
                                {"stamp":
                                    {"secs": 0,
                                    "nsecs": 0},
                                "frame_id": "",
                                "seq": 0},
                            "pose": {"position":
                                        {"y": 0.0,
                                        "x": 0.0,
                                        "z": 0.0},
                                    "orientation": {"y": 0.0,
                                                    "x": 0.0,
                                                    "z": 0.0,
                                                    "w": 0.0}
                                    }
                            }', #required
            "weight": 1, #optional
            "max_velocity": 0.3 #optional -- rad/s or m/s depending on joint; can not go higher than urdf limit
        }'
        :return:
        """
        root_P_goal = w.position_of(self.get_goal_pose())
        root_P_tip = w.position_of(self.get_fk(self.root, self.tip))
        root_V_start = w.position_of(self.get_parameter_as_symbolic_expression('start'))
        max_velocity = self.get_parameter_as_symbolic_expression('max_velocity')
        reference_velocity = self.get_parameter_as_symbolic_expression('reference_velocity')
        weight = self.get_parameter_as_symbolic_expression('weight')

        # Constraint to go to goal pos
        self.add_minimize_position_constraints(r_P_g=root_P_goal,
                                               reference_velocity=reference_velocity,
                                               max_velocity=max_velocity,
                                               root_frame=self.root,
                                               tip_frame=self.tip,
                                               weight=weight,
                                               name_suffix=u'goal')

        dist, nearest = w.distance_point_to_line_segment(root_P_tip,
                                                         root_V_start,
                                                         root_P_goal)
        # Constraint to stick to the line
        self.add_minimize_position_constraints(r_P_g=nearest,
                                               reference_velocity=max_velocity,
                                               root_frame=self.root,
                                               tip_frame=self.tip,
                                               name_suffix=u'line',
                                               weight=weight * 2)


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
    def __init__(self, root_link, tip_link, goal, translation_max_velocity=0.1,
                 translation_max_acceleration=0.1, rotation_max_velocity=0.5, rotation_max_acceleration=0.5,
                 weight=WEIGHT_ABOVE_CA, goal_constraint=True, **kwargs):
        super(CartesianPoseStraight, self).__init__(**kwargs)
        self.constraints = []
        self.add_constraints_of_goal(CartesianPositionStraight(root_link=root_link,
                                                               tip_link=tip_link,
                                                               goal=goal,
                                                               max_velocity=translation_max_velocity,
                                                               max_acceleration=translation_max_acceleration,
                                                               weight=weight,
                                                               goal_constraint=goal_constraint, **kwargs))
        self.add_constraints_of_goal(CartesianOrientation(root_link=root_link,
                                                          tip_link=tip_link,
                                                          goal=goal,
                                                          max_velocity=rotation_max_velocity,
                                                          max_accleration=rotation_max_acceleration,
                                                          weight=weight,
                                                          goal_constraint=goal_constraint, **kwargs))


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
