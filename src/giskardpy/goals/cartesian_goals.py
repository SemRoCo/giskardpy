from __future__ import division
from giskardpy import casadi_wrapper as w
from giskardpy.goals.goal import Goal, WEIGHT_ABOVE_CA
import giskardpy.utils.tfwrapper as tf

class BasicCartesianGoal(Goal):

    def __init__(self, god_map, root_link, tip_link, goal, reference_velocity=0.1, max_velocity=0.1,
                 weight=WEIGHT_ABOVE_CA, **kwargs):
        """
        dont use me
        """
        self.root = root_link
        self.tip = tip_link
        self.reference_velocity = reference_velocity
        self.goal = tf.transform_pose(self.root, goal)

        self.max_velocity = max_velocity
        self.weight = weight
        super(BasicCartesianGoal, self).__init__(god_map, **kwargs)

    def get_goal_pose(self):
        return self.get_input_PoseStamped(u'goal')

    def __str__(self):
        s = super(BasicCartesianGoal, self).__str__()
        return u'{}/{}/{}'.format(s, self.root, self.tip)


class CartesianPosition(BasicCartesianGoal):
    def __init__(self, god_map, root_link, tip_link, goal, reference_velocity=None, max_velocity=0.2,
                 weight=WEIGHT_ABOVE_CA, **kwargs):
        """
        This goal will use the kinematic chain between root and tip link to achieve a goal position for tip link
        :param root_link: str, root link of kinematic chain
        :param tip_link: str, tip link of kinematic chain
        :param goal: PoseStamped as json
        :param max_velocity: float, m/s, default 0.1
        :param weight: float, default WEIGHT_ABOVE_CA
        """
        if reference_velocity is None:
            reference_velocity = max_velocity
        super(CartesianPosition, self).__init__(god_map,
                                                root_link=root_link,
                                                tip_link=tip_link,
                                                goal=goal,
                                                reference_velocity=reference_velocity,
                                                max_velocity=max_velocity,
                                                weight=weight, **kwargs)

    def make_constraints(self):
        r_P_g = w.position_of(self.get_goal_pose())
        max_velocity = self.get_input_float(u'max_velocity')
        reference_velocity = self.get_input_float(u'reference_velocity')
        weight = self.get_input_float(u'weight')
        self.add_minimize_position_constraints(r_P_g=r_P_g,
                                               reference_velocity=reference_velocity,
                                               max_velocity=max_velocity,
                                               root=self.root,
                                               tip=self.tip,
                                               weight=weight)


class CartesianPositionStraight(BasicCartesianGoal):
    def __init__(self, god_map, root_link, tip_link, goal, reference_velocity=None, max_velocity=0.2,
                 weight=WEIGHT_ABOVE_CA, **kwargs):
        if reference_velocity is None:
            reference_velocity = max_velocity
        super(CartesianPositionStraight, self).__init__(god_map=god_map,
                                                        root_link=root_link,
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
                                               root=self.root,
                                               tip=self.tip,
                                               weight=weight,
                                               prefix=u'goal')

        dist, nearest = w.distance_point_to_line_segment(root_P_tip,
                                                         root_V_start,
                                                         root_P_goal)
        # Constraint to stick to the line
        self.add_minimize_position_constraints(r_P_g=nearest,
                                               reference_velocity=max_velocity,
                                               root=self.root,
                                               tip=self.tip,
                                               prefix=u'line',
                                               weight=weight*2)



class CartesianOrientation(BasicCartesianGoal):
    def __init__(self, god_map, root_link, tip_link, goal, reference_velocity=None, max_velocity=0.5,
                 max_accleration=0.5,
                 weight=WEIGHT_ABOVE_CA, goal_constraint=False, **kwargs):
        """
        This goal will the kinematic chain from root_link to tip_link to achieve a rotation goal for the tip link
        :param root_link: str, root link of the kinematic chain
        :param tip_link: str, tip link of the kinematic chain
        :param goal: PoseStamped as json
        :param max_velocity: float, rad/s, default 0.5
        :param weight: float, default WEIGHT_ABOVE_CA
        """
        if reference_velocity is None:
            reference_velocity = max_velocity
        super(CartesianOrientation, self).__init__(god_map=god_map,
                                                   root_link=root_link,
                                                   tip_link=tip_link,
                                                   goal=goal,
                                                   reference_velocity=reference_velocity,
                                                   max_velocity=max_velocity,
                                                   max_acceleration=max_accleration,
                                                   weight=weight,
                                                   goal_constraint=goal_constraint,
                                                   **kwargs)

    def make_constraints(self):
        """
        example:
        name='CartesianPosition'
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
        r_R_g = w.rotation_of(self.get_goal_pose())
        weight = self.get_input_float(u'weight')
        max_velocity = self.get_input_float(u'max_velocity')
        reference_velocity = self.get_input_float(u'reference_velocity')

        self.add_minimize_rotation_constraints(root_R_tipGoal=r_R_g,
                                               root=self.root,
                                               tip=self.tip,
                                               reference_velocity=reference_velocity,
                                               max_velocity=max_velocity,
                                               weight=weight)


class CartesianOrientationSlerp(CartesianOrientation):
    # TODO this is just here for backward compatibility
    pass


class CartesianPose(Goal):
    def __init__(self, god_map, root_link, tip_link, goal, max_linear_velocity=0.1,
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
        super(CartesianPose, self).__init__(god_map)
        self.constraints = []
        self.constraints.append(CartesianPosition(god_map=god_map,
                                                  root_link=root_link,
                                                  tip_link=tip_link,
                                                  goal=goal,
                                                  max_velocity=max_linear_velocity,
                                                  weight=weight,
                                                  **kwargs))
        self.constraints.append(CartesianOrientation(god_map=god_map,
                                                     root_link=root_link,
                                                     tip_link=tip_link,
                                                     goal=goal,
                                                     max_velocity=max_angular_velocity,
                                                     weight=weight,
                                                     **kwargs))

    def make_constraints(self):
        for constraint in self.constraints:
            self._constraints.update(constraint.get_constraints())


class CartesianPoseStraight(Goal):
    def __init__(self, god_map, root_link, tip_link, goal, translation_max_velocity=0.1,
                 translation_max_acceleration=0.1, rotation_max_velocity=0.5, rotation_max_acceleration=0.5,
                 weight=WEIGHT_ABOVE_CA, goal_constraint=True, **kwargs):
        super(CartesianPoseStraight, self).__init__(god_map)
        self.constraints = []
        self.constraints.append(CartesianPositionStraight(god_map=god_map,
                                                          root_link=root_link,
                                                          tip_link=tip_link,
                                                          goal=goal,
                                                          max_velocity=translation_max_velocity,
                                                          max_acceleration=translation_max_acceleration,
                                                          weight=weight,
                                                          goal_constraint=goal_constraint, **kwargs))
        self.constraints.append(CartesianOrientation(god_map=god_map,
                                                     root_link=root_link,
                                                     tip_link=tip_link,
                                                     goal=goal,
                                                     max_velocity=rotation_max_velocity,
                                                     max_accleration=rotation_max_acceleration,
                                                     weight=weight,
                                                     goal_constraint=goal_constraint, **kwargs))

    def make_constraints(self):
        for constraint in self.constraints:
            c, c_vel = constraint.get_constraints()
            self._constraints.update(c)
            self._velocity_constraints.update(c_vel)
            self.debug_expressions.update(constraint.debug_expressions)


class CartesianVelocityLimit(Goal):
    goal = u'goal'
    weight_id = u'weight'
    max_linear_velocity_id = u'max_linear_velocity'
    max_angular_velocity_id = u'max_angular_velocity'
    percentage = u'percentage'

    def __init__(self, god_map, root_link, tip_link, weight=WEIGHT_ABOVE_CA, max_linear_velocity=0.1,
                 max_angular_velocity=0.5, hard=True, **kwargs):
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
        super(CartesianVelocityLimit, self).__init__(god_map, **kwargs)
        self.root_link = root_link
        self.tip_link = tip_link
        self.hard = hard

        params = {self.weight_id: weight,
                  self.max_linear_velocity_id: max_linear_velocity,
                  self.max_angular_velocity_id: max_angular_velocity
                  }
        self.save_params_on_god_map(params)

    def make_constraints(self):
        weight = self.get_input_float(self.weight_id)
        max_linear_velocity = self.get_input_float(self.max_linear_velocity_id)
        max_angular_velocity = self.get_input_float(self.max_angular_velocity_id)
        sample_period = self.get_input_sampling_period()

        root_T_tip = self.get_fk(self.root_link, self.tip_link)
        tip_evaluated_T_root = self.get_fk_evaluated(self.tip_link, self.root_link)
        root_P_tip = w.position_of(root_T_tip)

        linear_weight = self.normalize_weight(max_linear_velocity, weight)

        if self.hard:
            slack_limit = 0
        else:
            slack_limit = 1e9

        self.add_constraint(u'/linear/x',
                            lower_velocity_limit=-max_linear_velocity * sample_period,
                            upper_velocity_limit=max_linear_velocity * sample_period,
                            weight=linear_weight,
                            expression=root_P_tip[0],
                            goal_constraint=False,
                            lower_slack_limit=-slack_limit,
                            upper_slack_limit=slack_limit)
        self.add_constraint(u'/linear/y',
                            lower_velocity_limit=-max_linear_velocity * sample_period,
                            upper_velocity_limit=max_linear_velocity * sample_period,
                            weight=linear_weight,
                            expression=root_P_tip[1],
                            goal_constraint=False,
                            lower_slack_limit=-slack_limit,
                            upper_slack_limit=slack_limit
                            )
        self.add_constraint(u'/linear/z',
                            lower_velocity_limit=-max_linear_velocity * sample_period,
                            upper_velocity_limit=max_linear_velocity * sample_period,
                            weight=linear_weight,
                            expression=root_P_tip[2],
                            goal_constraint=False,
                            lower_slack_limit=-slack_limit,
                            upper_slack_limit=slack_limit
                            )

        root_R_tip = w.rotation_of(root_T_tip)
        tip_evaluated_R_root = w.rotation_of(tip_evaluated_T_root)

        hack = w.rotation_matrix_from_axis_angle([0, 0, 1], 0.0001)

        axis, angle = w.axis_angle_from_matrix(w.dot(w.dot(tip_evaluated_R_root, hack), root_R_tip))
        angular_weight = self.normalize_weight(max_angular_velocity, weight)

        axis_angle = axis * angle

        self.add_constraint(u'/angular/x',
                            lower_velocity_limit=-max_angular_velocity * sample_period,
                            upper_velocity_limit=max_angular_velocity * sample_period,
                            weight=angular_weight,
                            expression=axis_angle[0],
                            goal_constraint=False,
                            lower_slack_limit=-slack_limit,
                            upper_slack_limit=slack_limit
                            )

        self.add_constraint(u'/angular/y',
                            lower_velocity_limit=-max_angular_velocity * sample_period,
                            upper_velocity_limit=max_angular_velocity * sample_period,
                            weight=angular_weight,
                            expression=axis_angle[1],
                            goal_constraint=False,
                            lower_slack_limit=-slack_limit,
                            upper_slack_limit=slack_limit
                            )

        self.add_constraint(u'/angular/z',
                            lower_velocity_limit=-max_angular_velocity * sample_period,
                            upper_velocity_limit=max_angular_velocity * sample_period,
                            weight=angular_weight,
                            expression=axis_angle[2],
                            goal_constraint=False,
                            lower_slack_limit=-slack_limit,
                            upper_slack_limit=slack_limit
                            )

    def __str__(self):
        s = super(CartesianVelocityLimit, self).__str__()
        return u'{}/{}/{}'.format(s, self.root_link, self.tip_link)

