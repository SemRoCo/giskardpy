from __future__ import division

from geometry_msgs.msg import PointStamped, PoseStamped, QuaternionStamped, Vector3Stamped

from giskardpy import casadi_wrapper as w
from giskardpy.goals.goal import Goal, WEIGHT_ABOVE_CA
from giskardpy.goals.pointing import PointingDiffDrive
from giskardpy.god_map import GodMap


class CartesianPosition(Goal):
    def __init__(self, root_link: str, tip_link: str, goal_point: PointStamped, reference_velocity: float = None,
                 max_velocity: float = 0.2, weight: float = WEIGHT_ABOVE_CA, **kwargs):
        """
        This goal will use the kinematic chain between root and tip link to achieve a goal position for tip link.
        :param root_link: root link of kinematic chain
        :param tip_link: tip link of kinematic chain
        :param goal: the goal, orientation part will be ignored
        :param max_velocity: m/s
        :param reference_velocity: m/s
        :param weight: default WEIGHT_ABOVE_CA
        """
        super(CartesianPosition, self).__init__(**kwargs)
        if reference_velocity is None:
            reference_velocity = max_velocity
        self.root_link = root_link
        self.tip_link = tip_link
        self.goal_point = self.transform_msg(self.root_link, goal_point)
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
        r_P_g = self.get_parameter_as_symbolic_expression('goal_point')
        r_P_c = w.position_of(self.get_fk(self.root_link, self.tip_link))
        # self.add_debug_expr('trans', w.norm(r_P_c))
        self.add_point_goal_constraints(frame_P_goal=r_P_g,
                                        frame_P_current=r_P_c,
                                        reference_velocity=self.reference_velocity,
                                        weight=self.weight)

    def __str__(self):
        s = super(CartesianPosition, self).__str__()
        return '{}/{}/{}'.format(s, self.root_link, self.tip_link)


class CartesianOrientation(Goal):
    def __init__(self, root_link, tip_link, goal_orientation, reference_velocity=None, max_velocity=0.5,
                 weight=WEIGHT_ABOVE_CA, **kwargs):
        super(CartesianOrientation, self).__init__(**kwargs)
        if reference_velocity is None:
            reference_velocity = max_velocity
        self.root_link = root_link
        self.tip_link = tip_link
        self.goal_orientation = self.transform_msg(self.root_link, goal_orientation)
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
        r_R_g = self.get_parameter_as_symbolic_expression('goal_orientation')
        r_R_c = self.get_fk(self.root_link, self.tip_link)
        c_R_r_eval = self.get_fk_evaluated(self.tip_link, self.root_link)
        self.add_rotation_goal_constraints(frame_R_current=r_R_c,
                                           frame_R_goal=r_R_g,
                                           current_R_frame_eval=c_R_r_eval,
                                           reference_velocity=self.reference_velocity,
                                           weight=self.weight)

    def __str__(self):
        s = super(CartesianOrientation, self).__str__()
        return '{}/{}/{}'.format(s, self.root_link, self.tip_link)


class CartesianPositionStraight(Goal):
    def __init__(self, root_link, tip_link, goal, reference_velocity=None, max_velocity=0.2,
                 weight=WEIGHT_ABOVE_CA, **kwargs):
        super(CartesianPositionStraight, self).__init__(**kwargs)
        if reference_velocity is None:
            reference_velocity = max_velocity
        self.reference_velocity = reference_velocity
        self.max_velocity = max_velocity
        self.weight = weight
        self.root_link = root_link
        self.tip_link = tip_link
        self.goal_pose = self.transform_msg(self.root_link, goal)

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
                                        name_suffix='goal')

        dist, nearest = w.distance_point_to_line_segment(root_P_tip,
                                                         root_V_start,
                                                         root_P_goal)
        # Constraint to stick to the line
        self.add_point_goal_constraints(frame_P_goal=nearest,
                                        frame_P_current=root_P_tip,
                                        reference_velocity=self.reference_velocity,
                                        name_suffix='line',
                                        weight=self.weight * 2)

        if self.max_velocity is not None:
            self.add_translational_velocity_limit(frame_P_current=root_P_tip,
                                                  max_velocity=self.max_velocity,
                                                  weight=self.weight)


class CartesianPose(Goal):
    def __init__(self, root_link: str, tip_link: str, goal_pose: PoseStamped, max_linear_velocity: float = 0.1,
                 max_angular_velocity: float = 0.5, weight: float = WEIGHT_ABOVE_CA, **kwargs):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose
        :param root_link: str, name of the root link of the kin chain
        :param tip_link: str, name of the tip link of the kin chain
        :param goal_pose: PoseStamped as json
        :param max_linear_velocity: float, m/s, default 0.1
        :param max_angular_velocity: float, rad/s, default 0.5
        :param weight: float, default WEIGHT_ABOVE_CA
        """
        super(CartesianPose, self).__init__(**kwargs)
        goal_point = PointStamped()
        goal_point.header = goal_pose.header
        goal_point.point = goal_pose.pose.position
        self.add_constraints_of_goal(CartesianPosition(root_link=root_link,
                                                       tip_link=tip_link,
                                                       goal_point=goal_point,
                                                       max_velocity=max_linear_velocity,
                                                       weight=weight,
                                                       **kwargs))
        goal_orientation = QuaternionStamped()
        goal_orientation.header = goal_pose.header
        goal_orientation.quaternion = goal_pose.pose.orientation
        self.add_constraints_of_goal(CartesianOrientation(root_link=root_link,
                                                          tip_link=tip_link,
                                                          goal_orientation=goal_orientation,
                                                          max_velocity=max_angular_velocity,
                                                          weight=weight,
                                                          **kwargs))


class DiffDriveBaseGoal(CartesianPose):

    def __init__(self, root_link: str, tip_link: str, goal_pose: PoseStamped, max_linear_velocity: float = 0.1,
                 max_angular_velocity: float = 0.5, weight: float = WEIGHT_ABOVE_CA, pointing_axis=None, **kwargs):
        super().__init__(root_link, tip_link, goal_pose, max_linear_velocity, max_angular_velocity, weight, **kwargs)
        goal_point = PointStamped()
        goal_point.header = goal_pose.header
        goal_point.point = goal_pose.pose.position
        if pointing_axis is None:
            pointing_axis = Vector3Stamped()
            pointing_axis.header.frame_id = tip_link
            pointing_axis.vector.x = 1
        #TODO handle weights properly
        self.add_constraints_of_goal(PointingDiffDrive(tip_link=tip_link,
                                                       root_link=root_link,
                                                       goal_point=goal_point,
                                                       pointing_axis=pointing_axis,
                                                       max_velocity=max_angular_velocity,
                                                       **kwargs))


class CartesianPoseStraight(Goal):
    def __init__(self, root_link, tip_link, goal_pose, max_linear_velocity=0.1, max_angular_velocity=0.5,
                 weight=WEIGHT_ABOVE_CA, **kwargs):
        super(CartesianPoseStraight, self).__init__(**kwargs)
        self.add_constraints_of_goal(CartesianPositionStraight(root_link=root_link,
                                                               tip_link=tip_link,
                                                               goal=goal_pose,
                                                               max_velocity=max_linear_velocity,
                                                               weight=weight,
                                                               **kwargs))
        self.add_constraints_of_goal(CartesianOrientation(root_link=root_link,
                                                          tip_link=tip_link,
                                                          goal_orientation=goal_pose,
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
        # self.add_debug_expr('limit', -self.max_velocity)
        if not self.hard:
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
        return '{}/{}/{}'.format(s, self.root_link, self.tip_link)


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
        return '{}/{}/{}'.format(s, self.root_link, self.tip_link)


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
