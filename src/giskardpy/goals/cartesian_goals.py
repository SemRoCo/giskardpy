from __future__ import division

import numpy as np
from geometry_msgs.msg import PointStamped, PoseStamped, QuaternionStamped, Vector3Stamped

from giskardpy import casadi_wrapper as w
from giskardpy.goals.goal import Goal, WEIGHT_ABOVE_CA, WEIGHT_BELOW_CA
from giskardpy.god_map import GodMap
from giskardpy.model.joints import DiffDrive
from giskardpy.utils.tfwrapper import msg_to_homogeneous_matrix, normalize


class CartesianPosition(Goal):
    def __init__(self, root_link: str, tip_link: str, goal_point: PointStamped, reference_velocity: float = None,
                 max_velocity: float = 0.2, weight: float = WEIGHT_ABOVE_CA, root_link2: str = None, **kwargs):
        """
        This goal will use the kinematic chain between root and tip link to achieve a goal position for tip link.
        :param root_link: root link of kinematic chain
        :param tip_link: tip link of kinematic chain
        :param goal: the goal, orientation part will be ignored
        :param max_velocity: m/s
        :param reference_velocity: m/s
        :param weight: default WEIGHT_ABOVE_CA
        """
        super().__init__(**kwargs)
        self.root_link2 = root_link2
        if reference_velocity is None:
            reference_velocity = max_velocity
        self.root_link = root_link
        self.tip_link = tip_link
        if self.root_link2 is not None:
            self.goal_point = self.transform_msg(self.root_link2, goal_point)
        else:
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
        r_P_g = w.ros_msg_to_matrix(self.goal_point)
        r_P_c = w.position_of(self.get_fk(self.root_link, self.tip_link))
        if self.root_link2 is not None:
            root_link2_T_root_link = self.get_fk_evaluated(self.root_link2, self.root_link)
            r_P_c = w.dot(root_link2_T_root_link, r_P_c)
        # self.add_debug_expr('trans', w.norm(r_P_c))
        self.add_point_goal_constraints(frame_P_goal=r_P_g,
                                        frame_P_current=r_P_c,
                                        reference_velocity=self.reference_velocity,
                                        weight=self.weight)

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.root_link}/{self.tip_link}'


class CartesianOrientation(Goal):
    def __init__(self, root_link, tip_link, goal_orientation, reference_velocity=None, max_velocity=0.5,
                 weight=WEIGHT_ABOVE_CA, root_link2: str = None, **kwargs):
        super().__init__(**kwargs)
        self.root_link2 = root_link2
        if reference_velocity is None:
            reference_velocity = max_velocity
        self.root_link = root_link
        self.tip_link = tip_link
        if self.root_link2 is not None:
            self.goal_orientation = self.transform_msg(self.root_link2, goal_orientation)
        else:
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
        r_R_g = w.ros_msg_to_matrix(self.goal_orientation)
        r_R_c = self.get_fk(self.root_link, self.tip_link)
        if self.root_link2 is not None:
            c_R_r_eval = self.get_fk_evaluated(self.tip_link, self.root_link2)
            root_link2_T_root_link = self.get_fk_evaluated(self.root_link2, self.root_link)
            # self.add_debug_matrix('root_link2_T_root_link', root_link2_T_root_link)
            r_R_c = w.dot(root_link2_T_root_link, r_R_c)
        else:
            c_R_r_eval = self.get_fk_evaluated(self.tip_link, self.root_link)
        # self.add_debug_expr('trans', w.norm(r_P_c))
        self.add_rotation_goal_constraints(frame_R_current=r_R_c,
                                           frame_R_goal=r_R_g,
                                           current_R_frame_eval=c_R_r_eval,
                                           reference_velocity=self.reference_velocity,
                                           weight=self.weight)

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.root_link}/{self.tip_link}'


class CartesianPositionStraight(Goal):
    def __init__(self, root_link: str, tip_link: str, goal_point: PointStamped, reference_velocity: float = None,
                 max_velocity: float = 0.2, weight: float = WEIGHT_ABOVE_CA, **kwargs):
        super(CartesianPositionStraight, self).__init__(**kwargs)
        if reference_velocity is None:
            reference_velocity = max_velocity
        self.reference_velocity = reference_velocity
        self.max_velocity = max_velocity
        self.weight = weight
        self.root_link = root_link
        self.tip_link = tip_link
        self.goal_point = self.transform_msg(self.root_link, goal_point)

    def make_constraints(self):
        root_P_goal = w.ros_msg_to_matrix(self.goal_point)
        root_P_tip = w.position_of(self.get_fk(self.root_link, self.tip_link))
        t_T_r = self.get_fk(self.tip_link, self.root_link)
        tip_P_goal = w.dot(t_T_r, root_P_goal)

        # Create rotation matrix, which rotates the tip link frame
        # such that its x-axis shows towards the goal position.
        # The goal frame is called 'a'.
        # Thus, the rotation matrix is called t_R_a.
        tip_P_error = tip_P_goal[:3]
        trans_error = w.norm(tip_P_error)
        # x-axis
        tip_P_intermediate_error = w.save_division(tip_P_error, trans_error)[:3]
        # y- and z-axis
        tip_P_intermediate_y = w.scale(w.Matrix(np.random.random((3,))), 1)
        y = w.cross(tip_P_intermediate_error, tip_P_intermediate_y)
        z = w.cross(tip_P_intermediate_error, y)
        t_R_a = w.Matrix([[tip_P_intermediate_error[0], -z[0], y[0], 0],
                          [tip_P_intermediate_error[1], -z[1], y[1], 0],
                          [tip_P_intermediate_error[2], -z[2], y[2], 0],
                          [0, 0, 0, 1]])
        t_R_a = w.normalize_rotation_matrix(t_R_a)

        # Apply rotation matrix on the fk of the tip link
        a_T_t = w.dot(w.inverse_frame(t_R_a),
                      self.get_fk_evaluated(self.tip_link, self.root_link),
                      self.get_fk(self.root_link, self.tip_link))
        expr_p = w.position_of(a_T_t)
        dist = w.norm(root_P_goal - root_P_tip)

        # self.add_debug_vector(self.tip_link + '_P_goal', tip_P_error)
        # self.add_debug_matrix(self.tip_link + '_R_frame', t_R_a)
        # self.add_debug_matrix(self.tip_link + '_T_a', w.inverse_frame(a_T_t))
        # self.add_debug_expr('error', dist)

        self.add_constraint_vector(reference_velocities=[self.reference_velocity] * 3,
                                   lower_errors=[dist, 0, 0],
                                   upper_errors=[dist, 0, 0],
                                   weights=[WEIGHT_ABOVE_CA, WEIGHT_ABOVE_CA * 2, WEIGHT_ABOVE_CA * 2],
                                   expressions=expr_p[:3],
                                   name_suffixes=['{}/x'.format('line'),
                                                  '{}/y'.format('line'),
                                                  '{}/z'.format('line')])

        if self.max_velocity is not None:
            self.add_translational_velocity_limit(frame_P_current=root_P_tip,
                                                  max_velocity=self.max_velocity,
                                                  weight=self.weight)


class CartesianPose(Goal):
    def __init__(self, root_link: str, tip_link: str, goal_pose: PoseStamped, max_linear_velocity: float = 0.1,
                 max_angular_velocity: float = 0.5, weight: float = WEIGHT_ABOVE_CA, root_link2: str = None, **kwargs):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose
        :param root_link: str, name of the root link of the kin chain
        :param tip_link: str, name of the tip link of the kin chain
        :param goal_pose: PoseStamped as json
        :param max_linear_velocity: float, m/s, default 0.1
        :param max_angular_velocity: float, rad/s, default 0.5
        :param weight: float, default WEIGHT_ABOVE_CA
        """
        super().__init__(**kwargs)
        goal_point = PointStamped()
        goal_point.header = goal_pose.header
        goal_point.point = goal_pose.pose.position
        self.add_constraints_of_goal(CartesianPosition(root_link=root_link,
                                                       tip_link=tip_link,
                                                       goal_point=goal_point,
                                                       max_velocity=max_linear_velocity,
                                                       weight=weight,
                                                       root_link2=root_link2,
                                                       **kwargs))
        goal_orientation = QuaternionStamped()
        goal_orientation.header = goal_pose.header
        goal_orientation.quaternion = goal_pose.pose.orientation
        self.add_constraints_of_goal(CartesianOrientation(root_link=root_link,
                                                          tip_link=tip_link,
                                                          goal_orientation=goal_orientation,
                                                          max_velocity=max_angular_velocity,
                                                          weight=weight,
                                                          root_link2=root_link2,
                                                          **kwargs))


class DiffDriveBaseGoal(Goal):

    def __init__(self, root_link: str, tip_link: str, goal_pose: PoseStamped, max_linear_velocity: float = 0.1,
                 max_angular_velocity: float = 0.5, weight: float = WEIGHT_ABOVE_CA, pointing_axis=None, **kwargs):
        super().__init__(**kwargs)
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        if pointing_axis is None:
            pointing_axis = Vector3Stamped()
            pointing_axis.header.frame_id = tip_link
            pointing_axis.vector.x = 1
        self.weight = weight
        self.map = root_link
        self.base_footprint = tip_link
        self.goal_pose = self.transform_msg(self.map, goal_pose)
        diff_drive_joints = [v for k, v in self.world.joints.items() if isinstance(v, DiffDrive)]
        assert len(diff_drive_joints) == 1
        self.joint: DiffDrive = diff_drive_joints[0]

        if pointing_axis is not None:
            self.base_footprint_V_pointing_axis = self.transform_msg(self.base_footprint, pointing_axis)
            self.base_footprint_V_pointing_axis.vector = normalize(self.base_footprint_V_pointing_axis.vector)
        else:
            self.base_footprint_V_pointing_axis = Vector3Stamped()
            self.base_footprint_V_pointing_axis.header.frame_id = self.base_footprint
            self.base_footprint_V_pointing_axis.vector.z = 1

    def make_constraints(self):
        map_T_base_footprint = self.get_fk(self.map, self.base_footprint)
        map_P_base_footprint = w.position_of(map_T_base_footprint)
        map_R_base_footprint = w.rotation_of(map_T_base_footprint)
        map_T_base_footprint_goal = w.ros_msg_to_matrix(self.goal_pose)
        map_P_base_footprint_goal = w.position_of(map_T_base_footprint_goal)
        map_R_base_footprint_goal = w.rotation_of(map_T_base_footprint_goal)
        base_footprint_V_pointing_axis = w.ros_msg_to_matrix(self.base_footprint_V_pointing_axis)

        map_V_goal_x = map_P_base_footprint_goal - map_P_base_footprint
        distance = w.norm(map_V_goal_x)

        map_V_pointing_axis = w.dot(map_T_base_footprint, base_footprint_V_pointing_axis)
        # map_goal_angle1 = w.angle_between_vector(map_V_goal_x, map_V_pointing_axis)
        axis, map_current_angle = w.axis_angle_from_matrix(map_R_base_footprint)
        map_current_angle = w.if_greater_zero(axis[2], map_current_angle, -map_current_angle)
        # rot_vel_symbol = self.joint.rot_vel.get_symbol(0)
        # map_current_angle = self.joint.rot.get_symbol(0)

        map_V_goal_x = w.scale(map_V_goal_x, 1)
        # map_V_z = w.vector3(0, 0, 1)
        # map_V_goal_y = w.cross(map_V_z, map_V_goal_x)
        # map_V_goal_y = w.scale(map_V_goal_y, 1)
        # map_R_goal = w.hstack([map_V_goal_x, map_V_goal_y, map_V_z, w.Matrix([0, 0, 0, 1])])
        # axis, map_goal_angle1 = w.axis_angle_from_matrix(map_R_goal)
        map_goal_angle1 = w.angle_between_vector(map_V_goal_x, w.vector3(1, 0, 0))
        map_goal_angle1 = w.if_greater_zero(map_V_goal_x[1], map_goal_angle1, -map_goal_angle1)
        # map_goal_angle1_2 = w.angle_between_vector(w.vector3(1, 0, 0), map_V_goal_x)
        angle_error1 = map_goal_angle1 - map_current_angle

        axis2, map_goal_angle2 = w.axis_angle_from_matrix(map_R_base_footprint_goal)
        map_goal_angle2 = w.if_greater_zero(axis2[2], map_goal_angle2, -map_goal_angle2)
        angle_error2 = map_goal_angle2 - map_current_angle

        eps = 0.01
        weight_rotate_to_goal = w.ca.if_else(w.logic_and(w.ca.ge(w.abs(angle_error1), eps),
                                                         w.ca.ge(w.abs(distance), eps)),
                                             self.weight,
                                             0)
        # weight_translation_raw = w.if_greater_eq(w.abs(distance), eps, self.weight, 0)
        weight_translation = w.ca.if_else(w.logic_and(w.ca.le(w.abs(angle_error1), eps * 2),
                                                      w.ca.ge(w.abs(distance), eps)),
                                          self.weight,
                                          0)
        # weight_translation = w.if_less_eq(weight_rotate_to_goal, eps, self.weight, 0)
        weight_final_rotation = w.ca.if_else(w.logic_and(w.ca.le(w.abs(distance), eps * 2),
                                                         w.ca.ge(w.abs(angle_error2), eps)),
                                             self.weight,
                                             0)

        self.add_debug_expr('map_goal_angle1', map_goal_angle1)
        self.add_debug_expr('map_current_angle', map_current_angle)
        self.add_debug_expr('angle_error1', angle_error1)
        self.add_debug_expr('weight_rotate_to_goal', weight_rotate_to_goal / 1000)
        self.add_debug_expr('distance', distance)
        self.add_debug_expr('weight_translation', weight_translation / 1000)
        self.add_debug_expr('angle_error2', angle_error2)
        self.add_debug_expr('weight_final_rotation', weight_final_rotation / 1000)

        # self.add_vector_goal_constraints(frame_V_current=map_V_pointing_axis,
        #                                  frame_V_goal=map_V_goal_x,
        #                                  reference_velocity=self.max_angular_velocity,
        #                                  weight=weight)

        self.add_constraint(reference_velocity=self.max_angular_velocity,
                            lower_error=angle_error1,
                            upper_error=angle_error1,
                            weight=weight_rotate_to_goal,
                            expression=map_current_angle,
                            name_suffix='/rot1')
        self.add_constraint(reference_velocity=self.max_linear_velocity,
                            lower_error=-distance,
                            upper_error=-distance,
                            weight=weight_translation,
                            expression=distance,
                            name_suffix='/dist')
        self.add_constraint(reference_velocity=self.max_angular_velocity,
                            lower_error=angle_error2,
                            upper_error=angle_error2,
                            weight=weight_final_rotation,
                            expression=map_current_angle,
                            name_suffix='/rot2')

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.map}/{self.base_footprint}'


class CartesianPoseStraight(Goal):
    def __init__(self, root_link: str, tip_link: str, goal_pose: PoseStamped, max_linear_velocity: float = 0.1,
                 max_angular_velocity: float = 0.5, weight: float = WEIGHT_ABOVE_CA, **kwargs):
        super(CartesianPoseStraight, self).__init__(**kwargs)
        goal_point = PointStamped()
        goal_point.header = goal_pose.header
        goal_point.point = goal_pose.pose.position
        self.add_constraints_of_goal(CartesianPositionStraight(root_link=root_link,
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
        super().__init__(**kwargs)

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
        s = super().__str__()
        return f'{s}/{self.root_link}/{self.tip_link}'


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
        super().__init__(**kwargs)
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
