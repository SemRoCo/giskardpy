from __future__ import division

from typing import Optional

import numpy as np
from geometry_msgs.msg import PointStamped, PoseStamped, QuaternionStamped
from geometry_msgs.msg import Vector3Stamped
from tf.transformations import rotation_from_matrix

from giskardpy import casadi_wrapper as w
from giskardpy.goals.goal import Goal, WEIGHT_ABOVE_CA, WEIGHT_BELOW_CA
from giskardpy.model.joints import DiffDrive, OmniDrivePR22
from giskardpy.my_types import Derivatives
from giskardpy.utils import logging
from giskardpy.utils.tfwrapper import normalize


class CartesianPosition(Goal):
    def __init__(self, root_link: str, tip_link: str, goal_point: PointStamped,
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 max_velocity: Optional[float] = None,
                 reference_velocity: Optional[float] = None,
                 weight: float = WEIGHT_ABOVE_CA, root_link2: str = None):
        """
        See CartesianPose.
        """
        super().__init__()
        if reference_velocity is None:
            reference_velocity = 0.2
        if reference_velocity is None:
            reference_velocity = max_velocity
        if isinstance(goal_point, PoseStamped):
            logging.logwarn('deprecated warning: CartesianPosition called with PoseStamped instead of PointStamped')
            p = PointStamped()
            p.header = goal_point.header
            p.point = goal_point.pose.position
            goal_point = p
        self.root_link = self.world.search_for_link_name(root_link, root_group)
        self.tip_link = self.world.search_for_link_name(tip_link, tip_group)
        if root_link2 is not None:
            self.root_link2 = self.world.search_for_link_name(root_link2, root_group)
            self.goal_point = self.transform_msg(self.root_link2, goal_point)
        else:
            self.root_link2 = None
            self.goal_point = self.transform_msg(self.root_link, goal_point)
        self.reference_velocity = reference_velocity
        self.max_velocity = max_velocity
        self.weight = weight
        if self.max_velocity is not None:
            self.add_constraints_of_goal(TranslationVelocityLimit(root_link=root_link,
                                                                  root_group=root_group,
                                                                  tip_link=tip_link,
                                                                  tip_group=tip_group,
                                                                  weight=weight,
                                                                  max_velocity=max_velocity,
                                                                  hard=False))

    @profile
    def make_constraints(self):
        r_P_g = w.Point3(self.goal_point)
        r_P_c = self.get_fk(self.root_link, self.tip_link).to_position()
        if self.root_link2 is not None:
            root_link2_T_root_link = self.get_fk_evaluated(self.root_link2, self.root_link)
            r_P_c = root_link2_T_root_link.dot(r_P_c)
        # self.add_debug_expr('trans', w.norm(r_P_c))
        self.add_point_goal_constraints(frame_P_goal=r_P_g,
                                        frame_P_current=r_P_c,
                                        reference_velocity=self.reference_velocity,
                                        weight=self.weight)

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.root_link}/{self.tip_link}'


class CartesianOrientation(Goal):
    def __init__(self,
                 root_link: str,
                 tip_link: str,
                 goal_orientation: QuaternionStamped,
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 reference_velocity: Optional[float] = None,
                 max_velocity: Optional[float] = None,
                 weight: float = WEIGHT_ABOVE_CA,
                 root_link2: str = None):
        """
        See CartesianPose.
        """
        super().__init__()
        if reference_velocity is None:
            reference_velocity = max_velocity
        if reference_velocity is None:
            reference_velocity = 0.5
        if isinstance(goal_orientation, PoseStamped):
            logging.logwarn(
                'deprecated warning: CartesianOrientation called with PoseStamped instead of QuaternionStamped')
            q = QuaternionStamped()
            q.header = goal_orientation.header
            q.quaternion = goal_orientation.pose.orientation
            goal_orientation = q
        self.root_link = self.world.search_for_link_name(root_link, root_group)
        self.tip_link = self.world.search_for_link_name(tip_link, tip_group)
        if root_link2 is not None:
            self.root_link2 = self.world.search_for_link_name(root_link2, root_group)
            self.goal_orientation = self.transform_msg(self.root_link2, goal_orientation)
        else:
            self.root_link2 = None
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
        #                                                        prefix=prefix,
        #                                                        **kwargs))

    def make_constraints(self):
        r_R_g = w.RotationMatrix(self.goal_orientation)
        r_R_c = self.get_fk(self.root_link, self.tip_link).to_rotation()
        if self.root_link2 is not None:
            c_R_r_eval = self.get_fk_evaluated(self.tip_link, self.root_link2).to_rotation()
            root_link2_T_root_link = self.get_fk_evaluated(self.root_link2, self.root_link)
            # self.add_debug_matrix('root_link2_T_root_link', root_link2_T_root_link)
            r_R_c = root_link2_T_root_link.dot(r_R_c)
        else:
            c_R_r_eval = self.get_fk_evaluated(self.tip_link, self.root_link).to_rotation()
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
    def __init__(self,
                 root_link: str,
                 tip_link: str,
                 goal_point: PointStamped,
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 reference_velocity: Optional[float] = None,
                 max_velocity: Optional[float] = None,
                 weight: float = WEIGHT_ABOVE_CA):
        """
        Same as CartesianPosition, but tries to move the tip_link in a straight line to the goal_point.
        """
        super().__init__()
        if reference_velocity is None:
            reference_velocity = max_velocity
        if reference_velocity is None:
            reference_velocity = 0.2
        self.reference_velocity = reference_velocity
        self.max_velocity = max_velocity
        self.weight = weight
        self.root_link = self.world.search_for_link_name(root_link, root_group)
        self.tip_link = self.world.search_for_link_name(tip_link, tip_group)
        self.goal_point = self.transform_msg(self.root_link, goal_point)

    def make_constraints(self):
        root_P_goal = w.Point3(self.goal_point)
        root_P_tip = self.get_fk(self.root_link, self.tip_link).to_position()
        t_T_r = self.get_fk(self.tip_link, self.root_link)
        tip_P_goal = t_T_r.dot(root_P_goal)

        # Create rotation matrix, which rotates the tip link frame
        # such that its x-axis shows towards the goal position.
        # The goal frame is called 'a'.
        # Thus, the rotation matrix is called t_R_a.
        tip_V_error = w.Vector3(tip_P_goal)
        trans_error = tip_V_error.norm()
        # x-axis
        tip_V_intermediate_error = w.save_division(tip_V_error, trans_error)
        # y- and z-axis
        tip_V_intermediate_y = w.Vector3(np.random.random((3,)))
        tip_V_intermediate_y.scale(1)
        y = tip_V_intermediate_error.cross(tip_V_intermediate_y)
        z = tip_V_intermediate_error.cross(y)
        t_R_a = w.RotationMatrix.from_vectors(x=tip_V_intermediate_error, y=-z, z=y)

        # Apply rotation matrix on the fk of the tip link
        a_T_t = t_R_a.inverse().dot(
            self.get_fk_evaluated(self.tip_link, self.root_link)).dot(
            self.get_fk(self.root_link, self.tip_link))
        expr_p = a_T_t.to_position()
        dist = w.norm(root_P_goal - root_P_tip)

        # self.add_debug_vector(self.tip_link + '_P_goal', tip_P_error)
        # self.add_debug_matrix(self.tip_link + '_R_frame', t_R_a)
        # self.add_debug_matrix(self.tip_link + '_T_a', w.inverse_frame(a_T_t))
        # self.add_debug_expr('error', dist)

        self.add_equality_constraint_vector(reference_velocities=[self.reference_velocity] * 3,
                                            equality_bounds=[dist, 0, 0],
                                            weights=[WEIGHT_ABOVE_CA, WEIGHT_ABOVE_CA * 2, WEIGHT_ABOVE_CA * 2],
                                            task_expression=expr_p[:3],
                                            names=['line/x',
                                                   'line/y',
                                                   'line/z'])

        if self.max_velocity is not None:
            self.add_translational_velocity_limit(frame_P_current=root_P_tip,
                                                  max_velocity=self.max_velocity,
                                                  weight=self.weight)

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.root_link}/{self.tip_link}'


class CartesianPose(Goal):
    def __init__(self, root_link: str, tip_link: str, goal_pose: PoseStamped,
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 max_linear_velocity: Optional[float] = None,
                 max_angular_velocity: Optional[float] = None,
                 reference_linear_velocity: Optional[float] = None,
                 reference_angular_velocity: Optional[float] = None,
                 weight=WEIGHT_ABOVE_CA, root_link2: Optional[str] = None):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose.
        The max velocities enforce a strict limit, but require a lot of additional constraints, thus making the
        system noticeably slower.
        The reference velocities don't enforce a strict limit, but also don't require any additional constraints.
        :param root_link: name of the root link of the kin chain
        :param tip_link: name of the tip link of the kin chain
        :param goal_pose: the goal pose
        :param root_group: a group name, where to search for root_link, only required to avoid name conflicts
        :param tip_group: a group name, where to search for tip_link, only required to avoid name conflicts
        :param max_linear_velocity: m/s
        :param max_angular_velocity: rad/s
        :param reference_linear_velocity: m/s
        :param reference_angular_velocity: rad/s
        :param weight: default WEIGHT_ABOVE_CA
        :param root_link2: experimental, don't use
        """
        self.root_link = root_link
        self.tip_link = tip_link
        super().__init__()
        goal_point = PointStamped()
        goal_point.header = goal_pose.header
        goal_point.point = goal_pose.pose.position
        self.add_constraints_of_goal(CartesianPosition(root_link=root_link,
                                                       root_group=root_group,
                                                       tip_link=tip_link,
                                                       tip_group=tip_group,
                                                       goal_point=goal_point,
                                                       max_velocity=max_linear_velocity,
                                                       reference_velocity=reference_linear_velocity,
                                                       weight=weight,
                                                       root_link2=root_link2))
        goal_orientation = QuaternionStamped()
        goal_orientation.header = goal_pose.header
        goal_orientation.quaternion = goal_pose.pose.orientation
        self.add_constraints_of_goal(CartesianOrientation(root_link=root_link,
                                                          root_group=root_group,
                                                          tip_link=tip_link,
                                                          tip_group=tip_group,
                                                          goal_orientation=goal_orientation,
                                                          max_velocity=max_angular_velocity,
                                                          reference_velocity=reference_angular_velocity,
                                                          weight=weight,
                                                          root_link2=root_link2))

    def make_constraints(self):
        pass

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.root_link}/{self.tip_link}'


class DiffDriveBaseGoal(Goal):

    def __init__(self, root_link: str, tip_link: str, goal_pose: PoseStamped, max_linear_velocity: float = 0.1,
                 max_angular_velocity: float = 0.5, weight: float = WEIGHT_ABOVE_CA, pointing_axis=None,
                 root_group: Optional[str] = None, tip_group: Optional[str] = None,
                 always_forward: bool = False):
        """
        Like a CartesianPose, but specifically for differential drives. It will achieve the goal in 3 phases.
        1. orient towards goal.
        2. drive to goal point.
        3. reach goal orientation.
        :param root_link: root link of the kinematic chain. typically map
        :param tip_link: tip link of the kinematic chain. typically base_footprint or similar
        :param goal_pose:
        :param max_linear_velocity:
        :param max_angular_velocity:
        :param weight:
        :param pointing_axis: the forward direction. default is x-axis
        :param always_forward: if false, it will drive backwards, if it requires less rotation.
        """
        super().__init__()
        self.always_forward = always_forward
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        if pointing_axis is None:
            pointing_axis = Vector3Stamped()
            pointing_axis.header.frame_id = tip_link
            pointing_axis.vector.x = 1
        self.weight = weight
        self.map = self.world.search_for_link_name(root_link, root_group)
        self.base_footprint = self.world.search_for_link_name(tip_link, tip_group)
        self.goal_pose = self.transform_msg(self.map, goal_pose)
        self.goal_pose.pose.position.z = 0
        diff_drive_joints = [v for k, v in self.world.joints.items() if isinstance(v, DiffDrive)]
        assert len(diff_drive_joints) == 1
        self.joint: DiffDrive = diff_drive_joints[0]
        self.odom = self.joint.parent_link_name

        if pointing_axis is not None:
            self.base_footprint_V_pointing_axis = self.transform_msg(self.base_footprint, pointing_axis)
            self.base_footprint_V_pointing_axis.vector = normalize(self.base_footprint_V_pointing_axis.vector)
        else:
            self.base_footprint_V_pointing_axis = Vector3Stamped()
            self.base_footprint_V_pointing_axis.header.frame_id = self.base_footprint
            self.base_footprint_V_pointing_axis.vector.z = 1

    def make_constraints(self):
        map_T_base_current = w.TransMatrix(self.world.compute_fk_np(self.map, self.base_footprint))
        map_T_odom_current = self.world.compute_fk_np(self.map, self.odom)
        map_odom_angle, _, _ = rotation_from_matrix(map_T_odom_current)
        map_R_base_current = map_T_base_current.to_rotation()
        axis_start, angle_start = map_R_base_current.to_axis_angle()
        angle_start = w.if_greater_zero(axis_start[2], angle_start, -angle_start)

        map_T_base_footprint = self.get_fk(self.map, self.base_footprint)
        map_P_base_footprint = map_T_base_footprint.to_position()
        # map_R_base_footprint = map_T_base_footprint.to_rotation()
        map_T_base_footprint_goal = w.TransMatrix(self.goal_pose)
        map_P_base_footprint_goal = map_T_base_footprint_goal.to_position()
        map_R_base_footprint_goal = map_T_base_footprint_goal.to_rotation()

        map_V_goal_x = map_P_base_footprint_goal - map_P_base_footprint
        distance = map_V_goal_x.norm()

        # axis, map_current_angle = map_R_base_footprint.to_axis_angle()
        # map_current_angle = w.if_greater_zero(axis.z, map_current_angle, -map_current_angle)
        odom_current_angle = self.joint.yaw.get_symbol(Derivatives.position)
        map_current_angle = map_odom_angle + odom_current_angle

        axis2, map_goal_angle2 = map_R_base_footprint_goal.to_axis_angle()
        map_goal_angle2 = w.if_greater_zero(axis2.z, map_goal_angle2, -map_goal_angle2)
        final_rotation_error = w.shortest_angular_distance(map_current_angle, map_goal_angle2)

        map_R_goal = w.RotationMatrix.from_vectors(x=map_V_goal_x, y=None, z=w.Vector3((0, 0, 1)))

        map_goal_angle_direction_f = map_R_goal.to_angle(lambda axis: axis[2])

        map_goal_angle_direction_b = w.if_less_eq(map_goal_angle_direction_f, 0,
                                                  if_result=map_goal_angle_direction_f + np.pi,
                                                  else_result=map_goal_angle_direction_f - np.pi)

        middle_angle = w.normalize_angle(
            map_goal_angle2 + w.shortest_angular_distance(map_goal_angle2, angle_start) / 2)

        middle_angle = self.god_map.evaluate_expr(middle_angle)
        a = self.god_map.evaluate_expr(w.shortest_angular_distance(map_goal_angle_direction_f, middle_angle))
        b = self.god_map.evaluate_expr(w.shortest_angular_distance(map_goal_angle_direction_b, middle_angle))
        eps = 0.01
        if self.always_forward:
            map_goal_angle1 = map_goal_angle_direction_f
        else:
            map_goal_angle1 = w.if_less_eq(w.abs(a) - w.abs(b), 0.03,
                                           if_result=map_goal_angle_direction_f,
                                           else_result=map_goal_angle_direction_b)
        rotate_to_goal_error = w.shortest_angular_distance(map_current_angle, map_goal_angle1)

        weight_final_rotation = w.if_else(w.logic_and(w.less_equal(w.abs(distance), eps * 2),
                                                      w.greater_equal(w.abs(final_rotation_error), 0)),
                                          self.weight,
                                          0)
        weight_rotate_to_goal = w.if_else(w.logic_and(w.greater_equal(w.abs(rotate_to_goal_error), eps),
                                                      w.greater_equal(w.abs(distance), eps),
                                                      w.less_equal(weight_final_rotation, eps)),
                                          self.weight,
                                          0)
        weight_translation = w.if_else(w.logic_and(w.less_equal(w.abs(rotate_to_goal_error), eps * 2),
                                                   w.greater_equal(w.abs(distance), eps)),
                                       self.weight,
                                       0)

        self.add_equality_constraint(reference_velocity=self.max_angular_velocity,
                                     equality_bound=rotate_to_goal_error,
                                     weight=weight_rotate_to_goal,
                                     task_expression=map_current_angle,
                                     name='/rot1')
        self.add_point_goal_constraints(frame_P_current=map_P_base_footprint,
                                        frame_P_goal=map_P_base_footprint_goal,
                                        reference_velocity=self.max_linear_velocity,
                                        weight=weight_translation)
        self.add_equality_constraint(reference_velocity=self.max_angular_velocity,
                                     equality_bound=final_rotation_error,
                                     weight=weight_final_rotation,
                                     task_expression=map_current_angle,
                                     name='/rot2')

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.map}/{self.base_footprint}'


class PR2DiffDriveBaseGoal(Goal):

    def __init__(self, goal_pose: PoseStamped, max_linear_velocity: float = 0.1,
                 max_angular_velocity: float = 0.5, weight: float = WEIGHT_ABOVE_CA, pointing_axis=None,
                 root_group: Optional[str] = None, tip_group: Optional[str] = None):
        super().__init__()
        self.max_angular_velocity = max_angular_velocity
        self.max_linear_velocity = max_linear_velocity
        diff_drive_joints = [v for k, v in self.world.joints.items() if isinstance(v, OmniDrivePR22)]
        assert len(diff_drive_joints) == 1
        self.joint: OmniDrivePR22 = diff_drive_joints[0]
        self.weight = weight
        self.root_link = self.joint.parent_link_name
        self.tip_link = self.joint.child_link_name
        self.root_T_goal = self.transform_msg(self.root_link, goal_pose)
        self.add_constraints_of_goal(CartesianPose(root_link=self.root_link,
                                                   tip_link=self.tip_link,
                                                   goal_pose=goal_pose,
                                                   reference_linear_velocity=self.max_linear_velocity,
                                                   reference_angular_velocity=self.max_angular_velocity,
                                                   weight=WEIGHT_BELOW_CA))

    def make_constraints(self):
        root_T_tip = self.get_fk(self.root_link, self.tip_link)
        root_P_tip = root_T_tip.to_position()

        root_T_goal = w.TransMatrix(self.root_T_goal)
        root_P_goal = root_T_goal.to_position()

        root_yaw1 = self.joint.yaw1_vel.get_symbol(Derivatives.position)
        root_V_forward = w.Vector3((w.cos(root_yaw1), w.sin(root_yaw1), 0))
        root_V_forward.vis_frame = self.tip_link

        root_V_goal = root_P_goal - root_P_tip
        root_V_goal.scale(1)
        root_V_goal.vis_frame = self.tip_link

        self.add_debug_expr('root_P_goal', root_P_goal)
        self.add_debug_expr('root_V_forward', root_V_forward)
        self.add_debug_expr('root_V_goal', root_V_goal)

        weight = w.if_greater(w.norm(root_P_goal - root_P_tip), 0.005, WEIGHT_ABOVE_CA, 0)

        self.add_vector_goal_constraints(frame_V_current=root_V_forward,
                                         frame_V_goal=root_V_goal,
                                         reference_velocity=self.max_angular_velocity,
                                         weight=weight,
                                         name='angle')

    def __str__(self) -> str:
        return super().__str__()


class CartesianPoseStraight(Goal):
    def __init__(self, root_link: str, tip_link: str, goal_pose: PoseStamped,
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 max_linear_velocity: Optional[float] = None,
                 max_angular_velocity: Optional[float] = None,
                 reference_linear_velocity: Optional[float] = None,
                 reference_angular_velocity: Optional[float] = None,
                 weight: float = WEIGHT_ABOVE_CA):
        """
        See CartesianPose. In contrast to it, this goal will try to move tip_link in a straight line.
        """
        self.root_link = root_link
        self.tip_link = tip_link
        super().__init__()
        goal_point = PointStamped()
        goal_point.header = goal_pose.header
        goal_point.point = goal_pose.pose.position
        self.add_constraints_of_goal(CartesianPositionStraight(root_link=root_link,
                                                               root_group=root_group,
                                                               tip_link=tip_link,
                                                               tip_group=tip_group,
                                                               goal_point=goal_point,
                                                               max_velocity=max_linear_velocity,
                                                               reference_velocity=reference_linear_velocity,
                                                               weight=weight))
        goal_orientation = QuaternionStamped()
        goal_orientation.header = goal_pose.header
        goal_orientation.quaternion = goal_pose.pose.orientation
        self.add_constraints_of_goal(CartesianOrientation(root_link=root_link,
                                                          root_group=root_group,
                                                          tip_link=tip_link,
                                                          tip_group=tip_group,
                                                          goal_orientation=goal_orientation,
                                                          max_velocity=max_angular_velocity,
                                                          reference_velocity=reference_angular_velocity,
                                                          weight=weight))

    def make_constraints(self):
        pass

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.root_link}/{self.tip_link}'


class TranslationVelocityLimit(Goal):
    def __init__(self, root_link: str, tip_link: str, root_group: Optional[str] = None, tip_group: Optional[str] = None,
                 weight=WEIGHT_ABOVE_CA, max_velocity=0.1, hard=True):
        """
        See CartesianVelocityLimit
        """
        super().__init__()
        self.root_link = self.world.search_for_link_name(root_link, root_group)
        self.tip_link = self.world.search_for_link_name(tip_link, tip_group)
        self.hard = hard
        self.weight = weight
        self.max_velocity = max_velocity

    def make_constraints(self):
        r_P_c = self.get_fk(self.root_link, self.tip_link).to_position()
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
        return f'{s}/{self.root_link}/{self.tip_link}'


class RotationVelocityLimit(Goal):
    def __init__(self, root_link: str, tip_link: str, root_group: Optional[str] = None, tip_group: Optional[str] = None,
                 weight=WEIGHT_ABOVE_CA, max_velocity=0.5, hard=True):
        """
        See CartesianVelocityLimit
        """

        super().__init__()
        self.root_link = self.world.search_for_link_name(root_link, root_group)
        self.tip_link = self.world.search_for_link_name(tip_link, tip_group)
        self.hard = hard

        self.weight = weight
        self.max_velocity = max_velocity

    def make_constraints(self):
        r_R_c = self.get_fk(self.root_link, self.tip_link).to_rotation()
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
    def __init__(self,
                 root_link: str,
                 tip_link: str,
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 max_linear_velocity: float = 0.1,
                 max_angular_velocity: float = 0.5,
                 weight: float = WEIGHT_ABOVE_CA,
                 hard: bool = False):
        """
        This goal will use put a strict limit on the Cartesian velocity. This will require a lot of constraints, thus
        slowing down the system noticeably.
        :param root_link: root link of the kinematic chain
        :param tip_link: tip link of the kinematic chain
        :param root_group: if the root_link is not unique, use this to say to which group the link belongs
        :param tip_group: if the tip_link is not unique, use this to say to which group the link belongs
        :param max_linear_velocity: m/s
        :param max_angular_velocity: rad/s
        :param weight: default WEIGHT_ABOVE_CA
        :param hard: Turn this into a hard constraint. This make create unsolvable optimization problems
        """
        self.root_link = root_link
        self.tip_link = tip_link
        super().__init__()
        self.add_constraints_of_goal(TranslationVelocityLimit(root_link=root_link,
                                                              root_group=root_group,
                                                              tip_link=tip_link,
                                                              tip_group=tip_group,
                                                              max_velocity=max_linear_velocity,
                                                              weight=weight,
                                                              hard=hard))
        self.add_constraints_of_goal(RotationVelocityLimit(root_link=root_link,
                                                           root_group=root_group,
                                                           tip_link=tip_link,
                                                           tip_group=tip_group,
                                                           max_velocity=max_angular_velocity,
                                                           weight=weight,
                                                           hard=hard))

    def make_constraints(self):
        pass

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.root_link}/{self.tip_link}'
