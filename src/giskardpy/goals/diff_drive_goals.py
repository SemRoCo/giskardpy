from __future__ import division

from typing import Optional

from geometry_msgs.msg import Vector3Stamped, PointStamped

import giskardpy.utils.tfwrapper as tf
from giskardpy import casadi_wrapper as w
from giskardpy.goals.goal import Goal, WEIGHT_ABOVE_CA


class DiffDriveTangentialToPoint(Goal):

    def __init__(self, goal_point: PointStamped, forward: Optional[Vector3Stamped] = None,
                 group_name: Optional[str] = None,
                 reference_velocity: float = 0.5, weight: bool = WEIGHT_ABOVE_CA, drive: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.goal_point = self.transform_msg(self.world.root_link_name, goal_point)
        self.goal_point.point.z = 0
        self.weight = weight
        self.tip = self.world.get_link_name('base_footprint', group_name)
        self.root = self.world.root_link_name
        self.drive = drive
        if forward is not None:
            self.tip_V_pointing_axis = tf.transform_vector(self.tip, forward)
            self.tip_V_pointing_axis.vector = tf.normalize(self.tip_V_pointing_axis.vector)
        else:
            self.tip_V_pointing_axis = Vector3Stamped()
            self.tip_V_pointing_axis.header.frame_id = self.tip
            self.tip_V_pointing_axis.vector.x = 1

    def make_constraints(self):
        map_P_center = w.ros_msg_to_matrix(self.goal_point)
        map_T_base = self.get_fk(self.root, self.tip)
        map_P_base = w.position_of(map_T_base)
        map_V_base_to_center = map_P_center - map_P_base
        map_V_base_to_center = w.scale(map_V_base_to_center, 1)
        map_V_up = w.Matrix([0, 0, 1, 0])
        map_V_tangent = w.cross(map_V_base_to_center, map_V_up)
        tip_V_pointing_axis = w.ros_msg_to_matrix(self.tip_V_pointing_axis)
        map_V_forward = w.dot(map_T_base, tip_V_pointing_axis)

        if self.drive:
            angle = w.abs(w.angle_between_vector(map_V_forward, map_V_tangent))
            self.add_constraint(reference_velocity=0.5,
                                lower_error=-angle,
                                upper_error=-angle,
                                weight=self.weight,
                                expression=angle,
                                name_suffix='/rot')
        else:
            # angle = w.abs(w.angle_between_vector(w.vector3(1,0,0), map_V_tangent))
            map_R_goal = w.rotation_matrix_from_vectors(x=map_V_tangent, y=None, z=w.vector3(0, 0, 1))
            goal_angle = w.angle_from_matrix(map_R_goal, lambda axis: axis[2])
            map_R_base = w.rotation_of(map_T_base)
            axis, map_current_angle = w.axis_angle_from_matrix(map_R_base)
            map_current_angle = w.if_greater_zero(axis[2], map_current_angle, -map_current_angle)
            angle_error = w.shortest_angular_distance(map_current_angle, goal_angle)
            self.add_constraint(reference_velocity=0.5,
                                lower_error=angle_error,
                                upper_error=angle_error,
                                weight=self.weight,
                                expression=map_current_angle,
                                name_suffix='/rot')


class PointingDiffDriveEEF(Goal):
    def __init__(self, base_tip, base_root, eef_tip, eef_root, pointing_axis=None, max_velocity=0.3,
                 weight=WEIGHT_ABOVE_CA, **kwargs):
        super().__init__(**kwargs)
        self.weight = weight
        self.max_velocity = max_velocity
        self.base_tip = base_tip
        self.base_root = base_root
        self.eef_tip = eef_tip
        self.eef_root = eef_root

        if pointing_axis is not None:
            self.tip_V_pointing_axis = tf.transform_vector(self.base_tip, pointing_axis)
            self.tip_V_pointing_axis.vector = tf.normalize(self.tip_V_pointing_axis.vector)
        else:
            self.tip_V_pointing_axis = Vector3Stamped()
            self.tip_V_pointing_axis.header.frame_id = self.base_tip
            self.tip_V_pointing_axis.vector.x = 1

    def make_constraints(self):
        fk_vel = self.get_fk_velocity(self.eef_root, self.eef_tip)
        eef_root_V_eef_tip = w.vector3(fk_vel[0], fk_vel[1], 0)
        eef_root_V_eef_tip_normed = w.scale(eef_root_V_eef_tip, 1)
        base_root_T_eef_root = self.get_fk(self.base_root, self.eef_root)
        base_root_V_eef_tip = w.dot(base_root_T_eef_root, eef_root_V_eef_tip_normed)

        tip_V_pointing_axis = w.ros_msg_to_matrix(self.tip_V_pointing_axis)
        base_root_T_base_tip = self.get_fk(self.base_root, self.base_tip)
        base_root_V_pointing_axis = w.dot(base_root_T_base_tip, tip_V_pointing_axis)

        # weight = w.if_less_eq(distance, 0.05, WEIGHT_BELOW_CA, WEIGHT_ABOVE_CA)
        # weight = WEIGHT_BELOW_CA
        # self.add_debug_expr('fk_vel/x', fk_vel[0])
        # self.add_debug_expr('fk_vel/y', fk_vel[1])
        # self.add_debug_vector('base_root_V_eef_tip', base_root_V_eef_tip)
        # self.add_debug_vector('eef_root_V_eef_tip', eef_root_V_eef_tip)
        # self.add_debug_vector('base_root_V_pointing_axis', base_root_V_pointing_axis)
        weight = WEIGHT_ABOVE_CA * w.norm(eef_root_V_eef_tip_normed)

        self.add_vector_goal_constraints(frame_V_current=base_root_V_pointing_axis,
                                         frame_V_goal=base_root_V_eef_tip,
                                         reference_velocity=self.max_velocity,
                                         weight=weight)

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.eef_root}/{self.eef_tip}'


class KeepHandInWorkspace(Goal):
    def __init__(self, tip_link, base_footprint=None, map_frame=None, pointing_axis=None, max_velocity=0.3,
                 group_name: Optional[str] = None, weight=WEIGHT_ABOVE_CA, **kwargs):
        super().__init__(**kwargs)
        if base_footprint is None:
            base_footprint = 'base_footprint'
        base_footprint = self.world.get_link_name(base_footprint, group_name)
        if map_frame is None:
            map_frame = self.world.root_link_name
        self.weight = weight
        self.max_velocity = max_velocity
        self.map_frame = map_frame
        self.tip_link = self.world.get_link_name(tip_link, group_name)
        self.base_footprint = base_footprint

        if pointing_axis is not None:
            self.map_V_pointing_axis = tf.transform_vector(self.base_footprint, pointing_axis)
            self.map_V_pointing_axis.vector = tf.normalize(self.map_V_pointing_axis.vector)
        else:
            self.map_V_pointing_axis = Vector3Stamped()
            self.map_V_pointing_axis.header.frame_id = self.map_frame
            self.map_V_pointing_axis.vector.x = 1

    def make_constraints(self):
        weight = WEIGHT_ABOVE_CA
        base_footprint_V_pointing_axis = w.ros_msg_to_matrix(self.map_V_pointing_axis)
        map_T_base_footprint = self.get_fk(self.map_frame, self.base_footprint)
        map_V_pointing_axis = w.dot(map_T_base_footprint, base_footprint_V_pointing_axis)
        map_T_tip = self.get_fk(self.map_frame, self.tip_link)
        map_V_tip = w.position_of(map_T_tip)
        map_V_tip[2] = 0
        map_V_tip[3] = 0
        map_P_tip = w.position_of(map_T_tip)
        map_P_tip[2] = 0
        map_P_base_footprint = w.position_of(map_T_base_footprint)
        map_P_base_footprint[2] = 0
        base_footprint_V_tip = map_P_tip - map_P_base_footprint
        distance_to_base = w.norm(base_footprint_V_tip)

        map_V_tip = w.scale(map_V_tip, 1)
        angle_error = w.angle_between_vector(base_footprint_V_tip, map_V_pointing_axis)
        # self.add_debug_expr('rot', angle_error)
        self.add_constraint(reference_velocity=0.5,
                            lower_error=-angle_error - 0.2,
                            upper_error=-angle_error + 0.2,
                            weight=weight,
                            expression=angle_error,
                            name_suffix='/rot')
        # self.add_vector_goal_constraints(frame_V_current=map_V_pointing_axis,
        #                                  frame_V_goal=base_footprint_V_tip,
        #                                  reference_velocity=0.5)

        # self.add_debug_expr('distance_to_base', distance_to_base)
        # self.add_constraint(reference_velocity=0.1,
        #                     lower_error=-distance_to_base + 0.35,
        #                     upper_error=-distance_to_base + 0.6,
        #                     weight=weight,
        #                     expression=distance_to_base,
        #                     name_suffix='/dist')

        # fk_vel = self.get_fk_velocity(self.base_footprint, self.tip_link)
        # eef_root_V_eef_tip = w.vector3(fk_vel[0], fk_vel[1], 0)
        # eef_root_V_eef_tip_normed = w.scale(eef_root_V_eef_tip, 1)
        # base_root_T_eef_root = self.get_fk(self.map_frame, self.base_footprint)
        # base_root_V_eef_tip = w.dot(base_root_T_eef_root, eef_root_V_eef_tip_normed)
        #
        # base_root_T_base_tip = self.get_fk(self.map_frame, self.base_tip)
        # base_root_V_pointing_axis = w.dot(base_root_T_base_tip, tip_V_pointing_axis)
        #
        # weight = WEIGHT_ABOVE_CA * w.norm(eef_root_V_eef_tip_normed)
        #
        # self.add_vector_goal_constraints(frame_V_current=base_root_V_pointing_axis,
        #                                  frame_V_goal=base_root_V_eef_tip,
        #                                  reference_velocity=self.max_velocity,
        #                                  weight=weight)

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.base_footprint}/{self.tip_link}'
