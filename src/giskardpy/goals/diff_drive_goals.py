from __future__ import division

from typing import Optional

from geometry_msgs.msg import Vector3Stamped, PointStamped

import giskardpy.utils.tfwrapper as tf
from giskardpy import casadi_wrapper as w
from giskardpy.goals.goal import Goal, WEIGHT_ABOVE_CA
from giskardpy.model.joints import OmniDrivePR22
from giskardpy.my_types import Derivatives


class DiffDriveTangentialToPoint(Goal):

    def __init__(self, goal_point: PointStamped, forward: Optional[Vector3Stamped] = None,
                 group_name: Optional[str] = None,
                 reference_velocity: float = 0.5, weight: bool = WEIGHT_ABOVE_CA, drive: bool = False):
        super().__init__()
        self.tip = self.world.search_for_link_name('base_footprint', group_name)
        self.root = self.world.root_link_name
        self.goal_point = self.transform_msg(self.world.root_link_name, goal_point)
        self.goal_point.point.z = 0
        self.weight = weight
        self.drive = drive
        if forward is not None:
            self.tip_V_pointing_axis = tf.transform_vector(self.tip, forward)
            self.tip_V_pointing_axis.vector = tf.normalize(self.tip_V_pointing_axis.vector)
        else:
            self.tip_V_pointing_axis = Vector3Stamped()
            self.tip_V_pointing_axis.header.frame_id = self.tip
            self.tip_V_pointing_axis.vector.x = 1

    def make_constraints(self):
        map_P_center = w.Point3(self.goal_point)
        map_T_base = self.get_fk(self.root, self.tip)
        map_P_base = map_T_base.to_position()
        map_V_base_to_center = map_P_center - map_P_base
        map_V_base_to_center = w.scale(map_V_base_to_center, 1)
        map_V_up = w.Expression([0, 0, 1, 0])
        map_V_tangent = w.cross(map_V_base_to_center, map_V_up)
        tip_V_pointing_axis = w.Vector3(self.tip_V_pointing_axis)
        map_V_forward = w.dot(map_T_base, tip_V_pointing_axis)

        if self.drive:
            angle = w.abs(w.angle_between_vector(map_V_forward, map_V_tangent))
            self.add_equality_constraint(reference_velocity=0.5,
                                         equality_bound=-angle,
                                         weight=self.weight,
                                         task_expression=angle,
                                         name='/rot')
        else:
            # angle = w.abs(w.angle_between_vector(w.vector3(1,0,0), map_V_tangent))
            map_R_goal = w.RotationMatrix.from_vectors(x=map_V_tangent, y=None, z=w.Vector3((0, 0, 1)))
            goal_angle = map_R_goal.to_angle(lambda axis: axis[2])
            map_R_base = map_T_base.to_rotation()
            axis, map_current_angle = map_R_base.to_axis_angle()
            map_current_angle = w.if_greater_zero(axis[2], map_current_angle, -map_current_angle)
            angle_error = w.shortest_angular_distance(map_current_angle, goal_angle)
            self.add_equality_constraint(reference_velocity=0.5,
                                         equality_bound=angle_error,
                                         weight=self.weight,
                                         task_expression=map_current_angle,
                                         name='/rot')

    def __str__(self) -> str:
        return f'{super().__str__()}/{self.root}/{self.tip}'


class PointingDiffDriveEEF(Goal):
    def __init__(self, base_tip, base_root, eef_tip, eef_root, pointing_axis=None, max_velocity=0.3,
                 weight=WEIGHT_ABOVE_CA):
        super().__init__()
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
        eef_root_V_eef_tip = w.Vector3((fk_vel[0], fk_vel[1], 0))
        eef_root_V_eef_tip_normed = w.scale(eef_root_V_eef_tip, 1)
        base_root_T_eef_root = self.get_fk(self.base_root, self.eef_root)
        base_root_V_eef_tip = w.dot(base_root_T_eef_root, eef_root_V_eef_tip_normed)

        tip_V_pointing_axis = w.Vector3(self.tip_V_pointing_axis)
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
                 group_name: Optional[str] = None, weight=WEIGHT_ABOVE_CA):
        super().__init__()
        if base_footprint is None:
            base_footprint = 'base_footprint'
        base_footprint = self.world.search_for_link_name(base_footprint, group_name)
        if map_frame is None:
            map_frame = self.world.root_link_name
        self.weight = weight
        self.max_velocity = max_velocity
        self.map_frame = map_frame
        self.tip_link = self.world.search_for_link_name(tip_link, group_name)
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
        base_footprint_V_pointing_axis = w.Vector3(self.map_V_pointing_axis)
        map_T_base_footprint = self.get_fk(self.map_frame, self.base_footprint)
        map_V_pointing_axis = w.dot(map_T_base_footprint, base_footprint_V_pointing_axis)
        map_T_tip = self.get_fk(self.map_frame, self.tip_link)
        map_V_tip = w.Vector3(map_T_tip.to_position())
        map_V_tip.y = 0
        map_V_tip.z = 0
        map_P_tip = map_T_tip.to_position()
        map_P_tip.z = 0
        map_P_base_footprint = map_T_base_footprint.to_position()
        map_P_base_footprint.z = 0
        base_footprint_V_tip = map_P_tip - map_P_base_footprint
        # distance_to_base = w.norm(base_footprint_V_tip)

        map_V_tip.scale(1)
        angle_error = w.angle_between_vector(base_footprint_V_tip, map_V_pointing_axis)
        # self.add_debug_expr('rot', angle_error)
        self.add_inequality_constraint(reference_velocity=0.5,
                                       lower_error=-angle_error - 0.2,
                                       upper_error=-angle_error + 0.2,
                                       weight=weight,
                                       task_expression=angle_error,
                                       name='/rot')
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


class PR2DiffDriveOrient(Goal):

    def __init__(self, eef_link, max_linear_velocity: float = 0.1,
                 max_angular_velocity: float = 0.5, weight: float = WEIGHT_ABOVE_CA, pointing_axis=None,
                 root_group: Optional[str] = None, tip_group: Optional[str] = None):
        super().__init__()
        self.max_angular_velocity = max_angular_velocity
        self.max_linear_velocity = max_linear_velocity
        diff_drive_joints = [v for k, v in self.world.joints.items() if isinstance(v, OmniDrivePR22)]
        assert len(diff_drive_joints) == 1
        self.joint: OmniDrivePR22 = diff_drive_joints[0]
        self.weight = weight
        self.base_root_link = self.joint.parent_link_name
        self.base_tip_link = self.joint.child_link_name
        self.eef_tip_link = self.world.get_link_name(eef_link)
        # self.root_T_goal = self.transform_msg(self.root_link, goal_pose)

    def make_constraints(self):
        base_root_T_base_tip = self.get_fk(self.base_root_link, self.base_tip_link)

        base_tip_T_eef_tip = self.get_fk(self.base_tip_link, self.eef_tip_link)
        base_tip_P_eef_tip = base_tip_T_eef_tip.to_position()
        base_tip_V_eef_vel = w.Vector3(self.get_expr_velocity(base_tip_P_eef_tip))
        base_root_V_eef_vel = base_root_T_base_tip.dot(base_tip_V_eef_vel)
        velocity_magnitude_mps = base_root_V_eef_vel.norm()
        base_root_V_eef_vel.scale(1)

        root_yaw1 = self.joint.caster_yaw1.get_symbol(Derivatives.position)
        root_V_forward = w.Vector3((w.cos(root_yaw1), w.sin(root_yaw1), 0))
        root_V_forward.vis_frame = self.base_tip_link

        self.add_debug_expr('root_V_forward', root_V_forward)
        self.add_debug_expr('base_root_V_eef_vel', base_root_V_eef_vel)

        weight = w.if_greater(velocity_magnitude_mps, 0.01, self.weight, 0)

        self.add_vector_goal_constraints(frame_V_current=root_V_forward,
                                         frame_V_goal=base_root_V_eef_vel,
                                         reference_velocity=self.max_angular_velocity,
                                         weight=weight,
                                         name='angle')

    def __str__(self) -> str:
        return super().__str__()
