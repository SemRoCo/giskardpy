from __future__ import division

from typing import Optional

import giskardpy.casadi_wrapper as cas
from giskardpy.data_types.data_types import PrefixName
from giskardpy.god_map import god_map
from giskardpy.motion_statechart.tasks.task import WEIGHT_ABOVE_CA, Task


class DiffDriveTangentialToPoint(Task):

    def __init__(self,
                 goal_point: cas.Point3,
                 forward: Optional[cas.Vector3] = None,
                 group_name: Optional[str] = None,
                 weight: bool = WEIGHT_ABOVE_CA, drive: bool = False,
                 name: Optional[str] = None):
        self.tip = god_map.world.search_for_link_name('base_footprint', group_name)
        self.root = god_map.world.root_link_name
        if name is None:
            name = f'{self.__class__.__name__}/{self.root}/{self.tip}'
        super().__init__(name=name)
        self.goal_point = god_map.world.transform(god_map.world.root_link_name, goal_point)
        self.goal_point.z = 0
        self.weight = weight
        self.drive = drive
        if forward is not None:
            self.tip_V_pointing_axis = god_map.world.transform(self.tip, forward)
            self.tip_V_pointing_axis.scale(1)
        else:
            self.tip_V_pointing_axis = cas.Vector3((1, 0, 0))
            self.tip_V_pointing_axis.reference_frame = self.tip

        map_P_center = cas.Point3(self.goal_point)
        map_T_base = god_map.world.compose_fk_expression(self.root, self.tip)
        map_P_base = map_T_base.to_position()
        map_V_base_to_center = map_P_center - map_P_base
        map_V_base_to_center = cas.scale(map_V_base_to_center, 1)
        map_V_up = cas.Expression([0, 0, 1, 0])
        map_V_tangent = cas.cross(map_V_base_to_center, map_V_up)
        tip_V_pointing_axis = cas.Vector3(self.tip_V_pointing_axis)
        map_V_forward = cas.dot(map_T_base, tip_V_pointing_axis)

        if self.drive:
            angle = cas.abs(cas.angle_between_vector(map_V_forward, map_V_tangent))
            self.add_equality_constraint(reference_velocity=0.5,
                                         equality_bound=-angle,
                                         weight=self.weight,
                                         task_expression=angle,
                                         name='/rot')
        else:
            # angle = cas.abs(cas.angle_between_vector(cas.vector3(1,0,0), map_V_tangent))
            map_R_goal = cas.RotationMatrix.from_vectors(x=map_V_tangent, y=None, z=cas.Vector3((0, 0, 1)))
            goal_angle = map_R_goal.to_angle(lambda axis: axis[2])
            map_R_base = map_T_base.to_rotation()
            axis, map_current_angle = map_R_base.to_axis_angle()
            map_current_angle = cas.if_greater_zero(axis[2], map_current_angle, -map_current_angle)
            angle_error = cas.shortest_angular_distance(map_current_angle, goal_angle)
            self.add_equality_constraint(reference_velocity=0.5,
                                         equality_bound=angle_error,
                                         weight=self.weight,
                                         task_expression=map_current_angle,
                                         name='/rot')


class KeepHandInWorkspace(Task):
    def __init__(self,
                 tip_link: PrefixName,
                 base_footprint: Optional[PrefixName] = None,
                 map_frame: Optional[PrefixName] = None,
                 pointing_axis: Optional[cas.Vector3] = None,
                 max_velocity: float = 0.3,
                 weight: float = WEIGHT_ABOVE_CA,
                 name: Optional[str] = None):
        if base_footprint is None:
            base_footprint = god_map.world.search_for_link_name('base_footprint')
        if map_frame is None:
            map_frame = god_map.world.root_link_name
        self.weight = weight
        self.max_velocity = max_velocity
        self.map_frame = map_frame
        self.tip_link = tip_link
        self.base_footprint = base_footprint
        if name is None:
            name = f'{self.__class__.__name__}/{self.base_footprint}/{self.tip_link}'
        super().__init__(name=name)

        if pointing_axis is not None:
            self.map_V_pointing_axis = god_map.world.transform(self.base_footprint, pointing_axis)
            self.map_V_pointing_axis.scale(1)
        else:
            self.map_V_pointing_axis = cas.Vector3((1, 0, 0))
            self.map_V_pointing_axis.reference_frame = self.map_frame

        weight = WEIGHT_ABOVE_CA
        base_footprint_V_pointing_axis = cas.Vector3(self.map_V_pointing_axis)
        map_T_base_footprint = god_map.world.compose_fk_expression(self.map_frame, self.base_footprint)
        map_V_pointing_axis = cas.dot(map_T_base_footprint, base_footprint_V_pointing_axis)
        map_T_tip = god_map.world.compose_fk_expression(self.map_frame, self.tip_link)
        map_V_tip = cas.Vector3(map_T_tip.to_position())
        map_V_tip.y = 0
        map_V_tip.z = 0
        map_P_tip = map_T_tip.to_position()
        map_P_tip.z = 0
        map_P_base_footprint = map_T_base_footprint.to_position()
        map_P_base_footprint.z = 0
        base_footprint_V_tip = map_P_tip - map_P_base_footprint

        map_V_tip.scale(1)
        angle_error = cas.angle_between_vector(base_footprint_V_tip, map_V_pointing_axis)
        self.add_inequality_constraint(reference_velocity=0.5,
                                       lower_error=-angle_error - 0.2,
                                       upper_error=-angle_error + 0.2,
                                       weight=weight,
                                       task_expression=angle_error,
                                       name='/rot')
