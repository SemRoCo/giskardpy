from __future__ import division

from typing import Optional

import giskardpy.casadi_wrapper as cas
from giskardpy.data_types.data_types import PrefixName
from giskardpy.motion_statechart.tasks.task import WEIGHT_ABOVE_CA, Task
from giskardpy.god_map import god_map


class GraspBar(Task):
    def __init__(self,
                 root_link: PrefixName,
                 tip_link: PrefixName,
                 tip_grasp_axis: cas.Vector3,
                 bar_center: cas.Point3,
                 bar_axis: cas.Vector3,
                 bar_length: float,
                 threshold: float = 0.01,
                 reference_linear_velocity: float = 0.1,
                 reference_angular_velocity: float = 0.5,
                 weight: float = WEIGHT_ABOVE_CA,
                 name: Optional[str] = None):
        """
        Like a CartesianPose but with more freedom.
        tip_link is allowed to be at any point along bar_axis, that is without bar_center +/- bar_length.
        It will align tip_grasp_axis with bar_axis, but allows rotation around it.
        :param root_link: root link of the kinematic chain
        :param tip_link: tip link of the kinematic chain
        :param tip_grasp_axis: axis of tip_link that will be aligned with bar_axis
        :param bar_center: center of the bar to be grasped
        :param bar_axis: alignment of the bar to be grasped
        :param bar_length: length of the bar to be grasped
        :param reference_linear_velocity: m/s
        :param reference_angular_velocity: rad/s
        :param weight: 
        """
        self.root = root_link
        self.tip = tip_link
        if name is None:
            name = f'{self.__class__.__name__}/{self.root}/{self.tip}'
        super().__init__(name=name)

        bar_center = god_map.world.transform(self.root, bar_center)

        tip_grasp_axis = god_map.world.transform(self.tip, tip_grasp_axis)
        tip_grasp_axis.scale(1)

        bar_axis = god_map.world.transform(self.root, bar_axis)
        bar_axis.scale(1)

        self.bar_axis = bar_axis
        self.tip_grasp_axis = tip_grasp_axis
        self.bar_center = bar_center
        self.bar_length = bar_length
        self.reference_linear_velocity = reference_linear_velocity
        self.reference_angular_velocity = reference_angular_velocity
        self.weight = weight


        root_V_bar_axis = self.bar_axis
        tip_V_tip_grasp_axis = self.tip_grasp_axis
        root_P_bar_center = self.bar_center

        root_T_tip = god_map.world.compose_fk_expression(self.root, self.tip)
        root_V_tip_normal = cas.dot(root_T_tip, tip_V_tip_grasp_axis)

        self.add_vector_goal_constraints(frame_V_current=root_V_tip_normal,
                                         frame_V_goal=root_V_bar_axis,
                                         reference_velocity=self.reference_angular_velocity,
                                         weight=self.weight)

        root_P_tip = god_map.world.compose_fk_expression(self.root, self.tip).to_position()

        root_P_line_start = root_P_bar_center + root_V_bar_axis * self.bar_length / 2
        root_P_line_end = root_P_bar_center - root_V_bar_axis * self.bar_length / 2

        dist, nearest = cas.distance_point_to_line_segment(root_P_tip, root_P_line_start, root_P_line_end)

        self.add_point_goal_constraints(frame_P_current=root_T_tip.to_position(),
                                        frame_P_goal=nearest,
                                        reference_velocity=self.reference_linear_velocity,
                                        weight=self.weight)

        self.expression = cas.less_equal(dist, threshold)
