from __future__ import division

from giskardpy import casadi_wrapper as w
from giskardpy.goals.goal import Goal, WEIGHT_ABOVE_CA
import giskardpy.utils.tfwrapper as tf

class GraspBar(Goal):
    def __init__(self, root_link, tip_link, tip_grasp_axis, bar_center, bar_axis, bar_length, root_group: str = None,
                 tip_group: str = None, max_linear_velocity=0.1, max_angular_velocity=0.5, weight=WEIGHT_ABOVE_CA,
                 **kwargs):
        """
        TODO update description
        This goal can be used to grasp bars. It's like a cartesian goal with some freedom along one axis.
        :param root_link: str, root link of the kin chain
        :param tip_link: str, tip link of the kin chain
        :param tip_grasp_axis: Vector3Stamped as json, this axis of the tip will be aligned with bar_axis
        :param bar_center: PointStamped as json, center of the bar
        :param bar_axis: Vector3Stamped as json, tip_grasp_axis will be aligned with this vector
        :param bar_length: float, length of the bar
        :param max_linear_velocity: float, m/s, default 0.1
        :param max_angular_velocity: float, rad/s, default 0.5
        :param weight: float default WEIGHT_ABOVE_CA
        """
        super().__init__(**kwargs)
        self.root = self.get_link(root_link, root_group)
        self.tip = self.get_link(tip_link, tip_group)

        bar_center = self.transform_msg(self.root, bar_center)

        tip_grasp_axis = self.transform_msg(self.tip, tip_grasp_axis)
        tip_grasp_axis.vector = tf.normalize(tip_grasp_axis.vector)

        bar_axis = self.transform_msg(self.root, bar_axis)
        bar_axis.vector = tf.normalize(bar_axis.vector)

        self.bar_axis = bar_axis
        self.tip_grasp_axis = tip_grasp_axis
        self.bar_center = bar_center
        self.bar_length = bar_length
        self.translation_max_velocity = max_linear_velocity
        self.rotation_max_velocity = max_angular_velocity
        self.weight = weight

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.root}/{self.tip}'

    def make_constraints(self):
        root_V_bar_axis = w.ros_msg_to_matrix(self.bar_axis)
        tip_V_tip_grasp_axis = w.ros_msg_to_matrix(self.tip_grasp_axis)
        root_P_bar_center = w.ros_msg_to_matrix(self.bar_center)

        root_T_tip = self.get_fk(self.root, self.tip)
        root_V_tip_normal = w.dot(root_T_tip, tip_V_tip_grasp_axis)

        self.add_vector_goal_constraints(frame_V_current=root_V_tip_normal,
                                         frame_V_goal=root_V_bar_axis,
                                         reference_velocity=self.rotation_max_velocity,
                                         weight=self.weight)

        root_P_tip = w.position_of(self.get_fk(self.root, self.tip))

        root_P_line_start = root_P_bar_center + root_V_bar_axis * self.bar_length / 2
        root_P_line_end = root_P_bar_center - root_V_bar_axis * self.bar_length / 2

        dist, nearest = w.distance_point_to_line_segment(root_P_tip, root_P_line_start, root_P_line_end)

        self.add_point_goal_constraints(frame_P_current=w.position_of(root_T_tip),
                                        frame_P_goal=nearest,
                                        reference_velocity=self.translation_max_velocity,
                                        weight=self.weight)
