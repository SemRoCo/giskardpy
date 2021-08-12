from __future__ import division

import giskardpy.identifier as identifier
from giskardpy import casadi_wrapper as w
from giskardpy.goals.goal import Goal, WEIGHT_BELOW_CA, WEIGHT_ABOVE_CA
import giskardpy.utils.tfwrapper as tf

class GraspBar(Goal):
    def __init__(self, root_link, tip_link, tip_grasp_axis, bar_center, bar_axis, bar_length,
                 max_linear_velocity=0.1, max_angular_velocity=0.5, weight=WEIGHT_ABOVE_CA, **kwargs):
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
        self.root = root_link
        self.tip = tip_link
        super(GraspBar, self).__init__(**kwargs)

        bar_center = tf.transform_point(self.root, bar_center)

        tip_grasp_axis = tf.transform_vector(self.tip, tip_grasp_axis)
        tip_grasp_axis.vector = tf.normalize(tip_grasp_axis.vector)

        bar_axis = tf.transform_vector(self.root, bar_axis)
        bar_axis.vector = tf.normalize(bar_axis.vector)

        self.bar_axis = bar_axis
        self.tip_grasp_axis = tip_grasp_axis
        self.bar_center = bar_center
        self.bar_length = bar_length
        self.translation_max_velocity = max_linear_velocity
        self.rotation_max_velocity = max_angular_velocity
        self.weight = weight

    def __str__(self):
        s = super(GraspBar, self).__str__()
        return u'{}/{}/{}'.format(s, self.root, self.tip)

    def get_bar_axis_vector(self):
        return self.get_parameter_as_symbolic_expression(u'bar_axis')

    def get_tip_grasp_axis_vector(self):
        return self.get_parameter_as_symbolic_expression(u'tip_grasp_axis')

    def get_bar_center_point(self):
        return self.get_parameter_as_symbolic_expression(u'bar_center')

    def make_constraints(self):
        translation_max_velocity = self.get_parameter_as_symbolic_expression(u'translation_max_velocity')
        rotation_max_velocity = self.get_parameter_as_symbolic_expression(u'rotation_max_velocity')
        weight = self.get_parameter_as_symbolic_expression(u'weight')

        bar_length = self.get_parameter_as_symbolic_expression(u'bar_length')
        root_V_bar_axis = self.get_bar_axis_vector()
        tip_V_tip_grasp_axis = self.get_tip_grasp_axis_vector()
        root_P_bar_center = self.get_bar_center_point()

        self.add_minimize_vector_angle_constraints(max_velocity=rotation_max_velocity,
                                                   root=self.root,
                                                   tip=self.tip,
                                                   tip_V_tip_normal=tip_V_tip_grasp_axis,
                                                   root_V_goal_normal=root_V_bar_axis,
                                                   weight=weight)

        root_P_tip = w.position_of(self.get_fk(self.root, self.tip))

        root_P_line_start = root_P_bar_center + root_V_bar_axis * bar_length / 2
        root_P_line_end = root_P_bar_center - root_V_bar_axis * bar_length / 2

        dist, nearest = w.distance_point_to_line_segment(root_P_tip, root_P_line_start, root_P_line_end)

        self.add_minimize_position_constraints(r_P_g=nearest,
                                               max_velocity=translation_max_velocity,
                                               root=self.root,
                                               tip=self.tip,
                                               weight=weight)
