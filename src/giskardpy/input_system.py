from angles import shortest_angular_distance
from tf.transformations import rotation_from_matrix, quaternion_matrix, quaternion_slerp

import symengine_wrappers as sw
from math import pi
import numpy as np


class InputArray(object):
    def __init__(self, to_expr, prefix, suffix, **kwargs):
        for param_name, identifier in kwargs.items():
            setattr(self, param_name, to_expr(list(prefix) + list(identifier) + list(suffix)))


class JointStatesInput(object):
    def __init__(self, to_expr, joint_names, prefix=(), suffix=()):
        self.joint_map = {}
        for joint_name in joint_names:
            self.joint_map[joint_name] = to_expr(list(prefix) + [joint_name] + list(suffix))


class Point3Input(InputArray):
    def __init__(self, to_expr, prefix=(), suffix=(), x=(u'0',), y=(u'1',), z=(u'2',)):
        super(Point3Input, self).__init__(to_expr, prefix, suffix,
                                          x=x, y=y, z=z)

    def get_expression(self):
        return sw.point3(self.x, self.y, self.z)


class Vector3Input(Point3Input):
    def get_expression(self):
        return sw.vector3(self.x, self.y, self.z)


class FrameInput(InputArray):
    def __init__(self, to_expr, translation_prefix=(), translation_suffix=(),
                 rotation_prefix=(), rotation_suffix=(),
                 x=(u'x',), y=(u'y',), z=(u'z',),
                 qx=(u'x',), qy=(u'y',), qz=(u'z',), qw=(u'w',)):
        super(FrameInput, self).__init__(to_expr, (), (),
                                         x=list(translation_prefix) + list(x) + list(translation_suffix),
                                         y=list(translation_prefix) + list(y) + list(translation_suffix),
                                         z=list(translation_prefix) + list(z) + list(translation_suffix),
                                         qx=list(rotation_prefix) + list(qx) + list(rotation_suffix),
                                         qy=list(rotation_prefix) + list(qy) + list(rotation_suffix),
                                         qz=list(rotation_prefix) + list(qz) + list(rotation_suffix),
                                         qw=list(rotation_prefix) + list(qw) + list(rotation_suffix))

    def get_frame(self):
        return sw.frame_quaternion(self.x, self.y, self.z, self.qx, self.qy, self.qz, self.qw)

    def get_position(self):
        return sw.point3(self.x, self.y, self.z)

    def get_rotation(self):
        return sw.rotation_matrix_from_quaternion(self.qx, self.qy, self.qz, self.qw)


class ShortestAngularDistanceInput(object):
    def __init__(self, f, prefix, current_angle, goal_angle):
        self.current_angle = current_angle
        self.goal_angle = goal_angle
        self.name = f(prefix + [self.get_key()])

    def __call__(self, god_map):
        """
        :param god_map:
        :type god_map: giskardpy.god_map.GodMap
        :return:
        :rtype: float
        """
        a1 = god_map.get_data(self.current_angle)  # type: float
        a2 = god_map.get_data(self.goal_angle)  # type: float
        return shortest_angular_distance(a1, a2)

    def get_key(self):
        return '__'.join(str(x) for x in [self.__class__.__name__] + self.current_angle + self.goal_angle)

    def get_expression(self):
        return self.name

# class SlerpInput(object):
#     def __init__(self, f, prefix, current_rotation, goal_rotation, p_gain, max_speed):
#         self.current_rotation = current_rotation
#         self.goal_rotation = goal_rotation
#         self.p_gain = p_gain
#         self.max_speed = max_speed
#         self.x = f(prefix + [self.get_key(), '0'])
#         self.y = f(prefix + [self.get_key(), '1'])
#         self.z = f(prefix + [self.get_key(), '2'])
#
#     def __call__(self, god_map):
#         """
#         :param god_map:
#         :type god_map: giskardpy.god_map.GodMap
#         :return:
#         :rtype: float
#         """
#         current_rotation = np.array(to_list(god_map.get_data(self.current_rotation)))
#         current_rotation_m = quaternion_matrix(current_rotation)
#         goal_rotation_q = god_map.get_data(self.goal_rotation)
#         if goal_rotation_q == 0:
#             goal_rotation = current_rotation
#             goal_rotation_m = current_rotation_m
#         else:
#             goal_rotation = np.array(to_list(goal_rotation_q))
#             goal_rotation_m = quaternion_matrix(goal_rotation)
#         p_gain = god_map.get_data(self.p_gain)
#         max_speed = god_map.get_data(self.max_speed)
#         rot_error, _, _ = rotation_from_matrix(current_rotation_m.T.dot(goal_rotation_m))
#
#         control = p_gain * rot_error
#         if max_speed - control > 0 or control == 0:
#             interpolation_value = 1
#         else:
#             interpolation_value = max_speed / control
#
#         intermediate_goal = georg_slerp(current_rotation,
#                                         goal_rotation,
#                                         interpolation_value)
#
#         rm = current_rotation_m.T.dot(quaternion_matrix(intermediate_goal))
#
#         angle2, axis2, _ = rotation_from_matrix(rm)
#         #
#         # r_rot_control = current_rotation_m[:3, :3].dot((axis2 * angle2))
#
#         # angle2, axis2, _ = rotation_from_matrix(quaternion_matrix(intermediate_goal))
#
#         r_rot_control = (axis2 * angle2)
#
#         return r_rot_control
#
#     def get_key(self):
#         return '__'.join(
#             str(x) for x in [self.__class__.__name__] + self.current_rotation + self.goal_rotation).replace(',', '__')
#
#     def get_expression(self):
#         return [self.x, self.y, self.z, ]
#
#     def get_x(self):
#         return self.x
#
#     def get_y(self):
#         return self.y
#
#     def get_z(self):
#         return self.z
