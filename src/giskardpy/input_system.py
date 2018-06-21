from angles import shortest_angular_distance
from tf.transformations import rotation_from_matrix, quaternion_matrix, quaternion_slerp

import symengine_wrappers as sw
from math import pi
import numpy as np

from giskardpy.utils import to_list, georg_slerp


class InputArray(object):
    def __init__(self, **kwargs):
        for param_name, identifier in kwargs.items():
            setattr(self, param_name, sw.Symbol(str(identifier)))


class JointStatesInput(InputArray):
    def __init__(self, joint_map):
        self.joint_map = joint_map
        for k, v in self.joint_map.items():
            self.joint_map[k] = sw.Symbol(str(v))
        super(JointStatesInput, self).__init__(**joint_map)

    @classmethod
    def prefix_constructor(cls, f, joints, prefix='', suffix=''):
        joint_map = {}
        for joint_name in joints:
            if prefix != '':
                prefix2 = '{}/'.format(prefix)
            else:
                prefix2 = ''
            if suffix != '':
                suffix2 = '/{}'.format(suffix)
            else:
                suffix2 = ''
            joint_map[joint_name] = f('{}{}{}'.format(prefix2, joint_name, suffix2))
        return cls(joint_map)


class Point3Input(InputArray):
    def __init__(self, x='', y='', z=''):
        super(Point3Input, self).__init__(x=x, y=y, z=z)

    def get_expression(self):
        return sw.point3(self.x, self.y, self.z)

    @classmethod
    def prefix(cls, f, prefix):
        return cls(x=f(prefix + ['0']),
                   y=f(prefix + ['1']),
                   z=f(prefix + ['2']))


class Vector3Input(Point3Input):
    def get_expression(self):
        return sw.vector3(self.x, self.y, self.z)


class FrameInput(InputArray):
    def __init__(self, x='', y='', z='', qx='', qy='', qz='', qw=''):
        super(FrameInput, self).__init__(x=x, y=y, z=z, qx=qx, qy=qy, qz=qz, qw=qw)

    @classmethod
    def prefix_constructor(self, point_prefix, orientation_prefix, f):
        if isinstance(point_prefix, str):
            return self(x=f('{}/x'.format(point_prefix)),
                        y=f('{}/y'.format(point_prefix)),
                        z=f('{}/z'.format(point_prefix)),
                        qx=f('{}/x'.format(orientation_prefix)),
                        qy=f('{}/y'.format(orientation_prefix)),
                        qz=f('{}/z'.format(orientation_prefix)),
                        qw=f('{}/w'.format(orientation_prefix)))
        else:
            return self(x=f(point_prefix + ['x']),
                        y=f(point_prefix + ['y']),
                        z=f(point_prefix + ['z']),
                        qx=f(orientation_prefix + ['x']),
                        qy=f(orientation_prefix + ['y']),
                        qz=f(orientation_prefix + ['z']),
                        qw=f(orientation_prefix + ['w']))

    def get_frame(self):
        return sw.frame3_quaternion(self.x, self.y, self.z, self.qx, self.qy, self.qz, self.qw)

    def get_position(self):
        return sw.point3(self.x, self.y, self.z)

    def get_rotation(self):
        return sw.rotation3_quaternion(self.qx, self.qy, self.qz, self.qw)


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


class SlerpInput(object):
    def __init__(self, f, prefix, current_rotation, goal_rotation, p_gain, max_speed):
        self.current_rotation = current_rotation
        self.goal_rotation = goal_rotation
        self.p_gain = p_gain
        self.max_speed = max_speed
        self.x = f(prefix + [self.get_key(), '0'])
        self.y = f(prefix + [self.get_key(), '1'])
        self.z = f(prefix + [self.get_key(), '2'])

    def __call__(self, god_map):
        """
        :param god_map:
        :type god_map: giskardpy.god_map.GodMap
        :return:
        :rtype: float
        """
        current_rotation = np.array(to_list(god_map.get_data(self.current_rotation)))
        current_rotation_m = quaternion_matrix(current_rotation)
        goal_rotation_q = god_map.get_data(self.goal_rotation)
        if goal_rotation_q == 0:
            goal_rotation = current_rotation
            goal_rotation_m = current_rotation_m
        else:
            goal_rotation = np.array(to_list(goal_rotation_q))
            goal_rotation_m = quaternion_matrix(goal_rotation)
        p_gain = god_map.get_data(self.p_gain)
        max_speed = god_map.get_data(self.max_speed)
        rot_error, _, _ = rotation_from_matrix(current_rotation_m.T.dot(goal_rotation_m))

        control = p_gain * rot_error
        if max_speed - control > 0 or control == 0:
            interpolation_value = 1
        else:
            interpolation_value = max_speed / control

        intermediate_goal = georg_slerp(current_rotation,
                                        goal_rotation,
                                        interpolation_value)

        rm = current_rotation_m.T.dot(quaternion_matrix(intermediate_goal))

        angle2, axis2, _ = rotation_from_matrix(rm)
        #
        # r_rot_control = current_rotation_m[:3, :3].dot((axis2 * angle2))

        # angle2, axis2, _ = rotation_from_matrix(quaternion_matrix(intermediate_goal))

        r_rot_control = (axis2 * angle2)

        return r_rot_control

    def get_key(self):
        return '__'.join(
            str(x) for x in [self.__class__.__name__] + self.current_rotation + self.goal_rotation).replace(',', '__')

    def get_expression(self):
        return [self.x, self.y, self.z, ]

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_z(self):
        return self.z
