from angles import shortest_angular_distance

import symengine_wrappers as sw
from math import pi


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
        return self(x=f('{}/x'.format(point_prefix)),
                    y=f('{}/y'.format(point_prefix)),
                    z=f('{}/z'.format(point_prefix)),
                    qx=f('{}/x'.format(orientation_prefix)),
                    qy=f('{}/y'.format(orientation_prefix)),
                    qz=f('{}/z'.format(orientation_prefix)),
                    qw=f('{}/w'.format(orientation_prefix)))

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
        self.name = f(prefix+[self.get_sdaffsafd()])

    def __call__(self, god_map):
        """
        :param god_map:
        :type god_map: giskardpy.god_map.GodMap
        :return:
        :rtype: float
        """
        # TODO get rid of unnecessary conversion
        # TODO potential speedup by optimizing shortest_angular_distance
        a1 = god_map.get_data(god_map.expr_to_key[str(self.current_angle)]) #type: float
        a2 = god_map.get_data(god_map.expr_to_key[str(self.goal_angle)]) #type: float
        return shortest_angular_distance(a1, a2)

    def get_sdaffsafd(self):
        # TODO rename me
        return '{}__{}__{}'.format(self.__class__.__name__, str(self.current_angle), str(self.goal_angle))

    def get_expression(self):
        return self.name
