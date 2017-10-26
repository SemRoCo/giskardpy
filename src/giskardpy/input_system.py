from collections import OrderedDict

import symengine as sp

from giskardpy.sympy_wrappers import point3, vec3, frame3_quaternion


class ControllerInputArray(object):
    separator = '__'

    def __init__(self, float_names, prefix='', suffix=''):
        if prefix == '':
            self._prefix = prefix
        else:
            self._prefix = '{}{}'.format(prefix, self.separator)
        if suffix == '':
            self._suffix = suffix
        else:
            self._suffix = '{}{}'.format(self.separator, suffix)

        self._symbol_map = OrderedDict({fn: sp.Symbol('{}{}{}'.format(self._prefix, fn, self._suffix))
                                        for fn in float_names})
        self._str_map = OrderedDict({k: str(v) for k, v in self._symbol_map.items()})

    def to_symbol(self, float_name):
        return self._symbol_map[float_name]

    def to_str_symbol(self, float_name):
        return self._str_map[float_name]

    def get_float_names(self):
        return self._symbol_map.keys()

    def get_update_dict(self, **kwargs):
        return {self._str_map[k]: v for k, v in kwargs.items() if k in self._symbol_map}

    def get_expression(self):
        return sp.Matrix([sp.Symbol(x) for x in self._symbol_map.values()])


class ScalarInput(ControllerInputArray):
    def __init__(self, prefix, suffix=''):
        super(ScalarInput, self).__init__(['v'], prefix, suffix)

    def get_update_dict(self, v):
        return super(ScalarInput, self).get_update_dict(v=v)

    def get_expression(self):
        return self._symbol_map['v']

    def get_symbol_str(self):
        return self._str_map['v']


class Point3(ControllerInputArray):
    def __init__(self, prefix, suffix=''):
        super(Point3, self).__init__(['x', 'y', 'z'], prefix, suffix)

    def get_update_dict(self, x, y, z):
        return super(Point3, self).get_update_dict(x=x, y=y, z=z)

    def get_x(self):
        return self._symbol_map['x']

    def get_y(self):
        return self._symbol_map['y']

    def get_z(self):
        return self._symbol_map['z']

    def get_expression(self):
        return point3(self.get_x(), self.get_y(), self.get_z())


class Vec3(Point3):
    def __init__(self, prefix, suffix=''):
        super(Vec3, self).__init__(prefix, suffix)

    def get_expression(self):
        return vec3(self.get_x(), self.get_y(), self.get_z())


class Quaternion(ControllerInputArray):
    def __init__(self, prefix, suffix=''):
        super(Quaternion, self).__init__(['x', 'y', 'z', 'w'], prefix, suffix)

    def get_update_dict(self, x, y, z, w):
        return super(Quaternion, self).get_update_dict(x=x, y=y, z=z, w=w)

    def get_x(self):
        return self._symbol_map['x']

    def get_y(self):
        return self._symbol_map['y']

    def get_z(self):
        return self._symbol_map['z']

    def get_w(self):
        return self._symbol_map['w']

    def get_expression(self):
        return rotation3_quaternion(self.get_x(), self.get_y(), self.get_z(), self.get_w())


class Frame3(ControllerInputArray):
    def __init__(self, prefix, suffix=''):
        super(Frame3, self).__init__(['qx', 'qy', 'qz', 'qw', 'x', 'y', 'z'], prefix, suffix)

    def get_update_dict(self, qx, qy, qz, qw, x, y, z):
        return super(Frame3, self).get_update_dict(qx=qx, qy=qy, qz=qz, qw=qw, x=x, y=y, z=z)

    def get_expression(self):
        return frame3_quaternion(self._symbol_map['qx'],
                                 self._symbol_map['qy'],
                                 self._symbol_map['qz'],
                                 self._symbol_map['qw'],
                                 point3(self._symbol_map['x'], self._symbol_map['y'], self._symbol_map['z']))

    def get_position(self):
        return point3(self._symbol_map['x'], self._symbol_map['y'], self._symbol_map['z'])

    def get_rotation(self):
        return rotation3_quaternion(self._symbol_map['qx'],
                                    self._symbol_map['qy'],
                                    self._symbol_map['qz'],
                                    self._symbol_map['qw'])