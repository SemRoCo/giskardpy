from collections import OrderedDict

from giskardpy import USE_SYMENGINE
if USE_SYMENGINE:
    import giskardpy.symengine_wrappers as spw
else:
    import giskardpy.sympy_wrappers as spw

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

        self._symbol_map = OrderedDict((fn, spw.Symbol(('{}{}{}'.format(self._prefix, fn, self._suffix))))
                                        for fn in float_names)
        self._str_map = OrderedDict((k, str(v)) for k, v in self._symbol_map.items())

    def to_symbol(self, float_name):
        return self._symbol_map[float_name]

    def to_str_symbol(self, float_name):
        return self._str_map[float_name]

    def get_float_names(self):
        return self._symbol_map.keys()

    def get_update_dict(self, **kwargs):
        return {self._str_map[k]: v for k, v in kwargs.items() if k in self._symbol_map}

    def get_expression(self):
        return spw.Matrix([spw.Symbol(x) for x in self._symbol_map.values()])


class ScalarInput(ControllerInputArray):
    def __init__(self, prefix, suffix=''):
        super(ScalarInput, self).__init__(['v'], prefix, suffix)

    def get_update_dict(self, v):
        return super(ScalarInput, self).get_update_dict(v=v)

    def get_expression(self):
        return self._symbol_map['v']

    def get_symbol_str(self):
        return self._str_map['v']

class HackInput(ControllerInputArray):
    def __init__(self, prefix, suffix=''):
        super(HackInput, self).__init__(['v'], prefix, suffix)

    def update(self, observable_dict):
        return 1

    def get_update_dict(self, v):
        return super(HackInput, self).get_update_dict(v=self.update)

    def get_expression(self):
        return self._symbol_map['v']

    def get_symbol_str(self):
        return self._str_map['v']

class slerp(ControllerInputArray):
    def __init__(self, prefix, suffix=''):
        super(slerp, self).__init__(['v'], prefix, suffix)

    def update(self, observable_dict):
        return 1

    def get_update_dict(self, v):
        return super(slerp, self).get_update_dict(v=self.update)

    def get_expression(self):
        return self._symbol_map['v']

    def get_symbol_str(self):
        return self._str_map['v']


class Point3Input(ControllerInputArray):
    def __init__(self, prefix, suffix=''):
        super(Point3Input, self).__init__(['x', 'y', 'z'], prefix, suffix)

    def get_update_dict(self, x, y, z):
        return super(Point3Input, self).get_update_dict(x=x, y=y, z=z)

    def get_x(self):
        return self._symbol_map['x']

    def get_y(self):
        return self._symbol_map['y']

    def get_z(self):
        return self._symbol_map['z']

    def get_expression(self):
        return spw.point3(*self._symbol_map.values())


class Vec3Input(Point3Input):
    def __init__(self, prefix, suffix=''):
        super(Vec3Input, self).__init__(prefix, suffix)

    def get_expression(self):
        return spw.vec3(*self._symbol_map.values())


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
        return spw.rotation3_quaternion(*self._symbol_map.values())


class FrameInput(ControllerInputArray):
    def __init__(self, prefix, suffix=''):
        super(FrameInput, self).__init__(['qx', 'qy', 'qz', 'qw', 'x', 'y', 'z'], prefix, suffix)

    def get_update_dict(self, qx, qy, qz, qw, x, y, z):
        return super(FrameInput, self).get_update_dict(qx=qx, qy=qy, qz=qz, qw=qw, x=x, y=y, z=z)

    def get_expression(self):
        return spw.frame3_quaternion(*(self._symbol_map.values()[:4] + [self._symbol_map.values()[4:]]))

    def get_position(self):
        return spw.point3(*self._symbol_map.values()[4:])

    def get_rotation(self):
        return spw.rotation3_quaternion(*self._symbol_map.values()[:4])