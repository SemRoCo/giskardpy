import symengine_wrappers as sw
from giskardpy.utils import keydefaultdict


class InputArray(object):
    def __init__(self, to_expr, prefix, suffix, **kwargs):
        for param_name, identifier in kwargs.items():
            setattr(self, param_name, to_expr(list(prefix) + list(identifier) + list(suffix)))


class JointStatesInput(object):
    def __init__(self, to_expr, joint_names, prefix=(), suffix=()):
        self.joint_map = keydefaultdict(lambda joint_name: to_expr(list(prefix) +
                                                                   [joint_name] +
                                                                   list(suffix)))
        for joint_name in joint_names:
            self.joint_map[joint_name] # trigger factory


class Point3Input(InputArray):
    def __init__(self, to_expr, prefix=(), suffix=(), x=(0,), y=(1,), z=(2,)):
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
