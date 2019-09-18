import symengine_wrappers as sw
from giskardpy.utils import KeyDefaultDict


class InputArray(object):
    def __init__(self, to_expr, prefix, suffix, **kwargs):
        for param_name, identifier in kwargs.items():
            setattr(self, param_name, to_expr(list(prefix) + list(identifier) + list(suffix)))


class JointStatesInput(object):
    def __init__(self, to_expr, joint_names, prefix=(), suffix=()):
        self.joint_map = KeyDefaultDict(lambda joint_name: to_expr(list(prefix) +
                                                                   [joint_name] +
                                                                   list(suffix)))
        for joint_name in joint_names:
            self.joint_map[joint_name]  # trigger factory


class Point3Input(InputArray):
    def __init__(self, to_expr, prefix=(), suffix=(), x=(0,), y=(1,), z=(2,)):
        super(Point3Input, self).__init__(to_expr, prefix, suffix,
                                          x=x, y=y, z=z)

    def get_expression(self):
        return sw.point3(self.x, self.y, self.z)


class Vector3Input(Point3Input):
    def get_expression(self):
        return sw.vector3(self.x, self.y, self.z)


class Vector3StampedInput(InputArray):
    def __init__(self, to_expr, vector_prefix=(), vector_suffix=(), x=(u'x',), y=(u'y',), z=(u'z',)):
        super(Vector3StampedInput, self).__init__(to_expr, (), (),
                                                  x=list(vector_prefix) + list(x) + list(vector_suffix),
                                                  y=list(vector_prefix) + list(y) + list(vector_suffix),
                                                  z=list(vector_prefix) + list(z) + list(vector_suffix))

    def get_expression(self):
        return sw.vector3(self.x, self.y, self.z)


class PointStampedInput(InputArray):
    def __init__(self, to_expr, prefix=(), suffix=(), x=(u'x',), y=(u'y',), z=(u'z',)):
        super(PointStampedInput, self).__init__(to_expr, (), (),
                                                x=list(prefix) + list(x) + list(suffix),
                                                y=list(prefix) + list(y) + list(suffix),
                                                z=list(prefix) + list(z) + list(suffix))

    def get_expression(self):
        return sw.point3(self.x, self.y, self.z)


class PoseStampedInput(InputArray):
    def __init__(self, to_expr, translation_prefix=(), translation_suffix=(),
                 rotation_prefix=(), rotation_suffix=(),
                 x=(u'x',), y=(u'y',), z=(u'z',),
                 qx=(u'x',), qy=(u'y',), qz=(u'z',), qw=(u'w',)):
        super(PoseStampedInput, self).__init__(to_expr, (), (),
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


class FrameInput(InputArray):
    def __init__(self, to_expr, prefix):
        super(FrameInput, self).__init__(to_expr, (), (),
                                         f00=list(prefix) + [0,0],
                                         f01=list(prefix) + [0,1],
                                         f02=list(prefix) + [0,2],
                                         f03=list(prefix) + [0,3],
                                         f10=list(prefix) + [1,0],
                                         f11=list(prefix) + [1,1],
                                         f12=list(prefix) + [1,2],
                                         f13=list(prefix) + [1,3],
                                         f20=list(prefix) + [2,0],
                                         f21=list(prefix) + [2,1],
                                         f22=list(prefix) + [2,2],
                                         f23=list(prefix) + [2,3])

    def get_frame(self):
        return sw.Matrix([[self.f00, self.f01, self.f02, self.f03],
                          [self.f10, self.f11, self.f12, self.f13],
                          [self.f20, self.f21, self.f22, self.f23],
                          [0, 0, 0, 1]])

    def get_position(self):
        return sw.position_of(self.get_frame())

    def get_rotation(self):
        return sw.rotation_of(self.get_frame())
