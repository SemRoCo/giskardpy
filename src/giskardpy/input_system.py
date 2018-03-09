import symengine_wrappers as sw


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


class FrameInput(InputArray):
    def __init__(self, x='', y='', z='', qx='', qy='', qz='', qw=''):
        super(FrameInput, self).__init__(x=x, y=y, z=z, qx=qx, qy=qy, qz=qz, qw=qw)

    def get_frame(self):
        return sw.frame3_quaternion(self.x, self.y, self.z, self.qx, self.qy, self.qz, self.qw)

    def get_position(self):
        return sw.point3(self.x, self.y, self.z)

    def get_rotation(self):
        return sw.rotation3_quaternion(self.qx, self.qy, self.qz, self.qw)
