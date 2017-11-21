from giskardpy.robot import Robot


class PointyBot(Robot):
    def __init__(self, weight, urdf='pointy.urdf', tip='eef'):
        super(PointyBot, self).__init__()
        self.load_from_urdf_path(urdf, 'base_link', [tip])
        for joint_name in self.weight_input.get_float_names():
            self.set_joint_weight(joint_name, weight)
