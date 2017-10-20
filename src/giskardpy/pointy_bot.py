from giskardpy.robot import Robot


class PointyBot(Robot):
    def __init__(self, weight, urdf='pointy.urdf'):
        super(PointyBot, self).__init__()
        self.load_from_urdf_path(urdf, 'base_link', ['eef'])
        for joint_name in self.get_joint_names():
            self.set_joint_weight(joint_name, weight)
