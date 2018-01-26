from giskardpy.robot_ros import RobotRos


class PointyBot(RobotRos):
    def __init__(self, urdf='pointy.urdf', tip='eef'):
        super(PointyBot, self).__init__()
        self.load_from_urdf_path(urdf, 'base_link', [tip])
