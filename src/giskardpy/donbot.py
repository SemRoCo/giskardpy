from giskardpy.robot_ros import RobotRos


class DonBot(RobotRos):
    def __init__(self, weight, urdf='iai_donbot.urdf', tip='gripper_tool_frame'):
        super(DonBot, self).__init__()
        self.load_from_urdf_path(urdf, 'base_footprint', [tip])
        for joint_name in self.weight_input.get_float_names():
            self.set_joint_weight(joint_name, weight)
