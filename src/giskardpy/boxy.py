from giskardpy.robot import Robot
from giskardpy.robot_ros import RobotRos

class Boxy(RobotRos):
    def __init__(self, default_joint_velocity=0.5, urdf_path='boxy.urdf', urdf_str=None):
        if urdf_str is None:
            with open(urdf_path, 'r') as file:
                urdf_str = file.read()
        super(Boxy, self).__init__(urdf_str=urdf_str, root_link='base_link',
                                  tip_links=['left_gripper_tool_frame', 'right_gripper_tool_frame', 'neck_ee_link'],
                                  default_joint_velocity=default_joint_velocity)

