from giskardpy.robot import Robot
from giskardpy.robot_ros import RobotRos

class PR2(RobotRos):
    def __init__(self, default_joint_velocity=0.5, urdf_path='pr2.urdf', urdf_str=None):
        if urdf_str is None:
            with open(urdf_path, 'r') as file:
                urdf_str = file.read()
        super(PR2, self).__init__(urdf_str=urdf_str, root_link='base_link',
                                  tip_links=['l_gripper_tool_frame', 'r_gripper_tool_frame'],
                                  default_joint_velocity=default_joint_velocity)

