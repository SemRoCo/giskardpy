from typing import Optional, List

from giskardpy.configs.drives import DriveInterface, OmniDriveCmdVelInterface, DiffDriveCmdVelInterface
from giskardpy.configs.follow_joint_trajectory import FollowJointTrajectoryInterface


class HardwareConfig:
    def __init__(self):
        self.drive_interface: Optional[DriveInterface] = None
        self.follow_joint_trajectory_interfaces: List[FollowJointTrajectoryInterface] = []

    def add_follow_joint_trajectory_server(self, namespace, state_topic):
        self.follow_joint_trajectory_interfaces.append(FollowJointTrajectoryInterface(
            namespace=namespace,
            state_topic=state_topic))

    def add_omni_drive_interface(self, cmd_vel_topic, parent_link_name, child_link_name,
                                 joint_name: str = 'brumbrum',
                                 odom_x_name: str = 'odom_x', odom_y_name: str = 'odom_y',
                                 odom_rot_name: str = 'odom_rot', **kwargs):
        self.drive_interface = OmniDriveCmdVelInterface(cmd_vel_topic=cmd_vel_topic,
                                                        parent_link_name=parent_link_name,
                                                        child_link_name=child_link_name,
                                                        joint_name=joint_name,
                                                        odom_x_name=odom_x_name,
                                                        odom_y_name=odom_y_name,
                                                        odom_rot_name=odom_rot_name,
                                                        **kwargs)

    def add_diff_drive_interface(self, cmd_vel_topic, parent_link_name, child_link_name,
                                 joint_name: str = 'brumbrum',
                                 odom_x_name: str = 'odom_x', odom_rot_name: str = 'odom_rot', **kwargs):
        self.drive_interface = DiffDriveCmdVelInterface(cmd_vel_topic=cmd_vel_topic,
                                                        parent_link_name=parent_link_name,
                                                        child_link_name=child_link_name,
                                                        joint_name=joint_name,
                                                        odom_x_name=odom_x_name,
                                                        odom_rot_name=odom_rot_name, **kwargs)
