from typing import Optional, List, Tuple

from giskardpy.configs.drives import DriveInterface, OmniDriveCmdVelInterface, DiffDriveCmdVelInterface
from giskardpy.configs.follow_joint_trajectory import FollowJointTrajectoryInterface
from giskardpy.model.joints import Joint


class HardwareConfig:
    def __init__(self):
        self.drive_interfaces: List[DriveInterface] = []
        self.follow_joint_trajectory_interfaces: List[FollowJointTrajectoryInterface] = []
        self.joint_state_topics: List[Tuple[str, str]] = []
        self.odometry_topics: List[Tuple[str, Joint]] = []

    def add_follow_joint_trajectory_server(self, namespace, state_topic, group_name):
        self.follow_joint_trajectory_interfaces.append(FollowJointTrajectoryInterface(
            action_namespace=namespace,
            state_topic=state_topic,
            group_name=group_name))

    def add_omni_drive_interface(self, group_name, cmd_vel_topic, parent_link_name, child_link_name,
                                 joint_name: str = 'brumbrum',
                                 odom_x_name: str = 'odom_x', odom_y_name: str = 'odom_y',
                                 odom_rot_name: str = 'odom_rot', **kwargs):
        self.drive_interfaces.append(OmniDriveCmdVelInterface(group_name=group_name,
                                                              cmd_vel_topic=cmd_vel_topic,
                                                              parent_link_name=parent_link_name,
                                                              child_link_name=child_link_name,
                                                              joint_name=joint_name,
                                                              odom_x_name=odom_x_name,
                                                              odom_y_name=odom_y_name,
                                                              odom_rot_name=odom_rot_name,
                                                              **kwargs))

    def add_diff_drive_interface(self, cmd_vel_topic, parent_link_name, child_link_name,
                                 joint_name: str = 'brumbrum',
                                 odom_x_name: str = 'odom_x', odom_rot_name: str = 'odom_rot',
                                 translation_velocity_limit: Optional[float] = 0.2,
                                 rotation_velocity_limit: Optional[float] = 0.2,
                                 translation_acceleration_limit: Optional[float] = None,
                                 rotation_acceleration_limit: Optional[float] = None,
                                 translation_jerk_limit: Optional[float] = 5,
                                 rotation_jerk_limit: Optional[float] = 10,
                                 **kwargs):
        self.drive_interfaces.append(DiffDriveCmdVelInterface(cmd_vel_topic=cmd_vel_topic,
                                                              parent_link_name=parent_link_name,
                                                              child_link_name=child_link_name,
                                                              joint_name=joint_name,
                                                              odom_x_name=odom_x_name,
                                                              odom_rot_name=odom_rot_name,
                                                              translation_velocity_limit=translation_velocity_limit,
                                                              rotation_velocity_limit=rotation_velocity_limit,
                                                              translation_acceleration_limit=translation_acceleration_limit,
                                                              rotation_acceleration_limit=rotation_acceleration_limit,
                                                              translation_jerk_limit=translation_jerk_limit,
                                                              rotation_jerk_limit=rotation_jerk_limit,
                                                              **kwargs))
