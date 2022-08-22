import abc
from abc import ABC
from typing import Optional

from giskardpy.data_types import PrefixName
from giskardpy.god_map import GodMap
from giskardpy.model.joints import OmniDrive, DiffDrive, Joint
from giskardpy.my_types import my_string
from giskardpy.tree.behaviors.send_trajectory_omni_drive_realtime import SendTrajectoryToCmdVel


class DriveInterface(ABC):
    @abc.abstractmethod
    def make_joint(self, god_map: GodMap) -> Joint:
        """
        """

    @abc.abstractmethod
    def make_plugin(self) -> SendTrajectoryToCmdVel:
        """
        """


class OmniDriveCmdVelInterface(DriveInterface):
    def __init__(self,
                 parent_link_name: str,
                 child_link_name: str,
                 cmd_vel_topic: Optional[str] = None,
                 translation_velocity_limit: float = 1,
                 rotation_velocity_limit: float = 1,
                 joint_name: str = 'brumbrum',
                 odom_x_name: str = 'odom_x',
                 odom_y_name: str = 'odom_y',
                 odom_rot_name: str = 'odom_rot',
                 **omni_drive_params):
        self.cmd_vel_topic = cmd_vel_topic
        self.parent_link_name = parent_link_name
        self.child_link_name = child_link_name
        self.translation_velocity_limit = translation_velocity_limit
        self.rotation_velocity_limit = rotation_velocity_limit
        self.omni_drive_params = omni_drive_params
        self.joint_name = joint_name
        self.odom_x_name = odom_x_name
        self.odom_y_name = odom_y_name
        self.odom_rot_name = odom_rot_name

    def make_joint(self, god_map):
        return OmniDrive(god_map=god_map,
                         parent_link_name=self.parent_link_name,
                         child_link_name=self.child_link_name,
                         translation_velocity_limit=self.translation_velocity_limit,
                         rotation_velocity_limit=self.rotation_velocity_limit,
                         name=self.joint_name,
                         x_name=self.odom_x_name,
                         y_name=self.odom_y_name,
                         rot_name=self.odom_rot_name,
                         **self.omni_drive_params)

    def make_plugin(self):
        return SendTrajectoryToCmdVel(self.cmd_vel_topic, self.cmd_vel_topic)


class DiffDriveCmdVelInterface(DriveInterface):
    def __init__(self,
                 parent_link_name: my_string,
                 child_link_name: my_string,
                 cmd_vel_topic: Optional[str] = None,
                 translation_velocity_limit: float = .2,
                 rotation_velocity_limit: float = .2,
                 joint_name: my_string = 'brumbrum',
                 odom_x_name: str = 'odom_x',
                 odom_rot_name: str = 'odom_rot',
                 **diff_drive_params):
        self.cmd_vel_topic = cmd_vel_topic
        if isinstance(parent_link_name, str):
            parent_link_name = PrefixName(parent_link_name, None)
        self.parent_link_name = parent_link_name
        if isinstance(child_link_name, str):
            child_link_name = PrefixName(child_link_name, None)
        self.child_link_name = child_link_name
        self.translation_velocity_limit = translation_velocity_limit
        self.rotation_velocity_limit = rotation_velocity_limit
        self.diff_drive_params = diff_drive_params
        if isinstance(joint_name, str):
            joint_name = PrefixName(joint_name, None)
        self.joint_name = joint_name
        self.odom_x_name = odom_x_name
        self.odom_rot_name = odom_rot_name

    def make_joint(self, god_map):
        return DiffDrive(god_map=god_map,
                         parent_link_name=self.parent_link_name,
                         child_link_name=self.child_link_name,
                         translation_velocity_limit=self.translation_velocity_limit,
                         rotation_velocity_limit=self.rotation_velocity_limit,
                         name=self.joint_name,
                         x_name=self.odom_x_name,
                         rot_name=self.odom_rot_name,
                         **self.diff_drive_params)

    def make_plugin(self):
        return SendTrajectoryToCmdVel(self.cmd_vel_topic, self.cmd_vel_topic)
