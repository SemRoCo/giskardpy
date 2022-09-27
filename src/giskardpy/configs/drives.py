import abc
from abc import ABC
from typing import Optional

from giskardpy.data_types import PrefixName
from giskardpy.god_map import GodMap
from giskardpy.model.joints import OmniDrive, DiffDrive, Joint
from giskardpy.my_types import my_string
from giskardpy.tree.behaviors.send_trajectory_omni_drive_realtime import SendTrajectoryToCmdVel


class DriveInterface(ABC):
    def __init__(self,
                 parent_link_name: my_string,
                 child_link_name: my_string,
                 cmd_vel_topic: Optional[str] = None,
                 joint_name: my_string = 'brumbrum',
                 odom_x_name: my_string = 'odom_x',
                 odom_y_name: my_string = 'odom_y',
                 odom_rot_name: my_string = 'odom_rot',
                 track_only_velocity: bool = False,
                 translation_velocity_limit: Optional[float] = 0.2,
                 rotation_velocity_limit: Optional[float] = 0.2,
                 translation_acceleration_limit: Optional[float] = None,
                 rotation_acceleration_limit: Optional[float] = None,
                 translation_jerk_limit: Optional[float] = 5,
                 rotation_jerk_limit: Optional[float] = 10,
                 **kwargs):
        self.cmd_vel_topic = cmd_vel_topic
        if isinstance(parent_link_name, str):
            parent_link_name = PrefixName(parent_link_name, None)
        self.parent_link_name = parent_link_name
        if isinstance(child_link_name, str):
            child_link_name = PrefixName(child_link_name, None)
        self.child_link_name = child_link_name
        if isinstance(joint_name, str):
            joint_name = PrefixName(joint_name, None)
        self.joint_name = joint_name
        self.odom_x_name = odom_x_name
        self.odom_y_name = odom_y_name
        self.odom_rot_name = odom_rot_name
        self.translation_velocity_limit = translation_velocity_limit
        self.rotation_velocity_limit = rotation_velocity_limit
        self.translation_velocity_limit = translation_velocity_limit
        self.rotation_velocity_limit = rotation_velocity_limit
        self.translation_acceleration_limit = translation_acceleration_limit
        self.rotation_acceleration_limit = rotation_acceleration_limit
        self.translation_jerk_limit = translation_jerk_limit
        self.rotation_jerk_limit = rotation_jerk_limit
        self.track_only_velocity = track_only_velocity
        self.kwargs = kwargs

    @abc.abstractmethod
    def make_joint(self, god_map: GodMap) -> Joint:
        """
        """

    @abc.abstractmethod
    def make_plugin(self) -> SendTrajectoryToCmdVel:
        """
        """


class OmniDriveCmdVelInterface(DriveInterface):

    def make_joint(self, god_map):
        return OmniDrive(god_map=god_map,
                         parent_link_name=self.parent_link_name,
                         child_link_name=self.child_link_name,
                         name=self.joint_name,
                         odom_x_name=self.odom_x_name,
                         odom_y_name=self.odom_y_name,
                         odom_rot_name=self.odom_rot_name,
                         translation_velocity_limit=self.translation_velocity_limit,
                         rotation_velocity_limit=self.rotation_velocity_limit,
                         translation_acceleration_limit=self.translation_acceleration_limit,
                         rotation_acceleration_limit=self.rotation_acceleration_limit,
                         translation_jerk_limit=self.translation_jerk_limit,
                         rotation_jerk_limit=self.rotation_jerk_limit,
                         **self.kwargs)

    def make_plugin(self):
        return SendTrajectoryToCmdVel(name=self.cmd_vel_topic,
                                      cmd_vel_topic=self.cmd_vel_topic,
                                      track_only_velocity=self.track_only_velocity)


class DiffDriveCmdVelInterface(DriveInterface):
    def make_joint(self, god_map):
        return DiffDrive(god_map=god_map,
                         parent_link_name=self.parent_link_name,
                         child_link_name=self.child_link_name,
                         name=self.joint_name,
                         x_name=self.odom_x_name,
                         rot_name=self.odom_rot_name,
                         translation_velocity_limit=self.translation_velocity_limit,
                         rotation_velocity_limit=self.rotation_velocity_limit,
                         translation_acceleration_limit=self.translation_acceleration_limit,
                         rotation_acceleration_limit=self.rotation_acceleration_limit,
                         translation_jerk_limit=self.translation_jerk_limit,
                         rotation_jerk_limit=self.rotation_jerk_limit,
                         **self.kwargs)

    def make_plugin(self):
        return SendTrajectoryToCmdVel(name=self.cmd_vel_topic,
                                      cmd_vel_topic=self.cmd_vel_topic,
                                      track_only_velocity=self.track_only_velocity)
