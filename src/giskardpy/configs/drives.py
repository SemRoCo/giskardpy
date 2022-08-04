from giskardpy.model.joints import OmniDrive, DiffDrive
from giskardpy.tree.behaviors.send_trajectory_omni_drive_realtime import OmniDriveCmdVel


class DriveInterface:
    pass


class OmniDriveCmdVelInterface(DriveInterface):
    def __init__(self,
                 cmd_vel_topic: str,
                 parent_link_name: str,
                 child_link_name: str):
        self.cmd_vel_topic = cmd_vel_topic
        self.parent_link_name = parent_link_name
        self.child_link_name = child_link_name

    def make_joint(self, god_map):
        return OmniDrive(god_map=god_map,
                         parent_link_name=self.parent_link_name,
                         child_link_name=self.child_link_name)

    def make_plugin(self):
        return OmniDriveCmdVel(self.cmd_vel_topic, self.cmd_vel_topic)


class DiffDriveCmdVelInterface(DriveInterface):
    def __init__(self,
                 cmd_vel_topic: str,
                 parent_link_name: str,
                 child_link_name: str,
                 translation_velocity_limit: float = 1,
                 rotation_velocity_limit: float = 1):
        self.cmd_vel_topic = cmd_vel_topic
        self.parent_link_name = parent_link_name
        self.child_link_name = child_link_name
        self.translation_velocity_limit = translation_velocity_limit
        self.rotation_velocity_limit = rotation_velocity_limit

    def make_joint(self, god_map):
        return DiffDrive(god_map=god_map,
                         parent_link_name=self.parent_link_name,
                         child_link_name=self.child_link_name,
                         translation_velocity_limit=self.translation_velocity_limit,
                         rotation_velocity_limit=self.rotation_velocity_limit)

    def make_plugin(self):
        return OmniDriveCmdVel(self.cmd_vel_topic, self.cmd_vel_topic)