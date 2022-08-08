from giskardpy.model.joints import OmniDrive, DiffDrive
from giskardpy.tree.behaviors.send_trajectory_omni_drive_realtime import SendTrajectoryToCmdVel


class DriveInterface:
    pass


class OmniDriveCmdVelInterface(DriveInterface):
    def __init__(self,
                 cmd_vel_topic: str,
                 parent_link_name: str,
                 child_link_name: str,
                 translation_velocity_limit: float = 1,
                 rotation_velocity_limit: float = 1,
                 **omni_drive_params):
        self.cmd_vel_topic = cmd_vel_topic
        self.parent_link_name = parent_link_name
        self.child_link_name = child_link_name
        self.translation_velocity_limit = translation_velocity_limit
        self.rotation_velocity_limit = rotation_velocity_limit
        self.omni_drive_params = omni_drive_params

    def make_joint(self, god_map):
        return OmniDrive(god_map=god_map,
                         parent_link_name=self.parent_link_name,
                         child_link_name=self.child_link_name,
                         translation_velocity_limit=self.translation_velocity_limit,
                         rotation_velocity_limit=self.rotation_velocity_limit,
                         **self.omni_drive_params)

    def make_plugin(self):
        return SendTrajectoryToCmdVel(self.cmd_vel_topic, self.cmd_vel_topic)


class DiffDriveCmdVelInterface(DriveInterface):
    def __init__(self,
                 cmd_vel_topic: str,
                 parent_link_name: str,
                 child_link_name: str,
                 translation_velocity_limit: float = 1,
                 rotation_velocity_limit: float = 1,
                 **diff_drive_params):
        self.cmd_vel_topic = cmd_vel_topic
        self.parent_link_name = parent_link_name
        self.child_link_name = child_link_name
        self.translation_velocity_limit = translation_velocity_limit
        self.rotation_velocity_limit = rotation_velocity_limit
        self.diff_drive_params = diff_drive_params

    def make_joint(self, god_map):
        return DiffDrive(god_map=god_map,
                         parent_link_name=self.parent_link_name,
                         child_link_name=self.child_link_name,
                         translation_velocity_limit=self.translation_velocity_limit,
                         rotation_velocity_limit=self.rotation_velocity_limit,
                         **self.diff_drive_params)

    def make_plugin(self):
        return SendTrajectoryToCmdVel(self.cmd_vel_topic, self.cmd_vel_topic)