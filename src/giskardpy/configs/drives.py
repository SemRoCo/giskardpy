from giskardpy.model.joints import OmniDrive
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