from typing import List

from py_trees import Sequence

from giskardpy.my_types import PrefixName
from giskardpy.tree.behaviors.joint_group_vel_controller_publisher import JointGroupVelController
from giskardpy.tree.behaviors.joint_vel_controller_publisher import JointVelController
from giskardpy.tree.behaviors.send_trajectory_omni_drive_realtime2 import SendCmdVel
from giskardpy.tree.composites.running_selector import RunningSelector
from giskardpy.tree.decorators import success_is_running, running_is_success


class SendControls(RunningSelector):
    def __init__(self, name: str = 'send controls'):
        super().__init__(name)

    def add_joint_velocity_controllers(self, namespaces: List[str]):
        self.add_child(JointVelController(namespaces=namespaces))

    def add_joint_velocity_group_controllers(self, namespace: str):
        self.add_child(JointGroupVelController(namespace))

    def add_send_cmd_velocity(self, cmd_vel_topic: str, joint_name: PrefixName = None):
        self.add_child(SendCmdVel(cmd_vel_topic, joint_name=joint_name))
