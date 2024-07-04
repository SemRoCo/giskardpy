from typing import List

from giskardpy.data_types.data_types import PrefixName
from giskardpy_ros.tree.behaviors.joint_group_vel_controller_publisher import JointGroupVelController
from giskardpy_ros.tree.behaviors.joint_pos_controller_publisher import JointPosController
from giskardpy_ros.tree.behaviors.joint_vel_controller_publisher import JointVelController
from giskardpy_ros.tree.behaviors.send_cmd_vel import SendCmdVel
from giskardpy_ros.tree.composites.running_selector import RunningSelector


class SendControls(RunningSelector):
    def __init__(self, name: str = 'send controls'):
        super().__init__(name)

    def add_joint_velocity_controllers(self, namespaces: List[str]):
        self.add_child(JointVelController(namespaces=namespaces))

    def add_joint_position_controllers(self, namespaces: List[str]):
        self.add_child(JointPosController(namespaces=namespaces))

    def add_joint_velocity_group_controllers(self, namespace: str):
        self.add_child(JointGroupVelController(namespace))

    def add_send_cmd_velocity(self, cmd_vel_topic: str, joint_name: PrefixName = None):
        self.add_child(SendCmdVel(cmd_vel_topic, joint_name=joint_name))
