from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, List

from giskardpy.configs.behavior_tree_config import ControlModes
from giskardpy.exceptions import GiskardException
from giskardpy.god_map_user import GodMapWorshipper
from giskardpy.my_types import my_string


class RobotInterfaceConfig(GodMapWorshipper, ABC):

    def set_defaults(self):
        pass

    @abstractmethod
    def setup(self):
        ...

    def sync_odometry_topic(self, odometry_topic: str, joint_name: str):
        joint_name = self.world.search_for_joint_name(joint_name)
        self.tree_manager.sync_odometry_topic(odometry_topic, joint_name)

    def sync_6dof_joint_with_tf_frame(self, joint_name: str, tf_parent_frame: str, tf_child_frame: str):
        """
        Tell Giskard to keep track of tf frames, e.g., for robot localization.
        """
        joint_name = self.world.search_for_joint_name(joint_name)
        self.tree_manager.sync_6dof_joint_with_tf_frame(joint_name, tf_parent_frame, tf_child_frame)

    def sync_joint_state_topic(self, topic_name: str, group_name: Optional[str] = None):
        if group_name is None:
            group_name = self.world.robot_name
        self.tree_manager.sync_joint_state_topic(group_name=group_name, topic_name=topic_name)

    def add_base_cmd_velocity(self,
                              cmd_vel_topic: str,
                              joint_name: my_string,
                              track_only_velocity: bool = False):
        """
        Used if the robot's base can be controlled with a Twist topic.
        :param cmd_vel_topic:
        :param track_only_velocity: The tracking mode. If true, any position error is not considered which makes
                                    the tracking smoother but less accurate.
        :param joint_name: name of the omni or diff drive joint. Doesn't need to be specified if there is only one.
        """
        joint_name = self.world.search_for_joint_name(joint_name)
        self.tree_manager.add_base_traj_action_server(cmd_vel_topic, track_only_velocity, joint_name)

    def register_controlled_joints(self, joint_names: List[str], group_name: Optional[str] = None):
        """
        Tell Giskard which joints can be controlled. Only used in standalone mode.
        :param joint_names:
        :param group_name: Only needs to be specified, if there are more than two robots.
        """
        if self.control_mode != ControlModes.standalone:
            raise GiskardException(f'Joints only need to be registered in {ControlModes.standalone.name} mode.')
        joint_names = [self.world.search_for_joint_name(j, group_name) for j in joint_names]
        self.world.register_controlled_joints(joint_names)

    def add_follow_joint_trajectory_server(self,
                                           namespace: str,
                                           state_topic: str,
                                           group_name: Optional[str] = None,
                                           fill_velocity_values: bool = False):
        """
        Connect Giskard to a follow joint trajectory server. It will automatically figure out which joints are offered
        and can be controlled.
        :param namespace: namespace of the action server
        :param state_topic: name of the state topic of the action server
        :param group_name: set if there are multiple robots
        :param fill_velocity_values: whether to fill the velocity entries in the message send to the robot
        """
        if group_name is None:
            group_name = self.world.robot_name
        self.tree_manager.add_follow_joint_traj_action_server(namespace=namespace, state_topic=state_topic,
                                                              group_name=group_name,
                                                              fill_velocity_values=fill_velocity_values)

    def add_joint_velocity_controller(self, namespaces: List[str]):
        self.tree_manager.add_joint_velocity_controllers(namespaces)

    def add_joint_velocity_group_controller(self, namespace: str):
        self.tree_manager.add_joint_velocity_group_controllers(namespace)


class StandAloneRobotInterfaceConfig(RobotInterfaceConfig):
    joint_names: List[str]

    def __init__(self, joint_names: List[str]):
        self.joint_names = joint_names

    def setup(self):
        self.register_controlled_joints(self.joint_names)


class JointTrajServerWithOmniBaseInterface(RobotInterfaceConfig):
    map_name: str
    localization_joint_name: str
    odom_link_name: str
    drive_joint_name: str

    def __init__(self,
                 map_name: str = 'map',
                 localization_joint_name: str = 'localization',
                 odom_link_name: str = 'odom_combined',
                 drive_joint_name: str = 'brumbrum'):
        self.map_name = map_name
        self.localization_joint_name = localization_joint_name
        self.odom_link_name = odom_link_name
        self.drive_joint_name = drive_joint_name

    def setup(self):
        self.sync_6dof_joint_with_tf_frame(joint_name=self.localization_joint_name,
                                           tf_parent_frame=self.map_name,
                                           tf_child_frame=self.odom_link_name)
        self.sync_joint_state_topic('/joint_states')
        self.sync_odometry_topic('/pr2/base_footprint', self.drive_joint_name)
        self.add_follow_joint_trajectory_server(
            namespace='/pr2/whole_body_controller/follow_joint_trajectory',
            state_topic='/pr2/whole_body_controller/state')
        self.add_follow_joint_trajectory_server(
            namespace='/pr2/l_gripper_l_finger_controller/follow_joint_trajectory',
            state_topic='/pr2/l_gripper_l_finger_controller/state')
        self.add_follow_joint_trajectory_server(
            namespace='/pr2/r_gripper_l_finger_controller/follow_joint_trajectory',
            state_topic='/pr2/r_gripper_l_finger_controller/state')
        self.add_base_cmd_velocity(cmd_vel_topic='/pr2/cmd_vel',
                                   track_only_velocity=True,
                                   joint_name=self.drive_joint_name)
