from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, List, Dict

from giskardpy import identifier
from giskardpy.configs.behavior_tree_config import ControlModes
from giskardpy.exceptions import GiskardException
from giskardpy.god_map import GodMap
from giskardpy.model.world import WorldTree
from giskardpy.my_types import my_string, PrefixName, Derivatives
from giskardpy.tree.garden import TreeManager


class RobotInterfaceConfig(ABC):
    god_map = GodMap()

    def set_defaults(self):
        pass

    @abstractmethod
    def setup(self):
        """
        Implement this method to configure how Giskard can talk to the robot using it's self. methods.
        """

    @property
    def world(self) -> WorldTree:
        return self.god_map.get_data(identifier.world)

    @property
    def robot_group_name(self) -> str:
        return self.world.robot_name

    def get_root_link_of_group(self, group_name: str) -> PrefixName:
        return self.world.groups[group_name].root_link_name

    @property
    def tree_manager(self) -> TreeManager:
        return self.god_map.get_data(identifier.tree_manager)

    @property
    def control_mode(self) -> ControlModes:
        return self.god_map.get_data(identifier.control_mode)

    def sync_odometry_topic(self, odometry_topic: str, joint_name: str):
        """
        Tell Giskard to sync an odometry joint added during by the world config.
        """
        joint_name = self.world.search_for_joint_name(joint_name)
        self.tree_manager.sync_odometry_topic(odometry_topic, joint_name)

    def sync_6dof_joint_with_tf_frame(self, joint_name: str, tf_parent_frame: str, tf_child_frame: str):
        """
        Tell Giskard to sync a 6dof joint with a tf frame.
        """
        joint_name = self.world.search_for_joint_name(joint_name)
        self.tree_manager.sync_6dof_joint_with_tf_frame(joint_name, tf_parent_frame, tf_child_frame)

    def sync_joint_state_topic(self, topic_name: str, group_name: Optional[str] = None):
        """
        Tell Giskard to sync the world state with a joint state topic
        """
        if group_name is None:
            group_name = self.world.robot_name
        self.tree_manager.sync_joint_state_topic(group_name=group_name, topic_name=topic_name)

    def add_base_cmd_velocity(self,
                              cmd_vel_topic: str,
                              joint_name: my_string,
                              track_only_velocity: bool = False):
        """
        Tell Giskard how it can control an odom joint of the robot.
        :param cmd_vel_topic: a Twist topic
        :param track_only_velocity: The tracking mode. If true, any position error is not considered which makes
                                    the tracking smoother but less accurate.
        :param joint_name: name of the omni or diff drive joint. Doesn't need to be specified if there is only one.
        """
        joint_name = self.world.search_for_joint_name(joint_name)
        self.tree_manager.add_base_traj_action_server(cmd_vel_topic, track_only_velocity, joint_name)

    def register_controlled_joints(self, joint_names: List[str], group_name: Optional[str] = None):
        """
        Tell Giskard which joints can be controlled. Giskard can usually figure this out on its own.
        Only used in standalone mode.
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
                                           fill_velocity_values: bool = False,
                                           path_tolerance: Dict[Derivatives, float] = None):
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
                                                              fill_velocity_values=fill_velocity_values,
                                                              path_tolerance=path_tolerance)

    def add_joint_velocity_controller(self, namespaces: List[str]):
        """
        For closed loop mode. Tell Giskard how it can send velocities to joints.
        :param namespaces: A list of namespaces where Giskard can find the topics and rosparams.
        """
        self.tree_manager.add_joint_velocity_controllers(namespaces)

    def add_joint_velocity_group_controller(self, namespace: str):
        """
        For closed loop mode. Tell Giskard how it can send velocities for a group of joints.
        :param namespace: where Giskard can find the topic and rosparams.
        """
        self.tree_manager.add_joint_velocity_group_controllers(namespace)


class StandAloneRobotInterfaceConfig(RobotInterfaceConfig):
    joint_names: List[str]

    def __init__(self, joint_names: List[str]):
        self.joint_names = joint_names

    def setup(self):
        self.register_controlled_joints(self.joint_names)
