from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, List, Dict

from giskardpy.god_map import god_map
from giskardpy.tree.branches.giskard_bt import GiskardBT
from giskardpy.tree.control_modes import ControlModes
from giskardpy.exceptions import GiskardException, SetupException
from giskardpy.model.world import WorldTree
from giskardpy.data_types import my_string, PrefixName, Derivatives


class RobotInterfaceConfig(ABC):
    def set_defaults(self):
        pass

    @abstractmethod
    def setup(self):
        """
        Implement this method to configure how Giskard can talk to the robot using it's self. methods.
        """

    @property
    def world(self) -> WorldTree:
        return god_map.world

    @property
    def robot_group_name(self) -> str:
        return self.world.robot_name

    def get_root_link_of_group(self, group_name: str) -> PrefixName:
        return self.world.groups[group_name].root_link_name

    @property
    def tree(self) -> GiskardBT:
        return god_map.tree

    @property
    def control_mode(self) -> ControlModes:
        return god_map.tree.control_mode

    def sync_odometry_topic(self, odometry_topic: str, joint_name: str):
        """
        Tell Giskard to sync an odometry joint added during by the world config.
        """
        joint_name = self.world.search_for_joint_name(joint_name)
        self.tree.wait_for_goal.synchronization.sync_odometry_topic(odometry_topic, joint_name)
        if god_map.is_closed_loop():
            self.tree.control_loop_branch.closed_loop_synchronization.sync_odometry_topic_no_lock(
                odometry_topic,
                joint_name)

    def sync_6dof_joint_with_tf_frame(self, joint_name: str, tf_parent_frame: str, tf_child_frame: str):
        """
        Tell Giskard to sync a 6dof joint with a tf frame.
        """
        joint_name = self.world.search_for_joint_name(joint_name)
        self.tree.wait_for_goal.synchronization.sync_6dof_joint_with_tf_frame(joint_name,
                                                                              tf_parent_frame,
                                                                              tf_child_frame)
        if god_map.is_closed_loop():
            self.tree.control_loop_branch.closed_loop_synchronization.sync_6dof_joint_with_tf_frame(
                joint_name,
                tf_parent_frame,
                tf_child_frame)

    def sync_joint_state_topic(self, topic_name: str, group_name: Optional[str] = None):
        """
        Tell Giskard to sync the world state with a joint state topic
        """
        if group_name is None:
            group_name = self.world.robot_name
        self.tree.wait_for_goal.synchronization.sync_joint_state_topic(group_name=group_name,
                                                                       topic_name=topic_name)
        if god_map.is_closed_loop():
            self.tree.control_loop_branch.closed_loop_synchronization.sync_joint_state2_topic(
                group_name=group_name,
                topic_name=topic_name)

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
        if god_map.is_closed_loop():
            self.tree.control_loop_branch.send_controls.add_send_cmd_velocity(cmd_vel_topic=cmd_vel_topic,
                                                                              joint_name=joint_name)
        elif god_map.is_planning():
            self.tree.execute_traj.add_base_traj_action_server(cmd_vel_topic=cmd_vel_topic,
                                                               joint_name=joint_name)

    def register_controlled_joints(self, joint_names: List[str], group_name: Optional[str] = None):
        """
        Tell Giskard which joints can be controlled. Giskard can usually figure this out on its own.
        Only used in standalone mode.
        :param group_name: Only needs to be specified, if there are more than two robots.
        """
        if self.control_mode != ControlModes.standalone:
            raise SetupException(f'Joints only need to be registered in {ControlModes.standalone.name} mode.')
        joint_names = [self.world.search_for_joint_name(j, group_name) for j in joint_names]
        self.world.register_controlled_joints(joint_names)

    def add_follow_joint_trajectory_server(self,
                                           namespace: str,
                                           group_name: Optional[str] = None,
                                           fill_velocity_values: bool = False,
                                           path_tolerance: Dict[Derivatives, float] = None):
        """
        Connect Giskard to a follow joint trajectory server. It will automatically figure out which joints are offered
        and can be controlled.
        :param namespace: namespace of the action server
        :param group_name: set if there are multiple robots
        :param fill_velocity_values: whether to fill the velocity entries in the message send to the robot
        """
        if group_name is None:
            group_name = self.world.robot_name
        if not god_map.is_planning():
            raise SetupException('add_follow_joint_trajectory_server only works in planning mode')
        self.tree.execute_traj.add_follow_joint_traj_action_server(namespace=namespace,
                                                                   group_name=group_name,
                                                                   fill_velocity_values=fill_velocity_values,
                                                                   path_tolerance=path_tolerance)

    def add_joint_velocity_controller(self, namespaces: List[str]):
        """
        For closed loop mode. Tell Giskard how it can send velocities to joints.
        :param namespaces: A list of namespaces where Giskard can find the topics and rosparams.
        """
        self.tree.control_loop_branch.send_controls.add_joint_velocity_controllers(namespaces)

    def add_joint_velocity_group_controller(self, namespace: str):
        """
        For closed loop mode. Tell Giskard how it can send velocities for a group of joints.
        :param namespace: where Giskard can find the topic and rosparams.
        """
        self.tree.control_loop_branch.send_controls.add_joint_velocity_group_controllers(
            namespace)


class StandAloneRobotInterfaceConfig(RobotInterfaceConfig):
    joint_names: List[str]

    def __init__(self, joint_names: List[str]):
        self.joint_names = joint_names

    def setup(self):
        self.register_controlled_joints(self.joint_names)
