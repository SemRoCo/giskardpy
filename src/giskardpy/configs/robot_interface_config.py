from __future__ import annotations
import abc
from abc import ABC
from collections import defaultdict
from typing import Dict, Optional, List, Union, DefaultDict

import numpy as np
import rospy
from numpy.typing import NDArray
from py_trees import Blackboard
from std_msgs.msg import ColorRGBA

from giskardpy import identifier
from giskardpy.configs.data_types import CollisionCheckerLib, ControlModes, SupportedQPSolver, \
    CollisionAvoidanceGroupConfig, CollisionAvoidanceConfigEntry, TfPublishingModes
from giskardpy.exceptions import GiskardException, SetupException
from giskardpy.goals.goal import Goal
from giskardpy.god_map import GodMap, GodMapUser
from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer
from giskardpy.model.joints import FixedJoint, OmniDrive, DiffDrive, Joint6DOF, OneDofJoint
from giskardpy.model.links import Link
from giskardpy.model.utils import robot_name_from_urdf_string
from giskardpy.model.world import WorldTree
from giskardpy.my_types import my_string, PrefixName, Derivatives, derivative_map
from giskardpy.tree.garden import OpenLoop, ClosedLoop, StandAlone, TreeManager
from giskardpy.utils import logging
from giskardpy.utils.utils import resolve_ros_iris, get_all_classes_in_package

class RobotInterfaceConfig(GodMapUser):

    def set_defaults(self):
        pass

    def sync_odometry_topic(self, odometry_topic: str, joint_name: str):
        joint_name = self.world.search_for_joint_name(joint_name)
        self._behavior_tree.sync_odometry_topic(odometry_topic, joint_name)

    def sync_6dof_joint_with_tf_frame(self, joint_name: str, tf_parent_frame: str, tf_child_frame: str):
        """
        Tell Giskard to keep track of tf frames, e.g., for robot localization.
        """
        joint_name = self.world.search_for_joint_name(joint_name)
        self._behavior_tree.sync_6dof_joint_with_tf_frame(joint_name, tf_parent_frame, tf_child_frame)

    def sync_joint_state_topic(self, topic_name: str, group_name: Optional[str] = None):
        if group_name is None:
            group_name = self.get_default_group_name()
        self._behavior_tree.sync_joint_state_topic(group_name=group_name, topic_name=topic_name)

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
        self._behavior_tree.add_base_traj_action_server(cmd_vel_topic, track_only_velocity, joint_name)

    def register_controlled_joints(self, joint_names: List[str], group_name: Optional[str] = None):
        """
        Tell Giskard which joints can be controlled. Only used in standalone mode.
        :param joint_names:
        :param group_name: Only needs to be specified, if there are more than two robots.
        """
        if self._execution_config.control_mode != ControlModes.standalone:
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
            group_name = self.get_default_group_name()
        self._behavior_tree.add_follow_joint_traj_action_server(namespace=namespace, state_topic=state_topic,
                                                                group_name=group_name,
                                                                fill_velocity_values=fill_velocity_values)

    def add_joint_velocity_controller(self, namespaces: List[str]):
        self._behavior_tree.add_joint_velocity_controllers(namespaces)

    def add_joint_velocity_group_controller(self, namespace: str):
        self._behavior_tree.add_joint_velocity_group_controllers(namespace)