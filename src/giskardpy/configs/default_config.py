import inspect
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from typing import Dict, Optional, List, Union, Tuple
import giskardpy.utils.tfwrapper as tf
import numpy as np
import rospy
from py_trees import Blackboard

from giskardpy import identifier
from giskardpy.configs.data_types import SupportedQPSolver, CollisionCheckerLib, GeneralConfig, \
    BehaviorTreeConfig, QPSolverConfig, CollisionAvoidanceConfig, ControlModes, RobotInterfaceConfig
from giskardpy.configs.hardware_interface_config import HardwareConfig
from giskardpy.configs.drives import DriveInterface, OmniDriveCmdVelInterface, DiffDriveCmdVelInterface
from giskardpy.configs.follow_joint_trajectory import FollowJointTrajectoryInterface
from giskardpy.data_types import PrefixName
from giskardpy.exceptions import GiskardException
from giskardpy.god_map import GodMap
from giskardpy.model.joints import Joint, FixedJoint
from giskardpy.model.world import WorldTree
from giskardpy.my_types import my_string
from giskardpy.tree.garden import OpenLoop, ClosedLoop, StandAlone
from giskardpy.utils import logging
from giskardpy.utils.time_collector import TimeCollector


class Giskard:
    def __init__(self):
        self.general_config: GeneralConfig = GeneralConfig()
        self.qp_solver_config: QPSolverConfig = QPSolverConfig()
        self.behavior_tree_config: BehaviorTreeConfig = BehaviorTreeConfig()
        self.collision_avoidance_config: CollisionAvoidanceConfig = CollisionAvoidanceConfig()
        self.robot_interface_configs: List[RobotInterfaceConfig] = []
        self.hardware_config: HardwareConfig = HardwareConfig()
        self._god_map = GodMap.init_from_paramserver()
        self._god_map.set_data(identifier.giskard, self)
        self._god_map.set_data(identifier.timer_collector, TimeCollector(self._god_map))
        self._controlled_joints = []
        self.root_link_name = None
        blackboard = Blackboard
        blackboard.god_map = self._god_map
        self._backup = {}

    def add_robot_urdf(self, urdf: str, **kwargs):
        robot = RobotInterfaceConfig(urdf, **kwargs)
        self.robot_interface_configs.append(robot)

    def add_robot_from_parameter_server(self, parameter_name: str = 'robot_description',
                                        joint_state_topics: List[str] = ('/joint_states',), **kwargs):
        urdf = rospy.get_param(parameter_name)
        self.hardware_config.joint_state_topics.extend(joint_state_topics)
        self.add_robot_urdf(urdf, **kwargs)

    def register_controlled_joints(self, joint_names: List[str]):
        self._controlled_joints.extend(joint_names)

    def disable_visualization(self):
        self.behavior_tree_config.plugin_config['CollisionMarker']['enabled'] = False
        self.behavior_tree_config.plugin_config['VisualizationBehavior']['enabled'] = False

    def disable_tf_publishing(self):
        self.behavior_tree_config.plugin_config['TFPublisher']['enabled'] = False

    def add_fixed_joint(self, parent_link: my_string, child_link: my_string, homo_transform: Optional[np.ndarray] = None):
        if homo_transform is None:
            homo_transform = np.eye(4)
        try:
            joints = self._god_map.get_data(identifier.joints_to_add)
        except KeyError:
            joints = []
        if isinstance(parent_link, str):
            parent_link = PrefixName(parent_link, None)
        if isinstance(child_link, str):
            child_link = PrefixName(child_link, None)
        joint_name = PrefixName(f'{parent_link}_{child_link}_fixed_joint', None)
        joint = FixedJoint(name=joint_name,
                           parent_link_name=parent_link,
                           child_link_name=child_link,
                           parent_T_child=homo_transform)
        joints.append(joint)
        self._god_map.set_data(identifier.joints_to_add, joints)

    def set_joint_states_topic(self, topic_name: str):
        self.robot_interface_configs.joint_state_topic = topic_name

    def add_sync_tf_frame(self, parent_link, child_link):
        self.add_fixed_joint(parent_link=parent_link, child_link=child_link)
        self.behavior_tree_config.add_sync_tf_frame(parent_link, child_link)

    def add_odometry_topic(self, odometry_topic):
        self.hardware_config.odometry_topics.append(odometry_topic)

    def add_follow_joint_trajectory_server(self, namespace, state_topic, fill_velocity_values=False):
        self.hardware_config.add_follow_joint_trajectory_server(namespace, state_topic,
                                                                fill_velocity_values=fill_velocity_values)

    def add_omni_drive_interface(self, parent_link_name, child_link_name,
                                 cmd_vel_topic: Optional[str] = None,
                                 track_only_velocity: bool = False,
                                 translation_velocity_limit: Optional[float] = 0.2,
                                 rotation_velocity_limit: Optional[float] = 0.2,
                                 translation_acceleration_limit: Optional[float] = None,
                                 rotation_acceleration_limit: Optional[float] = None,
                                 translation_jerk_limit: Optional[float] = 5,
                                 rotation_jerk_limit: Optional[float] = 10,
                                 odom_x_name: Optional[str] = 'odom_x',
                                 odom_y_name: Optional[str] = 'odom_y',
                                 odom_yaw_name: Optional[str] = 'odom_yaw'):
        self.hardware_config.add_omni_drive_interface(cmd_vel_topic=cmd_vel_topic,
                                                      parent_link_name=parent_link_name,
                                                      child_link_name=child_link_name,
                                                      track_only_velocity=track_only_velocity,
                                                      translation_velocity_limit=translation_velocity_limit,
                                                      rotation_velocity_limit=rotation_velocity_limit,
                                                      translation_acceleration_limit=translation_acceleration_limit,
                                                      rotation_acceleration_limit=rotation_acceleration_limit,
                                                      translation_jerk_limit=translation_jerk_limit,
                                                      rotation_jerk_limit=rotation_jerk_limit,
                                                      odom_x_name=odom_x_name,
                                                      odom_y_name=odom_y_name,
                                                      odom_yaw_name=odom_yaw_name)
        joints = self._god_map.get_data(identifier.joints_to_add, default=[])
        brumbrum_joint = self.hardware_config.drive_interfaces[-1].make_joint()
        joints.append(brumbrum_joint)
        self._controlled_joints.append(brumbrum_joint.name)

    def add_diff_drive_interface(self, parent_link_name: str, child_link_name: str,
                                 cmd_vel_topic: Optional[str] = None,
                                 track_only_velocity: bool = False,
                                 translation_velocity_limit: Optional[float] = 0.2,
                                 rotation_velocity_limit: Optional[float] = 0.2,
                                 translation_acceleration_limit: Optional[float] = None,
                                 rotation_acceleration_limit: Optional[float] = None,
                                 translation_jerk_limit: Optional[float] = 5,
                                 rotation_jerk_limit: Optional[float] = 10):
        self.hardware_config.add_diff_drive_interface(cmd_vel_topic=cmd_vel_topic,
                                                      parent_link_name=parent_link_name,
                                                      child_link_name=child_link_name,
                                                      track_only_velocity=track_only_velocity,
                                                      translation_velocity_limit=translation_velocity_limit,
                                                      rotation_velocity_limit=rotation_velocity_limit,
                                                      translation_acceleration_limit=translation_acceleration_limit,
                                                      rotation_acceleration_limit=rotation_acceleration_limit,
                                                      translation_jerk_limit=translation_jerk_limit,
                                                      rotation_jerk_limit=rotation_jerk_limit)
        joints = self._god_map.get_data(identifier.joints_to_add, default=[])
        brumbrum_joint = self.hardware_config.drive_interfaces[-1].make_joint(self._god_map)
        joints.append(brumbrum_joint)
        self._controlled_joints.append(brumbrum_joint.name)

    def reset_config(self):
        for parameter, value in self._backup.items():
            setattr(self, parameter, deepcopy(value))

    def _create_parameter_backup(self):
        self._backup = {'qp_solver_config': deepcopy(self.qp_solver_config),
                        'general_config': deepcopy(self.general_config)}

    def grow(self):
        if len(self.robot_interface_configs) == 0:
            self.add_robot_from_parameter_server()
        self._create_parameter_backup()
        if self.root_link_name is None:
            self.root_link_name = tf.get_tf_root()
        world = WorldTree(self.root_link_name, self._god_map)
        world.delete_all_but_robot()
        world.register_controlled_joints(self._controlled_joints)

        if self.collision_avoidance_config.collision_checker == CollisionCheckerLib.bpb:
            logging.loginfo('Using bpb for collision checking.')
            from giskardpy.model.better_pybullet_syncer import BetterPyBulletSyncer
            collision_scene = BetterPyBulletSyncer(world)
        elif self.collision_avoidance_config.collision_checker == CollisionCheckerLib.pybullet:
            logging.loginfo('Using pybullet for collision checking.')
            from giskardpy.model.pybullet_syncer import PyBulletSyncer
            collision_scene = PyBulletSyncer(world)
        elif self.collision_avoidance_config.collision_checker == CollisionCheckerLib.none:
            logging.logwarn('Using no collision checking.')
            from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer
            collision_scene = CollisionWorldSynchronizer(world)
        else:
            raise KeyError(f'Unknown collision checker {self.collision_avoidance_config.collision_checker}. '
                           f'Collision avoidance is disabled')
        self._god_map.set_data(identifier.collision_checker, self.collision_avoidance_config.collision_checker)
        self._god_map.set_data(identifier.collision_scene, collision_scene)
        if self.general_config.control_mode == ControlModes.open_loop:
            self._tree = OpenLoop(self._god_map)
        elif self.general_config.control_mode == ControlModes.close_loop:
            self._tree = ClosedLoop(self._god_map)
        elif self.general_config.control_mode == ControlModes.stand_alone:
            self._tree = StandAlone(self._god_map)
        else:
            raise KeyError(f'Robot interface mode \'{self.general_config.control_mode}\' is not supported.')

        self._controlled_joints_sanity_check()

    def _controlled_joints_sanity_check(self):
        world = self._god_map.get_data(identifier.world)
        non_controlled_joints = set(world.movable_joints).difference(set(world.controlled_joints))
        if len(world.controlled_joints) == 0:
            raise GiskardException('No joints are flagged as controlled.')
        logging.loginfo(f'The following joints are non-fixed according to the urdf, '
                        f'but not flagged as controlled: {non_controlled_joints}.')
        if len(self.hardware_config.drive_interfaces) == 0:
            logging.loginfo('No cmd_vel topic has been registered.')

    def live(self):
        self.grow()
        self._god_map.get_data(identifier.tree_manager).live()
