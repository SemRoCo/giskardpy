from collections import defaultdict
from copy import deepcopy
from typing import Dict, Optional, List, Union, Tuple

from std_msgs.msg import ColorRGBA
from tf2_py import LookupException

import giskardpy.utils.tfwrapper as tf
import numpy as np
import rospy
from py_trees import Blackboard

from giskardpy import identifier
from giskardpy.configs.data_types import CollisionCheckerLib, GeneralConfig, \
    BehaviorTreeConfig, QPSolverConfig, CollisionAvoidanceConfig, ControlModes, RobotInterfaceConfig, HardwareConfig, \
    TfPublishingModes, CollisionAvoidanceConfigEntry
from giskardpy.data_types import derivative_to_name
from giskardpy.exceptions import GiskardException
from giskardpy.god_map import GodMap
from giskardpy.model.utils import robot_name_from_urdf_string
from giskardpy.model.joints import Joint, FixedJoint, OmniDrive, DiffDrive
from giskardpy.model.world import WorldTree
from giskardpy.my_types import my_string, PrefixName
from giskardpy.qp.qp_solver import QPSolver
from giskardpy.tree.garden import OpenLoop, ClosedLoop, StandAlone
from giskardpy.utils import logging
from giskardpy.utils.time_collector import TimeCollector
from giskardpy.utils.utils import resolve_ros_iris


class Giskard:
    def __init__(self):
        self._collision_checker: CollisionCheckerLib = CollisionCheckerLib.bpb
        self._general_config: GeneralConfig = GeneralConfig()
        self._qp_solver_config: QPSolverConfig = QPSolverConfig()
        self.behavior_tree_config: BehaviorTreeConfig = BehaviorTreeConfig()
        self._collision_avoidance_configs: Dict[str, CollisionAvoidanceConfig] = defaultdict(CollisionAvoidanceConfig)
        self._god_map = GodMap.init_from_paramserver()
        self._god_map.set_data(identifier.giskard, self)
        self._god_map.set_data(identifier.timer_collector, TimeCollector(self._god_map))
        self._controlled_joints = []
        self.root_link_name = None
        blackboard = Blackboard
        blackboard.god_map = self._god_map
        self._backup = {}

    def get_collision_avoidance_config(self, group_name: Optional[str] = None):
        if group_name is None:
            group_name = self.get_default_group_name()
        return self._collision_avoidance_configs[group_name]

    def add_robot_urdf(self, urdf: str, name: str, **kwargs):
        if not hasattr(self, 'robot_interface_configs'):
            self.group_names = []
            self.robot_interface_configs: List[RobotInterfaceConfig] = []
            self.hardware_config: HardwareConfig = HardwareConfig()
        if name is None:
            name = robot_name_from_urdf_string(urdf)
            self.group_names.append(name)
        robot = RobotInterfaceConfig(urdf, name=name, **kwargs)
        self.robot_interface_configs.append(robot)

    def configure_VisualizationBehavior(self, enabled=True, in_planning_loop=False):
        self.behavior_tree_config.plugin_config['VisualizationBehavior']['enabled'] = enabled
        self.behavior_tree_config.plugin_config['VisualizationBehavior']['in_planning_loop'] = in_planning_loop

    def configure_CollisionMarker(self, enabled=True, in_planning_loop=False):
        self.behavior_tree_config.plugin_config['CollisionMarker']['enabled'] = enabled
        self.behavior_tree_config.plugin_config['CollisionMarker']['in_planning_loop'] = in_planning_loop

    def add_robot_from_parameter_server(self, parameter_name: str = 'robot_description',
                                        joint_state_topics: List[str] = ('/joint_states',),
                                        group_name: Optional[str] = None, **kwargs):
        urdf = rospy.get_param(parameter_name)
        self.add_robot_urdf(urdf, name=group_name, **kwargs)
        if group_name is None:
            group_name = self.get_default_group_name()
        js_kwargs = [{'group_name': group_name, 'joint_state_topic': topic} for topic in joint_state_topics]
        self.hardware_config.joint_state_topics_kwargs.extend(js_kwargs)

    def register_controlled_joints(self, joint_names: List[str], group_name: Optional[str] = None):
        if group_name is None:
            group_name = self.group_names[0]
        joint_names = [PrefixName(j, group_name) for j in joint_names]
        self._controlled_joints.extend(joint_names)

    def disable_visualization(self):
        self.behavior_tree_config.plugin_config['CollisionMarker']['enabled'] = False
        self.behavior_tree_config.plugin_config['VisualizationBehavior']['enabled'] = False

    def disable_tf_publishing(self):
        self.behavior_tree_config.plugin_config['TFPublisher']['enabled'] = False

    def publish_all_tf(self):
        self.behavior_tree_config.plugin_config['TFPublisher']['mode'] = TfPublishingModes.all

    def add_joint(self, joint: Joint):
        joints = self._god_map.get_data(identifier.joints_to_add, default=[])
        joints.append(joint)

    def add_fixed_joint(self, parent_link: my_string, child_link: my_string,
                        homo_transform: Optional[np.ndarray] = None):
        if homo_transform is None:
            homo_transform = np.eye(4)
        if isinstance(parent_link, str):
            parent_link = PrefixName(parent_link, None)
        if isinstance(child_link, str):
            child_link = PrefixName(child_link, None)
        joint_name = PrefixName(f'{parent_link}_{child_link}_fixed_joint', None)
        joint = FixedJoint(name=joint_name,
                           parent_link_name=parent_link,
                           child_link_name=child_link,
                           parent_T_child=homo_transform)
        self.add_joint(joint)

    def set_joint_states_topic(self, topic_name: str):
        self.robot_interface_configs.joint_state_topic = topic_name

    def add_sync_tf_frame(self, parent_link, child_link):
        if not tf.wait_for_transform(parent_link, child_link, rospy.Time(), rospy.Duration(1)):
            raise LookupException(f'Cannot get transform of {parent_link}<-{child_link}')
        self.add_fixed_joint(parent_link=parent_link, child_link=child_link)
        self.behavior_tree_config.add_sync_tf_frame(parent_link, child_link)

    def add_odometry_topic(self, odometry_topic, joint_name):
        self.hardware_config.odometry_node_kwargs.append({'odometry_topic': odometry_topic,
                                                          'joint_name': joint_name})

    def add_follow_joint_trajectory_server(self, namespace, state_topic, group_name=None, fill_velocity_values=False):
        if group_name is None:
            group_name = self.get_default_group_name()
        self.hardware_config.follow_joint_trajectory_interfaces_kwargs.append({'action_namespace': namespace,
                                                                               'state_topic': state_topic,
                                                                               'group_name': group_name,
                                                                               'fill_velocity_values': fill_velocity_values})

    def add_omni_drive_joint(self,
                             parent_link_name: str,
                             child_link_name: str,
                             robot_group_name: Optional[str] = None,
                             name: Optional[str] = 'brumbrum',
                             odometry_topic: Optional[str] = None,
                             translation_velocity_limit: Optional[float] = 0.2,
                             rotation_velocity_limit: Optional[float] = 0.2,
                             translation_acceleration_limit: Optional[float] = None,
                             rotation_acceleration_limit: Optional[float] = None,
                             translation_jerk_limit: Optional[float] = 5,
                             rotation_jerk_limit: Optional[float] = 10,
                             odom_x_name: Optional[str] = 'odom_x',
                             odom_y_name: Optional[str] = 'odom_y',
                             odom_yaw_name: Optional[str] = 'odom_yaw'):
        if robot_group_name is None:
            robot_group_name = self.get_default_group_name()
        brumbrum_joint = OmniDrive(parent_link_name=parent_link_name,
                                   child_link_name=PrefixName(child_link_name, robot_group_name),
                                   name=name,
                                   group_name=robot_group_name,
                                   odom_x_name=odom_x_name,
                                   odom_y_name=odom_y_name,
                                   odom_yaw_name=odom_yaw_name,
                                   translation_velocity_limit=translation_velocity_limit,
                                   rotation_velocity_limit=rotation_velocity_limit,
                                   translation_acceleration_limit=translation_acceleration_limit,
                                   rotation_acceleration_limit=rotation_acceleration_limit,
                                   translation_jerk_limit=translation_jerk_limit,
                                   rotation_jerk_limit=rotation_jerk_limit)
        self.add_joint(brumbrum_joint)
        if odometry_topic is not None:
            self.add_odometry_topic(odometry_topic=odometry_topic,
                                    joint_name=brumbrum_joint.name)

    def get_default_group_name(self):
        if len(self.group_names) > 1:
            raise AttributeError(f'group name has to be set if you have multiple robots')
        return self.group_names[0]

    def add_diff_drive_joint(self,
                             parent_link_name: str,
                             child_link_name: str,
                             name: Optional[str] = 'brumbrum',
                             odometry_topic: Optional[str] = None,
                             translation_velocity_limit: Optional[float] = 0.2,
                             rotation_velocity_limit: Optional[float] = 0.2,
                             translation_acceleration_limit: Optional[float] = None,
                             rotation_acceleration_limit: Optional[float] = None,
                             translation_jerk_limit: Optional[float] = 5,
                             rotation_jerk_limit: Optional[float] = 10,
                             odom_x_name: Optional[str] = 'odom_x',
                             odom_y_name: Optional[str] = 'odom_y',
                             odom_yaw_name: Optional[str] = 'odom_yaw'):
        brumbrum_joint = DiffDrive(parent_link_name=parent_link_name,
                                   child_link_name=child_link_name,
                                   name=name,
                                   odom_x_name=odom_x_name,
                                   odom_y_name=odom_y_name,
                                   odom_yaw_name=odom_yaw_name,
                                   translation_velocity_limit=translation_velocity_limit,
                                   rotation_velocity_limit=rotation_velocity_limit,
                                   translation_acceleration_limit=translation_acceleration_limit,
                                   rotation_acceleration_limit=rotation_acceleration_limit,
                                   translation_jerk_limit=translation_jerk_limit,
                                   rotation_jerk_limit=rotation_jerk_limit)
        self.add_joint(brumbrum_joint)
        if odometry_topic is not None:
            self.add_odometry_topic(odometry_topic=odometry_topic,
                                    joint_name=brumbrum_joint.name)

    def add_base_cmd_velocity(self,
                              cmd_vel_topic: str,
                              track_only_velocity: bool = False,
                              joint_name: Optional[my_string] = None):
        self.hardware_config.send_trajectory_to_cmd_vel_kwargs.append({'cmd_vel_topic': cmd_vel_topic,
                                                                       'track_only_velocity': track_only_velocity,
                                                                       'joint_name': joint_name})

    def reset_config(self):
        for parameter, value in self._backup.items():
            setattr(self, parameter, deepcopy(value))

    def _create_parameter_backup(self):
        self._backup = {'qp_solver_config': deepcopy(self._qp_solver_config),
                        'general_config': deepcopy(self._general_config)}

    def grow(self):
        if len(self.robot_interface_configs) == 0:
            self.add_robot_from_parameter_server()
        self._create_parameter_backup()
        if self.root_link_name is None:
            self.root_link_name = tf.get_tf_root()
        world = WorldTree(self.root_link_name, self._god_map)
        world.delete_all_but_robots()
        world.register_controlled_joints(self._controlled_joints)

        if self._collision_checker == CollisionCheckerLib.bpb:
            logging.loginfo('Using bpb for collision checking.')
            from giskardpy.model.better_pybullet_syncer import BetterPyBulletSyncer
            collision_scene = BetterPyBulletSyncer(world)
        elif self._collision_checker == CollisionCheckerLib.pybullet:
            logging.loginfo('Using pybullet for collision checking.')
            from giskardpy.model.pybullet_syncer import PyBulletSyncer
            collision_scene = PyBulletSyncer(world)
        elif self._collision_checker == CollisionCheckerLib.none:
            logging.logwarn('Using no collision checking.')
            from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer
            collision_scene = CollisionWorldSynchronizer(world)
        else:
            raise KeyError(f'Unknown collision checker {self._collision_checker}. '
                           f'Collision avoidance is disabled')
        self._god_map.set_data(identifier.collision_checker, self._collision_checker)
        self._god_map.set_data(identifier.collision_scene, collision_scene)
        if self._general_config.control_mode == ControlModes.open_loop:
            self._tree = OpenLoop(self._god_map)
        elif self._general_config.control_mode == ControlModes.close_loop:
            self._tree = ClosedLoop(self._god_map)
        elif self._general_config.control_mode == ControlModes.stand_alone:
            self._tree = StandAlone(self._god_map)
        else:
            raise KeyError(f'Robot interface mode \'{self._general_config.control_mode}\' is not supported.')

        self._controlled_joints_sanity_check()

    def _controlled_joints_sanity_check(self):
        world = self._god_map.get_data(identifier.world)
        non_controlled_joints = set(world.movable_joints).difference(set(world.controlled_joints))
        if len(world.controlled_joints) == 0:
            raise GiskardException('No joints are flagged as controlled.')
        logging.loginfo(f'The following joints are non-fixed according to the urdf, '
                        f'but not flagged as controlled: {non_controlled_joints}.')
        if len(self.hardware_config.send_trajectory_to_cmd_vel_kwargs) == 0:
            logging.loginfo('No cmd_vel topic has been registered.')

    @property
    def world(self) -> WorldTree:
        return self._god_map.get_data(identifier.world)

    def live(self):
        self.grow()
        self._god_map.get_data(identifier.tree_manager).live()

    def set_default_visualization_marker_color(self, r, g, b, a):
        self._general_config.default_link_color = ColorRGBA(r, g, b, a)

    # Collision avoidance

    def set_default_self_collision_avoidance(self,
                                             number_of_repeller: int = 1,
                                             soft_threshold: float = 0.05,
                                             hard_threshold: float = 0.0,
                                             max_velocity: float = 0.2,
                                             group_name: Optional[str] = None):
        """
        Sets the default self collision configuration. The default of this function are set automatically.
        If they are fine, you don't need to use this function.
        :param number_of_repeller: how many constraints are added for a particular link pair
        :param soft_threshold: will try to stay out of this threshold, but can violate
        :param hard_threshold: distance threshold not allowed to be violated
        :param max_velocity: how fast it will move away from collisions
        :param group_name: name of the group this default will be applied to
        """
        if group_name is None:
            group_name = self.get_default_group_name()
        new_default = CollisionAvoidanceConfigEntry(
            number_of_repeller=number_of_repeller,
            soft_threshold=soft_threshold,
            hard_threshold=hard_threshold,
            max_velocity=max_velocity
        )
        self._collision_avoidance_configs[group_name].self_collision_avoidance.default_factory = lambda: new_default

    def set_default_external_collision_avoidance(self,
                                                 number_of_repeller: int = 1,
                                                 soft_threshold: float = 0.05,
                                                 hard_threshold: float = 0.0,
                                                 max_velocity: float = 0.2):
        """
        Sets the default external collision configuration. The default of this function are set automatically.
        If they are fine, you don't need to use this function.
        :param number_of_repeller: How many constraints are added for a joint to avoid collisions
        :param soft_threshold: will try to stay out of this threshold, but can violate
        :param hard_threshold: distance threshold not allowed to be violated
        :param max_velocity: how fast it will move away from collisions
        """
        for config in self._collision_avoidance_configs.values():
            config.external_collision_avoidance.default_factory = lambda: CollisionAvoidanceConfigEntry(
                number_of_repeller=number_of_repeller,
                soft_threshold=soft_threshold,
                hard_threshold=hard_threshold,
                max_velocity=max_velocity
            )

    def overwrite_external_collision_avoidance(self,
                                               joint_name: str,
                                               group_name: Optional[str] = None,
                                               number_of_repeller: Optional[int] = None,
                                               soft_threshold: Optional[float] = None,
                                               hard_threshold: Optional[float] = None,
                                               max_velocity: Optional[float] = None):
        """
        :param joint_name:
        :param group_name: if there is only one robot, it will default to it
        :param number_of_repeller: How many constraints are added for a joint to avoid collisions
        :param soft_threshold: will try to stay out of this threshold, but can violate
        :param hard_threshold: distance threshold not allowed to be violated
        :param max_velocity: how fast it will move away from collisions
        """
        if group_name is None:
            group_name = self.get_default_group_name()
        config = self.get_collision_avoidance_config(group_name)
        joint_name = PrefixName(joint_name, group_name)
        if number_of_repeller is not None:
            config.external_collision_avoidance[joint_name].number_of_repeller = number_of_repeller
        if soft_threshold is not None:
            config.external_collision_avoidance[joint_name].soft_threshold = soft_threshold
        if hard_threshold is not None:
            config.external_collision_avoidance[joint_name].hard_threshold = hard_threshold
        if max_velocity is not None:
            config.external_collision_avoidance[joint_name].max_velocity = max_velocity

    def overwrite_self_collision_avoidance(self,
                                           link_name: str,
                                           group_name: Optional[str] = None,
                                           number_of_repeller: Optional[int] = None,
                                           soft_threshold: Optional[float] = None,
                                           hard_threshold: Optional[float] = None,
                                           max_velocity: Optional[float] = None):
        """
        :param link_name:
        :param group_name: if there is only one robot, it will default to it
        :param number_of_repeller: How many constraints are added for a joint to avoid collisions
        :param soft_threshold: will try to stay out of this threshold, but can violate
        :param hard_threshold: distance threshold not allowed to be violated
        :param max_velocity: how fast it will move away from collisions
        """
        if group_name is None:
            group_name = self.get_default_group_name()
        config = self.get_collision_avoidance_config(group_name)
        link_name = PrefixName(link_name, group_name)
        if number_of_repeller is not None:
            config.self_collision_avoidance[link_name].number_of_repeller = number_of_repeller
        if soft_threshold is not None:
            config.self_collision_avoidance[link_name].soft_threshold = soft_threshold
        if hard_threshold is not None:
            config.self_collision_avoidance[link_name].hard_threshold = hard_threshold
        if max_velocity is not None:
            config.self_collision_avoidance[link_name].max_velocity = max_velocity

    def load_moveit_self_collision_matrix(self, path_to_srdf: str, group_name: Optional[str] = None):
        """
        Giskard only has a limited ability to compute a self collision matrix. With this function you can load one
        from Moveit.
        :param path_to_srdf: path to the srdf, can handle ros package paths
        :param group_name: name of the robot for which it will be applied, only needs to be set if there are multiple robots.
        """
        import lxml.etree as ET
        path_to_srdf = resolve_ros_iris(path_to_srdf)
        srdf = ET.parse(path_to_srdf)
        srdf_root = srdf.getroot()
        for child in srdf_root:
            if hasattr(child, 'tag') and child.tag == 'disable_collisions':
                link1 = child.attrib['link1']
                link2 = child.attrib['link2']
                reason = child.attrib['reason']
                if reason in ['Never', 'Adjacent', 'Default']:
                    self.ignore_self_collisions_of_pair(link1, link2, group_name)
        logging.loginfo(f'loaded {path_to_srdf} for self collision avoidance matrix')

    def ignore_all_self_collisions_of_link(self, link_name: str, group_name: Optional[str] = None):
        """
        Completely turn off self collision avoidance for this link.
        """
        if group_name is None:
            group_name = self.get_default_group_name()
        config = self.get_collision_avoidance_config(group_name)
        link_name = PrefixName(link_name, group_name)
        config.ignored_self_collisions.append(link_name)

    def fix_joints_for_self_collision_avoidance(self, joint_names: List[str], group_name: Optional[str] = None):
        """
        Flag some joints as fixed for self collision avoidance. These joints will not be moved to avoid self
        collisions.
        """
        if group_name is None:
            group_name = self.get_default_group_name()
        config = self.get_collision_avoidance_config(group_name)
        joint_names = [PrefixName(joint_name, group_name) for joint_name in joint_names]
        config.fixed_joints_for_self_collision_avoidance.extend(joint_names)

    def fix_joints_for_external_collision_avoidance(self, joint_names: List[str], group_name: Optional[str] = None):
        """
        Flag some joints ad fixed for external collision avoidance. These joints will not be moved to avoid
        external collisions.
        """
        if group_name is None:
            group_name = self.get_default_group_name()
        config = self.get_collision_avoidance_config(group_name)
        joint_names = [PrefixName(joint_name, group_name) for joint_name in joint_names]
        config.fixed_joints_for_external_collision_avoidance.extend(joint_names)

    def ignore_self_collisions_of_pair(self, link_name1: str, link_name2: str, group_name: Optional[str] = None):
        """
        Ignore a certain pair of links for self collision avoidance.
        """
        if group_name is None:
            group_name = self.get_default_group_name()
        config = self.get_collision_avoidance_config(group_name)
        link_name1 = PrefixName(link_name1, group_name)
        link_name2 = PrefixName(link_name2, group_name)
        config.ignored_self_collisions.append((link_name1, link_name2))

    def add_self_collision(self, link_name1: str, link_name2: str, group_name: Optional[str] = None):
        """
        Specifically add a link pair for self collision avoidance.
        """
        if group_name is None:
            group_name = self.get_default_group_name()
        config = self.get_collision_avoidance_config(group_name)
        link_name1 = PrefixName(link_name1, group_name)
        link_name2 = PrefixName(link_name2, group_name)
        config.add_self_collisions.append((link_name1, link_name2))

    def set_collision_checker(self, new_collision_checker: CollisionCheckerLib):
        self._collision_checker = new_collision_checker

    def ignore_all_collisions_of_links(self, link_names: List[str], group_name: Optional[str] = None):
        if group_name is None:
            group_name = self.get_default_group_name()
        link_names = [PrefixName(link_name, group_name) for link_name in link_names]
        self._collision_avoidance_configs[group_name].ignored_collisions.extend(link_names)

    # QP stuff

    def set_prediction_horizon(self, new_prediction_horizon: float):
        """
        Set the prediction horizon for the MPC. If set to 1, it will turn off acceleration and jerk limits.
        :param new_prediction_horizon: should be 1 or >= 5
        """
        self._qp_solver_config.prediction_horizon = new_prediction_horizon

    def set_qp_solver(self, new_solver: QPSolver):
        self._qp_solver_config.qp_solver = new_solver

    def set_default_joint_limits(self,
                                 velocity_limit: float = 1,
                                 acceleration_limit: Optional[float] = 1e3,
                                 jerk_limit: Optional[float] = 30):
        if jerk_limit is not None and acceleration_limit is None:
            raise AttributeError('If jerk limits are set, acceleration limits also have to be set/')
        self._general_config.joint_limits = {
            derivative_to_name[1]: defaultdict(lambda: velocity_limit)
        }
        if acceleration_limit is not None:
            self._general_config.joint_limits[derivative_to_name[2]] = defaultdict(lambda: acceleration_limit)
        if jerk_limit is not None:
            self._general_config.joint_limits[derivative_to_name[3]] = defaultdict(lambda: jerk_limit)

    def overwrite_joint_velocity_limits(self, joint_name, velocity_limit: float, group_name: Optional[str] = None):
        if group_name is None:
            group_name = self.get_default_group_name()
        joint_name = PrefixName(joint_name, group_name)
        self._general_config.joint_limits[derivative_to_name[1]][joint_name] = velocity_limit

    def overwrite_joint_acceleration_limits(self, joint_name, acceleration_limit: float,
                                            group_name: Optional[str] = None):
        if group_name is None:
            group_name = self.get_default_group_name()
        joint_name = PrefixName(joint_name, group_name)
        self._general_config.joint_limits[derivative_to_name[2]][joint_name] = acceleration_limit

    def overwrite_joint_jerk_limits(self, joint_name, jerk_limit: float, group_name: Optional[str] = None):
        if group_name is None:
            group_name = self.get_default_group_name()
        joint_name = PrefixName(joint_name, group_name)
        self._general_config.joint_limits[derivative_to_name[3]][joint_name] = jerk_limit

    def set_default_weights(self,
                            velocity_weight: float = 0.001,
                            acceleration_weight: Optional[float] = None,
                            jerk_weight: Optional[float] = 0.001):
        self._qp_solver_config.joint_weights = {
            derivative_to_name[1]: defaultdict(lambda: velocity_weight)
        }
        if jerk_weight is not None:
            self._qp_solver_config.joint_weights[derivative_to_name[2]] = defaultdict(lambda: acceleration_weight)
            self._qp_solver_config.joint_weights[derivative_to_name[3]] = defaultdict(lambda: jerk_weight)
        elif acceleration_weight is not None:
            self._qp_solver_config.joint_weights[derivative_to_name[2]] = defaultdict(lambda: acceleration_weight)

    def overwrite_joint_velocity_weight(self,
                                        joint_name: str,
                                        velocity_weight: float,
                                        group_name: Optional[str] = None):
        if group_name is None:
            group_name = self.get_default_group_name()
        joint_name = PrefixName(joint_name, group_name)
        self._qp_solver_config.joint_weights[derivative_to_name[1]][joint_name] = velocity_weight

    def overwrite_joint_acceleration_weight(self,
                                            joint_name: str,
                                            acceleration_weight: float,
                                            group_name: Optional[str] = None):
        if group_name is None:
            group_name = self.get_default_group_name()
        joint_name = PrefixName(joint_name, group_name)
        self._qp_solver_config.joint_weights[derivative_to_name[2]][joint_name] = acceleration_weight

    def overwrite_joint_jerk_weight(self,
                                    joint_name: str,
                                    jerk_weight: float,
                                    group_name: Optional[str] = None):
        if group_name is None:
            group_name = self.get_default_group_name()
        joint_name = PrefixName(joint_name, group_name)
        self._qp_solver_config.joint_weights[derivative_to_name[3]][joint_name] = jerk_weight
