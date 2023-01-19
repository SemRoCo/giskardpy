from collections import defaultdict
from copy import deepcopy
from typing import Dict, Optional, List

import numpy as np
import rospy
from numpy.typing import NDArray
from py_trees import Blackboard
from std_msgs.msg import ColorRGBA
from tf2_py import LookupException

import giskardpy.utils.tfwrapper as tf
from giskardpy import identifier
from giskardpy.configs.data_types import CollisionCheckerLib, GeneralConfig, \
    BehaviorTreeConfig, QPSolverConfig, CollisionAvoidanceConfig, ControlModes, RobotInterfaceConfig, HardwareConfig, \
    TfPublishingModes, CollisionAvoidanceConfigEntry, SupportedQPSolver
from giskardpy.exceptions import GiskardException
from giskardpy.goals.goal import Goal
from giskardpy.god_map import GodMap
from giskardpy.model.joints import Joint, FixedJoint, OmniDrive, DiffDrive
from giskardpy.model.utils import robot_name_from_urdf_string
from giskardpy.model.world import WorldTree
from giskardpy.my_types import my_string, PrefixName, Derivatives
from giskardpy.tree.garden import OpenLoop, ClosedLoop, StandAlone
from giskardpy.utils import logging
from giskardpy.utils.time_collector import TimeCollector
from giskardpy.utils.utils import resolve_ros_iris, get_all_classes_in_package


class Giskard:
    def __init__(self):
        self._collision_checker: CollisionCheckerLib = CollisionCheckerLib.bpb
        self._general_config: GeneralConfig = GeneralConfig()
        self._qp_solver_config: QPSolverConfig = QPSolverConfig()
        self.behavior_tree_config: BehaviorTreeConfig = BehaviorTreeConfig()
        self._collision_avoidance_configs: Dict[str, CollisionAvoidanceConfig] = defaultdict(CollisionAvoidanceConfig)
        self._god_map = GodMap()
        self._god_map.set_data(identifier.giskard, self)
        self._god_map.set_data(identifier.timer_collector, TimeCollector(self._god_map))
        self._god_map.set_data(identifier.joints_to_add, [])
        self._controlled_joints = []
        self._root_link_name = None
        blackboard = Blackboard
        blackboard.god_map = self._god_map
        self._backup = {}
        self.goal_package_paths = ['giskardpy.goals']

    def add_goal_package_name(self, package_name: str):
        new_goals = get_all_classes_in_package(package_name, Goal)
        if len(new_goals) == 0:
            raise GiskardException(f'No classes of type \'Goal\' found in {package_name}')
        logging.loginfo(f'Made goal classes {new_goals} available Giskard.')
        self.goal_package_paths.append(package_name)

    def set_root_link_name(self, link_name: str):
        """
        Set the name of the root link of the world. Only required in standalone mode.
        """
        self._root_link_name = PrefixName.from_string(link_name, set_none_if_no_slash=True)

    def _get_collision_avoidance_config(self, group_name: Optional[str] = None):
        if group_name is None:
            group_name = self.get_default_group_name()
        return self._collision_avoidance_configs[group_name]

    def add_robot_urdf(self,
                       urdf: str,
                       group_name: str,
                       joint_state_topics: List[str] = ('/joint_states',),
                       add_drive_joint_to_group: bool = True):
        """
        Add a robot urdf to the world.
        :param urdf: robot urdf as string, not the path
        :param group_name:
        :param joint_state_topics:
        """
        if not hasattr(self, 'robot_interface_configs'):
            self.group_names = []
            self.robot_interface_configs: List[RobotInterfaceConfig] = []
            self.hardware_config: HardwareConfig = HardwareConfig()
        if group_name is None:
            group_name = robot_name_from_urdf_string(urdf)
            assert group_name not in self.group_names
            self.group_names.append(group_name)
        robot = RobotInterfaceConfig(urdf, name=group_name, add_drive_joint_to_group=add_drive_joint_to_group)
        self.robot_interface_configs.append(robot)
        js_kwargs = [{'group_name': group_name, 'joint_state_topic': topic} for topic in joint_state_topics]
        self.hardware_config.joint_state_topics_kwargs.extend(js_kwargs)

    def add_robot_from_parameter_server(self,
                                        parameter_name: str = 'robot_description',
                                        joint_state_topics: List[str] = ('/joint_states',),
                                        group_name: Optional[str] = None,
                                        add_drive_joint_to_group: bool = True):
        """
        Add a robot urdf from parameter server to Giskard.
        :param parameter_name:
        :param joint_state_topics: A list of topics where the robot's states are published. Joint names have to match
                                    with the urdf
        :param group_name: How to call the robot. If nothing is specified it will get the name it has in the urdf
        """
        urdf = rospy.get_param(parameter_name)
        self.add_robot_urdf(urdf, group_name=group_name, joint_state_topics=joint_state_topics,
                            add_drive_joint_to_group=add_drive_joint_to_group)

    def configure_MaxTrajectoryLength(self, enabled: bool = True, length: float = 30):
        self.behavior_tree_config.plugin_config['MaxTrajectoryLength']['enabled'] = enabled
        self.behavior_tree_config.plugin_config['MaxTrajectoryLength']['length'] = length

    def configure_VisualizationBehavior(self, enabled: bool = True, in_planning_loop: bool = False):
        """
        :param enabled: whether Giskard should publish markers during planning
        :param in_planning_loop: whether Giskard should update the markers after every control step. Will slow down
                                    the system.
        """
        self.behavior_tree_config.plugin_config['VisualizationBehavior']['enabled'] = enabled
        self.behavior_tree_config.plugin_config['VisualizationBehavior']['in_planning_loop'] = in_planning_loop

    def configure_PublishDebugExpressions(self, enabled: bool = True):
        """
        :param enabled: whether Giskard should publish markers during planning
        :param in_planning_loop: whether Giskard should update the markers after every control step. Will slow down
                                    the system.
        """
        self.behavior_tree_config.plugin_config['PublishDebugExpressions']['enabled'] = enabled
        # self.behavior_tree_config.plugin_config['VisualizationBehavior']['in_planning_loop'] = in_planning_loop

    def configure_CollisionMarker(self, enabled: bool = True, in_planning_loop: bool = False):
        """
        :param enabled: whether Giskard should publish collision markers during planning
        :param in_planning_loop: whether Giskard should update the markers after every control step. Will slow down
                                    the system.
        """
        self.behavior_tree_config.plugin_config['CollisionMarker']['enabled'] = enabled
        self.behavior_tree_config.plugin_config['CollisionMarker']['in_planning_loop'] = in_planning_loop

    def configure_PlotTrajectory(self, enabled: bool = False, normalize_position: bool = False):
        self.behavior_tree_config.plugin_config['PlotTrajectory']['enabled'] = enabled
        self.behavior_tree_config.plugin_config['PlotTrajectory']['normalize_position'] = normalize_position

    def configure_PlotDebugExpressions(self, enabled: bool = False):
        self.behavior_tree_config.plugin_config['PlotDebugExpressions']['enabled'] = enabled

    def configure_DebugMarkerPublisher(self, enabled: bool = False):
        self.behavior_tree_config.plugin_config['PlotDebugTF']['enabled'] = enabled

    def register_controlled_joints(self, joint_names: List[str], group_name: Optional[str] = None):
        """
        Tell Giskard which joints can be controlled. Only used in standalone mode.
        :param joint_names:
        :param group_name: Only needs to be specified, if there are more than two robots.
        """
        if group_name is None:
            group_name = self.group_names[0]
        joint_names = [PrefixName(j, group_name) for j in joint_names]
        self._controlled_joints.extend(joint_names)

    def disable_visualization(self):
        """
        Don't publish any visualization marker.
        """
        self.configure_VisualizationBehavior(enabled=False, in_planning_loop=False)
        self.configure_CollisionMarker(enabled=False, in_planning_loop=False)

    def disable_tf_publishing(self):
        self.behavior_tree_config.plugin_config['TFPublisher']['enabled'] = False

    def publish_all_tf(self):
        self.behavior_tree_config.plugin_config['TFPublisher']['mode'] = TfPublishingModes.all

    def _add_joint(self, joint: Joint):
        joints = self._god_map.get_data(identifier.joints_to_add, default=[])
        joints.append(joint)

    def add_fixed_joint(self, parent_link: my_string, child_link: my_string,
                        homogenous_transform: Optional[NDArray] = None):
        """
        Add a fixed joint to Giskard's world. Can be used to connect a non-mobile robot to the world frame.
        :param parent_link:
        :param child_link:
        :param homogenous_transform: a 4x4 transformation matrix.
        """
        if homogenous_transform is None:
            homogenous_transform = np.eye(4)

        parent_link = PrefixName.from_string(parent_link, set_none_if_no_slash=True)
        child_link = PrefixName.from_string(child_link, set_none_if_no_slash=True)
        joint_name = PrefixName(f'{parent_link}_{child_link}_fixed_joint', None)
        joint = FixedJoint(name=joint_name,
                           parent_link_name=parent_link,
                           child_link_name=child_link,
                           parent_T_child=homogenous_transform)
        self._add_joint(joint)

    def add_sync_tf_frame(self, parent_link: str, child_link: str):
        """
        Tell Giskard to keep track of tf frames, e.g., for robot localization.
        :param parent_link:
        :param child_link:
        """
        if not tf.wait_for_transform(parent_link, child_link, rospy.Time(), rospy.Duration(1)):
            raise LookupException(f'Cannot get transform of {parent_link}<-{child_link}')
        self.add_fixed_joint(parent_link=parent_link, child_link=child_link)
        self.behavior_tree_config.add_sync_tf_frame(parent_link, child_link)

    def _add_odometry_topic(self, odometry_topic: str, joint_name: str):
        self.hardware_config.odometry_node_kwargs.append({'odometry_topic': odometry_topic,
                                                          'joint_name': joint_name})

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
        """
        Use this to connect a robot urdf of a mobile robot to the world if it has an omni-directional drive.
        :param parent_link_name:
        :param child_link_name:
        :param robot_group_name: set if there are multiple robots
        :param name: Name of the new link. Has to be unique and may be required in other functions.
        :param odometry_topic: where the odometry gets published
        :param translation_velocity_limit: in m/s
        :param rotation_velocity_limit: in rad/s
        :param translation_acceleration_limit: in m/s**2
        :param rotation_acceleration_limit: in rad/s**2
        :param translation_jerk_limit: in m/s**3
        :param rotation_jerk_limit: in rad/s**3
        :param odom_x_name: how the degree of freedom along the x-axis is called
        :param odom_y_name: how the degree of freedom along the y-axis is called
        :param odom_yaw_name: how the degree of freedom about the z-axis is called
        """
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
        self._add_joint(brumbrum_joint)
        if odometry_topic is not None:
            self._add_odometry_topic(odometry_topic=odometry_topic,
                                     joint_name=brumbrum_joint.name)

    def get_default_group_name(self):
        """
        Returns the name of the robot, only works if there is only one.
        """
        if len(self.group_names) > 1:
            raise AttributeError(f'group name has to be set if you have multiple robots')
        return self.group_names[0]

    def add_diff_drive_joint(self,
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
        """
        Same as add_omni_drive_joint, but for a differential drive.
        """
        if robot_group_name is None:
            robot_group_name = self.get_default_group_name()
        brumbrum_joint = DiffDrive(parent_link_name=parent_link_name,
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
        self._add_joint(brumbrum_joint)
        if odometry_topic is not None:
            self._add_odometry_topic(odometry_topic=odometry_topic,
                                     joint_name=brumbrum_joint.name)

    def set_maximum_derivative(self, new_value: Derivatives = Derivatives.jerk):
        """
        Setting this to e.g. jerk will enable jerk and acceleration constraints.
        """
        self._general_config.maximum_derivative = new_value

    def add_base_cmd_velocity(self,
                              cmd_vel_topic: str,
                              track_only_velocity: bool = False,
                              joint_name: Optional[my_string] = None):
        """
        Used if the robot's base can be controlled with a Twist topic.
        :param cmd_vel_topic:
        :param track_only_velocity: The tracking mode. If true, any position error is not considered which makes
                                    the tracking smoother but less accurate.
        :param joint_name: name of the omni or diff drive joint. Doesn't need to be specified if there is only one.
        """
        self.hardware_config.send_trajectory_to_cmd_vel_kwargs.append({'cmd_vel_topic': cmd_vel_topic,
                                                                       'track_only_velocity': track_only_velocity,
                                                                       'joint_name': joint_name})

    def _reset_config(self):
        for parameter, value in self._backup.items():
            setattr(self, parameter, deepcopy(value))

    def _create_parameter_backup(self):
        self._backup = {'_qp_solver_config': deepcopy(self._qp_solver_config),
                        '_general_config': deepcopy(self._general_config)}

    def _create_collision_checker(self, world):
        if self._collision_checker == CollisionCheckerLib.bpb:
            logging.loginfo('Using betterpybullet for collision checking.')
            try:
                from giskardpy.model.better_pybullet_syncer import BetterPyBulletSyncer
                return BetterPyBulletSyncer(world)
            except ImportError as e:
                logging.logerr(f'{e}; turning off collision avoidance.')
                self._collision_checker = CollisionCheckerLib.none
        if self._collision_checker == CollisionCheckerLib.pybullet:
            logging.loginfo('Using pybullet for collision checking.')
            try:
                from giskardpy.model.pybullet_syncer import PyBulletSyncer
                return PyBulletSyncer(world)
            except ImportError as e:
                logging.logerr(f'{e}; turning off collision avoidance.')
                self._collision_checker = CollisionCheckerLib.none
        if self._collision_checker == CollisionCheckerLib.none:
            logging.logwarn('Using no collision checking.')
            from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer
            return CollisionWorldSynchronizer(world)
        if self._collision_checker not in CollisionCheckerLib:
            raise KeyError(f'Unknown collision checker {self._collision_checker}. '
                           f'Collision avoidance is disabled')

    def grow(self):
        """
        Initialize the behavior tree and world. You usually don't need to call this.
        """
        self._qp_solver_check()
        if len(self.robot_interface_configs) == 0:
            self.add_robot_from_parameter_server()
        self._create_parameter_backup()
        if self._root_link_name is None:
            self._root_link_name = PrefixName(tf.get_tf_root(), None)
        world = WorldTree(self._root_link_name, self._god_map)
        world.delete_all_but_robots()
        world.register_controlled_joints(self._controlled_joints)

        collision_scene = self._create_collision_checker(world)
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

    def _qp_solver_check(self):
        try:
            if self._qp_solver_config.qp_solver == SupportedQPSolver.gurobi:
                from giskardpy.qp.qp_solver_gurobi import QPSolverGurobi
            elif self._qp_solver_config.qp_solver == SupportedQPSolver.cplex:
                from giskardpy.qp.qp_solver_cplex import QPSolverCplex
            elif self._qp_solver_config.qp_solver not in SupportedQPSolver:
                raise KeyError(f'Solver \'{self._qp_solver_config.qp_solver}\' not supported.')
        except Exception as e:
            logging.logwarn(e)
            logging.logwarn('Defaulting back to qpoases.')
            self._qp_solver_config.qp_solver = SupportedQPSolver.qp_oases

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
        """
        Start Giskard.
        """
        self.grow()
        self._god_map.get_data(identifier.tree_manager).live()

    def set_default_visualization_marker_color(self, r: float, g: float, b: float, a: float):
        """
        :param r: 0-1
        :param g: 0-1
        :param b: 0-1
        :param a: 0-1
        """
        self._general_config.default_link_color = ColorRGBA(r, g, b, a)

    def set_control_mode(self, mode: ControlModes):
        self._general_config.control_mode = mode

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
        config = self._get_collision_avoidance_config(group_name)
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
        config = self._get_collision_avoidance_config(group_name)
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
        config = self._get_collision_avoidance_config(group_name)
        link_name = PrefixName(link_name, group_name)
        config.ignored_self_collisions.append(link_name)

    def fix_joints_for_self_collision_avoidance(self, joint_names: List[str], group_name: Optional[str] = None):
        """
        Flag some joints as fixed for self collision avoidance. These joints will not be moved to avoid self
        collisions.
        """
        if group_name is None:
            group_name = self.get_default_group_name()
        config = self._get_collision_avoidance_config(group_name)
        joint_names = [PrefixName(joint_name, group_name) for joint_name in joint_names]
        config.fixed_joints_for_self_collision_avoidance.extend(joint_names)

    def fix_joints_for_external_collision_avoidance(self, joint_names: List[str], group_name: Optional[str] = None):
        """
        Flag some joints as fixed for external collision avoidance. These joints will not be moved to avoid
        external collisions.
        """
        if group_name is None:
            group_name = self.get_default_group_name()
        config = self._get_collision_avoidance_config(group_name)
        joint_names = [PrefixName(joint_name, group_name) for joint_name in joint_names]
        config.fixed_joints_for_external_collision_avoidance.extend(joint_names)

    def ignore_self_collisions_of_pair(self, link_name1: str, link_name2: str, group_name: Optional[str] = None):
        """
        Ignore a certain pair of links for self collision avoidance.
        """
        if group_name is None:
            group_name = self.get_default_group_name()
        config = self._get_collision_avoidance_config(group_name)
        link_name1 = PrefixName(link_name1, group_name)
        link_name2 = PrefixName(link_name2, group_name)
        config.ignored_self_collisions.append((link_name1, link_name2))

    def add_self_collision(self, link_name1: str, link_name2: str, group_name: Optional[str] = None):
        """
        Specifically add a link pair for self collision avoidance.
        """
        if group_name is None:
            group_name = self.get_default_group_name()
        config = self._get_collision_avoidance_config(group_name)
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

    def set_qp_solver(self, new_solver: SupportedQPSolver):
        self._qp_solver_config.qp_solver = new_solver

    def set_default_joint_limits(self,
                                 velocity_limit: float = 1,
                                 acceleration_limit: Optional[float] = 1e3,
                                 jerk_limit: Optional[float] = 30):
        """
        The default values will be set automatically, even if this function is not called.
        :param velocity_limit: in m/s or rad/s
        :param acceleration_limit: in m/s**2 or rad/s**2
        :param jerk_limit: in m/s**3 or rad/s**3
        """
        if jerk_limit is not None and acceleration_limit is None:
            raise AttributeError('If jerk limits are set, acceleration limits also have to be set/')
        self._general_config.joint_limits = {
            Derivatives.velocity: defaultdict(lambda: velocity_limit)
        }
        if acceleration_limit is not None:
            self._general_config.joint_limits[Derivatives.acceleration] = defaultdict(lambda: acceleration_limit)
        if jerk_limit is not None:
            self._general_config.joint_limits[Derivatives.jerk] = defaultdict(lambda: jerk_limit)

    def overwrite_joint_velocity_limits(self, joint_name, velocity_limit: float, group_name: Optional[str] = None):
        if group_name is None:
            group_name = self.get_default_group_name()
        joint_name = PrefixName(joint_name, group_name)
        self._general_config.joint_limits[Derivatives.velocity][joint_name] = velocity_limit

    def overwrite_joint_acceleration_limits(self, joint_name, acceleration_limit: float,
                                            group_name: Optional[str] = None):
        if group_name is None:
            group_name = self.get_default_group_name()
        joint_name = PrefixName(joint_name, group_name)
        self._general_config.joint_limits[Derivatives.acceleration][joint_name] = acceleration_limit

    def overwrite_joint_jerk_limits(self, joint_name, jerk_limit: float, group_name: Optional[str] = None):
        if group_name is None:
            group_name = self.get_default_group_name()
        joint_name = PrefixName(joint_name, group_name)
        self._general_config.joint_limits[Derivatives.jerk][joint_name] = jerk_limit

    def set_default_weights(self,
                            velocity_weight: float = 0.001,
                            acceleration_weight: Optional[float] = None,
                            jerk_weight: Optional[float] = 0.001):
        """
        The default values are set automatically, even if this function is not called.
        A typical goal has a weight of 1, so the values in here should be sufficiently below that.
        """
        self._qp_solver_config.joint_weights = {
            Derivatives.velocity: defaultdict(lambda: velocity_weight)
        }
        if jerk_weight is not None:
            self._qp_solver_config.joint_weights[Derivatives.acceleration] = defaultdict(lambda: acceleration_weight)
            self._qp_solver_config.joint_weights[Derivatives.jerk] = defaultdict(lambda: jerk_weight)
        elif acceleration_weight is not None:
            self._qp_solver_config.joint_weights[Derivatives.acceleration] = defaultdict(lambda: acceleration_weight)

    def overwrite_joint_velocity_weight(self,
                                        joint_name: str,
                                        velocity_weight: float,
                                        group_name: Optional[str] = None):
        if group_name is None:
            group_name = self.get_default_group_name()
        joint_name = PrefixName(joint_name, group_name)
        self._qp_solver_config.joint_weights[Derivatives.velocity][joint_name] = velocity_weight

    def overwrite_joint_acceleration_weight(self,
                                            joint_name: str,
                                            acceleration_weight: float,
                                            group_name: Optional[str] = None):
        if group_name is None:
            group_name = self.get_default_group_name()
        joint_name = PrefixName(joint_name, group_name)
        self._qp_solver_config.joint_weights[Derivatives.acceleration][joint_name] = acceleration_weight

    def overwrite_joint_jerk_weight(self,
                                    joint_name: str,
                                    jerk_weight: float,
                                    group_name: Optional[str] = None):
        if group_name is None:
            group_name = self.get_default_group_name()
        joint_name = PrefixName(joint_name, group_name)
        self._qp_solver_config.joint_weights[Derivatives.jerk][joint_name] = jerk_weight
