import inspect
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from typing import Dict, Optional, List, Union, Tuple

from py_trees import Blackboard

from giskardpy import identifier
from giskardpy.configs.data_types import SupportedQPSolver, CollisionCheckerLib
from giskardpy.configs.drives import DriveInterface, OmniDriveCmdVelInterface, DiffDriveCmdVelInterface
from giskardpy.configs.follow_joint_trajectory import FollowJointTrajectoryInterface
from giskardpy.exceptions import GiskardException
from giskardpy.god_map import GodMap
from giskardpy.model.world import WorldTree
from giskardpy.tree.garden import OpenLoop, ClosedLoop
from giskardpy.utils import logging
from giskardpy.utils.time_collector import TimeCollector
from giskardpy.utils.utils import resolve_ros_iris


class ControlModes(Enum):
    open_loop = 1
    close_loop = 2


class GeneralConfig:
    control_mode: ControlModes = ControlModes.open_loop
    action_server_name: str = '~command'
    path_to_data_folder: str = resolve_ros_iris('package://giskardpy/data/')
    test_mode: bool = False
    debug: bool = False
    joint_limits: Dict[int, Dict[str, float]] = {
        'velocity': defaultdict(lambda: 1),
        'acceleration': defaultdict(lambda: 1e3),
        'jerk': defaultdict(lambda: 30)
    }


class QPSolverConfig:
    def __init__(self,
                 qp_solver: SupportedQPSolver = SupportedQPSolver.qp_oases,
                 prediction_horizon: int = 9,
                 retries_with_relaxed_constraints: int = 5,
                 added_slack: float = 100,
                 sample_period: float = 0.05,
                 weight_factor: float = 100,
                 joint_weights: Optional[Dict[int, Dict[str, float]]] = None):
        self.qp_solver = qp_solver
        self.prediction_horizon = prediction_horizon
        self.retries_with_relaxed_constraints = retries_with_relaxed_constraints
        self.added_slack = added_slack
        self.sample_period = sample_period
        self.weight_factor = weight_factor
        if joint_weights is None:
            self.joint_weights = {
                'velocity': defaultdict(lambda: 0.001),
                'acceleration': defaultdict(float),
                'jerk': defaultdict(lambda: 0.001)
            }
        else:
            self.joint_weights = joint_weights


class CollisionAvoidanceConfig:
    class CollisionAvoidanceEntry:
        def __init__(self,
                     number_of_repeller: int = 1,
                     soft_threshold: float = 0.05,
                     hard_threshold: float = 0.0,
                     max_velocity: float = 0.2):
            self.number_of_repeller = number_of_repeller
            self.soft_threshold = soft_threshold
            self.hard_threshold = hard_threshold
            self.max_velocity = max_velocity

        @classmethod
        def init_50mm(cls):
            return cls(soft_threshold=0.05, hard_threshold=0.0)

        @classmethod
        def init_100mm(cls):
            return cls(soft_threshold=0.1, hard_threshold=0.0)

        @classmethod
        def init_25mm(cls):
            return cls(soft_threshold=0.025, hard_threshold=0.0)

    collision_checker: CollisionCheckerLib = CollisionCheckerLib.bpb

    _add_self_collisions: List[Tuple[str, str]] = []
    _ignored_self_collisions: List[Union[str, Tuple[str, str]]] = []

    _external_collision_avoidance: Dict[str, CollisionAvoidanceEntry] = defaultdict(CollisionAvoidanceEntry)
    _self_collision_avoidance: Dict[str, CollisionAvoidanceEntry] = defaultdict(CollisionAvoidanceEntry)

    def ignore_all_self_collisions_of_link(self, link_name):
        self._ignored_self_collisions.append(link_name)

    def ignore_self_collisions_of_pair(self, link_name1, link_name2):
        self._ignored_self_collisions.append((link_name1, link_name2))

    def add_self_collision(self, link_name1, link_name2):
        self._add_self_collisions.append((link_name1, link_name2))

    def set_default_external_collision_avoidance(self,
                                                 number_of_repeller: int = 1,
                                                 soft_threshold: float = 0.05,
                                                 hard_threshold: float = 0.0,
                                                 max_velocity: float = 0.2):
        self._external_collision_avoidance.default_factory = lambda: self.CollisionAvoidanceEntry(
            number_of_repeller=number_of_repeller,
            soft_threshold=soft_threshold,
            hard_threshold=hard_threshold,
            max_velocity=max_velocity
        )

    def overwrite_external_collision_avoidance(self,
                                               joint_name: str,
                                               number_of_repeller: Optional[int] = None,
                                               soft_threshold: Optional[float] = None,
                                               hard_threshold: Optional[float] = None,
                                               max_velocity: Optional[float] = None):
        if number_of_repeller is not None:
            self._external_collision_avoidance[joint_name].number_of_repeller = number_of_repeller
        if soft_threshold is not None:
            self._external_collision_avoidance[joint_name].soft_threshold = soft_threshold
        if hard_threshold is not None:
            self._external_collision_avoidance[joint_name].hard_threshold = hard_threshold
        if max_velocity is not None:
            self._external_collision_avoidance[joint_name].max_velocity = max_velocity

    def set_default_self_collision_avoidance(self,
                                             number_of_repeller: int = 1,
                                             soft_threshold: float = 0.05,
                                             hard_threshold: float = 0.0,
                                             max_velocity: float = 0.2):
        self._self_collision_avoidance.default_factory = lambda: self.CollisionAvoidanceEntry(
            number_of_repeller=number_of_repeller,
            soft_threshold=soft_threshold,
            hard_threshold=hard_threshold,
            max_velocity=max_velocity
        )

    def overwrite_self_collision_avoidance(self,
                                           link_name: str,
                                           number_of_repeller: Optional[int] = None,
                                           soft_threshold: Optional[float] = None,
                                           hard_threshold: Optional[float] = None,
                                           max_velocity: Optional[float] = None):
        if number_of_repeller is not None:
            self._self_collision_avoidance[link_name].number_of_repeller = number_of_repeller
        if soft_threshold is not None:
            self._self_collision_avoidance[link_name].soft_threshold = soft_threshold
        if hard_threshold is not None:
            self._self_collision_avoidance[link_name].hard_threshold = hard_threshold
        if max_velocity is not None:
            self._self_collision_avoidance[link_name].max_velocity = max_velocity


class BehaviorTreeConfig:
    tree_tick_rate: float = 0.1

    plugin_config = {
        'GoalReached': {
            'joint_convergence_threshold': 0.01,
            'window_size': 21
        },
        'VisualizationBehavior': {
            'enabled': True,
            'in_planning_loop': False
        },
        'PublishDebugExpressions': {
            'enabled': False,
            'expression_filter': None
        },
        'CollisionMarker': {
            'enabled': True,
            'in_planning_loop': False
        },
        'PlotTrajectory': {
            'enabled': True,
            'history': 5,
            'velocity_threshold': 0.0,
            'cm_per_second': 2.5,
            'height_per_derivative': 6,
            'normalize_position': True,
            'order': 4,
            'tick_stride': 0.5,
        },
        'PlotDebugExpressions': {
            'enabled': True,
            'history': 5,
            'cm_per_second': 2.5,
            'height_per_derivative': 6,
            'order': 2,
            'tick_stride': 0.5,
        },
        'WiggleCancel': {
            'amplitude_threshold': 0.55,
            'window_size': 21,
            'frequency_range': 0.4,
        },
        'TFPublisher': {
            'publish_attached_objects': True,
            'publish_world_objects': False,
            'tf_topic': '/tf',
        },
        'MaxTrajectoryLength': {
            'enabled': True,
            'length': 30
        },
        'LoopDetector': {
            'precision': 3
        },
        'SyncTfFrames': {
            'frames': [],
        },
        'PlotDebugTF': {
            'enabled': False,
        },
    }

    def set_goal_reached_parameters(self, joint_convergence_threshold=0.01, window_size=21):
        self.plugin_config['GoalReached'] = {
            'joint_convergence_threshold': joint_convergence_threshold,
            'window_size': window_size
        }

    def add_sync_tf_frame(self, parent_link, child_link, add_after_robot=False):
        # TODO make data structure
        self.plugin_config['SyncTfFrames']['frames'].append([parent_link, child_link, add_after_robot])

    def set_odometry_topic(self, odometry_topic):
        self.plugin_config['SyncOdometry'] = {
            'odometry_topic': odometry_topic
        }


class RobotInterfaceConfig:
    joint_state_topic = '/joint_states'
    drive_interface: Optional[DriveInterface] = None
    follow_joint_trajectory_interfaces: List[FollowJointTrajectoryInterface] = []

    def add_follow_joint_trajectory_server(self, namespace, state_topic):
        self.follow_joint_trajectory_interfaces.append(FollowJointTrajectoryInterface(
            namespace=namespace,
            state_topic=state_topic))

    def add_omni_drive_interface(self, cmd_vel_topic, parent_link_name, child_link_name):
        self.drive_interface = OmniDriveCmdVelInterface(cmd_vel_topic=cmd_vel_topic,
                                                        parent_link_name=parent_link_name,
                                                        child_link_name=child_link_name)

    def add_diff_drive_interface(self, cmd_vel_topic, parent_link_name, child_link_name):
        self.drive_interface = DiffDriveCmdVelInterface(cmd_vel_topic=cmd_vel_topic,
                                                        parent_link_name=parent_link_name,
                                                        child_link_name=child_link_name)


class Giskard:
    general_config: GeneralConfig = GeneralConfig()
    behavior_tree_config: BehaviorTreeConfig = BehaviorTreeConfig()
    qp_solver_config: QPSolverConfig = QPSolverConfig()
    collision_avoidance_config: CollisionAvoidanceConfig = CollisionAvoidanceConfig()
    robot_interface_config: RobotInterfaceConfig = RobotInterfaceConfig()

    def __init__(self):
        self._god_map = GodMap.init_from_paramserver()
        self._god_map.set_data(identifier.giskard, self)
        self._god_map.set_data(identifier.timer_collector, TimeCollector(self._god_map))
        blackboard = Blackboard
        blackboard.god_map = self._god_map
        self._backup = {}

    def set_joint_states_topic(self, topic_name: str):
        self.robot_interface_config.joint_state_topic = topic_name

    def add_sync_tf_frame(self, parent_link, child_link, add_after_robot=False):
        self.behavior_tree_config.add_sync_tf_frame(parent_link, child_link, add_after_robot)

    def set_odometry_topic(self, odometry_topic):
        self.behavior_tree_config.set_odometry_topic(odometry_topic)

    def add_follow_joint_trajectory_server(self, namespace, state_topic):
        self.robot_interface_config.add_follow_joint_trajectory_server(namespace, state_topic)

    def add_omni_drive_interface(self, cmd_vel_topic, parent_link_name, child_link_name):
        self.robot_interface_config.add_omni_drive_interface(cmd_vel_topic=cmd_vel_topic,
                                                             parent_link_name=parent_link_name,
                                                             child_link_name=child_link_name)

    def add_diff_drive_interface(self, cmd_vel_topic, parent_link_name, child_link_name):
        self.robot_interface_config.add_diff_drive_interface(cmd_vel_topic=cmd_vel_topic,
                                                             parent_link_name=parent_link_name,
                                                             child_link_name=child_link_name)

    def reset_config(self):
        for parameter, value in self._backup.items():
            setattr(self, parameter, deepcopy(value))

    def _create_parameter_backup(self):
        self._backup = {'qp_solver_config': deepcopy(self.qp_solver_config),
                        'general_config': deepcopy(self.general_config)}

    def grow(self):
        self._create_parameter_backup()
        world = WorldTree(self._god_map)
        world.delete_all_but_robot()

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
        if self.robot_interface_config.drive_interface is None:
            logging.loginfo('No cmd_vel topic has been registered.')

    def live(self):
        self.grow()
        self._god_map.get_data(identifier.tree_manager).live()
