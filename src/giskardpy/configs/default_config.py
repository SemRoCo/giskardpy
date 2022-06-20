from collections import defaultdict
from enum import Enum
from typing import Dict, Optional, List

from py_trees import Blackboard

from giskardpy import identifier
from giskardpy.configs.data_types import SupportedQPSolver, CollisionCheckerLib
from giskardpy.configs.drives import DriveInterface
from giskardpy.configs.follow_joint_trajectory import FollowJointTrajectoryInterface
from giskardpy.god_map import GodMap
from giskardpy.model.world import WorldTree
from giskardpy.tree.garden import OpenLoop, ClosedLoop
from giskardpy.utils import logging
from giskardpy.utils.time_collector import TimeCollector
from giskardpy.utils.utils import resolve_ros_iris


class QPSolverConfig:
    def __init__(self,
                 qp_solver: SupportedQPSolver = SupportedQPSolver.gurobi,
                 prediction_horizon: int = 9,
                 retries_with_relaxed_constraints: int = 5,
                 added_slack: float = 100,
                 weight_factor: float = 100):
        self.qp_solver = qp_solver
        self.prediction_horizon = prediction_horizon
        self.retries_with_relaxed_constraints = retries_with_relaxed_constraints
        self.added_slack = added_slack
        self.weight_factor = weight_factor


class ControlModes(Enum):
    open_loop = 1
    close_loop = 2


class CollisionAvoidanceConfig:
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


class GiskardConfig:
    control_mode: ControlModes = ControlModes.open_loop
    action_server_name: str = '~command'
    path_to_data_folder: str = resolve_ros_iris('package://giskardpy/data/')
    enable_gui: bool = False
    sample_period: float = 0.05
    map_frame: str = 'map'
    test_mode: bool = False
    debug: bool = False
    tree_tick_rate: float = 0.1
    collision_checker: CollisionCheckerLib = CollisionCheckerLib.bpb
    external_collision_avoidance: Dict[str, CollisionAvoidanceConfig] = defaultdict(CollisionAvoidanceConfig)
    self_collision_avoidance: Dict[str, CollisionAvoidanceConfig] = defaultdict(CollisionAvoidanceConfig)
    add_self_collisions: List[str] = []
    ignored_self_collisions: List[str] = []
    prediction_horizon: int = 9
    qp_solver_config: QPSolverConfig = QPSolverConfig()
    joint_weights: Dict[int, Dict[str, float]] = {
        'velocity': defaultdict(lambda: 0.001),
        'acceleration': defaultdict(float),
        'jerk': defaultdict(lambda: 0.001)
    }
    joint_limits: Dict[int, Dict[str, float]] = {
        'velocity': defaultdict(lambda: 1),
        'acceleration': defaultdict(lambda: 1e3),
        'jerk': defaultdict(lambda: 30)
    }

    drive_interface: Optional[DriveInterface] = None
    follow_joint_trajectory_interfaces: List[FollowJointTrajectoryInterface] = []

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
            'joint_filter': ['base_footprint_x_vel',
                             'base_footprint_y_vel',
                             'base_footprint_rot_vel',
                             'odom_x',
                             'odom_y',
                             'odom_rot',
                             'r_shoulder_lift_joint'],
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
        }
    }

    def __init__(self):
        self.god_map = GodMap.init_from_paramserver()
        self.god_map.set_data(identifier.giskard, self)
        self.god_map.set_data(identifier.timer_collector, TimeCollector(self.god_map))
        blackboard = Blackboard
        blackboard.god_map = self.god_map

    def reset_config(self):
        #FIXME
        self.prediction_horizon = 9

    def grow(self):
        world = WorldTree(self.god_map)
        world.delete_all_but_robot()

        if self.collision_checker == CollisionCheckerLib.bpb:
            logging.loginfo('Using bpb for collision checking.')
            from giskardpy.model.better_pybullet_syncer import BetterPyBulletSyncer
            collision_scene = BetterPyBulletSyncer(world)
        elif self.collision_checker == CollisionCheckerLib.pybullet:
            logging.loginfo('Using pybullet for collision checking.')
            from giskardpy.model.pybullet_syncer import PyBulletSyncer
            collision_scene = PyBulletSyncer(world)
        elif self.collision_checker == CollisionCheckerLib.none:
            logging.logwarn('Using no collision checking.')
            from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer
            collision_scene = CollisionWorldSynchronizer(world)
        else:
            raise KeyError(f'Unknown collision checker {self.collision_checker}. Collision avoidance is disabled')
        self.god_map.set_data(identifier.collision_checker, self.collision_checker)
        self.god_map.set_data(identifier.collision_scene, collision_scene)
        if self.control_mode == ControlModes.open_loop:
            self.tree = OpenLoop(self.god_map)
        elif self.control_mode == ControlModes.close_loop:
            self.tree = ClosedLoop(self.god_map)
        else:
            raise KeyError(f'Robot interface mode \'{self.control_mode}\' is not supported.')

    def live(self):
        self.grow()
        self.god_map.get_data(identifier.tree_manager).live()
