from collections import defaultdict
from enum import Enum, IntEnum
from typing import Optional, List, Tuple, Dict, Union

from std_msgs.msg import ColorRGBA

from giskardpy.data_types import Derivatives
from giskardpy.model.utils import robot_name_from_urdf_string
from giskardpy.my_types import PrefixName
from giskardpy.utils.utils import resolve_ros_iris


class TfPublishingModes(Enum):
    nothing = 0
    all = 1
    attached_objects = 2

    world_objects = 4
    attached_and_world_objects = 6


class CollisionCheckerLib(Enum):
    bpb = 1
    pybullet = 2
    none = 3


class SupportedQPSolver(IntEnum):
    qpSWIFT = 1
    qpalm = 2
    gurobi = 3
    clarabel = 4
    qpOASES = 5
    osqp = 6
    quadprog = 7
    # cplex = 3
    # cvxopt = 7
    qp_solvers = 8
    # mosek = 9
    # scs = 11
    # casadi = 12
    # super_csc = 14
    # cvxpy = 15


class ControlModes(Enum):
    open_loop = 1
    close_loop = 2
    stand_alone = 3


class GeneralConfig:
    joint_limits: Dict[Derivatives, Dict[PrefixName, float]]

    def __init__(self):
        self.control_mode: ControlModes = ControlModes.open_loop
        self.maximum_derivative: Derivatives = Derivatives.jerk
        self.action_server_name: str = '~command'
        self.path_to_data_folder: str = resolve_ros_iris('package://giskardpy/tmp/')
        self.test_mode: bool = False
        self.debug: bool = False
        self.joint_limits = {d: defaultdict(lambda: 1e4) for d in Derivatives}
        self.default_link_color = ColorRGBA(1, 1, 1, 0.5)


class QPSolverConfig:
    joint_weights: Dict[Derivatives, Dict[PrefixName, float]]

    def __init__(self,
                 qp_solver: SupportedQPSolver = None,
                 prediction_horizon: int = 9,
                 retries_with_relaxed_constraints: int = 5,
                 added_slack: float = 100,
                 sample_period: float = 0.05,
                 weight_factor: float = 100):
        self.qp_solver = qp_solver
        self.prediction_horizon = prediction_horizon
        self.retries_with_relaxed_constraints = retries_with_relaxed_constraints
        self.added_slack = added_slack
        self.sample_period = sample_period
        self.weight_factor = weight_factor
        self.joint_weights = {d: defaultdict(float) for d in Derivatives}


class HardwareConfig:
    def __init__(self):
        self.send_trajectory_to_cmd_vel_kwargs: List[dict] = []
        self.follow_joint_trajectory_interfaces_kwargs: List[dict] = []
        self.joint_state_topics_kwargs: List[dict] = []
        self.odometry_node_kwargs: List[dict] = []


class CollisionAvoidanceConfigEntry:
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


class CollisionAvoidanceConfig:
    def __init__(self):
        self.add_self_collisions: List[Tuple[PrefixName, PrefixName]] = []
        self.ignored_self_collisions: List[Union[PrefixName, Tuple[PrefixName, PrefixName]]] = []
        self.ignored_collisions: List[PrefixName] = []
        self.fixed_joints_for_self_collision_avoidance = []
        self.fixed_joints_for_external_collision_avoidance = []

        self.external_collision_avoidance: Dict[PrefixName, CollisionAvoidanceConfigEntry] = defaultdict(
            CollisionAvoidanceConfigEntry)
        self.self_collision_avoidance: Dict[PrefixName, CollisionAvoidanceConfigEntry] = defaultdict(
            CollisionAvoidanceConfigEntry)

    def cal_max_param(self, parameter_name):
        external_distances = self.external_collision_avoidance
        self_distances = self.self_collision_avoidance
        default_distance = max(getattr(external_distances.default_factory(), parameter_name),
                               getattr(self_distances.default_factory(), parameter_name))
        for value in external_distances.values():
            default_distance = max(default_distance, getattr(value, parameter_name))
        for value in self_distances.values():
            default_distance = max(default_distance, getattr(value, parameter_name))
        return default_distance


class BehaviorTreeConfig:
    tree_tick_rate: float = 0.05

    plugin_config = {
        'GoalReached': {
            'joint_convergence_threshold': 0.01,
            'window_size': 21
        },
        'VisualizationBehavior': {
            'enabled': True,
            'in_planning_loop': False
        },
        'CollisionMarker': {
            'enabled': True,
            'in_planning_loop': False
        },
        'PublishDebugExpressions': {
            'enabled': False,
            'enabled_base': False,
            'expression_filter': None
        },
        'PlotTrajectory': {
            'enabled': False,
            'history': 5,
            'cm_per_second': 2.5,
            'height_per_derivative': 6,
            'normalize_position': False,
            # 'order': 5,
            'tick_stride': 0.5,
            'wait': False,
            # 'y_limits': [-0.5, 0.5]
            # 'y_limits': [-5, 5]
        },
        'PlotDebugExpressions': {
            'enabled': False,
            'history': 5,
            'cm_per_second': 2.5,
            'height_per_derivative': 6,
            # 'order': 2,
            'wait': False,
            'tick_stride': 0.5,
            # 'y_limits': [-0.5, 0.5]
        },
        'WiggleCancel': {
            'amplitude_threshold': 0.55,
            'window_size': 21,
            'frequency_range': 0.4,
        },
        'TFPublisher': {
            'enabled': True,
            'mode': TfPublishingModes.attached_objects,
            'tf_topic': '/tf',
        },
        'MaxTrajectoryLength': {
            'enabled': True,
            'length': 60  # seconds
        },
        'LoopDetector': {
            'precision': 4
        },
        'SyncTfFrames': {
            'joint_names': [],
        },
        'PlotDebugTF': {
            'enabled': False,
            'enabled_base': False,
        },
    }

    def set_goal_reached_parameters(self, joint_convergence_threshold=0.01, window_size=21):
        self.plugin_config['GoalReached'] = {
            'joint_convergence_threshold': joint_convergence_threshold,
            'window_size': window_size
        }

    def add_sync_tf_frame(self, joint_name):
        # TODO make data structure
        self.plugin_config['SyncTfFrames']['joint_names'].append(joint_name)


class RobotInterfaceConfig:
    def __init__(self, urdf: str, name: Optional[str] = None, add_drive_joint_to_group: bool = True):
        if name is None:
            name = robot_name_from_urdf_string(urdf)
        self.urdf = urdf
        self.name = name
        self.add_drive_joint_to_group = add_drive_joint_to_group
