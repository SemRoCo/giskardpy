from collections import defaultdict
from enum import Enum, IntEnum
from typing import List, Tuple, Dict, Union
from giskardpy.my_types import PrefixName


class TfPublishingModes(Enum):
    nothing = 0
    all = 1
    attached_objects = 2

    world_objects = 4
    attached_and_world_objects = 6


class CollisionCheckerLib(Enum):
    none = -1
    bpb = 1


class SupportedQPSolver(IntEnum):
    qpSWIFT = 1
    qpalm = 2
    gurobi = 3
    # clarabel = 4
    # qpOASES = 5
    # osqp = 6
    # quadprog = 7
    # cplex = 3
    # cvxopt = 7
    # qp_solvers = 8
    # mosek = 9
    # scs = 11
    # casadi = 12
    # super_csc = 14
    # cvxpy = 15


class ControlModes(Enum):
    open_loop = 1
    close_loop = 2
    stand_alone = 3


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


class CollisionAvoidanceGroupConfig:
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
