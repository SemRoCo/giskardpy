from enum import Enum


class CollisionChecker(Enum):
    bpb = 1
    pybullet = 2
    none = 3


class SupportedQPSolver(Enum):
    gurobi = 1
    qp_oases = 2
    cplex = 3
