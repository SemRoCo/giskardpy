from enum import Enum


class CollisionCheckerLib(Enum):
    bpb = 1
    pybullet = 2
    none = 3


class SupportedQPSolver(Enum):
    gurobi = 1
    qp_oases = 2
    cplex = 3


class FrameToAddToWorld:
    def __init__(self, parent_link, child_link, transform, add_after_robot):
        self.child_link = child_link
        self.parent_link = parent_link
        self.transform = transform
        self.add_after_robot = add_after_robot