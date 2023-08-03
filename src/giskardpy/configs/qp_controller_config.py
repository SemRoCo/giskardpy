from __future__ import annotations

from collections import defaultdict
from enum import IntEnum
from typing import Optional

from giskardpy.my_types import Derivatives


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


class QPControllerConfig:
    qp_solver: SupportedQPSolver
    prediction_horizon: int = 9
    sample_period: float = 0.05
    max_derivative: Derivatives = Derivatives.jerk
    max_trajectory_length: float = 30
    retries_with_relaxed_constraints: int = 5,
    added_slack: float = 100,
    weight_factor: float = 100

    def __init__(self,
                 qp_solver: Optional[SupportedQPSolver] = None,
                 prediction_horizon: int = 9,
                 sample_period: float = 0.05,
                 max_trajectory_length: float = 30,
                 retries_with_relaxed_constraints: int = 5,
                 added_slack: float = 100,
                 weight_factor: float = 100,
                 endless_mode: bool = False):
        self.__qp_solver = qp_solver
        if prediction_horizon < 7:
            raise ValueError('prediction horizon must be >= 7.')
        self.prediction_horizon = prediction_horizon
        self.__prediction_horizon = prediction_horizon
        self.__sample_period = sample_period
        self.__max_trajectory_length = max_trajectory_length
        self.__retries_with_relaxed_constraints = retries_with_relaxed_constraints
        self.__added_slack = added_slack
        self.__weight_factor = weight_factor
        self.__endless_mode = endless_mode
        self.set_defaults()

    def set_defaults(self):
        self.qp_solver = self.__qp_solver
        self.prediction_horizon = self.__prediction_horizon
        self.sample_period = self.__sample_period
        self.retries_with_relaxed_constraints = self.__retries_with_relaxed_constraints
        self.added_slack = self.__added_slack
        self.weight_factor = self.__weight_factor
        self.endless_mode = self.__endless_mode
        self.max_trajectory_length = self.__max_trajectory_length

    def set_qp_solver(self, new_solver: SupportedQPSolver):
        self.__qp_solver = new_solver
        self.qp_solver = new_solver
