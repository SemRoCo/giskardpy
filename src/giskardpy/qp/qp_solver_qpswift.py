from enum import IntEnum

import numpy as np
from giskardpy.exceptions import QPSolverException, InfeasibleException, HardConstraintsViolatedException
from giskardpy.qp.qp_solver import QPSolver
from giskardpy.utils import logging
import qpSWIFT

from giskardpy.utils.utils import record_time


class QPSolverQPSwift(QPSolver):
    """
    min_x 0.5 x^T P x + c^T x
    s.t.  Ax = b
          Gx <= h
    """

    opts = {}

    @profile
    @record_time
    def solve(self, weights: np.ndarray, g: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray, lbA: np.ndarray,
              ubA: np.ndarray) -> np.ndarray:
        A_b = np.eye(lb.shape[0])
        G = np.vstack([-A_b, A_b, -A, A])
        P = np.diag(weights)
        h = np.concatenate([-lb, ub, -lbA, ubA])
        return qpSWIFT.run(c=g, h=h, P=P, G=G, opts=self.opts)['sol']

    # @profile
    def solve_and_retry(self, weights, g, A, lb, ub, lbA, ubA):
        return self.solve(weights, g, A, lb, ub, lbA, ubA)
