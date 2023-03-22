
import numpy as np
from qpsolvers import solve_qp

from giskardpy.qp.qp_solver import QPSolver


class QPSolverQPSolvers(QPSolver):
    """
    min_x 0.5 x^T P x + q^T x
    s.t.  Ax = b
          Gx <= h
          lb <= x <= ub
    """

    opts = {}

    @profile
    def solve(self, weights: np.ndarray, g: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray, lbA: np.ndarray,
              ubA: np.ndarray) -> np.ndarray:
        # A_b = np.eye(lb.shape[0])
        G = np.vstack([-A, A])
        P = np.diag(weights)
        h = np.concatenate([-lbA, ubA])
        return solve_qp(P=P, q=g, G=G, h=h, lb=lb, ub=ub, solver='clarabel')

    # @profile
    def solve_and_retry(self, weights, g, A, lb, ub, lbA, ubA):
        return self.solve(weights, g, A, lb, ub, lbA, ubA)
