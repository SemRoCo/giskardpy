import cvxopt
import numpy as np
from scipy import sparse

from giskardpy.qp.qp_solver import QPSolver


class QPSolverCVXOPT(QPSolver):
    """
    min_x 0.5 x^T P x + q^T x
    s.t.  Ax = b
          Gx <= h
    """


    @profile
    def solve(self, weights: np.ndarray, g: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray, lbA: np.ndarray,
              ubA: np.ndarray) -> np.ndarray:
        A_b = np.eye(lb.shape[0])
        G = np.vstack([-A_b, A_b, -A, A])
        P = np.diag(weights)
        h = np.concatenate([-lb, ub, -lbA, ubA])
        return cvxopt.solvers.qp(P=P, q=g, G=G, h=h)

    # @profile
    def solve_and_retry(self, weights, g, A, lb, ub, lbA, ubA):
        return self.solve(weights, g, A, lb, ub, lbA, ubA)
