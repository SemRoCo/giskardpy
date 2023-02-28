import numpy as np
import quadprog

from giskardpy.exceptions import InfeasibleException
from giskardpy.qp.qp_solver import QPSolver


class QPSolverQuadprog(QPSolver):
    """
    min_x 0.5 x^T G x + a^T x
    s.t.  Cx <= h
    """

    def solve(self, weights: np.ndarray, g: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray, lbA: np.ndarray,
              ubA: np.ndarray) -> np.ndarray:
        A_b = np.eye(lb.shape[0])
        C = np.vstack([-A_b, A_b, -A, A]).T
        G = np.diag(weights)
        b = np.concatenate([-lb, ub, -lbA, ubA])
        return quadprog.solve_qp(G=G, a=g, C=C, b=b)[0]

    def solve_and_retry(self, weights: np.ndarray, g: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray,
                        lbA: np.ndarray, ubA: np.ndarray) -> np.ndarray:
        return self.solve(weights, g, A, lb, ub, lbA, ubA)
