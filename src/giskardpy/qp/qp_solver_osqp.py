
import numpy as np
from scipy import sparse

from giskardpy.qp.qp_solver import QPSolver
import osqp


class QPSolverOSQP(QPSolver):
    """
    min_x 0.5 x^T P x + q^T x
    s.t.  l <= Ax = u
    """
    settings = {
        'verbose': False,
    }


    @profile
    def solve(self, weights: np.ndarray, g: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray, lbA: np.ndarray,
              ubA: np.ndarray) -> np.ndarray:
        m = osqp.OSQP()
        A_b = np.eye(lb.shape[0])
        A = sparse.csc_matrix(np.vstack([A_b, A]))
        P = sparse.csc_matrix(np.diag(weights))
        l = np.concatenate([lb, lbA])
        u = np.concatenate([ub, ubA])
        m.setup(P=P, q=g, A=A, l=l, u=u, **self.settings)
        return m.solve().x

    # @profile
    def solve_and_retry(self, weights, g, A, lb, ub, lbA, ubA):
        return self.solve(weights, g, A, lb, ub, lbA, ubA)
