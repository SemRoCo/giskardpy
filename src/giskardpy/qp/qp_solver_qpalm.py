import numpy as np
import qpalm
from scipy import sparse

from giskardpy.configs.data_types import SupportedQPSolver
from giskardpy.qp.qp_solver import QPSolver

from giskardpy.utils.utils import record_time


class QPSolverQPalm(QPSolver):
    solver_id = SupportedQPSolver.qpalm
    """
    min_x 0.5 x^T Q x + q^T x
    s.t.  lb <= Ax <= ub
    https://github.com/kul-optec/QPALM
    """

    opts = {}
    settings = qpalm.Settings()
    settings.verbose = False
    settings.eps_abs = 1e-8

    @profile
    @record_time
    def solve(self, weights: np.ndarray, g: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray, lbA: np.ndarray,
              ubA: np.ndarray) -> np.ndarray:
        A_b = np.eye(lb.shape[0])
        Q = sparse.csc_matrix(np.diag(weights))
        q = g
        A = sparse.csc_matrix(np.vstack([A_b, A]))
        lb = np.concatenate([lb, lbA])
        ub = np.concatenate([ub, ubA])

        data = qpalm.Data(A.shape[1], A.shape[0])

        data.Q = Q
        data.A = A
        data.q = q
        data.bmax = ub
        data.bmin = lb

        solver = qpalm.Solver(data, self.settings)
        solver.solve()
        return solver.solution.x

    # @profile
    def solve_and_retry(self, weights, g, A, lb, ub, lbA, ubA):
        return self.solve(weights, g, A, lb, ub, lbA, ubA)
