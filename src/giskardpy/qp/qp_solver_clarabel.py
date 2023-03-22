import numpy as np
from clarabel import clarabel
from scipy import sparse

from giskardpy.configs.data_types import SupportedQPSolver
from giskardpy.qp.qp_solver import QPSolver
from giskardpy.utils.utils import record_time


class QPSolverClarabel(QPSolver):
    solver_id = SupportedQPSolver.clarabel
    """
    min_x 0.5 x^T P x + q^T x
    s.t.  Ax = b
          Gx <= h
          lb <= x <= ub
    """

    @profile
    @record_time
    def solve(self, weights: np.ndarray, g: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray, lbA: np.ndarray,
              ubA: np.ndarray) -> np.ndarray:
        A_b = np.eye(lb.shape[0])
        G = sparse.csc_matrix(np.vstack([-A_b, A_b, -A, A]))
        h = np.concatenate([-lb, ub, -lbA, ubA])
        P = sparse.csc_matrix(np.diag(weights))
        q = g

        cones = [clarabel.NonnegativeConeT(h.shape[0])]

        settings = clarabel.DefaultSettings()
        settings.verbose = False
        solver = clarabel.DefaultSolver(P, q, G, h, cones, settings)
        result = np.array(solver.solve().x)

        return result

    # @profile
    def solve_and_retry(self, weights, g, A, lb, ub, lbA, ubA):
        return self.solve(weights, g, A, lb, ub, lbA, ubA)
