from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import scipy.sparse as sp

import numpy as np
import scs

from giskardpy.qp.qp_solver import QPSolver


class QPSolverSuperSCS(QPSolver):
    """
    min_x 0.5 x^T P x + q^T x
    s.t.  Ax = b
          Gx <= h
          lb <= x <= ub
    """

    def solve(self, weights: np.ndarray, g: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray, lbA: np.ndarray,
              ubA: np.ndarray) -> np.ndarray:
        import scipy.sparse as sp
        A_b = np.eye(lb.shape[0])
        G = sp.csc_matrix(np.vstack([-A_b, A_b, -A, A]))
        h = np.concatenate([-lb, ub, -lbA, ubA])
        P = sp.csc_matrix(np.diag(weights))
        q = g

        data = {
            'P': P,
            'A': G,
            'b': h,
            'c': q
        }

        cone = {
            'l': h.shape[0],
        }

        solver = scs.SCS(data, cone, verbose=False)

        return solver.solve()['x']

    #
    def solve_and_retry(self, weights, g, A, lb, ub, lbA, ubA):
        return self.solve(weights, g, A, lb, ub, lbA, ubA)
