import sys
from copy import deepcopy

import mosek
import numpy as np
import scs
from qpsolvers import solve_qp
from scipy import sparse

from giskardpy.qp.qp_solver import QPSolver
from giskardpy.utils import logging


class QPSolverSCS(QPSolver):
    """
    min_x 0.5 x^T P x + q^T x
    s.t.  Ax = b
          Gx <= h
          lb <= x <= ub
    """



    @profile
    def solve(self, weights: np.ndarray, g: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray, lbA: np.ndarray,
              ubA: np.ndarray) -> np.ndarray:
        A = sparse.csc_matrix(np.vstack([-A, A]))
        P = sparse.csc_matrix(np.diag(weights))
        b = np.concatenate([-lbA, ubA])
        c = g

        data = {
            'P': P,
            'A': A,
            'b': b,
            'c': c
        }

        cone = {
            'l': b.shape[0],
        }

        solver = scs.SCS(data, cone, verbose=False)

        return solver.solve()['x']


    # @profile
    def solve_and_retry(self, weights, g, A, lb, ub, lbA, ubA):
        return self.solve(weights, g, A, lb, ub, lbA, ubA)
