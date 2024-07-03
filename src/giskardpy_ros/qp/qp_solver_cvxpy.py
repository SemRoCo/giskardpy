import sys
from copy import deepcopy

import cvxpy
import mosek
import numpy as np
import scs
from qpsolvers import solve_qp
from scipy import sparse

from giskardpy.qp.qp_solver import QPSolver
from giskardpy.utils import logging
from cvxpygen import cpg


class QPSolverCVXPY(QPSolver):
    """
    min_x 0.5 x^T P x + q^T x
    s.t.  Ax = b
          Gx <= h
    """

    @profile
    def solve(self, weights: np.ndarray, g: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray, lbA: np.ndarray,
              ubA: np.ndarray) -> np.ndarray:
        P, q, G, h = self.transform_problem1(weights, g, A, lb, ub, lbA, ubA)
        # Define and solve the CVXPY problem.
        x = cvxpy.Variable(weights.shape[0])
        problem = cvxpy.Problem(cvxpy.Minimize((1 / 2) * cvxpy.quad_form(x, P)),
                             [G @ x <= h])
        problem.solve(solver='COPT')

        return x.value

    # @profile
    def solve_and_retry(self, weights, g, A, lb, ub, lbA, ubA):
        return self.solve(weights, g, A, lb, ub, lbA, ubA)
