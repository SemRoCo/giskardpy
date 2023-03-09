
import numpy as np
from casadi import casadi
from qpsolvers import solve_qp

from giskardpy.qp.qp_solver import QPSolver


class QPSolverCasadi(QPSolver):
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
        h = casadi.DM(np.diag(weights))
        a = casadi.DM(A)
        g = casadi.DM(g)
        lba = casadi.DM(lbA)
        uba = casadi.DM(ubA)
        lbx = casadi.DM(lb)
        ubx = casadi.DM(ub)
        qp = {
            'h': h.sparsity(),
            'a': a.sparsity(),
        }
        solver = casadi.conic('S', 'qpoases', qp, verbose=False)
        result = solver(h=h,a=a,g=g,lba=lba,uba=uba,lbx=lbx,ubx=ubx)['x']
        result = np.array(result)[:,0]
        return result

    # @profile
    def solve_and_retry(self, weights, g, A, lb, ub, lbA, ubA):
        return self.solve(weights, g, A, lb, ub, lbA, ubA)
