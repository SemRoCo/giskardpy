import numpy as np
from scipy.sparse import csc_matrix

import osqp

from giskardpy.qp_solver import QPSolver


class OSQPSolver(QPSolver):

    def __init__(self, dim_a, dim_b):
        self.qpProblem = osqp.OSQP()

    def solve(self, H, g, A, lb, ub, lbA, ubA, nWSR=None):
        P = csc_matrix(H)
        q = g
        tmp = np.identity(len(lb))
        A = csc_matrix(np.vstack([tmp, A]))
        lbA = np.hstack([lb, lbA])
        ubA = np.hstack([ub, ubA])
        self.qpProblem.setup(P=P, q=q, A=A, l=lbA, u=ubA, verbose=False)
        res = self.qpProblem.solve()
        if res.info.status_val != self.qpProblem.constant('OSQP_SOLVED'):
            print("OSQP exited with status '%s'" % res.info.status)

        return res.x
