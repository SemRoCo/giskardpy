import numpy as np

import quadprog

from giskardpy.qp_solver import QPSolver

class QuadProgQPSolver(QPSolver):

    def __init__(self, dim_a, dim_b):
        pass

    def solve(self, H, g, A, lb, ub, lbA, ubA, nWSR=None):
        G = np.vstack([np.identity(len(ub)), -np.identity(len(ub)), A, -A])
        h = np.hstack([ub, -lb, ubA, -lbA])
        qp_G = H
        qp_a = -g
        qp_C = -G.T
        qp_b = -h
        meq = 0

        return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

