import numpy as np
from scipy.sparse import csc_matrix

import cvxopt

from giskardpy.qp_solver import QPSolver

def cvxopt_matrix(M):
    if type(M) is np.ndarray:
        return cvxopt.matrix(M)
    elif type(M) is cvxopt.spmatrix or type(M) is cvxopt.matrix:
        return M
    coo = M.tocoo()
    return cvxopt.spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=M.shape)

class CVXQPSolver(QPSolver):

    def __init__(self, dim_a, dim_b):
        # self.qpProblem = osqp.OSQP()
        pass

    def solve(self, H, g, A, lb, ub, lbA, ubA, nWSR=None):
        P = H
        q = g
        G = np.vstack([np.identity(len(ub)), -np.identity(len(ub)), A, -A])
        h = np.hstack([ub, -lb, ubA, -lbA])
        args = [cvxopt_matrix(P), cvxopt_matrix(q)]
        if G is not None:
            args.extend([cvxopt_matrix(G), cvxopt_matrix(h)])
        sol = cvxopt.solvers.qp(*args)
        # if 'optimal' not in sol['status']:
        #     return None

        return np.array(sol['x']).reshape((q.shape[0],))

