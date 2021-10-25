from copy import deepcopy

import numpy as np
import qpsolvers

from giskardpy.utils import logging
from giskardpy.qp.qp_solver import QPSolver


class QPSolverPackageWrapper(QPSolver):

    @profile
    def __init__(self, solver='mosek'):
        # TODO potential speed up by reusing model
        super().__init__()
        self.solver = solver

    def ensure_diag_positive_definite(self, M, v=1e-16):
        # https://en.wikipedia.org/wiki/Gershgorin_circle_theorem
        for i in range(0, M.shape[0]):
            sum_row = np.sum(M[i,:]) - M[i,i]
            if sum_row >= M[i,i]:
                M[i,i] = sum_row + v
        return M

    def get_qp(self, H, g, A, lb, ub, lbA, ubA):
        # Add vars with limits and linear objective if valid
        G = np.vstack((A, -A)) # Ax < ubA, -A < -lbA
        h = np.hstack((ubA, -lbA))
        P = 2.0 * np.eye(len(H)) * H
        P = self.ensure_diag_positive_definite(P)
        return P, np.zeros(P.shape[0]), G, h, np.zeros(G.shape), np.zeros(h.shape), lb, ub

    def print_debug(self):
        # Print QP problem stats
        logging.logwarn(u'')

    def round(self, data, decimal_places):
        return np.round(data, decimal_places)

    @profile
    def solve(self, H, g, A, lb, ub, lbA, ubA, tries=1, decimal_places=4):
        """
        x^T*H*x + x^T*g
        s.t.: lbA < A*x < ubA
        and    lb <  x  < ub
        :param H: 2d diagonal weight matrix, shape = (jc (joint constraints) + sc (soft constraints)) * (jc + sc)
        :type np.array
        :param g: 1d zero vector of len joint constraints + soft constraints
        :type np.array
        :param A: 2d jacobi matrix of hc (hard constraints) and sc, shape = (hc + sc) * (number of joints)
        :type np.array
        :param lb: 1d vector containing lower bound of x, len = jc + sc
        :type np.array
        :param ub: 1d vector containing upper bound of x, len = js + sc
        :type np.array
        :param lbA: 1d vector containing lower bounds for the change of hc and sc, len = hc+sc
        :type np.array
        :param ubA: 1d vector containing upper bounds for the change of hc and sc, len = hc+sc
        :type np.array
        :param nWSR:
        :type np.array
        :return: x according to the equations above, len = joint constraints + soft constraints
        :type np.array
        """
        for i in range(tries):
            P, q, G, h, A, b, lb, ub = deepcopy(self.get_qp(H, g, A, lb, ub, lbA, ubA))
            try:
                x = qpsolvers.solve_qp(P, q, G, h, A, b, lb, ub, solver=self.solver, verbose=True)
            except Exception:
                pass
            if x is not None:
                self.xdot_full = x
                break
            elif i < tries - 1:
                #self.print_debug()
                logging.logwarn(u'Solver returned non-optimal, retrying with data rounded to \'{}\' decimal places'.format(
                    decimal_places
                ))
                H = self.round(H, decimal_places)
                A = self.round(A, decimal_places)
                lb = self.round(lb, decimal_places)
                ub = self.round(ub, decimal_places)
                lbA = self.round(lbA, decimal_places)
                ubA = self.round(ubA, decimal_places)
        else:
            #self.print_debug()
            pass
        return self.xdot_full
