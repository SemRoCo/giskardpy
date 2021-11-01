from copy import deepcopy

import numpy as np
import qpsolvers

from giskardpy.utils import logging
from giskardpy.qp.qp_solver import QPSolver


class QPSolverPackageWrapper(QPSolver):

    @profile
    def __init__(self, solver='gurobi'):
        # TODO potential speed up by reusing model
        super().__init__()
        self.solver = solver

    @profile
    def ensure_diag_positive_definite(self, M, v=1e-16):
        # https://en.wikipedia.org/wiki/Gershgorin_circle_theorem
        for i in range(0, M.shape[0]):
            sum_row = np.sum(M[i, :]) - M[i, i]
            if sum_row >= M[i, i]:
                M[i, i] = sum_row + v
        return M

    @profile
    def zero_mask(self, a, b, M, mask):
        """
        :param a: vector
        :param b: vector
        :param M: matrix
        """
        if a.shape != b.shape or a.shape[0] != M.shape[0]:
            raise Exception(u'Invalid input shapes: a: {}, b: {}, M: {}'.format(a.shape, b.shape, M.shape))
        a_out = deepcopy(a)
        b_out = deepcopy(b)
        M_out = deepcopy(M)
        a_out[mask] = 0
        b_out[mask] = 0
        M_out[mask, :] = 0
        return a_out, b_out, M_out

    @profile
    def get_ineq(self, a, b, M):
        """
        :param a: vector
        :param b: vector
        :param M: matrix
        """
        return self.zero_mask(a, b, M, a == b)

    @profile
    def get_eq(self, a, b, M):
        """
        :param a: vector
        :param b: vector
        :param M: matrix
        """
        return self.zero_mask(a, b, M, a != b)

    @profile
    def get_inequality_constraints(self, ubA, lbA, A):
        ineq_ubA, ineq_lbA, ineq_A = self.get_ineq(ubA, lbA, A)
        G = np.vstack((ineq_A, -ineq_A))  # Ax < ubA, -A < -lbA
        h = np.hstack((ineq_ubA, -ineq_lbA))
        return G, h

    @profile
    def get_equality_constraints(self, ubA, lbA, A):
        eq_ubA, eq_lbA, eq_A = self.get_eq(ubA, lbA, A)
        A_out = np.vstack((eq_A, -eq_A))  # Ax < ubA, -A < -lbA
        b = np.hstack((eq_ubA, -eq_lbA))
        return A_out, b

    @profile
    def _solve(self, H, g, A, lb, ub, lbA, ubA):
        solver_inputs = self._get_qp(H, g, A, lb, ub, lbA, ubA)
        self.xdot_full = qpsolvers.solve_qp(*solver_inputs, solver=self.solver, verbose=False)

    @profile
    def _get_qp(self, H, g, A, lb, ub, lbA, ubA):
        # inequality constraints
        G, h = self.get_inequality_constraints(ubA, lbA, A)
        # equality constraints
        A_out, b = self.get_equality_constraints(ubA, lbA, A)
        # weight matrix
        P = 2.0 * np.eye(len(H)) * H
        P = self.ensure_diag_positive_definite(P)
        return deepcopy((P, g, G, h, A_out, b, lb, ub))

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
            try:
                self._solve(H, np.zeros(H.shape[0]), A, lb, ub, lbA, ubA)
            except Exception:
                pass
            if self.xdot_full is not None:
                break
            elif i < tries - 1:
                # self.print_debug()
                logging.logwarn(
                    u'Solver returned non-optimal, retrying with data rounded to \'{}\' decimal places'.format(
                        decimal_places
                    ))
                H = self.round(H, decimal_places)
                A = self.round(A, decimal_places)
                lb = self.round(lb, decimal_places)
                ub = self.round(ub, decimal_places)
                lbA = self.round(lbA, decimal_places)
                ubA = self.round(ubA, decimal_places)
        else:
            # self.print_debug()
            pass
        return self.xdot_full
