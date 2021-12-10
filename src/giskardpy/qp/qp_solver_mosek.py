from copy import deepcopy

import numpy as np
import sys
import mosek
import scipy.sparse

from giskardpy.exceptions import InfeasibleException
from giskardpy.utils import logging
from giskardpy.qp.qp_solver import QPSolver

unknown = [mosek.solsta.unknown]

optimal = [mosek.solsta.optimal]

infeasible = [mosek.solsta.dual_infeas_cer, mosek.solsta.prim_infeas_cer]


class QPSolverMosek(QPSolver):

    @profile
    def init(self, H, g, A, lb, ub, lbA, ubA):
        # TODO potential speed up by reusing model
        self.env = mosek.Env()
        #self.env.set_Stream(mosek.streamtype.log, self.streamprinter)
        self.qpProblem = self.env.Task(0, 0)
        #self.qpProblem.set_Stream(mosek.streamtype.log, self.streamprinter)
        self.set_qp(H, g, A, lb, ub, lbA, ubA)

    def set_qp(self, H, g, A, lb, ub, lbA, ubA):

        # Set up bounds
        # Set up and input bounds
        bkc = [mosek.boundkey.ra] * len(lbA)
        numcon = len(bkc)
        # Set lower and upper bounds for x
        bkx = [mosek.boundkey.ra] * len(lb)
        self.numvar = len(bkx)

        # Append 'numcon' empty constraints.
        # The constraints will initially have no bounds.
        self.qpProblem.appendcons(numcon)

        # Append 'numvar' variables.
        # The variables will initially be fixed at zero (x=0).
        self.qpProblem.appendvars(self.numvar)

        # Optionally add a constant term to the objective.
        self.qpProblem.putcfix(0.0)

        for j in range(self.numvar):
            # Set the linear term c_j in the objective.
            self.qpProblem.putcj(j, g[j])
            # Set the bounds on variable j
            # blx[j] <= x_j <= bux[j]
            self.qpProblem.putvarbound(j, bkx[j], lb[j], ub[j])

        # Input of A
        sA = scipy.sparse.bsr_matrix(A)
        rows, cols, vals = scipy.sparse.find(sA)
        self.qpProblem.putaijlist(rows, cols, vals)  # Non-zero Values

        for i in range(numcon):
            self.qpProblem.putconbound(i, bkc[i], lbA[i], ubA[i])

        # Set up and input quadratic objective
        qsubi = list(range(0, len(H)))
        qsubj = deepcopy(qsubi)
        qval = list(H)
        self.qpProblem.putqobj(qsubi, qsubj, qval)
        # Input the objective sense (minimize/maximize)
        self.qpProblem.putobjsense(mosek.objsense.minimize)

    # Define a stream printer to grab output from MOSEK
    def streamprinter(self, text):
        sys.stdout.write(text)
        sys.stdout.flush()

    def print_debug(self):
        # Print a summary containing information
        # about the solution for debugging purposes
        self.qpProblem.solutionsummary(mosek.streamtype.msg)

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
            self.init(H, g, A, lb, ub, lbA, ubA)
            self.qpProblem.optimize()
            # prosta = self.qpProblem.getprosta(mosek.soltype.itr)
            solsta = self.qpProblem.getsolsta(mosek.soltype.itr)
            if solsta in optimal:
                # Output a solution
                xx = [0.] * self.numvar
                self.qpProblem.getxx(mosek.soltype.itr, xx)
                self.xdot_full = xx
                break
            elif i < tries - 1:
                self.print_debug()
                logging.logwarn(u'Solver returned \'{}\', retrying with data rounded to \'{}\' decimal places'.format(
                    self.qpProblem.solution.get_status_string(),
                    decimal_places
                ))
                H = self.round(H, decimal_places)
                A = self.round(A, decimal_places)
                lb = self.round(lb, decimal_places)
                ub = self.round(ub, decimal_places)
                lbA = self.round(lbA, decimal_places)
                ubA = self.round(ubA, decimal_places)
        else:
            self.print_debug()
            if solsta in infeasible:
                error_message = u'Problem is infeasible.'
                raise InfeasibleException(error_message)
            elif solsta in unknown:
                error_message = u'Problem return unknown solution status.'
                raise InfeasibleException(error_message)
            else:
                error_message = u'Problem returned other solution status'
                raise InfeasibleException(error_message)
        return self.xdot_full
