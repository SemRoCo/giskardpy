import sys
from copy import deepcopy

import mosek
import numpy as np
from qpsolvers import solve_qp
from scipy import sparse

from giskardpy.qp.qp_solver import QPSolver
from giskardpy.utils import logging

unknown = [mosek.solsta.unknown]

optimal = [mosek.solsta.optimal]

infeasible = [mosek.solsta.dual_infeas_cer, mosek.solsta.prim_infeas_cer]

inf = 0.0


def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()


class QPSolverMosek(QPSolver):
    """
    min_x 0.5 x^T P x + q^T x
    s.t.  Ax = b
          Gx <= h
          lb <= x <= ub
    """

    def init(self, H, g, A, lb, ub, lbA, ubA):
        # TODO potential speed up by reusing model
        self.env = mosek.Env()
        # self.env.set_Stream(mosek.streamtype.log, self.streamprinter)
        self.qpProblem = self.env.Task(0, 0)
        # self.qpProblem.set_Stream(mosek.streamtype.log, self.streamprinter)
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
        sA = sparse.bsr_matrix(A)
        rows, cols, vals = sparse.find(sA)
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
    def solve(self, weights: np.ndarray, g: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray, lbA: np.ndarray,
              ubA: np.ndarray) -> np.ndarray:
        # Open MOSEK and create an environment and task
        # Make a MOSEK environment
        with mosek.Env() as env:
            # Attach a printer to the environment
            env.set_Stream(mosek.streamtype.log, streamprinter)
            # Create a task
            with env.Task() as task:
                task.set_Stream(mosek.streamtype.log, streamprinter)
                # Set up and input bounds and linear coefficients
                bkc = [mosek.boundkey.lo]
                blc = [1.0]
                buc = [inf]
                numvar = 3
                bkx = [mosek.boundkey.lo] * numvar
                blx = [0.0] * numvar
                bux = [inf] * numvar
                c = [0.0, -1.0, 0.0]
                asub = [[0], [0], [0]]
                aval = [[1.0], [1.0], [1.0]]

                numvar = lb.shape[0]
                numcon = len(bkc)

                # Append 'numcon' empty constraints.
                # The constraints will initially have no bounds.
                task.appendcons(numcon)

                # Append 'numvar' variables.
                # The variables will initially be fixed at zero (x=0).
                task.appendvars(numvar)

                for j in range(numvar):
                    # Set the linear term c_j in the objective.
                    task.putcj(j, c[j])
                    # Set the bounds on variable j
                    # blx[j] <= x_j <= bux[j]
                    task.putvarbound(j, bkx[j], blx[j], bux[j])
                    # Input column j of A
                    task.putacol(j,  # Variable (column) index.
                                 # Row index of non-zeros in column j.
                                 asub[j],
                                 aval[j])  # Non-zero Values of column j.
                for i in range(numcon):
                    task.putconbound(i, bkc[i], blc[i], buc[i])

                # Set up and input quadratic objective
                qsubi = [0, 1, 2, 2]
                qsubj = [0, 1, 0, 2]
                qval = [2.0, 0.2, -1.0, 2.0]

                task.putqobj(qsubi, qsubj, qval)

                # Input the objective sense (minimize/maximize)
                task.putobjsense(mosek.objsense.minimize)

                # Optimize
                task.optimize()
                # Print a summary containing information
                # about the solution for debugging purposes
                task.solutionsummary(mosek.streamtype.msg)

                prosta = task.getprosta(mosek.soltype.itr)
                solsta = task.getsolsta(mosek.soltype.itr)

                # Output a solution
                xx = task.getxx(mosek.soltype.itr)

                if solsta == mosek.solsta.optimal:
                    print("Optimal solution: %s" % xx)
                elif solsta == mosek.solsta.dual_infeas_cer:
                    print("Primal or dual infeasibility.\n")
                elif solsta == mosek.solsta.prim_infeas_cer:
                    print("Primal or dual infeasibility.\n")
                elif mosek.solsta.unknown:
                    print("Unknown solution status")
                else:
                    print("Other solution status")

    # @profile
    def solve_and_retry(self, weights, g, A, lb, ub, lbA, ubA):
        return self.solve(weights, g, A, lb, ub, lbA, ubA)
