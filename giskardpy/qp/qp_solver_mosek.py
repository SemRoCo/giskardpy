import sys
from copy import deepcopy
from line_profiler import profile
import mosek
import numpy as np
from giskardpy.qp.qp_solver_gurobi import QPSolverGurobi
from giskardpy.qp.qp_solver_ids import SupportedQPSolver
from scipy import sparse as sp


unknown = [mosek.solsta.unknown]

optimal = [mosek.solsta.optimal]

infeasible = [mosek.solsta.dual_infeas_cer, mosek.solsta.prim_infeas_cer]

inf = 0.0


def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()


class QPSolverMosek(QPSolverGurobi):
    """
    min_x 0.5 x^T P x + q^T x
    s.t.  Ax = b
          Gx <= h
          lb <= x <= ub
    """
    solver_id = SupportedQPSolver.mosek

    @profile
    def solver_call(self, H: sp.csc_matrix, g: np.ndarray, E: sp.csc_matrix, b: np.ndarray,
                    A: sp.csc_matrix, lb: np.ndarray, ub: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Solve the quadratic optimization problem:
            minimize    (1/2) * x^T H x + g^T x
            subject to  E x = b
                        A x <= h
                        lb <= x <= ub
        """
        H = sp.diags(H, offsets=0, format='coo')

        # Helper function to print log output from MOSEK
        def streamprinter(text):
            sys.stdout.write(text)
            sys.stdout.flush()

        # Create a MOSEK environment
        with mosek.Env() as env:
            # Create a task
            with env.Task(0, 0) as task:
                # Attach a log stream printer to the task
                # task.set_Stream(mosek.streamtype.log, streamprinter)

                n = H.shape[1]
                m = A.shape[0] if A is not None and A.shape[0] > 0 else 0
                p = E.shape[0] if E is not None and E.shape[0] > 0 else 0

                numvar = n
                numcon = m + p

                # Append 'numcon' empty constraints.
                task.appendcons(numcon)

                # Append 'numvar' variables.
                task.appendvars(numvar)

                # Set the bounds on variables
                inf = 0.0
                # inf = mosek.infinity
                bkx = []
                blx = []
                bux = []

                for i in range(n):
                    lbi = lb[i]
                    ubi = ub[i]
                    if np.isinf(lbi) and np.isinf(ubi):
                        bk = mosek.boundkey.fr
                        bl = -inf
                        bu = +inf
                    elif np.isinf(lbi):
                        bk = mosek.boundkey.up
                        bl = -inf
                        bu = ubi
                    elif np.isinf(ubi):
                        bk = mosek.boundkey.lo
                        bl = lbi
                        bu = +inf
                    elif lbi == ubi:
                        bk = mosek.boundkey.fx
                        bl = lbi
                        bu = ubi
                    else:
                        bk = mosek.boundkey.ra
                        bl = lbi
                        bu = ubi
                    # Set the variable bound
                    task.putvarbound(i, bk, bl, bu)

                # Set the constraints bounds
                bkc = []
                blc = []
                buc = []

                # Inequality constraints A x <= h
                for i in range(m):
                    bkc.append(mosek.boundkey.up)
                    blc.append(-inf)
                    buc.append(h[i])

                # Equality constraints E x = b
                for i in range(p):
                    bkc.append(mosek.boundkey.fx)
                    blc.append(b[i])
                    buc.append(b[i])

                # Set constraint bounds
                for i in range(numcon):
                    task.putconbound(i, bkc[i], blc[i], buc[i])

                # Stack A and E vertically to form constraints matrix
                if m > 0 and p > 0:
                    constraints = sp.vstack([A, E])
                elif m > 0:
                    constraints = A
                elif p > 0:
                    constraints = E
                else:
                    constraints = sp.csr_matrix((0, n))  # Empty matrix

                # Convert constraints to COO format
                constraints_coo = constraints.tocoo()
                # Get the row indices, column indices, and values
                subi = constraints_coo.row.tolist()
                subj = constraints_coo.col.tolist()
                valij = constraints_coo.data.tolist()

                # Input the linear constraints
                task.putaijlist(subi, subj, valij)

                # Set the linear objective term
                for i in range(n):
                    task.putcj(i, g[i])

                # Set up the quadratic objective terms
                # H should be symmetric, but we need to input only lower triangular part
                I = H.row
                J = H.col
                V = H.data
                tril_idx = I >= J
                qsubi = I[tril_idx].tolist()
                qsubj = J[tril_idx].tolist()
                qval = V[tril_idx].tolist()
                task.putqobj(qsubi, qsubj, qval)

                # Set the objective sense to minimization
                task.putobjsense(mosek.objsense.minimize)

                # Optimize the task
                task.optimize()

                # Get the solution status
                solsta = task.getsolsta(mosek.soltype.itr)

                # Output a solution
                xx = [0.] * n
                task.getxx(mosek.soltype.itr, xx)

                x = np.array(xx)

        if solsta == mosek.solsta.optimal:
            return x
        elif solsta == mosek.solsta.dual_infeas_cer:
            raise Exception("Dual infeasibility certificate found.")
        elif solsta == mosek.solsta.prim_infeas_cer:
            raise Exception("Primal infeasibility certificate found.")
        else:
            raise Exception(f"Unknown solution status: {solsta}")
