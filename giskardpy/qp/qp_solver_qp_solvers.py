import numpy as np
from giskardpy.qp.qp_solver_gurobi import QPSolverGurobi
from qpsolvers import solve_qp

from giskardpy.data_types.exceptions import QPSolverException, InfeasibleException, HardConstraintsViolatedException
from giskardpy.qp.qp_solver import QPSolver
from giskardpy.qp.qp_solver_ids import SupportedQPSolver
from line_profiler import profile
import scipy.sparse as sp


class QPSolverQPSolvers(QPSolverGurobi):
    """
    min_x 0.5 x^T P x + q^T x
    s.t.  Ax = b
          Gx <= h
          lb <= x <= ub
    """
    solver_id = SupportedQPSolver.qp_solvers

    opts = {}

    @profile
    def solver_call(self, H: np.ndarray, g: np.ndarray, E: sp.csc_matrix, b: np.ndarray, A: sp.csc_matrix,
                    lb: np.ndarray,
                    ub: np.ndarray, h: np.ndarray) -> np.ndarray:
        H = sp.diags(H, offsets=0, format='csc')
        # H = np.diag(H+self.regularization_value)
        # E = E.toarray()
        # try:
            # A = A.toarray()
        # except:
        #     A = None
        #     h = None
        result = np.array(solve_qp(P=H, q=g, G=A, h=h, A=E, b=b, lb=lb, ub=ub, solver='proxqp'))
        if len(result.shape) == 0:
            raise InfeasibleException('idk')
        return result
