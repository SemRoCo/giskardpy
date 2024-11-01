import numpy as np
import piqp
from giskardpy.data_types.exceptions import QPSolverException, InfeasibleException
from giskardpy.qp.qp_solver_ids import SupportedQPSolver
from line_profiler import profile
from giskardpy.qp.qp_solver_gurobi import QPSolverGurobi
from scipy import sparse as sp


class QPSolverPIQP(QPSolverGurobi):
    """
    min_x 0.5 x^T P x + c^T x
    s.t.  Ax = b
          Gx <= h
          lb <= x <= ub
    """
    solver_id = SupportedQPSolver.piqp
    sparse = True

    @profile
    def solver_call(self, H: np.ndarray, g: np.ndarray, E: sp.csc_matrix, b: np.ndarray, A: sp.csc_matrix,
                    lb: np.ndarray, ub: np.ndarray, h: np.ndarray) -> np.ndarray:
        H = sp.diags(H, offsets=0, format='csc')
        solver = piqp.SparseSolver()

        solver.setup(P=H, c=g, A=E, b=b, G=A, h=h, x_lb=lb, x_ub=ub)

        status = solver.solve()
        if status.value != piqp.PIQP_SOLVED:
            raise InfeasibleException("Solver status: {}".format(status.value))
        return solver.result.x
