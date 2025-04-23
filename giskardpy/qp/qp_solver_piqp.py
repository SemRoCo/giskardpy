from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import scipy.sparse as sp

import numpy as np
import piqp
from giskardpy.data_types.exceptions import InfeasibleException
from giskardpy.qp.qp_solver_ids import SupportedQPSolver
from giskardpy.qp.qp_solver_gurobi import QPSolverGurobi


class QPSolverPIQP(QPSolverGurobi):
    """
    min_x 0.5 x^T P x + c^T x
    s.t.  Ax = b
          Gx <= h
          lb <= x <= ub
    """
    solver_id = SupportedQPSolver.piqp
    sparse = True

    def analyze_infeasibility(self):
        pass

    def solver_call(self, H: np.ndarray, g: np.ndarray, E: sp.csc_matrix, b: np.ndarray, A: sp.csc_matrix,
                    lb: np.ndarray, ub: np.ndarray, h: np.ndarray) -> np.ndarray:
        import scipy.sparse as sp
        H = sp.diags(H + self.regularization_value * 10, offsets=0, format='csc')
        solver = piqp.SparseSolver()
        # solver.settings.eps_abs = 1e-3
        # solver.settings.eps_rel = 1e-4
        solver.settings.eps_duality_gap_rel = 5e-7
        # solver.settings.iterative_refinement_always_enabled = True
        solver.settings.delta_init = 7e-3
        if len(h) == 0:
            solver.setup(P=H, c=g, A=E, b=b, G=None, h=None, x_lb=lb, x_ub=ub)
        else:
            solver.setup(P=H, c=g, A=E, b=b, G=A, h=h, x_lb=lb, x_ub=ub)

        status = solver.solve()
        if status.value != piqp.PIQP_SOLVED:
            raise InfeasibleException("Solver status: {}".format(status.value))
        return solver.result.x
