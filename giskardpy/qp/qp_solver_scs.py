import numpy as np
import scs
from data_types.exceptions import InfeasibleException
from qp.qp_solver_clarabel import QPSolverClarabel
from qp.qp_solver_ids import SupportedQPSolver
from scipy import sparse as sp

from line_profiler import profile


class QPSolverSCS(QPSolverClarabel):
    solver_id = SupportedQPSolver.scs
    """
    min_x 0.5 x^T P x + c^T x
    s.t.  Ax = b
          Gx <= h
    """

    @profile
    def solver_call(self, H: sp.csc, g: np.ndarray, E: sp.csc_matrix, b: np.ndarray, A: sp.csc_matrix,
                    h: np.ndarray) -> np.ndarray:
        data = {
            'P': H,
            'A': sp.vstack([E, A]),
            'b': np.concatenate([b, h]),
            'c': g
        }

        cone = {
            'z': b.shape[0],
            'l': h.shape[0],
        }

        solver = scs.SCS(data, cone, verbose=False, eps_abs=5e-4, eps_rel=5e-4)
        result = solver.solve()
        if result['info']['status_val'] != scs.SOLVED:
            raise InfeasibleException(f'Failed to solve qp: {str(result["info"]["status"])}')
        return result['x']
