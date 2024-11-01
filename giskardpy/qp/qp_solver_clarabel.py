import numpy as np
from clarabel import clarabel
import scipy.sparse as sp
from data_types.exceptions import InfeasibleException
from qp.qp_solver_qpswift import QPSolverQPSwift
from scipy import sparse
from line_profiler import profile
from giskardpy.qp.qp_solver import QPSolver
from giskardpy.qp.qp_solver_ids import SupportedQPSolver
from giskardpy.utils.decorators import record_time


class QPSolverClarabel(QPSolverQPSwift):
    solver_id = SupportedQPSolver.clarabel
    """
    min_x 0.5 x^T P x + c^T x
    s.t.  Ax = b
          Gx <= h
    """

    settings = clarabel.DefaultSettings()
    settings.verbose = False

    @profile
    @record_time
    def solver_call(self, H: np.ndarray, g: np.ndarray, E: sp.csc_matrix, b: np.ndarray, A: sp.csc_matrix,
                    h: np.ndarray) -> np.ndarray:
        G = sp.vstack([E, A])

        cones = [clarabel.ZeroConeT(b.shape[0]), clarabel.NonnegativeConeT(h.shape[0])]
        h = np.concatenate([b, h])

        solver = clarabel.DefaultSolver(H, g, G, h, cones, self.settings)
        result = solver.solve()
        if result.status != clarabel.SolverStatus.Solved:
            raise InfeasibleException(f'Failed to solve qp: {str(result.status)}')
        return np.array(result.x)

