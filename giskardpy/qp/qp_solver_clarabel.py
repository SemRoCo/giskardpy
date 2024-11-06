from typing import Tuple

import numpy as np
import scipy.sparse as sp
from clarabel import clarabel
from data_types.exceptions import InfeasibleException
from giskardpy.qp.qp_solver_ids import SupportedQPSolver
from line_profiler import profile
from qp.qp_solver_qpswift import QPSolverQPSwift


class QPSolverClarabel(QPSolverQPSwift):
    solver_id = SupportedQPSolver.clarabel
    """
    min_x 0.5 x^T P x + c^T x
    s.t.  Ax = b
          Gx <= h
    """

    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_gap_abs = 1e-5

    @profile
    def problem_data_to_qp_format(self) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        H = sp.diags(self.weights, offsets=0, format='csc')
        if np.product(self.nA_A.shape) > 0:
            A = sp.vstack((self.nAi_Ai, self.nA_A))
        else:
            A = self.nAi_Ai
        nlb_ub_nlbA_ubA = np.concatenate((self.nlb, self.ub, self.nlbA_ubA))
        return H, self.g, self.E, self.bE, A, nlb_ub_nlbA_ubA

    @profile
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
