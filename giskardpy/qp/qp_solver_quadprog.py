from typing import Tuple

import numpy as np
import quadprog

from giskardpy.data_types.exceptions import InfeasibleException
from giskardpy.qp.qp_solver_clarabel import QPSolverClarabel
from giskardpy.qp.qp_solver_ids import SupportedQPSolver


class QPSolverQuadprog(QPSolverClarabel):
    solver_id = SupportedQPSolver.quadprog
    """
    min_x 0.5 x^T G x + a^T x
    s.t.  Cx >= b
    """

    
    def problem_data_to_qp_format(self) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        H = np.diag(self.weights + self.regularization_value)
        if np.product(self.nA_A.shape) > 0:
            A = np.vstack((self.nAi_Ai.toarray(), self.nA_A.toarray()))
        else:
            A = self.nAi_Ai.toarray()
        nlb_ub_nlbA_ubA = np.concatenate((self.nlb, self.ub, self.nlbA_ubA))
        return H, self.g, self.E, self.bE, A, nlb_ub_nlbA_ubA

    
    def solver_call(self, H: np.ndarray, g: np.ndarray, E: np.ndarray, b: np.ndarray, A: np.ndarray,
                    h: np.ndarray) -> np.ndarray:
        import scipy.sparse as sp
        C = -sp.vstack([E, A]).toarray().T
        bb = -np.concatenate([b, h])

        try:
            return quadprog.solve_qp(G=H, a=g, C=C, b=bb, meq=b.shape[0])[0]
        except ValueError as e:
            raise InfeasibleException(str(e))

