from typing import Tuple

import numpy as np
import osqp

from giskardpy.data_types.exceptions import InfeasibleException
from giskardpy.qp.qp_solver_ids import SupportedQPSolver
from giskardpy.qp.qp_solver_qpalm import QPSolverQPalm


class QPSolverOSQP(QPSolverQPalm):
    """
    min_x 0.5 x^T P x + q^T x
    s.t.  l <= Ax = u
    """
    solver_id = SupportedQPSolver.osqp
    settings = {
        'verbose': False,
        # 'eps_abs': 1e-6,
        # 'eps_rel': 1e-6,
        # 'eps_prim_inf': 1e-6,
        # 'polish': 1,
        # 'eps_abs': 1e-3,
        # 'eps_rel': 1e-2,  # default 1e-3
        # 'polish_refine_iter': 10, # default 3
    }


    def problem_data_to_qp_format(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        import scipy.sparse as sp
        H = sp.diags(self.weights + self.regularization_value)
        A = sp.vstack((self.Ai, self.A))
        return H, self.g, A, self.lb_bE_lbA, self.ub_bE_ubA


    def solver_call(self, H: np.ndarray, g: np.ndarray, A: np.ndarray, lbA: np.ndarray, ubA: np.ndarray) \
            -> np.ndarray:
        m = osqp.OSQP()
        m.setup(P=H, q=g, A=A, l=lbA, u=ubA, **self.settings)
        result = m.solve()
        if result.info.status != 'solved':
            raise InfeasibleException(f'unable to solve qp {result.info.status}')
        return result.x
