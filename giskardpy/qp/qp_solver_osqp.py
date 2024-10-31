
import numpy as np
from qp.qp_solver_qpalm import QPSolverQPalm
from scipy import sparse

from giskardpy.data_types.exceptions import QPSolverException, InfeasibleException, HardConstraintsViolatedException
from giskardpy.qp.qp_solver import QPSolver
import osqp
from line_profiler import profile
from giskardpy.qp.qp_solver_ids import SupportedQPSolver
from giskardpy.utils.decorators import record_time


class QPSolverOSQP(QPSolverQPalm):
    """
    min_x 0.5 x^T P x + q^T x
    s.t.  l <= Ax = u
    """
    solver_id = SupportedQPSolver.osqp
    settings = {
        'verbose': False,
        'polish': True,
        # 'eps_abs': 1e-3,
        'eps_rel': 1e-2,  # default 1e-3
        'polish_refine_iter': 4, # default 3
    }


    @profile
    def solver_call(self, H: np.ndarray, g: np.ndarray, A: np.ndarray, lbA: np.ndarray, ubA: np.ndarray) \
            -> np.ndarray:
        m = osqp.OSQP()
        m.setup(P=H, q=g, A=A, l=lbA, u=ubA, **self.settings)
        return m.solve().x

