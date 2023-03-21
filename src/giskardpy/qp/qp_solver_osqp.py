
import numpy as np
from scipy import sparse

from giskardpy.configs.data_types import SupportedQPSolver
from giskardpy.exceptions import QPSolverException, InfeasibleException, HardConstraintsViolatedException
from giskardpy.qp.qp_solver import QPSolver
import osqp

from giskardpy.utils import logging
from giskardpy.utils.utils import record_time


class QPSolverOSQP(QPSolver):
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
    @record_time
    def solve(self, weights: np.ndarray, g: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray, lbA: np.ndarray,
              ubA: np.ndarray) -> np.ndarray:
        m = osqp.OSQP()
        A_b = np.eye(lb.shape[0])
        A = sparse.csc_matrix(np.vstack([A_b, A]))
        P = sparse.csc_matrix(np.diag(weights))
        l = np.concatenate([lb, lbA])
        u = np.concatenate([ub, ubA])
        m.setup(P=P, q=g, A=A, l=l, u=u, **self.settings)
        return m.solve().x

    # @profile
    def solve_and_retry(self, weights, g, A, lb, ub, lbA, ubA):
        exception = None
        for i in range(2):
            try:
                return self.solve(weights, g, A, lb, ub, lbA, ubA)
            except QPSolverException as e:
                exception = e
                try:
                    weights, lb, ub = self.compute_relaxed_hard_constraints(weights, g, A, lb, ub, lbA, ubA)
                    logging.loginfo(f'{e}; retrying with relaxed hard constraints')
                except InfeasibleException as e2:
                    if isinstance(e2, HardConstraintsViolatedException):
                        raise e2
                    raise e
                continue
        raise exception
