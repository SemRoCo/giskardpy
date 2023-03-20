from enum import IntEnum

import numpy as np
import qpalm
from scipy import sparse

from giskardpy.configs.data_types import SupportedQPSolver
from giskardpy.exceptions import QPSolverException, InfeasibleException, HardConstraintsViolatedException
from giskardpy.qp.qp_solver import QPSolver
from giskardpy.utils import logging

from giskardpy.utils.utils import record_time


class QPALMInfo(IntEnum):
    SOLVED = 1  # status to indicate the problem is solved to optimality given the specified tolerances
    DUAL_TERMINATED = 2  # status to indicate the problem has a dual objective that is higher than the specified bound
    MAX_ITER_REACHED = -2  # status to indicate termination due to reaching the maximum number of iterations
    PRIMAL_INFEASIBLE = -3  # status to indicate the problem is primal infeasible
    DUAL_INFEASIBLE = -4  # status to indicate the problem is dual infeasible
    TIME_LIMIT_REACHED = -5  # status to indicate the problem's runtime has exceeded the specified time limit
    UNSOLVED = -10  # status to indicate the problem is unsolved.
    ERROR = 0


class QPSolverQPalm(QPSolver):
    solver_id = SupportedQPSolver.qpalm
    """
    min_x 0.5 x^T Q x + q^T x
    s.t.  lb <= Ax <= ub
    https://github.com/kul-optec/QPALM
    """

    settings = qpalm.Settings()
    settings.verbose = False
    settings.eps_abs = 1e-5
    settings.eps_rel = 1e-8
    settings.nonconvex = False
    # settings.max_iter = 100

    @profile
    @record_time
    def solve(self, weights: np.ndarray, g: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray, lbA: np.ndarray,
              ubA: np.ndarray) -> np.ndarray:
        A_b = np.eye(lb.shape[0])
        Q = sparse.csc_matrix(np.diag(weights))
        q = g
        A = sparse.csc_matrix(np.vstack([A_b, A]))
        lb = np.concatenate([lb, lbA])
        ub = np.concatenate([ub, ubA])

        data = qpalm.Data(A.shape[1], A.shape[0])

        data.Q = Q
        data.A = A
        data.q = q
        data.bmax = ub
        data.bmin = lb

        solver = qpalm.Solver(data, self.settings)
        solver.solve()
        # print(f'{solver.info.iter} {solver.info.iter_out}')
        if solver.info.status_val != QPALMInfo.SOLVED:
            raise InfeasibleException(f'Failed to solve qp: {str(QPALMInfo(solver.info.status_val))}')
        return solver.solution.x

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
