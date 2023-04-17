
import numpy as np
from qpsolvers import solve_qp

from giskardpy.configs.data_types import SupportedQPSolver
from giskardpy.exceptions import QPSolverException, InfeasibleException, HardConstraintsViolatedException
from giskardpy.qp.qp_solver import QPSolver
from giskardpy.utils import logging


class QPSolverQPSolvers(QPSolver):
    """
    min_x 0.5 x^T P x + q^T x
    s.t.  Ax = b
          Gx <= h
          lb <= x <= ub
    """
    solver_id = SupportedQPSolver.qp_solvers

    opts = {}

    @profile
    def solve(self, weights: np.ndarray, g: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray, lbA: np.ndarray,
              ubA: np.ndarray) -> np.ndarray:
        # A_b = np.eye(lb.shape[0])
        G = np.vstack([-A, A])
        P = np.diag(weights)
        h = np.concatenate([-lbA, ubA])
        result = solve_qp(P=P, q=g, G=G, h=h, lb=lb, ub=ub, solver='highs')
        if result is None:
            raise InfeasibleException('idk')
        return result

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
