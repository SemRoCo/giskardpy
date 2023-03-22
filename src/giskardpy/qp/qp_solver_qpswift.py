from enum import IntEnum

import numpy as np

from giskardpy.configs.data_types import SupportedQPSolver
from giskardpy.exceptions import QPSolverException, InfeasibleException, HardConstraintsViolatedException
from giskardpy.qp.qp_solver import QPSolver
import qpSWIFT

from giskardpy.utils import logging
from giskardpy.utils.utils import record_time


class QPSWIFTExitFlags(IntEnum):
    Optimal = 0  # Solution Found
    KKT_Failure = 1  # Failure in factorising KKT matrix
    MAX_ITER_REACHED = 2  # Maximum Number of Iterations Reached
    ERROR = 3  # Unknown Problem in Solver


class QPSolverQPSwift(QPSolver):
    solver_id = SupportedQPSolver.qpSWIFT
    """
    min_x 0.5 x^T P x + c^T x
    s.t.  Ax = b
          Gx <= h
    """

    opts = {
        'OUTPUT': 1,  # 0 = sol; 1 = sol + basicInfo; 2 = sol + basicInfo + advInfo
        # 'MAXITER': 100, # 0 < MAXITER < 200; default 100
        # 'ABSTOL': 1e-5, # 0 < ABSTOL < 1
        # 'RELTOL': 1, # 0 < RELTOL < 1
        # 'SIGMA': 1,
        # 'VERBOSE': 1  # 0 = no print; 1 = print
    }

    @record_time
    @profile
    def solve(self, weights: np.ndarray, g: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray, lbA: np.ndarray,
              ubA: np.ndarray) -> np.ndarray:
        P, g, G, h, A, b = self.transform_qpSWIFT(weights, g, A, lb, ub, lbA, ubA)
        result = qpSWIFT.run(c=g, h=h, P=P, G=G, A=A, b=b, opts=self.opts)
        # A_b = np.eye(lb.shape[0])
        # G = np.vstack([-A_b, A_b, -A, A])
        # P = np.diag(weights)
        # h = np.concatenate([-lb, ub, -lbA, ubA])
        # result = qpSWIFT.run(c=g, h=h, P=P, G=G, opts=self.opts)
        exit_flag = result['basicInfo']['ExitFlag']
        # print(result['basicInfo']['Iterations'])
        if exit_flag != 0:
            error_code = QPSWIFTExitFlags(exit_flag)
            if error_code == QPSWIFTExitFlags.MAX_ITER_REACHED:
                raise InfeasibleException(f'Failed to solve qp: {str(error_code)}')
            raise QPSolverException(f'Failed to solve qp: {str(error_code)}')
        return result['sol']

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
