from __future__ import annotations

from typing import TYPE_CHECKING, List

from giskardpy.qp.qp_adapter import QPData, GiskardToExplicitQPAdapter

if TYPE_CHECKING:
    pass
from enum import IntEnum

import numpy as np

from giskardpy.data_types.exceptions import QPSolverException, InfeasibleException
from giskardpy.qp.qp_solver import QPSolver
import qpSWIFT

from giskardpy.qp.qp_solver_ids import SupportedQPSolver


class QPSWIFTExitFlags(IntEnum):
    Optimal = 0  # Solution Found
    KKT_Failure = 1  # Failure in factorising KKT matrix
    MAX_ITER_REACHED = 2  # Maximum Number of Iterations Reached
    ERROR = 3  # Unknown Problem in Solver


class QPSolverQPSwift(QPSolver):
    """
    min_x 0.5 x^T P x + c^T x
    s.t.  Ax = b
          Gx <= h
    """
    solver_id = SupportedQPSolver.qpSWIFT
    required_adapter_type = GiskardToExplicitQPAdapter

    opts = {
        'OUTPUT': 1,  # 0 = sol; 1 = sol + basicInfo; 2 = sol + basicInfo + advInfo
        # 'MAXITER': 100,  # 0 < MAXITER < 200; default 100. maximum number of iterations needed
        # 'ABSTOL': 9e-4,  # 0 < ABSTOL < 1; default 1e-6. absolute tolerance
        'RELTOL': 3.5e-5,  # 0 < RELTOL < 1; default 1e-6. relative tolerance
        # 'SIGMA': 0.01,  # default 100. maximum centering allowed
        # 'VERBOSE': 1  # 0 = no print; 1 = print
    }

    def solver_call_batch(self, qps: List[QPData]) -> np.ndarray:
        result = qpSWIFT.solve_sparse_H_diag_batch(len(qps),
                                                   tuple([qp.as_qpSWIFT_data(self.opts) for qp in qps]))
        exit_flag = result['basicInfo']['ExitFlag']
        if exit_flag != 0:
            error_code = QPSWIFTExitFlags(exit_flag)
            if error_code == QPSWIFTExitFlags.MAX_ITER_REACHED:
                raise InfeasibleException(f'Failed to solve qp: {str(error_code)}')
            raise QPSolverException(f'Failed to solve qp: {str(error_code)}')
        return result['sol']

    def solver_call_explicit_interface(self, qp_data: QPData) -> np.ndarray:
        result = qpSWIFT.solve_sparse_H_diag(H=qp_data.quadratic_weights, g=qp_data.linear_weights,
                                             lb=qp_data.box_lower_constraints, ub=qp_data.box_upper_constraints,
                                             E=qp_data.eq_matrix, b=qp_data.eq_bounds,
                                             A=qp_data.neq_matrix, lbA=qp_data.neq_lower_bounds,
                                             ubA=qp_data.neq_upper_bounds,
                                             options=self.opts)
        exit_flag = result.exit_flag
        if exit_flag != 0:
            error_code = QPSWIFTExitFlags(exit_flag)
            if error_code == QPSWIFTExitFlags.MAX_ITER_REACHED:
                raise InfeasibleException(f'Failed to solve qp: {str(error_code)}')
            raise QPSolverException(f'Failed to solve qp: {str(error_code)}')
        return result.x

    solver_call = solver_call_explicit_interface
