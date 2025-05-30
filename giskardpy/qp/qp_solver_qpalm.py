from enum import IntEnum
from typing import Tuple

import numpy as np
import qpalm

import giskardpy.casadi_wrapper as cas
from giskardpy.data_types.exceptions import QPSolverException, InfeasibleException, HardConstraintsViolatedException
from giskardpy.qp.qp_adapter import QPData, GiskardToExplicitQPAdapter, GiskardToTwoSidedNeqQPAdapter
from giskardpy.qp.qp_solver import QPSolver
from giskardpy.qp.qp_solver_ids import SupportedQPSolver


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
    required_adapter_type = GiskardToTwoSidedNeqQPAdapter

    """
    min_x 0.5 x^T Q x + q^T x
    s.t.  lb <= Ax <= ub
    https://github.com/kul-optec/QPALM
    """
    settings = qpalm.Settings()
    settings.verbose = False
    settings.eps_abs = 3e-5
    settings.eps_rel = 1e-8
    settings.nonconvex = False


    def solver_call(self, qp_data: QPData) -> np.ndarray:
        data = qpalm.Data(qp_data.neq_matrix.shape[1], qp_data.neq_matrix.shape[0])

        data.Q = qp_data.sparse_hessian
        data.q = qp_data.linear_weights
        data.A = qp_data.neq_matrix
        data.bmin = qp_data.neq_lower_bounds
        data.bmax = qp_data.neq_upper_bounds

        solver = qpalm.Solver(data, self.settings)
        solver.solve()
        if solver.info.status_val != QPALMInfo.SOLVED:
            raise InfeasibleException(f'Failed to solve qp: {str(QPALMInfo(solver.info.status_val))}')
        return solver.solution.x

    def solver_call_explicit_interface(self, qp_data: QPData) -> np.ndarray:
        A2 = np.eye(len(ub))
        if len(E) > 0:
            A2 = np.vstack((A2, E))
        if len(A) > 0:
            A2 = np.vstack((A2, A))
        lbA = np.concatenate((lb, bE, lbA))
        ubA = np.concatenate((ub, bE, ubA))
        return self.solver_call(H, g, A2, lbA, ubA)
