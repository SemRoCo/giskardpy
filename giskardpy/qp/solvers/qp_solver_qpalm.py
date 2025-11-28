from enum import IntEnum

import numpy as np
import qpalm

from giskardpy.qp.exceptions import InfeasibleException
from giskardpy.qp.adapters.two_sided_neq_adapter import GiskardToTwoSidedNeqQPAdapter
from giskardpy.qp.qp_data import QPData
from giskardpy.qp.solvers.qp_solver import QPSolver
from giskardpy.qp.solvers.qp_solver_ids import SupportedQPSolver


class QPALMInfo(IntEnum):
    SOLVED = 1  # status to indicate the problem is solved to optimality given the specified tolerances
    DUAL_TERMINATED = 2  # status to indicate the problem has a dual objective that is higher than the specified bound
    MAX_ITER_REACHED = (
        -2
    )  # status to indicate termination due to reaching the maximum number of iterations
    PRIMAL_INFEASIBLE = -3  # status to indicate the problem is primal infeasible
    DUAL_INFEASIBLE = -4  # status to indicate the problem is dual infeasible
    TIME_LIMIT_REACHED = (
        -5
    )  # status to indicate the problem's runtime has exceeded the specified time limit
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
            raise InfeasibleException(
                f"Failed to solve qp: {str(QPALMInfo(solver.info.status_val))}"
            )
        return solver.solution.x

    def solver_call_explicit_interface(self, qp_data: QPData) -> np.ndarray:
        qp_data_qpalm = QPData()
        A2 = np.eye(len(qp_data.box_upper_constraints))
        if len(qp_data.eq_matrix) > 0:
            A2 = np.vstack((A2, qp_data.eq_matrix))
        if len(qp_data.neq_matrix) > 0:
            A2 = np.vstack((A2, qp_data.neq_matrix))
        qp_data_qpalm.quadratic_weights = qp_data.quadratic_weights
        qp_data_qpalm.linear_weights = qp_data.linear_weights
        qp_data_qpalm.neq_matrix = A2
        qp_data_qpalm.neq_lower_bounds = np.concatenate(
            (qp_data.box_lower_constraints, qp_data.eq_bounds, qp_data.neq_lower_bounds)
        )
        qp_data_qpalm.neq_upper_bounds = np.concatenate(
            (qp_data.box_upper_constraints, qp_data.eq_bounds, qp_data.neq_upper_bounds)
        )
        return self.solver_call(qp_data_qpalm)
