from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, List

if TYPE_CHECKING:
    import scipy.sparse as sp
from enum import IntEnum
from typing import Tuple

import numpy as np

from giskardpy.data_types.exceptions import QPSolverException, InfeasibleException
from giskardpy.qp.qp_solver import QPSWIFTFormatter, QPVerboseFormat
import qpSWIFT

from giskardpy.qp.qp_solver_ids import SupportedQPSolver


class QPSWIFTExitFlags(IntEnum):
    Optimal = 0  # Solution Found
    KKT_Failure = 1  # Failure in factorising KKT matrix
    MAX_ITER_REACHED = 2  # Maximum Number of Iterations Reached
    ERROR = 3  # Unknown Problem in Solver


@dataclass
class QP:
    H: np.ndarray
    g: np.ndarray
    E: sp.csc_matrix
    b: np.ndarray
    A_box: sp.csc_matrix
    h_box: np.ndarray
    A: sp.csc_matrix
    h: np.ndarray

    def as_qpSWIFT_data(self):
        return self.g, self.h_box, self.h, self.H, self.A_box, self.A, self.E, self.b


class QPSolverQPSwift(QPVerboseFormat):
    """
    min_x 0.5 x^T P x + c^T x
    s.t.  Ax = b
          Gx <= h
    """
    solver_id = SupportedQPSolver.qpSWIFT

    opts = {
        'OUTPUT': 1,  # 0 = sol; 1 = sol + basicInfo; 2 = sol + basicInfo + advInfo
        # 'MAXITER': 100,  # 0 < MAXITER < 200; default 100. maximum number of iterations needed
        # 'ABSTOL': 9e-4,  # 0 < ABSTOL < 1; default 1e-6. absolute tolerance
        'RELTOL': 3.5e-5,  # 0 < RELTOL < 1; default 1e-6. relative tolerance
        # 'SIGMA': 0.01,  # default 100. maximum centering allowed
        # 'VERBOSE': 1  # 0 = no print; 1 = print
    }

    def update_filters(self):
        self.weight_filter = self.weights != 0
        self.weight_filter[:-self.num_slack_variables] = True
        slack_part = self.weight_filter[-(self.num_eq_slack_variables + self.num_neq_slack_variables):]
        bE_part = slack_part[:self.num_eq_slack_variables]
        self.bA_part = slack_part[self.num_eq_slack_variables:]

        self.bE_filter = np.ones(self.E.shape[0], dtype=bool)
        self.num_filtered_eq_constraints = np.count_nonzero(np.invert(bE_part))
        if self.num_filtered_eq_constraints > 0:
            self.bE_filter[-len(bE_part):] = bE_part

        self.bA_filter = np.ones(self.A.shape[0], dtype=bool)
        self.num_filtered_neq_constraints = np.count_nonzero(np.invert(self.bA_part))
        if self.num_filtered_neq_constraints > 0:
            self.bA_filter[-len(self.bA_part):] = self.bA_part
        # if len(self.bA_part) > 0:
        #     l_bA = len(self.bA_part)
        #     lnbA_ubA_filter[self.num_neq_constraints - l_bA:self.num_neq_constraints] = self.bA_part
        #     lnbA_ubA_filter[self.num_neq_constraints * 2 - l_bA:self.num_neq_constraints * 2] = self.bA_part
        #     self.bA_filter = lnbA_ubA_filter[self.nlbA_ubA_finite_filter]
        #     self.nlbA_filter_half = self.bA_filter[:self.nlbA_finite_filter_size]
        #     self.ubA_filter_half = self.bA_filter[self.nlbA_finite_filter_size:]
        # else:
        #     self.bA_filter = np.empty(0, dtype=bool)


    def apply_filters(self):
        self.weights = self.weights[self.weight_filter]
        self.g = self.g[self.weight_filter]
        self.lb = self.lb[self.weight_filter]
        self.ub = self.ub[self.weight_filter]
        if self.num_filtered_eq_constraints > 0:
            self.E = self.E[self.bE_filter, :][:, self.weight_filter]
        else:
            # when no eq constraints were filtered, we can just cut off at the end, because that section is always all 0
            self.E = self.E[:, :self.weight_filter.sum()]
        self.bE = self.bE[self.bE_filter]
        if len(self.A.shape) > 1 and self.A.shape[0] * self.A.shape[1] > 0:
            self.A = self.A[:, self.weight_filter][self.bA_filter, :]
        self.lbA = self.lbA[self.bA_filter]
        self.ubA = self.ubA[self.bA_filter]

    def solver_call(self, H, g, lb, ub, E, bE, A, lbA, ubA) -> np.ndarray:
        import scipy.sparse as sp
        result = qpSWIFT.solve_sparse(sp.diags(H), g, lb, ub, E, bE, A, lbA, ubA, options=self.opts)
        exit_flag = result.exit_flag
        if exit_flag != 0:
            error_code = QPSWIFTExitFlags(exit_flag)
            if error_code == QPSWIFTExitFlags.MAX_ITER_REACHED:
                raise InfeasibleException(f'Failed to solve qp: {str(error_code)}')
            raise QPSolverException(f'Failed to solve qp: {str(error_code)}')
        return result.x

    def solver_call_batch(self, qps: List[QP]) -> np.ndarray:
        result = qpSWIFT.run_sparse_with_box_constraints_batch(len(qps),
                                                               tuple([qp.as_qpSWIFT_data(self.opts) for qp in qps]))
        exit_flag = result['basicInfo']['ExitFlag']
        if exit_flag != 0:
            error_code = QPSWIFTExitFlags(exit_flag)
            if error_code == QPSWIFTExitFlags.MAX_ITER_REACHED:
                raise InfeasibleException(f'Failed to solve qp: {str(error_code)}')
            raise QPSolverException(f'Failed to solve qp: {str(error_code)}')
        return result['sol']

    def default_interface_solver_call(self, H, g, lb, ub, E, bE, A, lbA, ubA) -> np.ndarray:
        A_lb_ub = np.eye(len(ub))
        if len(A) > 0:
            A_lb_ub = np.vstack((-A_lb_ub, A_lb_ub, -A, A))
            h = np.concatenate((-lb, ub, -lbA, ubA))
        else:
            A_lb_ub = np.vstack((-A_lb_ub, A_lb_ub))
            h = np.concatenate((-lb, ub))
        h_filter = np.isfinite(h)
        h = h[h_filter]
        A_lb_ub = A_lb_ub[h_filter, :]
        bE_filter = np.isfinite(bE)
        E = E[bE_filter, :]
        if len(E) == 0:
            result = qpSWIFT.run(c=g, h=h, P=H, G=A_lb_ub, opts=self.opts)
        else:
            result = qpSWIFT.run(c=g, h=h, P=H, G=A_lb_ub, A=E, b=bE, opts=self.opts)
        exit_flag = result['basicInfo']['ExitFlag']
        if exit_flag != 0:
            error_code = QPSWIFTExitFlags(exit_flag)
            if error_code == QPSWIFTExitFlags.MAX_ITER_REACHED:
                raise InfeasibleException(f'Failed to solve qp: {str(error_code)}')
            raise QPSolverException(f'Failed to solve qp: {str(error_code)}')
        return result['sol']

    def compute_violated_constraints(self, weights: np.ndarray, nA_A: np.ndarray, nlb: np.ndarray,
                                     ub: np.ndarray, nlbA_ubA: np.ndarray):
        nlb_relaxed = nlb.copy()
        ub_relaxed = ub.copy()
        if self.num_slack_variables > 0:
            # increase slack limit for constraints that are not inf or have 0 weight
            lb_filter = self.weight_filter & self.static_lb_finite_filter
            lb_filter[self.static_lb_finite_filter] &= self.nlb_secondary_finite_filter
            lb_filter = lb_filter[:self.num_non_slack_variables]
            ub_filter = self.weight_filter & self.static_ub_finite_filter
            ub_filter[self.static_ub_finite_filter] &= self.ub_secondary_finite_filter
            ub_filter = ub_filter[:self.num_non_slack_variables]
            lb_non_slack_without_inf = np.where(lb_filter)[0].shape[0]
            ub_non_slack_without_inf = np.where(ub_filter)[0].shape[0]
            nlb_relaxed[lb_non_slack_without_inf:] += self.retry_added_slack
            ub_relaxed[ub_non_slack_without_inf:] += self.retry_added_slack
        else:
            raise InfeasibleException('Can\'t relax constraints, because there are none.')
        # nlb_relaxed += 0.01
        # ub_relaxed += 0.01
        try:
            H, g, E, bE, A_box, _, A, _ = self.problem_data_to_qp_format()
            h_box = np.concatenate((nlb_relaxed, ub_relaxed))
            xdot_full = self.solver_call(H=H, g=g, E=E, b=bE, A_box=A_box, h_box=h_box, A=A, h=self.nlbA_ubA)
        except QPSolverException as e:
            self.retries_with_relaxed_constraints += 1
            raise e
        eps = 1e-4
        self.lb_filter = self.static_lb_finite_filter.copy()  # all static finites
        self.lb_filter[self.lb_filter] &= self.nlb_secondary_finite_filter  # all static and dynamic finites
        self.lb_filter = self.lb_filter[self.weight_filter]  # remove 0 weights

        self.ub_filter = self.static_ub_finite_filter.copy()
        self.ub_filter[self.ub_filter] &= self.ub_secondary_finite_filter
        self.ub_filter = self.ub_filter[self.weight_filter]

        # %% check which constraints are violated in general
        lower_violations = xdot_full[self.lb_filter] < - nlb - eps
        upper_violations = xdot_full[self.ub_filter] > ub + eps
        self.lb_filter[self.lb_filter] = lower_violations
        self.ub_filter[self.ub_filter] = upper_violations
        # revert previous relaxations
        nlb_relaxed[lb_non_slack_without_inf:] -= self.retry_added_slack
        ub_relaxed[ub_non_slack_without_inf:] -= self.retry_added_slack
        # add relaxations only to constraints that where violated
        nlb_relaxed[lower_violations] += self.retry_added_slack
        ub_relaxed[upper_violations] += self.retry_added_slack
        return self.lb_filter, nlb_relaxed, self.ub_filter, ub_relaxed

    def problem_data_to_qp_format(self):
        # nlb_ub = np.concatenate((self.nlb, self.ub))
        # if np.prod(self.nA_A.shape) == 0:
        #     return self.weights, self.g, self.E, self.bE, self.nAi_Ai, nlb_ub, None, None
        # return self.weights, self.g, self.E, self.bE, self.nAi_Ai, nlb_ub, self.nA_A, self.nlbA_ubA
        return self.weights, self.g, self.lb, self.ub, self.E, self.bE, self.A, self.lbA, self.ubA