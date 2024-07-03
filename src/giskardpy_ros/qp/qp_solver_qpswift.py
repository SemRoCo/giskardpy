import abc
from collections import defaultdict
from enum import IntEnum
from typing import Tuple, Iterable, List, Union, Optional, Dict

import numpy as np

from giskardpy.configs.qp_controller_config import SupportedQPSolver
from giskardpy.exceptions import QPSolverException, InfeasibleException, HardConstraintsViolatedException
from giskardpy.qp.qp_solver import QPSolver, record_solver_call_time, QPSWIFTFormatter
import qpSWIFT
import giskardpy.casadi_wrapper as cas
import scipy.sparse as sp

from giskardpy.utils.decorators import record_time


class QPSWIFTExitFlags(IntEnum):
    Optimal = 0  # Solution Found
    KKT_Failure = 1  # Failure in factorising KKT matrix
    MAX_ITER_REACHED = 2  # Maximum Number of Iterations Reached
    ERROR = 3  # Unknown Problem in Solver


class QPSolverQPSwift(QPSWIFTFormatter):
    """
    min_x 0.5 x^T P x + c^T x
    s.t.  Ax = b
          Gx <= h
    """
    solver_id = SupportedQPSolver.qpSWIFT
    _times: Dict[Tuple[int, int, int], list] = defaultdict(list)

    opts = {
        'OUTPUT': 1,  # 0 = sol; 1 = sol + basicInfo; 2 = sol + basicInfo + advInfo
        # 'MAXITER': 100,  # 0 < MAXITER < 200; default 100. maximum number of iterations needed
        # 'ABSTOL': 9e-4,  # 0 < ABSTOL < 1; default 1e-6. absolute tolerance
        'RELTOL': 3.e-5,  # 0 < RELTOL < 1; default 1e-6. relative tolerance
        # 'SIGMA': 0.01,  # default 100. maximum centering allowed
        # 'VERBOSE': 1  # 0 = no print; 1 = print
    }

    @profile
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

        # self.num_filtered_neq_constraints = np.count_nonzero(np.invert(self.bA_part))
        self.nlbA_filter_half = np.ones(self.num_neq_constraints, dtype=bool)
        self.ubA_filter_half = np.ones(self.num_neq_constraints, dtype=bool)
        if len(self.bA_part) > 0:
            self.nlbA_filter_half[-len(self.bA_part):] = self.bA_part
            self.ubA_filter_half[-len(self.bA_part):] = self.bA_part
            self.nlbA_filter_half = self.nlbA_filter_half[self.nlbA_finite_filter]
            self.ubA_filter_half = self.ubA_filter_half[self.ubA_finite_filter]
        self.bA_filter = np.concatenate((self.nlbA_filter_half, self.ubA_filter_half))

        self.nlb_secondary_finite_filter = np.isfinite(self.nlb)
        self.lb_finite_filter = self.static_lb_finite_filter.copy()
        self.lb_finite_filter[self.lb_finite_filter] = (self.lb_finite_filter[self.lb_finite_filter]
                                                        & self.nlb_secondary_finite_filter)

        self.ub_secondary_finite_filter = np.isfinite(self.ub)
        self.ub_finite_filter = self.static_ub_finite_filter.copy()
        self.ub_finite_filter[self.ub_finite_filter] = (self.ub_finite_filter[self.ub_finite_filter]
                                                        & self.ub_secondary_finite_filter)

        if self.compute_nI_I:
            self.nAi_Ai_filter = np.concatenate((self.lb_finite_filter[self.weight_filter],
                                                 self.ub_finite_filter[self.weight_filter]))

    @profile
    def apply_filters(self):
        self.weights = self.weights[self.weight_filter]
        self.g = self.g[self.weight_filter]
        self.nlb = self.nlb[self.weight_filter[self.static_lb_finite_filter] & self.nlb_secondary_finite_filter]
        self.ub = self.ub[self.weight_filter[self.static_ub_finite_filter] & self.ub_secondary_finite_filter]
        if self.num_filtered_eq_constraints > 0:
            self.E = self.E[self.bE_filter, :][:, self.weight_filter]
        else:
            # when no eq constraints were filtered, we can just cut off at the end, because that section is always all 0
            self.E = self.E[:, :np.count_nonzero(self.weight_filter)]
        self.bE = self.bE[self.bE_filter]
        if len(self.nA_A.shape) > 1 and self.nA_A.shape[0] * self.nA_A.shape[1] > 0:
            self.nA_A = self.nA_A[:, self.weight_filter][self.bA_filter, :]
        self.nlbA_ubA = self.nlbA_ubA[self.bA_filter]
        if self.compute_nI_I:
            # for constraints, both rows and columns are filtered, so I can start with weights dims
            # then only the rows need to be filtered for inf lb/ub
            self.nAi_Ai = self._direct_limit_model(self.weights.shape[0], self.nAi_Ai_filter, True)

    @profile
    def solver_call(self, H: np.ndarray, g: np.ndarray, E: sp.csc_matrix, b: np.ndarray, A: sp.csc_matrix,
                    h: np.ndarray) -> np.ndarray:
        A = A.toarray()
        E = E.toarray()
        result = qpSWIFT.run(c=g, h=h, P=H, G=A, A=E, b=b, opts=self.opts)
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

    def lb_ub_with_inf(self, nlb: np.ndarray, ub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lb_with_inf = (np.ones(self.static_lb_finite_filter.shape) * -np.inf)
        ub_with_inf = (np.ones(self.static_ub_finite_filter.shape) * np.inf)
        combined_filter = self.weight_filter & self.static_lb_finite_filter
        combined_filter[self.static_lb_finite_filter] &= self.nlb_secondary_finite_filter
        lb_with_inf[combined_filter] = -nlb
        lb_with_inf = lb_with_inf[self.weight_filter]

        combined_filter = self.weight_filter & self.static_ub_finite_filter
        combined_filter[self.static_ub_finite_filter] &= self.ub_secondary_finite_filter
        ub_with_inf[combined_filter] = ub
        ub_with_inf = ub_with_inf[self.weight_filter]
        return lb_with_inf, ub_with_inf


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
            H, g, E, bE, A, _ = self.problem_data_to_qp_format()
            nlb_ub_nlbA_ubA = np.concatenate((nlb_relaxed, ub_relaxed, self.nlbA_ubA))
            xdot_full = self.solver_call(H=H, g=g, E=E, b=bE, A=A, h=nlb_ub_nlbA_ubA)
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

    @profile
    def problem_data_to_qp_format(self) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        H = np.diag(self.weights)
        if np.product(self.nA_A.shape) > 0:
            A = sp.vstack((self.nAi_Ai, self.nA_A))
        else:
            A = self.nAi_Ai
        nlb_ub_nlbA_ubA = np.concatenate((self.nlb, self.ub, self.nlbA_ubA))
        return H, self.g, self.E, self.bE, A, nlb_ub_nlbA_ubA