from enum import IntEnum
from typing import Tuple, List

import numpy as np
import qpalm
from scipy import sparse as sp

from giskardpy.configs.data_types import SupportedQPSolver
from giskardpy.exceptions import QPSolverException, InfeasibleException, HardConstraintsViolatedException
from giskardpy.qp.qp_solver import QPSolver
from giskardpy.utils import logging

from giskardpy.utils.utils import record_time
import giskardpy.casadi_wrapper as cas


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

    sparse = True
    compute_nI_I = True
    settings = qpalm.Settings()
    settings.verbose = False
    settings.eps_abs = 1e-5
    settings.eps_rel = 1e-8
    settings.nonconvex = False

    # settings.max_iter = 100

    @profile
    def __init__(self, weights: cas.Expression, g: cas.Expression, lb: cas.Expression, ub: cas.Expression,
                 E: cas.Expression, E_slack: cas.Expression, bE: cas.Expression,
                 A: cas.Expression, A_slack: cas.Expression, lbA: cas.Expression, ubA: cas.Expression):
        """
        min_x 0.5 x^T H x + g^T x
        s.t.  lb <= Ax <= ub
        combined matrix format:
        """
        self.num_eq_constraints = bE.shape[0]
        self.num_neq_constraints = lbA.shape[0]
        self.num_free_variable_constraints = lb.shape[0]
        self.num_eq_slack_variables = E_slack.shape[1]
        self.num_neq_slack_variables = A_slack.shape[1]
        self.num_slack_variables = self.num_eq_slack_variables + self.num_neq_slack_variables
        self.num_non_slack_variables = self.num_free_variable_constraints - self.num_slack_variables

        self.len_b = lb.shape[0]
        self.len_bE = bE.shape[0]
        self.len_bA = lbA.shape[0]

        combined_A = cas.vstack([cas.hstack([E, E_slack, cas.zeros(E.shape[0], A_slack.shape[1])]),
                                 cas.hstack([A, cas.zeros(A.shape[0], E_slack.shape[1]), A_slack])])
        lbA = cas.vstack([lb, bE, lbA])
        ubA = cas.vstack([ub, bE, ubA])

        free_symbols = set(weights.free_symbols())
        free_symbols.update(combined_A.free_symbols())
        free_symbols.update(lbA.free_symbols())
        free_symbols.update(ubA.free_symbols())
        free_symbols = list(free_symbols)

        self.weights_f = weights.compile(parameters=free_symbols, sparse=False)
        self.lbA_f = lbA.compile(parameters=free_symbols, sparse=False)
        self.ubA_f = ubA.compile(parameters=free_symbols, sparse=False)
        self.A_f = combined_A.compile(parameters=free_symbols, sparse=self.sparse)

        self.free_symbols_str = [str(x) for x in free_symbols]

        if self.compute_nI_I:
            self._nAi_Ai_cache = {}

    @profile
    def apply_filters(self):
        self.weights = self.weights[self.weight_filter]
        self.g = np.zeros(*self.weights.shape)
        self.lbA = self.lbA[self.lbA_filter]
        self.ubA = self.ubA[self.ubA_filter]
        self.A = self.A[:, self.weight_filter][self.A_height_filter, :]
        if self.compute_nI_I:
            # for constraints, both rows and columns are filtered, so I can start with weights dims
            # then only the rows need to be filtered for inf lb/ub
            self.Ai = self._direct_limit_model(self.weights.shape[0])

    @profile
    def update_zero_filters(self):
        self.weight_filter = self.weights != 0
        self.weight_filter[:-self.num_slack_variables] = True
        slack_part = self.weight_filter[-(self.num_eq_slack_variables + self.num_neq_slack_variables):]
        self.bE_part = slack_part[:self.num_eq_slack_variables]
        self.bA_part = slack_part[self.num_eq_slack_variables:]

        self.lbA_filter = np.ones(self.lbA.shape[0], dtype=bool)
        self.ubA_filter = np.ones(self.ubA.shape[0], dtype=bool)
        # first part in lbA is b, so we need to copy the weight filter here
        self.lbA_filter[:self.len_b] = self.weight_filter
        self.ubA_filter[:self.len_b] = self.weight_filter
        self.len_b_filtered = np.where(self.weight_filter)[0].shape[0]

        # copy bA part at the end of the bE section
        bE_end = self.len_b + self.len_bE
        self.len_bE_filtered = self.len_bE - np.where(np.invert(self.bE_part))[0].shape[0]
        if self.num_eq_slack_variables > 0:
            self.lbA_filter[bE_end - len(self.bE_part):bE_end] = self.bE_part
            self.ubA_filter[bE_end - len(self.bE_part):bE_end] = self.bE_part
        self.len_bA_filtered = self.len_bA - np.where(np.invert(self.bA_part))[0].shape[0]
        if self.num_neq_slack_variables > 0:
            self.lbA_filter[-len(self.bA_part):] = self.bA_part
            self.ubA_filter[-len(self.bA_part):] = self.bA_part
        self.A_height_filter = self.lbA_filter[self.len_b:]

    @profile
    def solver_call(self, H: np.ndarray, g: np.ndarray, A: np.ndarray, lbA: np.ndarray, ubA: np.ndarray) \
            -> np.ndarray:
        data = qpalm.Data(A.shape[1], A.shape[0])

        data.Q = H
        data.q = g
        data.A = A
        data.bmin = lbA
        data.bmax = ubA

        solver = qpalm.Solver(data, self.settings)
        solver.solve()
        # print(f'{solver.info.iter} {solver.info.iter_out}')
        if solver.info.status_val != QPALMInfo.SOLVED:
            raise InfeasibleException(f'Failed to solve qp: {str(QPALMInfo(solver.info.status_val))}')
        return solver.solution.x

    @profile
    def relaxed_problem_data_to_qp_format(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.retries_with_relaxed_constraints -= 1
        if self.retries_with_relaxed_constraints <= 0:
            raise HardConstraintsViolatedException('Out of retries with relaxed hard constraints.')
        lb_filter, lbA_relaxed, ub_filter, ubA_relaxed = self.compute_violated_constraints(self.weights,
                                                                                          self.A,
                                                                                          self.lbA,
                                                                                          self.ubA)
        if np.any(lb_filter) or np.any(ub_filter):
            self.weights[ub_filter] *= self.retry_weight_factor
            self.weights[lb_filter] *= self.retry_weight_factor
            self.lbA = lbA_relaxed
            self.ubA = ubA_relaxed
            return self.problem_data_to_qp_format()
        self.retries_with_relaxed_constraints += 1
        raise InfeasibleException('')

    @profile
    def compute_violated_constraints(self, weights: np.ndarray, A: np.ndarray, lbA: np.ndarray,
                                     ubA: np.ndarray):
        lbA_relaxed = lbA.copy()
        ubA_relaxed = ubA.copy()
        if self.num_slack_variables > 0:
            num_bE_not_filtered = np.where(self.bE_part[self.bE_part])[0].shape[0]
            num_bA_not_filtered = np.where(self.bA_part[self.bA_part])[0].shape[0]
            b_slack_start = self.len_b_filtered - num_bE_not_filtered - num_bA_not_filtered
            if num_bE_not_filtered > 0:
                bE_start = b_slack_start
                bE_end = bE_start + num_bE_not_filtered
                lbA_relaxed[bE_start:bE_end] -= self.retry_added_slack
                ubA_relaxed[bE_start:bE_end] += self.retry_added_slack
            if num_bA_not_filtered > 0:
                bA_start = self.len_b_filtered - num_bA_not_filtered
                bA_end = self.len_b_filtered
                lbA_relaxed[bA_start:bA_end] -= self.retry_added_slack
                ubA_relaxed[bA_start:bA_end] += self.retry_added_slack
        else:
            raise InfeasibleException('Can\'t relax constraints, because there are none.')
        # nlb_relaxed += 0.01
        # ub_relaxed += 0.01
        try:
            H, g, A, lbA, ubA = self.problem_data_to_qp_format()
            xdot_full = self.solver_call(H=H, g=g, A=A, lbA=lbA_relaxed, ubA=ubA_relaxed)
        except QPSolverException as e:
            self.retries_with_relaxed_constraints += 1
            raise e
        # self.lb_filter = self.lb_inf_filter[self.weight_filter]
        # self.ub_filter = self.ub_inf_filter[self.weight_filter]
        eps = 1e-4
        lower_violations = xdot_full < lbA[:self.len_b_filtered] - eps
        upper_violations = xdot_full > ubA[:self.len_b_filtered] + eps
        # self.lb_filter[self.lb_filter] = lower_violations
        # self.ub_filter[self.ub_filter] = upper_violations
        lower_violations[:b_slack_start] = False
        upper_violations[:b_slack_start] = False
        return lower_violations, lbA_relaxed, upper_violations, ubA_relaxed

    @profile
    def problem_data_to_qp_format(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        H = sp.diags(self.weights)
        A = sp.vstack((self.Ai, self.A))
        return H, self.g, A, self.lbA, self.ubA

    @profile
    def evaluate_functions(self, substitutions: np.ndarray):
        self.weights = self.weights_f.fast_call(substitutions)
        self.g = np.zeros(self.weights.shape)
        self.lbA = self.lbA_f.fast_call(substitutions)
        self.ubA = self.ubA_f.fast_call(substitutions)
        self.A = self.A_f.fast_call(substitutions)

    @profile
    def get_problem_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        :return: weights, g, lb, ub, E, bE, A, lbA, ubA, weight_filter, bE_filter, bA_filter
        """
        weights = self.weights
        g = self.g
        lb = self.lbA[:self.len_b_filtered]
        ub = self.ubA[:self.len_b_filtered]
        bE = self.lbA[self.len_b_filtered:self.len_b_filtered+self.len_bE_filtered]
        lbA = self.lbA[self.len_b_filtered+self.len_bE_filtered:]
        ubA = self.ubA[self.len_b_filtered+self.len_bE_filtered:]

        E = self.A[:self.len_bE_filtered]
        A = self.A[self.len_bE_filtered:]
        if self.sparse:
            E = E.toarray()
            A = A.toarray()

        bE_filter = self.lbA_filter[self.len_b:self.len_b + self.len_bE]
        bA_filter = self.lbA_filter[self.len_b + self.len_bE:]

        return weights, g, lb, ub, E, bE, A, lbA, ubA, self.weight_filter, bE_filter, bA_filter
