from enum import IntEnum
from typing import Tuple

import numpy as np
import qpalm

import giskardpy.casadi_wrapper as cas
from giskardpy.data_types.exceptions import QPSolverException, InfeasibleException, HardConstraintsViolatedException
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
    """
    min_x 0.5 x^T Q x + q^T x
    s.t.  lb <= Ax <= ub
    https://github.com/kul-optec/QPALM
    """
    sparse = True
    compute_nI_I = True
    settings = qpalm.Settings()
    settings.verbose = False
    settings.eps_abs = 3e-5
    settings.eps_rel = 1e-8
    settings.nonconvex = False

    # settings.max_iter = 100

    
    def __init__(self, weights: cas.Expression, g: cas.Expression, lb: cas.Expression, ub: cas.Expression,
                 E: cas.Expression, E_slack: cas.Expression, bE: cas.Expression,
                 A: cas.Expression, A_slack: cas.Expression, lbA: cas.Expression, ubA: cas.Expression):
        """
        min_x 0.5 x^T H x + g^T x
        s.t.  lb <= Ax <= ub
        combined matrix format:
        """
        self.set_density(weights, E, E_slack, A, A_slack)
        self.num_eq_constraints = bE.shape[0]
        self.num_neq_constraints = lbA.shape[0]
        self.num_free_variable_constraints = lb.shape[0]
        self.num_eq_slack_variables = E_slack.shape[1]
        self.num_neq_slack_variables = A_slack.shape[1]
        self.num_slack_variables = self.num_eq_slack_variables + self.num_neq_slack_variables
        self.num_non_slack_variables = self.num_free_variable_constraints - self.num_slack_variables

        if len(A) == 0:
            combined_A = E = A = cas.hstack([E, E_slack])
        else:
            A = cas.hstack([A, cas.zeros(A.shape[0], E_slack.shape[1]), A_slack])
            E = cas.hstack([E, E_slack, cas.zeros(E.shape[0], A_slack.shape[1])])
            combined_A = cas.vstack([E, A])

        free_symbols = set(weights.free_symbols())
        free_symbols.update(combined_A.free_symbols())
        free_symbols.update(lb.free_symbols())
        free_symbols.update(ub.free_symbols())
        free_symbols.update(bE.free_symbols())
        free_symbols.update(lbA.free_symbols())
        free_symbols.update(ubA.free_symbols())
        free_symbols = list(free_symbols)
        self.free_symbols = free_symbols

        len_lb_be_lba_end = weights.shape[0] + lb.shape[0] + bE.shape[0] + lbA.shape[0]
        len_ub_be_uba_end = len_lb_be_lba_end + ub.shape[0] + bE.shape[0] + ubA.shape[0]

        self.combined_vector_f = cas.StackedCompiledFunction(expressions=[weights, lb, bE, lbA, ub, bE, ubA, g],
                                                             parameters=free_symbols,
                                                             additional_views=[
                                                                 slice(weights.shape[0], len_lb_be_lba_end),
                                                                 slice(len_lb_be_lba_end, len_ub_be_uba_end)])

        self.A_f = combined_A.compile(parameters=free_symbols, sparse=self.sparse)

        self.free_symbols_str = [str(x) for x in free_symbols]

        if self.compute_nI_I:
            self._nAi_Ai_cache = {}

    
    def evaluate_functions(self, substitutions: np.ndarray):

        self.weights, self.lb, self.bE, self.lbA, self.ub, _, self.ubA, self.g, self.lb_bE_lbA, self.ub_bE_ubA = self.combined_vector_f.fast_call(
            substitutions)
        self.A = self.A_f.fast_call(substitutions)

    
    def update_filters(self):
        self.weight_filter = self.weights != 0
        self.weight_filter[:-self.num_slack_variables] = True
        self.slack_part = self.weight_filter[-(self.num_eq_slack_variables + self.num_neq_slack_variables):]
        self.bE_part = self.slack_part[:self.num_eq_slack_variables]
        self.bA_part = self.slack_part[self.num_eq_slack_variables:]

        self.b_bE_bA_filter = np.ones(self.lb.shape[0] + self.bE.shape[0] + self.lbA.shape[0], dtype=bool)
        self.b_zero_inf_filter_view = self.b_bE_bA_filter[:self.lb.shape[0]]
        self.bE_filter_view = self.b_bE_bA_filter[self.lb.shape[0]:self.lb.shape[0] + self.bE.shape[0]]
        self.bA_filter_view = self.b_bE_bA_filter[self.lb.shape[0] + self.bE.shape[0]:]
        self.bE_bA_filter = self.b_bE_bA_filter[self.lb.shape[0]:]

        self.b_zero_filter = self.weight_filter.copy()
        self.b_finite_filter = np.isfinite(self.lb) | np.isfinite(self.ub)
        self.b_zero_inf_filter_view[::] = self.b_zero_filter & self.b_finite_filter
        self.Ai_inf_filter = self.b_finite_filter[self.b_zero_filter]

        if len(self.bE_part) > 0:
            self.bE_filter_view[-len(self.bE_part):] = self.bE_part

        if len(self.bA_part) > 0:
            self.bA_filter_view[-len(self.bA_part):] = self.bA_part

    
    def apply_filters(self):
        self.weights = self.weights[self.weight_filter]
        self.g = self.g[self.weight_filter]
        self.lb_bE_lbA = self.lb_bE_lbA[self.b_bE_bA_filter]
        self.ub_bE_ubA = self.ub_bE_ubA[self.b_bE_bA_filter]
        self.A = self.A[:, self.weight_filter][self.bE_bA_filter, :]
        if self.compute_nI_I:
            # for constraints, both rows and columns are filtered, so I can start with weights dims
            # then only the rows need to be filtered for inf lb/ub
            self.Ai = self._direct_limit_model(self.weights.shape[0], self.Ai_inf_filter)

    
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
        if solver.info.status_val != QPALMInfo.SOLVED:
            raise InfeasibleException(f'Failed to solve qp: {str(QPALMInfo(solver.info.status_val))}')
        return solver.solution.x

    def default_interface_solver_call(self, H, g, lb, ub, E, bE, A, lbA, ubA) -> np.ndarray:
        A2 = np.eye(len(ub))
        if len(E) > 0:
            A2 = np.vstack((A2, E))
        if len(A) > 0:
            A2 = np.vstack((A2, A))
        lbA = np.concatenate((lb, bE, lbA))
        ubA = np.concatenate((ub, bE, ubA))
        return self.solver_call(H, g, A2, lbA, ubA)

    
    def relaxed_problem_data_to_qp_format(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.retries_with_relaxed_constraints -= 1
        if self.retries_with_relaxed_constraints <= 0:
            raise HardConstraintsViolatedException('Out of retries with relaxed hard constraints.')
        lb_filter, lbA_relaxed, ub_filter, ubA_relaxed, relaxed_qp_data = self.compute_violated_constraints(
            self.lb[self.b_zero_inf_filter_view],
            self.ub[self.b_zero_inf_filter_view])
        if np.any(lb_filter) or np.any(ub_filter):
            self.weights[ub_filter | lb_filter] *= self.retry_weight_factor
            self.lb_bE_lbA = lbA_relaxed
            self.ub_bE_ubA = ubA_relaxed
            return self.problem_data_to_qp_format()
        # relaxing makes it solvable, without actually violating any of the old constraints
        return relaxed_qp_data
        # self.retries_with_relaxed_constraints += 1
        # raise InfeasibleException('')

    
    def compute_violated_constraints(self, filtered_lb: np.ndarray, filtered_ub: np.ndarray):
        lb_relaxed = filtered_lb.copy()
        ub_relaxed = filtered_ub.copy()
        slack_finite_filter = self.b_finite_filter[-(self.num_eq_slack_variables + self.num_neq_slack_variables):]
        num_constraints_not_filtered = np.where(self.slack_part[slack_finite_filter])[0].shape[0]
        lb_relaxed[-num_constraints_not_filtered:] -= self.retry_added_slack
        ub_relaxed[-num_constraints_not_filtered:] += self.retry_added_slack
        try:
            H, g, A, lbA, ubA = self.problem_data_to_qp_format()
            all_lbA_relaxed = np.concatenate([lb_relaxed, lbA[lb_relaxed.shape[0]:]])
            all_ubA_relaxed = np.concatenate([ub_relaxed, ubA[ub_relaxed.shape[0]:]])
            xdot_full = self.solver_call(H=H, g=g, A=A, lbA=all_lbA_relaxed, ubA=all_ubA_relaxed)
        except QPSolverException as e:
            self.retries_with_relaxed_constraints += 1
            raise e
        eps = 1e-4
        lower_violations = xdot_full < self.lb[self.b_zero_filter] - eps
        upper_violations = xdot_full > self.ub[self.b_zero_filter] + eps
        lower_violations[:-num_constraints_not_filtered] = False
        upper_violations[:-num_constraints_not_filtered] = False
        # revert previous relaxations
        lb_relaxed = filtered_lb.copy()
        ub_relaxed = filtered_ub.copy()
        # add relaxations only to constraints that where violated
        lb_relaxed[lower_violations[self.b_finite_filter[self.b_zero_filter]]] -= self.retry_added_slack
        ub_relaxed[upper_violations[self.b_finite_filter[self.b_zero_filter]]] += self.retry_added_slack
        lbA_relaxed = all_lbA_relaxed.copy()
        ubA_relaxed = all_ubA_relaxed.copy()
        lbA_relaxed[:lb_relaxed.shape[0]] = lb_relaxed
        ubA_relaxed[:ub_relaxed.shape[0]] = ub_relaxed
        return lower_violations, lbA_relaxed, upper_violations, ubA_relaxed, (H, g, A, all_lbA_relaxed, all_ubA_relaxed)

    
    def problem_data_to_qp_format(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        from scipy import sparse as sp
        H = sp.diags(self.weights)
        A = sp.vstack((self.Ai, self.A))
        return H, self.g, A, self.lb_bE_lbA, self.ub_bE_ubA

    def lb_ub_with_inf(self, nlb: np.ndarray, ub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lb_with_inf = (np.ones(self.b_finite_filter.shape) * -np.inf)
        ub_with_inf = (np.ones(self.b_finite_filter.shape) * np.inf)
        lb_with_inf[self.weight_filter & self.b_finite_filter] = -nlb
        ub_with_inf[self.weight_filter & self.b_finite_filter] = ub
        lb_with_inf = lb_with_inf[self.weight_filter]
        ub_with_inf = ub_with_inf[self.weight_filter]
        return lb_with_inf, ub_with_inf

    
    def get_problem_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        :return: weights, g, lb, ub, E, bE, A, lbA, ubA, weight_filter, b_filter, bE_filter, bA_filter
        """
        num_b = np.where(self.b_zero_inf_filter_view)[0].shape[0]
        num_bE = np.where(self.bE_filter_view)[0].shape[0]
        weights = self.weights
        g = self.g
        lb = self.lb_bE_lbA[:num_b]
        ub = self.ub_bE_ubA[:num_b]
        lb, ub = self.lb_ub_with_inf(lb, ub)
        lb = -lb
        bE = self.lb_bE_lbA[num_b:num_b + num_bE]
        lbA = self.lb_bE_lbA[num_b + num_bE:]
        ubA = self.ub_bE_ubA[num_b + num_bE:]

        E = self.A[:num_bE]
        A = self.A[num_bE:]
        if self.sparse:
            E = E.toarray()
            A = A.toarray()

        return weights, g, lb, ub, E, bE, A, lbA, ubA, self.weight_filter, self.bE_filter_view, self.bA_filter_view

    def compute_H_A_E_density(self) -> Tuple[float, float, float]:
        H, _, E, _, A, _ = self.problem_data_to_qp_format()
        h_count, a_count, e_count = H.count_nonzero(), A.count_nonzero(), E.count_nonzero()
        return h_count / (H.shape[0] * H.shape[1]), (a_count - (h_count * 2)) / (A.shape[0] + A.shape[1]), e_count / (
                    E.shape[0] + E.shape[1])
