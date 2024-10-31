from typing import Tuple

import numpy as np
import scipy.sparse as sp
from giskardpy.data_types.exceptions import QPSolverException, HardConstraintsViolatedException, InfeasibleException
from proxsuite.proxsuite_pywrap_avx2 import proxqp
from line_profiler import profile
from giskardpy.qp.qp_solver import QPSolver
from giskardpy.qp.qp_solver_ids import SupportedQPSolver
from giskardpy.utils.decorators import record_time
import giskardpy.casadi_wrapper as cas

class QPSolverProxsuite(QPSolver):
    solver_id = SupportedQPSolver.proxsuite
    """
    min_x 0.5 x^T H x + g^T x
    s.t.  Ex = b
          lbA <= Ax <= ubA
    """
    sparse = True

    @profile
    def __init__(self, weights: cas.Expression, g: cas.Expression, lb: cas.Expression, ub: cas.Expression,
                 E: cas.Expression, E_slack: cas.Expression, bE: cas.Expression,
                 A: cas.Expression, A_slack: cas.Expression, lbA: cas.Expression, ubA: cas.Expression):
        self.num_eq_constraints = bE.shape[0]
        self.num_neq_constraints = lbA.shape[0]
        self.num_free_variable_constraints = lb.shape[0]
        self.num_eq_slack_variables = E_slack.shape[1]
        self.num_neq_slack_variables = A_slack.shape[1]
        self.num_slack_variables = self.num_eq_slack_variables + self.num_neq_slack_variables
        self.num_non_slack_variables = self.num_free_variable_constraints - self.num_slack_variables

        combined_E = cas.hstack([E, E_slack, cas.zeros(E.shape[0], A_slack.shape[1])])
        combined_A = cas.hstack([A, cas.zeros(A.shape[0], E_slack.shape[1]), A_slack])

        free_symbols = set(weights.free_symbols())
        free_symbols.update(combined_E.free_symbols())
        free_symbols.update(combined_A.free_symbols())
        free_symbols.update(lb.free_symbols())
        free_symbols.update(ub.free_symbols())
        free_symbols.update(bE.free_symbols())
        free_symbols.update(lbA.free_symbols())
        free_symbols.update(ubA.free_symbols())
        free_symbols = list(free_symbols)
        self.free_symbols = free_symbols

        b_bA_len = lb.shape[0] + lbA.shape[0]

        self.combined_vector_f = cas.StackedCompiledFunction(expressions=[lb, lbA, weights, g, bE, ub, ubA],
                                                             parameters=free_symbols,
                                                             additional_views=[
                                                                 slice(0, b_bA_len),
                                                                 slice(-b_bA_len, None)])

        self.E_f = combined_E.compile(parameters=free_symbols, sparse=self.sparse)
        self.A_f = combined_A.compile(parameters=free_symbols, sparse=self.sparse)

        self.free_symbols_str = [str(x) for x in free_symbols]

        if self.compute_nI_I:
            self._nAi_Ai_cache = {}

    def default_interface_solver_call(self, H, g, lb, ub, E, bE, A, lbA, ubA) -> np.ndarray:
        A2 = np.eye(len(ub))
        if len(E) > 0:
            A2 = np.vstack((A2, E))
        if len(A) > 0:
            A2 = np.vstack((A2, A))
        lbA = np.concatenate((lb, bE, lbA))
        ubA = np.concatenate((ub, bE, ubA))
        return self.solver_call(H, g, A2, lbA, ubA)

    @profile
    def evaluate_functions(self, substitutions: np.ndarray):
        self.lb, self.lbA, self.weights, self.g, self.bE, self.ub, self.ubA, self.lb_lbA, self.ub_ubA = self.combined_vector_f.fast_call(
            substitutions)
        self.E = self.E_f.fast_call(substitutions)
        self.A = self.A_f.fast_call(substitutions)

    @profile
    def update_filters(self):
        self.weight_filter = self.weights != 0
        self.weight_filter[:-self.num_slack_variables] = True  # filter only slack variables with weight 0
        self.slack_part = self.weight_filter[-(self.num_eq_slack_variables + self.num_neq_slack_variables):]
        self.bE_part = self.slack_part[:self.num_eq_slack_variables]
        self.bA_part = self.slack_part[self.num_eq_slack_variables:]

        self.bE_filter = np.ones(self.E.shape[0], dtype=bool)
        self.num_filtered_eq_constraints = np.count_nonzero(np.invert(self.bE_part))
        if self.num_filtered_eq_constraints > 0:
            self.bE_filter[-len(self.bE_part):] = self.bE_part

        self.b_bA_filter = np.ones(self.lb.shape[0] + self.lbA.shape[0], dtype=bool)
        self.b_zero_inf_filter_view = self.b_bA_filter[:self.lb.shape[0]]
        self.bA_filter = self.b_bA_filter[self.lb.shape[0]:]

        self.b_zero_filter = self.weight_filter.copy()
        self.b_finite_filter = np.isfinite(self.lb) | np.isfinite(self.ub)
        self.b_zero_inf_filter_view[::] = self.b_zero_filter & self.b_finite_filter
        self.Ai_inf_filter = self.b_finite_filter[self.b_zero_filter]

        if len(self.bA_part) > 0:
            self.bA_filter[-len(self.bA_part):] = self.bA_part

    @profile
    def apply_filters(self):
        self.weights = self.weights[self.weight_filter]
        self.g = self.g[self.weight_filter]
        self.bE = self.bE[self.bE_filter]
        self.lb_lbA = self.lb_lbA[self.b_bA_filter]
        self.ub_ubA = self.ub_ubA[self.b_bA_filter]
        if self.num_filtered_eq_constraints > 0:
            self.E = self.E[self.bE_filter, :][:, self.weight_filter]
        else:
            # when no eq constraints were filtered, we can just cut off at the end, because that section is always all 0
            self.E = self.E[:, :np.count_nonzero(self.weight_filter)]
        if len(self.A.shape) > 1 and self.A.shape[0] * self.A.shape[1] > 0:
            self.A = self.A[:, self.weight_filter][self.bA_filter, :]
        if self.compute_nI_I:
            # for constraints, both rows and columns are filtered, so I can start with weights dims
            # then only the rows need to be filtered for inf lb/ub
            self.Ai = self._direct_limit_model(self.weights.shape[0], self.Ai_inf_filter)

    @profile
    def problem_data_to_qp_format(self) -> Tuple[sp.csc_matrix, np.ndarray, sp.csc_matrix, np.ndarray, sp.csc_matrix, np.ndarray, np.ndarray]:
        H = sp.diags(self.weights)
        if np.product(self.A.shape) > 0:
            A = sp.vstack((self.Ai, self.A))
        else:
            A = self.Ai
        return H, self.g, self.E, self.bE, A, self.lb_lbA, self.ub_ubA

    @record_time
    @profile
    def solver_call(self, H: sp.csc_matrix, g: np.ndarray, E: sp.csc_matrix, b: np.ndarray, A: sp.csc_matrix,
                    lbA: np.ndarray, ubA: np.ndarray) -> np.ndarray:
        H = H.toarray()
        A = A.toarray()
        E = E.toarray()
        qp = proxqp.dense.QP(H.shape[0], E.shape[0], A.shape[0])
        qp.init(H, g, E, b, A, lbA, ubA)
        qp.solve()
        if qp.results.info.status.value != 0:
            qp.settings.verbose = True
            qp.solve()
            raise InfeasibleException(f'Failed to solve qp: {qp.results.info.status.name}')
        return qp.results.x

    def lb_ub_with_inf(self, lb: np.ndarray, ub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lb_with_inf = (np.ones(self.b_finite_filter.shape) * -np.inf)
        ub_with_inf = (np.ones(self.b_finite_filter.shape) * np.inf)
        lb_with_inf[self.weight_filter & self.b_finite_filter] = lb
        ub_with_inf[self.weight_filter & self.b_finite_filter] = ub
        lb_with_inf = lb_with_inf[self.weight_filter]
        ub_with_inf = ub_with_inf[self.weight_filter]
        return lb_with_inf, ub_with_inf

    @profile
    def get_problem_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        :return: weights, g, lb, ub, E, bE, A, lbA, ubA, weight_filter, b_filter, bE_filter, bA_filter
        """
        num_b = np.where(self.b_zero_inf_filter_view)[0].shape[0]
        num_bE = np.where(self.bE_filter)[0].shape[0]
        weights = self.weights
        g = self.g
        lb = self.lb_lbA[:num_b]
        ub = self.ub_ubA[:num_b]
        lb, ub = self.lb_ub_with_inf(lb, ub)
        bE = self.bE
        lbA = self.lb_lbA[num_b:]
        ubA = self.ub_ubA[num_b:]

        E = self.E
        A = self.A
        if self.sparse:
            E = E.toarray()
            A = A.toarray()

        return weights, g, lb, ub, E, bE, A, lbA, ubA, self.weight_filter, self.bE_filter, self.bA_filter

    @profile
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

    @profile
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
