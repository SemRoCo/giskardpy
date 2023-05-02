import abc
from collections import defaultdict
from enum import IntEnum
from typing import Tuple, Iterable, List, Union, Optional, Dict

import numpy as np

from giskardpy.configs.data_types import SupportedQPSolver
from giskardpy.exceptions import QPSolverException, InfeasibleException, HardConstraintsViolatedException
from giskardpy.qp.qp_solver import QPSolver, record_solver_call_time
import qpSWIFT
import giskardpy.casadi_wrapper as cas
import scipy.sparse as sp

from giskardpy.utils.decorators import record_time


class QPSWIFTExitFlags(IntEnum):
    Optimal = 0  # Solution Found
    KKT_Failure = 1  # Failure in factorising KKT matrix
    MAX_ITER_REACHED = 2  # Maximum Number of Iterations Reached
    ERROR = 3  # Unknown Problem in Solver


class QPSWIFTFormatter(QPSolver):
    sparse: bool = True

    @profile
    def __init__(self, weights: cas.Expression, g: cas.Expression, lb: cas.Expression, ub: cas.Expression,
                 E: cas.Expression, E_slack: cas.Expression, bE: cas.Expression,
                 A: cas.Expression, A_slack: cas.Expression, lbA: cas.Expression, ubA: cas.Expression):
        """
        min_x 0.5 x^T H x + g^T x
        s.t.  Ex = b
              Ax <= lb/ub
        """
        self.num_eq_constraints = bE.shape[0]
        self.num_neq_constraints = lbA.shape[0]
        self.num_free_variable_constraints = lb.shape[0]
        self.num_eq_slack_variables = E_slack.shape[1]
        self.num_neq_slack_variables = A_slack.shape[1]
        self.num_slack_variables = self.num_eq_slack_variables + self.num_neq_slack_variables
        self.num_non_slack_variables = self.num_free_variable_constraints - self.num_slack_variables

        self.lb_inf_filter = self.to_inf_filter(lb)
        self.ub_inf_filter = self.to_inf_filter(ub)
        nlb_without_inf = -lb[self.lb_inf_filter]
        ub_without_inf = ub[self.ub_inf_filter]

        self.nlbA_inf_filter = self.to_inf_filter(lbA)
        self.ubA_inf_filter = self.to_inf_filter(ubA)
        nlbA_without_inf = -lbA[self.nlbA_inf_filter]
        ubA_without_inf = ubA[self.ubA_inf_filter]
        nA_without_inf = -A[self.nlbA_inf_filter]
        nA_slack_without_inf = -A_slack[self.nlbA_inf_filter]
        A_without_inf = A[self.ubA_inf_filter]
        A_slack_without_inf = A_slack[self.ubA_inf_filter]
        self.nlbA_ubA_inf_filter = np.concatenate((self.nlbA_inf_filter, self.ubA_inf_filter))
        self.len_lbA = nlbA_without_inf.shape[0]
        self.len_ubA = ubA_without_inf.shape[0]

        combined_E = cas.hstack([E, E_slack, cas.zeros(E_slack.shape[0], A_slack.shape[1])])
        combined_nA = cas.hstack([nA_without_inf,
                                  cas.zeros(nA_slack_without_inf.shape[0], E_slack.shape[1]),
                                  nA_slack_without_inf])
        combined_A = cas.hstack([A_without_inf,
                                 cas.zeros(A_slack_without_inf.shape[0], E_slack.shape[1]),
                                 A_slack_without_inf])
        nA_A = cas.vstack([combined_nA, combined_A])
        nlbA_ubA = cas.vstack([nlbA_without_inf, ubA_without_inf])

        free_symbols = set(weights.free_symbols())
        free_symbols.update(g.free_symbols())
        free_symbols.update(nlb_without_inf.free_symbols())
        free_symbols.update(ub_without_inf.free_symbols())
        free_symbols.update(combined_E.free_symbols())
        free_symbols.update(bE.free_symbols())
        free_symbols.update(nA_A.free_symbols())
        free_symbols.update(nlbA_ubA.free_symbols())
        free_symbols = list(free_symbols)

        self.E_f = combined_E.compile(parameters=free_symbols, sparse=self.sparse)
        self.nA_A_f = nA_A.compile(parameters=free_symbols, sparse=self.sparse)
        self.combined_vector_f = cas.StackedCompiledFunction([weights,
                                                              g,
                                                              nlb_without_inf,
                                                              ub_without_inf,
                                                              bE,
                                                              nlbA_ubA],
                                                             parameters=free_symbols)

        self.free_symbols_str = [str(x) for x in free_symbols]

        if self.compute_nI_I:
            self._nAi_Ai_cache = {}

    @profile
    def evaluate_functions(self, substitutions):
        self.nA_A = self.nA_A_f.fast_call(substitutions)
        self.E = self.E_f.fast_call(substitutions)
        self.weights, self.g, self.nlb, self.ub, self.bE, self.nlbA_ubA = self.combined_vector_f.fast_call(
            substitutions)

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

    @profile
    def relaxed_problem_data_to_qp_format(self) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.retries_with_relaxed_constraints -= 1
        if self.retries_with_relaxed_constraints <= 0:
            raise HardConstraintsViolatedException('Out of retries with relaxed hard constraints.')
        lb_filter, nlb_relaxed, ub_filter, ub_relaxed = self.compute_violated_constraints(self.weights,
                                                                                          self.nA_A,
                                                                                          self.nlb,
                                                                                          self.ub,
                                                                                          self.nlbA_ubA)
        if np.any(lb_filter) or np.any(ub_filter):
            self.weights[ub_filter | lb_filter] *= self.retry_weight_factor
            self.nlb = nlb_relaxed
            self.ub = ub_relaxed
            return self.problem_data_to_qp_format()
        self.retries_with_relaxed_constraints += 1
        raise InfeasibleException('')

    def compute_violated_constraints(self, weights: np.ndarray, nA_A: np.ndarray, nlb: np.ndarray,
                                     ub: np.ndarray, nlbA_ubA: np.ndarray):
        nlb_relaxed = nlb.copy()
        ub_relaxed = ub.copy()
        if self.num_slack_variables > 0:
            lb_non_slack_without_inf = np.where(self.lb_inf_filter[:self.num_non_slack_variables])[0].shape[0]
            ub_non_slack_without_inf = np.where(self.ub_inf_filter[:self.num_non_slack_variables])[0].shape[0]
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
        self.lb_filter = self.lb_inf_filter[self.weight_filter]
        self.ub_filter = self.ub_inf_filter[self.weight_filter]
        lower_violations = 1e-4 < xdot_full[self.lb_filter] - nlb
        upper_violations = 1e-4 < xdot_full[self.ub_filter] - ub
        self.lb_filter[self.lb_filter] = lower_violations
        self.ub_filter[self.ub_filter] = upper_violations
        self.lb_filter[:self.num_non_slack_variables] = False
        self.ub_filter[:self.num_non_slack_variables] = False
        nlb_relaxed[lb_non_slack_without_inf:] -= self.retry_added_slack
        ub_relaxed[ub_non_slack_without_inf:] -= self.retry_added_slack
        nlb_relaxed[lower_violations] += self.retry_added_slack
        ub_relaxed[upper_violations] += self.retry_added_slack
        return self.lb_filter, nlb_relaxed, self.ub_filter, ub_relaxed

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
            self.nlbA_filter_half = self.nlbA_filter_half[self.nlbA_inf_filter]
            self.ubA_filter_half = self.ubA_filter_half[self.ubA_inf_filter]
        self.bA_filter = np.concatenate((self.nlbA_filter_half, self.ubA_filter_half))
        if self.compute_nI_I:
            self.nAi_Ai_filter = np.concatenate((self.lb_inf_filter[self.weight_filter],
                                                 self.ub_inf_filter[self.weight_filter]))

    @profile
    def filter_inf_entries(self):
        self.lb_inf_filter = self.nlb != np.inf
        self.ub_inf_filter = self.ub != np.inf
        self.lbA_ubA_inf_filter = self.nlbA_ubA != np.inf
        self.lb_ub_inf_filter = np.concatenate((self.lb_inf_filter, self.ub_inf_filter))
        self.nAi_Ai = self.nAi_Ai[self.lb_ub_inf_filter]
        self.nlb = self.nlb[self.lb_inf_filter]
        self.ub = self.ub[self.ub_inf_filter]
        self.nlbA_ubA = self.nlbA_ubA[self.lbA_ubA_inf_filter]
        self.nA_A = self.nA_A[self.lbA_ubA_inf_filter]

    @profile
    def apply_filters(self):
        self.weights = self.weights[self.weight_filter]
        self.g = self.g[self.weight_filter]
        self.nlb = self.nlb[self.weight_filter[self.lb_inf_filter]]
        self.ub = self.ub[self.weight_filter[self.ub_inf_filter]]
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

    def lb_ub_with_inf(self, nlb: np.ndarray, ub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lb_with_inf = (np.ones(self.lb_inf_filter.shape) * -np.inf)
        ub_with_inf = (np.ones(self.ub_inf_filter.shape) * np.inf)
        lb_with_inf[self.weight_filter & self.lb_inf_filter] = -nlb
        ub_with_inf[self.weight_filter & self.ub_inf_filter] = ub
        lb_with_inf = lb_with_inf[self.weight_filter]
        ub_with_inf = ub_with_inf[self.weight_filter]
        return lb_with_inf, ub_with_inf

    @profile
    def get_problem_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        :return: weights, g, lb, ub, E, bE, A, lbA, ubA, weight_filter, bE_filter, bA_filter
        """
        weights = self.weights
        g = self.g
        lb, ub = self.lb_ub_with_inf(self.nlb, self.ub)

        num_nA_rows = np.where(self.nlbA_filter_half)[0].shape[0]
        # num_A_rows = np.where(self.ubA_filter_half)[0].shape[0]
        nA = self.nA_A[:num_nA_rows]
        A = self.nA_A[num_nA_rows:]
        merged_A = np.zeros((self.ubA_inf_filter.shape[0], self.weights.shape[0]))
        nlbA_filter = self.nlbA_inf_filter.copy()
        if len(nlbA_filter) > 0:
            nlbA_filter[nlbA_filter] = self.nlbA_filter_half
            if self.sparse:
                merged_A[nlbA_filter] = nA.toarray()
            else:
                merged_A[nlbA_filter] = nA

        ubA_filter = self.ubA_inf_filter.copy()
        if len(ubA_filter) > 0:
            ubA_filter[ubA_filter] = self.ubA_filter_half
            if self.sparse:
                merged_A[ubA_filter] = A.toarray()
            else:
                merged_A[ubA_filter] = A

        lbA = (np.ones(self.nlbA_inf_filter.shape) * -np.inf)
        ubA = (np.ones(self.ubA_inf_filter.shape) * np.inf)
        if len(nlbA_filter) > 0:
            lbA[nlbA_filter] = -self.nlbA_ubA[:num_nA_rows]
        if len(ubA_filter) > 0:
            ubA[ubA_filter] = self.nlbA_ubA[num_nA_rows:]

        bA_filter = np.ones(merged_A.shape[0], dtype=bool)
        if len(self.bA_part) > 0:
            bA_filter[-len(self.bA_part):] = self.bA_part

        if self.sparse:
            E = self.E.toarray()
        else:
            E = self.E
        bE = self.bE

        merged_A = merged_A[bA_filter]
        lbA = lbA[bA_filter]
        ubA = ubA[bA_filter]

        return weights, g, lb, ub, E, bE, merged_A, lbA, ubA, self.weight_filter, self.bE_filter, bA_filter

    @abc.abstractmethod
    def solver_call(self, H: np.ndarray, g: np.ndarray, E: np.ndarray, b: np.ndarray, A: np.ndarray, h: np.ndarray) \
            -> np.ndarray:
        pass


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
        # 'MAXITER': 100, # 0 < MAXITER < 200; default 100
        # 'ABSTOL': 1e-4, # 0 < ABSTOL < 1; default 1e-6
        'RELTOL': 1e-5,  # 0 < RELTOL < 1; default 1e-6
        # 'SIGMA': 1,
        # 'VERBOSE': 1  # 0 = no print; 1 = print
    }

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
        pass

