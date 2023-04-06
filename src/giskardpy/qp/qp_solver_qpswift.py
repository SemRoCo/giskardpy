import abc
from enum import IntEnum
from typing import Tuple, Iterable

import numpy as np

from giskardpy.configs.data_types import SupportedQPSolver
from giskardpy.exceptions import QPSolverException, InfeasibleException, HardConstraintsViolatedException
from giskardpy.qp.qp_solver import QPSolver
import qpSWIFT
import giskardpy.casadi_wrapper as cas

from giskardpy.utils.utils import record_time, memoize


class QPSWIFTExitFlags(IntEnum):
    Optimal = 0  # Solution Found
    KKT_Failure = 1  # Failure in factorising KKT matrix
    MAX_ITER_REACHED = 2  # Maximum Number of Iterations Reached
    ERROR = 3  # Unknown Problem in Solver


class QPSWIFTFormatter(QPSolver):
    sparse: bool = False

    @profile
    def __init__(self, weights: cas.Expression, g: cas.Expression, lb: cas.Expression, ub: cas.Expression,
                 E: cas.Expression, E_slack: cas.Expression, bE: cas.Expression,
                 A: cas.Expression, A_slack: cas.Expression, lbA: cas.Expression, ubA: cas.Expression):
        """
        min_x 0.5 x^T H x + g^T x
        s.t.  Ex = b
              Ax <= lb/ub
        combined matrix format:
                   |x|          1
             |---------------|-----|
        1    |    weights    |  0  |
        1    |      g        |  0  |
        1    |     -lb       |  0  |
        1    |      ub       |  0  |
        |b|  |      E        |  b  |
        |lbA||     -A        |-lbA |
        |lbA||      A        | ubA |
             |---------------|-----|
        """
        self.num_eq_constraints = bE.shape[0]
        self.num_neq_constraints = lbA.shape[0]
        self.num_free_variable_constraints = lb.shape[0]
        self.num_eq_slack_variables = E_slack.shape[1]
        self.num_neq_slack_variables = A_slack.shape[1]
        self.num_slack_variables = self.num_eq_slack_variables + self.num_neq_slack_variables
        self.num_non_slack_variables = self.num_free_variable_constraints - self.num_slack_variables

        self.lb_inf_filter = self.to_filter(lb)
        self.ub_inf_filter = self.to_filter(ub)
        nlb_without_inf = -lb[self.lb_inf_filter]
        ub_without_inf = ub[self.ub_inf_filter]

        self.nlbA_inf_filter = self.to_filter(lbA)
        self.ubA_inf_filter = self.to_filter(ubA)
        nlbA_without_inf = -lbA[self.nlbA_inf_filter]
        ubA_without_inf = ubA[self.ubA_inf_filter]
        nA_without_inf = -A[self.nlbA_inf_filter]
        nA_slack_without_inf = -A_slack[self.nlbA_inf_filter]
        A_without_inf = A[self.ubA_inf_filter]
        A_slack_without_inf = A_slack[self.ubA_inf_filter]
        self.nlbA_ubA_inf_filter = np.concatenate((self.nlbA_inf_filter, self.ubA_inf_filter))
        self.len_lbA = nlbA_without_inf.shape[0]
        self.len_ubA = ubA_without_inf.shape[0]

        combined_problem_data = cas.zeros(4 + bE.shape[0] + nlbA_without_inf.shape[0] + ubA_without_inf.shape[0],
                                          weights.shape[0] + 1)
        self.weights_slice = (0, slice(None, -1))
        self.g_slice = (1, slice(None, -1))
        self.nlb_slice = (2, slice(None, len(nlb_without_inf)))
        self.ub_slice = (3, slice(None, len(ub_without_inf)))
        offset = 4
        self.E_slice = (slice(offset, offset + self.num_eq_constraints), slice(None, E.shape[1]))
        self.E_slack_slice = (self.E_slice[0], slice(self.num_non_slack_variables, E.shape[1] + E_slack.shape[1]))
        self.E_E_slack_slice = (self.E_slice[0], slice(None, -1))
        self.bE_slice = (self.E_slice[0], -1)
        offset += self.num_eq_constraints

        next_offset = offset + nA_without_inf.shape[0]
        self.nA_slice = (slice(offset, next_offset), slice(None, A.shape[1]))
        self.nA_slack_slice = (self.nA_slice[0], slice(self.num_non_slack_variables + E_slack.shape[1], -1))
        self.nlbA_slice = (self.nA_slice[0], -1)
        offset = next_offset

        self.A_slice = (slice(offset, None), self.nA_slice[1])
        self.A_slack_slice = (self.A_slice[0], self.nA_slack_slice[1])
        self.ubA_slice = (self.A_slice[0], -1)

        self.nA_nA_slack_A_A_slack_slice = (slice(4 + self.num_eq_constraints, None), slice(None, -1))
        self.nlbA_ubA_slice = (self.nA_nA_slack_A_A_slack_slice[0], -1)

        combined_problem_data[self.weights_slice] = weights
        combined_problem_data[self.g_slice] = g
        combined_problem_data[self.nlb_slice] = nlb_without_inf
        combined_problem_data[self.ub_slice] = ub_without_inf
        if self.num_eq_constraints > 0:
            combined_problem_data[self.E_slice] = E
            combined_problem_data[self.E_slack_slice] = E_slack
            combined_problem_data[self.bE_slice] = bE
        if self.num_neq_constraints > 0:
            combined_problem_data[self.nA_slice] = nA_without_inf
            combined_problem_data[self.nA_slack_slice] = nA_slack_without_inf
            combined_problem_data[self.nlbA_slice] = nlbA_without_inf
            combined_problem_data[self.A_slice] = A_without_inf
            combined_problem_data[self.A_slack_slice] = A_slack_without_inf
            combined_problem_data[self.ubA_slice] = ubA_without_inf

        self.qp_setup_function = combined_problem_data.compile(sparse=self.sparse)
        self._nAi_Ai_cache = {}

    @staticmethod
    def to_filter(casadi_array):
        # FIXME, buggy if a function happens to evaluate with all 0 input
        if casadi_array.shape[0] == 0:
            return np.eye(0)
        compiled = casadi_array.compile()
        inf_filter = np.isfinite(compiled.fast_call(np.zeros(len(compiled.str_params))).T[0])
        return inf_filter

    @profile
    def split_and_filter_results(self, combined_problem_data: np.ndarray, relax_hard_constraints: bool = False) \
            -> Iterable[np.ndarray]:
        self.combined_problem_data = combined_problem_data
        self.weights = combined_problem_data[self.weights_slice]
        self.g = combined_problem_data[self.g_slice]
        self.nlb = combined_problem_data[self.nlb_slice]
        self.ub = combined_problem_data[self.ub_slice]
        self.E = combined_problem_data[self.E_E_slack_slice]
        self.bE = combined_problem_data[self.bE_slice]
        self.nA_A = combined_problem_data[self.nA_nA_slack_A_A_slack_slice]
        self.nlbA_ubA = combined_problem_data[self.nlbA_ubA_slice]

        self.update_filters()
        self.apply_filters()

        # self.filter_inf_entries()
        if relax_hard_constraints:
            return self.relaxed_problem_data_to_qp_format(self.weights, self.nA_A, self.nlb, self.ub,
                                                          self.nlbA_ubA)
        return self.problem_data_to_qp_format(self.weights, self.nA_A, self.nlb, self.ub, self.nlbA_ubA)

    @profile
    def problem_data_to_qp_format(self, weights: np.ndarray, nA_A: np.ndarray, nlb: np.ndarray,
                                  ub: np.ndarray, nlbA_ubA: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        H = np.diag(weights)
        A = np.concatenate((self.nAi_Ai, nA_A))
        nlb_ub_nlbA_ubA = np.concatenate((nlb, ub, nlbA_ubA))
        return H, self.g, self.E, self.bE, A, nlb_ub_nlbA_ubA

    @profile
    def relaxed_problem_data_to_qp_format(self, weights: np.ndarray, nA_A: np.ndarray, nlb: np.ndarray,
                                          ub: np.ndarray, nlbA_ubA: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.retries_with_relaxed_constraints -= 1
        if self.retries_with_relaxed_constraints <= 0:
            raise HardConstraintsViolatedException('Out of retries with relaxed hard constraints.')
        lb_filter, nlb_relaxed, ub_filter, ub_relaxed = self.compute_violated_constraints(weights, nA_A, nlb, ub,
                                                                                          nlbA_ubA)
        if np.any(lb_filter) or np.any(ub_filter):
            weights[ub_filter] *= self.retry_weight_factor
            weights[lb_filter] *= self.retry_weight_factor
            return self.problem_data_to_qp_format(weights=weights,
                                                  nA_A=nA_A,
                                                  nlb=nlb_relaxed,
                                                  ub=ub_relaxed,
                                                  nlbA_ubA=nlbA_ubA)
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
            relaxed_problem_data = self.problem_data_to_qp_format(weights=weights,
                                                                  nA_A=nA_A,
                                                                  nlb=nlb_relaxed,
                                                                  ub=ub_relaxed,
                                                                  nlbA_ubA=nlbA_ubA)
            xdot_full = self.solver_call(*relaxed_problem_data)
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
        return self.lb_filter, nlb_relaxed, self.ub_filter, ub_relaxed

    @profile
    def update_filters(self):
        self.weight_filter = self.weights != 0
        self.weight_filter[:-self.num_slack_variables] = True
        slack_part = self.weight_filter[-(self.num_eq_slack_variables + self.num_neq_slack_variables):]
        bE_part = slack_part[:self.num_eq_slack_variables]
        self.bA_part = slack_part[self.num_eq_slack_variables:]

        self.bE_filter = np.ones(self.E.shape[0], dtype=bool)
        if self.sparse:
            self.num_filtered_eq_constraints = bE_part.shape[1] - bE_part.nnz
        else:
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
        self.g = np.zeros(*self.weights.shape)
        self.nlb = self.nlb[self.weight_filter[self.lb_inf_filter]]
        self.ub = self.ub[self.weight_filter[self.ub_inf_filter]]
        if self.num_filtered_eq_constraints > 0:
            self.E = self.E[self.bE_filter, :][:, self.weight_filter]
        else:
            # when no eq constraints were filtered, we can just cut off at the end, because that section is always all 0
            self.E = self.E[:, :np.count_nonzero(self.weight_filter)]
        self.bE = self.bE[self.bE_filter]
        self.nA_A = self.nA_A[:, self.weight_filter][self.bA_filter, :]
        self.nlbA_ubA = self.nlbA_ubA[self.bA_filter]
        # for constraints, both rows and columns are filtered, so I can start with weights dims
        # then only the rows need to be filtered for inf lb/ub
        self.nAi_Ai = self._direct_limit_model(self.weights.shape[0], self.nAi_Ai_filter)

    @profile
    def _direct_limit_model(self, dimensions: int, nAi_Ai_filter: np.ndarray) -> np.ndarray:
        """
        These models are often identical, yet the computation is expensive. Caching to the rescue
        """
        key = hash((dimensions, nAi_Ai_filter.tostring()))
        if key not in self._nAi_Ai_cache:
            nI_I = self._cached_eyes(dimensions)
            self._nAi_Ai_cache[key] = nI_I[nAi_Ai_filter]
        return self._nAi_Ai_cache[key]

    @memoize
    def _cached_eyes(self, dimensions: int) -> np.ndarray:
        I = np.eye(dimensions)
        return np.concatenate([-I, I])

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
        nlbA_filter[nlbA_filter] = self.nlbA_filter_half
        merged_A[nlbA_filter] = nA

        ubA_filter = self.ubA_inf_filter.copy()
        ubA_filter[ubA_filter] = self.ubA_filter_half
        merged_A[ubA_filter] = A

        lbA = (np.ones(self.nlbA_inf_filter.shape) * -np.inf)
        ubA = (np.ones(self.ubA_inf_filter.shape) * np.inf)
        lbA[nlbA_filter] = -self.nlbA_ubA[:num_nA_rows]
        ubA[ubA_filter] = self.nlbA_ubA[num_nA_rows:]

        bA_filter = np.ones(merged_A.shape[0], dtype=bool)
        if len(self.bA_part) > 0:
            bA_filter[-len(self.bA_part):] = self.bA_part

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
    def solver_call(self, H: np.ndarray, g: np.ndarray, E: np.ndarray, b: np.ndarray, A: np.ndarray, h: np.ndarray) \
            -> np.ndarray:
        result = qpSWIFT.run(c=g, h=h, P=H, G=A, A=E, b=b, opts=self.opts)
        exit_flag = result['basicInfo']['ExitFlag']
        if exit_flag != 0:
            error_code = QPSWIFTExitFlags(exit_flag)
            if error_code == QPSWIFTExitFlags.MAX_ITER_REACHED:
                raise InfeasibleException(f'Failed to solve qp: {str(error_code)}')
            raise QPSolverException(f'Failed to solve qp: {str(error_code)}')
        return result['sol']
