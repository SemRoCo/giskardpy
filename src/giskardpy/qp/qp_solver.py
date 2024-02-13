import abc
from abc import ABC
from collections import defaultdict
from functools import wraps
from time import time
from typing import Tuple, List, Iterable, Optional, Sequence, Union, Dict
import scipy.sparse as sp
import numpy as np

import giskardpy.casadi_wrapper as cas
from giskardpy.configs.qp_controller_config import SupportedQPSolver
from giskardpy.exceptions import HardConstraintsViolatedException, InfeasibleException, QPSolverException
from giskardpy.utils import logging
from giskardpy.utils.decorators import memoize
from giskardpy.god_map import god_map
from visualization_msgs.msg import MarkerArray, Marker
import rospy

from giskardpy.utils.utils import is_running_in_pytest


def record_solver_call_time(function):
    if not is_running_in_pytest():
        return function

    @wraps(function)
    def wrapper(*args, **kwargs):
        self: QPSolver = args[0]
        start_time = time()
        result = function(*args, **kwargs)
        time_delta = time() - start_time
        if not ('relax_hard_constraints' in kwargs and kwargs['relax_hard_constraints']):
            key = (len(self.weights),
                   self.num_free_variable_constraints,
                   self.num_eq_constraints,
                   self.num_neq_constraints,
                   self.num_eq_slack_variables,
                   self.num_neq_slack_variables,
                   self.num_slack_variables)
            self._times[key].append(time_delta)
        else:
            logging.loginfo('skipped record time because hard constraints were violated')
        return result

    return wrapper


class QPSolver(ABC):
    free_symbols_str: List[str]
    free_symbols: List[cas.ca.SX]
    solver_id: SupportedQPSolver
    qp_setup_function: cas.CompiledFunction
    # num_non_slack = num_non_slack
    retry_added_slack = 100
    retry_weight_factor = 100
    retries_with_relaxed_constraints = 5
    _nAi_Ai_cache: dict = {}
    sparse: bool = False
    compute_nI_I: bool = True
    num_eq_constraints: int
    num_neq_constraints: int
    num_free_variable_constraints: int
    _times: Dict[Tuple[int, int, int, int], list]

    @abc.abstractmethod
    def __init__(self, weights: cas.Expression, g: cas.Expression, lb: cas.Expression, ub: cas.Expression,
                 A: cas.Expression, A_slack: cas.Expression, lbA: cas.Expression, ubA: cas.Expression,
                 E: cas.Expression, E_slack: cas.Expression, bE: cas.Expression, constraint_jacobian: cas.Expression,
                 grad_traces: [cas.Expression]):
        pass

    @classmethod
    def get_solver_times(cls) -> dict:
        if hasattr(cls, '_times'):
            return cls._times
        return {}

    def analyze_infeasibility(self):
        pass

    @record_solver_call_time
    @profile
    def solve(self, substitutions: np.ndarray, relax_hard_constraints: bool = False) -> np.ndarray:
        self.evaluate_functions(substitutions)
        self.update_filters()
        self.apply_filters()

        if relax_hard_constraints:
            problem_data = self.relaxed_problem_data_to_qp_format()
            try:
                return self.solver_call(*problem_data)
            except InfeasibleException as e:
                raise HardConstraintsViolatedException(str(e))
        else:
            problem_data = self.problem_data_to_qp_format()
            return self.solver_call(*problem_data)

    @staticmethod
    def to_finite_filter(casadi_array: cas.Expression) -> np.ndarray:
        if casadi_array.shape[0] == 0:
            return np.eye(0)
        finite_filter = []
        for i in range(len(casadi_array)):
            expr = casadi_array[i]
            if cas.is_constant(expr):
                finite_filter.append(np.isfinite(casadi_array[i].evaluate()))
            else:
                finite_filter.append(True)
        finite_filter = np.array(finite_filter)
        return finite_filter

    @abc.abstractmethod
    def apply_filters(self):
        pass

    @profile
    def solve_and_retry(self, substitutions: np.ndarray) -> np.ndarray:
        """
        Calls solve and retries on exception.
        """
        try:
            return self.solve(substitutions)
        except QPSolverException as e:
            try:
                logging.loginfo(f'{e}; retrying with relaxed constraints.')
                return self.solve(substitutions, relax_hard_constraints=True)
            except InfeasibleException as e2:
                logging.loginfo('Failed to relax constraints.')
                if isinstance(e2, HardConstraintsViolatedException):
                    raise e2
                raise e

    @abc.abstractmethod
    def update_filters(self):
        pass

    @profile
    def _direct_limit_model(self, dimensions_after_zero_filter: int,
                            Ai_inf_filter: Optional[np.ndarray] = None, nAi_Ai: bool = False) \
            -> Union[np.ndarray, sp.csc_matrix]:
        """
        These models are often identical, yet the computation is expensive. Caching to the rescue
        """
        if Ai_inf_filter is None:
            key = hash(dimensions_after_zero_filter)
        else:
            key = hash((dimensions_after_zero_filter, Ai_inf_filter.tostring()))
        if key not in self._nAi_Ai_cache:
            nI_I = self._cached_eyes(dimensions_after_zero_filter, nAi_Ai)
            if Ai_inf_filter is None:
                self._nAi_Ai_cache[key] = nI_I
            else:
                self._nAi_Ai_cache[key] = nI_I[Ai_inf_filter]
        return self._nAi_Ai_cache[key]

    @memoize
    def _cached_eyes(self, dimensions: int, nAi_Ai: bool = False) -> Union[np.ndarray, sp.csc_matrix]:
        if self.sparse:
            if nAi_Ai:
                d2 = dimensions * 2
                data = np.ones(d2, dtype=float)
                data[::2] *= -1
                r1 = np.arange(dimensions)
                r2 = np.arange(dimensions, d2)
                row_indices = np.empty((d2,), dtype=int)
                row_indices[0::2] = r1
                row_indices[1::2] = r2
                col_indices = np.arange(0, d2 + 1, 2)
                return sp.csc_matrix((data, row_indices, col_indices))
            else:
                data = np.ones(dimensions, dtype=float)
                row_indices = np.arange(dimensions)
                col_indices = np.arange(dimensions + 1)
                return sp.csc_matrix((data, row_indices, col_indices))
        else:
            I = np.eye(dimensions)
            if nAi_Ai:
                return np.concatenate([-I, I])
            else:
                return I

    @classmethod
    def empty(cls):
        self = cls(weights=cas.Expression(),
                   g=cas.Expression(),
                   lb=cas.Expression(),
                   ub=cas.Expression(),
                   E=cas.Expression(),
                   E_slack=cas.Expression(),
                   bE=cas.Expression(),
                   A=cas.Expression(),
                   A_slack=cas.Expression(),
                   lbA=cas.Expression(),
                   ubA=cas.Expression(),
                   constraint_jacobian=cas.Expression(),
                   grad_traces=cas.Expression())
        return self

    @abc.abstractmethod
    def solver_call(self, *args, **kwargs) -> np.ndarray:
        pass

    @abc.abstractmethod
    def default_interface_solver_call(self, H, g, lb, ub, E, bE, A, lbA, ubA) -> np.ndarray:
        pass

    @abc.abstractmethod
    def relaxed_problem_data_to_qp_format(self) -> List[np.ndarray]:
        pass

    @abc.abstractmethod
    def problem_data_to_qp_format(self) -> List[np.ndarray]:
        pass

    @abc.abstractmethod
    def evaluate_functions(self, substitutions: np.ndarray):
        pass

    @abc.abstractmethod
    def get_problem_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        :return: weights, g, lb, ub, E, bE, A, lbA, ubA, weight_filter, bE_filter, bA_filter
        """

    @profile
    def init_manipulability_variables(self, constraint_jacobian, grad_traces):
        self.use_manipulability = False
        if constraint_jacobian:
            self.JJT_f = constraint_jacobian.dot(constraint_jacobian.T).compile(self.free_symbols)
            self.pred_horizon = god_map.manip_constraints[list(god_map.manip_constraints)[0]].prediction_horizon
            self.use_manipulability = True
            self.manip_gain = god_map.manip_constraints[list(god_map.manip_constraints)[0]].gain
            self.det_symbol = cas.Symbol('det')
            self.m_symbolic = cas.sqrt(self.det_symbol)
            self.grad_traces_augmented = cas.vstack([cas.vstack(grad_traces)] * self.pred_horizon) * -self.manip_gain

    @profile
    def calc_det_of_jjt_manipulability(self, substitutions):
        if self.use_manipulability:
            self.JJT = self.JJT_f.fast_call(substitutions)
            self.det = np.linalg.det(self.JJT)


class QPSWIFTFormatter(QPSolver):
    sparse: bool = True

    @profile
    def __init__(self, weights: cas.Expression, g: cas.Expression, lb: cas.Expression, ub: cas.Expression,
                 E: cas.Expression, E_slack: cas.Expression, bE: cas.Expression,
                 A: cas.Expression, A_slack: cas.Expression, lbA: cas.Expression, ubA: cas.Expression,
                 constraint_jacobian: cas.Expression, grad_traces: [cas.Expression]):
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

        self.static_lb_finite_filter = self.to_finite_filter(lb)
        self.static_ub_finite_filter = self.to_finite_filter(ub)
        nlb_without_inf = -lb[self.static_lb_finite_filter]
        ub_without_inf = ub[self.static_ub_finite_filter]

        self.nlbA_finite_filter = self.to_finite_filter(lbA)
        self.ubA_finite_filter = self.to_finite_filter(ubA)
        nlbA_without_inf = -lbA[self.nlbA_finite_filter]
        ubA_without_inf = ubA[self.ubA_finite_filter]
        nA_without_inf = -A[self.nlbA_finite_filter]
        nA_slack_without_inf = -A_slack[self.nlbA_finite_filter]
        A_without_inf = A[self.ubA_finite_filter]
        A_slack_without_inf = A_slack[self.ubA_finite_filter]
        self.nlbA_ubA_finite_filter = np.concatenate((self.nlbA_finite_filter, self.ubA_finite_filter))
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
        self.free_symbols = list(free_symbols)

        self.E_f = combined_E.compile(parameters=self.free_symbols, sparse=self.sparse)
        self.nA_A_f = nA_A.compile(parameters=self.free_symbols, sparse=self.sparse)

        self.init_manipulability_variables(constraint_jacobian, grad_traces)

        if self.use_manipulability:
            self.combined_vector_f = cas.StackedCompiledFunction([weights,
                                                                  g,
                                                                  nlb_without_inf,
                                                                  ub_without_inf,
                                                                  bE,
                                                                  nlbA_ubA,
                                                                  self.grad_traces_augmented * self.m_symbolic,
                                                                  self.m_symbolic],
                                                                 parameters=self.free_symbols + [self.det_symbol.s])
        else:
            self.combined_vector_f = cas.StackedCompiledFunction([weights,
                                                                  g,
                                                                  nlb_without_inf,
                                                                  ub_without_inf,
                                                                  bE,
                                                                  nlbA_ubA],
                                                                 parameters=self.free_symbols)

        self.free_symbols_str = [str(x) for x in self.free_symbols]

        if self.compute_nI_I:
            self._nAi_Ai_cache = {}

    @profile
    def evaluate_functions(self, substitutions):
        self.nA_A = self.nA_A_f.fast_call(substitutions)
        self.E = self.E_f.fast_call(substitutions)
        if self.use_manipulability:
            self.calc_det_of_jjt_manipulability(substitutions)
            self.weights, self.g, self.nlb, self.ub, self.bE, self.nlbA_ubA, m_grad, m = self.combined_vector_f.fast_call(
                np.append(substitutions, self.det))
            self.g[:len(m_grad)] = m_grad
            god_map.qp_controller.manipulability_indexes[1] = god_map.qp_controller.manipulability_indexes[0]
            god_map.qp_controller.manipulability_indexes[0] = m
        else:
            self.weights, self.g, self.nlb, self.ub, self.bE, self.nlbA_ubA = self.combined_vector_f.fast_call(
                substitutions)

    @abc.abstractmethod
    def compute_violated_constraints(self, *args, **kwargs):
        pass

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
        merged_A = np.zeros((self.ubA_finite_filter.shape[0], self.weights.shape[0]))
        nlbA_filter = self.nlbA_finite_filter.copy()
        if len(nlbA_filter) > 0:
            nlbA_filter[nlbA_filter] = self.nlbA_filter_half
            if self.sparse:
                merged_A[nlbA_filter] = nA.toarray()
            else:
                merged_A[nlbA_filter] = nA

        ubA_filter = self.ubA_finite_filter.copy()
        if len(ubA_filter) > 0:
            ubA_filter[ubA_filter] = self.ubA_filter_half
            if self.sparse:
                merged_A[ubA_filter] = A.toarray()
            else:
                merged_A[ubA_filter] = A

        lbA = (np.ones(self.nlbA_finite_filter.shape) * -np.inf)
        ubA = (np.ones(self.ubA_finite_filter.shape) * np.inf)
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
