import abc
from abc import ABC
from collections import defaultdict
from functools import wraps
from time import time
from typing import Tuple, List, Iterable, Optional, Sequence, Union, Dict
import scipy.sparse as sp
import numpy as np

import giskardpy.casadi_wrapper as cas
from giskardpy.configs.data_types import SupportedQPSolver
from giskardpy.exceptions import HardConstraintsViolatedException, InfeasibleException, QPSolverException
from giskardpy.utils import logging
from giskardpy.utils.decorators import memoize


def record_solver_call_time(function):
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
                 E: cas.Expression, E_slack: cas.Expression, bE: cas.Expression):
        pass

    @classmethod
    def get_solver_times(self) -> dict:
        if hasattr(self, '_times'):
            return self._times
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
    def to_inf_filter(casadi_array):
        # FIXME, buggy if a function happens to evaluate with all 0 input
        if casadi_array.shape[0] == 0:
            return np.eye(0)
        compiled = casadi_array.compile()
        inf_filter = np.isfinite(compiled.fast_call(np.zeros(len(compiled.str_params))))
        return inf_filter

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
                   ubA=cas.Expression())
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
