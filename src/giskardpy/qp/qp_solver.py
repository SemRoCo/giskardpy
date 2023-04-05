import abc
from abc import ABC
from typing import Tuple, List, Iterable, Optional

import numpy as np

import giskardpy.casadi_wrapper as cas
from giskardpy.configs.data_types import SupportedQPSolver
from giskardpy.exceptions import HardConstraintsViolatedException, InfeasibleException, QPSolverException
from giskardpy.utils import logging
from giskardpy.utils.utils import memoize


class QPSolver(ABC):
    solver_id: SupportedQPSolver
    qp_setup_function: cas.CompiledFunction
    # num_non_slack = num_non_slack
    retry_added_slack = 100
    retry_weight_factor = 100
    retries_with_relaxed_constraints = 5

    @abc.abstractmethod
    def __init__(self, weights: cas.Expression, g: cas.Expression, lb: cas.Expression, ub: cas.Expression,
                 A: cas.Expression, A_slack: cas.Expression, lbA: cas.Expression, ubA: cas.Expression,
                 E: cas.Expression, E_slack: cas.Expression, bE: cas.Expression):
        pass

    @profile
    def solve(self, substitutions: List[float], relax_hard_constraints: bool = False) -> np.ndarray:
        problem_data = self.qp_setup_function.fast_call(substitutions)
        split_problem_data = self.split_and_filter_results(problem_data, relax_hard_constraints)
        if relax_hard_constraints:
            try:
                return self.solver_call(*split_problem_data)
            except InfeasibleException as e:
                raise HardConstraintsViolatedException(str(e))
        return self.solver_call(*split_problem_data)

    @profile
    def solve_and_retry(self, substitutions: List[float]) -> np.ndarray:
        """
        Calls solve and retries on exception.
        """
        try:
            return self.solve(substitutions)
        except QPSolverException as e:
            try:
                logging.loginfo(f'{e}; retrying with relaxed hard constraints')
                return self.solve(substitutions, relax_hard_constraints=True)
            except InfeasibleException as e2:
                if isinstance(e2, HardConstraintsViolatedException):
                    raise e2
                raise e

    @abc.abstractmethod
    def split_and_filter_results(self, problem_data: np.ndarray, relax_hard_constraints: bool = False) \
            -> List[np.ndarray]:
        pass

    @abc.abstractmethod
    def solver_call(self, *args, **kwargs) -> np.ndarray:
        pass

    @abc.abstractmethod
    def relaxed_problem_data_to_qp_format(self, *args, **kwargs) -> List[np.ndarray]:
        pass

    @abc.abstractmethod
    def get_problem_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        :return: weights, g, lb, ub, E, bE, A, lbA, ubA, weight_filter, bE_filter, bA_filter
        """


