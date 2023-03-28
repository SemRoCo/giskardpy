import abc
from abc import ABC
from typing import Tuple, List, Iterable

import numpy as np

import giskardpy.casadi_wrapper as cas
from giskardpy.configs.data_types import SupportedQPSolver
from giskardpy.exceptions import HardConstraintsViolatedException, InfeasibleException, QPSolverException
from giskardpy.utils import logging


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
                 E: cas.Expression, E_slack: cas.Expression, b: cas.Expression):
        pass

    @profile
    def solve(self, substitutions: List[float]) -> np.ndarray:
        problem_data = self.qp_setup_function.fast_call(substitutions)
        split_problem_data = self.split_results(problem_data)
        # todo filter
        return self.solver_call(*split_problem_data)

    @profile
    def solve_and_retry(self, substitutions: List[float]) -> np.ndarray:
        """
        Calls solve and retries on exception.
        """
        exception = None
        for i in range(2):
            try:
                return self.solve(substitutions)
            except QPSolverException as e:
                exception = e
                # try:
                #     weights, lb, ub = self.compute_relaxed_hard_constraints(weights, g, A, lb, ub, lbA, ubA)
                #     logging.loginfo(f'{e}; retrying with relaxed hard constraints')
                # except InfeasibleException as e2:
                #     if isinstance(e2, HardConstraintsViolatedException):
                #         raise e2
                #     raise e
                continue
        raise exception

    @abc.abstractmethod
    def split_results(self, problem_data: np.ndarray) -> List[np.ndarray]:
        pass

    @abc.abstractmethod
    def solver_call(self, *args, **kwargs) -> np.ndarray:
        pass

    # def __init__(self,
    #              num_non_slack: int,
    #              retry_added_slack: float,
    #              retry_weight_factor: float,
    #              retries_with_relaxed_constraints: int,
    #              on_fail_round_to: int = 4):
    #     self.num_non_slack = num_non_slack
    #     self.retry_added_slack = retry_added_slack
    #     self.retry_weight_factor = retry_weight_factor
    #     self.retries_with_relaxed_constraints = retries_with_relaxed_constraints
    #     self.on_fail_round_to = on_fail_round_to

    def compute_relaxed_hard_constraints(self, weights, g, A, lb, ub, lbA, ubA) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.retries_with_relaxed_constraints -= 1
        if self.retries_with_relaxed_constraints <= 0:
            raise HardConstraintsViolatedException('Out of retries with relaxed hard constraints.')
        num_of_slack = len(lb) - self.num_non_slack
        lb_relaxed = lb.copy()
        ub_relaxed = ub.copy()
        lb_relaxed[-num_of_slack:] = -self.retry_added_slack
        ub_relaxed[-num_of_slack:] = self.retry_added_slack
        try:
            xdot_full = self.solve(weights, g, A, lb_relaxed, ub_relaxed, lbA, ubA)
        except QPSolverException as e:
            self.retries_with_relaxed_constraints += 1
            raise e
        upper_violations = ub < xdot_full
        lower_violations = lb > xdot_full
        if np.any(upper_violations) or np.any(lower_violations):
            weights[upper_violations | lower_violations] *= self.retry_weight_factor
            return weights, lb_relaxed, ub_relaxed
        self.retries_with_relaxed_constraints += 1
        raise InfeasibleException('')


class QPSWIFTFormatter(QPSolver):

    def __init__(self, weights: cas.Expression, g: cas.Expression, lb: cas.Expression, ub: cas.Expression,
                 A: cas.Expression, A_slack: cas.Expression, lbA: cas.Expression, ubA: cas.Expression,
                 E: cas.Expression, E_slack: cas.Expression, b: cas.Expression):
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
        |lb| |     -A        |-lbA |
        |lb| |      A        | ubA |
             |---------------|-----|
        """
        self.num_eq_constraints = b.shape[0]
        self.num_neq_constraints = lbA.shape[0]
        self.num_free_variable_constraints = lb.shape[0]

        empty_E_slack = cas.zeros(self.num_neq_constraints, E_slack.shape[1])
        empty_A_slack = cas.zeros(self.num_eq_constraints, A_slack.shape[1])
        A = cas.hstack([A, empty_E_slack, A_slack])
        E = cas.hstack([E, E_slack, empty_A_slack])

        combined_problem_data = cas.zeros(4 + b.shape[0] + lb.shape[0] * 2, weights.shape[0] + 1)
        combined_problem_data[0, :-1] = weights
        combined_problem_data[1, :-1] = g
        combined_problem_data[2, :-1] = -lb
        combined_problem_data[3, :-1] = ub
        offset = 4
        combined_problem_data[offset:offset + self.num_eq_constraints, :-1] = E
        combined_problem_data[offset:offset + self.num_eq_constraints, -1] = b
        offset += self.num_eq_constraints
        combined_problem_data[offset:offset + self.num_neq_constraints, :-1] = -A
        combined_problem_data[offset:offset + self.num_neq_constraints, -1] = -lbA
        offset += self.num_neq_constraints
        combined_problem_data[offset:, :-1] = A
        combined_problem_data[offset:, -1] = ubA
        self.qp_setup_function = combined_problem_data.compile()

    def split_results(self, combined_problem_data: np.ndarray) -> Iterable[np.ndarray]:
        weights = combined_problem_data[0, :-1]
        g = combined_problem_data[1, :-1]
        nlb = combined_problem_data[2, :-1]
        ub = combined_problem_data[3, :-1]
        E = combined_problem_data[4:self.num_eq_constraints, :-1]
        b = combined_problem_data[4:self.num_eq_constraints, -1]
        A = combined_problem_data[4 + self.num_eq_constraints:, :-1]
        nlb_ub = combined_problem_data[4 + self.num_eq_constraints:, -1]
        H = np.diag(weights)
        I = np.eye(self.num_free_variable_constraints)
        A = np.vstack([-I, I, A])
        nlb_ub_nlbA_ubA = np.concatenate([nlb, ub, nlb_ub])
        return H, g, E, b, A, nlb_ub_nlbA_ubA

    @abc.abstractmethod
    def solver_call(self, H: np.ndarray, g: np.ndarray, E: np.ndarray, b: np.ndarray, A: np.ndarray, h: np.ndarray) \
            -> np.ndarray:
        pass
