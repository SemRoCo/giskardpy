import abc
from abc import ABC
from typing import Tuple, List, Iterable

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

    @abc.abstractmethod
    def get_problem_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        :return: weights, g, lb, ub, E, E_slack, b, A, A_slack, lbA, ubA
        """


class QPSWIFTFormatter(QPSolver):

    @profile
    def __init__(self, weights: cas.Expression, g: cas.Expression, lb: cas.Expression, ub: cas.Expression,
                 E: cas.Expression, E_slack: cas.Expression, b: cas.Expression,
                 A: cas.Expression, A_slack: cas.Expression, lbA: cas.Expression, ubA: cas.Expression, ):
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
        self.num_eq_constraints = b.shape[0]
        self.num_neq_constraints = lbA.shape[0]
        self.num_free_variable_constraints = lb.shape[0]

        combined_problem_data = cas.zeros(4 + b.shape[0] + lbA.shape[0] * 2, weights.shape[0] + 1)
        self.weights_slice = (0, slice(None, -1))
        self.g_slice = (1, slice(None, -1))
        self.nlb_slice = (2, slice(None, -1))
        self.ub_slice = (3, slice(None, -1))
        offset = 4
        self.E_slice = (slice(offset, offset + self.num_eq_constraints), slice(None, E.shape[1]))
        self.E_slack_slice = (self.E_slice[0], slice(E.shape[1], E.shape[1] + E_slack.shape[1]))
        self.E_E_slack_slice = (self.E_slice[0], slice(None, -1))
        self.b_slice = (self.E_slice[0], -1)
        offset += self.num_eq_constraints

        self.nA_slice = (slice(offset, offset + self.num_neq_constraints), slice(None, A.shape[1]))
        self.nA_slack_slice = (self.nA_slice[0], slice(A.shape[1] + E_slack.shape[1], -1))
        self.nlbA_slice = (self.nA_slice[0], -1)
        offset += self.num_neq_constraints

        self.A_slice = (slice(offset, None), self.nA_slice[1])
        self.A_slack_slice = (self.A_slice[0], self.nA_slack_slice[1])
        self.ubA_slice = (self.A_slice[0], -1)

        self.nA_nA_slack_A_A_slack_slice = (slice(4 + self.num_eq_constraints, None), slice(None, -1))
        self.nlb_ub_slice = (self.nA_nA_slack_A_A_slack_slice[0], -1)

        combined_problem_data[self.weights_slice] = weights
        combined_problem_data[self.g_slice] = g
        combined_problem_data[self.nlb_slice] = -lb
        combined_problem_data[self.ub_slice] = ub
        combined_problem_data[self.E_slice] = E
        combined_problem_data[self.E_slack_slice] = E_slack
        combined_problem_data[self.b_slice] = b
        combined_problem_data[self.nA_slice] = -A
        combined_problem_data[self.nlbA_slice] = -lbA
        combined_problem_data[self.A_slice] = A
        combined_problem_data[self.A_slack_slice] = A_slack
        combined_problem_data[self.ubA_slice] = ubA

        self.qp_setup_function = combined_problem_data.compile()

    @profile
    def split_results(self, combined_problem_data: np.ndarray) -> Iterable[np.ndarray]:
        self.combined_problem_data = combined_problem_data
        self.weights = combined_problem_data[self.weights_slice]
        self.g = combined_problem_data[self.g_slice]
        self.nlb = combined_problem_data[self.nlb_slice]
        self.ub = combined_problem_data[self.ub_slice]
        self.E = combined_problem_data[self.E_E_slack_slice]
        self.b = combined_problem_data[self.b_slice]
        self.nA_A = combined_problem_data[self.nA_nA_slack_A_A_slack_slice]
        self.nlb_ub = combined_problem_data[self.nlb_ub_slice]

        self.H = np.diag(self.weights)
        A_d = self.__direct_limit_model(self.num_free_variable_constraints)
        A = np.concatenate([A_d, self.nA_A])
        nlb_ub_nlbA_ubA = np.concatenate([self.nlb, self.ub, self.nlb_ub])
        return self.H, self.g, self.E, self.b, A, nlb_ub_nlbA_ubA

    @memoize
    def __direct_limit_model(self, dimensions):
        I = np.eye(dimensions)
        return np.concatenate([-I, I])

    @profile
    def get_problem_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        :return: weights, g, lb, ub, E, E_slack, b, A, A_slack, lbA, ubA
        """
        weights = self.weights
        g = self.g
        lb = -self.nlb
        ub = self.ub
        A = self.combined_problem_data[self.A_slice]
        A_slack = self.combined_problem_data[self.A_slack_slice]
        lbA = -self.combined_problem_data[self.nlbA_slice]
        ubA = self.combined_problem_data[self.ubA_slice]
        E = self.combined_problem_data[self.E_slice]
        E_slack = self.combined_problem_data[self.E_slack_slice]
        b = self.b
        return weights, g, lb, ub, E, E_slack, b, A, A_slack, lbA, ubA

    @abc.abstractmethod
    def solver_call(self, H: np.ndarray, g: np.ndarray, E: np.ndarray, b: np.ndarray, A: np.ndarray, h: np.ndarray) \
            -> np.ndarray:
        pass
