import abc
from abc import ABC
from typing import Tuple

import numpy as np

from giskardpy.exceptions import HardConstraintsViolatedException, InfeasibleException, QPSolverException


class QPSolver(ABC):

    def __init__(self,
                 num_non_slack: int,
                 retry_added_slack: float,
                 retry_weight_factor: float,
                 retries_with_relaxed_constraints: int,
                 on_fail_round_to: int = 4):
        self.num_non_slack = num_non_slack
        self.retry_added_slack = retry_added_slack
        self.retry_weight_factor = retry_weight_factor
        self.retries_with_relaxed_constraints = retries_with_relaxed_constraints
        self.on_fail_round_to = on_fail_round_to

    @abc.abstractmethod
    def solve(self, weights: np.ndarray, g: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray, lbA: np.ndarray,
              ubA: np.ndarray) -> np.ndarray:
        """
        x^T*H*x + x^T*g
        s.t.: lbA < A*x < ubA
        and    lb <  x  < ub
        :param weights: 1d vector, len = (jc (joint constraints) + sc (soft constraints))
        :param g: 1d zero vector of len joint constraints + soft constraints
        :param A: 2d jacobi matrix of hc (hard constraints) and sc, shape = (hc + sc) * (number of joints)
        :param lb: 1d vector containing lower bound of x, len = jc + sc
        :param ub: 1d vector containing upper bound of x, len = js + sc
        :param lbA: 1d vector containing lower bounds for the change of hc and sc, len = hc+sc
        :param ubA: 1d vector containing upper bounds for the change of hc and sc, len = hc+sc
        :return: x according to the equations above, len = joint constraints + soft constraints
        """

    @abc.abstractmethod
    def solve_and_retry(self, weights: np.ndarray, g: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray,
                        lbA: np.ndarray, ubA: np.ndarray) -> np.ndarray:
        """
        Calls solve and retries on exception.
        """

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
