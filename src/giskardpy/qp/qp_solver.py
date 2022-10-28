import abc
from abc import ABC
import numpy as np


class QPSolver(ABC):

    @abc.abstractmethod
    def solve(self, H: np.ndarray, g: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray, lbA: np.ndarray,
              ubA: np.ndarray) -> np.ndarray:
        """
        x^T*H*x + x^T*g
        s.t.: lbA < A*x < ubA
        and    lb <  x  < ub
        :param H: 2d diagonal weight matrix, shape = (jc (joint constraints) + sc (soft constraints)) * (jc + sc)
        :param g: 1d zero vector of len joint constraints + soft constraints
        :param A: 2d jacobi matrix of hc (hard constraints) and sc, shape = (hc + sc) * (number of joints)
        :param lb: 1d vector containing lower bound of x, len = jc + sc
        :param ub: 1d vector containing upper bound of x, len = js + sc
        :param lbA: 1d vector containing lower bounds for the change of hc and sc, len = hc+sc
        :param ubA: 1d vector containing upper bounds for the change of hc and sc, len = hc+sc
        :return: x according to the equations above, len = joint constraints + soft constraints
        """
