from __future__ import annotations

import abc
from abc import ABC
from functools import wraps
from time import time
from typing import TYPE_CHECKING

import numpy as np
from giskardpy.middleware import get_middleware
from giskardpy.qp.qp_formatter import QPData
from giskardpy.utils.utils import is_running_in_pytest

if TYPE_CHECKING:
    pass


class QPSolver(ABC):
    @abc.abstractmethod
    def solver_call(self, *args, **kwargs) -> np.ndarray:
        ...

    @abc.abstractmethod
    def solver_call_explicit_interface(self, qp_data: QPData) -> np.ndarray:
        """
        min_x 0.5 x^T H x + g^T x
        s.t.  lb <= x <= ub     (box constraints)
                   Ex <= bE     (equality constraints)
            lbA <= Ax <= ubA    (lower/upper inequality constraints)
        :param H:
        :param g:
        :param lb:
        :param ub:
        :param E:
        :param bE:
        :param A:
        :param lbA:
        :param ubA:
        :return: x
        """
        ...
