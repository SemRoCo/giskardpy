from __future__ import annotations

import abc
from abc import ABC
from typing import TYPE_CHECKING, Type

import numpy as np

from giskardpy.qp.qp_adapter import QPData, GiskardToQPAdapter
from giskardpy.qp.qp_solver_ids import SupportedQPSolver

if TYPE_CHECKING:
    pass


class QPSolver(ABC):
    solver_id: SupportedQPSolver
    required_adapter_type: Type[GiskardToQPAdapter]

    @abc.abstractmethod
    def solver_call(self, qp_data: QPData) -> np.ndarray:
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
