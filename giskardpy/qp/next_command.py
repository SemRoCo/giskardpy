from __future__ import annotations
from typing import List, Dict
import numpy as np


from giskardpy.data_types.data_types import Derivatives, PrefixName
from giskardpy.qp.free_variable import FreeVariable
import giskardpy.utils.math as giskard_math
from giskardpy.utils.decorators import memoize
from line_profiler import profile

@memoize
@profile
def derivative_offset(offset, prediction_horizon, derivative, max_derivative):
    x = prediction_horizon - max_derivative + 1
    return int(offset * ((derivative - 1) * x + giskard_math.gauss(derivative - 2)))


@memoize
@profile
def joint_derivative_filter(offset, prediction_horizon, max_derivative):
    return np.array([derivative_offset(offset, prediction_horizon, derivative, max_derivative)
                     for derivative in Derivatives.range(Derivatives.velocity, max_derivative)])


class NextCommands:
    free_variable_data: Dict[PrefixName, np.ndarray]

    @profile
    def __init__(self):
        self.free_variable_data = {}

    @classmethod
    @profile
    def from_xdot(cls, free_variables: List[FreeVariable], xdot: np.ndarray, max_derivative: Derivatives,
                  prediction_horizon: int) -> NextCommands:
        self = cls()
        self.free_variable_data = {}
        offset = len(free_variables)
        joint_derivative_filter_ = joint_derivative_filter(offset, prediction_horizon, max_derivative)
        self.free_variable_data = {
            free_variable.name: xdot[joint_derivative_filter_ + i]
            for i, free_variable in enumerate(free_variables)
        }
        return self
