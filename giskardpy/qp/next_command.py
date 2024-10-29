from __future__ import annotations
from typing import List, Dict, TYPE_CHECKING
import numpy as np


from giskardpy.data_types.data_types import Derivatives, PrefixName
from giskardpy.qp.free_variable import FreeVariable
import giskardpy.utils.math as giskard_math
from giskardpy.utils.decorators import memoize
from line_profiler import profile
if TYPE_CHECKING:
    from model.world import WorldTree


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

    @classmethod
    @profile
    def from_xdot_implicit(cls, free_variables: List[FreeVariable], xdot: np.ndarray, max_derivative: Derivatives,
                  prediction_horizon: int, world: WorldTree, dt: float) -> NextCommands:
        self = cls()
        self.free_variable_data = {}
        offset = len(free_variables)
        last_state = np.array([world.state[v.name].state[1:max_derivative+1] for v in free_variables])
        joint_derivative_filter_ = joint_derivative_filter(offset, prediction_horizon, Derivatives.velocity)
        self.free_variable_data = {
            free_variable.name: [float(xdot[joint_derivative_filter_ + i]),
                                 float(xdot[joint_derivative_filter_ + i])/dt-last_state[i][0]/dt ,
                                 float(xdot[joint_derivative_filter_ + i])/dt**2-last_state[i][0]/dt**2-last_state[i][1]/dt ] for i, free_variable in enumerate(free_variables)
        }
        return self

    @classmethod
    @profile
    def from_xdot_explicit_no_acc(cls, free_variables: List[FreeVariable], xdot: np.ndarray, max_derivative: Derivatives,
                  prediction_horizon: int, world: WorldTree, dt: float) -> NextCommands:
        self = cls()
        self.free_variable_data = {}
        offset = len(free_variables)
        last_state = np.array([world.state[v.name].state[1:max_derivative+1] for v in free_variables])
        joint_derivative_filter_ = joint_derivative_filter(offset, prediction_horizon, Derivatives.jerk)[:2]
        self.free_variable_data = {
            free_variable.name: [float(xdot[joint_derivative_filter_ + i][0]),
                                 float(xdot[joint_derivative_filter_ + i][0])/dt-last_state[i][0]/dt,
                                 float(xdot[joint_derivative_filter_ + i][1])] for i, free_variable in enumerate(free_variables)
        }
        return self
