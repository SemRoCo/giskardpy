from collections import OrderedDict
from typing import List, Dict

import numpy as np

from giskardpy import identifier
from giskardpy.god_map import GodMap
from giskardpy.my_types import Derivatives, PrefixName
from giskardpy.qp.free_variable import FreeVariable
import giskardpy.utils.math as giskard_math

class NextCommands:
    def __init__(self, free_variables: List[FreeVariable], xdot: np.ndarray, max_derivative: Derivatives,
                 prediction_horizon: int):
        self.free_variable_data: Dict[PrefixName, List[float]] = {}
        offset = len(free_variables)
        self.xdot_velocity = xdot[:offset]
        x = prediction_horizon - max_derivative + 1
        derivative_offset = {d: offset*((d-1)*x+giskard_math.gauss(d-2)) for d in Derivatives.range(Derivatives.velocity, max_derivative)}
        joint_derivative_filter = np.array([int(derivative_offset[derivative])
                                            for derivative in Derivatives.range(Derivatives.velocity, max_derivative)])
        # for derivative in Derivatives.range(Derivatives.velocity, max_derivative):
        for i, free_variable in enumerate(free_variables):
            try:
                self.free_variable_data[free_variable.name] = xdot[joint_derivative_filter + i]
            except IndexError as e:
                pass
                # OrderedDict((x.name, xdot[i + offset * prediction_horizon * (derivative - 1)]) for i, x in
                #                                      enumerate(free_variables))
            # _JointState
            # split[Derivatives(derivative)] = OrderedDict((x.position_name,
            #                                               xdot[i + offset * self.prediction_horizon * (derivative - 1)])
            #                                              for i, x in enumerate(self.free_variables))
