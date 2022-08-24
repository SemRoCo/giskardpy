from typing import Union, Dict

import genpy

from giskardpy.data_types import PrefixName
import casadi as ca


goal_parameter = Union[str, float, bool, genpy.Message, dict, list]
my_string = Union[str, PrefixName]
expr_symbol = Union[ca.SX, float]
expr_matrix = ca.SX
derivative_map = Dict[int, float]
derivative_joint_map = Dict[int, Dict[my_string, float]]