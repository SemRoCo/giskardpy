from typing import Union, Dict

import genpy

from giskardpy.data_types import PrefixName
import giskardpy.casadi_wrapper as w


goal_parameter = Union[str, float, bool, genpy.Message, dict]
my_string = Union[str, PrefixName]
expr_symbol = w.ca.SX
expr_matrix = w.ca.SX
derivative_joint_map = Dict[int, Dict[my_string, float]]