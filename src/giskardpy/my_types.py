from typing import Union, Dict

import genpy

from giskardpy.data_types import PrefixName

goal_parameter = Union[str, float, bool, genpy.Message, dict]
my_string = Union[str, PrefixName]
