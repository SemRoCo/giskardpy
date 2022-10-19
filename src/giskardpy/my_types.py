from __future__ import annotations
from typing import Union, Dict
import genpy

import casadi as ca


class PrefixName:
    def __init__(self, name, prefix, separator='/'):
        self.short_name = name
        self.prefix = prefix
        self.separator = separator
        if prefix:
            self.long_name = f'{self.prefix}{self.separator}{self.short_name}'
        else:
            self.long_name = name

    @classmethod
    def from_string(cls, name: my_string):
        if isinstance(name, PrefixName):
            return name
        parts = name.split('/')
        if len(parts):
            raise AttributeError(f'{name} can not be converted to a {str(cls)}.')
        return cls(parts[1], parts[0])

    def __str__(self):
        return self.long_name.__str__()

    def __repr__(self):
        return self.long_name.__repr__()

    def __hash__(self):
        return self.long_name.__hash__()

    def __eq__(self, other):
        return self.long_name.__eq__(other.__str__())

    def __ne__(self, other):
        return self.long_name.__ne__(other.__str__())

    def __le__(self, other):
        return self.long_name.__le__(other.__str__())

    def __ge__(self, other):
        return self.long_name.__ge__(other.__str__())

    def __gt__(self, other):
        return self.long_name.__gt__(other.__str__())

    def __lt__(self, other):
        return self.long_name.__lt__(other.__str__())

    def __contains__(self, item):
        return self.long_name.__contains__(item.__str__())


goal_parameter = Union[str, float, bool, genpy.Message, dict, list]
my_string = Union[str, PrefixName]
expr_symbol = Union[ca.SX, float]
expr_matrix = ca.SX
derivative_map = Dict[int, float]
derivative_joint_map = Dict[int, Dict[my_string, float]]
