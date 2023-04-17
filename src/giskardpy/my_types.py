from __future__ import annotations
import numpy as np
from enum import IntEnum
from typing import Union, Dict

import genpy
from geometry_msgs.msg import PoseStamped, PointStamped, Vector3Stamped, QuaternionStamped


class PrefixName:
    separator = '/'

    def __init__(self, name, prefix):
        self.short_name = name
        self.prefix = prefix
        if prefix:
            self.long_name = f'{self.prefix}{self.separator}{self.short_name}'
        else:
            self.long_name = name

    @classmethod
    def from_string(cls, name: my_string, set_none_if_no_slash: bool = False):
        if isinstance(name, PrefixName):
            return name
        parts = name.split(cls.separator)
        if len(parts) != 2:
            if set_none_if_no_slash:
                return cls(parts[0], None)
            else:
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


class Derivatives(IntEnum):
    position = 0
    velocity = 1
    acceleration = 2
    jerk = 3
    snap = 4
    crackle = 5
    pop = 6

    @classmethod
    def range(cls, start: Derivatives, stop: Derivatives, step: int = 1):
        """
        Includes stop!
        """
        return [item for item in cls if start <= item <= stop][::step]


number = Union[int, float, np.number]
my_string = Union[str, PrefixName]
goal_parameter = Union[my_string, float, bool, genpy.Message, dict, list, IntEnum, None]
derivative_map = Dict[Derivatives, float]
derivative_joint_map = Dict[Derivatives, Dict[my_string, float]]
transformable_message = Union[PoseStamped, PointStamped, Vector3Stamped, QuaternionStamped]
