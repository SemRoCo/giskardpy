from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from enum import IntEnum
from typing import Optional, Generic, TypeVar, Dict, Union

import genpy
import numpy as np
from geometry_msgs.msg import PoseStamped, PointStamped, Vector3Stamped, QuaternionStamped
from sensor_msgs.msg import JointState


class PrefixName:
    primary_separator = '/'
    secondary_separator = '_'

    def __init__(self, name: str, prefix: Optional[Union[str, PrefixName]]):
        if isinstance(prefix, PrefixName):
            self.prefix = prefix.prefix
            old_suffix = prefix.short_name
            self.short_name = f'{old_suffix}{self.secondary_separator}{name}'
        else:
            self.short_name = name
            self.prefix = prefix
        if prefix:
            self.long_name = f'{self.prefix}{self.primary_separator}{self.short_name}'
        else:
            self.long_name = name

    def __len__(self) -> int:
        return len(self.short_name) + len(self.long_name)

    def __getitem__(self, item) -> str:
        return str(self)[item]

    @classmethod
    def from_string(cls, name: my_string, set_none_if_no_slash: bool = False):
        if isinstance(name, PrefixName):
            return name
        parts = name.split(cls.primary_separator)
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

    def encode(self, param: str):
        return self.long_name.__str__().encode(param)


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


my_string = Union[str, PrefixName]
goal_parameter = Union[my_string, float, bool, genpy.Message, dict, list, IntEnum, None]
derivative_map = Dict[Derivatives, float]
derivative_joint_map = Dict[Derivatives, Dict[my_string, float]]
transformable_message = Union[PoseStamped, PointStamped, Vector3Stamped, QuaternionStamped]


class KeyDefaultDict(defaultdict):
    """
    A default dict where the key is passed as parameter to the factory function.
    """

    def __missing__(self, key, cache=True):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            if cache:
                ret = self[key] = self.default_factory(key)
                return ret
            else:
                return self.default_factory(key)


class FIFOSet(set):
    def __init__(self, data, max_length=None):
        if len(data) > max_length:
            raise ValueError('len(data) > max_length')
        super(FIFOSet, self).__init__(data)
        self.max_length = max_length
        self._data_queue = deque(data)

    def add(self, item):
        if len(self._data_queue) == self.max_length:
            to_delete = self._data_queue.popleft()
            super(FIFOSet, self).remove(to_delete)
            self._data_queue.append(item)
        super(FIFOSet, self).add(item)

    def remove(self, item):
        self.remove(item)
        self._data_queue.remove(item)


class _JointState:
    def __init__(self,
                 position: float = 0,
                 velocity: float = 0,
                 acceleration: float = 0,
                 jerk: float = 0,
                 snap: float = 0,
                 crackle: float = 0,
                 pop: float = 0):
        self.state: np.ndarray = [position, velocity, acceleration, jerk, snap, crackle, pop]

    def __getitem__(self, derivative):
        return self.state[derivative]

    def __setitem__(self, derivative, value):
        self.state[derivative] = value

    @property
    def position(self) -> float:
        return self.state[Derivatives.position]

    @position.setter
    def position(self, value: float):
        self.state[Derivatives.position] = value

    @property
    def velocity(self) -> float:
        return self.state[Derivatives.velocity]

    @velocity.setter
    def velocity(self, value: float):
        self.state[Derivatives.velocity] = value

    @property
    def acceleration(self) -> float:
        return self.state[Derivatives.acceleration]

    @acceleration.setter
    def acceleration(self, value: float):
        self.state[Derivatives.acceleration] = value

    @property
    def jerk(self) -> float:
        return self.state[Derivatives.jerk]

    @jerk.setter
    def jerk(self, value: float):
        self.state[Derivatives.jerk] = value

    @property
    def snap(self) -> float:
        return self.state[Derivatives.snap]

    @snap.setter
    def snap(self, value: float):
        self.state[Derivatives.snap] = value

    @property
    def crackle(self) -> float:
        return self.state[Derivatives.crackle]

    @crackle.setter
    def crackle(self, value: float):
        self.state[Derivatives.crackle] = value

    @property
    def pop(self) -> float:
        return self.state[Derivatives.pop]

    @pop.setter
    def pop(self, value: float):
        self.state[Derivatives.pop] = value

    def set_derivative(self, d: Derivatives, item: float):
        self.state[d] = item

    def __str__(self):
        return f'{self.position}'

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, memodict=None):
        return _JointState(*self.state)


K = TypeVar('K', bound=PrefixName)
V = TypeVar('V', bound=_JointState)


class JointStates(defaultdict, Dict[K, V], Generic[K, V]):
    def __init__(self, *args, **kwargs):
        super().__init__(_JointState, *args, **kwargs)

    def __setitem__(self, key: PrefixName, value: JointState):
        if not isinstance(key, PrefixName):
            if isinstance(key, str):
                key = PrefixName.from_string(key, set_none_if_no_slash=True)
            else:
                raise KeyError(f'{key} is not of type {PrefixName}')
        super().__setitem__(key, value)

    @classmethod
    def from_msg(cls, msg: JointState, prefix: Optional[str] = None) -> JointStates:
        self = cls()
        for i, joint_name in enumerate(msg.name):
            joint_name = PrefixName(joint_name, prefix)
            sjs = _JointState(position=msg.position[i],
                              velocity=0,
                              acceleration=0,
                              jerk=0,
                              snap=0,
                              crackle=0,
                              pop=0)
            self[joint_name] = sjs
        return self

    def __deepcopy__(self, memodict={}):
        new_js = JointStates()
        for joint_name, joint_state in self.items():
            new_js[joint_name] = deepcopy(joint_state)
        return new_js

    def to_position_dict(self) -> Dict[PrefixName, float]:
        return {k: v.position for k, v in self.items()}

    def pretty_print(self):
        for joint_name, joint_state in sorted(self.items()):
            print(f'{joint_name}:')
            print(f'\tposition: {joint_state.position}')
            print(f'\tvelocity: {joint_state.velocity}')
            print(f'\tacceleration: {joint_state.acceleration}')
            print(f'\tjerk: {joint_state.jerk}')


class BiDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse[value] = key

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key)
        super().__setitem__(key, value)
        self.inverse[value] = key

    def __delitem__(self, key):
        self.inverse.setdefault(self[key], []).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]:
            del self.inverse[self[key]]
        super().__delitem__(key)


class TaskState(IntEnum):
    not_started = 0
    running = 1
    on_hold = 2
    succeeded = 3
    failed = 4
