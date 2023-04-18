from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from typing import Optional, Dict, List

import numpy as np
from sensor_msgs.msg import JointState

from giskardpy.my_types import PrefixName, Derivatives


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
        self.state: np.ndarray = np.array([position, velocity, acceleration, jerk, snap, crackle, pop], dtype=float)

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


class JointStates(defaultdict):
    def __init__(self, *args, **kwargs):
        super().__init__(_JointState, *args, **kwargs)

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

    def to_position_dict(self):
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
