from __future__ import annotations

from collections import defaultdict, deque, OrderedDict
from copy import deepcopy
from enum import IntEnum, Enum
from typing import Optional, Generic, TypeVar, Dict, Union, List

import numpy as np
import giskardpy.casadi_wrapper as cas
from collections.abc import MutableMapping


class PrefixName:
    primary_separator = '/'
    secondary_separator = '_'

    def __init__(self, name: str, prefix: Optional[Union[str, PrefixName]] = None):
        prefix = prefix or ''
        if isinstance(prefix, PrefixName):
            self.prefix = prefix.long_name
        else:
            self.prefix = prefix
        self.short_name = name
        if self.prefix:
            self.long_name = f'{self.prefix}{self.primary_separator}{self.short_name}'
        else:
            self.long_name = name

    def __len__(self) -> int:
        return len(self.short_name) + len(self.long_name)

    def __getitem__(self, item) -> str:
        return str(self)[item]

    @classmethod
    def from_string(cls, name: Union[str, PrefixName], set_none_if_no_slash: bool = False):
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


class ColorRGBA:
    _r: float
    _g: float
    _b: float
    _a: float

    def __init__(self, r: float, g: float, b: float, a: float):
        self.r, self.g, self.b, self.a = float(r), float(g), float(b), float(a)

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, value):
        self._r = float(value)

    @property
    def g(self):
        return self._g

    @g.setter
    def g(self, value):
        self._g = float(value)

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        self._b = float(value)

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        self._a = float(value)


class Derivatives(IntEnum):
    position = 0
    velocity = 1
    acceleration = 2
    jerk = 3

    @classmethod
    def range(cls, start: Union[Derivatives, int], stop: Union[Derivatives, int], step: int = 1):
        """
        Includes stop!
        """
        return [item for item in cls if start <= item <= stop][::step]


my_string = Union[str, PrefixName]
goal_parameter = Union[str, float, bool, dict, list, IntEnum, None]
derivative_map = Dict[Derivatives, float]


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


class JointStateView:
    def __init__(self, data: np.ndarray):
        self.data = data

    def __getitem__(self, item: Derivatives) -> float:
        return self.data[item]

    def __setitem__(self, key: Derivatives, value: float) -> None:
        self.data[key] = value

    @property
    def position(self) -> float:
        return self.data[Derivatives.position]

    @position.setter
    def position(self, value: float):
        self.data[Derivatives.position] = value

    @property
    def velocity(self) -> float:
        return self.data[Derivatives.velocity]

    @velocity.setter
    def velocity(self, value: float):
        self.data[Derivatives.velocity] = value

    @property
    def acceleration(self) -> float:
        return self.data[Derivatives.acceleration]

    @acceleration.setter
    def acceleration(self, value: float):
        self.data[Derivatives.acceleration] = value

    @property
    def jerk(self) -> float:
        return self.data[Derivatives.jerk]

    @jerk.setter
    def jerk(self, value: float):
        self.data[Derivatives.jerk] = value


class JointStates(MutableMapping):
    # 4 rows (pos, vel, acc, jerk), columns are joints
    data: np.ndarray

    # list of joint names in column order
    _names: List[PrefixName]

    # maps joint_name -> column index
    _index: Dict[PrefixName, int]

    def __init__(self):
        self.data = np.zeros((4, 0), dtype=float)
        self._names = []
        self._index = {}

    def _add_joint(self, name: PrefixName) -> None:
        idx = len(self._names)
        self._names.append(name)
        self._index[name] = idx
        # append a zero column
        new_col = np.zeros((4, 1), dtype=float)
        if self.data.shape[1] == 0:
            self.data = new_col
        else:
            self.data = np.hstack((self.data, new_col))

    def __getitem__(self, name: PrefixName) -> JointStateView:
        if name not in self._index:
            self._add_joint(name)
        idx = self._index[name]
        # return the column view (shape (4,))
        return JointStateView(self.data[:, idx])

    def __setitem__(self, name: PrefixName, value: np.ndarray) -> None:
        arr = np.asarray(value, dtype=float)
        if arr.shape != (4,):
            raise ValueError(f"Value for '{name}' must be length-4 array (pos, vel, acc, jerk).")
        if name not in self._index:
            self._add_joint(name)
        idx = self._index[name]
        self.data[:, idx] = arr

    def __delitem__(self, name: PrefixName) -> None:
        if name not in self._index:
            raise KeyError(name)
        idx = self._index.pop(name)
        self._names.pop(idx)
        # remove column from data
        self.data = np.delete(self.data, idx, axis=1)
        # rebuild indices
        for i, nm in enumerate(self._names):
            self._index[nm] = i

    def __iter__(self) -> iter:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def keys(self) -> List[PrefixName]:
        return self._names

    def items(self) -> List[tuple[PrefixName, np.ndarray]]:
        return [(name, self.data[:, self._index[name]].copy()) for name in self._names]

    def values(self) -> List[np.ndarray]:
        return [self.data[:, self._index[name]].copy() for name in self._names]

    def __contains__(self, name: PrefixName) -> bool:
        return name in self._index

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({{ " + ", ".join(f"{n}: {list(self.data[:, i])}"
                                                            for i, n in enumerate(self._names)) + " })"

    def to_position_dict(self) -> Dict[PrefixName, float]:
        return {joint_name: self[joint_name].position for joint_name in self._names}

    @property
    def positions(self) -> np.ndarray:
        return self.data[0, :]

    @property
    def velocities(self) -> np.ndarray:
        return self.data[1, :]

    @property
    def accelerations(self) -> np.ndarray:
        return self.data[2, :]

    @property
    def jerks(self) -> np.ndarray:
        return self.data[3, :]


class ExecutionMode(IntEnum):
    Execute = 1
    Projection = 2


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


class LifeCycleState(IntEnum):
    not_started = 0
    running = 1
    paused = 2
    succeeded = 3
    failed = 4


class ObservationState:
    false = cas.TrinaryFalse
    unknown = cas.TrinaryUnknown
    true = cas.TrinaryTrue
