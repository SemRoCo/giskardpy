from collections import defaultdict, deque
from copy import deepcopy

from sensor_msgs.msg import JointState


class KeyDefaultDict(defaultdict):
    """
    A default dict where the key is passed as parameter to the factory function.
    """

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


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


derivative_to_name = {
    0: 'position',
    1: 'velocity',
    2: 'acceleration',
    3: 'jerk',
    4: 'snap',
    5: 'crackle',
    6: 'pop',
}


class _JointState(object):
    def __init__(self, position=0, velocity=0, acceleration=0, jerk=0, snap=0, crackle=0, pop=0):
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.jerk = jerk
        self.snap = snap
        self.crackle = crackle
        self.pop = pop

    def set_derivative(self, d, item):
        setattr(self, derivative_to_name[d], item)

    def __str__(self):
        return '{}'.format(self.position)

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, memodict={}):
        return _JointState(self.position, self.velocity, self.acceleration, self.jerk, self.snap, self.crackle,
                           self.pop)


class JointStates(defaultdict):
    def __init__(self, *args, **kwargs):
        super().__init__(_JointState, *args, **kwargs)

    @classmethod
    def from_msg(cls, msg, prefix=None):
        """
        :type msg: JointState
        :rtype: JointStates
        """
        self = cls()
        for i, joint_name in enumerate(msg.name):
            joint_name = PrefixName(joint_name, prefix)
            sjs = _JointState(position=msg.position[i],
                              velocity=msg.velocity[i] if msg.velocity else 0,
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


class BiDict(dict):
    # TODO test me
    def __init__(self, *args, **kwargs):
        super(BiDict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse[value] = key

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key)
        super(BiDict, self).__setitem__(key, value)
        self.inverse[value] = key

    def __delitem__(self, key):
        self.inverse.setdefault(self[key], []).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]:
            del self.inverse[self[key]]
        super(BiDict, self).__delitem__(key)


class PrefixName(object):
    def __init__(self, name, prefix, separator='/'):
        self.short_name = name
        self.prefix = prefix
        self.separator = separator
        if prefix:
            self.long_name = '{}{}{}'.format(self.prefix, self.separator, self.short_name)
        else:
            self.long_name = name

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


order_map = BiDict({
    0: 'position',
    1: 'velocity',
    2: 'acceleration',
    3: 'jerk',
    4: 'snap',
    5: 'crackle',
    6: 'pop'
})
