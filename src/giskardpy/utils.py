from collections import defaultdict
from math import pi

class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret

# def shortest_angular_distance(from_angle, to_angle):
#     diff = to_angle - from_angle
#     pi2 = 2 * pi
#     diff = ((diff % pi2) + pi2) % pi2
#     if diff > pi:
#         return diff - pi2
#     return diff