from collections import OrderedDict, defaultdict

from sortedcontainers import SortedList


class SingleJointState(object):
    def __init__(self, name='', position=0.0, velocity=0.0, effort=0.0):
        self.name = name
        self.position = position
        self.velocity = velocity
        self.effort = effort

    def __str__(self):
        return u'{}: {}, {}, {}'.format(self.name, self.position, self.velocity, self.effort)


class Trajectory(object):
    def __init__(self):
        self._points = OrderedDict()

    def get_exact(self, time):
        return self._points[time]

    def get_closest(self, time):
        pass

    def get_sub_trajectory(self, start_time, end_time):
        pass

    def set(self, time, point):
        if len(self._points) > 0 and self._points.keys()[-1] > time:
            raise KeyError(u'Cannot append a trajectory point that is before the current end time of the trajectory.')
        self._points[time] = point

    def items(self):
        return self._points.items()

    def keys(self):
        return self._points.keys()

    def values(self):
        return self._points.values()


class ClosestPointInfo(object):
    # TODO why no named tuple?
    def __init__(self, position_on_a, position_on_b, contact_distance, min_dist, link_a, body_b, link_b, contact_normal,
                 old_key):
        self.position_on_a = position_on_a
        self.position_on_b = position_on_b
        self.contact_distance = contact_distance
        self.contact_normal = contact_normal
        self.min_dist = min_dist
        self.link_a = link_a
        self.body_b = body_b
        self.link_b = link_b
        self.old_key = old_key


class Collisions(object):
    def __init__(self):
        self.data = defaultdict(list)
        self.key_to_key = defaultdict(set)

    def add(self, key, contact):
        """
        :type key: list
        :type contact: ClosestPointInfo
        :return:
        """
        self.data[key].append(contact)
        self.key_to_key[(key[0],)].add(key)
        self.key_to_key[key[0], key[1]].add(key)

    def get(self, link_a, body_b=None, link_b=None):

        if body_b is not None and link_b is not None:
            r = self.data[(link_a, body_b, link_b)]
        elif body_b is not None and link_b is None:
            r = []
            for k in self.key_to_key[(link_a, body_b)]:
                r.extend(self.data[k])
        else:
            r = []
            for k in self.key_to_key[(link_a,)]:
                r.extend(self.data[k])
        if len(r) == 0:
            return [ClosestPointInfo((10, 0, 0),
                                     (0, 0, 0),
                                     100,
                                     0.0,
                                     (link_a, body_b, link_b),
                                     '',
                                     '',
                                     (1, 0, 0), (link_a, body_b, link_b))]
        return list(sorted(r, key=lambda x: x.contact_distance))

    def __getitem__(self, item):
        return self.get(*item)

    def __contains__(self, item):
        return item in self.data
