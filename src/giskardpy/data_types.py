from collections import OrderedDict


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
    #TODO why no named tuple?
    def __init__(self, position_on_a, position_on_b, contact_distance, min_dist, link_a, link_b, contact_normal, old_key):
        self.position_on_a = position_on_a
        self.position_on_b = position_on_b
        self.contact_distance = contact_distance
        self.contact_normal = contact_normal
        self.min_dist = min_dist
        self.link_a = link_a
        self.link_b = link_b
        self.old_key = old_key
