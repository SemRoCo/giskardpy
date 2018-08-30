from collections import OrderedDict


class SingleJointState(object):
    def __init__(self, name='', position=0.0, velocity=0.0, effort=0.0):
        self.name = name
        self.position = position
        self.velocity = velocity
        self.effort = effort

    def __str__(self):
        return u'{}: {}, {}, {}'.format(self.name, self.position, self.velocity, self.effort)


# class MultiJointState(object):
#     def __init__(self):
#         self._states = OrderedDict()
#
#     def get(self, name):
#         return self._states[name]
#
#     def set(self, state):
#         self._states[state.name] = state
#
#     def keys(self):
#         return self._states.keys()
#
#     def values(self):
#         return self._states.values()
#
#     def items(self):
#         return self._states.items()


class Point(object):
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return u'{}(x={}, y={}, z={})'.format(self.__class__.__name__, self.x, self.y, self.z)


class Quaternion(object):
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def __repr__(self):
        return u'{}(x={}, y={}, z={}, w={})'.format(self.__class__.__name__, self.x, self.y, self.z, self.w)


class Transform(object):
    def __init__(self, translation=Point(), rotation=Quaternion()):
        self.translation = translation
        self.rotation = rotation

    def __repr__(self):
        return u'{}(translation={}, rotation={})'.format(self.__class__.__name__, self.translation, self.rotation)


# class Stamped(object):
#     def __init__(self, reference_name=''):
#         self.reference_name = reference_name


# class PointStamped(Point, Stamped):
#     pass


# class QuaternionStamped(Quaternion, Stamped):
#     pass


# class TransformStamped(Transform, Stamped):
#     pass


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
    def __init__(self, position_on_a, position_on_b, contact_distance, min_dist, link_a, link_b, contact_normal):
        self.position_on_a = position_on_a
        self.position_on_b = position_on_b
        self.contact_distance = contact_distance
        self.contact_normal = contact_normal
        self.min_dist = min_dist
        self.link_a = link_a
        self.link_b = link_b
