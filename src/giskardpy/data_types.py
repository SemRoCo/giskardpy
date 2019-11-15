from collections import OrderedDict, defaultdict

from sortedcontainers import SortedKeyList


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
                 old_key, frame=u'base_footprint'):
        self.position_on_a = position_on_a
        self.position_on_b = position_on_b
        self.contact_distance = contact_distance
        self.contact_normal = contact_normal
        self.min_dist = min_dist
        self.link_a = link_a
        self.body_b = body_b
        self.link_b = link_b
        self.old_key = old_key
        self.frame = frame


class Collisions(object):
    def __init__(self, robot):
        """
        :type robot: giskardpy.symengine_robot.Robot
        """
        self.robot = robot

        # FIXME I'm assuming that self collisions only has collisions for pairs of objects
        #   which results in a list of length 1 always, which is why I don't sort to safe time
        #   I start with a list of default collisions in case there was none
        def f1():
            return [self._default_collision('', '', '')]

        self.self_collisions = defaultdict(f1)

        def f2():
            return SortedKeyList([self._default_collision('', '', '')] * 20,
                                 key=lambda x: x.contact_distance)

        self.external_collision = defaultdict(f2)
        self.all_collisions = set()

    def add(self, key, contact):
        """
        :type key: list
        :type contact: ClosestPointInfo
        :return:
        """
        body_b = key[1]
        movable_joint = self.robot.get_controlled_parent_joint(key[0])
        self.all_collisions.add(contact)

        if body_b == self.robot.get_name():
            # self.self_collisions[key].add(contact)
            # self.self_collisions[key[:-1]].add(contact)
            # self.self_collisions[key[:-2]].add(contact)
            self.self_collisions[key[0], key[2]].insert(0, contact)
            # self.self_collisions[movable_joint].add(contact)
        else:
            # self.external_collision[key].add(contact)
            # self.external_collision[key[:-1]].add(contact)
            # self.external_collision[key[:-2]].add(contact)
            self.external_collision[movable_joint].add(contact)

    def _default_collision(self, link_a, body_b, link_b):
        return ClosestPointInfo([0, 0, 0],
                                [0, 0, 0],
                                100,
                                0,
                                link_a,
                                body_b,
                                link_b,
                                [0, 0, 1],
                                (link_a, body_b, link_b))

    # def get(self, key):
    #     if key in self.external_collision:
    #         return self.external_collision[key]
    #     elif key in self.self_collisions:
    #         return self. self_collisions[key]

    def get_external_collisions(self, joint_name):
        """
        Collisions are saved as a list for each movable robot joint, sorted by contact distance
        :type joint_name: str
        :rtype: ClosestPointInfo
        """
        return self.external_collision[joint_name]

    def get_self_collisions(self, link_a, link_b):
        """
        Make sure that link_a < link_b, the reverse collision is not saved.
        :type link_a: str
        :type link_b: str
        :return:
        :rtype: ClosestPointInfo
        """
        # FIXME maybe check for reverse key?
        return self.self_collisions[link_a, link_b]

    def __contains__(self, item):
        return item in self.self_collisions or item in self.external_collision

    def items(self):
        return self.all_collisions
