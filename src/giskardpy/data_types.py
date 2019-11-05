from collections import OrderedDict, defaultdict
from itertools import chain

import numpy as np

from giskardpy.tfwrapper import to_np


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
        self.self_collisions = defaultdict(list)
        self.external_collision = defaultdict(list)
        self.all_collisions = set()
        # self.key_to_key = defaultdict(set)

    def add(self, key, contact):
        """
        :type key: list
        :type contact: ClosestPointInfo
        :return:
        """
        body_b = key[1]
        movable_joint = self.robot.get_movable_parent_joint(key[0])
        self.all_collisions.add(contact)

        if body_b == self.robot.get_name():
            self.self_collisions[key].append(contact)
            self.self_collisions[key[:-1]].append(contact)
            self.self_collisions[key[:-2]].append(contact)
            self.self_collisions[movable_joint].append(contact)
        else:
            self.external_collision[key].append(contact)
            self.external_collision[key[:-1]].append(contact)
            self.external_collision[key[:-2]].append(contact)
            self.external_collision[movable_joint].append(contact)

        # self.data[key].append(contact)
        # self.key_to_key[(key[0],)].add(key)
        # self.key_to_key[key[0], key[1]].add(key)

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

    def get(self, link_a, body_b=None, link_b=None):

        if body_b is not None and link_b is not None:
            r = self.external_collision[(link_a, body_b, link_b)]
        elif body_b is not None and link_b is None:
            r = []
            for k in self.key_to_key[(link_a, body_b)]:
                r.extend(self.external_collision[k])
        else:
            r = []
            for k in self.key_to_key[(link_a,)]:
                r.extend(self.external_collision[k])
        if len(r) == 0:
            return [self._default_collision(link_a, body_b, link_b)]
        return list(sorted(r, key=lambda x: x.contact_distance))

    def get_external_collisions(self, joint_name):
        collisions = self.external_collision[joint_name]
        if collisions:
            r = collisions
        else:
            link_a = self.robot.get_child_link_of_joint(joint_name)
            r = [self._default_collision(link_a, None, None)]
        # collisions = self.get(link_a)
        # return [x for x in collisions if x.body_b != robot_name]
        return list(sorted(r, key=lambda x: x.contact_distance))

    def __getitem__(self, item):
        return self.get(*item)

    def __contains__(self, item):
        return item in self.self_collisions or item in self.external_collision

    def items(self):
        # return chain(self.external_collision)
        # return self.data.items()
        return self.all_collisions
