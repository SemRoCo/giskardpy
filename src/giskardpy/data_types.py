from collections import OrderedDict, defaultdict

import numpy as np
from sortedcontainers import SortedKeyList

from giskardpy.tfwrapper import to_np, np_vector, np_point


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


class Collision(object):
    # TODO why no named tuple?
    def __init__(self, link_a, body_b, link_b, position_on_a, position_on_b, contact_normal, contact_distance):
        self.__position_on_a = position_on_a
        self.__position_on_b = position_on_b
        self.__contact_distance = contact_distance
        self.__contact_normal = contact_normal
        self.__original_link_a = link_a
        self.__link_a = link_a
        self.__body_b = body_b
        self.__original_link_b = link_b
        self.__link_b = link_b
        self.__old_key = (link_a, body_b, link_a)

    def get_position_on_a_in_map(self):
        return self.__position_on_a

    def get_position_on_a_in_a(self):
        return self.__position_on_a_in_a

    def get_position_on_b_in_map(self):
        return self.__position_on_b

    def get_position_on_b_in_root(self):
        return self.__position_on_b_in_root

    def get_position_on_b_in_b(self):
        return self.__position_on_b_in_b

    def get_contact_normal_in_map(self):
        return self.__contact_normal

    def get_contact_normal_in_b(self):
        return self.__contact_normal_in_b

    def get_contact_normal_in_root(self):
        return self.__contact_normal_in_root

    def get_contact_distance(self):
        return self.__contact_distance

    def get_original_link_a(self):
        return self.__original_link_a

    def get_link_a(self):
        return self.__link_a

    def get_original_link_b(self):
        return self.__original_link_b

    def get_link_b(self):
        return self.__link_b

    def get_body_b(self):
        return self.__body_b

    def set_position_on_a_in_a(self, position):
        self.__position_on_a_in_a = position

    def set_position_on_b_in_root(self, position):
        self.__position_on_b_in_root = position

    def set_position_on_b_in_b(self, position):
        self.__position_on_b_in_b = position

    def set_contact_normal_in_b(self, normal):
        self.__contact_normal_in_b = normal

    def set_contact_normal_in_root(self, normal):
        self.__contact_normal_in_root = normal

    def set_link_a(self, link_a):
        self.__link_a = link_a

    def set_link_b(self, link_b):
        self.__link_b = link_b


class Collisions(object):
    def __init__(self, robot):
        """
        :type robot: giskardpy.symengine_robot.Robot
        """
        self.robot = robot
        self.root_T_map = to_np(self.robot.root_T_map)
        self.robot_root = self.robot.get_root()

        def default_f():
            return SortedKeyList([self._default_collision('', '', '')] * 20,
                                 key=lambda x: x.get_contact_distance())

        self.self_collisions = defaultdict(default_f)
        self.external_collision = defaultdict(default_f)
        self.all_collisions = set()

    def add(self, collision):
        """
        :type collision: Collision
        :return:
        """
        collision = self.transform_closest_point(collision)
        self.all_collisions.add(collision)

        if collision.get_body_b() == self.robot.get_name():
            self.self_collisions[collision.get_link_a(), collision.get_link_b()].add(collision)
        else:
            self.external_collision[collision.get_link_a()].add(collision)

    def transform_closest_point(self, collision):
        """
        :type collision: Collision
        :rtype: Collision
        """
        if collision.get_body_b() == self.robot.get_name():
            return self.transform_self_collision(collision)
        else:
            return self.transform_external_collision(collision)

    def transform_self_collision(self, collision):
        """
        :type collision: Collision
        :rtype: Collision
        """
        link_a = collision.get_original_link_a()
        link_b = collision.get_original_link_b()
        new_link_a, new_link_b = self.robot.get_chain_reduced_to_controlled_joints(link_a, link_b)
        new_b_T_r = self.robot.get_fk_np(new_link_b, self.robot_root)
        new_a_T_r = self.robot.get_fk_np(new_link_a, self.robot_root)
        collision.set_link_a(new_link_a)
        collision.set_link_b(new_link_b)

        new_b_T_map = np.dot(new_b_T_r, self.root_T_map)

        new_a_P_pa = np.dot(np.dot(new_a_T_r, self.root_T_map), np_point(*collision.get_position_on_a_in_map()))
        new_b_P_pb = np.dot(new_b_T_map, np_point(*collision.get_position_on_b_in_map()))
        # r_P_pb = np.dot(self.root_T_map, np_point(*closest_point.position_on_b))
        new_b_V_n = np.dot(new_b_T_map, np_vector(*collision.get_contact_normal_in_map()))
        collision.set_position_on_a_in_a(new_a_P_pa[:-1])
        collision.set_position_on_b_in_b(new_b_P_pb[:-1])
        collision.set_contact_normal_in_b(new_b_V_n[:-1])
        return collision

    def transform_external_collision(self, collision):
        """
        :type collision: Collision
        :rtype: Collision
        """
        movable_joint = self.robot.get_controlled_parent_joint(collision.get_original_link_a())
        new_a = self.robot.get_child_link_of_joint(movable_joint)
        new_a_T_r = self.robot.get_fk_np(new_a, self.robot_root)
        collision.set_link_a(new_a)

        new_a_P_pa = np.dot(np.dot(new_a_T_r, self.root_T_map), np_point(*collision.get_position_on_a_in_map()))
        r_P_pb = np.dot(self.root_T_map, np_point(*collision.get_position_on_b_in_map()))
        r_V_n = np.dot(self.root_T_map, np_vector(*collision.get_contact_normal_in_map()))
        collision.set_position_on_a_in_a(new_a_P_pa[:-1])
        collision.set_position_on_b_in_root(r_P_pb[:-1])
        collision.set_contact_normal_in_root(r_V_n[:-1])
        return collision

    def _default_collision(self, link_a, body_b, link_b):
        return Collision(link_a, body_b, link_b, [0, 0, 0], [0, 0, 0], [0, 0, 1], 100)

    def get_external_collisions(self, joint_name):
        """
        Collisions are saved as a list for each movable robot joint, sorted by contact distance
        :type joint_name: str
        :rtype: SortedKeyList
        """
        return self.external_collision[joint_name]

    def get_self_collisions(self, link_a, link_b):
        """
        Make sure that link_a < link_b, the reverse collision is not saved.
        :type link_a: str
        :type link_b: str
        :return:
        :rtype: SortedKeyList
        """
        # FIXME maybe check for reverse key?
        return self.self_collisions[link_a, link_b]

    def __contains__(self, item):
        return item in self.self_collisions or item in self.external_collision

    def items(self):
        return self.all_collisions
