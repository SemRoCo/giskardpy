from collections import OrderedDict, defaultdict, namedtuple

import numpy as np
from sortedcontainers import SortedKeyList

from giskardpy.tfwrapper import kdl_to_np, np_vector, np_point

# SoftConstraint = namedtuple(u'SoftConstraint', [u'lbA_v', u'ubA_v',
#                                                 u'lbA_a', u'ubA_a',
#                                                 u'weight_v', u'weight_a',
#                                                 u'expression', u'expression_dot', u'goal_constraint',
#                                                 u'lower_slack_limit_v', u'lower_slack_limit_a',
#                                                 u'upper_slack_limit_v', u'upper_slack_limit_a',
#                                                 u'linear_weight'])
# HardConstraint = namedtuple(u'HardConstraint', [u'lower', u'upper', u'expression'])
# JointConstraint = namedtuple(u'JointConstraint', [u'lower_p', u'upper_p',
#                                                   u'lower_v', u'upper_v', u'weight_v',
#                                                   u'lower_a', u'upper_a', u'weight_a',
#                                                   u'lower_j', u'upper_j', u'weight_j',
#                                                   u'joint_symbol', u'joint_velocity_symbol',
#                                                   u'joint_acceleration_symbol',
#                                                   u'linear_weight'])
DebugConstraint = namedtuple(u'debug', [u'expr'])


class Constraint(object):
    lower_position_limit = None
    upper_position_limit = None
    lower_velocity_limit = -1e4
    upper_velocity_limit = 1e4
    lower_acceleration_limit = -1e4
    upper_acceleration_limit = 1e4
    lower_jerk_limit = -1e4
    upper_jerk_limit = 1e4
    lower_slack_limit = -1e4
    upper_slack_limit = 1e4
    linear_weight = 0

    def __init__(self, name, expression,
                 lower_position_limit, upper_position_limit,
                 lower_velocity_limit, upper_velocity_limit,
                 lower_acceleration_limit, upper_acceleration_limit,
                 lower_jerk_limit, upper_jerk_limit,
                 lower_slack_limit, upper_slack_limit,
                 quadratic_velocity_weight, linear_weight):
        self.name = name
        self.expression = expression
        self.quadratic_velocity_weight = quadratic_velocity_weight
        self.lower_position_limit = lower_position_limit
        self.upper_position_limit = upper_position_limit
        if lower_velocity_limit is not None:
            self.lower_velocity_limit = lower_velocity_limit
        if upper_velocity_limit is not None:
            self.upper_velocity_limit = upper_velocity_limit
        if lower_acceleration_limit is not None:
            self.lower_acceleration_limit = lower_acceleration_limit
        if upper_acceleration_limit is not None:
            self.upper_acceleration_limit = upper_acceleration_limit
        if lower_jerk_limit is not None:
            self.lower_jerk_limit = lower_jerk_limit
        if upper_jerk_limit is not None:
            self.upper_jerk_limit = upper_jerk_limit
        if lower_slack_limit is not None:
            self.lower_slack_limit = lower_slack_limit
        if upper_slack_limit is not None:
            self.upper_slack_limit = upper_slack_limit
        if linear_weight is not None:
            self.linear_weight = linear_weight

    def __str__(self):
        return self.name


class VelocityConstraint(Constraint):
    def __init__(self, name, expression, lower_velocity_limit, upper_velocity_limit, quadratic_velocity_weight,
                 lower_slack_limit=None, upper_slack_limit=None, linear_weight=None):
        super(VelocityConstraint, self).__init__(name=name,
                                                 expression=expression,
                                                 lower_position_limit=None,
                                                 upper_position_limit=None,
                                                 lower_velocity_limit=lower_velocity_limit,
                                                 upper_velocity_limit=upper_velocity_limit,
                                                 lower_acceleration_limit=None,
                                                 upper_acceleration_limit=None,
                                                 lower_jerk_limit=None,
                                                 upper_jerk_limit=None,
                                                 lower_slack_limit=lower_slack_limit,
                                                 upper_slack_limit=upper_slack_limit,
                                                 quadratic_velocity_weight=quadratic_velocity_weight,
                                                 linear_weight=linear_weight)


class PositionConstraint(Constraint):
    def __init__(self, name, expression, lower_position_limit, upper_position_limit, quadratic_velocity_weight,
                 lower_slack_limit=None, upper_slack_limit=None, linear_weight=None):
        super(PositionConstraint, self).__init__(name=name,
                                                 expression=expression,
                                                 lower_position_limit=lower_position_limit,
                                                 upper_position_limit=upper_position_limit,
                                                 lower_velocity_limit=None,
                                                 upper_velocity_limit=None,
                                                 lower_acceleration_limit=None,
                                                 upper_acceleration_limit=None,
                                                 lower_jerk_limit=None,
                                                 upper_jerk_limit=None,
                                                 lower_slack_limit=lower_slack_limit,
                                                 upper_slack_limit=upper_slack_limit,
                                                 quadratic_velocity_weight=quadratic_velocity_weight,
                                                 linear_weight=linear_weight)


class FreeVariable(object):
    def __init__(self, position_symbol,
                 lower_position_limit, upper_position_limit,
                 lower_velocity_limit, upper_velocity_limit,
                 lower_acceleration_limit, upper_acceleration_limit,
                 lower_jerk_limit, upper_jerk_limit,
                 quadratic_velocity_weight, quadratic_acceleration_weight, quadratic_jerk_weight,
                 velocity_symbol=None, acceleration_symbol=None, jerk_symbol=None, linear_weight=None):
        self.position_symbol = position_symbol
        self.velocity_symbol = velocity_symbol
        self.acceleration_symbol = acceleration_symbol
        self.jerk_symbol = jerk_symbol
        self.lower_position_limit = lower_position_limit
        self.upper_position_limit = upper_position_limit
        self.lower_velocity_limit = lower_velocity_limit
        self.upper_velocity_limit = upper_velocity_limit
        self.lower_acceleration_limit = lower_acceleration_limit
        self.upper_acceleration_limit = upper_acceleration_limit
        self.lower_jerk_limit = lower_jerk_limit
        self.upper_jerk_limit = upper_jerk_limit
        self.quadratic_velocity_weight = quadratic_velocity_weight
        self.linear_weight = linear_weight
        self.quadratic_acceleration_weight = quadratic_acceleration_weight
        self.quadratic_jerk_weight = quadratic_jerk_weight
        self.name = str(self.position_symbol)

    def has_position_limits(self):
        return self.lower_position_limit is not None and abs(self.upper_position_limit) < 100

    def __str__(self):
        return self.name


class SingleJointState(object):
    def __init__(self, name='', position=0.0, velocity=0.0, acceleration=0.0, jerk=0.0):
        self.name = name
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.jerk = jerk

    def __str__(self):
        return u'{}: {}, {}, {}, {}'.format(self.name, self.position, self.velocity, self.acceleration, self.jerk)


class Trajectory(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self._points = OrderedDict()

    def get_exact(self, time):
        return self._points[time]

    def set(self, time, point):
        if len(self._points) > 0 and list(self._points.keys())[-1] > time:
            raise KeyError(u'Cannot append a trajectory point that is before the current end time of the trajectory.')
        self._points[time] = point

    def delete(self, time):
        del self._points[time]

    def delete_last(self):
        self.delete(self._points.keys()[-1])

    def items(self):
        return self._points.items()

    def keys(self):
        return self._points.keys()

    def values(self):
        return self._points.values()


class Collision(object):
    def __init__(self, link_a, body_b, link_b, position_on_a, position_on_b, contact_normal, contact_distance):
        self.__position_on_a = position_on_a
        self.__position_on_a_in_a = position_on_a
        self.__position_on_b = position_on_b
        self.__position_on_b_in_b = position_on_b
        self.__position_on_b_in_root = position_on_b
        self.__contact_distance = contact_distance
        self.__contact_normal = contact_normal
        self.__contact_normal_in_root = contact_normal
        self.__contact_normal_in_b = contact_normal
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

    # @profile
    def get_link_b_hash(self):
        return self.get_link_b().__hash__()

    def get_body_b(self):
        return self.__body_b

    # @profile
    def get_body_b_hash(self):
        return self.get_body_b().__hash__()

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

    def reverse(self):
        return Collision(link_a=self.get_original_link_b(),
                         body_b=self.get_body_b(),
                         link_b=self.get_original_link_a(),
                         position_on_a=self.get_position_on_b_in_map(),
                         position_on_b=self.get_position_on_a_in_map(),
                         contact_normal=[-self.__contact_normal[0],
                                         -self.__contact_normal[1],
                                         -self.__contact_normal[2]],
                         contact_distance=self.get_contact_distance())


class Collisions(object):

    def __init__(self, robot, collision_list_size):
        """
        :type robot: giskardpy.robot.Robot
        """
        self.robot = robot
        self.root_T_map = kdl_to_np(self.robot.root_T_map)
        self.robot_root = self.robot.get_root()
        self.collision_list_size = collision_list_size

        # @profile
        def sort(x):
            return x.get_contact_distance()

        # @profile
        def default_f():
            return SortedKeyList([self._default_collision('', '', '')] * collision_list_size,
                                 key=sort)

        self.default_result = default_f()

        self.self_collisions = defaultdict(default_f)
        self.external_collision = defaultdict(default_f)
        self.external_collision_long_key = defaultdict(lambda: self._default_collision('', '', ''))
        self.all_collisions = set()
        self.number_of_self_collisions = defaultdict(int)
        self.number_of_external_collisions = defaultdict(int)

    # @profile
    def add(self, collision):
        """
        :type collision: Collision
        :return:
        """
        collision = self.transform_closest_point(collision)
        self.all_collisions.add(collision)

        if collision.get_body_b() == self.robot.get_name():
            key = collision.get_link_a(), collision.get_link_b()
            self.self_collisions[key].add(collision)
            self.number_of_self_collisions[key] = min(self.collision_list_size,
                                                      self.number_of_self_collisions[key] + 1)
        else:
            key = collision.get_link_a()
            self.external_collision[key].add(collision)
            self.number_of_external_collisions[key] = min(self.collision_list_size,
                                                          self.number_of_external_collisions[key] + 1)
            key_long = (collision.get_original_link_a(), collision.get_body_b(), collision.get_original_link_b())
            if key_long not in self.external_collision_long_key:
                self.external_collision_long_key[key_long] = collision
            else:
                self.external_collision_long_key[key_long] = min(collision, self.external_collision_long_key[key_long],
                                                                 key=lambda x: x.get_contact_distance())

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
        if new_link_a > new_link_b:
            collision = collision.reverse()
            new_link_a, new_link_b = new_link_b, new_link_a

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

    # @profile
    def get_external_collisions(self, joint_name):
        """
        Collisions are saved as a list for each movable robot joint, sorted by contact distance
        :type joint_name: str
        :rtype: SortedKeyList
        """
        if joint_name in self.external_collision:
            return self.external_collision[joint_name]
        return self.default_result

    def get_external_collisions_long_key(self, link_a, body_b, link_b):
        """
        Collisions are saved as a list for each movable robot joint, sorted by contact distance
        :type joint_name: str
        :rtype: SortedKeyList
        """
        return self.external_collision_long_key[link_a, body_b, link_b]

    def get_number_of_external_collisions(self, joint_name):
        return self.number_of_external_collisions[joint_name]

    # @profile
    def get_self_collisions(self, link_a, link_b):
        """
        Make sure that link_a < link_b, the reverse collision is not saved.
        :type link_a: str
        :type link_b: str
        :return:
        :rtype: SortedKeyList
        """
        # FIXME maybe check for reverse key?
        if (link_a, link_b) in self.self_collisions:
            return self.self_collisions[link_a, link_b]
        return self.default_result

    def get_number_of_self_collisions(self, link_a, link_b):
        return self.number_of_self_collisions[link_a, link_b]

    def __contains__(self, item):
        return item in self.self_collisions or item in self.external_collision

    def items(self):
        return self.all_collisions
