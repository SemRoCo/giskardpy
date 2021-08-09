from collections import OrderedDict, defaultdict, namedtuple

import numpy as np
from sortedcontainers import SortedKeyList

from giskardpy.utils.tfwrapper import kdl_to_np, np_vector, np_point

DebugConstraint = namedtuple(u'debug', [u'expr'])


class Constraint(object):
    lower_error = -1e4
    upper_error = 1e4
    lower_slack_limit = -1e4
    upper_slack_limit = 1e4
    linear_weight = 0

    def __init__(self, name, expression,
                 lower_error, upper_error,
                 velocity_limit,
                 quadratic_weight, control_horizon, linear_weight=None,
                 lower_slack_limit=None, upper_slack_limit=None, ):
        self.name = name
        self.expression = expression
        self.quadratic_weight = quadratic_weight
        self.control_horizon = control_horizon
        self.velocity_limit = velocity_limit
        self.lower_error = lower_error
        self.upper_error = upper_error
        if lower_slack_limit is not None:
            self.lower_slack_limit = lower_slack_limit
        if upper_slack_limit is not None:
            self.upper_slack_limit = upper_slack_limit
        if linear_weight is not None:
            self.linear_weight = linear_weight

    def __str__(self):
        return self.name

    def normalized_weight(self, prediction_horizon):
        weight_normalized = self.quadratic_weight * (1 / (self.velocity_limit)) ** 2
        return weight_normalized


class VelocityConstraint(object):
    lower_acceleration_limit = -1e4
    upper_acceleration_limit = 1e4
    lower_slack_limit = -1e3
    upper_slack_limit = 1e3
    linear_weight = 0

    def __init__(self, name, expression,
                 lower_velocity_limit, upper_velocity_limit,
                 quadratic_weight,
                 lower_slack_limit=None, upper_slack_limit=None,
                 linear_weight=None, control_horizon=None,
                 horizon_function=None):
        self.name = name
        self.expression = expression
        self.quadratic_weight = quadratic_weight
        self.control_horizon = control_horizon
        self.lower_velocity_limit = lower_velocity_limit
        self.upper_velocity_limit = upper_velocity_limit
        if lower_slack_limit is not None:
            self.lower_slack_limit = lower_slack_limit
        if upper_slack_limit is not None:
            self.upper_slack_limit = upper_slack_limit
        if linear_weight is not None:
            self.linear_weight = linear_weight

        def default_horizon_function(weight, t):
            return weight

        self.horizon_function = default_horizon_function
        if horizon_function is not None:
            self.horizon_function = horizon_function

    def normalized_weight(self, t):
        weight_normalized = self.quadratic_weight * (1 / self.upper_velocity_limit) ** 2
        return self.horizon_function(weight_normalized, t)


class FreeVariable(object):
    def __init__(self, symbols, lower_limits, upper_limits, quadratic_weights, horizon_functions=None):
        """
        :type symbols:  dict
        :type lower_limits: dict
        :type upper_limits: dict
        :type quadratic_weights: dict
        :type horizon_functions: dict
        """
        self._symbols = symbols
        self.name = str(self._symbols[0])
        self._lower_limits = lower_limits
        self._upper_limits = upper_limits
        self._quadratic_weights = quadratic_weights
        assert max(self._symbols.keys()) == len(self._symbols) - 1
        assert len(self._symbols) == len(self._quadratic_weights) + 1
        self.order = len(self._symbols)

        # def default_horizon_f(weight, t):
        #     return weight

        self.horizon_functions = defaultdict(float)
        self.horizon_functions.update(horizon_functions)

    def get_symbol(self, order):
        try:
            return self._symbols[order]
        except KeyError:
            raise KeyError(u'Free variable {} doesn\'t have symbol for derivative of order {}'.format(self, order))

    def get_lower_limit(self, order):
        try:
            return self._lower_limits[order]
        except KeyError:
            raise KeyError(u'Free variable {} doesn\'t have lower limit for derivative of order {}'.format(self, order))

    def get_upper_limit(self, order):
        try:
            return self._upper_limits[order]
        except KeyError:
            raise KeyError(u'Free variable {} doesn\'t have upper limit for derivative of order {}'.format(self, order))

    def has_position_limits(self):
        try:
            lower_limit = self.get_lower_limit(0)
            upper_limit = self.get_upper_limit(0)
            return lower_limit is not None and abs(lower_limit) < 100 \
                   and upper_limit is not None and abs(upper_limit) < 100
        except Exception:
            return False

    def get_quadratic_weights(self, order):
        try:
            return self._quadratic_weights[order]
        except KeyError:
            raise KeyError(u'Free variable {} doesn\'t have weight for derivative of order {}'.format(self, order))

    def normalized_weight(self, t, order, prediction_horizon):
        weight = self.get_quadratic_weights(order)
        start = weight * self.horizon_functions[order]
        a = (weight - start) / (prediction_horizon)
        weight = a*t + start
        # weight = (self.get_quadratic_weights(order), t)
        return weight * (1 / self.get_upper_limit(order)) ** 2

    def __str__(self):
        return self.name


class SingleJointState(object):
    def __init__(self, name='', position=0.0, velocity=0.0, acceleration=0.0, jerk=0.0, snap=0.0, crackle=0.0, pop=0.0):
        self.name = name
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.jerk = jerk
        self.snap = snap
        self.crackle = crackle
        self.pop = pop

    def __str__(self):
        return u'{}: {}, {}, {}, {}, {}, {}, {}'.format(self.name,
                                            self.position,
                                            self.velocity,
                                            self.acceleration,
                                            self.jerk,
                                            self.snap,
                                            self.crackle,
                                            self.pop)


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
