from collections import OrderedDict, defaultdict, deque
from copy import deepcopy

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from sortedcontainers import SortedKeyList
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from giskardpy import RobotName


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


class _JointState(object):
    derivative_to_name = {
        0: 'position',
        1: 'velocity',
        2: 'acceleration',
        3: 'jerk',
        4: 'snap',
        5: 'crackle',
        6: 'pop',
    }

    def __init__(self, position=0, velocity=0, acceleration=0, jerk=0, snap=0, crackle=0, pop=0):
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.jerk = jerk
        self.snap = snap
        self.crackle = crackle
        self.pop = pop

    def set_derivative(self, d, item):
        setattr(self, self.derivative_to_name[d], item)

    def __str__(self):
        return '{}'.format(self.position)

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, memodict={}):
        return _JointState(self.position, self.velocity, self.acceleration, self.jerk, self.snap, self.crackle, self.pop)

class JointStates(defaultdict):
    def __init__(self, *args, **kwargs):
        super(JointStates, self).__init__(_JointState, *args, **kwargs)

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

class Trajectory(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self._points = OrderedDict()

    def get_exact(self, time):
        return self._points[time]

    def set(self, time, point):
        if len(self._points) > 0 and list(self._points.keys())[-1] > time:
            raise KeyError('Cannot append a trajectory point that is before the current end time of the trajectory.')
        self._points[time] = point

    def delete(self, time):
        del self._points[time]

    def delete_last(self):
        self.delete(list(self._points.keys())[-1])

    def get_last(self):
        return list(self._points.values())[-1]

    def items(self):
        return self._points.items()

    def keys(self):
        return self._points.keys()

    def values(self):
        return self._points.values()

    def to_msg(self, sample_period, controlled_joints, fill_velocity_values):
        """
        :type traj: giskardpy.data_types.Trajectory
        :return: JointTrajectory
        """
        trajectory_msg = JointTrajectory()
        trajectory_msg.header.stamp = rospy.get_rostime() + rospy.Duration(0.5)
        trajectory_msg.joint_names = controlled_joints
        for time, traj_point in self.items():
            p = JointTrajectoryPoint()
            p.time_from_start = rospy.Duration(time * sample_period)
            for joint_name in controlled_joints:
                if joint_name in traj_point:
                    p.positions.append(traj_point[joint_name].position)
                    if fill_velocity_values:
                        p.velocities.append(traj_point[joint_name].velocity)
                else:
                    raise NotImplementedError('generated traj does not contain all joints')
            trajectory_msg.points.append(p)
        return trajectory_msg


class Collision(object):
    def __init__(self, link_a, body_b, link_b, contact_distance,
                 map_P_pa=None, map_P_pb=None, map_V_n=None,
                 a_P_pa=None, b_P_pb=None):
        self.contact_distance = contact_distance
        self.body_b = body_b
        self.link_a = link_a
        self.original_link_a = link_a
        self.link_b = link_b
        self.original_link_b = link_b

        self.map_P_pa = self.__point_to_4d(map_P_pa)
        self.map_P_pb = self.__point_to_4d(map_P_pb)
        self.map_V_n = self.__vector_to_4d(map_V_n)
        self.old_key = (link_a, body_b, link_a)
        self.a_P_pa = self.__point_to_4d(a_P_pa)
        self.b_P_pb = self.__point_to_4d(b_P_pb)

        self.new_a_P_pa = None
        self.new_b_P_pb = None
        self.new_b_V_n = None

    def __point_to_4d(self, point):
        if point is None:
            return point
        point = np.array(point)
        if len(point) == 3:
            return np.append(point, 1)
        return point

    def __vector_to_4d(self, vector):
        if vector is None:
            return vector
        vector = np.array(vector)
        if len(vector) == 3:
            return np.append(vector, 0)
        return vector

    def get_link_b_hash(self):
        return self.link_b.__hash__()

    def get_body_b_hash(self):
        return self.body_b.__hash__()

    def reverse(self):
        return Collision(link_a=self.original_link_b,
                         body_b=self.body_b,
                         link_b=self.original_link_a,
                         map_P_pa=self.map_P_pb,
                         map_P_pb=self.map_P_pa,
                         map_V_n=-self.map_V_n,
                         a_P_pa=self.b_P_pb,
                         b_P_pb=self.a_P_pa,
                         contact_distance=self.contact_distance)


class Collisions(object):
    @profile
    def __init__(self, world, collision_list_size):
        """
        :type robot: giskardpy.model.world.WorldTree
        """
        self.world = world
        self.robot = self.world.groups[RobotName]
        self.robot_root = self.robot.root_link_name
        self.root_T_map = self.robot.compute_fk_np(self.robot_root, self.world.root_link_name)
        self.collision_list_size = collision_list_size

        # @profile
        def sort(x):
            return x.contact_distance

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

    @profile
    def add(self, collision):
        """
        :type collision: Collision
        :return:
        """
        collision = self.transform_closest_point(collision)
        self.all_collisions.add(collision)

        if collision.body_b == self.robot.name:
            key = collision.link_a, collision.link_b
            self.self_collisions[key].add(collision)
            self.number_of_self_collisions[key] = min(self.collision_list_size,
                                                      self.number_of_self_collisions[key] + 1)
        else:
            key = collision.link_a
            self.external_collision[key].add(collision)
            self.number_of_external_collisions[key] = min(self.collision_list_size,
                                                          self.number_of_external_collisions[key] + 1)
            key_long = (collision.original_link_a, None, collision.original_link_b)
            if key_long not in self.external_collision_long_key:
                self.external_collision_long_key[key_long] = collision
            else:
                self.external_collision_long_key[key_long] = min(collision, self.external_collision_long_key[key_long],
                                                                 key=lambda x: x.contact_distance)

    def transform_closest_point(self, collision):
        """
        :type collision: Collision
        :rtype: Collision
        """
        if collision.body_b == self.robot.name:
            return self.transform_self_collision(collision)
        else:
            return self.transform_external_collision(collision)

    @profile
    def transform_self_collision(self, collision):
        """
        :type collision: Collision
        :rtype: Collision
        """
        link_a = collision.original_link_a
        link_b = collision.original_link_b
        new_link_a, new_link_b = self.world.compute_chain_reduced_to_controlled_joints(link_a, link_b)
        if new_link_a > new_link_b:
            collision = collision.reverse()
            new_link_a, new_link_b = new_link_b, new_link_a
        collision.link_a = new_link_a
        collision.link_b = new_link_b

        new_b_T_r = self.world.compute_fk_np(new_link_b, self.world.root_link_name)
        new_b_T_map = np.dot(new_b_T_r, self.root_T_map)
        collision.new_b_V_n = np.dot(new_b_T_map, collision.map_V_n)

        if collision.map_P_pa is not None:
            new_a_T_r = self.world.compute_fk_np(new_link_a, self.world.root_link_name)
            new_a_P_pa = np.dot(np.dot(new_a_T_r, self.root_T_map), collision.map_P_pa)
            new_b_P_pb = np.dot(new_b_T_map, collision.map_P_pb)
        else:
            new_a_T_a = self.world.compute_fk_np(new_link_a, collision.original_link_a)
            new_a_P_pa = np.dot(new_a_T_a, collision.a_P_pa)
            new_b_T_b = self.world.compute_fk_np(new_link_b, collision.original_link_b)
            new_b_P_pb = np.dot(new_b_T_b, collision.b_P_pb)
        collision.new_a_P_pa = new_a_P_pa
        collision.new_b_P_pb = new_b_P_pb
        return collision

    @profile
    def transform_external_collision(self, collision):
        """
        :type collision: Collision
        :rtype: Collision
        """
        movable_joint = self.robot.get_controlled_parent_joint_of_link(collision.original_link_a)
        new_a = self.robot.joints[movable_joint].child_link_name
        collision.link_a = new_a
        if collision.map_P_pa is not None:
            new_a_T_map = self.world.compute_fk_np(new_a, self.world.root_link_name)
            new_a_P_a = np.dot(new_a_T_map, collision.map_P_pa)
        else:
            new_a_T_a = self.world.compute_fk_np(new_a, collision.original_link_a)
            new_a_P_a = np.dot(new_a_T_a, collision.a_P_pa)

        collision.new_a_P_pa = new_a_P_a
        return collision

    def _default_collision(self, link_a, body_b, link_b):
        c = Collision(link_a=link_a,
                      body_b=body_b,
                      link_b=link_b,
                      contact_distance=100,
                      map_P_pa=[0, 0, 0, 1],
                      map_P_pb=[0, 0, 0, 1],
                      map_V_n=[0, 0, 1, 0],
                      a_P_pa=[0, 0, 0, 1],
                      b_P_pb=[0, 0, 0, 1])
        c.new_a_P_pa = [0, 0, 0, 1]
        c.new_b_P_pb = [0, 0, 0, 1]
        c.new_b_V_n = [0, 0, 1, 0]
        return c

    @profile
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

    @profile
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
