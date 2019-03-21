import hashlib
import os
import numpy as np
import errno
import pickle
from itertools import combinations, product

from geometry_msgs.msg import Pose, Quaternion

from giskardpy.data_types import SingleJointState
from giskardpy.tfwrapper import msg_to_kdl
from giskardpy.urdf_object import URDFObject


class WorldObject(URDFObject):
    def __init__(self, urdf, base_pose=None, controlled_joints=None, path_to_data_folder=u'', *args, **kwargs):
        super(WorldObject, self).__init__(urdf, *args, **kwargs)
        self.path_to_data_folder = path_to_data_folder + u'collision_matrix/'
        self.controlled_joints = controlled_joints
        if base_pose is None:
            p = Pose()
            p.orientation.w = 1
            self.base_pose = p
        # FIXME using .joint_state creates a chicken egg problem in pybulletworldobject
        self._js = self.get_zero_joint_state()
        self._self_collision_matrix = set()

    @classmethod
    def from_urdf_file(cls, urdf_file, *args, **kwargs):
        return super(WorldObject, cls).from_urdf_file(urdf_file, *args, **kwargs)

    @classmethod
    def from_world_body(cls, world_body, *args, **kwargs):
        return super(WorldObject, cls).from_world_body(world_body, *args, **kwargs)

    @classmethod
    def from_parts(cls, robot_name, links, joints, *args, **kwargs):
        return super(WorldObject, cls).from_parts(robot_name, links, joints, *args, **kwargs)

    @property
    def joint_state(self):
        return self._js

    @joint_state.setter
    def joint_state(self, value):
        self._js = value

    @property
    def base_pose(self):
        return self._base_pose

    @base_pose.setter
    def base_pose(self, value):
        """
        :type value: Pose
        :return:
        """
        orientation_vector = np.array([value.orientation.x,
                                       value.orientation.y,
                                       value.orientation.z,
                                       value.orientation.w])
        self._base_pose = value
        self._base_pose.orientation = Quaternion(*orientation_vector / np.linalg.norm(orientation_vector))
        self.T_base___map = msg_to_kdl(self._base_pose).Inverse()

    @property
    def controlled_joints(self):
        # FIXME reinitialize does not handle newly added or removed controllable joints
        if self._controlled_joints is None:
            self._controlled_joints = self.get_controllable_joints()
        return self._controlled_joints

    @controlled_joints.setter
    def controlled_joints(self, value):
        self._controlled_joints = value

    def suicide(self):
        pass

    def __del__(self):
        self.suicide()

    def get_controlled_links(self):
        controllable_links = set()
        for joint_name in self.controlled_joints:
            controllable_links.update(self.get_sub_tree_link_names_with_collision(joint_name))
        return controllable_links

    def get_self_collision_matrix(self):
        """
        :return: (link1, link2) -> min allowed distance
        """
        return self._self_collision_matrix

    def calc_collision_matrix(self, link_combinations, d=0.05, d2=0.0, num_rnd_tries=2000):
        """
        :param link_combinations: set with link name tuples
        :type link_combinations: set
        :param d: distance threshold to detect links that are always in collision
        :type d: float
        :param d2: distance threshold to find links that are sometimes in collision
        :type d2: float
        :param num_rnd_tries:
        :type num_rnd_tries: int
        :return: set of link name tuples which are sometimes in collision.
        :rtype: set
        """
        # TODO computational expansive because of too many collision checks
        print(u'calculating self collision matrix')
        np.random.seed(1337)
        always = set()

        # find meaningless self-collisions
        for link_a, link_b in link_combinations:
            if self.are_linked(link_a, link_b):
                always.add((link_a, link_b))
        rest = link_combinations.difference(always)
        self.joint_state = self.get_zero_joint_state()
        always = always.union(self.check_collisions(rest, d))
        rest = rest.difference(always)

        # find meaningful self-collisions
        self.joint_state = self.get_min_joint_state()
        sometimes = self.check_collisions(rest, d2)
        rest = rest.difference(sometimes)
        self.joint_state = self.get_max_joint_state()
        sometimes2 = self.check_collisions(rest, d2)
        rest = rest.difference(sometimes2)
        sometimes = sometimes.union(sometimes2)
        for i in range(num_rnd_tries):
            self.joint_state = self.get_rnd_joint_state()
            sometimes2 = self.check_collisions(rest, d2)
            if len(sometimes2) > 0:
                rest = rest.difference(sometimes2)
                sometimes = sometimes.union(sometimes2)
        return sometimes

    def get_possible_collisions(self, link):
        # TODO speed up by saving this
        possible_collisions = set()
        for link1, link2 in self.get_self_collision_matrix():
            if link == link1:
                possible_collisions.add(link2)
            elif link == link2:
                possible_collisions.add(link1)
        return possible_collisions

    def check_collisions(self, link_combinations, distance):
        in_collision = set()
        for link_a, link_b in link_combinations:
            if self.in_collision(link_a, link_b, distance):
                in_collision.add((link_a, link_b))
        return in_collision

    def in_collision(self, link_a, link_b, distance):
        return self.are_linked(link_a, link_b)

    def get_zero_joint_state(self):
        # FIXME 0 might not be a valid joint value
        return self.generate_joint_state(lambda x: 0)

    def get_max_joint_state(self):
        def f(joint_name):
            _, upper_limit = self.get_joint_limits(joint_name)
            if upper_limit is None:
                return np.pi * 2
            return upper_limit

        return self.generate_joint_state(f)

    def get_min_joint_state(self):
        def f(joint_name):
            lower_limit, _ = self.get_joint_limits(joint_name)
            if lower_limit is None:
                return -np.pi * 2
            return lower_limit

        return self.generate_joint_state(f)

    def get_rnd_joint_state(self):
        def f(joint_name):
            lower_limit, upper_limit = self.get_joint_limits(joint_name)
            if lower_limit is None:
                return np.random.random() * np.pi * 2
            lower_limit = max(lower_limit, -10)
            upper_limit = min(upper_limit, 10)
            return (np.random.random() * (upper_limit - lower_limit)) + lower_limit

        return self.generate_joint_state(f)

    def generate_joint_state(self, f):
        """
        :param f: lambda joint_info: float
        :return:
        """
        # TODO possible optimization, if some joints are not controlled, the collision matrix might get smaller
        js = {}
        for joint_name in self.get_controllable_joints():
            sjs = SingleJointState()
            sjs.name = joint_name
            sjs.position = f(joint_name)
            js[joint_name] = sjs
        return js

    def add_self_collision_entries(self, object_name):
        link_pairs = {(object_name, link_name) for link_name in self.get_link_names()}
        link_pairs.remove((object_name, object_name))
        self_collision_with_object = self.calc_collision_matrix(link_pairs)
        self._self_collision_matrix.update(self_collision_with_object)

    def remove_self_collision_entries(self, object_name):
        self._self_collision_matrix = {(link1, link2) for link1, link2 in self.get_self_collision_matrix()
                                       if link1 != object_name and link2 != object_name}

    def update_self_collision_matrix(self, added_links=None, removed_links=None):
        if not self.load_self_collision_matrix(self.path_to_data_folder):
            if added_links is None:
                added_links = set()
            if removed_links is None:
                removed_links = set()
            self._self_collision_matrix = {x for x in self._self_collision_matrix if x[0] not in removed_links and
                                           x[1] not in removed_links}
            self._self_collision_matrix.update(self.calc_collision_matrix(added_links))
            self.safe_self_collision_matrix(self.path_to_data_folder)

    def load_self_collision_matrix(self, path):
        """
        :rtype: bool
        """
        urdf_hash = hashlib.md5(self.get_urdf()).hexdigest()
        path = u'{}/{}/{}'.format(path, self.get_name(), urdf_hash)
        if os.path.isfile(path):
            with open(path) as f:
                self._self_collision_matrix = pickle.load(f)
                print(u'loaded self collision matrix {}'.format(path))
                return True
        return False

    def safe_self_collision_matrix(self, path):
        urdf_hash = hashlib.md5(self.get_urdf()).hexdigest()
        path = u'{}/{}/{}'.format(path, self.get_name(), urdf_hash)
        if not os.path.exists(os.path.dirname(path)):
            try:
                dir_name = os.path.dirname(path)
                if dir_name != u'':
                    os.makedirs(dir_name)
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        with open(path, u'w') as file:
            print(u'saved self collision matrix {}'.format(path))
            pickle.dump(self._self_collision_matrix, file)

    def as_marker_msg(self, ns=u'', id=1):
        m = super(WorldObject, self).as_marker_msg(ns, id)
        m.pose = self.base_pose
        return m

    def attach_urdf_object(self, urdf_object, parent_link, pose):
        super(WorldObject, self).attach_urdf_object(urdf_object, parent_link, pose)
        self.update_self_collision_matrix(added_links=set(product(self.get_link_names(), urdf_object.get_link_names())))

    def detach_sub_tree(self, joint_name):
        sub_tree = super(WorldObject, self).detach_sub_tree(joint_name)
        self.update_self_collision_matrix(removed_links=sub_tree.get_link_names())
        return sub_tree

    def reset(self):
        super(WorldObject, self).reset()
        self.update_self_collision_matrix()

