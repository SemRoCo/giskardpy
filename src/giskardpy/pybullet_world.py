import pickle
import pybullet as p
import string
import random
import os
from collections import namedtuple, OrderedDict, defaultdict
from copy import copy
from itertools import combinations
from pybullet import JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_PLANAR, JOINT_SPHERICAL

import errno
from numpy.random.mtrand import seed

import giskardpy
from giskardpy import DEBUG
from giskardpy.exceptions import UnknownBodyException, RobotExistsException, DuplicateNameException
from giskardpy.data_types import SingleJointState, Transform, Point, Quaternion
import numpy as np

from giskardpy.utils import keydefaultdict, suppress_stdout, NullContextManager, resolve_ros_iris

from giskardpy.object import UrdfObject, FixedJoint, to_urdf_string, BoxShape, \
    CollisionProperty, remove_outer_tag, SphereShape
import hashlib

JointInfo = namedtuple(u'JointInfo', [u'joint_index', u'joint_name', u'joint_type', u'q_index', u'u_index', u'flags',
                                      u'joint_damping', u'joint_friction', u'joint_lower_limit', u'joint_upper_limit',
                                      u'joint_max_force', u'joint_max_velocity', u'link_name', u'joint_axis',
                                      u'parent_frame_pos', u'parent_frame_orn', u'parent_index'])

ContactInfo = namedtuple(u'ContactInfo', [u'contact_flag', u'body_unique_id_a', u'body_unique_id_b', u'link_index_a',
                                          u'link_index_b', u'position_on_a', u'position_on_b', u'contact_normal_on_b',
                                          u'contact_distance', u'normal_force', u'lateralFriction1',
                                          u'lateralFrictionDir1',
                                          u'lateralFriction2', u'lateralFrictionDir2'])


def write_urdf_to_disc(filename, urdf_string):
    """
    Writes a URDF string into a temporary file on disc. Used to deliver URDFs to PyBullet that only loads file.
    :param filename: Name of the temporary file without any path information, e.g. 'pr2.urdf'
    :type filename: str
    :param urdf_string: URDF as an XML string that shall be written to disc.
    :type urdf_string: str
    :return: Complete path to where the urdf was written, e.g. '/tmp/pr2.urdf'
    :rtype: str
    """
    new_path = u'/tmp/{}'.format(filename)
    with open(new_path, u'w') as o:
        o.write(urdf_string)
    return new_path


def random_string(size=6):
    """
    Creates and returns a random string.
    :param size: Number of characters that the string shall contain.
    :type size: int
    :return: Generated random sequence of chars.
    :rtype: str
    """
    return u''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(size))


def load_urdf_string_into_bullet(urdf_string, pose):
    """
    Loads a URDF string into the bullet world.
    :param urdf_string: XML string of the URDF to load.
    :type urdf_string: str
    :param pose: Pose at which to load the URDF into the world.
    :type pose: Transform
    :return: internal PyBullet id of the loaded urdf
    :rtype: intload_urdf_string_into_bullet
    """
    filename = write_urdf_to_disc(u'{}.urdf'.format(random_string()), urdf_string)
    with NullContextManager() if giskardpy.PRINT_LEVEL == DEBUG else suppress_stdout():
        id = p.loadURDF(filename, [pose.translation.x, pose.translation.y, pose.translation.z],
                        [pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w],
                        flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
    os.remove(filename)
    return id


class PyBulletRobot(object):
    """
    Keeps track of and offers convenience functions for an urdf object in bullet.
    """
    # TODO maybe merge symengine robot with this class?
    base_link_name = u'base'

    def __init__(self, name, urdf, controlled_joints, base_pose=Transform(), calc_self_collision_matrix=True,
                 path_to_data_folder=''):
        """
        :type name: str
        :param urdf: Path to URDF file, or content of already loaded URDF file.
        :type urdf: str
        :type base_pose: Transform
        :type calc_self_collision_matrix: bool
        :param path_to_data_folder: where the self collision matrix is stored
        :type path_to_data_folder: str
        """
        self.path_to_data_folder = path_to_data_folder + u'collision_matrix/'
        self.name = name
        # TODO i think resolve iris can be removed because I do this at the start
        self.base_urdf = resolve_ros_iris(urdf)
        self.id = load_urdf_string_into_bullet(self.base_urdf, base_pose)
        self.sync_with_bullet()
        self.attached_objects = {}
        self.controlled_joints = controlled_joints
        if calc_self_collision_matrix:
            if not self.load_self_collision_matrix():
                self.init_self_collision_matrix()
                self.save_self_collision_matrix()
        else:
            self.self_collision_matrix = set()

    def init_self_collision_matrix(self):
        self.self_collision_matrix = self.calc_collision_matrix(set(combinations(self.get_link_names(), 2)))

    def load_self_collision_matrix(self):
        """
        :rtype: bool
        """
        urdf_hash = hashlib.md5(self.base_urdf).hexdigest()
        path = self.path_to_data_folder + urdf_hash
        if os.path.isfile(path):
            with open(path) as f:
                self.self_collision_matrix = pickle.load(f)
                print(u'loaded self collision matrix {}'.format(urdf_hash))
                return True
        return False

    def save_self_collision_matrix(self):
        urdf_hash = hashlib.md5(self.base_urdf).hexdigest()
        path = self.path_to_data_folder + urdf_hash
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
            pickle.dump(self.self_collision_matrix, file)

    def get_attached_objects(self):
        """
        :rtype: dict
        """
        return self.attached_objects

    def set_joint_state(self, multi_joint_state):
        """

        :param multi_joint_state:
        :type multi_joint_state: dict
        :return:
        """
        for joint_name, singe_joint_state in multi_joint_state.items():
            p.resetJointState(self.id, self.joint_name_to_info[joint_name].joint_index, singe_joint_state.position)

    def set_base_pose(self, position=(0, 0, 0), orientation=(0, 0, 0, 1)):
        """
        Set base pose in bullet world frame.
        :param position:
        :type position: list
        :param orientation:
        :type orientation: list
        """
        p.resetBasePositionAndOrientation(self.id, position, orientation)

    def get_base_pose(self):
        """
        Retrieves the current base pose of the robot in the PyBullet world.
        :return: Base pose of the robot in the world.
        :rtype: Transform
        """
        [position, orientation] = p.getBasePositionAndOrientation(self.id)
        return Transform(Point(*position), Quaternion(*orientation))

    def sync_with_bullet(self):
        """
        Syncs joint and link infos with bullet
        """
        self.joint_id_map = {}
        self.link_name_to_id = {}
        self.link_id_to_name = {}
        self.joint_name_to_info = OrderedDict()
        self.joint_id_to_info = OrderedDict()
        self.joint_name_to_info[self.base_link_name] = JointInfo(*([-1, self.base_link_name] + [None] * 10 +
                                                                   [self.base_link_name] + [None] * 4))
        self.joint_id_to_info[-1] = JointInfo(*([-1, self.base_link_name] + [None] * 10 +
                                                [self.base_link_name] + [None] * 4))
        self.link_id_to_name[-1] = self.base_link_name
        self.link_name_to_id[self.base_link_name] = -1
        for joint_index in range(p.getNumJoints(self.id)):
            joint_info = JointInfo(*p.getJointInfo(self.id, joint_index))
            self.joint_name_to_info[joint_info.joint_name] = joint_info
            self.joint_id_to_info[joint_info.joint_index] = joint_info
            self.joint_id_map[joint_index] = joint_info.joint_name
            self.joint_id_map[joint_info.joint_name] = joint_index
            self.link_name_to_id[joint_info.link_name] = joint_index
            self.link_id_to_name[joint_index] = joint_info.link_name

    def get_self_collision_matrix(self):
        """
        :return: (link1, link2) -> min allowed distance
        """
        return self.self_collision_matrix

    def get_joint_states(self):
        mjs = dict()
        for joint_info in self.joint_name_to_info.values():
            if joint_info.joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                sjs = SingleJointState()
                sjs.name = joint_info.joint_name
                sjs.position = p.getJointState(self.id, joint_info.joint_index)[0]
                mjs[sjs.name] = sjs
        return mjs

    def calc_collision_matrix(self, link_combinations, d=0.05, d2=0.0, num_rnd_tries=1000):
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
        seed(1337)
        always = set()

        # find meaningless self-collisions
        for link_name_a, link_name_b in link_combinations:
            if self.get_parent_link_name(link_name_a) == link_name_b or \
                    self.get_parent_link_name(link_name_b) == link_name_a:
                # if self.joint_id_to_info[link_name_a].parent_index == link_name_b or \
                #         self.joint_id_to_info[link_name_b].parent_index == link_name_a:
                always.add((link_name_a, link_name_b))
        rest = link_combinations.difference(always)
        self.set_joint_state(self.get_zero_joint_state())
        always = always.union(self._check_collisions(rest, d))
        rest = rest.difference(always)

        # find meaningful self-collisions
        self.set_joint_state(self.get_min_joint_state())
        sometimes = self._check_collisions(rest, d2)
        rest = rest.difference(sometimes)
        self.set_joint_state(self.get_max_joint_state())
        sometimes2 = self._check_collisions(rest, d2)
        rest = rest.difference(sometimes2)
        sometimes = sometimes.union(sometimes2)
        for i in range(num_rnd_tries):
            self.set_joint_state(self.get_rnd_joint_state())
            sometimes2 = self._check_collisions(rest, d2)
            if len(sometimes2) > 0:
                rest = rest.difference(sometimes2)
                sometimes = sometimes.union(sometimes2)
        return sometimes

    def get_parent_link_name(self, child_link_name):
        return self.joint_id_to_info[self.get_link_id(child_link_name)].parent_index

    def get_possible_collisions(self, link):
        # TODO speed up by saving this
        possible_collisions = set()
        for link1, link2 in self.get_self_collision_matrix():
            if link == link1:
                possible_collisions.add(link2)
            elif link == link2:
                possible_collisions.add(link1)
        return possible_collisions

    def _check_collisions(self, link_combinations, d):
        in_collision = set()
        for link_name_1, link_name_2 in link_combinations:
            link_id_1 = self.get_link_id(link_name_1)
            link_id_2 = self.get_link_id(link_name_2)
            if len(p.getClosestPoints(self.id, self.id, d, link_id_1, link_id_2)) > 0:
                in_collision.add((link_name_1, link_name_2))
        return in_collision

    def get_zero_joint_state(self):
        return self.generate_joint_state(lambda x: 0)

    def get_max_joint_state(self):
        return self.generate_joint_state(lambda x: x.joint_upper_limit)

    def get_min_joint_state(self):
        return self.generate_joint_state(lambda x: x.joint_lower_limit)

    def get_rnd_joint_state(self):
        def f(joint_info):
            lower_limit = joint_info.joint_lower_limit
            upper_limit = joint_info.joint_upper_limit
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
        js = {}
        for joint_name in self.controlled_joints:
            joint_info = self.joint_name_to_info[joint_name]
            if joint_info.joint_type in [JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_PLANAR, JOINT_SPHERICAL]:
                sjs = SingleJointState()
                sjs.name = joint_name
                sjs.position = f(joint_info)
                js[joint_name] = sjs
        return js

    def get_link_names(self):
        return self.link_name_to_id.keys()

    def get_link_ids(self):
        return self.link_id_to_name.keys()

    def get_link_name(self, link_id):
        """
        :type link_id: int 
        :rtype: str 
        """
        return self.link_id_to_name[link_id]

    def get_link_id(self, link_name):
        """
        :type link_name: str
        :rtype: int
        """
        return self.link_name_to_id[link_name]

    def get_link_id_to_names(self):
        """
        :rtype: dict
        """
        return self.link_id_to_name

    def has_attached_object(self, object_name):
        """
        Checks whether an object with this name has already been attached to the robot.
        :type object_name: str
        :rtype: bool
        """
        return object_name in self.attached_objects.keys()

    def attach_object(self, object_, parent_link_name, transform):
        """
        Rigidly attach another object to the robot.
        :param object_: Object that shall be attached to the robot.
        :type object_: UrdfObject
        :param parent_link_name: Name of the link to which the object shall be attached.
        :type parent_link_name: str
        :param transform: Hom. transform between the reference frames of the parent link and the object.
        :type transform: Transform
        """
        # TODO should only be called through world because this class does not know which objects exist
        if self.has_attached_object(object_.name):
            # TODO: choose better exception type
            raise DuplicateNameException(
                u'An object \'{}\' has already been attached to the robot.'.format(object_.name))

        # assemble and store URDF string of new link and fixed joint
        self.extend_urdf(object_, transform, parent_link_name)

        # remove last robot and load new robot from new URDF
        self.sync_urdf_with_bullet()

        # update the collision matrix for the newly attached object
        self.add_self_collision_entries(object_.name)
        print(u'object {} attached to {} in pybullet world'.format(object_.name, self.name))

    def add_self_collision_entries(self, object_name):
        link_pairs = {(object_name, link_name) for link_name in self.get_link_names()}
        link_pairs.remove((object_name, object_name))
        self_collision_with_object = self.calc_collision_matrix(link_pairs)
        self.self_collision_matrix.update(self_collision_with_object)

    def remove_self_collision_entries(self, object_name):
        self.self_collision_matrix = {(link1, link2) for link1, link2 in self.get_self_collision_matrix()
                                      if link1 != object_name and link2 != object_name}

    def extend_urdf(self, object_, transform, parent_link_name):
        new_joint = FixedJoint(u'{}_joint'.format(object_.name), transform, parent_link_name,
                               object_.name)
        self.attached_objects[object_.name] = u'{}{}'.format(to_urdf_string(new_joint),
                                                             remove_outer_tag(object_.get_urdf()))

    def sync_urdf_with_bullet(self):
        joint_state = self.get_joint_states()
        base_pose = self.get_base_pose()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.removeBody(self.id)
        self.id = load_urdf_string_into_bullet(self.get_urdf(), base_pose)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        self.sync_with_bullet()
        self.set_joint_state(joint_state)

    def get_urdf(self):
        """
        :rtype: str
        """
        # for each attached object, insert the corresponding URDF sub-string into the original URDF string
        new_urdf_string = self.base_urdf
        for sub_string in self.attached_objects.values():
            new_urdf_string = new_urdf_string.replace(u'</robot>', u'{}</robot>'.format(sub_string))
        return new_urdf_string

    # def as_urdf_object(self):
    #     return urdfob

    def detach_object(self, object_name):
        """
        Detaches an attached object from the robot.
        :param object_name: Name of the object that shall be detached from the robot.
        :type object_name: str
        """
        if not self.has_attached_object(object_name):
            # TODO: choose better exception type
            raise RuntimeError(u"No object '{}' has been attached to the robot.".format(object_name))

        del (self.attached_objects[object_name])

        self.sync_urdf_with_bullet()
        self.remove_self_collision_entries(object_name)
        print(u'object {} detachted from {} in pybullet world'.format(object_name, self.name))

    def detach_all_objects(self):
        """
        Detaches all object that have been attached to the robot.
        """
        if self.attached_objects:
            for attached_object in self.attached_objects:
                self.remove_self_collision_entries(attached_object)
            self.attached_objects = {}
            self.sync_urdf_with_bullet()

    def __str__(self):
        return u'{}/{}'.format(self.name, self.id)


class PyBulletWorld(object):
    """
    Wraps around the shitty pybullet api.
    """

    def __init__(self, enable_gui=False, path_to_data_folder=u''):
        """
        :type enable_gui: bool
        :param path_to_data_folder: location where compiled collision matrices are stored
        :type path_to_data_folder: str
        """
        self._gui = enable_gui
        self._object_names_to_objects = {}
        self._object_id_to_name = {}
        self._robot = None
        self.path_to_data_folder = path_to_data_folder

    def spawn_robot_from_urdf_file(self, robot_name, urdf_file, controlled_joints, base_pose=Transform()):
        """
        Spawns a new robot into the world, reading its URDF from disc.
        :param robot_name: Name of the new robot to spawn.
        :type robot_name: str
        :param urdf_file: Valid and existing filename of the URDF to load, e.g. '/home/foo/bar/pr2.urdf'
        :type urdf_file: str
        :param base_pose: Pose at which to spawn the robot.
        :type base_pose: Transform
        """
        with open(urdf_file, u'r') as f:
            self.spawn_robot_from_urdf(robot_name, f.read(), controlled_joints, base_pose)

    def spawn_robot_from_urdf(self, robot_name, urdf, controlled_joints, base_pose=Transform()):
        """
        :type robot_name: str
        :param urdf: URDF to spawn as loaded XML string.
        :type urdf: str
        :type base_pose: Transform
        """
        if self.has_robot():
            raise RobotExistsException(u'A robot is already loaded')
        if self.has_object(robot_name):
            raise DuplicateNameException(
                u'can\'t add robot; object with name "{}" already exists'.format(robot_name))
        self.deactivate_rendering()
        self._robot = PyBulletRobot(robot_name, urdf, controlled_joints, base_pose,
                                    path_to_data_folder=self.path_to_data_folder)
        self.activate_rendering()

    def spawn_object_from_urdf_str(self, name, urdf, base_pose=Transform()):
        """
        :type name: str
        :param urdf: Path to URDF file, or content of already loaded URDF file.
        :type urdf: str
        :type base_pose: Transform
        """
        if self.has_object(name):
            raise DuplicateNameException(u'object with name "{}" already exists'.format(name))
        if self.has_robot() and self.get_robot().name == name:
            raise DuplicateNameException(u'robot with name "{}" already exists'.format(name))
        self.deactivate_rendering()
        self._object_names_to_objects[name] = PyBulletRobot(name, urdf, [], base_pose, False)
        self._object_id_to_name[self._object_names_to_objects[name].id] = name
        self.activate_rendering()
        print(u'object {} added to pybullet world'.format(name))

    def spawn_object_from_urdf_file(self, object_name, urdf_file, base_pose=Transform()):
        """
        Spawns a new robot into the world, reading its URDF from disc.
        :param robot_name: Name of the new robot to spawn.
        :type robot_name: str
        :param urdf_file: Valid and existing filename of the URDF to load, e.g. '/home/foo/bar/pr2.urdf'
        :type urdf_file: str
        :param base_pose: Pose at which to spawn the robot.
        :type base_pose: Transform
        """
        with open(urdf_file, u'r') as f:
            self.spawn_object_from_urdf_str(object_name, f.read(), base_pose)

    def spawn_urdf_object(self, urdf_object, base_pose=Transform()):
        """
        Spawns a new object into the Bullet world at a given pose.
        :param urdf_object: New object to add to the world.
        :type urdf_object: UrdfObject
        :param base_pose: Pose at which to spawn the object.
        :type base_pose: Transform
        """
        self.spawn_object_from_urdf_str(urdf_object.name, to_urdf_string(urdf_object), base_pose)

    def attach_object(self, object_, parent_link, transform):
        """
        :type object_: UrdfObject
        :type parent_link: str
        :param transform:
        :return:
        """
        if self.has_object(object_.name):
            object_ = self.get_object(object_.name)
            self.delete_object(object_.name)
            # raise DuplicateNameException(
            #     u'Can\'t attach existing object \'{}\'.'.format(object.name))
        self.get_robot().attach_object(object_, parent_link, transform)

    def has_robot(self):
        """
        :rtype: bool
        """
        return self._robot is not None

    def has_object(self, object_name):
        """
        Checks for objects with the same name.
        :type object_name: str
        :rtype: bool
        """
        return object_name in self._object_names_to_objects

    def get_robot(self):
        """
        :rtype: PyBulletRobot
        """
        return self._robot

    def get_object(self, name):
        """
        :type name: str
        :rtype: PyBulletRobot
        """
        return self._object_names_to_objects[name]

    def get_object_names(self):
        """
        :rtype: list
        """
        return list(self._object_names_to_objects.keys())

    def get_object_id(self, name):
        return self._object_names_to_objects[name].id

    def get_object_name(self, id):
        return self._object_id_to_name[id]

    def set_robot_joint_state(self, joint_state):
        """
        Set the current joint state readings for a robot in the world.
        :param joint_state: joint name -> SingleJointState
        :type joint_state: dict
        """
        self._robot.set_joint_state(joint_state)

    def set_object_joint_state(self, object_name, joint_state):
        """
        :type object_name: str
        :param joint_state: joint name -> SingleJointState
        :type joint_state: dict
        """
        self.get_object(object_name).set_joint_state(joint_state)

    def delete_robot(self):
        if self._robot is not None:
            p.removeBody(self._robot.id)
            self._robot = None

    def delete_object(self, object_name):
        """
        Deletes an object with a specific name from the world.
        :type object_name: str
        """
        if not self.has_object(object_name):
            raise UnknownBodyException(u'Cannot delete unknown object {}'.format(object_name))
        self.deactivate_rendering()
        p.removeBody(self.get_object_id(object_name))
        self.activate_rendering()
        del (self._object_id_to_name[self.get_object_id(object_name)])
        del (self._object_names_to_objects[object_name])
        print(u'object {} deleted from pybullet world'.format(object_name))

    def delete_all_objects(self, remaining_objects=(u'plane',)):
        """
        Deletes all objects in world. Optionally, one can specify a list of objects that shall remain in the world.
        :param remaining_objects: Names of objects that shall remain in the world.
        :type remaining_objects: list
        """
        for object_name in self.get_object_names():
            if not object_name in remaining_objects:
                self.delete_object(object_name)

    def check_collisions(self, cut_off_distances):
        """
        :param cut_off_distances: (robot_link, body_b, link_b) -> cut off distance. Contacts between objects not in this
                                    dict or further away than the cut off distance will be ignored.
        :type cut_off_distances: dict
        :param self_collision_d: distances grater than this value will be ignored
        :type self_collision_d: float
        :type enable_self_collision: bool
        :return: (robot_link, body_b, link_b) -> ContactInfo
        :rtype: dict
        """
        # TODO I think I have to multiply distance with something
        collisions = defaultdict(lambda: None)
        for k, distance in cut_off_distances.items():
            (robot_link, body_b, link_b) = k
            robot_link_id = self.get_robot().link_name_to_id[robot_link]
            if self.get_robot().name == body_b:
                object_id = self.get_robot().id
                link_b_id = self.get_robot().link_name_to_id[link_b]
            else:
                object_id = self.get_object_id(body_b)
                link_b_id = self.get_object(body_b).link_name_to_id[link_b]
            # FIXME redundant checks for robot link pairs
            contacts = [ContactInfo(*x) for x in p.getClosestPoints(self._robot.id, object_id,
                                                                    distance * 3,
                                                                    robot_link_id, link_b_id)]
            if len(contacts) > 0:
                collisions.update({k: min(contacts, key=lambda x: x.contact_distance)})
                # asdf = self.should_switch(contacts[0])
                pass
        return collisions

    def should_flip_contact_info(self, contact_info):
        """
        :type contact_info: ContactInfo
        :rtype: bool
        """
        contact_info2 = ContactInfo(*min(p.getClosestPoints(contact_info.body_unique_id_b,
                                                            contact_info.body_unique_id_a,
                                                            abs(contact_info.contact_distance) * 1.05,
                                                            contact_info.link_index_b, contact_info.link_index_a),
                                         key=lambda x: x[8]))
        if not np.isclose(contact_info2.contact_normal_on_b, contact_info.contact_normal_on_b).all():
            return False
        pa = np.array(contact_info.position_on_a)
        # pb = np.array(contact_info.position_on_b)

        self.move_hack(pa)
        try:
            contact_info3 = ContactInfo(*[x for x in p.getClosestPoints(self.get_object_id(u'pybullet_sucks'),
                                                                        contact_info.body_unique_id_a, 0.001) if
                                          np.allclose(x[8], -0.005)][0])
            if contact_info3.body_unique_id_b == contact_info.body_unique_id_a and \
                    contact_info3.link_index_b == contact_info.link_index_a:
                return False
        except Exception as e:
            return True
        return True

    def flip_contact_info(self, contact_info):
        return ContactInfo(contact_info.contact_flag,
                           contact_info.body_unique_id_a, contact_info.body_unique_id_b,
                           contact_info.link_index_a, contact_info.link_index_b,
                           contact_info.position_on_b, contact_info.position_on_a,
                           (-np.array(contact_info.contact_normal_on_b)).tolist(), contact_info.contact_distance,
                           contact_info.normal_force,
                           contact_info.lateralFriction1, contact_info.lateralFrictionDir1,
                           contact_info.lateralFriction2,
                           contact_info.lateralFrictionDir2)

    def activate_viewer(self):
        if self._gui:
            # TODO expose opengl2 option for gui?
            self.physicsClient = p.connect(p.GUI, options=u'--opengl2')  # or p.DIRECT for non-graphical version
        else:
            self.physicsClient = p.connect(p.DIRECT)  # or p.DIRECT for non-graphical version
        p.setGravity(0, 0, -9.8)
        self.add_ground_plane()
        self.add_pybullet_bug_fix_hack()

    def clear_world(self):
        self.delete_all_objects()
        self.delete_robot()
        self.add_ground_plane()
        self.add_pybullet_bug_fix_hack()

    def deactivate_viewer(self):
        p.disconnect()

    def add_ground_plane(self, name=u'plane'):
        """
        Adds a ground plane to the Bullet World.
        """
        # like in the PyBullet examples: spawn a big collision box in the origin
        if not self.has_object(name):
            self.spawn_urdf_object(UrdfObject(name=name,
                                              collision_props=[CollisionProperty(geometry=BoxShape(30, 30, 10))]),
                                   Transform(translation=Point(0, 0, -5)))

    def add_pybullet_bug_fix_hack(self, name=u'pybullet_sucks'):
        """
        Adds a ground plane to the Bullet World.
        """
        # like in the PyBullet examples: spawn a big collision box in the origin
        if not self.has_object(name):
            self.spawn_urdf_object(UrdfObject(name=name,
                                              collision_props=[CollisionProperty(geometry=SphereShape(0.005))]),
                                   Transform(translation=Point(10, 10, -10)))

    def move_hack(self, position):
        self.get_object(u'pybullet_sucks').set_base_pose(position)

    def deactivate_rendering(self):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    def activate_rendering(self):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
