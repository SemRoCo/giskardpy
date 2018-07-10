import pickle
import pybullet as p
import rospkg
import string
import random
import os
from collections import namedtuple, OrderedDict, defaultdict
from itertools import combinations
from pybullet import JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_PLANAR, JOINT_SPHERICAL
from time import time

import errno
from numpy.random.mtrand import seed

from giskardpy.exceptions import UnknownBodyException, DuplicateObjectNameException, DuplicateRobotNameException
from giskardpy.data_types import MultiJointState, SingleJointState, Transform, Point, Quaternion
import numpy as np

from giskardpy.utils import keydefaultdict

from giskardpy.object import WorldObject, FixedJoint, from_msg, to_urdf_string, BoxShape, CollisionProperty
import hashlib

JointInfo = namedtuple('JointInfo', ['joint_index', 'joint_name', 'joint_type', 'q_index', 'u_index', 'flags',
                                     'joint_damping', 'joint_friction', 'joint_lower_limit', 'joint_upper_limit',
                                     'joint_max_force', 'joint_max_velocity', 'link_name', 'joint_axis',
                                     'parent_frame_pos', 'parent_frame_orn', 'parent_index'])

ContactInfo = namedtuple('ContactInfo', ['contact_flag', 'body_unique_id_a', 'body_unique_id_b', 'link_index_a',
                                         'link_index_b', 'position_on_a', 'position_on_b', 'contact_normal_on_b',
                                         'contact_distance', 'normal_force'])


def resolve_ros_iris(input_urdf):
    """
    Replace all instances of ROS IRIs with a urdf string with global paths in the file system.
    :param input_urdf: URDF in which the ROS IRIs shall be replaced.
    :type input_urdf: str
    :return: URDF with replaced ROS IRIs.
    :rtype: str
    """
    rospack = rospkg.RosPack()
    output_urdf = ''
    for line in input_urdf.split('\n'):
        if 'package://' in line:
            package_name = line.split('package://', 1)[-1].split('/', 1)[0]
            real_path = rospack.get_path(package_name)
            output_urdf += line.replace(package_name, real_path)
        else:
            output_urdf += line
        output_urdf += '\n'
    return output_urdf


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
    new_path = '/tmp/{}'.format(filename)
    with open(new_path, 'w') as o:
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
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(size))


def load_urdf_string_into_bullet(urdf_string, pose):
    """
    Loads a URDF string into the bullet world.
    :param urdf_string: XML string of the URDF to load.
    :type urdf_string: str
    :param pose: Pose at which to load the URDF into the world.
    :type pose: Transform
    :return: internal PyBullet id of the loaded urdf
    :rtype: int
    """
    filename = write_urdf_to_disc('{}.urdf'.format(random_string()), urdf_string)
    id = p.loadURDF(filename, [pose.translation.x, pose.translation.y, pose.translation.z],
                    [pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w],
                    flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
    os.remove(filename)
    return id


class PyBulletRobot(object):
    def __init__(self, name, urdf, base_pose=Transform(), calc_self_collision_matrix=True, path_to_data_folder=''):
        """

        :param name:
        :param urdf: Path to URDF file, or content of already loaded URDF file.
        :type urdf: str
        :param base_pose:
        :type base_pose: Transform
        """
        self.path_to_data_folder = path_to_data_folder
        self.name = name
        self.original_urdf = resolve_ros_iris(urdf)
        self.id = load_urdf_string_into_bullet(self.original_urdf, base_pose)
        self.init_js_info()
        self.attached_objects = {}
        if calc_self_collision_matrix:
            if not self.load_self_collision_matrix():
                self.sometimes = self.calc_self_collision_matrix(set(combinations(self.joint_id_to_info.keys(), 2)))
                self.save_self_collision_matrix()
        else:
            self.sometimes = set()

    def load_self_collision_matrix(self):
        urdf_hash = hashlib.md5(self.original_urdf).hexdigest()
        path = self.path_to_data_folder + urdf_hash
        if os.path.isfile(path):
            with open(path) as f:
                self.sometimes = pickle.load(f)
                print('loaded self collision matrix {}'.format(urdf_hash))
                return True
        return False

    def save_self_collision_matrix(self):
        urdf_hash = hashlib.md5(self.original_urdf).hexdigest()
        path = self.path_to_data_folder + urdf_hash
        if not os.path.exists(os.path.dirname(path)):
            try:
                dir_name = os.path.dirname(path)
                if dir_name != '':
                    os.makedirs(dir_name)
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        with open(path, 'w') as file:
            print('saved self collision matrix {}'.format(path))
            pickle.dump(self.sometimes, file)

    def set_joint_state(self, multi_joint_state):
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

    def init_js_info(self):
        self.joint_id_map = {}
        self.link_name_to_id = {}
        self.link_id_to_name = {}
        self.joint_name_to_info = OrderedDict()
        self.joint_id_to_info = OrderedDict()
        self.joint_name_to_info['base'] = JointInfo(*([-1, 'base'] + [None] * 10 + ['base'] + [None] * 4))
        self.joint_id_to_info[-1] = JointInfo(*([-1, 'base'] + [None] * 10 + ['base'] + [None] * 4))
        self.link_id_to_name[-1] = 'base'
        self.link_name_to_id['base'] = -1
        for joint_index in range(p.getNumJoints(self.id)):
            joint_info = JointInfo(*p.getJointInfo(self.id, joint_index))
            self.joint_name_to_info[joint_info.joint_name] = joint_info
            self.joint_id_to_info[joint_info.joint_index] = joint_info
            self.joint_id_map[joint_index] = joint_info.joint_name
            self.joint_id_map[joint_info.joint_name] = joint_index
            self.link_name_to_id[joint_info.link_name] = joint_index
            self.link_id_to_name[joint_index] = joint_info.link_name

    def check_self_collision(self, d=0.2, whitelist=None):
        if whitelist is None:
            whitelist = self.sometimes

        def default_contact_info(k):
            return ContactInfo(None, self.id, self.id, k[0], k[1], (0, 0, 0), (0, 0, 0), (0, 0, 0), 1e9, 0)

        contact_infos = keydefaultdict(default_contact_info)
        contact_infos.update({(self.link_id_to_name[link_a], self.name, self.link_id_to_name[link_b]): ContactInfo(*x)
                              for (link_a, link_b) in whitelist for x in
                              p.getClosestPoints(self.id, self.id, d, link_a, link_b)})
        contact_infos.update({(link_b, name, link_a): ContactInfo(ci.contact_flag, ci.body_unique_id_a,
                                                                  ci.body_unique_id_b, ci.link_index_b,
                                                                  ci.link_index_a, ci.position_on_b,
                                                                  ci.position_on_a,
                                                                  [-x for x in ci.contact_normal_on_b],
                                                                  ci.contact_distance,
                                                                  ci.normal_force)
                              for (link_a, name, link_b), ci in contact_infos.items()})
        return contact_infos

    def get_joint_states(self):
        mjs = MultiJointState()
        for joint_info in self.joint_name_to_info.values():
            if joint_info.joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                sjs = SingleJointState()
                sjs.name = joint_info.joint_name
                sjs.position = p.getJointState(self.id, joint_info.joint_index)[0]
                mjs.set(sjs)
        return mjs

    def calc_self_collision_matrix(self, combis, d=0.05, d2=0.0, num_rnd_tries=1000):
        # TODO computational expansive because of too many collision checks
        print('calculating self collision matrix')
        seed(1337)
        always = set()

        # find meaningless self-collisions
        for link_a, link_b in combis:
            if self.joint_id_to_info[link_a].parent_index == link_b or \
                    self.joint_id_to_info[link_b].parent_index == link_a:
                always.add((link_a, link_b))
        rest = combis.difference(always)
        always = always.union(self._check_all_collisions(rest, d, self.get_zero_joint_state()))
        rest = rest.difference(always)

        # find meaningful self-collisions
        sometimes = self._check_all_collisions(rest, d2, self.get_min_joint_state())
        rest = rest.difference(sometimes)
        sometimes2 = self._check_all_collisions(rest, d2, self.get_max_joint_state())
        rest = rest.difference(sometimes2)
        sometimes = sometimes.union(sometimes2)
        for i in range(num_rnd_tries):
            sometimes2 = self._check_all_collisions(rest, d2, self.get_rnd_joint_state())
            if len(sometimes2) > 0:
                rest = rest.difference(sometimes2)
                sometimes = sometimes.union(sometimes2)
        return sometimes

    def _check_all_collisions(self, test_links, d, js):
        self.set_joint_state(js)
        sometimes = set()
        for link_a, link_b in test_links:
            if len(p.getClosestPoints(self.id, self.id, d, link_a, link_b)) > 0:
                sometimes.add((link_a, link_b))
        return sometimes

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
        for joint_name, joint_info in self.joint_name_to_info.items():
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

    def has_attached_object(self, object_name):
        """
        Checks whether an object has already been attached to the robot.
        :param object_name: Name of the object for which to check.
        :type object_name: str
        :return: True if one of the attached objects has that name, else False.
        :rtype: bool
        """
        return object_name in self.attached_objects.keys()

    def attach_object(self, object, parent_link_name, transform):
        """
        Rigidly attach another object to the robot.
        :param object: Object that shall be attached to the robot.
        :type object: WorldObject
        :param parent_link_name: Name of the link to which the object shall be attached.
        :type parent_link_name: str
        :param transform: Hom. transform between the reference frames of the parent link and the object.
        :type Transform
        :return: Nothing
        """
        if self.has_attached_object(object.name):
            # TODO: choose better exception type
            raise DuplicateObjectNameException(
                "An object '{}' has already been attached to the robot.".format(object.name))

        # salvage last joint state and base pose
        joint_state = self.get_joint_states()
        base_pose = self.get_base_pose()

        # salvage last collision matrix, and save collisions as pairs of link names
        collision_matrix = set()
        for collision in self.sometimes:
            collision_matrix.add((self.link_id_to_name[collision[0]], self.link_id_to_name[collision[1]]))

        # assemble and store URDF string of new link and fixed joint
        new_joint = FixedJoint('{}_joint'.format(object.name), transform, parent_link_name,
                               object.name)
        self.attached_objects[object.name] = '{}{}'.format(to_urdf_string(new_joint), to_urdf_string(object, True))

        new_urdf_string = self.get_urdf()

        # remove last robot and load new robot from new URDF
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.removeBody(self.id)
        self.id = load_urdf_string_into_bullet(new_urdf_string, base_pose)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        # reload joint info and last joint state
        self.init_js_info()
        self.set_joint_state(joint_state)

        # reload last collision matrix as pairs of link IDs
        self.sometimes = set()
        for collision in collision_matrix:
            self.sometimes.add((self.link_name_to_id[collision[0]], self.link_name_to_id[collision[1]]))

        # update the collision matrix for the newly attached object
        object_id = self.link_name_to_id[object.name]
        link_pairs = {(object_id, link_id) for link_id in self.joint_id_to_info.keys()}
        new_collisions = self.calc_self_collision_matrix(link_pairs)
        self.sometimes.union(new_collisions)
        print('object {} attached to {} in pybullet world'.format(object.name, self.name))

    def get_urdf(self):
        # for each attached object, insert the corresponding URDF sub-string into the original URDF string
        new_urdf_string = self.original_urdf
        for sub_string in self.attached_objects.values():
            new_urdf_string = new_urdf_string.replace('</robot>', '{}</robot>'.format(sub_string))
        return new_urdf_string

    def detach_object(self, object_name):
        """
        Detaches an attached object from the robot.
        :param object_name: Name of the object that shall be detached from the robot.
        :type object_name: str
        :return: Nothing.
        """
        if not self.has_attached_object(object_name):
            # TODO: choose better exception type
            raise RuntimeError("No object '{}' has been attached to the robot.".format(object_name))

        # salvage last joint state and base pose
        base_pose = self.get_base_pose()
        joint_state = self.get_joint_states()

        # salvage last collision matrix, and save collisions as pairs of link names
        collision_matrix = set()
        for collision in self.sometimes:
            collision_matrix.add((self.link_id_to_name[collision[0]], self.link_id_to_name[collision[1]]))

        # remove all collision entries related to the object that shall be detached
        collision_matrix = filter((lambda (collision): object_name not in collision), collision_matrix)

        # forget about the object that shall be detached
        del (self.attached_objects[object_name])

        # for each attached object, insert the corresponding URDF sub-string into the original URDF string
        new_urdf_string = self.get_urdf()

        # remove last robot and load new robot from new URDF
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.removeBody(self.id)
        self.id = load_urdf_string_into_bullet(new_urdf_string, base_pose)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        # reload joint info and last joint state
        self.init_js_info()
        self.set_joint_state(joint_state)

        # reload last collision matrix as pairs of link IDs
        self.sometimes = set()
        for collision in collision_matrix:
            self.sometimes.add((self.link_name_to_id[collision[0]], self.link_name_to_id[collision[1]]))
        print('object {} detachted from {} in pybullet world'.format(object_name, self.name))

    def detach_all_objects(self):
        """
        Detaches all object that have been attached to the robot.
        :return: Nothing.
        """
        if self.attached_objects:
            # salvage last joint state and base pose
            base_pose = self.get_base_pose()
            joint_state = self.get_joint_states()

            # salvage last collision matrix, and save collisions as pairs of link names
            collision_matrix = set()
            for collision in self.sometimes:
                collision_matrix.add((self.link_id_to_name[collision[0]], self.link_id_to_name[collision[1]]))

            # remove all collision entries related to attached objects
            for object_name in self.attached_objects.keys():
                collision_matrix = filter((lambda (collision): object_name not in collision), collision_matrix)

            # remove last robot and reload with original URDF
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.removeBody(self.id)
            self.id = load_urdf_string_into_bullet(self.original_urdf, base_pose)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

            # reload joint info and last joint state
            self.init_js_info()
            self.set_joint_state(joint_state)

            # reload original collision matrix
            self.sometimes = set()
            for collision in collision_matrix:
                self.sometimes.add((self.link_name_to_id[collision[0]], self.link_name_to_id[collision[1]]))

            # forget about previously attached objects
            self.attached_objects = {}

    def __str__(self):
        return '{}/{}'.format(self.name, self.id)


class PyBulletWorld(object):
    def __init__(self, gui=False, path_to_data_folder=''):
        self._gui = gui
        self._objects = {}
        self._robot = None
        self.path_to_data_folder = path_to_data_folder

    def spawn_robot_from_urdf_file(self, robot_name, urdf_file, base_pose=Transform()):
        """
        Spawns a new robot into the world, reading its URDF from disc.
        :param robot_name: Name of the new robot to spawn.
        :type robot_name: str
        :param urdf_file: Valid and existing filename of the URDF to load, e.g. '/home/foo/bar/pr2.urdf'
        :type urdf_file: str
        :param base_pose: Pose at which to spawn the robot.
        :type base_pose: Transform
        :return: Nothing
        """
        with open(urdf_file, 'r') as f:
            self.spawn_robot_from_urdf(robot_name, f.read(), base_pose)

    def spawn_robot_from_urdf(self, robot_name, urdf, base_pose=Transform()):
        """

        :param robot_name:
        :param urdf: URDF to spawn as loaded XML string.
        :type urdf: str
        :param base_pose:
        :type base_pose: Transform
        :return:
        """
        if self.has_robot():
            raise Exception('A robot is already loaded')
        self.deactivate_rendering()
        self._robot = PyBulletRobot(robot_name, urdf, base_pose, path_to_data_folder=self.path_to_data_folder)
        self.activate_rendering()

    def spawn_object_from_urdf_file(self, object_name, urdf_file, base_pose=Transform()):
        """
        Spawns a new robot into the world, reading its URDF from disc.
        :param robot_name: Name of the new robot to spawn.
        :type robot_name: str
        :param urdf_file: Valid and existing filename of the URDF to load, e.g. '/home/foo/bar/pr2.urdf'
        :type urdf_file: str
        :param base_pose: Pose at which to spawn the robot.
        :type base_pose: Transform
        :return: Nothing
        """
        with open(urdf_file, 'r') as f:
            self.spawn_object_from_urdf(object_name, f.read(), base_pose)

    def spawn_object_from_urdf(self, name, urdf, base_pose=Transform()):
        """

        :param name:
        :param urdf: Path to URDF file, or content of already loaded URDF file.
        :type urdf: str
        :param base_pose:
        :type base_pose: Transform
        :return:
        """
        if self.has_object(name):
            raise DuplicateObjectNameException('Cannot spawn object "{}" because an object with such a '
                                               'name already exists'.format(name))
        self.deactivate_rendering()
        self._objects[name] = PyBulletRobot(name, urdf, base_pose, False)
        self.activate_rendering()

    def get_robot(self):
        """
        :param robot_name:
        :type robot_name: str
        :return:
        :rtype: PyBulletRobot
        """
        return self._robot

    def has_robot(self):
        """
        Checks whether this world already contains a robot with a specific name.
        :param robot_name: Identifier of the robot that shall be checked.
        :type robot_name: str
        :return: True if robot with that name is already in the world. Else: returns False.
        """
        return self._robot is not None

    def get_object(self, name):
        """
        :param name:
        :type name: str
        :return:
        :rtype: PyBulletRobot
        """
        return self._objects[name]

    def set_robot_joint_state(self, joint_state):
        """
        Set the current joint state readings for a robot in the world.
        :param robot_name: name of the robot to update
        :type string
        :param joint_state: sensor readings for the entire robot
        :type dict{string, MultiJointState}
        """
        self._robot.set_joint_state(joint_state)

    def get_robot_joint_state(self):
        return self._robot.get_joint_states()

    def set_object_joint_state(self, object_name, joint_state):
        self.get_object(object_name).set_joint_state(joint_state)

    def get_object_joint_state(self, object_name):
        return self.get_object(object_name).get_joint_states()

    def delete_robot(self):
        if self._robot is not None:
            p.removeBody(self._robot.id)
            self._robot = None

    def spawn_object(self, object, base_pose=Transform()):
        """
        Spawns a new object into the Bullet world at a given pose.
        :param object: New object to add to the world.
        :type object: WorldObject
        :param base_pose: Pose at which to spawn the object.
        :type base_pose: Transform
        :return: Nothing.
        """
        self.spawn_object_from_urdf(object.name, to_urdf_string(object), base_pose)
        print('object {} added to pybullet world'.format(object.name))

    def get_object_list(self):
        return list(self._objects.keys())

    def delete_object(self, object_name):
        """
        Deletes an object with a specific name from the world.
        :param object_name: Name of the object that shall be deleted.
        :type object_name: str
        """
        if not self.has_object(object_name):
            raise UnknownBodyException('Cannot delete unknown object {}'.format(object_name))
        self.deactivate_rendering()
        p.removeBody(self._objects[object_name].id)
        self.activate_rendering()
        del (self._objects[object_name])
        print('object {} deleted from pybullet world'.format(object_name))

    def delete_all_objects(self, remaining_objects=['plane']):
        """
        Deletes all objects in world. Optionally, one can specify a list of objects that shall remain in the world.
        :param remaining_objects: Names of objects that shall remain in the world.
        :type remaining_objects: list
        """
        for object_name in self.get_object_list():
            if not object_name in remaining_objects:
                self.delete_object(object_name)

    def has_object(self, object_name):
        """
        Checks whether this world already contains an object with a specific name.
        :param object_name: Identifier of the object that shall be checked.
        :type object_name: str
        :return: True if object with that name is already in the world. Else: returns False.
        """
        return object_name in self._objects.keys()

    def check_collisions(self, cut_off_distances, allowed_collision=set(), self_collision_d=0.1, self_collision=True,
                         controllable_links=None):
        """
        :param cut_off_distances:
        :type cut_off_distances: dict
        :param self_collision:
        :type self_collision: bool
        :return:
        :rtype: dict
        """
        # TODO implement a cooler way to remove wheel/plane collisions but detect eg. arm/plane collisions
        allowed_collision.add('plane')

        # TODO set self collision cut off distance in a cool way
        def default_contact_info(k):
            # TODO I think this is buggy but never gets called
            return ContactInfo(None, -2, -2, k[0], k[1], (0, 0, 0), (0, 0, 0), (1, 0, 0), 1e9, 0)

        collisions = keydefaultdict(default_contact_info)
        if self_collision and self.get_robot().name not in allowed_collision:
            # TODO use cut_off_distances in self collision
            if collisions is None:
                collisions = self._robot.check_self_collision(self_collision_d)
            else:
                collisions.update(self._robot.check_self_collision(self_collision_d))
        for object_name, object in self._objects.items():  # type: (str, PyBulletRobot)
            if object_name not in allowed_collision:
                for robot_link_name, robot_link in self._robot.link_name_to_id.items():
                    if controllable_links is not None and robot_link_name not in controllable_links:
                        continue
                    for object_link_name, object_link in object.link_name_to_id.items():
                        key = (robot_link_name, object_name, object_link_name)
                        if key not in allowed_collision:
                            contacts = [ContactInfo(*x) for x in p.getClosestPoints(self._robot.id, object.id,
                                                                                    cut_off_distances[
                                                                                        key] + self_collision_d,
                                                                                    robot_link, object_link)]
                            if len(contacts) > 0:
                                collisions.update({key: min(contacts, key=lambda x: x.contact_distance)})
        return collisions

    def check_trajectory_collision(self):
        pass

    def activate_viewer(self):
        if self._gui:
            # TODO expose opengl2 option for gui?
            self.physicsClient = p.connect(p.GUI, options='--opengl2')  # or p.DIRECT for non-graphical version
        else:
            self.physicsClient = p.connect(p.DIRECT)  # or p.DIRECT for non-graphical version
        p.setGravity(0, 0, -9.8)
        self.add_ground_plane()

    def clear_world(self):
        self.delete_all_objects()
        self.delete_robot()
        # self.delete_all_robots()

    def deactivate_viewer(self):
        p.disconnect()

    def add_ground_plane(self, name='plane'):
        """
        Adds a ground plane to the Bullet World.
        :return: Nothing.
        """
        # like in the PyBullet examples: spawn a big collision box in the origin
        if not self.has_object(name):
            self.spawn_object(WorldObject(name=name,
                                          collision_props=[CollisionProperty(geometry=BoxShape(30, 30, 10))]),
                              Transform(translation=Point(0, 0, -5)))

    def deactivate_rendering(self):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    def activate_rendering(self):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
