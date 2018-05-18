import pybullet as p
import rospkg
from collections import namedtuple, OrderedDict, defaultdict
from itertools import combinations
from pybullet import JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_PLANAR, JOINT_SPHERICAL
from time import time

import pybullet_data
from copy import deepcopy
from numpy.random.mtrand import seed

from giskardpy.exceptions import UnknownBodyException, DuplicateObjectNameException, DuplicateRobotNameException
from giskardpy.trajectory import MultiJointState, SingleJointState, Transform
import numpy as np

from giskardpy.utils import keydefaultdict

from giskardpy.object import WorldObject, FixedJoint, from_msg, to_urdf_string

JointInfo = namedtuple('JointInfo', ['joint_index', 'joint_name', 'joint_type', 'q_index', 'u_index', 'flags',
                                     'joint_damping', 'joint_friction', 'joint_lower_limit', 'joint_upper_limit',
                                     'joint_max_force', 'joint_max_velocity', 'link_name', 'joint_axis',
                                     'parent_frame_pos', 'parent_frame_orn', 'parent_index'])

ContactInfo = namedtuple('ContactInfo', ['contact_flag', 'body_unique_id_a', 'body_unique_id_b', 'link_index_a',
                                         'link_index_b', 'position_on_a', 'position_on_b', 'contact_normal_on_b',
                                         'contact_distance', 'normal_force'])


def replace_paths(urdf, name):
    new_path = '/tmp/{}.urdf'.format(name)
    with open(new_path, 'w') as o:
        if urdf.endswith('.urdf'):
            try:
                # TODO find cleaner solution
                with open(urdf, 'r') as f:
                    urdf = f.read()
            except IOError:
                return urdf
        o.write(resolve_ros_iris(urdf))
    return new_path


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
    return output_urdf


class PyBulletRobot(object):
    def __init__(self, name, urdf, base_pose=Transform()):
        """

        :param name:
        :param urdf: Path to URDF file, or content of already loaded URDF file.
        :type urdf: str
        :param base_pose:
        :type base_pose: Transform
        """
        self.name = name
        self.original_urdf = replace_paths(urdf, name)
        self.id = p.loadURDF(self.original_urdf, [base_pose.translation.x, base_pose.translation.y, base_pose.translation.z],
                             [base_pose.rotation.x, base_pose.rotation.y, base_pose.rotation.z, base_pose.rotation.w],
                             flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
        self.sometimes = set()
        self.init_js_info()
        self.attached_objects = {}
        self.original_collision_matrix = deepcopy(self.sometimes) # TODO: translate from IDs to names

    def set_joint_state(self, multi_joint_state):
        for joint_name, singe_joint_state in multi_joint_state.items():
            p.resetJointState(self.id, self.joint_name_to_info[joint_name].joint_index, singe_joint_state.position)

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
        self.generate_self_collision_matrix()

    def check_self_collision(self, d=0.5, whitelist=None):
        if whitelist is None:
            whitelist = self.sometimes

        def default_contact_info(k):
            return ContactInfo(None, self.id, self.id, k[0], k[1], (0, 0, 0), (0, 0, 0), (0, 0, 0), 1e9, 0)
        contact_infos = keydefaultdict(default_contact_info)
        contact_infos.update({(self.link_id_to_name[link_a], self.name, self.link_id_to_name[link_b]): ContactInfo(*x)
                              for (link_a, link_b) in whitelist for x in
                              p.getClosestPoints(self.id, self.id, d, link_a, link_b)})
        contact_infos.update({(link_b, name, link_a): ContactInfo(ci.contact_flag, ci.body_unique_id_a, ci.body_unique_id_b,
                                                            ci.link_index_b, ci.link_index_a, ci.position_on_b,
                                                            ci.position_on_a,
                                                            [-x for x in ci.contact_normal_on_b], ci.contact_distance,
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

    def generate_self_collision_matrix(self, d=0.05, num_rnd_tries=200):
        # TODO computational expansive because of too many collision checks
        seed(1337)
        always = set()
        self.all = set(combinations(self.joint_id_to_info.keys(), 2))
        for link_a, link_b in self.all:
            if self.joint_id_to_info[link_a].parent_index == link_b or \
                    self.joint_id_to_info[link_b].parent_index == link_a:
                always.add((link_a, link_b))
        rest = self.all.difference(always)
        always = always.union(self._check_all_collisions(rest, d, self.get_zero_joint_state()))
        rest = rest.difference(always)
        sometimes = self._check_all_collisions(rest, d, self.get_min_joint_state())
        rest = rest.difference(sometimes)
        sometimes2 = self._check_all_collisions(rest, d, self.get_max_joint_state())
        rest = rest.difference(sometimes2)
        sometimes = sometimes.union(sometimes2)
        for i in range(num_rnd_tries):
            sometimes2 = self._check_all_collisions(rest, d, self.get_rnd_joint_state())
            if len(sometimes2) > 0:
                rest = rest.difference(sometimes2)
                sometimes = sometimes.union(sometimes2)
        self.sometimes = sometimes
        self.never = rest

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
            raise RuntimeError("An object '{}' has already been attached to the robot.".format(object.name))

        # remember last joint state and base pose
        # TODO: implement me

        # assemble and store URDF string of new link and fixed joint
        new_joint = FixedJoint('{}_joint'.format(object.name), transform, parent_link_name, '{}_link'.format(object.name))
        self.attached_objects[object.name] = '{}{}'.format(to_urdf_string(new_joint), to_urdf_string(object, True))

        # for each attached object, insert the corresponding URDF sub-string into the original URDF string
        new_urdf_string = self.original_urdf
        for sub_string in self.attached_objects.values():
            new_urdf_string = new_urdf_string.replace('</robot>', '{}</robot>'.format(sub_string))

        # remove last robot and load new robot from new URDF
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.removeBody(self.id)
        self.id = p.loadURDF(new_urdf_string, (0,0,0), (0,0,0,1), flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        # reload joint info
        # TODO: implement me

        # salvage last joint state and base pose
        # TODO: implement me

        # salvage original collision matrix
        # TODO: implement me

        # for each attached object, extend collision matrix
        # TODO: implement me

    def detach_object(self, object_name):
        pass

    def detach_all_objects(self):
        """

        :return: Nothing.
        """
        if self.attached_objects:
            # TODO: remember last joint state and base pose

            # remove last robot and reload with original URDF
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.removeBody(self.id)
            self.id = p.loadURDF(self.original_urdf, (0, 0, 0), (0, 0, 0, 1), flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)


            # TODO: reload joint info

            # TODO: salve last joint state and base pose

            # TODO: salve original collision matrix

            # forget about previously attached objects
            self.attached_objects = {}

    def __str__(self):
        return '{}/{}'.format(self.name, self.id)


class PyBulletWorld(object):
    def __init__(self, gui=False):
        self.gui = gui
        self._objects = {}
        self._robots = {}

    def spawn_robot_from_urdf(self, robot_name, urdf, base_pose=Transform()):
        """

        :param robot_name:
        :param urdf: Path to URDF file, or content of already loaded URDF file.
        :type urdf: str
        :param base_pose:
        :type base_pose: Transform
        :return:
        """
        if self.has_robot(robot_name):
            raise DuplicateRobotNameException('Cannot spawn robot "{}" because a robot with such a '
                                               'name already exists'.format(robot_name))
        self.deactivate_rendering()
        self._robots[robot_name] = PyBulletRobot(robot_name, urdf, base_pose)
        self.activate_rendering()

    def get_robot_list(self):
        return list(self._robots.keys())

    def get_robot(self, robot_name):
        """
        :param robot_name:
        :type robot_name: str
        :return:
        :rtype: PyBulletRobot
        """
        return self._robots[robot_name]

    def has_robot(self, robot_name):
        """
        Checks whether this world already contains a robot with a specific name.
        :param robot_name: Identifier of the robot that shall be checked.
        :type robot_name: str
        :return: True if robot with that name is already in the world. Else: returns False.
        """
        return robot_name in self._robots.keys()

    def get_object(self, name):
        """
        :param name:
        :type name: str
        :return:
        :rtype: PyBulletRobot
        """
        return self._objects[name]

    def set_joint_state(self, robot_name, joint_state):
        """
        Set the current joint state readings for a robot in the world.
        :param robot_name: name of the robot to update
        :type string
        :param joint_state: sensor readings for the entire robot
        :type dict{string, MultiJointState}
        """
        self._robots[robot_name].set_joint_state(joint_state)

    def get_joint_state(self, robot_name):
        return self._robots[robot_name].get_joint_states()

    def delete_robot(self, robot_name):
        p.removeBody(self._robots[robot_name].id)
        del (self._robots[robot_name])

    # TODO: refactor into using Transform
    def spawn_object_from_urdf(self, name, urdf, base_pose=Transform()):
        """

        :param name:
        :param urdf: Path to URDF file, or content of already loaded URDF file.
        :type urdf: str
        :param base_position:
        :param base_orientation:
        :return:
        """
        if self.has_object(name):
            raise DuplicateObjectNameException('Cannot spawn object "{}" because an object with such a '
                                               'name already exists'.format(name))
        self.deactivate_rendering()
        self._objects[name] = PyBulletRobot(name, urdf, base_pose)
        self.activate_rendering()

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
        p.removeBody(self._objects[object_name].id)
        del (self._objects[object_name])

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

    def check_collisions(self, cut_off_distances=None, allowed_collision=set(), d=0.1, self_collision=True):
        """
        :param cut_off_distances:
        :type cut_off_distances: dict
        :param self_collision:
        :type self_collision: bool
        :return:
        :rtype: dict
        """
        if cut_off_distances is None:
            cut_off_distances = defaultdict(lambda: d)
        # TODO set self collision cut off distance in a cool way
        def default_contact_info(k):
            return ContactInfo(None, self.id, self.id, k[0], k[1], (0, 0, 0), (0, 0, 0), (0, 0, 0), 1e9, 0)
        collisions = keydefaultdict(default_contact_info)
        if self_collision:
            for robot in self._robots.values():  # type: PyBulletRobot
                if collisions is None:
                    collisions = robot.check_self_collision(d)
                else:
                    collisions.update(robot.check_self_collision(d))
        # TODO robot/robot collisions are missing
        for robot_name, robot in self._robots.items():  # type: (str, PyBulletRobot)
            for robot_link_name, robot_link in robot.link_name_to_id.items():
                # TODO skip if collisions with all links of an object are allowed
                for object_name, object in self._objects.items():  # type: (str, PyBulletRobot)
                    for object_link_name, object_link in object.link_name_to_id.items():
                        key = (robot_link_name, object_name, object_link_name)
                        if key not in allowed_collision:
                            collisions.update({key: ContactInfo(*x) for x in
                                               p.getClosestPoints(robot.id, object.id, cut_off_distances[key] + d,
                                                                  robot_link, object_link)})
        return collisions

    def check_trajectory_collision(self):
        pass

    def activate_viewer(self):
        if self.gui:
            self.physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        else:
            self.physicsClient = p.connect(p.DIRECT)  # or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        # print(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        self.spawn_object_from_urdf('plane', 'plane.urdf')

    def clear_world(self):
        for i in range(p.getNumBodies()):
            p.removeBody(p.getBodyUniqueId(i))

    def deactivate_viewer(self):
        p.disconnect()

    def add_ground_plane(self):
        pass

    def muh(self, gui=True):
        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

    def deactivate_rendering(self):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    def activate_rendering(self):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
