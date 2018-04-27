import pybullet as p
import rospkg
from collections import namedtuple, OrderedDict, defaultdict
from itertools import combinations
from pybullet import JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_PLANAR, JOINT_SPHERICAL

import pybullet_data
from numpy.random.mtrand import seed

from giskardpy.trajectory import MultiJointState, SingleJointState
import numpy as np

from giskardpy.utils import keydefaultdict

JointInfo = namedtuple('JointInfo', ['joint_index', 'joint_name', 'joint_type', 'q_index', 'u_index', 'flags',
                                     'joint_damping', 'joint_friction', 'joint_lower_limit', 'joint_upper_limit',
                                     'joint_max_force', 'joint_max_velocity', 'link_name', 'joint_axis',
                                     'parent_frame_pos', 'parent_frame_orn', 'parent_index'])

ContactInfo = namedtuple('ContactInfo', ['contact_flag', 'body_unique_id_a', 'body_unique_id_b', 'link_index_a',
                                         'link_index_b', 'position_on_a', 'position_on_b', 'contact_normal_on_b',
                                         'contact_distance', 'normal_force'])


def replace_paths(urdf_str):
    rospack = rospkg.RosPack()
    with open('/tmp/robot.urdf', 'w') as o:
        for line in urdf_str.split('\n'):
            if 'package://' in line:
                package_name = line.split('package://', 1)[-1].split('/', 1)[0]
                real_path = rospack.get_path(package_name)
                o.write(line.replace(package_name, real_path))
            else:
                o.write(line)


class PyBulletRobot(object):
    def __init__(self, name, urdf_string, base_position=(0, 0, 0), base_orientation=(0, 0, 0, 1)):
        self.name = name
        replace_paths(urdf_string)
        self.id = p.loadURDF('/tmp/robot.urdf', base_position, base_orientation,
                             flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
        self.init_js_info()

    def set_joint_state(self, multi_joint_state):
        for joint_name, singe_joint_state in multi_joint_state.items():
            p.resetJointState(self.id, self.joint_name_to_info[joint_name].joint_index, singe_joint_state.position)

    def init_js_info(self):
        self.joint_id_map = {}
        self.link_id_map = {}
        self.joint_name_to_info = OrderedDict()
        self.joint_id_to_info = OrderedDict()
        # self.ignored_collisions = defaultdict(bool)
        self.joint_name_to_info['base'] = JointInfo(*([-1, 'base'] + [None] * 10 + ['base'] + [None] * 4))
        self.joint_id_to_info[-1] = JointInfo(*([-1, 'base'] + [None] * 10 + ['base'] + [None] * 4))
        self.link_id_map[-1] = 'base'
        self.link_id_map['base'] = -1
        for joint_index in range(p.getNumJoints(self.id)):
            joint_info = JointInfo(*p.getJointInfo(self.id, joint_index))
            self.joint_name_to_info[joint_info.joint_name] = joint_info
            self.joint_id_to_info[joint_info.joint_index] = joint_info
            self.joint_id_map[joint_index] = joint_info.joint_name
            self.joint_id_map[joint_info.joint_name] = joint_index
            self.link_id_map[joint_info.link_name] = joint_index
            self.link_id_map[joint_index] = joint_info.link_name
        self.generate_self_collision_matrix()

    def check_self_collision(self, d=0.5, whitelist=None):
        if whitelist is None:
            whitelist = self.sometimes
        o = (0, 0, 0)
        contact_infos = keydefaultdict(lambda k: ContactInfo(None, self.id, self.id, k[0], k[1], o, o, o, 1e9, 0))
        contact_infos.update({(self.link_id_map[link_a], self.link_id_map[link_b]): ContactInfo(*x)
                              for (link_a, link_b) in whitelist for x in p.getClosestPoints(self.id, self.id, d, link_a, link_b)})
        contact_infos.update({(link_b, link_a): ContactInfo(ci.contact_flag, ci.body_unique_id_a, ci.body_unique_id_b,
                                                            ci.link_index_b, ci.link_index_a, ci.position_on_b,
                                                            ci.position_on_a,
                                                            [-x for x in ci.contact_normal_on_b], ci.contact_distance,
                                                            ci.normal_force)
                              for (link_a, link_b), ci in contact_infos.items()})
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


class PyBulletWorld(object):
    def __init__(self, gui=False):
        self.gui = gui
        self._objects = {}
        self._robots = {}

    def spawn_urdf_str_robot(self, robot_name, urdf_string, base_position=(0, 0, 0), base_orientation=(0, 0, 0, 1)):
        self._deactivate_rendering()
        self._robots[robot_name] = PyBulletRobot(robot_name, urdf_string, base_position, base_orientation)
        self._activate_rendering()

    def spawn_urdf_file_robot(self, robot_name, urdf_file, base_position=(0, 0, 0), base_orientation=(0, 0, 0, 1)):
        with open(urdf_file, 'r') as f:
            urdf_string = f.read().replace('\n', '')
        self.spawn_urdf_str_robot(robot_name, urdf_string, base_position, base_orientation)

    def get_robot_list(self):
        return list(self._robots.keys())

    def get_robot(self, name):
        return self._robots[name]

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

    def spawn_object_from_urdf(self, name, urdf):
        self._deactivate_rendering()
        self._objects[name] = p.loadURDF(urdf)
        self._activate_rendering()

    def get_object_list(self):
        return list(self._objects.keys())

    def delete_object(self, object_name):
        p.removeBody(self._objects[object_name])

    def attach_object(self):
        pass

    def release_object(self):
        pass

    def check_collision(self):
        collisions = None
        for robot in self._robots.values():
            if collisions is None:
                collisions = robot.check_self_collision()
            else:
                collisions.update(robot.check_self_collision())
        return collisions

    def check_trajectory_collision(self):
        pass

    def activate_viewer(self):
        if self.gui:
            self.physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        else:
            self.physicsClient = p.connect(p.DIRECT)  # or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        print(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        self.spawn_object_from_urdf('plane', 'plane.urdf')

    def clear_world(self):
        for i in range(p.getNumBodies()):
            p.removeBody(p.getBodyUniqueId(i))

    def deactivate_viewer(self):
        p.disconnect()

    def _deactivate_rendering(self):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    def _activate_rendering(self):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
