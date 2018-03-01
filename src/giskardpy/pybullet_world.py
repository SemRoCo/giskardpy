import pybullet as p
import rospkg
from collections import namedtuple, OrderedDict, defaultdict

import pybullet_data

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

    def set_joint_state(self, joint_state):
        for i, joint_name in enumerate(joint_state.name):
            p.resetJointState(self.id, self.joint_name_to_info[joint_name].joint_index, joint_state.position[i])

    def init_js_info(self):
        self.joint_name_to_info = OrderedDict()
        self.joint_id_to_info = OrderedDict()
        self.ignored_collisions = defaultdict(bool)
        self.joint_name_to_info['base'] = JointInfo(*([-1,'base']+[None]*15))
        self.joint_id_to_info[-1] = JointInfo(*([-1,'base']+[None]*15))
        for joint_index in range(p.getNumJoints(self.id)):
            joint_info = JointInfo(*p.getJointInfo(self.id, joint_index))
            self.joint_name_to_info[joint_info.joint_name] = joint_info
            self.joint_id_to_info[joint_info.joint_index] = joint_info
        initial_distances = self.check_self_collision(0.05)
        for (link_a, link_b) in initial_distances:
            self.ignored_collisions[link_a, link_b] = True
            self.ignored_collisions[link_b, link_a] = True

    def check_self_collision(self, d=0.02):
        contact_infos = [ContactInfo(*x) for x in p.getClosestPoints(self.id, self.id, d)]
        distances = {}
        for ci in contact_infos:
            link_a = self.joint_id_to_info[ci.link_index_a].link_name
            link_b = self.joint_id_to_info[ci.link_index_b].link_name
            if ci.link_index_a != ci.link_index_b and not self.ignored_collisions[link_a, link_b]:
                distances[link_a, link_b] = ci.contact_distance
        return distances

class PyBulletWorld(object):
    def __init__(self):
        self._objects = {}
        self._robots = {}

    def spawn_urdf_robot(self, urdf_string, robot_name, base_position=(0, 0, 0), base_orientation=(0, 0, 0, 1)):

        self._deactivate_rendering()
        self._robots[robot_name] = PyBulletRobot(robot_name, urdf_string, base_position, base_orientation)
        self._activate_rendering()

    def get_robot_list(self):
        return list(self._robots.keys())

    def set_joint_state(self, robot_name, joint_state):
        """
        Set the current joint state readings for a robot in the world.
        :param robot_name: name of the robot to update
        :type string
        :param joint_state: sensor readings for the entire robot
        :type dict{string, JointState}
        """
        self._robots[robot_name].set_joint_state(joint_state)

    def delete_robot(self, robot_name):
        p.removeBody(self._robots[robot_name].id)

    def spawn_object_from_urdf(self, name, urdf):
        self._deactivate_rendering()
        self._objects[name] = p.loadURDF(urdf)
        self._activate_rendering()

    def get_object_list(self):
        return self._objects

    def delete_object(self, object_name):
        p.removeBody(self._objects[object_name])

    def attach_object(self):
        pass

    def release_object(self):
        pass

    def check_collision(self):
        collisions = {}
        for robot in self._robots.values():
            collisions.update(robot.check_self_collision())
        return collisions


    def check_trajectory_collision(self):
        pass

    def activate_viewer(self):
        self.physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -9.8)
        self.spawn_object_from_urdf('plane', 'plane.urdf')

    def deactivate_viewer(self):
        p.disconnect()

    def _deactivate_rendering(self):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    def _activate_rendering(self):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
