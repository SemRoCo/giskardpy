import numpy as np
from collections import defaultdict

import PyKDL
import pybullet as p
from geometry_msgs.msg import Point, Pose
from giskard_msgs.msg import CollisionEntry

import giskardpy
from giskardpy.data_types import ClosestPointInfo
from giskardpy.pybullet_world_object import PyBulletWorldObject
from giskardpy.pybullet_wrapper import ContactInfo
from giskardpy.tfwrapper import msg_to_kdl
from giskardpy.utils import resolve_ros_iris
from giskardpy.world import World
from giskardpy.world_object import WorldObject

# TODO globally define map

MAP = u'map'


class PyBulletWorld(World):
    """
    Wraps around the shitty pybullet api.
    """
    ground_plane_name = u'ground_plane'
    hack_name = u'pybullet_hack'
    hidden_objects = [ground_plane_name, hack_name]

    def __init__(self, enable_gui=False, path_to_data_folder=u''):
        """
        :type enable_gui: bool
        :param path_to_data_folder: location where compiled collision matrices are stored
        :type path_to_data_folder: str
        """
        super(PyBulletWorld, self).__init__(path_to_data_folder)
        self._gui = enable_gui
        self._object_names_to_objects = {}
        self._object_id_to_name = {}
        self._robot = None
        self.setup()

    def __get_pybullet_object_id(self, name):
        return self.get_object(name).get_pybullet_id()

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
        checked_things = set()
        for k, distance in cut_off_distances.items():
            (robot_link, body_b, link_b) = k
            r_k = (link_b, body_b, robot_link)
            if r_k in checked_things:
                if r_k in collisions:
                    collisions[k] = self.__flip_contact_info(collisions[r_k])
                continue
            robot_link_id = self.robot.get_pybullet_link_id(robot_link)
            if self.robot.get_name() == body_b:
                object_id = self.robot.get_pybullet_id()
                link_b_id = self.robot.get_pybullet_link_id(link_b)
            else:
                object_id = self.__get_pybullet_object_id(body_b)
                if link_b != CollisionEntry.ALL:
                    link_b_id = self.get_object(body_b).get_pybullet_link_id(link_b)
            if body_b == self.robot.get_name() or link_b != CollisionEntry.ALL:
                contacts = [ContactInfo(*x) for x in p.getClosestPoints(self.robot.get_pybullet_id(), object_id,
                                                                        distance * 3,
                                                                        robot_link_id, link_b_id)]
            else:
                contacts = [ContactInfo(*x) for x in p.getClosestPoints(self.robot.get_pybullet_id(), object_id,
                                                                        distance * 3,
                                                                        robot_link_id)]
            if len(contacts) > 0:
                collisions.update({k: min(contacts, key=lambda x: x.contact_distance)})
                pass
            checked_things.add(k)
        return collisions

    def __should_flip_contact_info(self, contact_info):
        """
        :type contact_info: ContactInfo
        :rtype: bool
        """
        # contact_info2 = ContactInfo(*min(p.getClosestPoints(contact_info.body_unique_id_b,
        #                                                     contact_info.body_unique_id_a,
        #                                                     abs(contact_info.contact_distance) * 1.05,
        #                                                     contact_info.link_index_b, contact_info.link_index_a),
        #                                  key=lambda x: x[8]))
        # do i get different results with flipped closest point check
        # if not np.isclose(contact_info2.contact_normal_on_b, contact_info.contact_normal_on_b).all():
        #     return False
        # if they are identical figure out which one is correct
        pa = np.array(contact_info.position_on_a)

        new_p = Pose()
        new_p.position = Point(*pa)
        new_p.orientation.w = 1

        self.__move_hack(new_p)
        try:
            contact_info3 = ContactInfo(
                *[x for x in p.getClosestPoints(self.__get_pybullet_object_id(self.hack_name),
                                                contact_info.body_unique_id_a, 0.001) if
                  abs(x[8]+0.005) < 1e-5][0])
            if contact_info3.body_unique_id_b == contact_info.body_unique_id_a and \
                    contact_info3.link_index_b == contact_info.link_index_a:
                return False
        except Exception as e:
            return True
        return True

    def __flip_contact_info(self, contact_info):
        return ContactInfo(contact_info.contact_flag,
                           contact_info.body_unique_id_b, contact_info.body_unique_id_a,
                           contact_info.link_index_b, contact_info.link_index_a,
                           contact_info.position_on_b, contact_info.position_on_a,
                           [-contact_info.contact_normal_on_b[0],
                            -contact_info.contact_normal_on_b[1],
                            -contact_info.contact_normal_on_b[2]], contact_info.contact_distance,
                           contact_info.normal_force,
                           contact_info.lateralFriction1, contact_info.lateralFrictionDir1,
                           contact_info.lateralFriction2,
                           contact_info.lateralFrictionDir2)

    def setup(self):
        self.__add_ground_plane()
        self.__add_pybullet_bug_fix_hack()

    def soft_reset(self):
        super(PyBulletWorld, self).soft_reset()
        self.__add_ground_plane()
        self.__add_pybullet_bug_fix_hack()

    def __add_ground_plane(self):
        """
        Adds a ground plane to the Bullet World.
        """
        if not self.has_object(self.ground_plane_name):
            path = resolve_ros_iris(u'package://giskardpy/urdfs/ground_plane.urdf')
            plane = WorldObject.from_urdf_file(path)
            plane.set_name(self.ground_plane_name)
            self.add_object(plane)

    def __add_pybullet_bug_fix_hack(self):
        if not self.has_object(self.hack_name):
            path = resolve_ros_iris(u'package://giskardpy/urdfs/tiny_ball.urdf')
            plane = WorldObject.from_urdf_file(path)
            plane.set_name(self.hack_name)
            self.add_object(plane)

    def __move_hack(self, pose):
        self.get_object(self.hack_name).base_pose = pose

    def get_objects(self):
        objects = super(PyBulletWorld, self).get_objects()
        return {k: v for k, v in objects.items() if k not in self.hidden_objects}

    def get_object_names(self):
        return [x for x in super(PyBulletWorld, self).get_object_names() if x not in self.hidden_objects]

    def add_object(self, object_):
        """
        :type object_: giskardpy.world_object.WorldObject
        :return:
        """
        # TODO create from world object to avoid basepose and joint state getting lost?
        pwo = PyBulletWorldObject.from_urdf_object(object_)
        pwo.base_pose = object_.base_pose
        pwo.joint_state = object_.joint_state
        return super(PyBulletWorld, self).add_object(pwo)

    def collisions_to_closest_point(self, collisions, min_allowed_distance):
        """
        :param collisions: (robot_link, body_b, link_b) -> ContactInfo
        :type collisions: dict
        :param min_allowed_distance: (robot_link, body_b, link_b) -> min allowed distance
        :type min_allowed_distance: dict
        :return: robot_link -> ClosestPointInfo of closest thing
        :rtype: dict
        """
        closest_point = super(PyBulletWorld, self).collisions_to_closest_point(collisions, min_allowed_distance)
        for key, cpi in closest_point.items():  # type: (str, ClosestPointInfo)
            if self.__should_flip_contact_info(collisions[cpi.old_key]):
                root_T_link = msg_to_kdl(self.robot.get_fk_pose(self.robot.get_root(), cpi.link_a))
                b_link = PyKDL.Vector(*cpi.position_on_a)
                a_root = PyKDL.Vector(*cpi.position_on_b)
                b_root = root_T_link * b_link
                a_link = root_T_link.Inverse() * a_root
                closest_point[key] = ClosestPointInfo(a_link, b_root, cpi.contact_distance,
                                                      cpi.min_dist,
                                                      cpi.link_a,
                                                      cpi.body_b,
                                                      cpi.link_b,
                                                      -np.array(cpi.contact_normal), key)
        return closest_point

    def remove_robot(self):
        self.robot.suicide()
        super(PyBulletWorld, self).remove_robot()
