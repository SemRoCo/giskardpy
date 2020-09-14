import pybullet as p
from geometry_msgs.msg import Point, Pose
from giskard_msgs.msg import CollisionEntry
from pybullet import error

import giskardpy
from giskardpy.data_types import Collision, Collisions
from giskardpy.exceptions import CorruptShapeException
from giskardpy.pybullet_world_object import PyBulletWorldObject
from giskardpy.pybullet_wrapper import ContactInfo
from giskardpy.utils import resolve_ros_iris
from giskardpy.world import World
from giskardpy.world_object import WorldObject


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

    # @profile
    def check_collisions(self, cut_off_distances, collision_list_size=15):
        """
        :param cut_off_distances: (robot_link, body_b, link_b) -> cut off distance. Contacts between objects not in this
                                    dict or further away than the cut off distance will be ignored.
        :type cut_off_distances: dict
        :param self_collision_d: distances grater than this value will be ignored
        :type self_collision_d: float
        :type enable_self_collision: bool
        :return: (robot_link, body_b, link_b) -> Collision
        :rtype: Collisions
        """
        collisions = Collisions(self.robot, collision_list_size)
        robot_name = self.robot.get_name()
        for (robot_link, body_b, link_b), distance in cut_off_distances.items():
            if robot_name == body_b:
                object_id = self.robot.get_pybullet_id()
                link_b_id = self.robot.get_pybullet_link_id(link_b)
            else:
                object_id = self.__get_pybullet_object_id(body_b)
                if link_b != CollisionEntry.ALL:
                    link_b_id = self.get_object(body_b).get_pybullet_link_id(link_b)

            robot_link_id = self.robot.get_pybullet_link_id(robot_link)
            if body_b == robot_name or link_b != CollisionEntry.ALL:
                contacts = [ContactInfo(*x) for x in p.getClosestPoints(self.robot.get_pybullet_id(), object_id,
                                                                        distance * 1.1,
                                                                        robot_link_id, link_b_id)]
            else:
                contacts = [ContactInfo(*x) for x in p.getClosestPoints(self.robot.get_pybullet_id(), object_id,
                                                                        distance * 1.1,
                                                                        robot_link_id)]
            if len(contacts) > 0:
                try:
                    body_b_object = self.get_object(body_b)
                except KeyError:
                    body_b_object = self.robot
                for contact in contacts:  # type: ContactInfo
                    if link_b == CollisionEntry.ALL:
                        link_b = body_b_object.pybullet_link_id_to_name(contact.link_index_b)
                    if self.__should_flip_collision(contact.position_on_a, robot_link):
                        flipped_normal = [-contact.contact_normal_on_b[0],
                                          -contact.contact_normal_on_b[1],
                                          -contact.contact_normal_on_b[2]]
                        collision = Collision(robot_link, body_b, link_b,
                                              contact.position_on_b, contact.position_on_a,
                                              flipped_normal, contact.contact_distance)
                        collisions.add(collision)
                    else:
                        collision = Collision(robot_link, body_b, link_b,
                                              contact.position_on_a, contact.position_on_b,
                                              contact.contact_normal_on_b, contact.contact_distance)
                        collisions.add(collision)
        return collisions

    def __should_flip_collision(self, position_on_a_in_map, link_a):
        """
        :type collision: ContactInfo
        :rtype: bool
        """
        new_p = Pose()
        new_p.position = Point(*position_on_a_in_map)
        new_p.orientation.w = 1

        self.__move_hack(new_p)
        hack_id = self.__get_pybullet_object_id(self.hack_name)
        body_a_id = self.robot.get_pybullet_id()
        try:
            contact_info3 = ContactInfo(
                *[x for x in p.getClosestPoints(hack_id,
                                                body_a_id, 0.001) if
                  abs(x[8] + 0.005) < 0.0005][0])
            return not (contact_info3.body_unique_id_b == body_a_id and
                        contact_info3.link_index_b == self.robot.get_pybullet_link_id(link_a))
        except Exception as e:
            return True

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
        try:
            pwo = PyBulletWorldObject.from_urdf_object(object_)
            pwo.base_pose = object_.base_pose
            pwo.joint_state = object_.joint_state
        except Exception as e:
            if isinstance(e, error):
                raise CorruptShapeException(e)
            raise e
        return super(PyBulletWorld, self).add_object(pwo)

    def remove_robot(self):
        self.robot.suicide()
        super(PyBulletWorld, self).remove_robot()
