import PyKDL
from geometry_msgs.msg import PoseStamped
from giskard_msgs.msg import CollisionEntry

from giskardpy.data_types import ClosestPointInfo
from giskardpy.exceptions import RobotExistsException, DuplicateNameException, PhysicsWorldException, \
    UnknownBodyException, UnsupportedOptionException
from giskardpy.symengine_robot import Robot
from giskardpy.tfwrapper import msg_to_kdl, kdl_to_pose, np_to_kdl, to_np
from giskardpy.urdf_object import URDFObject, FIXED_JOINT
from giskardpy.utils import KeyDefaultDict, np_point, np_vector
from giskardpy.world_object import WorldObject
from giskardpy import logging
import numpy as np

class World(object):
    def __init__(self, path_to_data_folder=u''):
        self._objects = {}
        self._robot = None  # type: Robot
        if path_to_data_folder is None:
            path_to_data_folder = u''
        self._path_to_data_folder = path_to_data_folder

    # General ----------------------------------------------------------------------------------------------------------
    
    def soft_reset(self):
        """
        keeps robot and other important objects like ground plane
        """
        self.remove_all_objects()
        if self._robot is not None:
            self._robot.reset()

    def hard_reset(self):
        """
        removes everything
        """
        self.soft_reset()
        self.remove_robot()

    def check_collisions(self, cut_off_distances):
        pass

    # Objects ----------------------------------------------------------------------------------------------------------

    def add_object(self, object_):
        """
        :type object_: URDFObject
        """
        # FIXME this interface seems unintuitive, why not pass base pose as well?
        if self.has_robot() and self.robot.get_name() == object_.get_name():
            raise DuplicateNameException(u'object and robot have the same name')
        if self.has_object(object_.get_name()):
            raise DuplicateNameException(u'object with that name already exists')
        self._objects[object_.get_name()] = object_
        logging.loginfo(u'--> added {} to world'.format(object_.get_name()))

    def set_object_pose(self, name, pose):
        """
        :type pose: Pose
        :return:
        """
        self.get_object(name).base_pose = pose

    def get_object(self, name):
        """
        :type name: str
        :rtype: WorldObject
        """
        return self._objects[name]

    def get_objects(self):
        return self._objects

    def get_object_names(self):
        """
        :rtype: list
        """
        return list(self._objects.keys())

    def has_object(self, name):
        """
        Checks for objects with the same name.
        :type name: str
        :rtype: bool
        """
        return name in self.get_objects()

    def set_object_joint_state(self, name, joint_state):
        """
        :type name: str
        :param joint_state: joint name -> SingleJointState
        :type joint_state: dict
        """
        self.get_object(name).joint_state = joint_state

    def remove_object(self, name):
        if self.has_object(name):
            self._objects[name].suicide()
            logging.loginfo(u'<-- removed object {} to world'.format(name))
            del (self._objects[name])
        else:
            raise UnknownBodyException(u'can\'t remove object \'{}\', because it doesn\' exist'.format(name))

    def remove_all_objects(self):
        for object_ in self._objects.values():
            object_.suicide()
        self._objects = {}

    # Robot ------------------------------------------------------------------------------------------------------------

    def add_robot(self, robot, base_pose, controlled_joints, joint_vel_limit, joint_acc_limit, joint_weights,
                  calc_self_collision_matrix, ignored_pairs, added_pairs):
        """
        :type robot: giskardpy.world_object.WorldObject
        :type controlled_joints: list
        :type base_pose: PoseStamped
        """
        if not isinstance(robot, WorldObject):
            raise TypeError(u'only WorldObject can be added to world')
        if self.has_robot():
            raise RobotExistsException(u'A robot is already loaded')
        if self.has_object(robot.get_name()):
            raise DuplicateNameException(
                u'can\'t add robot; object with name "{}" already exists'.format(robot.get_name()))
        self._robot = Robot.from_urdf_object(urdf_object=robot,
                                             base_pose=base_pose,
                                             controlled_joints=controlled_joints,
                                             path_to_data_folder=self._path_to_data_folder,
                                             joint_vel_limit=joint_vel_limit,
                                             joint_acc_limit=joint_acc_limit,
                                             calc_self_collision_matrix=calc_self_collision_matrix,
                                             joint_weights=joint_weights,
                                             ignored_pairs=ignored_pairs,
                                             added_pairs=added_pairs)

    @property
    def robot(self):
        """
        :rtype: Robot
        """
        return self._robot

    def has_robot(self):
        """
        :rtype: bool
        """
        return self._robot is not None

    def set_robot_joint_state(self, joint_state):
        """
        Set the current joint state readings for a robot in the world.
        :param joint_state: joint name -> SingleJointState
        :type joint_state: dict
        """
        self._robot.joint_state = joint_state

    def remove_robot(self):
        self._robot = None

    def attach_existing_obj_to_robot(self, name, link, pose):
        """
        :param name: name of the existing object
        :type name: name
        """
        # TODO this should know the object pose and not require it as input
        self._robot.attach_urdf_object(self.get_object(name), link, pose)
        self.remove_object(name)

    def detach(self, joint_name, from_obj=None):
        if joint_name not in self.robot.get_joint_names():
            raise UnknownBodyException(u'can\'t detach: {}'.format(joint_name))
        if from_obj is None or self.robot.get_name() == from_obj:
            # this only works because attached simple objects have joint names equal to their name
            p = self.robot.get_fk_pose(self.robot.get_root(), joint_name)
            p_map = kdl_to_pose(self.robot.root_T_map.Inverse() * msg_to_kdl(p))

            cut_off_obj = self.robot.detach_sub_tree(joint_name)
        else:
            raise UnsupportedOptionException(u'only detach from robot supported')
        wo = WorldObject.from_urdf_object(cut_off_obj)  # type: WorldObject
        wo.base_pose = p_map
        self.add_object(wo)

    def get_robot_collision_matrix(self, min_dist):
        robot_name = self.robot.get_name()
        collision_matrix = self.robot.get_self_collision_matrix()
        collision_matrix2 = {}
        for link1, link2 in collision_matrix:
            # FIXME should I use the minimum of both distances?
            if link1 < link2:
                collision_matrix2[link1, robot_name, link2] = min_dist[link1][u'zero_weight_distance']
            else:
                collision_matrix2[link2, robot_name, link1] = min_dist[link1][u'zero_weight_distance']
        return collision_matrix2

    def collision_goals_to_collision_matrix(self, collision_goals, min_dist):
        """
        :param collision_goals: list of CollisionEntry
        :type collision_goals: list
        :return: dict mapping (robot_link, body_b, link_b) -> min allowed distance
        :rtype: dict
        """
        collision_goals = self.verify_collision_entries(collision_goals, min_dist)
        min_allowed_distance = {}
        for collision_entry in collision_goals:  # type: CollisionEntry
            if self.is_avoid_all_self_collision(collision_entry):
                min_allowed_distance.update(self.get_robot_collision_matrix(min_dist))
                continue
            assert len(collision_entry.robot_links) == 1
            assert len(collision_entry.link_bs) == 1
            key = (collision_entry.robot_links[0], collision_entry.body_b, collision_entry.link_bs[0])
            if self.is_allow_collision(collision_entry):
                if self.all_link_bs(collision_entry):
                    for key2 in list(min_allowed_distance.keys()):
                        if key[0] == key2[0] and key[1] == key2[1]:
                            del min_allowed_distance[key2]
                elif key in min_allowed_distance:
                    del min_allowed_distance[key]

            elif self.is_avoid_collision(collision_entry):
                min_allowed_distance[key] = min_dist[key[0]][u'zero_weight_distance']
            else:
                raise Exception('todo')
        return min_allowed_distance

    def verify_collision_entries(self, collision_goals, min_dist):
        for ce in collision_goals:  # type: CollisionEntry
            if ce.type in [CollisionEntry.ALLOW_ALL_COLLISIONS,
                           CollisionEntry.AVOID_ALL_COLLISIONS]:
                # logging.logwarn(u'ALLOW_ALL_COLLISIONS and AVOID_ALL_COLLISIONS deprecated, use AVOID_COLLISIONS and'
                #               u'ALLOW_COLLISIONS instead with ALL constant instead.')
                if ce.type == CollisionEntry.ALLOW_ALL_COLLISIONS:
                    ce.type = CollisionEntry.ALLOW_COLLISION
                else:
                    ce.type = CollisionEntry.AVOID_COLLISION

        for ce in collision_goals:  # type: CollisionEntry
            if CollisionEntry.ALL in ce.robot_links and len(ce.robot_links) != 1:
                raise PhysicsWorldException(u'ALL used in robot_links, but it\'s not the only entry')
            if CollisionEntry.ALL in ce.link_bs and len(ce.link_bs) != 1:
                raise PhysicsWorldException(u'ALL used in link_bs, but it\'s not the only entry')
            if ce.body_b == CollisionEntry.ALL and not self.all_link_bs(ce):
                raise PhysicsWorldException(u'if body_b == ALL, link_bs has to be ALL as well')

        self.are_entries_known(collision_goals)

        for ce in collision_goals:
            if not ce.robot_links:
                ce.robot_links = [CollisionEntry.ALL]
            if not ce.link_bs:
                ce.link_bs = [CollisionEntry.ALL]

        for i, ce in enumerate(reversed(collision_goals)):
            if self.is_avoid_all_collision(ce):
                collision_goals = collision_goals[len(collision_goals) - i - 1:]
                break
            if self.is_allow_all_collision(ce):
                collision_goals = collision_goals[len(collision_goals) - i:]
                break
        else:
            ce = CollisionEntry()
            ce.type = CollisionEntry.AVOID_COLLISION
            ce.robot_links = [CollisionEntry.ALL]
            ce.body_b = CollisionEntry.ALL
            ce.link_bs = [CollisionEntry.ALL]
            ce.min_dist = -1
            collision_goals.insert(0, ce)

        # split body bs
        collision_goals = self.split_body_b(collision_goals)

        # split robot links
        collision_goals = self.robot_related_stuff(collision_goals)

        # split link_bs
        collision_goals = self.split_link_bs(collision_goals)

        return collision_goals

    def are_entries_known(self, collision_goals):
        robot_name = self.robot.get_name()
        robot_links = set(self.robot.get_link_names())
        for collision_entry in collision_goals:
            if not (collision_entry.body_b == robot_name or
                    collision_entry.body_b in self.get_object_names() or
                    self.all_body_bs(collision_entry)):
                raise UnknownBodyException(u'body b \'{}\' unknown'.format(collision_entry.body_b))
            if not self.all_robot_links(collision_entry):
                for robot_link in collision_entry.robot_links:
                    if robot_link not in robot_links:
                        raise UnknownBodyException(u'robot link \'{}\' unknown'.format(robot_link))
            if collision_entry.body_b == robot_name:
                for robot_link in collision_entry.link_bs:
                    if robot_link != CollisionEntry.ALL and robot_link not in robot_links:
                        raise UnknownBodyException(u'link b \'{}\' of body \'{}\' unknown'.format(robot_link, collision_entry.body_b))
            elif not self.all_body_bs(collision_entry) and not self.all_link_bs(collision_entry):
                object_links = self.get_object(collision_entry.body_b).get_link_names()
                for link_b in collision_entry.link_bs:
                    if link_b not in object_links:
                        raise UnknownBodyException(u'link b \'{}\' of body \'{}\' unknown'.format(link_b, collision_entry.body_b))


    def split_link_bs(self, collision_goals):
        # FIXME remove the side effects of these three methods
        i = 0
        while i < len(collision_goals):
            collision_entry = collision_goals[i]
            if self.is_avoid_all_self_collision(collision_entry):
                i += 1
                continue
            if self.all_link_bs(collision_entry):
                if collision_entry.body_b == self.robot.get_name():
                    new_ces = []
                    link_bs = self.robot.get_possible_collisions(list(collision_entry.robot_links)[0])
                elif [x for x in collision_goals[i:] if
                      x.robot_links == collision_entry.robot_links and
                      x.body_b == collision_entry.body_b and not self.all_link_bs(x)]:
                    new_ces = []
                    link_bs = self.get_object(collision_entry.body_b).get_link_names_with_collision()
                else:
                    i += 1
                    continue
                collision_goals.remove(collision_entry)
                for link_b in link_bs:
                    ce = CollisionEntry()
                    ce.type = collision_entry.type
                    ce.robot_links = collision_entry.robot_links
                    ce.body_b = collision_entry.body_b
                    ce.min_dist = collision_entry.min_dist
                    ce.link_bs = [link_b]
                    new_ces.append(ce)
                for new_ce in new_ces:
                    collision_goals.insert(i, new_ce)
                i += len(new_ces)
                continue
            elif len(collision_entry.link_bs) > 1:
                collision_goals.remove(collision_entry)
                for link_b in collision_entry.link_bs:
                    ce = CollisionEntry()
                    ce.type = collision_entry.type
                    ce.robot_links = collision_entry.robot_links
                    ce.body_b = collision_entry.body_b
                    ce.link_bs = [link_b]
                    ce.min_dist = collision_entry.min_dist
                    collision_goals.insert(i, ce)
                i += len(collision_entry.link_bs)
                continue
            i += 1
        return collision_goals

    def robot_related_stuff(self, collision_goals):
        i = 0
        controlled_robot_links = self.robot.get_controlled_links()
        while i < len(collision_goals):
            collision_entry = collision_goals[i]
            if self.is_avoid_all_self_collision(collision_entry):
                i += 1
                continue
            if self.all_robot_links(collision_entry):
                collision_goals.remove(collision_entry)

                new_ces = []
                for robot_link in controlled_robot_links:
                    ce = CollisionEntry()
                    ce.type = collision_entry.type
                    ce.robot_links = [robot_link]
                    ce.body_b = collision_entry.body_b
                    ce.min_dist = collision_entry.min_dist
                    ce.link_bs = collision_entry.link_bs
                    new_ces.append(ce)

                for new_ce in new_ces:
                    collision_goals.insert(i, new_ce)
                i += len(new_ces)
                continue
            elif len(collision_entry.robot_links) > 1:
                collision_goals.remove(collision_entry)
                for robot_link in collision_entry.robot_links:
                    ce = CollisionEntry()
                    ce.type = collision_entry.type
                    ce.robot_links = [robot_link]
                    ce.body_b = collision_entry.body_b
                    ce.min_dist = collision_entry.min_dist
                    ce.link_bs = collision_entry.link_bs
                    collision_goals.insert(i, ce)
                i += len(collision_entry.robot_links)
                continue
            i += 1
        return collision_goals

    def split_body_b(self, collision_goals):
        i = 0
        while i < len(collision_goals):
            collision_entry = collision_goals[i]
            if self.all_body_bs(collision_entry):
                collision_goals.remove(collision_entry)
                new_ces = []
                for body_b in [self.robot.get_name()] + self.get_object_names():
                    ce = CollisionEntry()
                    ce.type = collision_entry.type
                    ce.robot_links = collision_entry.robot_links
                    ce.min_dist = collision_entry.min_dist
                    ce.body_b = body_b
                    ce.link_bs = collision_entry.link_bs
                    new_ces.append(ce)
                for new_ce in reversed(new_ces):
                    collision_goals.insert(i, new_ce)
                i += len(new_ces)
                continue
            i += 1
        return collision_goals

    def all_robot_links(self, collision_entry):
        return CollisionEntry.ALL in collision_entry.robot_links and len(collision_entry.robot_links) == 1

    def all_link_bs(self, collision_entry):
        return CollisionEntry.ALL in collision_entry.link_bs and len(collision_entry.link_bs) == 1 or \
               not collision_entry.link_bs

    def all_body_bs(self, collision_entry):
        return collision_entry.body_b == CollisionEntry.ALL

    def is_avoid_collision(self, collision_entry):
        return collision_entry.type in [CollisionEntry.AVOID_COLLISION, CollisionEntry.AVOID_ALL_COLLISIONS]

    def is_allow_collision(self, collision_entry):
        return collision_entry.type in [CollisionEntry.ALLOW_COLLISION, CollisionEntry.ALLOW_ALL_COLLISIONS]

    def is_avoid_all_collision(self, collision_entry):
        """
        :type collision_entry: CollisionEntry
        :return: bool
        """
        return self.is_avoid_collision(collision_entry) \
               and self.all_robot_links(collision_entry) \
               and self.all_body_bs(collision_entry) \
               and self.all_link_bs(collision_entry)

    def is_avoid_all_self_collision(self, collision_entry):
        """
        :type collision_entry: CollisionEntry
        :return: bool
        """
        return self.is_avoid_collision(collision_entry) \
               and self.all_robot_links(collision_entry) \
               and collision_entry.body_b == self.robot.get_name() \
               and self.all_link_bs(collision_entry)

    def is_allow_all_collision(self, collision_entry):
        """
        :type collision_entry: CollisionEntry
        :return: bool
        """
        return self.is_allow_collision(collision_entry) \
               and self.all_robot_links(collision_entry) \
               and self.all_body_bs(collision_entry) \
               and self.all_link_bs(collision_entry)

    
    def transform_contact_info(self, collisions):
        root_T_map = to_np(self.robot.root_T_map)
        robot_root = self.robot.get_root()
        for contact_info in collisions.items():  # type: ClosestPointInfo
            movable_joint = self.robot.get_controlled_parent_joint(contact_info.link_a)
            f = self.robot.get_child_link_of_joint(movable_joint)
            f_T_r = self.robot.get_fk_np(f, robot_root)
            contact_info.frame = f

            f_P_pa = np.dot(np.dot(f_T_r, root_T_map), np_point(*contact_info.position_on_a))
            r_P_pb = np.dot(root_T_map, np_point(*contact_info.position_on_b))
            r_V_n = np.dot(root_T_map, np_vector(*contact_info.contact_normal))
            contact_info.position_on_a = f_P_pa[:-1]
            contact_info.position_on_b = r_P_pb[:-1]
            contact_info.contact_normal = r_V_n[:-1]
        return collisions
