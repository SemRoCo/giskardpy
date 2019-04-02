import PyKDL
import rospy
from geometry_msgs.msg import PoseStamped
from giskard_msgs.msg import CollisionEntry

from giskardpy.data_types import ClosestPointInfo
from giskardpy.exceptions import RobotExistsException, DuplicateNameException, PhysicsWorldException, \
    UnknownBodyException, UnsupportedOptionException
from giskardpy.symengine_robot import Robot
from giskardpy.tfwrapper import msg_to_kdl, kdl_to_pose
from giskardpy.urdf_object import URDFObject
from giskardpy.utils import keydefaultdict
from giskardpy.world_object import WorldObject


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
        if self.has_robot() and self.robot.get_name() == object_.get_name():
            raise DuplicateNameException(u'object and robot have the same name')
        if self.has_object(object_.get_name()):
            raise DuplicateNameException(u'object with that name already exists')
        self._objects[object_.get_name()] = object_
        print(u'--> added {} to world'.format(object_.get_name()))

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
            print(u'<-- removed object {} to world'.format(name))
            del (self._objects[name])
        else:
            raise UnknownBodyException(u'can\'t remove object \'{}\', because it doesn\' exist'.format(name))

    def remove_all_objects(self):
        for object_ in self._objects.values():
            object_.suicide()
        self._objects = {}

    # Robot ------------------------------------------------------------------------------------------------------------

    def add_robot(self, robot, base_pose, controlled_joints, default_joint_vel_limit, default_joint_weight,
                  calc_self_collision_matrix):
        """
        :type robot: giskardpy.world_object.WorldObject
        :type controlled_joints: list
        :type base_pose: PoseStamped
        """
        if self.has_robot():
            raise RobotExistsException(u'A robot is already loaded')
        if self.has_object(robot.get_name()):
            raise DuplicateNameException(
                u'can\'t add robot; object with name "{}" already exists'.format(robot.get_name()))
        self._robot = Robot.from_urdf_object(robot, base_pose, controlled_joints, self._path_to_data_folder,
                                             default_joint_vel_limit, default_joint_weight, calc_self_collision_matrix)

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
        self._robot.attach_urdf_object(self.get_object(name), link, pose)
        self.remove_object(name)

    def detach(self, joint_name, from_obj=None):
        if from_obj is None or self.robot.get_name() == from_obj:
            # this only works because attached simple objects have joint names equal to their name
            p = self.robot.get_fk(self.robot.get_root(), joint_name)
            p_map = kdl_to_pose(self.robot.T_base___map.Inverse() * msg_to_kdl(p))

            cut_off_obj = self.robot.detach_sub_tree(joint_name)
        else:
            raise UnsupportedOptionException(u'only detach from robot supported')
        wo = WorldObject.from_urdf_object(cut_off_obj)  # type: WorldObject
        wo.base_pose = p_map
        self.add_object(wo)

    def get_robot_collision_matrix(self, min_dist):
        robot_name = self.robot.get_name()
        collision_matrix = self.robot.get_self_collision_matrix()
        return {(link1, robot_name, link2): min_dist for link1, link2 in collision_matrix}

    def collision_goals_to_collision_matrix(self, collision_goals, min_dist):
        """
        :param collision_goals: list of CollisionEntry
        :type collision_goals: list
        :return: dict mapping (robot_link, body_b, link_b) -> min allowed distance
        :rtype: dict
        """
        min_allowed_distance = {}
        collision_goals = self.verify_collision_entries(collision_goals, min_dist)
        for collision_entry in collision_goals:  # type: CollisionEntry
            assert len(collision_entry.robot_links) == 1
            assert len(collision_entry.link_bs) == 1
            key = (collision_entry.robot_links.pop(), collision_entry.body_b, collision_entry.link_bs.pop())
            if self.is_allow_collision(collision_entry):
                if self.all_link_bs(collision_entry):
                    for key2 in list(min_allowed_distance.keys()):
                        if key[0] == key2[0] and key[1] == key2[1]:
                            del min_allowed_distance[key2]
                elif key in min_allowed_distance:
                    del min_allowed_distance[key]

            elif self.is_avoid_collision(collision_entry):
                    min_allowed_distance[key] = collision_entry.min_dist
            else:
                raise Exception('todo')
        return min_allowed_distance

        # # TODO split this into smaller functions
        # robot_name = self.robot.get_name()
        # robot_links = self.robot.get_link_names_with_collision()
        #
        # collision_matrix = self.get_robot_collision_matrix(min_dist)
        # min_allowed_distance = collision_matrix
        #
        # if len([x for x in collision_goals if x.type in [CollisionEntry.AVOID_ALL_COLLISIONS,
        #                                                  CollisionEntry.ALLOW_ALL_COLLISIONS]]) == 0:
        #     # add avoid all collision if there is no other avoid or allow all
        #     collision_goals.insert(0, CollisionEntry(type=CollisionEntry.AVOID_ALL_COLLISIONS,
        #                                              min_dist=min_dist))
        #
        # controllable_links = self.robot.get_controlled_links()
        #
        # collision_goals = self.verify_collision_entries(collision_goals)
        #
        # for collision_entry in collision_goals:  # type: CollisionEntry
        #     if collision_entry.type in [CollisionEntry.ALLOW_ALL_COLLISIONS,
        #                                 CollisionEntry.AVOID_ALL_COLLISIONS]:
        #         if collision_entry.robot_links != []:
        #             rospy.logwarn(u'type==AVOID_ALL_COLLISION but robot_links is set, did you mean AVOID_COLLISION?')
        #             collision_entry.robot_links = []
        #         if collision_entry.body_b != u'':
        #             rospy.logwarn(u'type==AVOID_ALL_COLLISION but body_b is set, it will be ignored.')
        #             collision_entry.body_b = u''
        #         if collision_entry.link_bs != []:
        #             rospy.logwarn(u'type==AVOID_ALL_COLLISION but link_bs is set, it will be ignored.')
        #             collision_entry.link_bs = []
        #
        #         if collision_entry.type == CollisionEntry.ALLOW_ALL_COLLISIONS:
        #             min_allowed_distance = {}
        #             continue
        #         else:
        #             min_allowed_distance = collision_matrix
        #
        #     # check if msg got properly filled
        #     if collision_entry.body_b == u'' and collision_entry.link_bs != []:
        #         raise PhysicsWorldException(u'body_b is empty but link_b is not')
        #
        #     # if robot link is empty, use all robot links
        #     if collision_entry.robot_links == []:
        #         robot_links = set(robot_links)
        #     else:
        #         for robot_link in collision_entry.robot_links:
        #             if robot_link not in self.robot.get_link_names():
        #                 raise UnknownBodyException(u'robot_link \'{}\' unknown'.format(robot_link))
        #         robot_links = set(collision_entry.robot_links)
        #
        #     # remove all non controllable links
        #     robot_links.intersection_update(controllable_links)
        #
        #     # if body_b is empty, use all objects
        #     if collision_entry.body_b == u'':
        #         bodies_b = self.get_object_names()
        #         # if collision_entry.type == CollisionEntry.AVOID_COLLISION:
        #         bodies_b.append(robot_name)
        #     elif self.has_object(collision_entry.body_b) or \
        #             collision_entry.body_b == robot_name:
        #         bodies_b = [collision_entry.body_b]
        #     else:
        #         raise UnknownBodyException(u'body_b \'{}\' unknown'.format(collision_entry.body_b))
        #
        #     link_b_was_set = len(collision_entry.link_bs) > 0
        #
        #     for body_b in bodies_b:
        #         # if link_b is empty, use all links from body_b
        #         link_bs = collision_entry.link_bs
        #         if body_b != robot_name:
        #             if link_bs == []:
        #                 # link_bs = self.get_object(body_b).get_link_names_with_collision()
        #                 link_bs = [u'']
        #             elif link_bs != []:
        #                 for link_b in link_bs:
        #                     # TODO use sets and intersection to safe time
        #                     if link_b not in self.get_object(body_b).get_link_names():
        #                         raise UnknownBodyException(u'link_b \'{}\' unknown'.format(link_b))
        #
        #         for robot_link in robot_links:
        #             if not link_b_was_set and body_b == robot_name:
        #                 link_bs = self.robot.get_possible_collisions(robot_link)
        #             for link_b in link_bs:
        #                 keys = [(robot_link, body_b, link_b)]
        #                 if body_b == robot_name:
        #                     if link_b not in self.robot.get_possible_collisions(robot_link):
        #                         continue
        #                     keys.append((link_b, body_b, robot_link))
        #
        #                 for key in keys:
        #                     if collision_entry.type == CollisionEntry.ALLOW_COLLISION:
        #                         if key[2] == u'':
        #                             for key2 in list(min_allowed_distance.keys()):
        #                                 if key[0] == key2[0] and key[1] == key2[1]:
        #                                     del min_allowed_distance[key2]
        #                         elif key in min_allowed_distance:
        #                             del min_allowed_distance[key]
        #
        #                     elif collision_entry.type == CollisionEntry.AVOID_COLLISION or \
        #                             collision_entry.type == CollisionEntry.AVOID_ALL_COLLISIONS:
        #                         min_allowed_distance[key] = collision_entry.min_dist
        #
        # return min_allowed_distance

    def verify_collision_entries(self, collision_goals, min_dist):
        for ce in collision_goals:  # type: CollisionEntry
            if ce.type in [CollisionEntry.ALLOW_ALL_COLLISIONS,
                           CollisionEntry.AVOID_ALL_COLLISIONS]:
                rospy.logwarn(u'ALLOW_ALL_COLLISIONS and AVOID_ALL_COLLISIONS deprecated, use AVOID_COLLISIONS and'
                              u'ALLOW_COLLISIONS instead with ALL constant instead.')

        for ce in collision_goals:  # type: CollisionEntry
            if CollisionEntry.ALL in ce.robot_links and len(ce.robot_links) != 1:
                raise PhysicsWorldException(u'ALL used in robot_links, but it\'s not the only entry')
            if CollisionEntry.ALL in ce.link_bs and len(ce.link_bs) != 1:
                raise PhysicsWorldException(u'ALL used in link_bs, but it\'s not the only entry')
            if ce.body_b == CollisionEntry.ALL and not self.all_link_bs(ce):
                raise PhysicsWorldException(u'if body_b == ALL, link_bs has to be ALL as well')

        for ce in collision_goals:
            if not ce.robot_links:
                ce.robot_links = [CollisionEntry.ALL]
            if not ce.link_bs:
                ce.link_bs = [CollisionEntry.ALL]

        for ce in collision_goals:
            ce.robot_links = set(ce.robot_links)
            ce.link_bs = set(ce.link_bs)

        if not collision_goals:
            ce = CollisionEntry()
            ce.type = CollisionEntry.AVOID_COLLISION
            ce.robot_links = [CollisionEntry.ALL]
            ce.body_b = CollisionEntry.ALL
            ce.link_bs = [CollisionEntry.ALL]
            ce.min_dist = min_dist
            collision_goals.append(ce)

        for i, ce in enumerate(reversed(collision_goals)):
            if self.is_avoid_all_collision(ce) or self.is_allow_all_collision(ce):
                break
        else:
            i = 0
        collision_goals = collision_goals[len(collision_goals) - i - 1:]

        # split body bs
        collision_goals = self.split_body_b(collision_goals)

        # split robot links
        collision_goals = self.robot_related_stuff(collision_goals)

        # split link_bs
        collision_goals = self.split_link_bs(collision_goals)

        # TODO if body b is robot filter with collision matrix

        return collision_goals

    def split_link_bs(self, collision_goals):
        # FIXME remove the side effects of these three methods
        i = 0
        while i < len(collision_goals):
            collision_entry = collision_goals[i]
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
                    ce.link_bs = {link_b}
                    new_ces.append(ce)
                for new_ce in new_ces:
                    collision_goals.insert(i, new_ce)
                i += len(new_ces)
                continue
            i += 1
        return collision_goals

    def robot_related_stuff(self, collision_goals):
        i = 0
        controlled_robot_links =self.robot.get_controlled_links()
        while i < len(collision_goals):
            collision_entry = collision_goals[i]
            if self.all_robot_links(collision_entry):
                collision_goals.remove(collision_entry)

                new_ces = []
                for robot_link in controlled_robot_links:
                    ce = CollisionEntry()
                    ce.type = collision_entry.type
                    ce.robot_links = {robot_link}
                    ce.body_b = collision_entry.body_b
                    ce.min_dist = collision_entry.min_dist
                    ce.link_bs = collision_entry.link_bs
                    new_ces.append(ce)

                for new_ce in new_ces:
                    collision_goals.insert(i, new_ce)
                i += len(new_ces)
                continue
            i += 1
        return collision_goals

    # def split_robot_links(self, collision_entry):
    #     ces = []
    #     for robot_link in self.robot.get_controlled_links():
    #         ce = CollisionEntry()
    #         ce.type = collision_entry.type
    #         ce.robot_links = {robot_link}
    #         ce.body_b = collision_entry.body_b
    #         ce.min_dist = collision_entry.min_dist
    #         ce.link_bs = collision_entry.link_bs
    #         ces.append(ce)
    #     return ces

    def split_body_b(self, collision_goals):
        i = 0
        while i < len(collision_goals):
            collision_entry = collision_goals[i]
            if self.all_body_bs(collision_entry):
                collision_goals.remove(collision_entry)
                new_ces = []
                for body_b in self.get_object_names() + [self.robot.get_name()]:
                    ce = CollisionEntry()
                    ce.type = collision_entry.type
                    ce.robot_links = collision_entry.robot_links
                    ce.min_dist = collision_entry.min_dist
                    ce.body_b = body_b
                    ce.link_bs = collision_entry.link_bs
                    new_ces.append(ce)
                for new_ce in new_ces:
                    collision_goals.insert(i, new_ce)
                i += len(new_ces)
                continue
            i += 1
        return collision_goals

    def all_robot_links(self, collision_entry):
        return CollisionEntry.ALL in collision_entry.robot_links and len(collision_entry.robot_links) == 1

    def all_link_bs(self, collision_entry):
        return CollisionEntry.ALL in collision_entry.link_bs and len(collision_entry.link_bs) == 1

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

    def is_allow_all_collision(self, collision_entry):
        """
        :type collision_entry: CollisionEntry
        :return: bool
        """
        return self.is_allow_collision(collision_entry) \
               and self.all_robot_links(collision_entry) \
               and self.all_body_bs(collision_entry) \
               and self.all_link_bs(collision_entry)

    def collisions_to_closest_point(self, collisions, min_allowed_distance):
        """
        :param collisions: (robot_link, body_b, link_b) -> ContactInfo
        :type collisions: dict
        :param min_allowed_distance: (robot_link, body_b, link_b) -> min allowed distance
        :type min_allowed_distance: dict
        :return: robot_link -> ClosestPointInfo of closest thing
        :rtype: dict
        """
        closest_point = keydefaultdict(lambda k: ClosestPointInfo((10, 0, 0),
                                                                  (0, 0, 0),
                                                                  1e9,
                                                                  0.0,
                                                                  k,
                                                                  '',
                                                                  (1, 0, 0), k))
        T_base___map = self.robot.T_base___map
        for key, contact_info in collisions.items():  # type: ((str, str, str), ContactInfo)
            if contact_info is None:
                continue
            link1 = key[0]
            a_in_robot_root = T_base___map * PyKDL.Vector(*contact_info.position_on_a)
            b_in_robot_root = T_base___map * PyKDL.Vector(*contact_info.position_on_b)
            n_in_robot_root = T_base___map.M * PyKDL.Vector(*contact_info.contact_normal_on_b)
            try:
                cpi = ClosestPointInfo(a_in_robot_root, b_in_robot_root, contact_info.contact_distance,
                                       min_allowed_distance[key], key[0], u'{} - {}'.format(key[1], key[2]),
                                       n_in_robot_root, key)
            except KeyError:
                continue
            if link1 in closest_point:
                closest_point[link1] = min(closest_point[link1], cpi, key=lambda x: x.contact_distance)
            else:
                closest_point[link1] = cpi
        return closest_point
