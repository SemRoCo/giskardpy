import numbers
import traceback
from copy import deepcopy, copy

import urdf_parser_py.urdf as up
from geometry_msgs.msg import PoseStamped
from giskard_msgs.msg import CollisionEntry
from tf.transformations import euler_matrix

from giskardpy import casadi_wrapper as w
from giskardpy import identifier
from giskardpy.data_types import JointStates, KeyDefaultDict
from giskardpy.exceptions import RobotExistsException, DuplicateNameException, PhysicsWorldException, \
    UnknownBodyException, UnsupportedOptionException
from giskardpy.god_map import GodMap
from giskardpy.model.joints import FixedJoint, PrismaticJoint, RevoluteJoint, ContinuousJoint, MovableJoint, MimicJoint
from giskardpy.model.robot import Robot
from giskardpy.model.urdf_object import hacky_urdf_parser_fix
from giskardpy.model.world_object import WorldObject
from giskardpy.qp.free_variable import FreeVariable
from giskardpy.utils import logging
from giskardpy.utils.tfwrapper import msg_to_kdl, kdl_to_pose, homo_matrix_to_pose
from giskardpy.utils.utils import suppress_stderr, memoize

class LinkGeometry(object):
    def __init__(self, link_T_geometry):
        self.link_T_geometry = link_T_geometry

class MeshGeometry(LinkGeometry):
    def __init__(self, link_T_geometry, file_name, scale=None):
        super(MeshGeometry, self).__init__(link_T_geometry)
        self.file_name = file_name
        if scale is None:
            self.scale = [1,1,1]
        else:
            self.scale = scale

class BoxGeometry(LinkGeometry):
    def __init__(self, link_T_geometry):
        super(BoxGeometry, self).__init__(link_T_geometry)


class Link(object):
    def __init__(self, name):
        self.name = name
        self.visuals = None
        self.collisions = None
        self.parent_joint_name = None
        self.child_names = []

    @classmethod
    def from_urdf(cls, urdf_link):
        link = cls(urdf_link.name)
        for urdf_collision in urdf_link.collisions:
            urdf_geometry = urdf_collision.geometry
            if isinstance(urdf_geometry, up.Mesh):
                link_T_geometry = euler_matrix(*urdf_collision.origin.rpy)
                link_T_geometry[0,3] = urdf_collision.origin.xyz[0]
                link_T_geometry[1,3] = urdf_collision.origin.xyz[1]
                link_T_geometry[2,3] = urdf_collision.origin.xyz[2]
                # link_T_geometry[]
                geometry = MeshGeometry(link_T_geometry, urdf_collision.geometry)
                # geometry

        return link

    def has_visuals(self):
        return self.visuals is not None

    def has_collisions(self):
        return self.collisions is not None

    def __repr__(self):
        return self.name


class WorldTree(object):
    def __init__(self, god_map=None):
        # public
        self.state = JointStates()
        self.root_link_name = 'root'
        self.links = {self.root_link_name: Link('root')}
        self.joints = {}
        self.groups = {}

        # private
        self.god_map = god_map  # type: GodMap
        self._free_variables = {}

    def add_group(self, name, root_link_name):
        if root_link_name not in self.links:
            raise KeyError('World doesn\'t have link \'{}\''.format(root_link_name))
        self.groups[name] = SubWorldTree(name, root_link_name, self)

    @property
    def root_link(self):
        return self.links[self.root_link_name]

    @property
    def link_names(self):
        return list(self.links.keys())

    @property
    def link_names_with_visuals(self):
        return [link.name for link in self.links.values() if link.has_visual()]


    @property
    def joint_names(self):
        return list(self.joints.keys())

    def load_urdf(self, urdf, parent_link_name=None):
        # create group?
        with suppress_stderr():
            parsed_urdf = up.URDF.from_xml_string(hacky_urdf_parser_fix(urdf))  # type: up.Robot

        if parent_link_name is None:
            parent_link = self.root_link
        else:
            parent_link = self.links[parent_link_name]
        child_link = Link(parsed_urdf.get_root())
        connecting_joint = FixedJoint(parsed_urdf.name, parent_link, child_link, None, None)
        self.link_joint_to_links(connecting_joint)

        def helper(urdf, parent_link_name):
            if parent_link_name not in urdf.child_map:
                return
            for child_joint_name, child_link_name in urdf.child_map[parent_link_name]:
                self.add_urdf_link(urdf.link_map[child_link_name])
                self.add_urdf_joint(urdf.joint_map[child_joint_name])
                helper(urdf, child_link_name)

        helper(parsed_urdf, child_link.name)

    @property
    def movable_joints(self):
        return [j.name for j in self.joints.values() if isinstance(j, MovableJoint)]

    def add_urdf_link(self, urdf_link):
        link = Link.from_urdf(urdf_link)
        self.links[link.name] = link

    def add_urdf_joint(self, urdf_joint):
        parent_link = self.links[urdf_joint.parent]
        child_link = self.links[urdf_joint.child]
        if urdf_joint.origin is not None:
            translation_offset = urdf_joint.origin.xyz
            rotation_offset = urdf_joint.origin.rpy
        else:
            translation_offset = None
            rotation_offset = None
        if urdf_joint.type == 'fixed':
            joint = FixedJoint(urdf_joint.name, parent_link, child_link, translation_offset, rotation_offset)
        else:
            lower_limits = {}
            upper_limits = {}
            if not urdf_joint.type == 'continuous':
                try:
                    lower_limits[0] = max(urdf_joint.safety_controller.soft_lower_limit, urdf_joint.limit.lower)
                    upper_limits[0] = min(urdf_joint.safety_controller.soft_upper_limit, urdf_joint.limit.upper)
                except AttributeError:
                    try:
                        lower_limits[0] = urdf_joint.limit.lower
                        upper_limits[0] = urdf_joint.limit.upper
                    except AttributeError:
                        lower_limits[0] = None
                        upper_limits[0] = None
            else:
                lower_limits[0] = None
                upper_limits[0] = None
            try:
                lower_limits[1] = -urdf_joint.limit.velocity
                upper_limits[1] = urdf_joint.limit.velocity
            except AttributeError:
                lower_limits[1] = None
                upper_limits[1] = None
            lower_limits[2] = -1e3
            upper_limits[2] = 1e3
            lower_limits[3] = -30
            upper_limits[3] = 30

            # TODO get rosparam data
            free_variable = FreeVariable(symbols={
                0: self.god_map.to_symbol(identifier.joint_states + [urdf_joint.name, 'position']),
                1: self.god_map.to_symbol(identifier.joint_states + [urdf_joint.name, 'velocity']),
                2: self.god_map.to_symbol(identifier.joint_states + [urdf_joint.name, 'acceleration']),
                3: self.god_map.to_symbol(identifier.joint_states + [urdf_joint.name, 'jerk']),
            },
                lower_limits=lower_limits,
                upper_limits=upper_limits,
                quadratic_weights={1: 0.01, 2: 0, 3: 0})

            if urdf_joint.type == 'revolute':
                joint = RevoluteJoint(urdf_joint.name, parent_link, child_link, translation_offset, rotation_offset,
                                      free_variable,
                                      urdf_joint.axis)
            elif urdf_joint.type == 'prismatic':
                joint = PrismaticJoint(urdf_joint.name, parent_link, child_link, translation_offset, rotation_offset,
                                       free_variable, urdf_joint.axis)
            elif urdf_joint.type == 'continuous':
                joint = ContinuousJoint(urdf_joint.name, parent_link, child_link, translation_offset, rotation_offset,
                                        free_variable, urdf_joint.axis)
            else:
                raise NotImplementedError('Joint of type {} is not supported'.format(urdf_joint.type))
        self.link_joint_to_links(joint)
        # TODO this has to move somewhere else
        self.init_fast_fks()

    def soft_reset(self):
        pass

    @property
    def joint_constraints(self):
        return {j.name: j.free_variable for j in self.joints.values() if isinstance(j, MovableJoint)}

    def link_joint_to_links(self, joint):
        self.joints[joint.name] = joint
        joint.child.parent_joint_name = joint.name
        self.links[joint.child.name] = joint.child
        parent_link = self.links[joint.parent_link_name]
        assert joint.name not in parent_link.child_names
        parent_link.child_names.append(joint.name)

    def move_branch(self, old_parent_joint, new_joint):
        # replace joint
        pass

    def delete_branch(self, parent_joint):
        pass

    def compute_chain(self, root_link_name, tip_link_name, joints=True, links=True, fixed=True):
        chain = []
        if links:
            chain.append(tip_link_name)
        link = self.links[tip_link_name]
        while link.name != root_link_name:
            if link.parent_joint_name not in self.joints:
                raise ValueError('{} and {} are not connected'.format(root_link_name, tip_link_name))
            parent_joint = self.joints[link.parent_joint_name]
            parent_link = parent_joint.parent
            if joints:
                if fixed or not isinstance(parent_joint, FixedJoint):
                    chain.append(parent_joint.name)
            if links:
                chain.append(parent_link.name)
            link = parent_link
        chain.reverse()
        return chain

    def compute_split_chain(self, root, tip, joints=True, links=True, fixed=True):
        if root == tip:
            return [], [], []
        root_chain = self.compute_chain(self.root_link_name, root, False, True, True)
        tip_chain = self.compute_chain(self.root_link_name, tip, False, True, True)
        for i in range(min(len(root_chain), len(tip_chain))):
            if root_chain[i] != tip_chain[i]:
                break
        else:
            i += 1
        connection = tip_chain[i - 1]
        root_chain = self.compute_chain(connection, root, joints, links, fixed)
        if links:
            root_chain = root_chain[1:]
        root_chain.reverse()
        tip_chain = self.compute_chain(connection, tip, joints, links, fixed)
        if links:
            tip_chain = tip_chain[1:]
        return root_chain, [connection] if links else [], tip_chain

    def compose_fk_expression(self, root_link, tip_link):
        fk = w.eye(4)
        root_chain, _, tip_chain = self.compute_split_chain(root_link, tip_link, links=False)
        for joint_name in root_chain:
            fk = w.dot(fk, w.inverse_frame(self.joints[joint_name].parent_T_child))
        for joint_name in tip_chain:
            fk = w.dot(fk, self.joints[joint_name].parent_T_child)
        # FIXME there is some reference fuckup going on, but i don't know where; deepcopy is just a quick fix
        return deepcopy(fk)

    def init_fast_fks(self):
        def f(key):
            root, tip = key
            fk = self.compose_fk_expression(root, tip)
            m = w.speed_up(fk, w.free_symbols(fk))
            return m

        self._fks = KeyDefaultDict(f)

    def compute_fk_pose(self, root, tip):
        try:
            homo_m = self.compute_fk_np(root, tip)
            p = PoseStamped()
            p.header.frame_id = root
            p.pose = homo_matrix_to_pose(homo_m)
        except Exception as e:
            print(e)
            traceback.print_exc()
            pass
        return p

    @memoize
    def compute_fk_np(self, root, tip):
        return self._fks[root, tip].call2(self.god_map.get_values(self._fks[root, tip].str_params))

    def set_joint_limits(self, linear_limits, angular_limits, order):
        for joint in self.joints.values():
            if self.is_joint_fixed(joint.name):
                continue
            if self.is_joint_rotational(joint.name):
                new_limits = angular_limits
            else:
                new_limits = linear_limits

            old_upper_limits = joint.free_variable.upper_limits[order]
            if old_upper_limits is None:
                joint.free_variable.upper_limits[order] = new_limits[joint.name]
            else:
                joint.free_variable.upper_limits[order] = w.min(old_upper_limits,
                                                                new_limits[joint.name])

            old_lower_limits = joint.free_variable.lower_limits[order]
            if old_lower_limits is None:
                joint.free_variable.lower_limits[order] = new_limits[joint.name]
            else:
                joint.free_variable.lower_limits[order] = w.max(old_lower_limits,
                                                                -new_limits[joint.name])

    def joint_limit_expr(self, joint_name, order):
        upper_limit = self.joints[joint_name].free_variable.get_upper_limit(order)
        lower_limit = self.joints[joint_name].free_variable.get_lower_limit(order)
        return upper_limit, lower_limit

    def compute_joint_limits(self, joint_name, order):
        lower_limit, upper_limit = self.joint_limit_expr(joint_name, order)
        if not isinstance(lower_limit, numbers.Number) and lower_limit is not None:
            f = w.speed_up(lower_limit, w.free_symbols(lower_limit))
            lower_limit = f.call2(self.god_map.get_values(f.str_params))[0][0]
        if not isinstance(upper_limit, numbers.Number) and upper_limit is not None:
            f = w.speed_up(upper_limit, w.free_symbols(upper_limit))
            upper_limit = f.call2(self.god_map.get_values(f.str_params))[0][0]
        return upper_limit, lower_limit

    def get_joint_position_limits(self, joint_name):
        return self.compute_joint_limits(joint_name, 0)

    def get_joint_velocity_limits(self, joint_name):
        return self.compute_joint_limits(joint_name, 1)

    def get_all_joint_position_limits(self):
        return {j: self.get_joint_position_limits(j) for j in self.movable_joints}

    def is_joint_prismatic(self, joint_name):
        return isinstance(self.joints[joint_name], PrismaticJoint)

    def is_joint_fixed(self, joint_name):
        return not isinstance(self.joints[joint_name], MovableJoint)

    def is_joint_revolute(self, joint_name):
        return isinstance(self.joints[joint_name], RevoluteJoint)

    def is_joint_continuous(self, joint_name):
        return isinstance(self.joints[joint_name], ContinuousJoint)

    def is_joint_mimic(self, joint_name):
        return isinstance(self.joints[joint_name], MimicJoint)

    def is_joint_rotational(self, joint_name):
        return self.is_joint_revolute(joint_name) or self.is_joint_continuous(joint_name)

    def has_joint(self, joint_name):
        return joint_name in self.joints

class SubWorldTree(WorldTree):
    def __init__(self, name, root_link_name, world):
        """
        :type name: str
        :type root_link_name: str
        :type world: WorldTree
        """
        self.name = name
        self.root_link_name = root_link_name
        self.world = world

    @property
    def state(self):
        return self.world.state

    @property
    def god_map(self):
        return self.world.god_map

    @property
    def root_link(self):
        return self.world.links[self.root_link_name]

    @property
    def joints(self):
        def helper(root_link):
            """
            :type root_link: Link
            :rtype: dict
            """
            joints = {j: self.world.joints[j] for j in root_link.child_names}
            for j in root_link.child_names: # type: FixedJoint
                joints.update(helper(self.world.joints[j].child))
            return joints
        return helper(self.root_link)

    @property
    def links(self):
        def helper(root_link):
            """
            :type root_link: Link
            :rtype: list
            """
            links = {root_link.name: root_link}
            for j in root_link.child_names: # type: FixedJoint
                links.update(helper(self.world.joints[j].child))
            return links
        return helper(self.root_link)

    def add_group(self, name, root_link_name):
        raise NotImplementedError()

    def link_joint_to_links(self, joint):
        raise NotImplementedError()

    def add_urdf_joint(self, urdf_joint):
        raise NotImplementedError()

    def delete_branch(self, parent_joint):
        raise NotImplementedError()


class World(object):
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

    def check_collisions(self, cut_off_distances, collision_list_size=20):
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
        :type name: Union[str, unicode]
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
        :type name: Union[str, unicode]
        :rtype: bool
        """
        return name in self.get_objects()

    def set_object_joint_state(self, name, joint_state):
        """
        :type name: Union[str, unicode]
        :param joint_state: joint name -> SingleJointState
        :type joint_state: dict
        """
        self.get_object(name).joint_state = joint_state

    def remove_object(self, name):
        if self.has_object(name):
            self._objects[name].suicide()
            logging.loginfo(u'<-- removed object {} from world'.format(name))
            del (self._objects[name])
        else:
            raise UnknownBodyException(u'can\'t remove object \'{}\', because it doesn\' exist'.format(name))

    def remove_all_objects(self):
        for object_name in self._objects.keys():
            # I'm not using remove object, because has object ignores hidden objects in pybullet world
            self._objects[object_name].suicide()
            logging.loginfo(u'<-- removed object {} from world'.format(object_name))
        self._objects = {}

    # Robot ------------------------------------------------------------------------------------------------------------

    @profile
    def add_robot(self, robot, base_pose, controlled_joints, ignored_pairs, added_pairs):
        """
        :type robot: giskardpy.world_object.WorldObject
        :type controlled_joints: list
        :type base_pose: Pose
        """
        if not isinstance(robot, WorldObject):
            raise TypeError(u'only WorldObject can be added to world')
        if self.has_robot():
            raise RobotExistsException(u'A robot is already loaded')
        if self.has_object(robot.get_name()):
            raise DuplicateNameException(
                u'can\'t add robot; object with name "{}" already exists'.format(robot.get_name()))
        if base_pose is None:
            base_pose = robot.base_pose
        self._robot = Robot.from_urdf_object(urdf_object=robot,
                                             base_pose=base_pose,
                                             controlled_joints=controlled_joints,
                                             path_to_data_folder=self._path_to_data_folder,
                                             ignored_pairs=ignored_pairs,
                                             added_pairs=added_pairs)
        logging.loginfo(u'--> added {} to world'.format(robot.get_name()))

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
        logging.loginfo(u'--> attached object {} on link {}'.format(name, link))

    def detach(self, joint_name, from_obj=None):
        if joint_name not in self.robot.get_joint_names():
            raise UnknownBodyException(u'can\'t detach: {}'.format(joint_name))
        if from_obj is None or self.robot.get_name() == from_obj:
            # this only works because attached simple objects have joint names equal to their name
            p = self.robot.get_fk_pose(self.robot.get_root(), joint_name)
            p_map = kdl_to_pose(self.robot.root_T_map.Inverse() * msg_to_kdl(p))

            parent_link = self.robot.get_parent_link_of_joint(joint_name)
            cut_off_obj = self.robot.detach_sub_tree(joint_name)
            logging.loginfo(u'<-- detached {} from link {}'.format(joint_name, parent_link))
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
            if self.robot.link_order(link1, link2):
                collision_matrix2[link1, robot_name, link2] = min_dist[link1]
            else:
                collision_matrix2[link2, robot_name, link1] = min_dist[link1]
        return collision_matrix2

    def collision_goals_to_collision_matrix(self, collision_goals, min_dist):
        """
        :param collision_goals: list of CollisionEntry
        :type collision_goals: list
        :return: dict mapping (robot_link, body_b, link_b) -> min allowed distance
        :rtype: dict
        """
        collision_goals = self.verify_collision_entries(collision_goals)
        min_allowed_distance = {}
        for collision_entry in collision_goals:  # type: CollisionEntry
            if self.is_avoid_all_self_collision(collision_entry):
                min_allowed_distance.update(self.get_robot_collision_matrix(min_dist))
                continue
            assert len(collision_entry.robot_links) == 1
            assert len(collision_entry.link_bs) == 1
            key = (collision_entry.robot_links[0], collision_entry.body_b, collision_entry.link_bs[0])
            r_key = (collision_entry.link_bs[0], collision_entry.body_b, collision_entry.robot_links[0])
            if self.is_allow_collision(collision_entry):
                if self.all_link_bs(collision_entry):
                    for key2 in list(min_allowed_distance.keys()):
                        if key[0] == key2[0] and key[1] == key2[1]:
                            del min_allowed_distance[key2]
                elif key in min_allowed_distance:
                    del min_allowed_distance[key]
                elif r_key in min_allowed_distance:
                    del min_allowed_distance[r_key]

            elif self.is_avoid_collision(collision_entry):
                min_allowed_distance[key] = min_dist[key[0]]
            else:
                raise Exception('todo')
        return min_allowed_distance

    def verify_collision_entries(self, collision_goals):
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
                        raise UnknownBodyException(
                            u'link b \'{}\' of body \'{}\' unknown'.format(robot_link, collision_entry.body_b))
            elif not self.all_body_bs(collision_entry) and not self.all_link_bs(collision_entry):
                object_links = self.get_object(collision_entry.body_b).get_link_names()
                for link_b in collision_entry.link_bs:
                    if link_b not in object_links:
                        raise UnknownBodyException(
                            u'link b \'{}\' of body \'{}\' unknown'.format(link_b, collision_entry.body_b))

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

    def is_avoid_all_self_collision(self, collision_entry):
        """
        :type collision_entry: CollisionEntry
        :return: bool
        """
        return self.is_avoid_collision(collision_entry) \
               and self.all_robot_links(collision_entry) \
               and collision_entry.body_b == self.robot.get_name() \
               and self.all_link_bs(collision_entry)

    def is_allow_all_self_collision(self, collision_entry):
        """
        :type collision_entry: CollisionEntry
        :return: bool
        """
        return self.is_allow_collision(collision_entry) \
               and self.all_robot_links(collision_entry) \
               and collision_entry.body_b == self.robot.get_name() \
               and self.all_link_bs(collision_entry)

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
