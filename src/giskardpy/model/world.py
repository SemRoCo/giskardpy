from __future__ import annotations

import abc
import hashlib
from abc import ABC
from copy import deepcopy
from functools import cached_property
from itertools import combinations
from typing import Dict, Union, Tuple, Set, Optional, List, Callable, Sequence

import numpy as np
import urdf_parser_py.urdf as up
from geometry_msgs.msg import PoseStamped, Pose, PointStamped, Point, Vector3Stamped, Vector3, QuaternionStamped
from std_msgs.msg import ColorRGBA
from tf2_msgs.msg import TFMessage

import giskardpy.utils.math as mymath
from giskard_msgs.msg import WorldBody
from giskardpy import casadi_wrapper as w, identifier
from giskardpy.casadi_wrapper import CompiledFunction
from giskardpy.data_types import JointStates
from giskardpy.exceptions import DuplicateNameException, UnknownGroupException, UnknownLinkException, \
    PhysicsWorldException, GiskardException
from giskardpy.god_map import GodMap
from giskardpy.god_map_user import GodMapWorshipper
from giskardpy.model.joints import Joint, FixedJoint, PrismaticJoint, RevoluteJoint, OmniDrive, DiffDrive, \
    urdf_to_joint, VirtualFreeVariables, MovableJoint, Joint6DOF
from giskardpy.model.links import Link, MeshGeometry
from giskardpy.model.utils import hacky_urdf_parser_fix
from giskardpy.my_types import PrefixName, Derivatives, derivative_joint_map, derivative_map
from giskardpy.my_types import my_string
from giskardpy.qp.free_variable import FreeVariable
from giskardpy.qp.next_command import NextCommands
from giskardpy.utils import logging
from giskardpy.utils.tfwrapper import homo_matrix_to_pose, np_to_pose, msg_to_homogeneous_matrix, make_transform
from giskardpy.utils.utils import suppress_stderr, clear_cached_properties
from giskardpy.utils.decorators import memoize, copy_memoize, clear_memo


class TravelCompanion:
    def link_call(self, link_name: PrefixName) -> bool:
        """
        :return: return True to stop climbing up the branch
        """
        return False

    def joint_call(self, joint_name: PrefixName) -> bool:
        """
        :return: return True to stop climbing up the branch
        """
        return False


class WorldTreeInterface(ABC):
    joints: Dict[PrefixName, Union[Joint, OmniDrive]]
    links: Dict[PrefixName, Link]
    groups: Dict[str, WorldBranch]

    @property
    def joint_names(self):
        return list(self.joints.keys())

    @property
    def link_names(self):
        return list(self.links.keys())

    @property
    @abc.abstractmethod
    def controlled_joints(self) -> List[PrefixName]:
        ...

    @property
    def link_names_with_visuals(self) -> Set[PrefixName]:
        return set(link.name for link in self.links.values() if link.has_visuals())

    @cached_property
    def link_names_with_collisions(self) -> Set[PrefixName]:
        return set(link.name for link in self.links.values() if link.has_collisions())

    @profile
    def to_hash(self):
        s = ''
        for link_name in sorted(self.link_names_with_collisions):
            link = self.links[link_name]
            for collision in link.collisions:
                s += collision.to_hash()
        # s += str(sorted(self.controlled_joints))
        hash_object = hashlib.sha256()
        hash_object.update(s.encode('utf-8'))
        return hash_object.hexdigest()

    @cached_property
    def link_names_without_collisions(self) -> Set[PrefixName]:
        return self.link_names_as_set.difference(self.link_names_with_collisions)

    @cached_property
    def movable_joint_names(self) -> List[PrefixName]:
        return [j.name for j in self.joints.values() if isinstance(j, MovableJoint)]

    @cached_property
    def link_names_as_set(self) -> Set[PrefixName]:
        return set(self.links.keys())

    @property
    def joint_names_as_set(self) -> Set[PrefixName]:
        return set(self.joints.keys())

    @property
    def group_names(self) -> Set[str]:
        return set(self.groups.keys())

    def reset_cache(self):
        for group in self.groups.values():
            group.reset_cache()
        clear_cached_properties(self)


class WorldModelUpdateContextManager:
    first: bool = True

    def __init__(self, world: WorldTree):
        self.world = world

    def __enter__(self):
        if self.world.context_manager_active:
            self.first = False
        self.world.context_manager_active = True
        return self.world

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None and self.first:
            self.world.context_manager_active = False
            self.world.notify_model_change()


class ResetJointStateContextManager:
    def __init__(self, world: WorldTree):
        self.world = world

    def __enter__(self):
        self.joint_state_tmp = deepcopy(self.world.state)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.world.state = self.joint_state_tmp
            self.world.notify_state_change()


class WorldTree(WorldTreeInterface, GodMapWorshipper):
    joints: Dict[PrefixName, Union[Joint, OmniDrive]]
    links: Dict[PrefixName, Link]
    state: JointStates
    free_variables: Dict[PrefixName, FreeVariable]
    virtual_free_variables: Dict[PrefixName, FreeVariable]
    context_manager_active: bool = False
    _default_limits: Dict[Derivatives, float]
    _default_weights: Dict[Derivatives, float]
    _root_link_name: PrefixName = None

    def __init__(self):
        self.default_link_color = ColorRGBA(1, 1, 1, 0.75)
        self.god_map.set_data(identifier.world, self)
        self.connection_prefix = 'connection'
        self.fast_all_fks = None
        self._state_version = 0
        self._model_version = 0
        self._clear()

    @classmethod
    def empty_world(cls):
        self = WorldTree()
        self._default_weights = {
            Derivatives.velocity: 1,
            Derivatives.acceleration: 1,
            Derivatives.jerk: 1,
        }
        self._default_limits = {
            Derivatives.velocity: 1,
        }
        identifier.max_derivative = ['max_derivative']
        self.god_map.set_data(identifier.max_derivative, Derivatives.jerk)
        return self

    @property
    def root_link_name(self) -> PrefixName:
        return self._root_link_name

    @property
    def root_link(self) -> Link:
        if self._root_link_name is None:
            raise PhysicsWorldException('no root_link set')
        return self.links[self._root_link_name]

    @property
    def default_limits(self):
        if self._default_limits is None:
            raise AttributeError(f'Please set default limits.')
        return self._default_limits

    def update_default_limits(self, new_limits: Dict[Derivatives, float]):
        if not hasattr(self, '_default_limits'):
            self._default_limits = {}
        for derivative, limit in new_limits.items():
            self._default_limits[derivative] = limit
        assert len(self._default_limits) == max(self._default_limits)

    def update_default_weights(self, new_weights: Dict[Derivatives, float]):
        if not hasattr(self, '_default_weights'):
            self._default_weights = {}
        for derivative, weight in new_weights.items():
            self._default_weights[derivative] = weight
        assert len(self._default_weights) == max(self._default_weights)

    def get_joint_name(self, joint_name: my_string, group_name: Optional[str] = None) -> PrefixName:
        logging.logwarn(f'Deprecated warning: use \'search_for_joint_name\' instead of \'get_joint_name\'.')
        return self.search_for_joint_name(joint_name, group_name)

    def search_for_joint_name(self, joint_name: str, group_name: Optional[str] = None) -> PrefixName:
        """
        Will search the worlds joint for one that matches joint_name. group_name is only needed if there are multiple
        joints with the same name.
        :param joint_name: a joint name e.g. torso_lift_joint
        :param group_name: only needed if there are name conflicts, e.g., when there are 2 torso_lift_joints
        :return: how the joint is called inside the world tree e.g. pr2/torso_lift_joint
        """
        if group_name == '':
            group_name = None
        if group_name is not None:
            return self.groups[group_name].search_for_joint_name(joint_name)

        matches = []
        for internal_joint_name in self.joint_names:
            if joint_name == internal_joint_name or joint_name == internal_joint_name.short_name:
                matches.append(internal_joint_name)
        if len(matches) > 1:
            raise ValueError(f'Multiple matches for \'{joint_name}\' found: \'{matches}\'.')
        if len(matches) == 0:
            raise ValueError(f'No matches for \'{joint_name}\' found: \'{matches}\'.')
        return matches[0]

    def rename_link(self, old_name: PrefixName, new_name: PrefixName):
        if old_name not in self.link_names:
            self._raise_if_link_does_not_exist(old_name)
        if new_name in self.link_names:
            self._raise_if_link_exists(new_name)
        link = self.links[old_name]
        link.name = new_name
        for joint in self.joints.values():
            if joint.parent_link_name == old_name:
                joint.parent_link_name = new_name
            elif joint.child_link_name == old_name:
                joint.child_link_name = new_name

    def get_joint(self, joint_name: my_string, group_name: Optional[str] = None) -> Joint:
        """
        Like get_joint_name, but returns the actual joint.
        """
        return self.joints[self.search_for_joint_name(joint_name, group_name)]

    def get_link_name(self, link_name: str, group_name: Optional[str] = None) -> PrefixName:
        logging.logwarn(f'Deprecated warning: use \'search_for_link_name\' instead of \'get_link_name\'.')
        return self.search_for_link_name(link_name, group_name)

    def search_for_link_name(self, link_name: str, group_name: Optional[str] = None) -> PrefixName:
        """
        Like get_joint_name but for links.
        """
        if group_name == '':
            group_name = None
        if link_name == '':
            link_name = None
        if link_name is None:
            if group_name is None:
                return self.root_link_name
            else:
                return self.groups[group_name].root_link_name
        if group_name is not None:
            return self.groups[group_name].search_for_link_name(link_name)

        matches = []
        for internal_link_name in self.link_names:
            if link_name == internal_link_name or link_name == internal_link_name.short_name:
                matches.append(internal_link_name)
        if len(matches) > 1:
            raise UnknownLinkException(f'Multiple links matches for \'{link_name}\' found: \'{matches}\'.')
        if len(matches) == 0:
            raise UnknownLinkException(f'Link \'{link_name}\' not found.')
        return matches[0]

    def get_link(self, link_name: str, group_name: Optional[str] = None) -> Link:
        """
        Like get_joint but for links.
        """
        return self.links[self.search_for_link_name(link_name, group_name)]

    @property
    def version(self) -> Tuple[int, int]:
        """
        Can be used to determine if the world has changed
        :return: tuple of model and state version. The first number indicates if the world itself has changed.
                    and the second number says if the it's state has changed.
        """
        return self._model_version, self._state_version

    @property
    def model_version(self) -> int:
        """
        :return: number that increased every time the world model has changed
        """
        return self._model_version

    @property
    def state_version(self) -> int:
        """
        :return: number that increases every time the world state has changed.
        """
        return self._state_version

    @profile
    def notify_state_change(self):
        """
        If you have changed the state of the world, call this function to trigger necessary events and increase
        the state version.
        """
        clear_memo(self.compute_fk_pose)
        clear_memo(self.compute_fk_pose_with_collision_offset)
        self._recompute_fks()
        self._state_version += 1

    def reset_cache(self):
        super().reset_cache()
        clear_memo(self.get_directly_controlled_child_links_with_collisions)
        clear_memo(self.get_directly_controlled_child_links_with_collisions)
        clear_memo(self.compute_chain_reduced_to_controlled_joints)
        clear_memo(self.get_movable_parent_joint)
        clear_memo(self.get_controlled_parent_joint_of_link)
        clear_memo(self.get_controlled_parent_joint_of_joint)
        clear_memo(self.compute_split_chain)
        clear_memo(self.are_linked)
        clear_memo(self.compose_fk_expression)
        clear_memo(self.compute_chain)
        clear_memo(self.is_link_controlled)
        for free_variable in self.free_variables.values():
            free_variable.reset_cache()

    @profile
    def notify_model_change(self):
        """
        Call this function if you have changed the model of the world to trigger necessary events and increase
        the model version number.
        """
        if not self.context_manager_active:
            with self.god_map:
                self.fix_tree_structure()
                self.reset_cache()
                self.init_all_fks()
                self.notify_state_change()
                self._model_version += 1

    def travel_branch(self, link_name: PrefixName, companion: TravelCompanion):
        """
        Do a depth first search on a branch starting at link_name.
        Use companion to do whatever you want. It link_call and joint_call are called on every link/joint it sees.
        The traversion is stopped once they return False.
        :param link_name: starting point of the search
        :param companion: payload. Implement your own Travelcompanion for your purpose.
        """
        link = self.links[link_name]
        if not companion.link_call(link_name):
            for child_joint_name in link.child_joint_names:
                if companion.joint_call(child_joint_name):
                    continue
                child_link_name = self.joints[child_joint_name].child_link_name
                self.travel_branch(child_link_name, companion)

    def search_branch(self,
                      link_name: PrefixName,
                      stop_at_joint_when: Optional[Callable[[PrefixName], bool]] = None,
                      stop_at_link_when: Optional[Callable[[PrefixName], bool]] = None,
                      collect_joint_when: Optional[Callable[[PrefixName], bool]] = None,
                      collect_link_when: Optional[Callable[[PrefixName], bool]] = None) -> \
            Tuple[List[PrefixName], List[PrefixName]]:
        """
        Do a depth first search starting at link_name.
        :param link_name: starting point of the search
        :param stop_at_joint_when: If None, 'lambda joint_name: False' is used.
        :param stop_at_link_when: If None, 'lambda link_name: False' is used.
        :param collect_joint_when: If None, 'lambda joint_name: False' is used.
        :param collect_link_when: If None, 'lambda link_name: False' is used.
        :return: Collected link names and joint names. Might include 'link_name'
        """

        class CollectorCompanion(TravelCompanion):
            def __init__(self, collect_joint_when=None, collect_link_when=None,
                         stop_at_joint_when=None, stop_at_link_when=None):
                self.collected_link_names = []
                self.collected_joint_names = []

                if stop_at_joint_when is None:
                    def stop_at_joint_when(_):
                        return False
                if stop_at_link_when is None:
                    def stop_at_link_when(_):
                        return False
                self.stop_at_joint_when = stop_at_joint_when
                self.stop_at_link_when = stop_at_link_when

                if collect_joint_when is None:
                    def collect_joint_when(_):
                        return False
                if collect_link_when is None:
                    def collect_link_when(_):
                        return False
                self.collect_joint_when = collect_joint_when
                self.collect_link_when = collect_link_when

            def link_call(self, link_name):
                if self.collect_link_when(link_name):
                    self.collected_link_names.append(link_name)
                return self.stop_at_link_when(link_name)

            def joint_call(self, joint_name):
                if self.collect_joint_when(joint_name):
                    self.collected_joint_names.append(joint_name)
                return self.stop_at_joint_when(joint_name)

        collector_companion = CollectorCompanion(collect_joint_when=collect_joint_when,
                                                 collect_link_when=collect_link_when,
                                                 stop_at_joint_when=stop_at_joint_when,
                                                 stop_at_link_when=stop_at_link_when)
        self.travel_branch(link_name, companion=collector_companion)
        return collector_companion.collected_link_names, collector_companion.collected_joint_names

    @memoize
    def get_directly_controlled_child_links_with_collisions(self,
                                                            joint_name: PrefixName,
                                                            joints_to_exclude: Optional[Tuple] = None) \
            -> List[PrefixName]:
        """
        Collect all links with collisions that are connected to joint_name by fixed joints or joints in
        joints_to_exclude.
        :param joint_name:
        :param joints_to_exclude: List of joints to be considered as fixed for this search
        :return:
        """
        if joints_to_exclude is None:
            joints_to_exclude = set()
        else:
            joints_to_exclude = set(joints_to_exclude)

        def stopper(joint_name):
            return joint_name not in joints_to_exclude and self.is_joint_controlled(joint_name)

        child_link_name = self.joints[joint_name].child_link_name
        links, joints = self.search_branch(link_name=child_link_name,
                                           stop_at_joint_when=stopper,
                                           collect_link_when=self.has_link_collisions)
        return links

    def get_siblings_with_collisions(self, joint_name: PrefixName) -> List[PrefixName]:
        """
        Goes up the tree until the first controlled joint and then down again until another controlled joint or
        the joint_name is reached again. Collects all links with collision along the way.
        :param joint_name:
        :return: list of link names
        """
        try:
            parent_joint = self.search_for_parent_joint(joint_name, stop_when=self.is_joint_controlled)
        except KeyError as e:
            return []

        def stop_at_joint_when(other_joint_name):
            return joint_name == other_joint_name or self.is_joint_controlled(other_joint_name)

        child_link_name = self.joints[parent_joint].child_link_name
        link_names, joint_names = self.search_branch(link_name=child_link_name,
                                                     stop_at_joint_when=stop_at_joint_when,
                                                     collect_link_when=self.has_link_collisions)
        return link_names

    def register_group(self, name: str, root_link_name: PrefixName, actuated: bool = False):
        """
        Create a new subgroup at root_link_name.
        :param name:
        :param root_link_name:
        :param actuated: Whether this group is controlled by giskard. Important for self collision avoidance
        """
        if root_link_name not in self.links:
            raise KeyError(f'World doesn\'t have link \'{root_link_name}\'')
        if name in self.groups:
            raise DuplicateNameException(f'Group with name {name} already exists')
        new_group = WorldBranch(name, root_link_name, self, actuated=actuated)
        # if the group is a subtree of a subtree, register it for the subtree as well
        # for group in self.groups.values():
        #     if root_link_name in group.links:
        #         group.groups[name] = new_group
        self.groups[name] = new_group

    def deregister_group(self, name: str):
        del self.groups[name]

    @property
    def robots(self) -> List[WorldBranch]:
        """
        :return: All actuated groups
        """
        return [self.groups[group_name] for group_name in list(self.group_names)
                if self.groups[group_name].actuated]

    @property
    def robot_names(self) -> List[str]:
        """
        :return: The names of all actuated groups
        """
        return [r.name for r in self.robots]

    @property
    def robot_name(self) -> str:
        return self.robot_names[0]

    @property
    def minimal_group_names(self) -> Set[str]:
        """
        :return: All groups that are not part of another group.
        """
        group_names = self.group_names
        for group in self.groups.values():
            for group_name in group.group_names:
                if group_name in group_names:
                    group_names.remove(group_name)
        return group_names

    def add_free_variable(self,
                          name: PrefixName,
                          lower_limits: derivative_map,
                          upper_limits: derivative_map) -> FreeVariable:
        free_variable = FreeVariable(name=name,
                                     lower_limits=lower_limits,
                                     upper_limits=upper_limits,
                                     quadratic_weights=self._default_weights)
        if free_variable.has_position_limits():
            lower_limit = free_variable.get_lower_limit(derivative=Derivatives.position,
                                                        evaluated=True)
            upper_limit = free_variable.get_upper_limit(derivative=Derivatives.position,
                                                        evaluated=True)
            initial_value = min(max(0, lower_limit), upper_limit)
            self.state[name].position = initial_value
        self.free_variables[name] = free_variable
        return free_variable

    def add_virtual_free_variable(self, name: PrefixName) -> FreeVariable:
        free_variable = FreeVariable(name=name,
                                     lower_limits={},
                                     upper_limits={},
                                     quadratic_weights=self._default_weights)
        self.virtual_free_variables[name] = free_variable
        return free_variable

    def update_state(self, next_commands: NextCommands, dt: float):
        max_derivative = self.god_map.get_data(identifier.max_derivative)
        for free_variable_name, command in next_commands.free_variable_data.items():
            self.state[free_variable_name][:max_derivative] += command * dt
            self.state[free_variable_name][max_derivative] = command[-1]
        for joint in self.joints.values():
            if isinstance(joint, VirtualFreeVariables):
                joint.update_state(dt)

    def add_urdf(self,
                 urdf: str,
                 group_name: Optional[str] = None,
                 parent_link_name: Optional[PrefixName] = None,
                 pose: Optional[w.TransMatrix] = None,
                 actuated: bool = False):
        """
        Add a urdf to the world at parent_link_name and create a SubWorldTree named group_name for it.
        :param urdf: urdf as str, not a file path
        :param group_name: name of the group that will be created. default is name in urdf.
        :param parent_link_name: where the urdf will be attached
        :param actuated: if the urdf is controlled by Giskard, important for self collision avoidance
        """
        with suppress_stderr():
            parsed_urdf: up.Robot = up.URDF.from_xml_string(hacky_urdf_parser_fix(urdf))
        if group_name in self.groups:
            raise DuplicateNameException(
                f'Failed to add group \'{group_name}\' because one with such a name already exists')

        urdf_root_link_name = parsed_urdf.link_map[parsed_urdf.get_root()].name
        urdf_root_link_name_prefixed = PrefixName(urdf_root_link_name, group_name)

        if parent_link_name is not None:
            urdf_link = parsed_urdf.link_map[urdf_root_link_name]
            urdf_root_link = Link.from_urdf(urdf_link=urdf_link,
                                            prefix=group_name,
                                            color=self.default_link_color)
            self._add_link(urdf_root_link)
            pose_msg = Pose()
            position = pose.to_position().evaluate()
            orientation = pose.to_rotation().to_quaternion().evaluate()
            pose_msg.position.x = position[0][0]
            pose_msg.position.y = position[1][0]
            pose_msg.position.z = position[2][0]
            pose_msg.orientation.x = orientation[0][0]
            pose_msg.orientation.y = orientation[1][0]
            pose_msg.orientation.z = orientation[2][0]
            pose_msg.orientation.w = orientation[3][0]
            joint = Joint6DOF(name=PrefixName(group_name, self.connection_prefix),
                              parent_link_name=parent_link_name,
                              child_link_name=urdf_root_link.name)
            joint.update_transform(pose_msg)
            self._add_joint(joint)
        else:
            urdf_root_link = Link(urdf_root_link_name_prefixed)
            self._add_link(urdf_root_link)
            # urdf_root_link = self.links[urdf_root_link_name]

        def helper(urdf, parent_link):
            short_name = parent_link.name.short_name
            if short_name not in urdf.child_map:
                return  # stop because link has no child links
            for child_joint_name, child_link_name in urdf.child_map[short_name]:
                urdf_link = urdf.link_map[child_link_name]
                child_link = Link.from_urdf(urdf_link=urdf_link,
                                            prefix=group_name,
                                            color=self.default_link_color)
                self._add_link(child_link)

                urdf_joint: up.Joint = urdf.joint_map[child_joint_name]

                joint = urdf_to_joint(urdf_joint, group_name)
                if not isinstance(joint, FixedJoint):
                    for derivative, limit in self.default_limits.items():
                        joint.free_variable.set_lower_limit(derivative, -limit)
                        joint.free_variable.set_upper_limit(derivative, limit)

                self._link_joint_to_links(joint)
                helper(urdf, child_link)

        number_of_links_before = len(self.links)
        helper(parsed_urdf, urdf_root_link)
        if number_of_links_before + len(parsed_urdf.links) - 1 != len(self.links):
            # -1 because root link already exists
            raise GiskardException(f'Failed to add urdf \'{group_name}\' to world')

        # if add_drive_joint_to_group:
        #     root_link = self.get_parent_link_of_link(urdf_root_link_name)
        #     self.register_group(group_name, root_link, actuated=actuated)
        # else:
        self.register_group(group_name, urdf_root_link_name_prefixed, actuated=actuated)
        self.notify_model_change()

    def _add_fixed_joint(self, parent_link: Link, child_link: Link, joint_name: str = None,
                         transform: Optional[w.TransMatrix] = None):
        self._raise_if_link_does_not_exist(parent_link.name)
        self._raise_if_link_does_not_exist(child_link.name)
        if joint_name is None:
            joint_name = PrefixName(f'{parent_link.name}_{child_link.name}_fixed_joint', None)
        connecting_joint = FixedJoint(name=joint_name,
                                      parent_link_name=parent_link.name,
                                      child_link_name=child_link.name,
                                      parent_T_child=transform)
        self._link_joint_to_links(connecting_joint)

    def _replace_joint(self, new_joint: Joint):
        child_link = new_joint.child_link_name
        parent_link = new_joint.parent_link_name

        for i, link_or_joint in enumerate(self.compute_chain(parent_link, child_link, add_joints=True, add_links=True,
                                                             add_fixed_joints=True, add_non_controlled_joints=True)[
                                          1:-1]):
            if i == 0:
                parent_link = self.joints[link_or_joint].parent_link_name
                self.links[parent_link].child_joint_names.remove(link_or_joint)
            if i % 2 == 0:
                del self.joints[link_or_joint]
            else:
                del self.links[link_or_joint]

        self._link_joint_to_links(new_joint)

    def get_parent_link_of_link(self, link_name: PrefixName) -> PrefixName:
        return self.joints[self.links[link_name].parent_joint_name].parent_link_name

    def get_group_of_joint(self, joint_name: PrefixName):
        ret = set()
        for group_name, subtree in self.groups.items():
            if joint_name in subtree.joints:
                ret.add(subtree)
        if len(ret) == 0:
            raise KeyError(f'No groups found with joint name {joint_name}.')
        if len(ret) > 1:
            raise KeyError(f'Multiple groups {[x.name for x in ret]} found with joint name {joint_name}.')
        else:
            return ret.pop()

    def _get_robots_containing_link_short_name(self, link_name: Union[PrefixName, str]) -> Set[str]:
        groups = set()
        for group_name, subtree in self.groups.items():
            if subtree.actuated:
                try:
                    subtree.get_link_short_name_match(link_name)
                except KeyError:
                    continue
                groups.add(group_name)
        return groups

    def _get_parents_of_group_name(self, group_name: str) -> Set[str]:
        ancestry = list()
        traversed = False
        parent = self.get_parent_group_name(group_name)
        while not traversed:
            if parent in ancestry:
                traversed = True
            else:
                ancestry.append(parent)
            parent = self.get_parent_group_name(ancestry[-1])
        return set(ancestry)

    def create_group_ancestry(self) -> Dict[str, str]:
        ancestry = {}
        for group_name in self.group_names:
            possible_parents = []
            for possible_parent_name, possible_parent_group in self.groups.items():
                if group_name in possible_parent_group.group_names:
                    possible_parents.append(possible_parent_name)
            ancestry[group_name] = possible_parents

        while not np.all([len(direct_children) <= 1 for direct_children in ancestry.values()]):
            for group_name, parents in list(ancestry.items()):
                if len(parents) > 1:
                    for possible_direct_parent in parents:
                        for grand_parent in ancestry[possible_direct_parent]:
                            ancestry[group_name].remove(grand_parent)
        for group_name, ancestors in list(ancestry.items()):
            if len(ancestors) == 0:
                ancestry[group_name] = None
            else:
                ancestry[group_name] = ancestors[0]
        return ancestry

    def _get_group_name_containing_link(self, link_name: Union[PrefixName, str]) -> str:
        groups = self.get_group_names_containing_link(link_name)
        ret = self._get_group_from_groups(groups)
        if ret is None:
            raise UnknownGroupException(f'Did not find any group containing the link {link_name}.')
        return ret

    def _get_group_from_groups(self, groups: Set[str]) -> str:
        if len(groups) == 1:
            return list(groups)[0]
        else:
            groups_l = list(groups)
            group = None
            for i in range(len(groups_l)):
                g_a = groups_l[i]
                if i + 1 == len(groups):
                    break
                else:
                    g_b = groups_l[i + 1]
                if g_a != g_b:
                    g_ancestry = self._get_parents_of_group_name(g_a)
                    group_ancestry = self._get_parents_of_group_name(g_b)
                    relatives = list(g_ancestry & group_ancestry)
                    if relatives and relatives[0] in groups:
                        group = relatives[0]
            return group

    def get_group_names_containing_link(self, link_name: Union[PrefixName, str]) -> Set[str]:
        groups = set()
        for group_name, subtree in self.groups.items():
            if link_name in subtree.link_names_as_set:
                groups.add(group_name)
        return groups

    def _get_robots_containing_link(self, link_name: Union[PrefixName, str]) -> Set[str]:
        groups = set()
        for group_name, subtree in self.groups.items():
            if subtree.actuated:
                if link_name in subtree.link_names_as_set:
                    groups.add(group_name)
        return groups

    @memoize
    def compute_chain_reduced_to_controlled_joints(self,
                                                   link_a: PrefixName,
                                                   link_b: PrefixName,
                                                   joints_to_exclude: Optional[tuple] = None) \
            -> Tuple[PrefixName, PrefixName]:
        """
        1. Compute kinematic chain of links between link_a and link_b.
        2. Remove all entries from link_a downward until one is connected with a non fixed joint.
        2. Remove all entries from link_b upward until one is connected with a non fixed joint.
        :param link_a:
        :param link_b:
        :param joints_to_exclude: non fixed joints to be assumed as fixed.
        :return: start and end link of the reduced chain
        """
        if joints_to_exclude is None:
            joints_to_exclude = set()
        joint_list = [j for j in self.controlled_joints if j not in joints_to_exclude]
        chain1, connection, chain2 = self.compute_split_chain(link_b, link_a, add_joints=True, add_links=True,
                                                              add_fixed_joints=True,
                                                              add_non_controlled_joints=True)
        chain = chain1 + connection + chain2
        for i, thing in enumerate(chain):
            if i % 2 == 1 and thing in joint_list:
                new_link_b = chain[i - 1]
                break
        else:
            raise KeyError(f'no controlled joint in chain between {link_a} and {link_b}')
        for i, thing in enumerate(reversed(chain)):
            if i % 2 == 1 and thing in joint_list:
                new_link_a = chain[len(chain) - i]
                break
        else:
            raise KeyError(f'no controlled joint in chain between {link_a} and {link_b}')
        return new_link_a, new_link_b

    @memoize
    def get_movable_parent_joint(self, link_name: PrefixName) -> PrefixName:
        joint = self.links[link_name].parent_joint_name
        while not self.is_joint_movable(joint):
            joint = self.links[self.joints[joint].parent_link_name].parent_joint_name
        return joint

    def get_parent_group_name(self, group_name: str) -> str:
        for potential_parent_group in self.minimal_group_names:
            if group_name in self.groups[potential_parent_group].groups:
                return potential_parent_group
        return group_name

    @profile
    def add_world_body(self,
                       group_name: str,
                       msg: WorldBody,
                       pose: Pose,
                       parent_link_name: PrefixName):
        """
        Add a WorldBody to the world.
        :param group_name: Name of the SubWorldTree that will be created.
        :param msg:
        :param pose: pose of WorldBody relative to parent_link_name
        :param parent_link_name: where the WorldBody will be attached
        """
        if group_name in self.groups:
            raise DuplicateNameException(f'Group with name \'{group_name}\' already exists')
        self._raise_if_link_does_not_exist(parent_link_name)
        if isinstance(parent_link_name, str):
            parent_link_name = PrefixName(parent_link_name, None)
        if msg.type == msg.URDF_BODY:
            self.add_urdf(urdf=msg.urdf,
                          parent_link_name=parent_link_name,
                          group_name=group_name,
                          pose=w.TransMatrix(pose))
            self.notify_model_change()
        else:
            link = Link.from_world_body(link_name=PrefixName(group_name, group_name), msg=msg,
                                        color=self.default_link_color)
            self._add_link(link)
            joint = Joint6DOF(name=PrefixName(group_name, self.connection_prefix),
                              parent_link_name=parent_link_name,
                              child_link_name=link.name)
            joint.update_transform(pose)
            self._link_joint_to_links(joint)
            self.register_group(group_name, link.name)
            self.notify_model_change()

    def _clear(self):
        self.state = JointStates()
        self.links = {}
        self.joints = {}
        self.free_variables = {}
        self.virtual_free_variables = {}
        self.groups: Dict[my_string, WorldBranch] = {}
        self.reset_cache()

    def delete_all_but_robots(self):
        """
        Resets Giskard to the state from when it was started.
        """
        self._clear()
        with self.modify_world():
            self.god_map.get_data(identifier.giskard).world_config.setup()

    def _add_joint_and_create_child(self, joint: Joint):
        self._raise_if_joint_exists(joint.name)
        self._raise_if_link_exists(joint.child_link_name)
        child_link = Link(joint.child_link_name)
        self._add_link(child_link)
        self._link_joint_to_links(joint)

    def _link_joint_to_links(self, joint: Joint):
        self._raise_if_joint_exists(joint.name)
        self._raise_if_link_does_not_exist(joint.child_link_name)
        self._raise_if_link_does_not_exist(joint.parent_link_name)
        child_link = self.links[joint.child_link_name]
        parent_link = self.links[joint.parent_link_name]
        self.joints[joint.name] = joint
        child_link.parent_joint_name = joint.name
        assert joint.name not in parent_link.child_joint_names
        parent_link.child_joint_names.append(joint.name)

    def fix_tree_structure(self):
        for joint in self.joints.values():
            self._raise_if_link_does_not_exist(joint.parent_link_name)
            self._raise_if_link_does_not_exist(joint.child_link_name)
            parent_link = self.links[joint.parent_link_name]
            if joint.name not in parent_link.child_joint_names:
                parent_link.child_joint_names.append(joint.name)
            child_link = self.links[joint.child_link_name]
            if child_link.parent_joint_name is None:
                child_link.parent_joint_name = joint.name
            else:
                assert child_link.parent_joint_name == joint.name
        self.fix_root_link()
        for link in self.links.values():
            if link != self.root_link:
                self._raise_if_joint_does_not_exist(link.parent_joint_name)
            for child_joint_name in link.child_joint_names:
                self._raise_if_joint_does_not_exist(child_joint_name)

    def fix_root_link(self):
        orphans = []
        for link_name, link in self.links.items():
            if link.parent_joint_name is None:
                orphans.append(link_name)
        if len(orphans) > 1:
            raise PhysicsWorldException(f'Found multiple orphaned links: {orphans}.')
        self._root_link_name = orphans[0]

    def _raise_if_link_does_not_exist(self, link_name: my_string):
        if link_name not in self.links:
            raise PhysicsWorldException(f'Link \'{link_name}\' does not exist.')

    def _raise_if_link_exists(self, link_name: my_string):
        if link_name in self.links:
            raise DuplicateNameException(f'Link \'{link_name}\' does already exist.')

    def _raise_if_joint_does_not_exist(self, joint_name: my_string):
        if joint_name not in self.joints:
            raise PhysicsWorldException(f'Joint \'{joint_name}\' does not exist.')

    def _raise_if_joint_exists(self, joint_name: my_string):
        if joint_name in self.joints:
            raise DuplicateNameException(f'Joint \'{joint_name}\' does already exist.')

    @profile
    def move_branch(self, joint_name: PrefixName, new_parent_link_name: PrefixName):
        """
        Removed joint_name and creates a fixed joint between the old parent and child link.
        :param joint_name:
        :param new_parent_link_name:
        """
        # TODO: change parent link from TFJoints
        # if not self.is_joint_fixed(joint_name):
        #     raise NotImplementedError('Can only change fixed joints')
        joint = self.joints[joint_name]
        old_parent_link = self.links[joint.parent_link_name]
        new_parent_link = self.links[new_parent_link_name]

        if isinstance(joint, FixedJoint):
            fk = w.TransMatrix(self.compute_fk_np(new_parent_link_name, joint.child_link_name))
            joint.parent_link_name = new_parent_link_name
            joint.parent_T_child = fk
        elif isinstance(joint, Joint6DOF):
            pose = self.compute_fk_pose(new_parent_link_name, joint.child_link_name)
            joint.parent_link_name = new_parent_link_name
            joint.update_transform(pose.pose)
        else:
            raise NotImplementedError('Can only change fixed joints and TFJoints')

        old_parent_link.child_joint_names.remove(joint_name)
        new_parent_link.child_joint_names.append(joint_name)
        self.notify_model_change()

    @profile
    def update_joint_parent_T_child(self,
                                    joint_name: PrefixName,
                                    new_parent_T_child: w.TransMatrix,
                                    notify: bool = True):
        joint = self.joints[joint_name]
        if not isinstance(joint, FixedJoint):
            raise NotImplementedError('Can only change fixed joints')
        joint.parent_T_child = new_parent_T_child
        if notify:
            self.notify_model_change()

    def cleanup_unused_free_variable(self):
        used_variables = []
        for joint_name in self.movable_joint_names:
            joint = self.joints[joint_name]
            used_variables.extend(free_variable.name for free_variable in joint.free_variables)
        for free_variable_name in self.free_variables:
            if free_variable_name not in used_variables:
                del self.state[free_variable_name]

    def move_group(self, group_name: str, new_parent_link_name: PrefixName):
        """
        Removed the joint connecting group_name to the world and attach it to new_parent_link_name.
        The pose relative to the self.root_link does not change.
        """
        group = self.groups[group_name]
        joint_name = self.links[group.root_link_name].parent_joint_name
        if self.joints[joint_name].parent_link_name == new_parent_link_name:
            raise DuplicateNameException(f'\'{group_name}\' is already attached to \'{new_parent_link_name}\'')
        self.move_branch(joint_name, new_parent_link_name)

    def delete_group(self, group_name: str):
        """
        Delete the group and all links and joints contained in it.
        """
        if group_name not in self.groups:
            raise UnknownGroupException(f'Can\'t delete unknown group: \'{group_name}\'')
        self.delete_branch(self.groups[group_name].root_link_name)

    def delete_branch(self, link_name: PrefixName):
        """
        Delete every link and joint from link_name downward, including the link.
        """
        self.delete_branch_at_joint(self.links[link_name].parent_joint_name)

    @profile
    def delete_branch_at_joint(self, joint_name: PrefixName):
        """
        Delete every link and joint from joint_name downward, including the joint.
        """
        joint = self.joints.pop(joint_name)  # type: Joint
        self.links[joint.parent_link_name].child_joint_names.remove(joint_name)

        def helper(link_name):
            link = self.links.pop(link_name)
            for group_name in list(self.groups.keys()):
                if self.groups[group_name].root_link_name == link_name:
                    del self.groups[group_name]
                    logging.loginfo(f'Deleted group \'{group_name}\', because it\'s root link got removed.')
            for child_joint_name in link.child_joint_names:
                child_joint = self.joints.pop(child_joint_name)  # type: Joint
                helper(child_joint.child_link_name)

        helper(joint.child_link_name)
        self.notify_model_change()

    def link_order(self, link_a: PrefixName, link_b: PrefixName) -> bool:
        """
        this function is used when deciding for which order to calculate the collisions
        true if link_a < link_b
        """
        return link_a < link_b

    def sort_links(self, link_a: PrefixName, link_b: PrefixName) -> Tuple[PrefixName, PrefixName]:
        """
        A deterministic way of sorting links, not necessary dependent on the alphabetical order.
        """
        if self.link_order(link_a, link_b):
            return link_a, link_b
        return link_b, link_a

    @property
    def controlled_joints(self) -> List[PrefixName]:
        try:
            return self.god_map.unsafe_get_data(identifier.controlled_joints)
        except KeyError:
            return []

    def register_controlled_joints(self, controlled_joints: List[PrefixName]):
        """
        Flag these joints as controlled.
        """
        old_controlled_joints = set(self.controlled_joints)
        new_controlled_joints = set(controlled_joints)
        double_joints = old_controlled_joints.intersection(new_controlled_joints)
        if double_joints:
            raise DuplicateNameException(f'Controlled joints \'{double_joints}\' are already registered!')
        unknown_joints = new_controlled_joints.difference(self.joint_names_as_set)
        if unknown_joints:
            raise UnknownGroupException(f'Trying to register unknown joints: \'{unknown_joints}\'')
        old_controlled_joints.update(new_controlled_joints)
        self.god_map.set_data(identifier.controlled_joints, list(sorted(old_controlled_joints)))

    @memoize
    def get_controlled_parent_joint_of_link(self, link_name: PrefixName) -> PrefixName:
        joint = self.links[link_name].parent_joint_name
        if self.is_joint_controlled(joint):
            return joint
        return self.get_controlled_parent_joint_of_joint(joint)

    @memoize
    def get_controlled_parent_joint_of_joint(self, joint_name: PrefixName) -> PrefixName:
        return self.search_for_parent_joint(joint_name, self.is_joint_controlled)

    def search_for_parent_joint(self,
                                joint_name: PrefixName,
                                stop_when: Optional[Callable[[PrefixName], bool]] = None) -> PrefixName:
        try:
            joint = self.links[self.joints[joint_name].parent_link_name].parent_joint_name
            while stop_when is not None and not stop_when(joint):
                joint = self.search_for_parent_joint(joint)
        except KeyError as e:
            raise KeyError(f'\'{joint_name}\' has no fitting parent joint.')
        return joint

    @profile
    @memoize
    def compute_chain(self,
                      root_link_name: PrefixName,
                      tip_link_name: PrefixName,
                      add_joints: bool,
                      add_links: bool,
                      add_fixed_joints: bool,
                      add_non_controlled_joints: bool) -> List[PrefixName]:
        """
        Computes a chain between root_link_name and tip_link_name. Only works if root_link_name is above tip_link_name
        in the world tree.
        :param root_link_name:
        :param tip_link_name:
        :param add_joints:
        :param add_links:
        :param add_fixed_joints: only used if add_joints == True
        :param add_non_controlled_joints: only used if add_joints == True
        :return:
        """
        chain = []
        if add_links:
            chain.append(tip_link_name)
        link = self.links[tip_link_name]
        while link.name != root_link_name:
            if link.parent_joint_name not in self.joints:
                raise ValueError(f'{root_link_name} and {tip_link_name} are not connected')
            parent_joint = self.joints[link.parent_joint_name]
            parent_link = self.links[parent_joint.parent_link_name]
            if add_joints:
                if (add_fixed_joints or not isinstance(parent_joint, FixedJoint)) and \
                        (add_non_controlled_joints or parent_joint.name in self.controlled_joints):
                    chain.append(parent_joint.name)
            if add_links:
                chain.append(parent_link.name)
            link = parent_link
        chain.reverse()
        return chain

    @memoize
    def compute_split_chain(self,
                            root_link_name: PrefixName,
                            tip_link_name: PrefixName,
                            add_joints: bool,
                            add_links: bool,
                            add_fixed_joints: bool,
                            add_non_controlled_joints: bool) \
            -> Tuple[List[PrefixName], List[PrefixName], List[PrefixName]]:
        """
        Computes the chain between root_link_name and tip_link_name. Can handle chains that start and end anywhere
        in the tree.
        :param root_link_name:
        :param tip_link_name:
        :param add_joints:
        :param add_links:
        :param add_fixed_joints: only used if add_joints == True
        :param add_non_controlled_joints: only used if add_joints == True
        :return: tuple containing
                    1. chain from root_link_name to the connecting link
                    2. the connecting link, if add_lins is True
                    3. chain from connecting link to tip_link_name
        """
        if root_link_name == tip_link_name:
            return [], [], []
        root_chain = self.compute_chain(self.root_link_name, root_link_name, False, True, True, True)
        tip_chain = self.compute_chain(self.root_link_name, tip_link_name, False, True, True, True)
        for i in range(min(len(root_chain), len(tip_chain))):
            if root_chain[i] != tip_chain[i]:
                break
        else:
            i += 1
        connection = tip_chain[i - 1]
        root_chain = self.compute_chain(connection, root_link_name, add_joints, add_links, add_fixed_joints,
                                        add_non_controlled_joints)
        if add_links:
            root_chain = root_chain[1:]
        root_chain = root_chain[::-1]
        tip_chain = self.compute_chain(connection, tip_link_name, add_joints, add_links, add_fixed_joints,
                                       add_non_controlled_joints)
        if add_links:
            tip_chain = tip_chain[1:]
        return root_chain, [connection] if add_links else [], tip_chain

    def modify_world(self):
        return WorldModelUpdateContextManager(self)

    def reset_joint_state_context(self):
        return ResetJointStateContextManager(self)

    @copy_memoize
    @profile
    def compose_fk_expression(self, root_link: PrefixName, tip_link: PrefixName) -> w.TransMatrix:
        """
        Multiplies all transformation matrices in the chain between root_link and tip_link
        :param root_link:
        :param tip_link:
        :return: 4x4 homogenous transformation matrix
        """
        fk = w.TransMatrix()
        root_chain, _, tip_chain = self.compute_split_chain(root_link, tip_link, add_joints=True, add_links=False,
                                                            add_fixed_joints=True, add_non_controlled_joints=True)
        for joint_name in root_chain:
            a = self.joints[joint_name].parent_T_child
            ai = a.inverse()
            fk = fk.dot(ai)
        for joint_name in tip_chain:
            a = self.joints[joint_name].parent_T_child
            fk = fk.dot(a)
        return fk

    @memoize
    def compute_fk_pose(self, root: my_string, tip: my_string) -> PoseStamped:
        root = self.search_for_link_name(root)
        tip = self.search_for_link_name(tip)
        homo_m = self.compute_fk_np(root, tip)
        p = PoseStamped()
        p.header.frame_id = str(root)
        p.pose = homo_matrix_to_pose(homo_m)
        return p

    def compute_fk_point(self, root: my_string, tip: my_string) -> PointStamped:
        root_T_tip = self.compute_fk_pose(root, tip)
        root_P_tip = PointStamped()
        root_P_tip.header = root_T_tip.header
        root_P_tip.point = root_T_tip.pose.position
        return root_P_tip

    @memoize
    def compute_fk_pose_with_collision_offset(self, root: PrefixName, tip: PrefixName,
                                              collision_id: int) -> PoseStamped:
        root_T_tip = self.compute_fk_np(root, tip)
        tip_link = self.links[tip]
        root_T_tip = root_T_tip.dot(tip_link.collisions[collision_id].link_T_geometry.evaluate())
        p = PoseStamped()
        p.header.frame_id = str(root)
        p.pose = homo_matrix_to_pose(root_T_tip)
        return p

    @profile
    def as_tf_msg(self, include_prefix: bool) -> TFMessage:
        """
        Create a tfmessage for the whole world tree.
        """
        tf_msg = TFMessage()
        for joint_name, joint in self.joints.items():
            p_T_c = self.compute_fk_pose(root=joint.parent_link_name, tip=joint.child_link_name)
            if include_prefix:
                parent_link_name = joint.parent_link_name
                child_link_name = joint.child_link_name
            else:
                parent_link_name = joint.parent_link_name.short_name
                child_link_name = joint.child_link_name.short_name
            p_T_c = make_transform(parent_frame=parent_link_name,
                                   child_frame=child_link_name,
                                   pose=p_T_c.pose,
                                   normalize_quaternion=False)
            tf_msg.transforms.append(p_T_c)
        return tf_msg

    @profile
    def compute_all_collision_fks(self):
        params = self.god_map.unsafe_get_values(self._fk_computer.fast_collision_fks.str_params)
        return self._fk_computer.fast_collision_fks.fast_call(params)

    @profile
    def init_all_fks(self):
        class ExpressionCompanion(TravelCompanion):
            idx_start: Dict[PrefixName, int]
            fast_collision_fks: CompiledFunction
            fast_all_fks: CompiledFunction
            str_params: List[str]

            def __init__(self, world: WorldTree):
                self.world = world
                self.god_map = GodMap()
                self.fks = {self.world.root_link_name: w.TransMatrix()}

            @profile
            def joint_call(self, joint_name: my_string) -> bool:
                joint = self.world.joints[joint_name]
                map_T_parent = self.fks[joint.parent_link_name]
                self.fks[joint.child_link_name] = map_T_parent.dot(joint.parent_T_child)
                return False

            @profile
            def compile_fks(self):
                all_fks = w.vstack([self.fks[link_name] for link_name in self.world.link_names_as_set])
                collision_fks = []
                # collision_ids = []
                for link_name in sorted(self.world.link_names_with_collisions):
                    if link_name == self.world.root_link_name:
                        continue
                    # link = self.world.links[link_name]
                    # for collision_id, geometry in enumerate(link.collisions):
                    #     link_name_with_id = link.name_with_collision_id(collision_id)
                    collision_fks.append(self.fks[link_name])
                    # collision_ids.append(link_name_with_id)
                collision_fks = w.vstack(collision_fks)
                # self.collision_link_order = list(collision_ids)
                params = set()
                params.update(all_fks.free_symbols())
                params.update(collision_fks.free_symbols())
                params = list(params)
                self.str_params = [str(v) for v in params]
                self.fast_all_fks = all_fks.compile(parameters=params)
                self.fast_collision_fks = collision_fks.compile(parameters=params)
                self.idx_start = {link_name: i * 4 for i, link_name in enumerate(self.world.link_names_as_set)}

            @profile
            def recompute(self):
                self.compute_fk_np.memo.clear()
                self.fks = self.fast_all_fks.fast_call(self.god_map.unsafe_get_values(self.fast_all_fks.str_params))

            @memoize
            @profile
            def compute_fk_np(self, root, tip):
                if root == self.world.root_link_name:
                    map_T_root = np.eye(4)
                else:
                    map_T_root = self.fks[self.idx_start[root]:self.idx_start[root] + 4]
                if tip == self.world.root_link_name:
                    map_T_tip = np.eye(4)
                else:
                    map_T_tip = self.fks[self.idx_start[tip]:self.idx_start[tip] + 4]
                root_T_map = mymath.inverse_frame(map_T_root)
                root_T_tip = np.dot(root_T_map, map_T_tip)
                return root_T_tip

        self._fk_computer = ExpressionCompanion(self)
        self.travel_branch(self.root_link_name, self._fk_computer)
        self._fk_computer.compile_fks()

    @profile
    def _recompute_fks(self):
        self._fk_computer.recompute()

    @profile
    def compute_fk_np(self, root: PrefixName, tip: PrefixName) -> np.ndarray:
        return self._fk_computer.compute_fk_np(root, tip)

    @memoize
    @profile
    def are_linked(self, link_a: PrefixName, link_b: PrefixName,
                   do_not_ignore_non_controlled_joints: bool = False,
                   joints_to_be_assumed_fixed: Optional[Sequence[PrefixName]] = None):
        """
        Return True if all joints between link_a and link_b are fixed.
        """
        chain1, connection, chain2 = self.compute_split_chain(link_a, link_b, add_joints=True, add_links=False,
                                                              add_fixed_joints=False,
                                                              add_non_controlled_joints=do_not_ignore_non_controlled_joints)
        if joints_to_be_assumed_fixed is not None:
            chain1 = [x for x in chain1 if x not in joints_to_be_assumed_fixed]
            chain2 = [x for x in chain2 if x not in joints_to_be_assumed_fixed]
        return not chain1 and not connection and not chain2

    def _add_link(self, link: Link):
        self._raise_if_link_exists(link.name)
        self.links[link.name] = link

    def _add_joint(self, joint: Joint):
        self._raise_if_joint_exists(joint.name)
        self.joints[joint.name] = joint

    def joint_limit_expr(self, joint_name: PrefixName, order: Derivatives) \
            -> Tuple[Optional[w.symbol_expr_float], Optional[w.symbol_expr_float]]:
        return self.joints[joint_name].get_limit_expressions(order)

    def transform_msg(self, target_frame: PrefixName,
                      msg: Union[PointStamped, PoseStamped, Vector3Stamped, QuaternionStamped]) \
            -> Union[PointStamped, PoseStamped, Vector3Stamped, QuaternionStamped]:
        if isinstance(msg, PoseStamped):
            return self.transform_pose(target_frame, msg)
        elif isinstance(msg, PointStamped):
            return self.transform_point(target_frame, msg)
        elif isinstance(msg, Vector3Stamped):
            return self.transform_vector(target_frame, msg)
        elif isinstance(msg, QuaternionStamped):
            return self.transform_quaternion(target_frame, msg)
        else:
            raise NotImplementedError(f'World can\'t transform message of type \'{type(msg)}\'')

    def transform_pose(self, target_frame: PrefixName, pose: PoseStamped) -> PoseStamped:
        f_T_p = msg_to_homogeneous_matrix(pose.pose)
        t_T_f = self.compute_fk_np(target_frame, pose.header.frame_id)
        t_T_p = np.dot(t_T_f, f_T_p)
        result = PoseStamped()
        result.header.frame_id = target_frame
        result.pose = np_to_pose(t_T_p)
        return result

    def transform_quaternion(self, target_frame: PrefixName, quaternion: QuaternionStamped) -> QuaternionStamped:
        p = PoseStamped()
        p.header = quaternion.header
        p.pose.orientation = quaternion.quaternion
        new_pose = self.transform_pose(target_frame, p)
        new_quaternion = QuaternionStamped()
        new_quaternion.header = new_pose.header
        new_quaternion.quaternion = new_pose.pose.orientation
        return new_quaternion

    def transform_point(self, target_frame: PrefixName, point: PointStamped) -> PointStamped:
        f_P_p = msg_to_homogeneous_matrix(point)
        t_T_f = self.compute_fk_np(target_frame, point.header.frame_id)
        t_P_p = np.dot(t_T_f, f_P_p)
        result = PointStamped()
        result.header.frame_id = target_frame
        result.point = Point(*t_P_p[:3])
        return result

    def transform_vector(self, target_frame: PrefixName, vector: Vector3Stamped) -> Vector3Stamped:
        f_V_p = msg_to_homogeneous_matrix(vector)
        t_T_f = self.compute_fk_np(target_frame, vector.header.frame_id)
        t_V_p = np.dot(t_T_f, f_V_p)
        result = Vector3Stamped()
        result.header.frame_id = target_frame
        result.vector = Vector3(*t_V_p[:3])
        return result

    def compute_joint_limits(self, joint_name: PrefixName, order: Derivatives) \
            -> Tuple[Optional[w.symbol_expr_float], Optional[w.symbol_expr_float]]:
        try:
            lower_limit, upper_limit = self.joint_limit_expr(joint_name, order)
        except KeyError:
            # joint has no limits for this derivative
            return None, None
        if not isinstance(lower_limit, (int, float)) and lower_limit is not None:
            lower_limit = self.god_map.evaluate_expr(lower_limit)
        if not isinstance(upper_limit, (int, float)) and upper_limit is not None:
            upper_limit = self.god_map.evaluate_expr(upper_limit)
        return lower_limit, upper_limit

    def get_joint_position_limits(self, joint_name: my_string) -> Tuple[Optional[float], Optional[float]]:
        """
        :return: minimum position, maximum position as float
        """
        return self.compute_joint_limits(joint_name, Derivatives.position)

    def get_joint_velocity_limits(self, joint_name) -> Tuple[float, float]:
        return self.compute_joint_limits(joint_name, Derivatives.velocity)

    def dye_group(self, group_name: str, color: ColorRGBA):
        if group_name in self.groups:
            for _, link in self.groups[group_name].links.items():
                link.dye_collisions(color)
        else:
            raise UnknownGroupException(f'No group named {group_name}')

    def dye_world(self, color: ColorRGBA):
        for link in self.links.values():
            link.dye_collisions(color)

    def get_all_free_variable_velocity_limits(self) -> Dict[PrefixName, float]:
        limits = {}
        for free_variable_name, free_variable in self.free_variables.items():
            limits[free_variable_name] = free_variable.get_upper_limit(derivative=Derivatives.velocity,
                                                                       default=False,
                                                                       evaluated=True)
        return limits

    def is_joint_prismatic(self, joint_name: PrefixName) -> bool:
        return isinstance(self.joints[joint_name], PrismaticJoint)

    def is_joint_fixed(self, joint_name: PrefixName) -> bool:
        return isinstance(self.joints[joint_name], FixedJoint)

    def is_joint_movable(self, joint_name: PrefixName) -> bool:
        return not self.is_joint_fixed(joint_name)

    def is_joint_controlled(self, joint_name: PrefixName) -> bool:
        return joint_name in self.controlled_joints

    @memoize
    def is_link_controlled(self, link_name: PrefixName) -> bool:
        try:
            self.get_controlled_parent_joint_of_link(link_name)
            return True
        except KeyError as e:
            return False

    def is_joint_revolute(self, joint_name: PrefixName) -> bool:
        return isinstance(self.joints[joint_name], RevoluteJoint) and not self.is_joint_continuous(joint_name)

    def is_joint_continuous(self, joint_name: PrefixName) -> bool:
        joint = self.joints[joint_name]
        return isinstance(joint, RevoluteJoint) and not joint.free_variable.has_position_limits()

    def is_joint_rotational(self, joint_name: PrefixName) -> bool:
        return self.is_joint_revolute(joint_name) or self.is_joint_continuous(joint_name)

    def has_joint(self, joint_name: PrefixName) -> bool:
        return joint_name in self.joints

    def has_link_collisions(self, link_name: PrefixName) -> bool:
        return self.links[link_name].has_collisions()

    def has_link_visuals(self, link_name: PrefixName) -> bool:
        return self.links[link_name].has_visuals()

    def save_graph_pdf(self):
        import pydot
        def joint_type_to_color(joint):
            color = 'lightgrey'
            if isinstance(joint, PrismaticJoint):
                color = 'red'
            elif isinstance(joint, RevoluteJoint):
                color = 'yellow'
            elif isinstance(joint, (OmniDrive, DiffDrive)):
                color = 'orange'
            return color

        world_graph = pydot.Dot('world_tree', bgcolor='white', rank='source')
        group_clusters = {group_name: pydot.Cluster(group_name, label=group_name) for group_name in self.group_names}
        ancestry = self.create_group_ancestry()
        for group_name, ancestor in ancestry.items():
            if ancestor is None:
                world_graph.add_subgraph(group_clusters[group_name])
            else:
                group_clusters[ancestor].add_subgraph(group_clusters[group_name])

        for link_name, link in self.links.items():
            node_label = f'{str(link_name)}\nGeometry: {[type(x).__name__ for x in link.collisions]}'
            link_node = pydot.Node(str(link_name), label=node_label)
            world_graph.add_node(link_node)
            for group_name, group in self.groups.items():
                if link_name in group.link_names:
                    group_clusters[group_name].add_node(link_node)

        for joint_name, joint in self.joints.items():
            joint_node_name = f'{joint_name}\n{type(joint).__name__}'
            if self.is_joint_controlled(joint_name):
                peripheries = 2
            else:
                peripheries = 1
            joint_node = pydot.Node(str(joint_name), label=joint_node_name, shape='box', style='filled',
                                    fillcolor=joint_type_to_color(joint), peripheries=peripheries)
            world_graph.add_node(joint_node)
            for group_name, group in self.groups.items():
                if joint_name in group.joint_names:
                    group_clusters[group_name].add_node(joint_node)

            child_edge = pydot.Edge(str(joint_name), str(joint.child_link_name))
            world_graph.add_edge(child_edge)
            parent_edge = pydot.Edge(str(joint.parent_link_name), str(joint_name))
            world_graph.add_edge(parent_edge)
        file_name = f'{self.god_map.get_data(identifier.tmp_folder)}/world_tree.pdf'
        world_graph.write_pdf(file_name)


class WorldBranch(WorldTreeInterface):
    def __init__(self, name: str, root_link_name: PrefixName, world: WorldTree, actuated: bool = False):
        self.name = name
        self.root_link_name = root_link_name
        self.world = world
        self.actuated = actuated

    def get_siblings_with_collisions(self, joint_name: PrefixName) -> List[PrefixName]:
        siblings = self.world.get_siblings_with_collisions(joint_name)
        return [x for x in siblings if x in self.link_names_as_set]

    def get_link(self, link_name: str) -> Link:
        return self.world.get_link(link_name, self.name)

    def get_joint(self, joint_name: str) -> Joint:
        return self.world.get_joint(joint_name, self.name)

    def search_for_link_name(self, link_name: str) -> PrefixName:
        matches = []
        for internal_link_name in self.link_names:
            if internal_link_name.short_name == link_name or internal_link_name == link_name:
                matches.append(internal_link_name)
        if len(matches) > 1:
            raise ValueError(f'Multiple matches for \'{link_name}\' found: \'{matches}\'.')
        if len(matches) == 0:
            raise ValueError(f'No matches for \'{link_name}\' found: \'{matches}\'.')
        return matches[0]

    def search_for_joint_name(self, joint_name: str) -> PrefixName:
        matches = []
        for internal_joint_name in self.joint_names:
            if joint_name == internal_joint_name or joint_name == internal_joint_name.short_name:
                matches.append(internal_joint_name)
        if len(matches) > 1:
            raise ValueError(f'Multiple matches for \'{joint_name}\' found: \'{matches}\'.')
        if len(matches) == 0:
            raise ValueError(f'No matches for \'{joint_name}\' found: \'{matches}\'.')
        return matches[0]

    @cached_property
    def controlled_joints(self) -> List[PrefixName]:
        return [j for j in self.god_map.unsafe_get_data(identifier.controlled_joints) if j in self.joint_names_as_set]

    @property
    def parent_link_of_root(self) -> PrefixName:
        return self.world.get_parent_link_of_link(self.world.groups[self.name].root_link_name)

    @profile
    def possible_collision_combinations(self) -> Set[Tuple[PrefixName, PrefixName]]:
        links = self.link_names_with_collisions
        link_combinations = {self.world.sort_links(link_a, link_b) for link_a, link_b in combinations(links, 2)}
        for link_name in links:
            direct_children = set()
            for child_joint_name in self.links[link_name].child_joint_names:
                if self.world.is_joint_controlled(child_joint_name):
                    continue
                child_link_name = self.joints[child_joint_name].child_link_name
                links, joints = self.world.search_branch(link_name=child_link_name,
                                                         stop_at_joint_when=self.world.is_joint_controlled,
                                                         stop_at_link_when=None,
                                                         collect_joint_when=None,
                                                         collect_link_when=self.world.has_link_collisions)

                direct_children.update(links)
            direct_children.add(link_name)
            link_combinations.difference_update(
                self.world.sort_links(link_a, link_b) for link_a, link_b in combinations(direct_children, 2))
        return link_combinations

    def get_unmovable_links(self) -> List[PrefixName]:
        unmovable_links, _ = self.world.search_branch(link_name=self.root_link_name,
                                                      stop_at_joint_when=lambda
                                                          joint_name: joint_name in self.controlled_joints,
                                                      collect_link_when=self.world.has_link_collisions)
        return unmovable_links

    @property
    def base_pose(self) -> Pose:
        return self.world.compute_fk_pose(self.world.root_link_name, self.root_link_name).pose

    @property
    def state(self) -> JointStates:
        return JointStates({j: self.world.state[j] for j in self.joints if j in self.world.state})

    def get_link_short_name_match(self, link_name: my_string) -> PrefixName:
        matches = []
        for link_name2 in self.link_names_as_set:
            if link_name == link_name2 or link_name == link_name2.short_name:
                matches.append(link_name2)
        if len(matches) > 1:
            raise ValueError(f'Found multiple link matches \'{matches}\'.')
        if len(matches) == 0:
            raise UnknownLinkException(f'Found no link match for \'{link_name}\'.')
        return matches[0]

    def reset_cache(self):
        super().reset_cache()
        try:
            del self.joints
        except:
            pass
        try:
            del self.links
        except:
            pass
        try:
            del self.groups
        except:
            pass
        try:
            del self.controlled_joints
        except:
            pass

    @property
    def god_map(self) -> GodMap:
        return self.world.god_map

    @property
    def root_link(self) -> Link:
        return self.world.links[self.root_link_name]

    @cached_property
    def joints(self) -> Dict[PrefixName, Joint]:
        def helper(root_link: Link) -> Dict[PrefixName, Joint]:
            joints = {j: self.world.joints[j] for j in root_link.child_joint_names}
            for joint_name in root_link.child_joint_names:
                joint = self.world.joints[joint_name]
                child_link = self.world.links[joint.child_link_name]
                joints.update(helper(child_link))
            return joints

        return helper(self.root_link)

    @cached_property
    def groups(self) -> Dict[str, WorldBranch]:
        return {group_name: group for group_name, group in self.world.groups.items() if
                group.root_link_name in self.links and group.name != self.name}

    @cached_property
    def links(self) -> Dict[PrefixName, Link]:
        def helper(root_link: Link) -> Dict[my_string, Link]:
            links = {root_link.name: root_link}
            for j in root_link.child_joint_names:
                j = self.world.joints[j]
                child_link = self.world.links[j.child_link_name]
                links.update(helper(child_link))
            return links

        return helper(self.root_link)

    def compute_fk_pose(self, root: PrefixName, tip: PrefixName) -> PoseStamped:
        return self.world.compute_fk_pose(root, tip)

    def compute_fk_pose_with_collision_offset(self, root: PrefixName, tip: PrefixName, collision_id: int) \
            -> PoseStamped:
        return self.world.compute_fk_pose_with_collision_offset(root, tip, collision_id)

    def is_link_controlled(self, link_name: PrefixName) -> bool:
        return self.world.is_link_controlled(link_name)
