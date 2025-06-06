from __future__ import annotations

import abc
import hashlib
from abc import ABC
from collections import OrderedDict
from copy import deepcopy
from functools import cached_property, wraps
from itertools import combinations
from typing import Dict, Union, Tuple, Set, Optional, List, Callable, Sequence, Type, overload
from xml.etree.ElementTree import ParseError

import numpy as np
import urdf_parser_py.urdf as up

import giskardpy.utils.math as mymath
from giskardpy import casadi_wrapper as cas
from giskardpy.casadi_wrapper import CompiledFunction
from giskardpy.data_types.data_types import JointStates, ColorRGBA
from giskardpy.data_types.exceptions import DuplicateNameException, UnknownGroupException, UnknownLinkException, \
    WorldException, UnknownJointException, CorruptURDFException
from giskardpy.model.joints import Joint, FixedJoint, PrismaticJoint, RevoluteJoint, OmniDrive, DiffDrive, \
    VirtualFreeVariables, MovableJoint, Joint6DOF, OneDofJoint
from giskardpy.model.links import Link
from giskardpy.model.utils import hacky_urdf_parser_fix, robot_name_from_urdf_string
from giskardpy.data_types.data_types import PrefixName, Derivatives, derivative_map
from giskardpy.qp.free_variable import FreeVariable
from giskardpy.qp.next_command import NextCommands
from giskardpy.symbol_manager import symbol_manager
from giskardpy.middleware import get_middleware
from giskardpy.utils.utils import suppress_stderr, clear_cached_properties
from giskardpy.utils.decorators import memoize, copy_memoize, clear_memo
from line_profiler import profile


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
    def link_names(self) -> List[PrefixName]:
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
        if self.first:
            self.world.context_manager_active = False
            if exc_type is None:
                self.world._notify_model_change()


def modifies_world(func):
    @wraps(func)
    def wrapper(self: WorldTree, *args, **kwargs):
        with self.modify_world():
            result = func(self, *args, **kwargs)
            return result

    return wrapper


class ResetJointStateContextManager:
    def __init__(self, world: WorldTree):
        self.world = world

    def __enter__(self):
        self.joint_state_tmp = deepcopy(self.world.state)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.world.state = self.joint_state_tmp
            self.world.notify_state_change()


class WorldTree(WorldTreeInterface):
    joints: Dict[PrefixName, Union[Joint, OmniDrive]]
    links: Dict[PrefixName, Link]
    state: JointStates
    free_variables: Dict[PrefixName, FreeVariable]
    virtual_free_variables: Dict[PrefixName, FreeVariable]
    context_manager_active: bool = False
    _default_limits: Dict[Derivatives, float]
    _default_weights: Dict[Derivatives, float]
    _root_link_name: PrefixName = None
    _controlled_joints: List[PrefixName]

    def __init__(self):
        self.default_link_color = ColorRGBA(1, 1, 1, 0.75)
        self.connection_prefix = 'connection'
        self.fast_all_fks = None
        self._state_version = 0
        self._model_version = 0
        self._controlled_joints = []
        self.clear()
        self.set_default_weights()

    def set_default_weights(self,
                            velocity_weight: float = 0.01,
                            acceleration_weight: float = 0,
                            jerk_weight: float = 0.0):
        """
        The default values are set automatically, even if this function is not called.
        A typical goal has a weight of 1, so the values in here should be sufficiently below that.
        """
        self.update_default_weights({Derivatives.velocity: velocity_weight,
                                     Derivatives.acceleration: acceleration_weight,
                                     Derivatives.jerk: jerk_weight})

    @property
    def root_link_name(self) -> PrefixName:
        return self._root_link_name

    @property
    def root_link(self) -> Link:
        if self._root_link_name is None:
            raise WorldException('no root_link set')
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
        for v in self.free_variables.values():
            for d, new_limit in new_limits.items():
                v.set_lower_limit(d, -new_limit if new_limit is not None else None)
                v.set_upper_limit(d, new_limit)

    def update_default_weights(self, new_weights: Dict[Derivatives, float]):
        if not hasattr(self, '_default_weights'):
            self._default_weights = {}
        for derivative, weight in new_weights.items():
            self._default_weights[derivative] = weight
        assert len(self._default_weights) == max(self._default_weights)

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
            raise UnknownJointException(f'Multiple matches for \'{joint_name}\' found: \'{matches}\'.')
        if len(matches) == 0:
            raise UnknownJointException(f'No matches for \'{joint_name}\' found: \'{matches}\'.')
        return matches[0]

    def search_for_joint_of_type(self, joint_types: Tuple[Type[Joint]]) -> List[Joint]:
        return [j for j in self.joints.values() if isinstance(j, joint_types)]

    def get_drive_joint(self, possible_types: Tuple[Type[Joint]] = (OmniDrive, DiffDrive),
                        joint_name: Optional[PrefixName] = None) -> Joint:
        if joint_name is None:
            joints = self.search_for_joint_of_type(possible_types)
            if len(joints) == 1:
                return joints[0]
            elif len(joints) == 0:
                raise ValueError('No joints found')
            else:
                raise ValueError(f'Multiple joints found: {joints}')
        else:
            joint = self.joints[joint_name]
            assert isinstance(joint, possible_types)
            return joint

    @modifies_world
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

    def get_joint(self, joint_name: PrefixName, group_name: Optional[str] = None) -> Joint:
        """
        Like get_joint_name, but returns the actual joint.
        """
        return self.joints[joint_name]

    def get_one_dof_joint_symbol(self, joint_name: PrefixName, derivative: Derivatives) -> Union[cas.Expression, float]:
        """
        returns a symbol that refers to the given joint
        """
        if not self.has_joint(joint_name):
            raise KeyError(f'World doesn\'t have joint named: {joint_name}.')
        joint = self.joints[joint_name]
        if isinstance(joint, OneDofJoint):
            return joint.get_symbol(derivative)
        raise TypeError(f'get_joint_position_symbol is only supported for OneDofJoint, not {type(joint)}')

    def get_link_name(self, link_name: str, group_name: Optional[str] = None) -> PrefixName:
        get_middleware().logwarn(f'Deprecated warning: use \'search_for_link_name\' instead of \'get_link_name\'.')
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
        clear_memo(self.compute_fk)
        clear_memo(self.compute_fk_with_collision_offset_np)
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
    def _notify_model_change(self):
        """
        Call this function if you have changed the model of the world to trigger necessary events and increase
        the model version number.
        """
        if not self.context_manager_active:
            self._fix_tree_structure()
            self.reset_cache()
            self.init_all_fks()
            self._cleanup_unused_free_variable()
            self.notify_state_change()
            self._model_version += 1

    def _travel_branch(self, link_name: PrefixName, companion: TravelCompanion):
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
                self._travel_branch(child_link_name, companion)

    def _search_branch(self,
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
        self._travel_branch(link_name, companion=collector_companion)
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
        links, joints = self._search_branch(link_name=child_link_name,
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
        link_names, joint_names = self._search_branch(link_name=child_link_name,
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
        new_group = WorldBranch(name, root_link_name, world=self, actuated=actuated)
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

    @modifies_world
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

    @modifies_world
    def add_virtual_free_variable(self, name: PrefixName) -> FreeVariable:
        free_variable = FreeVariable(name=name,
                                     lower_limits={},
                                     upper_limits={},
                                     quadratic_weights={})
        self.virtual_free_variables[name] = free_variable
        return free_variable

    def update_state(self, next_commands: NextCommands, dt: float, max_derivative: Derivatives) -> None:
        for free_variable_name, command in next_commands.free_variable_data.items():
            self.state[free_variable_name][max_derivative] = command[-1]
            for i in range(max_derivative - 1, -1, -1):
                self.state[free_variable_name][i] += self.state[free_variable_name][i + 1] * dt
        for joint in self.joints.values():
            if isinstance(joint, VirtualFreeVariables):
                joint.update_state(dt)

    @modifies_world
    @profile
    def add_urdf(self,
                 urdf: str,
                 group_name: Optional[str] = None,
                 parent_link_name: Optional[PrefixName] = None,
                 pose: Optional[cas.TransMatrix] = None,
                 actuated: bool = False):
        """
        Add a urdf to the world at parent_link_name and create a SubWorldTree named group_name for it.
        :param urdf: urdf as str, not a file path
        :param group_name: name of the group that will be created. default is name in urdf.
        :param parent_link_name: where the urdf will be attached
        :param actuated: if the urdf is controlled by Giskard, important for self collision avoidance
        """
        with suppress_stderr():
            try:
                parsed_urdf: up.Robot = up.URDF.from_xml_string(hacky_urdf_parser_fix(urdf))
            except Exception as e:
                raise CorruptURDFException(str(e))
        if group_name is None:
            group_name = robot_name_from_urdf_string(urdf)
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
            self.add_link(urdf_root_link)
            joint = Joint6DOF(name=PrefixName(group_name, self.connection_prefix),
                              parent_link_name=parent_link_name,
                              child_link_name=urdf_root_link.name)
            joint.update_transform(pose)
            self.add_joint(joint)
        else:
            urdf_link = parsed_urdf.link_map[urdf_root_link_name]
            urdf_root_link = Link.from_urdf(urdf_link=urdf_link,
                                            prefix=group_name,
                                            color=self.default_link_color)
            self.add_link(urdf_root_link)

        def recursive_parser(urdf: up.Robot, parent_link: Link):
            short_name = parent_link.name.short_name
            if short_name not in urdf.child_map:
                return  # stop because link has no child links
            for child_joint_name, child_link_name in urdf.child_map[short_name]:
                # add link
                urdf_link = urdf.link_map[child_link_name]
                child_link = Link.from_urdf(urdf_link=urdf_link,
                                            prefix=group_name,
                                            color=self.default_link_color)
                self.add_link(child_link)

                # add joint
                urdf_joint: up.Joint = urdf.joint_map[child_joint_name]
                joint = Joint.from_urdf(urdf_joint, group_name)
                if not isinstance(joint, FixedJoint):
                    for derivative, limit in self.default_limits.items():
                        joint.free_variable.set_lower_limit(derivative, -limit if limit is not None else None)
                        joint.free_variable.set_upper_limit(derivative, limit)
                self.add_joint(joint)

                recursive_parser(urdf, child_link)

        number_of_links_before = len(self.links)
        recursive_parser(parsed_urdf, urdf_root_link)
        if number_of_links_before + len(parsed_urdf.links) - 1 != len(self.links):
            # -1 because root link already exists
            raise WorldException(f'Failed to add urdf \'{group_name}\' to world')

        self.register_group(group_name, urdf_root_link_name_prefixed, actuated=actuated)

    @modifies_world
    def add_fixed_joint(self, parent_link: Link, child_link: Link, joint_name: Optional[PrefixName] = None,
                        transform: Optional[cas.TransMatrix] = None) -> None:
        self._raise_if_link_does_not_exist(parent_link.name)
        self._raise_if_link_does_not_exist(child_link.name)
        if joint_name is None:
            joint_name = PrefixName(f'{parent_link.name}_{child_link.name}_fixed_joint', None)
        self.add_joint(FixedJoint(name=joint_name,
                                  parent_link_name=parent_link.name,
                                  child_link_name=child_link.name,
                                  parent_T_child=transform))

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

    def get_group_name_containing_link(self, link_name: Union[PrefixName, str]) -> str:
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

    def clear(self):
        self.state = JointStates()
        self.links = {}
        self.joints = {}
        self.free_variables = {}
        self.virtual_free_variables = {}
        self.groups: Dict[str, WorldBranch] = {}
        self.reset_cache()

    def _fix_tree_structure(self) -> None:
        """
        This function fixes the tree structure based on the parent and child link of joints
        """
        for joint in self.joints.values():
            # parent and child link must exist
            self._raise_if_link_does_not_exist(joint.parent_link_name)
            self._raise_if_link_does_not_exist(joint.child_link_name)
            # add this joint as child joint of parent link
            parent_link = self.links[joint.parent_link_name]
            if joint.name not in parent_link.child_joint_names:
                parent_link.child_joint_names.append(joint.name)
            # set this joint as parent joint of child link
            child_link = self.links[joint.child_link_name]
            if child_link.parent_joint_name is None:
                child_link.parent_joint_name = joint.name
            else:
                assert child_link.parent_joint_name == joint.name
        self._fix_root_link()
        for link in self.links.values():
            if link != self.root_link:
                # if not root link, the parent joint has to exist
                self._raise_if_joint_does_not_exist(link.parent_joint_name)
            # all child joints have to exist
            for child_joint_name in link.child_joint_names:
                self._raise_if_joint_does_not_exist(child_joint_name)

    def _fix_root_link(self):
        # search for links with no parent joint
        orphans = []
        for link_name, link in self.links.items():
            if link.parent_joint_name is None:
                orphans.append(link_name)
        # if there are multiple links with no parent joints, we have multiple trees
        if len(orphans) > 1:
            raise WorldException(f'Found multiple orphaned links: {orphans}.')
        self._root_link_name = orphans[0]

    def _raise_if_link_does_not_exist(self, link_name: PrefixName):
        if link_name not in self.links:
            raise UnknownLinkException(f'Link \'{link_name}\' does not exist.')

    def _raise_if_link_exists(self, link_name: PrefixName):
        if link_name in self.links:
            raise DuplicateNameException(f'Link \'{link_name}\' does already exist.')

    def _raise_if_joint_does_not_exist(self, joint_name: PrefixName):
        if joint_name not in self.joints:
            raise UnknownJointException(f'Joint \'{joint_name}\' does not exist.')

    def _raise_if_joint_exists(self, joint_name: PrefixName):
        if joint_name in self.joints:
            raise DuplicateNameException(f'Joint \'{joint_name}\' does already exist.')

    @modifies_world
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
            fk = cas.TransMatrix(self.compute_fk_np(new_parent_link_name, joint.child_link_name))
            joint.parent_link_name = new_parent_link_name
            joint.parent_T_child = fk
        elif isinstance(joint, Joint6DOF):
            pose = self.compute_fk(new_parent_link_name, joint.child_link_name)
            joint.parent_link_name = new_parent_link_name
            joint.update_transform(pose)
        else:
            raise NotImplementedError('Can only change fixed joints and TFJoints')

        old_parent_link.child_joint_names.remove(joint_name)
        new_parent_link.child_joint_names.append(joint_name)

    def _cleanup_unused_free_variable(self):
        used_variables = []
        for joint_name in self.movable_joint_names:
            joint = self.joints[joint_name]
            used_variables.extend(free_variable.name for free_variable in joint.free_variables)
        for free_variable_name in self.free_variables:
            if free_variable_name not in used_variables:
                try:
                    del self.state[free_variable_name]
                except KeyError as e:
                    # idk why this sometimes throws a KeyError, even though the key existed, but the deletion
                    # seems to work anyway ...
                    pass

    @modifies_world
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

    @modifies_world
    def delete_group(self, group_name: str):
        """
        Delete the group and all links and joints contained in it.
        """
        if group_name not in self.groups:
            raise UnknownGroupException(f'Can\'t delete unknown group: \'{group_name}\'')
        self.delete_branch(self.groups[group_name].root_link_name)

    @modifies_world
    def delete_branch(self, link_name: PrefixName):
        """
        Delete every link and joint from link_name downward, including the link.
        """
        self.delete_branch_at_joint(self.links[link_name].parent_joint_name)

    @modifies_world
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
                    get_middleware().loginfo(f'Deleted group \'{group_name}\', because it\'s root link got removed.')
            for child_joint_name in link.child_joint_names:
                child_joint = self.joints.pop(child_joint_name)  # type: Joint
                helper(child_joint.child_link_name)

        helper(joint.child_link_name)

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
        return self._controlled_joints

    @controlled_joints.setter
    def controlled_joints(self, value: List[PrefixName]):
        self._controlled_joints = value

    def register_controlled_joints(self, controlled_joints: List[PrefixName]) -> None:
        """
        Flag these joints as controlled.
        """
        old_controlled_joints = set(self.controlled_joints)
        new_controlled_joints = set(controlled_joints)
        for joint in new_controlled_joints:
            if self.is_joint_fixed(joint):
                raise WorldException(f'Can\'t register fixed joint as controllable')
        double_joints = old_controlled_joints.intersection(new_controlled_joints)
        if double_joints:
            raise DuplicateNameException(f'Controlled joints \'{double_joints}\' are already registered!')
        unknown_joints = new_controlled_joints.difference(self.joint_names_as_set)
        if unknown_joints:
            raise UnknownGroupException(f'Trying to register unknown joints: \'{unknown_joints}\'')
        old_controlled_joints.update(new_controlled_joints)
        self.controlled_joints = list(sorted(old_controlled_joints))

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
    def compose_fk_expression(self, root_link: PrefixName, tip_link: PrefixName) -> cas.TransMatrix:
        """
        Multiplies all transformation matrices in the chain between root_link and tip_link
        :param root_link:
        :param tip_link:
        :return: 4x4 homogenous transformation matrix
        """
        fk = cas.TransMatrix()
        root_chain, _, tip_chain = self.compute_split_chain(root_link, tip_link, add_joints=True, add_links=False,
                                                            add_fixed_joints=True, add_non_controlled_joints=True)
        for joint_name in root_chain:
            a = self.joints[joint_name].parent_T_child
            ai = a.inverse()
            fk = fk.dot(ai)
        for joint_name in tip_chain:
            a = self.joints[joint_name].parent_T_child
            fk = fk.dot(a)
        fk.reference_frame = root_link
        fk.child_frame = tip_link
        return fk

    def get_fk_velocity(self, root: PrefixName, tip: PrefixName) -> cas.Expression:
        # FIXME, only use symbols of fk expr?
        r_T_t = self.compose_fk_expression(root, tip)
        r_R_t = r_T_t.to_rotation()
        axis, angle = r_R_t.to_axis_angle()
        r_R_t_axis_angle = axis * angle
        r_P_t = r_T_t.to_position()
        fk = cas.Expression([r_P_t[0],
                             r_P_t[1],
                             r_P_t[2],
                             r_R_t_axis_angle[0],
                             r_R_t_axis_angle[1],
                             r_R_t_axis_angle[2]])
        return cas.total_derivative(fk,
                                    self.joint_position_symbols,
                                    self.joint_velocity_symbols)

    @copy_memoize
    def compute_fk(self, root_link: PrefixName, tip_link: PrefixName) -> cas.TransMatrix:
        result = cas.TransMatrix(self.compute_fk_np(root_link, tip_link))
        result.reference_frame = root_link
        result.child_frame = tip_link
        return result

    def compute_fk_point(self, root_link: PrefixName, tip_link: PrefixName) -> cas.Point3:
        return self.compute_fk(root_link=root_link, tip_link=tip_link).to_position()

    @memoize
    @profile
    def compute_fk_with_collision_offset_np(self, root_link: PrefixName, tip_link: PrefixName,
                                            collision_id: int) -> np.ndarray:
        root_T_tip = self.compute_fk_np(root_link, tip_link)
        tip_link = self.links[tip_link]
        return np.dot(root_T_tip, tip_link.collisions[collision_id].link_T_geometry.to_np())

    @profile
    def compute_all_collision_fks(self):
        return self._fk_computer.compiled_collision_fks.fast_call(self._fk_computer.subs)

    @profile
    def init_all_fks(self):
        class ExpressionCompanion(TravelCompanion):
            idx_start: Dict[PrefixName, int]
            compiled_collision_fks: CompiledFunction
            compiled_all_fks: CompiledFunction
            str_params: List[str]
            fks: np.ndarray
            fks_exprs: Dict[PrefixName, cas.TransMatrix]

            def __init__(self, world: WorldTree):
                self.world = world
                self.fks_exprs = {self.world.root_link_name: cas.TransMatrix()}
                self.tf = OrderedDict()

            @profile
            def joint_call(self, joint_name: PrefixName) -> bool:
                joint = self.world.joints[joint_name]
                map_T_parent = self.fks_exprs[joint.parent_link_name]
                self.fks_exprs[joint.child_link_name] = map_T_parent.dot(joint.parent_T_child)
                self.tf[(joint.parent_link_name, joint.child_link_name)] = joint.parent_T_child_as_pos_quaternion()
                return False

            @profile
            def compile_fks(self):
                all_fks = cas.vstack([self.fks_exprs[link_name] for link_name in self.world.link_names_as_set])
                tf = cas.vstack([pose for pose in self.tf.values()])
                collision_fks = []
                for link_name in sorted(self.world.link_names_with_collisions):
                    if link_name == self.world.root_link_name:
                        continue
                    collision_fks.append(self.fks_exprs[link_name])
                collision_fks = cas.vstack(collision_fks)
                params = set()
                params.update(all_fks.free_symbols())
                params.update(collision_fks.free_symbols())
                params = list(params)
                self.str_params = [str(v) for v in params]
                self.compiled_all_fks = all_fks.compile(parameters=params)
                self.compiled_collision_fks = collision_fks.compile(parameters=params)
                self.compiled_tf = tf.compile(parameters=params)
                self.idx_start = {link_name: i * 4 for i, link_name in enumerate(self.world.link_names_as_set)}

            @profile
            def recompute(self):
                self.compute_fk_np.memo.clear()
                self.subs = symbol_manager.resolve_symbols(self.compiled_all_fks.str_params)
                self.fks = self.compiled_all_fks.fast_call(self.subs)

            def compute_tf(self):
                return self.compiled_tf.fast_call(self.subs)

            @memoize
            @profile
            def compute_fk_np(self, root: PrefixName, tip: PrefixName) -> np.ndarray:
                root_is_world = root == self.world.root_link_name
                tip_is_world = tip == self.world.root_link_name

                if not tip_is_world:
                    i = self.idx_start[tip]
                    map_T_tip = self.fks[i:i + 4]
                    if root_is_world:
                        return map_T_tip

                if not root_is_world:
                    i = self.idx_start[root]
                    map_T_root = self.fks[i:i + 4]
                    root_T_map = mymath.inverse_frame(map_T_root)
                    if tip_is_world:
                        return root_T_map

                if tip_is_world and root_is_world:
                    return np.eye(4)

                return root_T_map @ map_T_tip

        new_fks = ExpressionCompanion(self)
        self._travel_branch(self.root_link_name, new_fks)
        new_fks.compile_fks()
        self._fk_computer = new_fks

    @profile
    def _recompute_fks(self):
        self._fk_computer.recompute()

    @profile
    def compute_fk_np(self, root: PrefixName, tip: PrefixName) -> np.ndarray:
        return self._fk_computer.compute_fk_np(root, tip)

    @profile
    def compose_fk_evaluated_expression(self, root: PrefixName, tip: PrefixName) -> cas.TransMatrix:
        result: cas.TransMatrix = symbol_manager.get_expr(f'god_map.world.compute_fk_np(\'{root}\', \'{tip}\')',
                                                          output_type_hint=cas.TransMatrix)
        result.reference_frame = root
        result.child_frame = tip
        return result

    @memoize
    @profile
    def are_linked(self, link_a: PrefixName, link_b: PrefixName,
                   do_not_ignore_non_controlled_joints: bool = False,
                   joints_to_be_assumed_fixed: Optional[Sequence[PrefixName]] = None) -> bool:
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

    @modifies_world
    def add_link(self, link: Link) -> None:
        self._raise_if_link_exists(link.name)
        self.links[link.name] = link

    @modifies_world
    def add_joint(self, joint: Joint):
        self._raise_if_joint_exists(joint.name)
        self.joints[joint.name] = joint

    def joint_limit_expr(self, joint_name: PrefixName, order: Derivatives) \
            -> Tuple[Optional[cas.symbol_expr_float], Optional[cas.symbol_expr_float]]:
        return self.joints[joint_name].get_limit_expressions(order)

    @overload
    def transform(self, target_frame: PrefixName, geometric_cas_object: cas.Point3) -> cas.Point3:
        ...

    @overload
    def transform(self, target_frame: PrefixName, geometric_cas_object: cas.TransMatrix) -> cas.TransMatrix:
        ...

    @overload
    def transform(self, target_frame: PrefixName, geometric_cas_object: cas.Vector3) -> cas.Vector3:
        ...

    @overload
    def transform(self, target_frame: PrefixName, geometric_cas_object: cas.Quaternion) -> cas.Quaternion:
        ...

    @overload
    def transform(self, target_frame: PrefixName, geometric_cas_object: cas.RotationMatrix) -> cas.RotationMatrix:
        ...

    def transform(self, target_frame, geometric_cas_object):
        if geometric_cas_object.reference_frame is None:
            raise WorldException('Can\'t transform an object without reference_frame.')
        target_frame_T_reference_frame = self.compute_fk(root_link=target_frame,
                                                         tip_link=geometric_cas_object.reference_frame)
        if isinstance(geometric_cas_object, cas.Quaternion):
            reference_frame_R = geometric_cas_object.to_rotation_matrix()
            target_frame_R = target_frame_T_reference_frame.dot(reference_frame_R)
            return target_frame_R.to_quaternion()
        else:
            return target_frame_T_reference_frame.dot(geometric_cas_object)

    def compute_joint_limits(self, joint_name: PrefixName, order: Derivatives) \
            -> Tuple[Optional[cas.symbol_expr_float], Optional[cas.symbol_expr_float]]:
        try:
            lower_limit, upper_limit = self.joint_limit_expr(joint_name, order)
        except KeyError:
            # joint has no limits for this derivative
            return None, None
        if not isinstance(lower_limit, (int, float)) and lower_limit is not None:
            lower_limit = symbol_manager.evaluate_expr(lower_limit)
        if not isinstance(upper_limit, (int, float)) and upper_limit is not None:
            upper_limit = symbol_manager.evaluate_expr(upper_limit)
        return lower_limit, upper_limit

    def get_joint_position_limits(self, joint_name: PrefixName) -> Tuple[Optional[float], Optional[float]]:
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

    def save_graph_pdf(self, folder_name: str) -> None:
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
        file_name = f'{folder_name}/world_tree.pdf'
        world_graph.write_pdf(file_name)


class WorldBranch(WorldTreeInterface):
    def __init__(self, name: str, root_link_name: PrefixName, world: WorldTree, actuated: bool = False):
        self.name = name
        self.root_link_name = root_link_name
        self.actuated = actuated
        self.world = world

    def get_siblings_with_collisions(self, joint_name: PrefixName) -> List[PrefixName]:
        siblings = self.world.get_siblings_with_collisions(joint_name)
        return [x for x in siblings if x in self.link_names_as_set]

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
        return [j for j in self.world.controlled_joints if j in self.joint_names_as_set]

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
                links, joints = self.world._search_branch(link_name=child_link_name,
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
        unmovable_links, _ = self.world._search_branch(link_name=self.root_link_name,
                                                       stop_at_joint_when=lambda
                                                           joint_name: joint_name in self.controlled_joints,
                                                       collect_link_when=self.world.has_link_collisions)
        return unmovable_links

    @property
    def base_pose(self) -> cas.TransMatrix:
        return self.world.compute_fk(self.world.root_link_name, self.root_link_name)

    @property
    def state(self) -> JointStates:
        return JointStates({j: self.world.state[j] for j in self.joints if j in self.world.state})

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
                group.root_link_name in self.links
                and group.root_link_name != self.root_link_name
                and group.name != self.name}

    @cached_property
    def links(self) -> Dict[PrefixName, Link]:
        def helper(root_link: Link) -> Dict[PrefixName, Link]:
            links = {root_link.name: root_link}
            for j in root_link.child_joint_names:
                j = self.world.joints[j]
                child_link = self.world.links[j.child_link_name]
                links.update(helper(child_link))
            return links

        return helper(self.root_link)

    def compute_fk_pose(self, root: PrefixName, tip: PrefixName) -> cas.TransMatrix:
        return self.world.compute_fk(root, tip)

    def compute_fk_pose_with_collision_offset(self, root: PrefixName, tip: PrefixName, collision_id: int) \
            -> cas.TransMatrix:
        return self.world.compute_fk_with_collision_offset_np(root, tip, collision_id)

    def is_link_controlled(self, link_name: PrefixName) -> bool:
        return self.world.is_link_controlled(link_name)
