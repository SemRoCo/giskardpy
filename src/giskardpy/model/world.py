import numbers
import traceback
from copy import deepcopy
from functools import cached_property
from itertools import combinations
from typing import Dict, Union, Tuple, Set, Optional

import numpy as np
import urdf_parser_py.urdf as up
from geometry_msgs.msg import PoseStamped, Pose, PointStamped, Point, Vector3Stamped, Vector3, QuaternionStamped

import giskardpy.utils.math as mymath
from giskard_msgs.msg import WorldBody
from giskardpy import casadi_wrapper as w, identifier
from giskardpy.casadi_wrapper import CompiledFunction
from giskardpy.data_types import JointStates, KeyDefaultDict, order_map
from giskardpy.data_types import PrefixName
from giskardpy.exceptions import DuplicateNameException, UnknownGroupException, UnknownLinkException
from giskardpy.god_map import GodMap
from giskardpy.model.joints import Joint, PrismaticJoint, RevoluteJoint, ContinuousJoint, MovableJoint, \
    FixedJoint, MimicJoint, DiffDriveWheelsJoint
from giskardpy.model.joints import OneDofJoint
from giskardpy.model.links import Link
from giskardpy.model.utils import hacky_urdf_parser_fix
from giskardpy.utils import logging
from giskardpy.utils.tfwrapper import homo_matrix_to_pose, np_to_pose, msg_to_homogeneous_matrix, kdl_to_quaternion, \
    np_to_kdl
from giskardpy.utils.utils import suppress_stderr, memoize


class TravelCompanion(object):
    def link_call(self, link_name: Union[PrefixName, str]) -> bool:
        """
        :return: return True to stop climbing up the branch
        """
        return False

    def joint_call(self, joint_name: Union[PrefixName, str]) -> bool:
        """
        :return: return True to stop climbing up the branch
        """
        return False


class WorldTree(object):
    joints: Dict[Union[PrefixName, str], Joint]
    links: Dict[Union[PrefixName, str], Link]
    god_map: GodMap

    def __init__(self, god_map=None):
        self.god_map = god_map
        if self.god_map is not None:
            self.god_map.set_data(identifier.world, self)
        self.connection_prefix = 'connection'
        self._state_version = 0
        self._model_version = 0
        self._clear()

    @property
    def version(self):
        return self._model_version, self._state_version

    @property
    def model_version(self):
        return self._model_version

    @property
    def state_version(self):
        return self._state_version

    def _clear_memo(self, f):
        try:
            if hasattr(f, 'memo'):
                f.memo.clear()
            else:
                del f
        except:
            pass

    @profile
    def notify_state_change(self):
        self._clear_memo(self.compute_fk_pose)
        self._clear_memo(self.compute_fk_pose_with_collision_offset)
        self._recompute_fks()
        self._state_version += 1

    @profile
    def notify_model_change(self):
        for group in self.groups.values():
            group.reset_cache()
        try:
            del self.link_names
        except:
            pass  # property wasn't called
        try:
            del self.link_names_with_collisions
        except:
            pass  # property wasn't called
        try:
            del self.movable_joints_as_set
        except:
            pass  # property wasn't called
        try:
            del self.movable_joints
        except:
            pass  # property wasn't called
        self._clear_memo(self.get_directly_controlled_child_links_with_collisions)
        self._clear_memo(self.get_directly_controlled_child_links_with_collisions)
        self._clear_memo(self.compute_chain_reduced_to_controlled_joints)
        self._clear_memo(self.get_movable_parent_joint)
        self._clear_memo(self.get_controlled_parent_joint_of_link)
        self._clear_memo(self.get_controlled_parent_joint_of_joint)
        self._clear_memo(self.compute_split_chain)
        self._clear_memo(self.are_linked)
        self._clear_memo(self.compose_fk_expression)
        self._clear_memo(self.compute_chain)
        self._clear_memo(self.is_link_controlled)
        self.init_all_fks()
        self.notify_state_change()
        self._model_version += 1

    def travel_branch(self, link_name, companion: TravelCompanion):
        link = self.links[link_name]
        if not companion.link_call(link_name):
            for child_joint_name in link.child_joint_names:
                if companion.joint_call(child_joint_name):
                    continue
                child_link_name = self.joints[child_joint_name].child_link_name
                self.travel_branch(child_link_name, companion)

    def search_branch(self, link_name,
                      stop_at_joint_when=None, stop_at_link_when=None,
                      collect_joint_when=None, collect_link_when=None):
        """

        :param joint_name:
        :param stop_at_joint_when: If None, 'lambda joint_name: False' is used.
        :param stop_at_link_when: If None, 'lambda link_name: False' is used.
        :param collect_joint_when: If None, 'lambda joint_name: False' is used.
        :param collect_link_when: If None, 'lambda link_name: False' is used.
        :return: Collected links and joints. Might include 'link_name'
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
    def get_directly_controlled_child_links_with_collisions(self, joint_name):
        child_link_name = self.joints[joint_name].child_link_name
        links, joints = self.search_branch(child_link_name,
                                           stop_at_joint_when=self.is_joint_controlled,
                                           collect_link_when=self.has_link_collisions)
        return links

    def get_siblings_with_collisions(self, joint_name):
        """
        Goes up the tree until the first controlled joint and then down again until another controlled joint or
        the joint_name is reached again. Collects all links with collision along the way.
        :param joint_name:
        :return:
        """
        try:
            parent_joint = self.search_for_parent_joint(joint_name, stop_when=self.is_joint_controlled)
        except KeyError as e:
            return []

        def stop_at_joint_when(other_joint_name):
            return joint_name == other_joint_name or self.is_joint_controlled(other_joint_name)

        child_link_name = self.joints[parent_joint].child_link_name
        links, joints = self.search_branch(child_link_name,
                                           stop_at_joint_when=stop_at_joint_when,
                                           collect_link_when=self.has_link_collisions)
        return links

    def register_group(self, name, root_link_name):
        if root_link_name not in self.links:
            raise KeyError('World doesn\'t have link \'{}\''.format(root_link_name))
        if name in self.groups:
            raise DuplicateNameException('Group with name {} already exists'.format(name))
        new_group = SubWorldTree(name, root_link_name, self)
        # if the group is a subtree of a subtree, register it for the subtree as well
        for group in self.groups.values():
            if root_link_name in group.links:
                group.groups[name] = new_group
        self.groups[name] = new_group

    @property
    def group_names(self):
        return set(self.groups.keys())

    @property
    def minimal_group_names(self):
        group_names = self.group_names
        for group in self.groups.values():
            for group_name in group.group_names:
                if group_name in group_names:
                    group_names.remove(group_name)
        return group_names

    @property
    def root_link(self):
        return self.links[self.root_link_name]

    @cached_property
    def link_names(self):
        return set(self.links.keys())

    @property
    def link_names_with_visuals(self):
        return set(link.name for link in self.links.values() if link.has_visuals())

    @cached_property
    def link_names_with_collisions(self):
        return set(link.name for link in self.links.values() if link.has_collisions())

    @cached_property
    def link_names_without_collisions(self):
        return self.link_names.difference(self.link_names_with_collisions)

    @property
    def joint_names(self):
        return list(self.joints.keys())

    @property
    def joint_names_as_set(self):
        return set(self.joints.keys())

    def add_urdf(self, urdf, prefix=None, parent_link_name=None, group_name=None):
        with suppress_stderr():
            parsed_urdf = up.URDF.from_xml_string(hacky_urdf_parser_fix(urdf))  # type: up.Robot
        if group_name is None:
            group_name = parsed_urdf.name
        if group_name in self.groups:
            raise DuplicateNameException(
                'Failed to add group \'{}\' because one with such a name already exists'.format(group_name))
        if parent_link_name is None:
            parent_link = self.root_link
        else:
            parent_link = self.links[parent_link_name]
        if False:
            base_footprint, odom = self._add_diff_drive_joint(urdf=parsed_urdf,
                                                              map_link=parent_link,
                                                              prefix=prefix)
        else:
            odom = Link.from_urdf(parsed_urdf.link_map[parsed_urdf.get_root()], None)
            base_footprint = odom
            self._add_fixed_joint(parent_link, odom,
                                  joint_name=PrefixName(PrefixName(parsed_urdf.name, prefix), self.connection_prefix))

        def helper(urdf, parent_link):
            short_name = parent_link.name.short_name
            if short_name not in urdf.child_map:
                return
            for child_joint_name, child_link_name in urdf.child_map[short_name]:
                urdf_link = urdf.link_map[child_link_name]
                child_link = Link.from_urdf(urdf_link, prefix)

                urdf_joint = urdf.joint_map[child_joint_name]
                joint = Joint.from_urdf(urdf_joint, prefix, parent_link.name, child_link.name, self.god_map)

                self._link_joint_to_links(joint, child_link)
                helper(urdf, child_link)

        helper(parsed_urdf, base_footprint)

        if group_name is not None:
            self.register_group(group_name, odom.name)
        if self.god_map is not None:
            self.sync_with_paramserver()
        self._set_free_variables_on_mimic_joints(group_name)

    def _add_fixed_joint(self, parent_link, child_link, joint_name=None):
        if joint_name is None:
            joint_name = '{}_{}_fixed_joint'.format(parent_link.name, child_link.name)
        connecting_joint = FixedJoint(name=joint_name,
                                      parent_link_name=parent_link.name,
                                      child_link_name=child_link.name)
        self._link_joint_to_links(connecting_joint, child_link)

    def _add_diff_drive_joint(self, urdf, map_link, odom_link_name='odom', prefix=None):
        # create odom and link to map with fixed joint
        odom = Link(name=PrefixName(odom_link_name, prefix))
        self._add_fixed_joint(parent_link=map_link,
                              child_link=odom,
                              joint_name=PrefixName(PrefixName(urdf.name, prefix), self.connection_prefix))

        # create urdf root link and fix to odom with diff drive
        base_footprint_name = urdf.get_robot_root_link()
        base_footprint = Link.from_urdf(urdf.link_map[base_footprint_name], None)

        trans_lower_limits = {1: -1}
        trans_upper_limits = {1: 1}
        rot_lower_limits = {1: -1}
        rot_upper_limits = {1: 1}
        # diff_drive_joint = DiffDriveJoint(name=PrefixName('diff_drive', None),
        diff_drive_joint = DiffDriveWheelsJoint(name=PrefixName('diff_drive', None),
                                                parent_link_name=odom_link_name,
                                                child_link_name=base_footprint_name,
                                                god_map=self.god_map)
        diff_drive_joint.create_free_variables(identifier.joint_states, trans_lower_limits, trans_upper_limits,
                                               rot_lower_limits, rot_upper_limits)
        self._link_joint_to_links(diff_drive_joint, base_footprint)
        return base_footprint, odom

    def _set_free_variables_on_mimic_joints(self, group_name):
        # TODO prevent this from happening twice
        for joint_name, joint in self.groups[group_name].joints.items():  # type: (PrefixName, MimicJoint)
            if self.is_joint_mimic(joint_name):
                mimed_joint = self.joints[joint.mimed_joint_name]  # type: OneDofJoint
                joint.set_mimed_free_variable(mimed_joint.free_variable)

    def get_parent_link_of_link(self, link_name: Union[PrefixName, str]) -> PrefixName:
        return self.joints[self.links[link_name].parent_joint_name].parent_link_name

    @memoize
    def compute_chain_reduced_to_controlled_joints(self, link_a, link_b):
        chain1, connection, chain2 = self.compute_split_chain(link_b, link_a, joints=True, links=True, fixed=True,
                                                              non_controlled=True)
        chain = chain1 + connection + chain2
        for i, thing in enumerate(chain):
            if i % 2 == 1 and thing in self.controlled_joints:
                new_link_b = chain[i - 1]
                break
        else:
            raise KeyError(f'no controlled joint in chain between {link_a} and {link_b}')
        for i, thing in enumerate(reversed(chain)):
            if i % 2 == 1 and thing in self.controlled_joints:
                new_link_a = chain[len(chain) - i]
                break
        else:
            raise KeyError('no controlled joint in chain between {} and {}'.format(link_a, link_b))
        return new_link_a, new_link_b

    @memoize
    def get_movable_parent_joint(self, link_name):
        joint = self.links[link_name].parent_joint_name
        while not self.is_joint_movable(joint):
            joint = self.links[self.joints[joint].parent_link_name].parent_joint_name
        return joint

    def get_parent_group_name(self, group_name):
        for potential_parent_group in self.minimal_group_names:
            if group_name in self.groups[potential_parent_group].groups:
                return potential_parent_group
        return group_name


    @profile
    def add_world_body(self, group_name: str,
                       msg: WorldBody,
                       pose: Pose,
                       parent_link_name: Union[str, PrefixName]):
        if group_name in self.groups:
            raise DuplicateNameException(f'Group with name \'{group_name}\' already exists')

        if msg.type == msg.URDF_BODY:
            self.add_urdf(urdf=msg.urdf,
                          parent_link_name=parent_link_name,
                          group_name=group_name,
                          prefix=None)
        else:
            link = Link.from_world_body(prefix=group_name, msg=msg)
            joint = FixedJoint(PrefixName(group_name, self.connection_prefix), parent_link_name, link.name,
                               parent_T_child=w.Matrix(msg_to_homogeneous_matrix(pose)))
            self._link_joint_to_links(joint, link)
            self.register_group(group_name, link.name)
            self.notify_model_change()

    @cached_property
    def movable_joints(self):
        return [j.name for j in self.joints.values() if isinstance(j, MovableJoint) and not isinstance(j, MimicJoint)]

    @cached_property
    def movable_joints_as_set(self):
        return set(self.movable_joints)

    def _clear(self):
        self.state = JointStates()
        if self.god_map is not None:
            self.root_link_name = PrefixName(self.god_map.unsafe_get_data(identifier.map_frame), None)
        else:
            self.root_link_name = 'map'
        self.links = {self.root_link_name: Link(self.root_link_name)}
        self.joints = {}
        self.groups = {}

    @profile
    def delete_all_but_robot(self):
        self._clear()
        self.add_urdf(self.god_map.unsafe_get_data(identifier.robot_description), group_name=None, prefix=None)
        self.notify_model_change()

    def _set_joint_limits(self, linear_limits, angular_limits, order):
        for joint_name in self.movable_joints:  # type: OneDofJoint
            joint = self.joints[joint_name]
            joint.update_limits(linear_limits, angular_limits, order)

    @profile
    def sync_with_paramserver(self):
        new_lin_limits = {}
        new_ang_limits = {}
        for i in range(1, len(self.god_map.unsafe_get_data(identifier.joint_limits)) + 1):
            diff = order_map[i]

            class Linear(object):
                def __init__(self, god_map, diff):
                    self.god_map = god_map
                    self.diff = diff

                def __call__(self, key):
                    return self.god_map.to_symbol(identifier.joint_limits + [self.diff] + ['linear', 'override', key])

            class Angular(object):
                def __init__(self, god_map, diff):
                    self.god_map = god_map
                    self.diff = diff

                def __call__(self, key):
                    return self.god_map.to_symbol(identifier.joint_limits + [self.diff] + ['angular', 'override', key])

            d_linear = KeyDefaultDict(Linear(self.god_map, diff))
            d_angular = KeyDefaultDict(Angular(self.god_map, diff))
            new_lin_limits[i] = d_linear
            new_ang_limits[i] = d_angular
        for joint_name in self.movable_joints:  # type: OneDofJoint
            joint = self.joints[joint_name]
            joint.update_limits(new_lin_limits, new_ang_limits)

        new_weights = {}
        for i in range(1, len(self.god_map.unsafe_get_data(identifier.joint_weights)) + 1):
            def default(joint_name):
                return self.god_map.to_symbol(identifier.joint_weights + [order_map[i], 'override', joint_name])

            d = KeyDefaultDict(default)
            new_weights[i] = d
            # self._set_joint_weights(i, d)
        self.overwrite_joint_weights(new_weights)
        self.notify_model_change()

    def overwrite_joint_weights(self, new_weights):
        for joint_name in self.movable_joints:
            joint = self.joints[joint_name]
            if not self.is_joint_mimic(joint_name):
                joint.update_weights(new_weights)

    @property
    def joint_constraints(self):
        joint_constraints = []
        for joint_name, joint in self.joints.items():
            if joint.has_free_variables():  # FIXME check name clashes
                joint_constraints.extend(joint.free_variable_list)
        return joint_constraints

    def _link_joint_to_links(self, joint, child_link):
        """
        :type joint: Joint
        :type child_link: Link
        """
        parent_link = self.links[joint.parent_link_name]
        if joint.name in self.joints:
            raise DuplicateNameException('Cannot add joint named \'{}\' because already exists'.format(joint.name))
        self.joints[joint.name] = joint
        child_link.parent_joint_name = joint.name
        if child_link.name in self.links:
            raise DuplicateNameException('Cannot add link named \'{}\' because already exists'.format(child_link.name))
        self.links[child_link.name] = child_link
        assert joint.name not in parent_link.child_joint_names
        parent_link.child_joint_names.append(joint.name)

    @profile
    def move_branch(self, joint_name, new_parent_link_name):
        if not self.is_joint_fixed(joint_name):
            raise NotImplementedError('Can only change fixed joints')
        joint = self.joints[joint_name]
        fk = w.Matrix(self.compute_fk_np(new_parent_link_name, joint.child_link_name))
        old_parent_link = self.links[joint.parent_link_name]
        new_parent_link = self.links[new_parent_link_name]

        joint.parent_link_name = new_parent_link_name
        joint.parent_T_child = fk
        old_parent_link.child_joint_names.remove(joint_name)
        new_parent_link.child_joint_names.append(joint_name)
        self.notify_model_change()

    @profile
    def update_joint_parent_T_child(self, joint_name, new_parent_T_child):
        joint = self.joints[joint_name]
        joint.parent_T_child = new_parent_T_child
        self.notify_model_change()

    def move_group(self, group_name, new_parent_link_name):
        group = self.groups[group_name]
        joint_name = self.links[group.root_link_name].parent_joint_name
        if self.joints[joint_name].parent_link_name == new_parent_link_name:
            raise DuplicateNameException(
                '\'{}\' is already attached to \'{}\''.format(group_name, new_parent_link_name))
        self.move_branch(joint_name, new_parent_link_name)

    def delete_group(self, group_name):
        if group_name not in self.groups:
            raise UnknownGroupException(f'Can\'t delete unknown group: \'{group_name}\'')
        self.delete_branch(self.groups[group_name].root_link_name)

    def delete_branch(self, link_name):
        self.delete_branch_at_joint(self.links[link_name].parent_joint_name)

    @profile
    def delete_branch_at_joint(self, joint_name):
        joint = self.joints.pop(joint_name)  # type: Joint
        self.links[joint.parent_link_name].child_joint_names.remove(joint_name)

        def helper(link_name):
            link = self.links.pop(link_name)
            for group_name in list(self.groups.keys()):
                if self.groups[group_name].root_link_name == link_name:
                    del self.groups[group_name]
                    logging.loginfo('Deleted group \'{}\', because it\'s root link got removed.'.format(group_name))
            for child_joint_name in link.child_joint_names:
                child_joint = self.joints.pop(child_joint_name)  # type: Joint
                helper(child_joint.child_link_name)

        helper(joint.child_link_name)
        self.notify_model_change()

    def link_order(self, link_a, link_b):
        """
        TODO find a better name
        this function is used when deciding for which order to calculate the collisions
        true if link_a < link_b
        :type link_a: str
        :type link_b: str
        :rtype: bool
        """
        if self.is_link_controlled(link_a) and not self.is_link_controlled(link_b):
            return True
        elif not self.is_link_controlled(link_a) and self.is_link_controlled(link_b):
            return False
        return link_a < link_b

    def sort_links(self, link_a: Union[PrefixName, str], link_b: Union[PrefixName, str]) \
            -> Tuple[Union[PrefixName, str], Union[PrefixName, str]]:
        if self.link_order(link_a, link_b):
            return link_a, link_b
        return link_b, link_a

    @property
    def controlled_joints(self):
        try:
            return self.god_map.unsafe_get_data(identifier.controlled_joints)
        except KeyError:
            return []

    def register_controlled_joints(self, controlled_joints):
        old_controlled_joints = set(self.controlled_joints)
        new_controlled_joints = set(controlled_joints)
        double_joints = old_controlled_joints.intersection(new_controlled_joints)
        if double_joints:
            raise DuplicateNameException('Controlled joints \'{}\' are already registered!'.format(double_joints))
        unknown_joints = new_controlled_joints.difference(self.joint_names_as_set)
        if unknown_joints:
            raise UnknownGroupException('Trying to register unknown joints: \'{}\''.format(unknown_joints))
        old_controlled_joints.update(new_controlled_joints)
        self.god_map.set_data(identifier.controlled_joints, list(sorted(old_controlled_joints)))

    @memoize
    def get_controlled_parent_joint_of_link(self, link_name):
        joint = self.links[link_name].parent_joint_name
        if self.is_joint_controlled(joint):
            return joint
        return self.get_controlled_parent_joint_of_joint(joint)

    @memoize
    def get_controlled_parent_joint_of_joint(self, joint_name):
        return self.search_for_parent_joint(joint_name, self.is_joint_controlled)

    def search_for_parent_joint(self, joint_name, stop_when=None):
        try:
            joint = self.links[self.joints[joint_name].parent_link_name].parent_joint_name
            while stop_when is not None and not stop_when(joint):
                joint = self.search_for_parent_joint(joint)
        except KeyError as e:
            raise KeyError('\'{}\' has no fitting parent joint'.format(joint_name))
        return joint

    @profile
    @memoize
    def compute_chain(self, root_link_name, tip_link_name, joints, links, fixed, non_controlled):
        # FIXME memoizing this function results in weird errors...
        chain = []
        if links:
            chain.append(tip_link_name)
        link = self.links[tip_link_name]
        while link.name != root_link_name:
            if link.parent_joint_name not in self.joints:
                raise ValueError('{} and {} are not connected'.format(root_link_name, tip_link_name))
            parent_joint = self.joints[link.parent_joint_name]
            parent_link = self.links[parent_joint.parent_link_name]
            if joints:
                if (fixed or not isinstance(parent_joint, FixedJoint)) and \
                        (non_controlled or parent_joint.name in self.controlled_joints):
                    chain.append(parent_joint.name)
            if links:
                chain.append(parent_link.name)
            link = parent_link
        chain.reverse()
        return chain

    @memoize
    def compute_split_chain(self, root, tip, joints, links, fixed, non_controlled):
        if root == tip:
            return [], [], []
        root_chain = self.compute_chain(self.root_link_name, root, False, True, True, True)
        tip_chain = self.compute_chain(self.root_link_name, tip, False, True, True, True)
        for i in range(min(len(root_chain), len(tip_chain))):
            if root_chain[i] != tip_chain[i]:
                break
        else:
            i += 1
        connection = tip_chain[i - 1]
        root_chain = self.compute_chain(connection, root, joints, links, fixed, non_controlled)
        if links:
            root_chain = root_chain[1:]
        root_chain.reverse()
        tip_chain = self.compute_chain(connection, tip, joints, links, fixed, non_controlled)
        if links:
            tip_chain = tip_chain[1:]
        return root_chain, [connection] if links else [], tip_chain

    @memoize
    @profile
    def compose_fk_expression(self, root_link, tip_link):
        fk = w.eye(4)
        root_chain, _, tip_chain = self.compute_split_chain(root_link, tip_link, joints=True, links=False, fixed=True,
                                                            non_controlled=True)
        for joint_name in root_chain:
            fk = w.dot(fk, w.inverse_frame(self.joints[joint_name].parent_T_child))
        for joint_name in tip_chain:
            fk = w.dot(fk, self.joints[joint_name].parent_T_child)
        # FIXME there is some reference fuckup going on, but i don't know where; deepcopy is just a quick fix
        return deepcopy(fk)

    @profile
    def init_fast_fks(self):
        def f(key):
            root, tip = key
            fk = self.compose_fk_expression(root, tip)
            m = w.speed_up(fk, w.free_symbols(fk))
            return m

        self._fks = KeyDefaultDict(f)

    @memoize
    def compute_fk_pose(self, root, tip):
        homo_m = self.compute_fk_np(root, tip)
        p = PoseStamped()
        p.header.frame_id = str(root)
        p.pose = homo_matrix_to_pose(homo_m)
        return p

    @memoize
    def compute_fk_pose_with_collision_offset(self, root, tip):
        try:
            root_T_tip = self.compute_fk_np(root, tip)
            tip_link = self.links[tip]
            root_T_tip = w.dot(root_T_tip, tip_link.collisions[0].link_T_geometry)
            p = PoseStamped()
            p.header.frame_id = str(root)
            p.pose = homo_matrix_to_pose(root_T_tip)
        except Exception as e:
            print(e)
            traceback.print_exc()
            pass
        return p

    def compute_fk_np(self, root, tip):
        return self.get_fk(root, tip)

    @profile
    def compute_all_fks(self):
        tfs = self.compute_all_fks_matrix()
        result = {}
        for i, link in enumerate(self.link_names_with_collisions):
            fk = tfs[i*4:i*4 + 4]
            position = fk[:, 3]
            q = kdl_to_quaternion(np_to_kdl(fk))
            orientation = [q.x, q.y, q.z, q.w]
            result[link] = np.append(position, orientation)
        return result

    @profile
    def compute_all_fks_matrix(self):
        return self._fk_computer.collision_fk_matrix

    @profile
    def init_all_fks(self):
        class ExpressionCompanion(TravelCompanion):
            idx_start: Dict[PrefixName, int]
            fast_collision_fks: CompiledFunction
            fast_all_fks: CompiledFunction

            def __init__(self, world: WorldTree):
                self.world = world
                self.god_map = self.world.god_map
                self.fks = {self.world.root_link_name: w.eye(4)}

            @profile
            def joint_call(self, joint_name: Union[PrefixName, str]) -> bool:
                joint = self.world.joints[joint_name]
                map_T_parent = self.fks[joint.parent_link_name]
                self.fks[joint.child_link_name] = w.dot(map_T_parent, joint.parent_T_child)
                return False

            @profile
            def compile_fks(self):
                all_fks = w.vstack([self.fks[link_name] for link_name in self.world.link_names])
                for link_name in self.world.link_names_with_collisions:
                    link = self.world.links[link_name]
                    self.fks[link_name] = w.dot(self.fks[link_name], link.collisions[0].link_T_geometry)
                collision_fks = w.vstack([self.fks[link_name] for link_name in self.world.link_names_with_collisions if
                                          link_name != self.world.root_link_name])
                self.fast_all_fks = w.speed_up(all_fks, w.free_symbols(all_fks))
                self.fast_collision_fks = w.speed_up(collision_fks, w.free_symbols(collision_fks))
                self.idx_start = {link_name: i * 4 for i, link_name in enumerate(self.world.link_names)}

            @profile
            def recompute(self):
                self.get_fk.memo.clear()
                self.fks = self.fast_all_fks.call2(self.god_map.unsafe_get_values(self.fast_all_fks.str_params))
                self.collision_fk_matrix = self.fast_collision_fks.call2(self.god_map.unsafe_get_values(self.fast_collision_fks.str_params))

            @memoize
            @profile
            def get_fk(self, root, tip):
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
    def get_fk(self, root, tip):
        return self._fk_computer.get_fk(root, tip)

    @memoize
    @profile
    def are_linked(self, link_a, link_b, non_controlled=False, fixed=False):
        """
        Return True if all joints between link_a and link_b are fixed.
        """
        chain1, connection, chain2 = self.compute_split_chain(link_a, link_b, joints=True, links=False, fixed=fixed,
                                                              non_controlled=non_controlled)
        return not chain1 and not connection and not chain2

    def _delete_joint_limits(self):
        for joint_name in self.movable_joints:  # type: OneDofJoint
            joint = self.joints[joint_name]
            joint.delete_limits()
            joint.delete_weights()

    def joint_limit_expr(self, joint_name, order):
        return self.joints[joint_name].get_limit_expressions(order)

    def transform_msg(self, target_frame, msg):
        if isinstance(msg, PoseStamped):
            return self.transform_pose(target_frame, msg)
        elif isinstance(msg, PointStamped):
            return self.transform_point(target_frame, msg)
        elif isinstance(msg, Vector3Stamped):
            return self.transform_vector(target_frame, msg)
        elif isinstance(msg, QuaternionStamped):
            return self.transform_quaternion(target_frame, msg)
        else:
            raise NotImplementedError('World can\'t transform message of type \'{}\''.format(type(msg)))

    def transform_pose(self, target_frame, pose):
        """
        :type target_frame: Union[str, PrefixName]
        :type pose: PoseStamped
        :rtype: PoseStamped
        """
        f_T_p = msg_to_homogeneous_matrix(pose.pose)
        t_T_f = self.compute_fk_np(target_frame, pose.header.frame_id)
        t_T_p = np.dot(t_T_f, f_T_p)
        result = PoseStamped()
        result.header.frame_id = target_frame
        result.pose = np_to_pose(t_T_p)
        return result

    def transform_quaternion(self, target_frame: str, quaternion: QuaternionStamped) -> QuaternionStamped:
        p = PoseStamped()
        p.header = quaternion.header
        p.pose.orientation = quaternion.quaternion
        new_pose = self.transform_pose(target_frame, p)
        new_quaternion = QuaternionStamped()
        new_quaternion.header = new_pose.header
        new_quaternion.quaternion = new_pose.pose.orientation
        return new_quaternion

    def transform_point(self, target_frame, point):
        """
        :type target_frame: Union[str, PrefixName]
        :type point: PointStamped
        :rtype: PointStamped
        """
        f_P_p = msg_to_homogeneous_matrix(point)
        t_T_f = self.compute_fk_np(target_frame, point.header.frame_id)
        t_P_p = np.dot(t_T_f, f_P_p)
        result = PointStamped()
        result.header.frame_id = target_frame
        result.point = Point(*t_P_p[:3])
        return result

    def transform_vector(self, target_frame, vector):
        """
        :type target_frame: Union[str, PrefixName]
        :type vector: Vector3Stamped
        :rtype: Vector3Stamped
        """
        f_V_p = msg_to_homogeneous_matrix(vector)
        t_T_f = self.compute_fk_np(target_frame, vector.header.frame_id)
        t_V_p = np.dot(t_T_f, f_V_p)
        result = Vector3Stamped()
        result.header.frame_id = target_frame
        result.vector = Vector3(*t_V_p[:3])
        return result

    def compute_joint_limits(self, joint_name, order):
        try:
            lower_limit, upper_limit = self.joint_limit_expr(joint_name, order)
        except KeyError:
            # joint has no limits for this derivative
            return None, None
        if not isinstance(lower_limit, numbers.Number) and lower_limit is not None:
            f = w.speed_up(lower_limit, w.free_symbols(lower_limit))
            lower_limit = f.call2(self.god_map.get_values(f.str_params))[0][0]
        if not isinstance(upper_limit, numbers.Number) and upper_limit is not None:
            f = w.speed_up(upper_limit, w.free_symbols(upper_limit))
            upper_limit = f.call2(self.god_map.get_values(f.str_params))[0][0]
        return lower_limit, upper_limit

    @profile
    def possible_collision_combinations(self, group_name: Optional[str] = None) -> Set[Tuple[PrefixName, PrefixName]]:
        if group_name is None:
            links = self.link_names_with_collisions
        else:
            links = self.groups[group_name].link_names_with_collisions
        link_combinations = {self.sort_links(link_a, link_b) for link_a, link_b in combinations(links, 2)}
        for link_name in links:
            direct_children = set()
            for child_joint_name in self.links[link_name].child_joint_names:
                if self.is_joint_controlled(child_joint_name):
                    continue
                child_link_name = self.joints[child_joint_name].child_link_name
                links, joints = self.search_branch(link_name=child_link_name,
                                                   stop_at_joint_when=self.is_joint_controlled,
                                                   stop_at_link_when=None,
                                                   collect_joint_when=None,
                                                   collect_link_when=self.has_link_collisions)

                direct_children.update(links)
            direct_children.add(link_name)
            link_combinations.difference_update(
                self.sort_links(link_a, link_b) for link_a, link_b in combinations(direct_children, 2))
        return link_combinations

    def get_joint_position_limits(self, joint_name):
        """
        :return: minimum position, maximum position as float
        """
        return self.compute_joint_limits(joint_name, 0)

    def get_joint_velocity_limits(self, joint_name):
        return self.compute_joint_limits(joint_name, 1)

    def get_all_joint_position_limits(self):
        return {j: self.get_joint_position_limits(j) for j in self.movable_joints}

    def is_joint_prismatic(self, joint_name):
        return isinstance(self.joints[joint_name], PrismaticJoint)

    def is_joint_fixed(self, joint_name):
        return not isinstance(self.joints[joint_name], MovableJoint)

    def is_joint_movable(self, joint_name):
        return not self.is_joint_fixed(joint_name)

    def is_joint_controlled(self, joint_name):
        return joint_name in self.controlled_joints

    @memoize
    def is_link_controlled(self, link_name):
        try:
            self.get_controlled_parent_joint_of_link(link_name)
            return True
        except KeyError as e:
            return False

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

    def has_link_collisions(self, link_name) -> bool:
        return self.links[link_name].has_collisions()

    def has_link_visuals(self, link_name):
        return self.links[link_name].has_visuals()


class SubWorldTree(WorldTree):
    def __init__(self, name, root_link_name, world):
        """
        :type name: str
        :type root_link_name: PrefixName
        :type world: WorldTree
        """
        self.name = name
        self.root_link_name = root_link_name
        self.world = world

    @property
    def controlled_joints(self):
        return [j for j in self.god_map.unsafe_get_data(identifier.controlled_joints) if j in self.joint_names_as_set]

    @property
    def attachment_joint_name(self):
        return self.world.links[self.root_link_name].parent_joint_name

    @property
    def parent_link_of_root(self):
        return self.world.get_parent_link_of_link(self.world.groups[self.name].root_link_name)

    @property
    def _fk_computer(self):
        return self.world._fk_computer

    def delete_all_but_robot(self):
        raise NotImplementedError('Can\'t hard reset a SubWorldTree.')

    @property
    def base_pose(self):
        return self.world.compute_fk_pose(self.world.root_link_name, self.root_link_name).pose

    @property
    def _fks(self):
        return self.world._fks

    @property
    def state(self):
        """
        :rtype: JointStates
        """
        return JointStates({j: self.world.state[j] for j in self.joints if j in self.world.state})

    @state.setter
    def state(self, value):
        self.world.state = value

    def notify_model_change(self):
        raise NotImplementedError()

    def get_link_short_name_match(self, link_name):
        matches = []
        for link_name2 in self.link_names:
            if link_name == link_name2 or link_name == link_name2.short_name:
                matches.append(link_name2)
        if len(matches) > 1:
            raise ValueError(f'Found multiple link matches \'{matches}\'.')
        if len(matches) == 0:
            raise UnknownLinkException(f'Found no link match for \'{link_name}\'.')
        return matches[0]


    def reset_cache(self):
        try:
            del self.joints
        except:
            pass  # property wasn't called
        try:
            del self.links
        except:
            pass  # property wasn't called
        try:
            del self.link_names
        except:
            pass  # property wasn't called
        try:
            del self.link_names_with_collisions
        except:
            pass  # property wasn't called
        try:
            del self.groups
        except:
            pass  # property wasn't called

    @property
    def god_map(self):
        return self.world.god_map

    @property
    def root_link(self):
        return self.world.links[self.root_link_name]

    @cached_property
    def joints(self):
        def helper(root_link):
            """
            :type root_link: Link
            :rtype: dict
            """
            joints = {j: self.world.joints[j] for j in root_link.child_joint_names}
            for j in root_link.child_joint_names:  # type: Joint
                j = self.world.joints[j]
                child_link = self.world.links[j.child_link_name]
                joints.update(helper(child_link))
            return joints

        return helper(self.root_link)

    @cached_property
    def groups(self):
        return {group_name: group for group_name, group in self.world.groups.items() if
                group.root_link_name in self.links and group.name != self.name}

    @cached_property
    def links(self):
        def helper(root_link):
            """
            :type root_link: Link
            :rtype: list
            """
            links = {root_link.name: root_link}
            for j in root_link.child_joint_names:  # type: Joint
                j = self.world.joints[j]
                child_link = self.world.links[j.child_link_name]
                links.update(helper(child_link))
            return links

        return helper(self.root_link)

    def compute_fk_pose(self, root, tip):
        return self.world.compute_fk_pose(root, tip)

    def compute_fk_pose_with_collision_offset(self, root, tip):
        return self.world.compute_fk_pose_with_collision_offset(root, tip)

    def register_group(self, name, root_link_name):
        raise NotImplementedError()

    def _link_joint_to_links(self, connecting_joint, child_link):
        raise NotImplementedError()

    def add_urdf_joint(self, urdf_joint):
        raise NotImplementedError()

    def delete_branch(self, parent_joint):
        raise NotImplementedError()
