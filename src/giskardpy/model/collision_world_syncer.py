import os
import itertools
from collections import defaultdict
from enum import Enum
from copy import deepcopy
from itertools import product, combinations_with_replacement, combinations
from time import time
from typing import List, Dict, Optional, Tuple, Iterable, Set, DefaultDict
from lxml import etree
import hashlib
import numpy as np
from progress.bar import Bar
from sortedcontainers import SortedKeyList

from giskard_msgs.msg import CollisionEntry
from giskardpy import identifier
from giskardpy.configs.data_types import CollisionAvoidanceGroupConfig
from giskardpy.data_types import JointStates
from giskardpy.exceptions import UnknownGroupException
from giskardpy.god_map import GodMap
from giskardpy.model.world import WorldBranch
from giskardpy.model.world import WorldTree
from giskardpy.my_types import my_string, Derivatives, PrefixName
from giskardpy.qp.free_variable import FreeVariable
from giskardpy.utils import logging
from giskardpy.utils.utils import resolve_ros_iris

np.random.seed(1337)


class Collision:
    def __init__(self, link_a, link_b, contact_distance,
                 map_P_pa=None, map_P_pb=None, map_V_n=None,
                 a_P_pa=None, b_P_pb=None):
        self.contact_distance = contact_distance
        self.link_a = link_a
        self.original_link_a = link_a
        self.link_b = link_b
        self.link_b_hash = self.link_b.__hash__()
        self.original_link_b = link_b
        self.is_external = None

        self.map_P_pa = self.__point_to_4d(map_P_pa)
        self.map_P_pb = self.__point_to_4d(map_P_pb)
        self.map_V_n = self.__vector_to_4d(map_V_n)
        self.a_P_pa = self.__point_to_4d(a_P_pa)
        self.b_P_pb = self.__point_to_4d(b_P_pb)

        self.new_a_P_pa = None
        self.new_b_P_pb = None
        self.new_b_V_n = None

    def __str__(self):
        return f'{self.original_link_a}|-|{self.original_link_b}: {self.contact_distance}'

    def __repr__(self):
        return str(self)

    def __point_to_4d(self, point):
        if point is None:
            return point
        point = np.array(point)
        if len(point) == 3:
            return np.append(point, 1)
        return point

    def __vector_to_4d(self, vector):
        if vector is None:
            return vector
        vector = np.array(vector)
        if len(vector) == 3:
            return np.append(vector, 0)
        return vector

    def reverse(self):
        return Collision(link_a=self.original_link_b,
                         link_b=self.original_link_a,
                         map_P_pa=self.map_P_pb,
                         map_P_pb=self.map_P_pa,
                         map_V_n=-self.map_V_n,
                         a_P_pa=self.b_P_pb,
                         b_P_pb=self.a_P_pa,
                         contact_distance=self.contact_distance)


class Collisions:
    all_collisions: Set[Collision]

    @profile
    def __init__(self, collision_list_size):
        self.god_map = GodMap()
        self.collision_scene: CollisionWorldSynchronizer = self.god_map.get_data(identifier.collision_scene)
        self.collision_avoidance_configs: Dict[str, CollisionAvoidanceGroupConfig] = self.god_map.get_data(
            identifier.collision_avoidance_configs)
        self.fixed_joints = self.collision_scene.fixed_joints
        self.world: WorldTree = self.god_map.get_data(identifier.world)
        self.collision_list_size = collision_list_size

        # @profile
        def sort(x):
            return x.contact_distance

        # @profile
        def default_f():
            return SortedKeyList([self._default_collision('', '', '')] * collision_list_size,
                                 key=sort)

        self.default_result = default_f()

        self.self_collisions = defaultdict(default_f)
        self.external_collision = defaultdict(default_f)
        self.external_collision_long_key = defaultdict(lambda: self._default_collision('', '', ''))
        self.all_collisions = set()
        self.number_of_self_collisions = defaultdict(int)
        self.number_of_external_collisions = defaultdict(int)

    def get_robot_from_self_collision(self, collision):
        link_a, link_b = collision.link_a, collision.link_b
        for robot in self.collision_scene.robots:
            if link_a in robot.link_names_as_set and link_b in robot.link_names_as_set:
                return robot

    @profile
    def add(self, collision: Collision):
        robot = self.get_robot_from_self_collision(collision)
        # is_external = collision.link_b not in robot.link_names_with_collisions \
        #               or collision.link_a not in robot.link_names_with_collisions
        collision.is_external = robot is None
        if collision.is_external:
            collision = self.transform_external_collision(collision)
            key = collision.link_a
            self.external_collision[key].add(collision)
            self.number_of_external_collisions[key] = min(self.collision_list_size,
                                                          self.number_of_external_collisions[key] + 1)
            key_long = (collision.original_link_a, None, collision.original_link_b)
            if key_long not in self.external_collision_long_key:
                self.external_collision_long_key[key_long] = collision
            else:
                self.external_collision_long_key[key_long] = min(collision, self.external_collision_long_key[key_long],
                                                                 key=lambda x: x.contact_distance)
        else:
            collision = self.transform_self_collision(collision, robot)
            key = collision.link_a, collision.link_b
            self.self_collisions[key].add(collision)
            self.number_of_self_collisions[key] = min(self.collision_list_size,
                                                      self.number_of_self_collisions[key] + 1)
        self.all_collisions.add(collision)

    @profile
    def transform_self_collision(self, collision: Collision, robot: WorldBranch) -> Collision:
        link_a = collision.original_link_a
        link_b = collision.original_link_b
        new_link_a, new_link_b = self.world.compute_chain_reduced_to_controlled_joints(link_a, link_b,
                                                                                       self.fixed_joints)
        if not self.world.link_order(new_link_a, new_link_b):
            collision = collision.reverse()
            new_link_a, new_link_b = new_link_b, new_link_a
        collision.link_a = new_link_a
        collision.link_b = new_link_b

        new_b_T_r = self.world.compute_fk_np(new_link_b, robot.root_link_name)
        root_T_map = self.world.compute_fk_np(robot.root_link_name, self.world.root_link_name)
        new_b_T_map = np.dot(new_b_T_r, root_T_map)
        collision.new_b_V_n = np.dot(new_b_T_map, collision.map_V_n)

        if collision.map_P_pa is not None:
            new_a_T_r = self.world.compute_fk_np(new_link_a, robot.root_link_name)
            new_a_P_pa = np.dot(np.dot(new_a_T_r, root_T_map), collision.map_P_pa)
            new_b_P_pb = np.dot(new_b_T_map, collision.map_P_pb)
        else:
            new_a_T_a = self.world.compute_fk_np(new_link_a, collision.original_link_a)
            new_a_P_pa = np.dot(new_a_T_a, collision.a_P_pa)
            new_b_T_b = self.world.compute_fk_np(new_link_b, collision.original_link_b)
            new_b_P_pb = np.dot(new_b_T_b, collision.b_P_pb)
        collision.new_a_P_pa = new_a_P_pa
        collision.new_b_P_pb = new_b_P_pb
        return collision

    @profile
    def transform_external_collision(self, collision: Collision) -> Collision:
        try:
            link_name = collision.original_link_a
            joint = self.world.links[link_name].parent_joint_name
            if self.world.is_joint_controlled(joint) and joint not in self.fixed_joints:
                movable_joint = joint

            def stopper(joint_name):
                return self.world.is_joint_controlled(joint_name) and joint_name not in self.fixed_joints

            movable_joint = self.world.search_for_parent_joint(joint, stopper)
            # movable_joint = self.world.get_controlled_parent_joint_of_link(collision.original_link_a)
        except Exception as e:
            pass
        new_a = self.world.joints[movable_joint].child_link_name
        collision.link_a = new_a
        if collision.map_P_pa is not None:
            new_a_T_map = self.world.compute_fk_np(new_a, self.world.root_link_name)
            new_a_P_a = np.dot(new_a_T_map, collision.map_P_pa)
        else:
            new_a_T_a = self.world.compute_fk_np(new_a, collision.original_link_a)
            new_a_P_a = np.dot(new_a_T_a, collision.a_P_pa)

        collision.new_a_P_pa = new_a_P_a
        return collision

    def _default_collision(self, link_a, body_b, link_b):
        c = Collision(link_a=link_a,
                      link_b=link_b,
                      contact_distance=100,
                      map_P_pa=[0, 0, 0, 1],
                      map_P_pb=[0, 0, 0, 1],
                      map_V_n=[0, 0, 1, 0],
                      a_P_pa=[0, 0, 0, 1],
                      b_P_pb=[0, 0, 0, 1])
        c.new_a_P_pa = [0, 0, 0, 1]
        c.new_b_P_pb = [0, 0, 0, 1]
        c.new_b_V_n = [0, 0, 1, 0]
        return c

    @profile
    def get_external_collisions(self, joint_name: str) -> SortedKeyList:
        """
        Collisions are saved as a list for each movable robot joint, sorted by contact distance
        """
        if joint_name in self.external_collision:
            return self.external_collision[joint_name]
        return self.default_result

    @profile
    def get_number_of_external_collisions(self, joint_name):
        return self.number_of_external_collisions[joint_name]

    # @profile
    def get_self_collisions(self, link_a: my_string, link_b: my_string) -> SortedKeyList:
        """
        Make sure that link_a < link_b, the reverse collision is not saved.
        """
        if (link_a, link_b) in self.self_collisions:
            return self.self_collisions[link_a, link_b]
        return self.default_result

    def get_number_of_self_collisions(self, link_a, link_b):
        return self.number_of_self_collisions[link_a, link_b]

    def __contains__(self, item):
        return item in self.self_collisions or item in self.external_collision


class DisableCollisionReason(Enum):
    Unknown = -1
    Never = 1
    Adjacent = 2
    Default = 3
    AlmostAlways = 4


class CollisionWorldSynchronizer:
    black_list: Set[Tuple[PrefixName, PrefixName]]
    self_collision_matrix_paths: Dict[str, str]

    def __init__(self, world):
        self.black_list = set()
        self.self_collision_matrix_paths = {}
        self.world = world  # type: WorldTree
        self.collision_avoidance_configs: Dict[str, CollisionAvoidanceGroupConfig] = self.god_map.get_data(
            identifier.collision_avoidance_configs)
        self.fixed_joints = []
        self.links_to_ignore = set()
        self.ignored_self_collion_pairs = set()
        self.white_list_pairs = set()
        self.black_list = set()
        for robot_name, collision_avoidance_config in self.collision_avoidance_configs.items():
            self.fixed_joints.extend(collision_avoidance_config.fixed_joints_for_self_collision_avoidance)
            self.links_to_ignore.update(set(collision_avoidance_config.ignored_collisions))
            self.ignored_self_collion_pairs.update(collision_avoidance_config.ignored_self_collisions)
            self.white_list_pairs.update(collision_avoidance_config.add_self_collisions)
        self.fixed_joints = tuple(self.fixed_joints)

        self.world_version = -1

    def _sort_white_list(self):
        self.white_list_pairs = set(
            tuple(x) if self.world.link_order(*x) else tuple(reversed(x)) for x in self.white_list_pairs)
        self.ignored_self_collion_pairs = set(
            tuple(x) if self.world.link_order(*x) else tuple(reversed(x)) for x in self.ignored_self_collion_pairs)

    def _sort_black_list(self):
        self.black_list = set(
            tuple(x) if self.world.link_order(*x) else tuple(reversed(x)) for x in self.black_list)

    def has_world_changed(self):
        if self.world_version != self.world.model_version:
            self.world_version = self.world.model_version
            return True
        return False

    def load_self_collision_matrix_in_tmp(self, group_name: str):
        file_name = self._get_path_to_self_collision_matrix(group_name)
        self.load_black_list_from_srdf(file_name, group_name)

    def load_black_list_from_srdf(self, path: str, group_name: str) -> Dict[
        Tuple[PrefixName, PrefixName], DisableCollisionReason]:
        path_to_srdf = resolve_ros_iris(path)
        if not os.path.exists(path_to_srdf):
            raise AttributeError(f'file {path_to_srdf} does not exist')
        srdf = etree.parse(path_to_srdf)
        srdf_root = srdf.getroot()
        black_list = set()
        reasons = {}
        expected_hash = self.world.groups[group_name].to_hash()
        actual_hash = None
        for child in srdf_root:
            if hasattr(child, 'tag') and child.tag == 'hash':
                actual_hash = child.text
            if hasattr(child, 'tag') and child.tag == 'disable_collisions':
                link_a = child.attrib['link1']
                link_b = child.attrib['link2']
                link_a = self.world.search_for_link_name(link_a)
                link_b = self.world.search_for_link_name(link_b)
                reason_id = child.attrib['reason']
                if link_a not in self.world.link_names_with_collisions \
                        or link_b not in self.world.link_names_with_collisions:
                    continue
                try:
                    reason = DisableCollisionReason[reason_id]
                except KeyError as e:
                    reason = DisableCollisionReason.Unknown
                combi = self.world.sort_links(link_a, link_b)
                black_list.add(combi)
                reasons[combi] = reason
        if actual_hash is not None and actual_hash != expected_hash:
            logging.logwarn(f'Self collision matrix \'{path_to_srdf}\' not loaded because it appears to be outdated.')
            return {}
        for link_name in self.world.link_names_with_collisions:
            black_list.add((link_name, link_name))
        self.black_list = black_list
        logging.loginfo(f'Loaded self collision avoidance matrix: {path_to_srdf}')
        self.self_collision_matrix_paths[group_name] = path_to_srdf
        return reasons

    def robot(self, robot_name: str = '') -> WorldBranch:
        for robot in self.robots:
            if robot.name == robot_name:
                return robot
        raise KeyError(f'robot names {robot_name} does not exist')

    @property
    def robots(self) -> List[WorldBranch]:
        return [self.world.groups[robot_name] for robot_name in self.world.groups.keys()
                if self.world.groups[robot_name].actuated]

    @property
    def group_names(self):
        return list(self.world.groups.keys())

    @property
    def robot_names(self):
        return [r.name for r in self.robots]

    @property
    def god_map(self):
        """
        :rtype: giskardpy.god_map.GodMap
        """
        return self.world.god_map

    def add_object(self, link):
        """
        :type link: giskardpy.model.world.Link
        """
        pass

    # def reset_collision_blacklist(self):
    #     self.black_list = set()
    #     self.update_collision_blacklist(white_list_combinations=self.white_list_pairs)

    def remove_black_list_entries(self, part_list: set):
        self.black_list = {x for x in self.black_list if x[0] not in part_list and x[1] not in part_list}

    def update_self_collision_matrix(self, group_name: str):
        pass

    def update_group_blacklist(self,
                               group_name: str,
                               link_combinations: Optional[set] = None,
                               white_list_combinations: Optional[set] = None,
                               distance_threshold_zero: float = 0.05,
                               distance_threshold_rnd: float = 0.0,
                               non_controlled: bool = False,
                               steps: int = 10):
        self._sort_white_list()
        group: WorldBranch = self.world.groups[group_name]
        if link_combinations is None:
            link_combinations = set(combinations_with_replacement(group.link_names_with_collisions, 2))
        # logging.loginfo('calculating self collision matrix')
        joint_state_tmp = deepcopy(self.world.state)
        t = time()
        # find meaningless collisions
        for link_a, link_b in link_combinations:
            link_combination = self.world.sort_links(link_a, link_b)
            if link_combination in self.black_list:
                continue
            try:
                if link_a == link_b \
                        or link_a in self.links_to_ignore \
                        or link_b in self.links_to_ignore \
                        or link_a in self.ignored_self_collion_pairs \
                        or link_b in self.ignored_self_collion_pairs \
                        or (link_a, link_b) in self.ignored_self_collion_pairs \
                        or (link_b, link_a) in self.ignored_self_collion_pairs \
                        or self.world.are_linked(link_a, link_b, do_not_ignore_non_controlled_joints=non_controlled,
                                                 joints_to_be_assumed_fixed=self.fixed_joints) \
                        or (not group.is_link_controlled(link_a) and not group.is_link_controlled(link_b)):
                    self.add_black_list_entry(*link_combination)
            except Exception as e:
                pass

        unknown = link_combinations.difference(self.black_list)
        self.set_joint_state_to_zero()
        for link_a, link_b in self.find_colliding_combinations(unknown, distance_threshold_zero):
            link_combination = self.world.sort_links(link_a, link_b)
            self.add_black_list_entry(*link_combination)
        unknown = unknown.difference(self.black_list)

        # Remove combinations which can never touch
        # by checking combinations which a single joint can influence
        joints = [j for j in group.controlled_joints if j not in self.fixed_joints]
        for joint_name in joints:
            parent_links = group.get_siblings_with_collisions(joint_name)
            if not parent_links:
                continue
            child_links = self.world.get_directly_controlled_child_links_with_collisions(joint_name, self.fixed_joints)
            if self.world.is_joint_continuous(joint_name):
                min_position = -np.pi
                max_position = np.pi
            else:
                min_position, max_position = self.world.get_joint_position_limits(joint_name)

            # joint_name can make these links touch.
            current_combinations = set(product(parent_links, child_links))
            # Filter for combinations which are still unknown
            # and make sure the link_a, link_b order is same as in unknown
            subset_of_unknown = [x for x in unknown if
                                 x in current_combinations or (x[1], x[0]) in current_combinations]
            if not subset_of_unknown:
                continue
            sometimes = set()
            for position in np.linspace(min_position, max_position, steps):
                self.world.state[joint_name].position = position
                self.world.notify_state_change()
                self.sync()
                for link_a, link_b in subset_of_unknown:
                    if self.in_collision(link_a, link_b, distance_threshold_rnd):
                        sometimes.add(self.world.sort_links(link_a, link_b))
            never = set(subset_of_unknown).difference(sometimes)
            unknown = unknown.difference(never)
            self.add_black_list_entries(never)

        logging.logdebug(f'Calculated self collision matrix in {time() - t:.3f}s')
        self.world.state = joint_state_tmp
        self.world.notify_state_change()
        # unknown.update(self.white_list_pairs)
        if white_list_combinations is not None:
            self.black_list.difference_update(white_list_combinations)
        # self.black_list[group_name] = unknown
        # return self.collision_matrices[group_name]

    @profile
    def compute_self_collision_matrix(self,
                                      group_name: str,
                                      link_combinations: Optional[set] = None,
                                      distance_threshold_zero: float = 0.0,
                                      distance_threshold_never: float = 0.0,
                                      distance_threshold_always: float = 0.005,
                                      non_controlled: bool = False,
                                      steps: int = 10) -> Dict[Tuple[PrefixName, PrefixName], DisableCollisionReason]:
        np.random.seed(1337)
        joint_state_tmp = self.world.state
        black_list = []
        white_list = set()
        default_ = set()
        reasons = {}
        group = self.world.groups[group_name]
        # 0. GENERATE ALL POSSIBLE LINK PAIRS
        if link_combinations is None:
            link_combinations = set(combinations_with_replacement(group.link_names_with_collisions, 2))
        # sort links
        for link_a, link_b in list(link_combinations):
            white_list.add(self.world.sort_links(link_a, link_b))
        # 1. FIND CONNECTING LINKS and DISABLE ALL ADJACENT LINK COLLISIONS
        # find meaningless collisions
        adjacent = set()
        for link_a, link_b in list(white_list):
            element = link_a, link_b
            if link_a == link_b \
                    or self.world.are_linked(link_a, link_b, do_not_ignore_non_controlled_joints=non_controlled,
                                             joints_to_be_assumed_fixed=self.fixed_joints) \
                    or (not group.is_link_controlled(link_a) and not group.is_link_controlled(link_b)):
                white_list.remove(element)
                adjacent.add(element)
                reasons[element] = DisableCollisionReason.Adjacent
        # 2. DISABLE "DEFAULT" COLLISIONS
        self.set_default_joint_state(group)
        for link_a, link_b in self.find_colliding_combinations(white_list, distance_threshold_zero, True):
            link_combination = self.world.sort_links(link_a, link_b)
            white_list.remove(link_combination)
            default_.add(link_combination)
            reasons[link_combination] = DisableCollisionReason.Default
        # 3. (almost) ALWAYS IN COLLISION
        always_tries = 200
        almost_always = set()
        counts: DefaultDict[Tuple[PrefixName, PrefixName], int] = defaultdict(int)
        for try_id in range(always_tries):
            self.set_rnd_joint_state(group)
            for link_a, link_b in self.find_colliding_combinations(white_list, distance_threshold_always, True):
                link_combination = self.world.sort_links(link_a, link_b)
                counts[link_combination] += 1
        for link_combination, count in counts.items():
            if count > always_tries * .95:
                white_list.remove(link_combination)
                almost_always.add(link_combination)
                reasons[link_combination] = DisableCollisionReason.AlmostAlways
        # 4. NEVER IN COLLISION
        never_tries = 10000
        sometimes = set()
        update_query = True
        with Bar('never in collision', max=never_tries) as bar:
            for try_id in range(never_tries):
                self.set_rnd_joint_state(group)
                contacts = self.find_colliding_combinations(white_list, distance_threshold_never, update_query)
                update_query = False
                for link_a, link_b in contacts:
                    link_combination = self.world.sort_links(link_a, link_b)
                    white_list.remove(link_combination)
                    sometimes.add(link_combination)
                    update_query = True
                bar.next()
        never_in_contact = white_list
        for combi in never_in_contact:
            reasons[combi] = DisableCollisionReason.Never
        self.world.state = joint_state_tmp
        self.white_list = white_list
        self.black_list = never_in_contact.union(default_).union(almost_always).union(adjacent)
        self.save_black_list(group, adjacent, default_, almost_always, never_in_contact)
        return reasons

    def save_black_list(self,
                        group: WorldBranch,
                        adjacent: Set[Tuple[PrefixName, PrefixName]],
                        by_default: Set[Tuple[PrefixName, PrefixName]],
                        almost_always: Set[Tuple[PrefixName, PrefixName]],
                        never: Set[Tuple[PrefixName, PrefixName]]):
        # Create the root element
        root = etree.Element('robot')
        root.set('name', group.name)

        child = etree.SubElement(root, 'hash')
        child.text = group.to_hash()

        for link_a, link_b in sorted(itertools.chain(adjacent, by_default, almost_always, never)):
            child = etree.SubElement(root, 'disable_collisions')
            child.set('link1', link_a.short_name)
            child.set('link2', link_b.short_name)
            if (link_a, link_b) in adjacent:
                child.set('reason', 'Adjacent')
            elif (link_a, link_b) in by_default:
                child.set('reason', 'Default')
            elif (link_a, link_b) in almost_always:
                child.set('reason', 'AlmostAlways')
            else:
                child.set('reason', 'Never')

        # Create the XML tree
        tree = etree.ElementTree(root)

        file_name = self._get_path_to_self_collision_matrix(group.name)
        logging.loginfo(f'Saved self collision matrix for {group.name} in {file_name}.')
        tree.write(file_name, pretty_print=True, xml_declaration=True, encoding=tree.docinfo.encoding)
        self.self_collision_matrix_paths[group.name] = file_name

    def _get_path_to_self_collision_matrix(self, group_name: str) -> str:
        path_to_tmp = self.god_map.get_data(identifier.tmp_folder)
        return f'{path_to_tmp}{group_name}.srdf'

    def add_black_list_entry(self, link_a, link_b):
        self.black_list.add((link_a, link_b))

    def add_black_list_entries(self, entries):
        self.black_list.update(entries)

    @profile
    def update_collision_blacklist(self,
                                   link_combinations: Optional[set] = None,
                                   white_list_combinations: Optional[set] = None,
                                   distance_threshold_zero: float = 0.05,
                                   distance_threshold_rnd: float = 0.0,
                                   non_controlled: bool = False,
                                   steps: int = 10):
        # for group_name in self.world.minimal_group_names:
        #     self.update_group_blacklist(group_name, link_combinations, white_list_combinations, distance_threshold_zero,
        #                                 distance_threshold_rnd, non_controlled, steps)
        self.blacklist_inter_group_collisions()

    def blacklist_inter_group_collisions(self):
        for group_a_name, group_b_name in combinations(self.world.minimal_group_names, 2):
            if group_a_name in self.robot_names or group_b_name in self.robot_names:
                if group_a_name in self.robot_names:
                    robot_group = self.world.groups[group_a_name]
                    other_group = self.world.groups[group_b_name]
                else:
                    robot_group = self.world.groups[group_b_name]
                    other_group = self.world.groups[group_a_name]
                unmovable_links = robot_group.get_unmovable_links()
                if len(unmovable_links) > 0:
                    for link_a, link_b in product(unmovable_links,
                                                  other_group.link_names_with_collisions):
                        self.add_black_list_entry(*self.world.sort_links(link_a, link_b))
                continue
            group_a: WorldBranch = self.world.groups[group_a_name]
            group_b: WorldBranch = self.world.groups[group_b_name]
            for link_a, link_b in product(group_a.link_names_with_collisions, group_b.link_names_with_collisions):
                self.add_black_list_entry(*self.world.sort_links(link_a, link_b))

    def get_map_T_geometry(self, link_name, collision_id=0):
        return self.world.compute_fk_pose_with_collision_offset(self.world.root_link_name, link_name, collision_id)

    def set_joint_state_to_zero(self):
        for free_variable in self.world.free_variables:
            self.world.state[free_variable].position = 0

    def set_default_joint_state(self, group: WorldBranch):
        for joint_name in group.controlled_joints:
            free_variable: FreeVariable
            for free_variable in group.joints[joint_name].free_variables:
                if free_variable.has_position_limits():
                    lower_limit = free_variable.get_lower_limit(Derivatives.position)
                    upper_limit = free_variable.get_upper_limit(Derivatives.position)
                    self.world.state[free_variable.name].position = (upper_limit + lower_limit) / 2

    @profile
    def set_rnd_joint_state(self, group: WorldBranch):
        for joint_name in group.controlled_joints:
            free_variable: FreeVariable
            for free_variable in group.joints[joint_name].free_variables:
                if free_variable.has_position_limits():
                    lower_limit = free_variable.get_lower_limit(Derivatives.position)
                    upper_limit = free_variable.get_upper_limit(Derivatives.position)
                    rnd_position = (np.random.random() * (upper_limit - lower_limit)) + lower_limit
                else:
                    rnd_position = np.random.random() * np.pi * 2
                self.world.state[joint_name].position = rnd_position

    def find_colliding_combinations(self, link_combinations: Iterable[Tuple[PrefixName, PrefixName]],
                                    distance: float,
                                    update_query: bool) -> Set[Tuple[PrefixName, PrefixName]]:
        raise NotImplementedError('Collision checking is turned off.')

    def check_collisions(self, cut_off_distances: dict, collision_list_size: float = 15) -> Collisions:
        """
        :param cut_off_distances: (robot_link, body_b, link_b) -> cut off distance. Contacts between objects not in this
                                    dict or further away than the cut off distance will be ignored.
        """
        pass

    def in_collision(self, link_a: my_string, link_b: my_string, distance: float) -> bool:
        return False

    def sync(self):
        """
        :type world: giskardpy.model.world.WorldTree
        """
        pass

    def collision_goals_to_collision_matrix(self,
                                            collision_goals: List[CollisionEntry],
                                            min_dist: dict) -> dict:
        """
        :param collision_goals: list of CollisionEntry
        :return: dict mapping (robot_link, body_b, link_b) -> min allowed distance
        """
        self._sort_black_list()
        collision_goals = self.verify_collision_entries(collision_goals)
        min_allowed_distance = {}
        for collision_entry in collision_goals:  # type: CollisionEntry
            if collision_entry.group1 == collision_entry.ALL:
                group1_links = self.world.link_names_with_collisions
            else:
                group1_links = self.world.groups[collision_entry.group1].link_names_with_collisions
            if collision_entry.group2 == collision_entry.ALL:
                group2_links = self.world.link_names_with_collisions
            else:
                group2_links = self.world.groups[collision_entry.group2].link_names_with_collisions
            for link1 in group1_links:
                if link1 in self.links_to_ignore:
                    continue
                for link2 in group2_links:
                    if link2 in self.links_to_ignore:
                        continue
                    key = self.world.sort_links(link1, link2)
                    r_key = (key[1], key[0])
                    if self.is_allow_collision(collision_entry):
                        if key in min_allowed_distance:
                            del min_allowed_distance[key]
                        elif r_key in min_allowed_distance:
                            del min_allowed_distance[r_key]
                    elif self.is_avoid_collision(collision_entry):
                        if key not in self.black_list:
                            if collision_entry.distance == -1:
                                min_allowed_distance[key] = min_dist[key[0]]
                            else:
                                min_allowed_distance[key] = collision_entry.distance
                    else:
                        raise AttributeError(f'Invalid collision entry type: {collision_entry.type}')
        return min_allowed_distance

    def verify_collision_entries(self, collision_goals: List[CollisionEntry]) -> List[CollisionEntry]:
        for collision_entry in collision_goals:
            if collision_entry.group1 != collision_entry.ALL and collision_entry.group1 not in self.world.groups:
                raise UnknownGroupException(f'group1 \'{collision_entry.group1}\' unknown.')
            if collision_entry.group2 != collision_entry.ALL and collision_entry.group2 not in self.world.groups:
                raise UnknownGroupException(f'group2 \'{collision_entry.group2}\' unknown.')

        for i, ce in enumerate(reversed(collision_goals)):
            if self.is_avoid_all_collision(ce):
                # remove everything before the avoid all
                collision_goals = collision_goals[len(collision_goals) - i - 1:]
                break
            if self.is_allow_all_collision(ce):
                # remove everything before the allow all, including the allow all
                collision_goals = collision_goals[len(collision_goals) - i:]
                break
        else:
            # put an avoid all at the front
            ce = CollisionEntry()
            ce.type = CollisionEntry.AVOID_COLLISION
            ce.distance = -1
            collision_goals.insert(0, ce)

        return collision_goals

    def is_avoid_collision(self, collision_entry: CollisionEntry) -> bool:
        return collision_entry.type == CollisionEntry.AVOID_COLLISION

    def is_allow_collision(self, collision_entry: CollisionEntry) -> bool:
        return collision_entry.type == CollisionEntry.ALLOW_COLLISION

    def is_avoid_all_self_collision(self, collision_entry: CollisionEntry) -> bool:
        return self.is_avoid_collision(collision_entry) \
            and collision_entry.group1 == collision_entry.group2 \
            and collision_entry.group1 in self.robot_names

    def is_allow_all_self_collision(self, collision_entry: CollisionEntry) -> bool:
        return self.is_allow_collision(collision_entry) \
            and collision_entry.group1 == collision_entry.group2 \
            and collision_entry.group1 in self.robot_names

    def is_avoid_all_collision(self, collision_entry: CollisionEntry) -> bool:
        """
        :type collision_entry: CollisionEntry
        :return: bool
        """
        return self.is_avoid_collision(collision_entry) \
            and collision_entry.group1 == collision_entry.ALL and collision_entry.group2 == collision_entry.ALL

    def is_allow_all_collision(self, collision_entry: CollisionEntry) -> bool:
        """
        :type collision_entry: CollisionEntry
        :return: bool
        """
        return self.is_allow_collision(collision_entry) \
            and collision_entry.group1 == collision_entry.ALL and collision_entry.group2 == collision_entry.ALL

    def reset_cache(self):
        pass
