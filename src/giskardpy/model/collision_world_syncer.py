import os
import itertools
from collections import defaultdict
from enum import Enum
from copy import deepcopy
from itertools import product, combinations_with_replacement, combinations
from time import time
from typing import List, Dict, Optional, Tuple, Iterable, Set, DefaultDict, Callable, Union

from geometry_msgs.msg import Pose
from lxml import etree
import hashlib
import numpy as np

from giskard_msgs.msg import CollisionEntry
from giskardpy import identifier
from giskardpy.data_types import JointStates
from giskardpy.exceptions import UnknownGroupException, UnknownLinkException
from giskardpy.god_map import GodMap
from giskardpy.god_map_user import GodMapWorshipper
from giskardpy.model.world import WorldBranch
from giskardpy.model.world import WorldTree
from giskardpy.my_types import my_string, Derivatives, PrefixName
from giskardpy.qp.free_variable import FreeVariable
from giskardpy.utils import logging
from giskardpy.utils.utils import resolve_ros_iris

np.random.seed(1337)


class CollisionCheckerLib(Enum):
    none = -1
    bpb = 1


class CollisionAvoidanceThresholds:
    def __init__(self,
                 number_of_repeller: int = 1,
                 soft_threshold: float = 0.05,
                 hard_threshold: float = 0.0,
                 max_velocity: float = 0.2):
        self.number_of_repeller = number_of_repeller
        self.soft_threshold = soft_threshold
        self.hard_threshold = hard_threshold
        self.max_velocity = max_velocity

    @classmethod
    def init_50mm(cls):
        return cls(soft_threshold=0.05, hard_threshold=0.0)

    @classmethod
    def init_100mm(cls):
        return cls(soft_threshold=0.1, hard_threshold=0.0)

    @classmethod
    def init_25mm(cls):
        return cls(soft_threshold=0.025, hard_threshold=0.0)


class CollisionAvoidanceGroupThresholds:
    def __init__(self):
        self.external_collision_avoidance: Dict[PrefixName, CollisionAvoidanceThresholds] = defaultdict(
            CollisionAvoidanceThresholds)
        self.self_collision_avoidance: Dict[PrefixName, CollisionAvoidanceThresholds] = defaultdict(
            CollisionAvoidanceThresholds)

    def max_num_of_repeller(self):
        external_distances = self.external_collision_avoidance
        self_distances = self.self_collision_avoidance
        default_distance = max(external_distances.default_factory().number_of_repeller,
                               self_distances.default_factory().number_of_repeller)
        for value in external_distances.values():
            default_distance = max(default_distance, value.number_of_repeller)
        for value in self_distances.values():
            default_distance = max(default_distance, value.number_of_repeller)
        return default_distance


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


class SortedCollisionResults:
    data: list
    default_result = Collision(link_a='',
                               link_b='',
                               contact_distance=100,
                               map_P_pa=[0, 0, 0, 1],
                               map_P_pb=[0, 0, 0, 1],
                               map_V_n=[0, 0, 1, 0],
                               a_P_pa=[0, 0, 0, 1],
                               b_P_pb=[0, 0, 0, 1])
    default_result.new_a_P_pa = [0, 0, 0, 1]
    default_result.new_b_P_pb = [0, 0, 0, 1]
    default_result.new_b_V_n = [0, 0, 1, 0]

    def __init__(self):
        self.data = []

        def sort(x):
            return x.contact_distance

        self.key = sort

    def add(self, element):
        self.data.append(element)
        self.data = list(sorted(self.data, key=self.key))

    def __getitem__(self, item):
        try:
            return self.data[item]
        except (KeyError, IndexError) as e:
            return SortedCollisionResults.default_result


class Collisions:
    all_collisions: Set[Collision]

    @profile
    def __init__(self, collision_list_size):
        self.god_map = GodMap()
        self.collision_scene: CollisionWorldSynchronizer = self.god_map.get_data(identifier.collision_scene)
        self.fixed_joints = self.collision_scene.fixed_joints
        self.world: WorldTree = self.god_map.get_data(identifier.world)
        self.collision_list_size = collision_list_size

        self.self_collisions = defaultdict(SortedCollisionResults)
        self.external_collision = defaultdict(SortedCollisionResults)
        self.external_collision_long_key = defaultdict(lambda: SortedCollisionResults.default_result)
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
        collision.is_external = robot is None
        if collision.is_external:
            collision = self.transform_external_collision(collision)
            key = collision.link_a
            self.external_collision[key].add(collision)
            self.number_of_external_collisions[key] = min(self.collision_list_size,
                                                          self.number_of_external_collisions[key] + 1)
            key_long = (collision.original_link_a, collision.original_link_b)
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
        link_name = collision.original_link_a
        joint = self.world.links[link_name].parent_joint_name

        def stopper(joint_name):
            return self.world.is_joint_controlled(joint_name) and joint_name not in self.fixed_joints

        try:
            movable_joint = self.world.search_for_parent_joint(joint, stopper)
        except KeyError as e:
            movable_joint = joint
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

    @profile
    def get_external_collisions(self, link_name: str) -> SortedCollisionResults:
        """
        Collisions are saved as a list for each movable robot joint, sorted by contact distance
        """
        if link_name in self.external_collision:
            return self.external_collision[link_name]
        return SortedCollisionResults()

    def get_external_collisions_long_key(self, link_a, link_b):
        """
        Collisions are saved as a list for each movable robot joint, sorted by contact distance
        :type joint_name: str
        :rtype: SortedKeyList
        """
        return self.external_collision_long_key[link_a, link_b]

    @profile
    def get_number_of_external_collisions(self, joint_name):
        return self.number_of_external_collisions[joint_name]

    # @profile
    def get_self_collisions(self, link_a: my_string, link_b: my_string) -> SortedCollisionResults:
        """
        Make sure that link_a < link_b, the reverse collision is not saved.
        """
        if (link_a, link_b) in self.self_collisions:
            return self.self_collisions[link_a, link_b]
        return SortedCollisionResults()

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


class CollisionWorldSynchronizer(GodMapWorshipper):
    self_collision_matrix: Dict[Tuple[PrefixName, PrefixName], DisableCollisionReason]
    self_collision_matrix_paths: Dict[str, str]
    world: WorldTree
    collision_avoidance_configs: Dict[str, CollisionAvoidanceGroupThresholds]
    disabled_links: Set[PrefixName]
    srdf_disable_all_collisions = 'disable_all_collisions'
    srdf_disable_self_collision = 'disable_self_collision'
    srdf_moveit_disable_collisions = 'disable_collisions'
    collision_checker_id = CollisionCheckerLib.none
    _fixed_joints: Tuple[PrefixName]

    def __init__(self):
        self.self_collision_matrix = {}
        self.self_collision_matrix_paths = {}
        self.disabled_links = set()
        self.collision_avoidance_configs = defaultdict(CollisionAvoidanceGroupThresholds)
        self._fixed_joints = tuple()
        self.world_version = -1

    @property
    def fixed_joints(self) -> Tuple[PrefixName]:
        return self._fixed_joints

    def add_fixed_joint(self, joint_name: PrefixName):
        fixed_joint_list = list(self._fixed_joints)
        fixed_joint_list.append(joint_name)
        self._fixed_joints = tuple(fixed_joint_list)

    @classmethod
    def empty(cls):
        self = cls()
        giskard = {
            'collision_avoidance': {
                'collision_checker_id': CollisionCheckerLib.none
            },
        }
        self.god_map.set_data(identifier.giskard, giskard)
        self.god_map.set_data(identifier.tmp_folder, resolve_ros_iris('package://giskardpy/tmp/'))
        self.god_map.set_data(identifier.collision_scene, self)
        return self

    def has_world_changed(self):
        if self.world_version != self.world.model_version:
            self.world_version = self.world.model_version
            return True
        return False

    def load_or_compute_self_collision_matrix_in_tmp(self, group_name: str):
        try:
            file_name = self.get_path_to_self_collision_matrix(group_name)
            recompute = not self.load_self_collision_matrix_from_srdf(file_name, group_name)
        except AttributeError as e:
            logging.logerr('No self collision matrix loaded, computing new one. '
                           'You might want to verify the result using collision_matrix_tool.py.')
            recompute = True
        if recompute:
            self.compute_self_collision_matrix(group_name)

    @property
    def is_collision_checking_enabled(self) -> bool:
        return self.collision_checker_id != CollisionCheckerLib.none

    def load_self_collision_matrix_from_srdf(self, path: str, group_name: str) \
            -> Tuple[Optional[dict], Set[PrefixName]]:
        if not self.is_collision_checking_enabled:
            return {}, set()
        path_to_srdf = resolve_ros_iris(path)
        logging.loginfo(f'loading self collision matrix: {path_to_srdf}')
        if not os.path.exists(path_to_srdf):
            raise AttributeError(f'file {path_to_srdf} does not exist')
        srdf = etree.parse(path_to_srdf)
        srdf_root = srdf.getroot()
        self_collision_matrix = {}
        for child in srdf_root:
            if hasattr(child, 'tag'):
                if child.tag == self.srdf_moveit_disable_collisions \
                        or child.tag == self.srdf_disable_self_collision:
                    link_a = child.attrib['link1']
                    link_b = child.attrib['link2']
                    try:
                        link_a = self.world.search_for_link_name(link_a)
                        link_b = self.world.search_for_link_name(link_b)
                    except UnknownLinkException as e:
                        logging.logwarn(e)
                        continue
                    reason_id = child.attrib['reason']
                    if link_a not in self.world.link_names_with_collisions \
                            or link_b not in self.world.link_names_with_collisions:
                        continue
                    try:
                        reason = DisableCollisionReason[reason_id]
                    except KeyError as e:
                        reason = DisableCollisionReason.Unknown
                    combi = self.world.sort_links(link_a, link_b)
                    self_collision_matrix[combi] = reason
                elif child.tag == self.srdf_disable_all_collisions:
                    try:
                        link_name = self.world.search_for_link_name(child.attrib['link'])
                    except UnknownLinkException as e:
                        logging.logwarn(e)
                        continue
                    self.disabled_links.add(link_name)

        # %% update matrix according to currently controlled joints
        group = self.world.groups[group_name]
        link_combinations = set(combinations_with_replacement(group.link_names_with_collisions, 2))
        link_combinations = {self.world.sort_links(*x) for x in link_combinations}
        _, matrix_updates = self.compute_self_collision_matrix_adjacent(link_combinations, group)
        self_collision_matrix.update(matrix_updates)
        logging.loginfo(f'Loaded self collision matrix: {path_to_srdf}')
        self.self_collision_matrix_paths[group_name] = path_to_srdf
        self.self_collision_matrix = self_collision_matrix
        return self_collision_matrix, self.disabled_links

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

    def add_object(self, link):
        """
        :type link: giskardpy.model.world.Link
        """
        pass

    def remove_links_from_self_collision_matrix(self, link_names: Set[PrefixName]):
        for (link1, link2), reason in list(self.self_collision_matrix.items()):
            if link1 in link_names or link2 in link_names:
                del self.self_collision_matrix[link1, link2]

    def update_self_collision_matrix(self, group_name: str, new_links: Iterable[PrefixName]):
        group = self.world.groups[group_name]
        if group.actuated:
            link_combinations = list(product(group.link_names_with_collisions, new_links))
            self.compute_self_collision_matrix(group_name, link_combinations, save_to_tmp=False)
        else:
            combinations = set(combinations_with_replacement(group.link_names_with_collisions, 2))
            matrix_update = {self.world.sort_links(link1, link2): DisableCollisionReason.Unknown
                             for link1, link2 in combinations}
            self.self_collision_matrix.update(matrix_update)

    @profile
    def compute_self_collision_matrix(self,
                                      group_name: str,
                                      link_combinations: Optional[Iterable] = None,
                                      distance_threshold_zero: float = 0.0,
                                      distance_threshold_always: float = 0.005,
                                      distance_threshold_never_max: float = 0.05,
                                      distance_threshold_never_min: float = -0.02,
                                      distance_threshold_never_range: float = 0.05,
                                      distance_threshold_never_zero: float = 0.0,
                                      number_of_tries_always: int = 200,
                                      almost_percentage: float = 0.95,
                                      number_of_tries_never: int = 10000,
                                      save_to_tmp: bool = True,
                                      progress_callback: Optional[Callable[[int, str], None]] = None) \
            -> Dict[Tuple[PrefixName, PrefixName], DisableCollisionReason]:
        if progress_callback is None:
            progress_callback = lambda value, text: None
        if not self.is_collision_checking_enabled:
            return {}
        np.random.seed(1337)
        remaining_pairs = set()
        self_collision_matrix = {}
        group = self.world.groups[group_name]
        # 0. GENERATE ALL POSSIBLE LINK PAIRS
        if link_combinations is None:
            link_combinations = set(combinations_with_replacement(group.link_names_with_collisions, 2))
        for link_a, link_b in list(link_combinations):
            remaining_pairs.add(self.world.sort_links(link_a, link_b))

        # %%
        remaining_pairs, matrix_updates = self.compute_self_collision_matrix_adjacent(remaining_pairs,
                                                                                      group)
        self_collision_matrix.update(matrix_updates)

        # %%
        remaining_pairs, matrix_updates = self.compute_self_collision_matrix_default(remaining_pairs,
                                                                                     group,
                                                                                     distance_threshold_zero)
        self_collision_matrix.update(matrix_updates)

        # %%
        remaining_pairs, matrix_updates = self.compute_self_collision_matrix_always(
            link_combinations=remaining_pairs,
            group=group,
            distance_threshold_always=distance_threshold_always,
            number_of_tries=number_of_tries_always,
            almost_percentage=almost_percentage)
        self_collision_matrix.update(matrix_updates)

        # %%
        remaining_pairs, matrix_updates = self.compute_self_collision_matrix_never(
            link_combinations=remaining_pairs,
            group=group,
            distance_threshold_never_initial=distance_threshold_never_max,
            distance_threshold_never_min=distance_threshold_never_min,
            distance_threshold_never_range=distance_threshold_never_range,
            distance_threshold_never_zero=distance_threshold_never_zero,
            number_of_tries=number_of_tries_never,
            progress_callback=progress_callback)
        self_collision_matrix.update(matrix_updates)

        if save_to_tmp:
            self.self_collision_matrix = self_collision_matrix
            self.save_self_collision_matrix(group=group,
                                            self_collision_matrix=self_collision_matrix,
                                            disabled_links=set())
        else:
            self.self_collision_matrix.update(self_collision_matrix)
        return self_collision_matrix

    def compute_self_collision_matrix_adjacent(self,
                                               link_combinations: Set[Tuple[PrefixName, PrefixName]],
                                               group: WorldBranch) \
            -> Tuple[Set[Tuple[PrefixName, PrefixName]], Dict[Tuple[PrefixName, PrefixName], DisableCollisionReason]]:
        """
        Find connecting links and disable all adjacent link collisions
        """
        self_collision_matrix = {}
        remaining_pairs = deepcopy(link_combinations)
        for link_a, link_b in list(link_combinations):
            element = link_a, link_b
            if link_a == link_b \
                    or self.world.are_linked(link_a, link_b, do_not_ignore_non_controlled_joints=False,
                                             joints_to_be_assumed_fixed=self.fixed_joints) \
                    or (not group.is_link_controlled(link_a) and not group.is_link_controlled(link_b)):
                remaining_pairs.remove(element)
                self_collision_matrix[element] = DisableCollisionReason.Adjacent
        return remaining_pairs, self_collision_matrix

    def compute_self_collision_matrix_default(self,
                                              link_combinations: Set[Tuple[PrefixName, PrefixName]],
                                              group: WorldBranch,
                                              distance_threshold_zero: float) \
            -> Tuple[Set[Tuple[PrefixName, PrefixName]], Dict[Tuple[PrefixName, PrefixName], DisableCollisionReason]]:
        """
        Disable link pairs that are in collision in default state
        """
        with self.world.reset_joint_state_context():
            self.set_default_joint_state(group)
            self_collision_matrix = {}
            remaining_pairs = deepcopy(link_combinations)
            for link_a, link_b, _ in self.find_colliding_combinations(remaining_pairs, distance_threshold_zero, True):
                link_combination = self.world.sort_links(link_a, link_b)
                remaining_pairs.remove(link_combination)
                self_collision_matrix[link_combination] = DisableCollisionReason.Default
        return remaining_pairs, self_collision_matrix

    def compute_self_collision_matrix_always(self,
                                             link_combinations: Set[Tuple[PrefixName, PrefixName]],
                                             group: WorldBranch,
                                             distance_threshold_always: float,
                                             number_of_tries: int = 200,
                                             almost_percentage: float = 0.95) \
            -> Tuple[Set[Tuple[PrefixName, PrefixName]], Dict[Tuple[PrefixName, PrefixName], DisableCollisionReason]]:
        """
        Disable link pairs that are (almost) always in collision.
        """
        with self.world.reset_joint_state_context():
            self_collision_matrix = {}
            remaining_pairs = deepcopy(link_combinations)
            counts: DefaultDict[Tuple[PrefixName, PrefixName], int] = defaultdict(int)
            for try_id in range(int(number_of_tries)):
                self.set_rnd_joint_state(group)
                for link_a, link_b, _ in self.find_colliding_combinations(remaining_pairs, distance_threshold_always,
                                                                          True):
                    link_combination = self.world.sort_links(link_a, link_b)
                    counts[link_combination] += 1
            for link_combination, count in counts.items():
                if count > number_of_tries * almost_percentage:
                    remaining_pairs.remove(link_combination)
                    self_collision_matrix[link_combination] = DisableCollisionReason.AlmostAlways
        return remaining_pairs, self_collision_matrix

    def compute_self_collision_matrix_never(self,
                                            link_combinations: Set[Tuple[PrefixName, PrefixName]],
                                            group: WorldBranch,
                                            distance_threshold_never_initial: float,
                                            distance_threshold_never_min: float,
                                            distance_threshold_never_range: float,
                                            distance_threshold_never_zero: float,
                                            number_of_tries: int = 10000,
                                            progress_callback: Optional[Callable[[int, str], None]] = None) \
            -> Tuple[Set[Tuple[PrefixName, PrefixName]], Dict[Tuple[PrefixName, PrefixName], DisableCollisionReason]]:
        """
        Disable link pairs that are never in collision.
        """
        with self.world.reset_joint_state_context():
            one_percent = number_of_tries // 100
            self_collision_matrix = {}
            remaining_pairs = deepcopy(link_combinations)
            update_query = True
            distance_ranges: Dict[Tuple[PrefixName, PrefixName], Tuple[float, float]] = {}
            once_without_contact = set()
            for try_id in range(int(number_of_tries)):
                self.set_rnd_joint_state(group)
                contacts = self.find_colliding_combinations(remaining_pairs, distance_threshold_never_initial,
                                                            update_query)
                update_query = False
                contact_keys = set()
                for link_a, link_b, distance in contacts:
                    key = self.world.sort_links(link_a, link_b)
                    contact_keys.add(key)
                    if key in distance_ranges:
                        old_min, old_max = distance_ranges[key]
                        distance_ranges[key] = (min(old_min, distance), max(old_max, distance))
                    else:
                        distance_ranges[key] = (distance, distance)
                    if distance < distance_threshold_never_min:
                        remaining_pairs.remove(key)
                        update_query = True
                        del distance_ranges[key]
                once_without_contact.update(remaining_pairs.difference(contact_keys))
                if try_id % one_percent == 0:
                    progress_callback(try_id // one_percent, 'checking collisions')
            never_in_contact = remaining_pairs
            for key in once_without_contact:
                if key in distance_ranges:
                    old_min, old_max = distance_ranges[key]
                    distance_ranges[key] = (old_min, np.inf)
            for key, (min_, max_) in list(distance_ranges.items()):
                if (max_ - min_) < distance_threshold_never_range or min_ > distance_threshold_never_zero:
                    never_in_contact.add(key)
                    del distance_ranges[key]

            for combi in never_in_contact:
                self_collision_matrix[combi] = DisableCollisionReason.Never
        return remaining_pairs, self_collision_matrix

    def save_self_collision_matrix(self,
                                   group: WorldBranch,
                                   self_collision_matrix: Dict[Tuple[PrefixName, PrefixName], DisableCollisionReason],
                                   disabled_links: Set[PrefixName],
                                   file_name: Optional[str] = None):
        # Create the root element
        root = etree.Element('robot')
        root.set('name', group.name)

        # %% disabled links
        for link_name in sorted(disabled_links):
            child = etree.SubElement(root, 'disable_all_collisions')
            child.set('link', link_name.short_name)

        # %% self collision matrix
        for (link_a, link_b), reason in sorted(self_collision_matrix.items()):
            child = etree.SubElement(root, 'disable_self_collision')
            child.set('link1', link_a.short_name)
            child.set('link2', link_b.short_name)
            child.set('reason', reason.name)

        # Create the XML tree
        tree = etree.ElementTree(root)

        if file_name is None:
            file_name = self.get_path_to_self_collision_matrix(group.name)
        logging.loginfo(f'Saved self collision matrix for {group.name} in {file_name}.')
        tree.write(file_name, pretty_print=True, xml_declaration=True, encoding=tree.docinfo.encoding)
        self.self_collision_matrix_paths[group.name] = file_name

    def get_path_to_self_collision_matrix(self, group_name: str) -> str:
        path_to_tmp = self.god_map.get_data(identifier.tmp_folder)
        return f'{path_to_tmp}{group_name}/{group_name}.srdf'

    @profile
    def update_collision_blacklist(self,
                                   link_combinations: Optional[set] = None,
                                   white_list_combinations: Optional[set] = None,
                                   distance_threshold_zero: float = 0.05,
                                   distance_threshold_rnd: float = 0.0,
                                   non_controlled: bool = False,
                                   steps: int = 10):
        # FIXME
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
                        self.self_collision_matrix[
                            self.world.sort_links(link_a, link_b)] = DisableCollisionReason.Unknown
                continue
            group_a: WorldBranch = self.world.groups[group_a_name]
            group_b: WorldBranch = self.world.groups[group_b_name]
            for link_a, link_b in product(group_a.link_names_with_collisions, group_b.link_names_with_collisions):
                self.self_collision_matrix[self.world.sort_links(link_a, link_b)] = DisableCollisionReason.Unknown

    def get_map_T_geometry(self, link_name: PrefixName, collision_id: int = 0) -> Pose:
        return self.world.compute_fk_pose_with_collision_offset(self.world.root_link_name, link_name, collision_id).pose

    def set_joint_state_to_zero(self):
        for free_variable in self.world.free_variables:
            self.world.state[free_variable].position = 0

    def set_default_joint_state(self, group: WorldBranch):
        for joint_name in group.movable_joint_names:
            free_variable: FreeVariable
            for free_variable in group.joints[joint_name].free_variables:
                if free_variable.has_position_limits():
                    lower_limit = free_variable.get_lower_limit(Derivatives.position)
                    upper_limit = free_variable.get_upper_limit(Derivatives.position)
                    self.world.state[free_variable.name].position = (upper_limit + lower_limit) / 2

    @profile
    def set_rnd_joint_state(self, group: WorldBranch):
        for joint_name in group.movable_joint_names:
            free_variable: FreeVariable
            for free_variable in group.joints[joint_name].free_variables:
                if free_variable.has_position_limits():
                    lower_limit = free_variable.get_lower_limit(Derivatives.position)
                    upper_limit = free_variable.get_upper_limit(Derivatives.position)
                    rnd_position = (np.random.random() * (upper_limit - lower_limit)) + lower_limit
                else:
                    rnd_position = np.random.random() * np.pi * 2
                self.world.state[joint_name].position = rnd_position

    def has_self_collision_matrix(self):
        return len(self.self_collision_matrix_paths) > 0

    def find_colliding_combinations(self, link_combinations: Iterable[Tuple[PrefixName, PrefixName]],
                                    distance: float,
                                    update_query: bool) -> Set[Tuple[PrefixName, PrefixName]]:
        raise NotImplementedError('Collision checking is turned off.')

    def check_collisions(self, cut_off_distances: dict, collision_list_size: float = 15, buffer: float = 0.05) \
            -> Collisions:
        """
        :param cut_off_distances: (link_a, link_b) -> cut off distance. Contacts between objects not in this
                                    dict or further away than the cut off distance will be ignored.
        """
        pass

    def in_collision(self, link_a: my_string, link_b: my_string, distance: float) -> bool:
        return False

    def sync(self):
        pass

    def sync_links_with_world(self):
        for old_link in list(self.disabled_links):
            if old_link not in self.world.link_names:
                self.disabled_links.discard(old_link)
        for key, value in list(self.self_collision_matrix.items()):
            link1, link2 = key
            if link1 not in self.world.link_names or link2 not in self.world.link_names:
                del self.self_collision_matrix[key]

    def collision_goals_to_collision_matrix(self,
                                            collision_goals: List[CollisionEntry],
                                            collision_check_distances: Dict[PrefixName, float]) \
            -> Dict[Tuple[PrefixName, PrefixName], float]:
        """
        :param collision_goals: list of CollisionEntry
        :return: dict mapping (link_a, link_b) -> collision checking distance
        """
        collision_goals = self.verify_collision_entries(collision_goals)
        collision_matrix = {}
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
                if link1 in self.disabled_links:
                    continue
                for link2 in group2_links:
                    if link2 in self.disabled_links:
                        continue
                    black_list_key = robot_link, env_link = self.world.sort_links(link1, link2)
                    if not self.world.is_link_controlled(robot_link) and self.world.is_link_controlled(env_link):
                        robot_link, env_link = env_link, robot_link
                    collision_matrix_key = (robot_link, env_link)
                    if self.is_allow_collision(collision_entry):
                        if collision_matrix_key in collision_matrix:
                            del collision_matrix[collision_matrix_key]
                    elif self.is_avoid_collision(collision_entry):
                        if black_list_key not in self.self_collision_matrix:
                            if collision_entry.distance == -1:
                                collision_matrix[collision_matrix_key] = collision_check_distances[robot_link]
                            else:
                                collision_matrix[collision_matrix_key] = collision_entry.distance
                    else:
                        raise AttributeError(f'Invalid collision entry type: {collision_entry.type}')
        return collision_matrix

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
