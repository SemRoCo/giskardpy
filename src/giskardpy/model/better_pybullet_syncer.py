from collections import defaultdict
from typing import Dict, Tuple, DefaultDict, List, Set, Optional, Iterable

import betterpybullet as bpb
import numpy as np
from betterpybullet import ClosestPair
from betterpybullet import ContactPoint
from geometry_msgs.msg import PoseStamped, Quaternion, Pose
from sortedcontainers import SortedDict

from giskardpy import identifier
from giskardpy.configs.collision_avoidance_config import CollisionCheckerLib
from giskardpy.model.bpb_wrapper import create_cube_shape, create_object, create_sphere_shape, create_cylinder_shape, \
    load_convex_mesh_shape, create_shape_from_link, to_giskard_collision
from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer, Collision, Collisions
from giskardpy.model.links import BoxGeometry, SphereGeometry, CylinderGeometry, MeshGeometry, Link
from giskardpy.my_types import PrefixName
from giskardpy.utils import logging
from giskardpy.utils.tfwrapper import np_to_pose


class BetterPyBulletSyncer(CollisionWorldSynchronizer):
    collision_checker_id = CollisionCheckerLib.bpb

    def __init__(self,):
        self.kw = bpb.KineverseWorld()
        self.object_name_to_id: Dict[PrefixName, bpb.CollisionObject] = {}
        self.query: Optional[DefaultDict[PrefixName, Set[Tuple[bpb.CollisionObject, float]]]] = None
        super().__init__()

    @classmethod
    def empty(cls):
        self = super().empty()
        self.god_map.set_data(identifier.collision_checker, CollisionCheckerLib.bpb)
        return self

    @profile
    def add_object(self, link: Link):
        if not link.has_collisions():
            return
        o = create_shape_from_link(link)
        self.kw.add_collision_object(o)
        self.object_name_to_id[link.name] = o

    def reset_cache(self):
        self.query = None
        for method_name in dir(self):
            try:
                getattr(self, method_name).memo.clear()
            except:
                pass

    @profile
    def cut_off_distances_to_query(self, cut_off_distances: Dict[Tuple[PrefixName, PrefixName], float],
                                   buffer: float = 0.05) -> DefaultDict[PrefixName, Set[Tuple[bpb.CollisionObject, float]]]:
        if self.query is None:
            self.query = {(self.object_name_to_id[a], self.object_name_to_id[b]): v + buffer for (a, b), v in cut_off_distances.items()}
        return self.query

    @profile
    def check_collisions(self, cut_off_distances: Dict[Tuple[PrefixName, PrefixName], float],
                         collision_list_sizes: int, buffer: float = 0.05) -> Collisions:
        """
        :param cut_off_distances: (link_a, link_b) -> max distance. Contacts between objects not in this
                                    dict or further away than the cutoff distance will be ignored.
        :param collision_list_sizes: max number of collisions
        """

        query = self.cut_off_distances_to_query(cut_off_distances, buffer=buffer)
        result: List[bpb.Collision] = self.kw.get_closest_filtered_map_batch(query)
        return self.bpb_result_to_collisions(result, collision_list_sizes)

    @profile
    def find_colliding_combinations(self, link_combinations: Iterable[Tuple[PrefixName, PrefixName]],
                                    distance: float,
                                    update_query: bool) -> Set[Tuple[PrefixName, PrefixName, float]]:
        if update_query:
            self.query = None
            cut_off_distance = {link_combination: distance for link_combination in link_combinations}
        else:
            cut_off_distance = {}
        self.sync()
        collisions = self.check_collisions(cut_off_distance, 15, buffer=0.0)
        colliding_combinations = {(c.original_link_a, c.original_link_b, c.contact_distance) for c in collisions.all_collisions
                                  if c.contact_distance <= distance}
        return colliding_combinations

    @profile
    def bpb_result_to_collisions(self, result: List[bpb.Collision],
                                 collision_list_size: int) -> Collisions:
        collisions = Collisions(collision_list_size)

        for collision in result:
            giskard_collision = to_giskard_collision(collision)
            collisions.add(giskard_collision)
        return collisions

    def check_collision(self, link_a, link_b, distance):
        self.sync()
        query = defaultdict(set)
        query[self.object_name_to_id[link_a]].add((self.object_name_to_id[link_b], distance))
        return self.kw.get_closest_filtered_POD_batch(query)

    @profile
    def in_collision(self, link_a, link_b, distance):
        result = False
        for link_id_a in self.object_name_to_id[link_a]:
            for link_id_b in self.object_name_to_id[link_b]:
                query_result = self.kw.get_distance(link_id_a, link_id_b)
                result |= len(query_result) > 0 and query_result[0].distance < distance
        return result

    @profile
    def sync(self):
        super().sync()
        if self.has_world_changed():
            self.sync_links_with_world()
            self.reset_cache()
            logging.logdebug('hard sync')
            for o in self.kw.collision_objects:
                self.kw.remove_collision_object(o)
            self.object_name_to_id = {}
            self.objects_in_order = []

            for link_name in sorted(self.world.link_names_with_collisions):
                link = self.world.links[link_name]
                self.add_object(link)
                self.objects_in_order.append(self.object_name_to_id[link_name])
            bpb.batch_set_transforms(self.objects_in_order, self.world.compute_all_collision_fks())
        else:
            bpb.batch_set_transforms(self.objects_in_order, self.world.compute_all_collision_fks())

    @profile
    def get_map_T_geometry(self, link_name: PrefixName, collision_id: int = 0) -> Pose:
        collision_object = self.object_name_to_id[link_name]
        return collision_object.compound_transform(collision_id)
