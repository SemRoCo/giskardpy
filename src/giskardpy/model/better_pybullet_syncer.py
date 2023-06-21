from collections import defaultdict
from typing import Dict, Tuple, DefaultDict, List, Set, Optional, Iterable

import betterpybullet as bpb
import numpy as np
from betterpybullet import ClosestPair
from betterpybullet import ContactPoint
from geometry_msgs.msg import PoseStamped, Quaternion, Pose
from sortedcontainers import SortedDict

from giskardpy.model.bpb_wrapper import create_cube_shape, create_object, create_sphere_shape, create_cylinder_shape, \
    load_convex_mesh_shape, MyCollisionObject, create_shape_from_link
from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer, Collision, Collisions
from giskardpy.model.links import BoxGeometry, SphereGeometry, CylinderGeometry, MeshGeometry, Link
from giskardpy.my_types import PrefixName
from giskardpy.utils import logging
from giskardpy.utils.tfwrapper import np_to_pose


class BetterPyBulletSyncer(CollisionWorldSynchronizer):
    def __init__(self, world):
        self.kw = bpb.KineverseWorld()
        self.object_name_to_id: Dict[PrefixName, MyCollisionObject] = {}
        self.query: Optional[DefaultDict[PrefixName, Set[Tuple[MyCollisionObject, float]]]] = None
        super().__init__(world)


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
                                   buffer: float = 0.05) -> DefaultDict[PrefixName, Set[Tuple[MyCollisionObject, float]]]:
        if self.query is None:
            self.query = defaultdict(set)
            for (link_a, link_b), dist in cut_off_distances.items():
                for collision_object_a in self.object_name_to_id[link_a]:
                    for collision_object_b in self.object_name_to_id[link_b]:
                        self.query[collision_object_a].add((collision_object_b, dist+buffer))
        return self.query

    @profile
    def check_collisions(self, cut_off_distances: Dict[Tuple[PrefixName, PrefixName], float],
                         collision_list_sizes: int) -> Collisions:
        """
        :param cut_off_distances: (link_a, link_b) -> max distance. Contacts between objects not in this
                                    dict or further away than the cutoff distance will be ignored.
        :param collision_list_sizes: max number of collisions
        """

        query = self.cut_off_distances_to_query(cut_off_distances, buffer=0.0)
        result: Dict[MyCollisionObject, List[ClosestPair]] = self.kw.get_closest_filtered_POD_batch(query)

        return self.bpb_result_to_collisions(result, collision_list_sizes)

    @profile
    def find_colliding_combinations(self, link_combinations: Iterable[Tuple[PrefixName, PrefixName]],
                                    distance: float,
                                    update_query: bool) -> Set[Tuple[PrefixName, PrefixName]]:
        if update_query:
            self.query = None
            cut_off_distance = {link_combination: distance for link_combination in link_combinations}
        else:
            cut_off_distance = {}
        self.sync()
        collisions = self.check_collisions(cut_off_distance, 15)
        colliding_combinations = {(c.original_link_a, c.original_link_b) for c in collisions.all_collisions
                                  if c.contact_distance <= distance}
        return colliding_combinations

    @profile
    def bpb_result_to_list(self, result: Dict[MyCollisionObject, List[ClosestPair]]) -> List[Collision]:
        result_list = []
        for obj_a, contacts in result.items():
            if not contacts:
                continue
            map_T_a = obj_a.np_transform
            link_a = obj_a.name
            for contact in contacts:  # type: ClosestPair
                map_T_b = contact.obj_b.np_transform
                link_b = contact.obj_b.name
                for p in contact.points:  # type: ContactPoint
                    map_P_a = map_T_a.dot(p.point_a.reshape(4))
                    map_P_b = map_T_b.dot(p.point_b.reshape(4))
                    c = Collision(link_a=link_a,
                                  link_b=link_b,
                                  contact_distance=p.distance,
                                  map_V_n=p.normal_world_b.reshape(4),
                                  map_P_pa=map_P_a,
                                  map_P_pb=map_P_b)
                    result_list.append(c)
        return result_list

    def bpb_result_to_dict(self, result: Dict[MyCollisionObject, List[ClosestPair]]):
        result_dict = {}
        for c in self.bpb_result_to_list(result):
            result_dict[c.link_a, c.link_b] = c
        return SortedDict({k: v for k, v in sorted(result_dict.items())})

    @profile
    def bpb_result_to_collisions(self, result: Dict[MyCollisionObject, List[ClosestPair]],
                                 collision_list_size: int) -> Collisions:
        collisions = Collisions(collision_list_size)
        for c in self.bpb_result_to_list(result):
            collisions.add(c)
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
        if self.has_world_changed():
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
        bpb.batch_set_transforms(self.objects_in_order, self.world.compute_all_collision_fks())

    @profile
    def get_map_T_geometry(self, link_name: PrefixName, link_T_geometry: np.ndarray) -> Pose:
        collision_object = self.object_name_to_id[link_name]
        map_T_link = collision_object.transform.to_np()
        map_T_geometry = np.dot(map_T_link, link_T_geometry)
        return np_to_pose(map_T_geometry)
