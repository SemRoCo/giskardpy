from collections import defaultdict

import betterpybullet as bpb
from betterpybullet import ClosestPair
from betterpybullet import ContactPoint
from geometry_msgs.msg import PoseStamped, Quaternion
from sortedcontainers import SortedDict

from giskardpy.model.bpb_wrapper import create_cube_shape, create_object, create_sphere_shape, create_cylinder_shape, \
    load_convex_mesh_shape
from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer, Collision, Collisions
from giskardpy.model.links import BoxGeometry, SphereGeometry, CylinderGeometry, MeshGeometry, Link
from giskardpy.utils import logging


class BetterPyBulletSyncer(CollisionWorldSynchronizer):
    def __init__(self, world):
        self.kw = bpb.KineverseWorld()
        self.object_name_to_id = defaultdict(list)
        self.query = None
        super().__init__(world)


    @profile
    def add_object(self, link: Link):
        if not link.has_collisions():
            return False
        for collision_id, geometry in enumerate(link.collisions):
            if isinstance(geometry, BoxGeometry):
                shape = create_cube_shape([geometry.depth, geometry.width, geometry.height])
            elif isinstance(geometry, SphereGeometry):
                shape = create_sphere_shape(geometry.radius * 2)
            elif isinstance(geometry, CylinderGeometry):
                shape = create_cylinder_shape(geometry.radius * 2, geometry.height)
            elif isinstance(geometry, MeshGeometry):
                shape = load_convex_mesh_shape(geometry.file_name, scale=geometry.scale)
                geometry.file_name = 'file://' + shape.file_path
            else:
                raise NotImplementedError()
            map_T_o = bpb.Transform()
            map_T_o.origin = bpb.Vector3(0, 0, 0)
            map_T_o.rotation = bpb.Quaternion(0, 0, 0, 1)
            o = create_object(link.name, shape, map_T_o, collision_id)
            self.kw.add_collision_object(o)
            self.object_name_to_id[link.name].append(o)

    def reset_cache(self):
        self.query = None
        for method_name in dir(self):
            try:
                getattr(self, method_name).memo.clear()
            except:
                pass

    @profile
    def cut_off_distances_to_query(self, cut_off_distances, buffer=0.05):
        if self.query is None:
            self.query = defaultdict(set)
            for (link_a, link_b), dist in cut_off_distances.items():
                for collision_object_a in self.object_name_to_id[link_a]:
                    for collision_object_b in self.object_name_to_id[link_b]:
                        self.query[collision_object_a].add((collision_object_b, dist+buffer))
        return self.query

    @profile
    def check_collisions(self, cut_off_distances, collision_list_sizes):
        """
        :param cut_off_distances: (robot_link, body_b, link_b) -> cut off distance. Contacts between objects not in this
                                    dict or further away than the cut off distance will be ignored.
        :type cut_off_distances: dict
        :param self_collision_d: distances grater than this value will be ignored
        :type self_collision_d: float
        :type enable_self_collision: bool
        :return: (robot_link, body_b, link_b) -> Collision
        :rtype: Collisions
        """

        query = self.cut_off_distances_to_query(cut_off_distances)
        result = self.kw.get_closest_filtered_POD_batch(query)

        return self.bpb_result_to_collisions(result, collision_list_sizes)

    @profile
    def bpb_result_to_list(self, result):
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

    def bpb_result_to_dict(self, result):
        result_dict = {}
        for c in self.bpb_result_to_list(result):
            result_dict[c.link_a, c.link_b] = c
        return SortedDict({k: v for k, v in sorted(result_dict.items())})

    @profile
    def bpb_result_to_collisions(self, result, collision_list_size):
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
            self.object_name_to_id = defaultdict(list)

            for link_name in self.world.link_names_with_collisions:
                link = self.world.links[link_name]
                self.add_object(link)
            self.objects_in_order = [x for link_name in self.world._fk_computer.collision_link_order for x in self.object_name_to_id[link_name]]
            # self.objects_in_order = [self.object_name_to_id[link_name] for link_name in self.world.link_names_with_collisions]
            bpb.batch_set_transforms(self.objects_in_order, self.world.compute_all_fks_matrix())
            # self.update_collision_blacklist()
        bpb.batch_set_transforms(self.objects_in_order, self.world.compute_all_fks_matrix())

    @profile
    def get_pose(self, link_name, collision_id=0):
        collision_object = self.object_name_to_id[link_name][collision_id]
        map_T_link = PoseStamped()
        map_T_link.header.frame_id = self.world.root_link_name
        map_T_link.pose.position.x = collision_object.transform.origin.x
        map_T_link.pose.position.y = collision_object.transform.origin.y
        map_T_link.pose.position.z = collision_object.transform.origin.z
        map_T_link.pose.orientation = Quaternion(*collision_object.transform.rotation)
        return map_T_link
