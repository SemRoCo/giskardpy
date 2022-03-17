from collections import defaultdict

import betterpybullet as bpb
from geometry_msgs.msg import PoseStamped, Quaternion
from sortedcontainers import SortedDict

from giskardpy.data_types import BiDict, Collisions, Collision
from giskardpy.model.bpb_wrapper import create_cube_shape, create_object, create_sphere_shape, create_cylinder_shape, \
    load_convex_mesh_shape
from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer
from giskardpy.model.links import BoxGeometry, SphereGeometry, CylinderGeometry, MeshGeometry
from giskardpy.utils import logging


class BetterPyBulletSyncer(CollisionWorldSynchronizer):
    def __init__(self, world):
        super(BetterPyBulletSyncer, self).__init__(world)
        self.kw = bpb.KineverseWorld()
        self.object_name_to_id = BiDict()
        self.query = None

    @profile
    def add_object(self, link):
        """
        :type link: giskardpy.model.world.Link
        """
        if not link.has_collisions():
            return False
        geometry = link.collisions[0]
        if isinstance(geometry, BoxGeometry):
            shape = create_cube_shape([geometry.depth, geometry.width, geometry.height])
        elif isinstance(geometry, SphereGeometry):
            shape = create_sphere_shape(geometry.radius * 2)
        elif isinstance(geometry, CylinderGeometry):
            shape = create_cylinder_shape(geometry.radius * 2, geometry.height)
        elif isinstance(geometry, MeshGeometry):
            shape = load_convex_mesh_shape(geometry.file_name, scale=geometry.scale)
        else:
            raise NotImplementedError()
        map_T_o = bpb.Transform()
        map_T_o.origin = bpb.Vector3(0, 0, 0)
        map_T_o.rotation = bpb.Quaternion(0, 0, 0, 1)
        o = create_object(link.name, shape, map_T_o)
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
    def cut_off_distances_to_query(self, cut_off_distances):
        if self.query is None:
            self.query = defaultdict(set)
            for (_, link_a, _, link_b), dist in cut_off_distances.items():
                self.query[self.object_name_to_id[link_a]].add((self.object_name_to_id[link_b], dist))
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

        return self.bpb_result_to_collisions(self.robot_names, result, collision_list_sizes)

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
                try:
                    link_b = contact.obj_b.name
                except Exception as e:
                    result_list = []
                    pass
                for p in contact.points:  # type: ContactPoint
                    map_P_a = map_T_a.dot(p.point_a)
                    map_P_b = map_T_b.dot(p.point_b)
                    body_b = link_b.prefix if link_b in self.robot(link_b.prefix).link_names else ' '
                    c = Collision(link_a=link_a,
                                  body_b=body_b,
                                  link_b=link_b,
                                  contact_distance=p.distance,
                                  map_V_n=p.normal_world_b,
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
    def bpb_result_to_collisions(self, robot_names, result, collision_list_sizes):

        all_collisions = dict()
        for robot_name in robot_names:
            collision_list_size = collision_list_sizes[robot_name]
            collisions = Collisions(self.world, collision_list_size, robot_name)
            all_collisions[robot_name] = collisions

        for c in self.bpb_result_to_list(result):
            all_collisions[c.link_a.prefix].add(c)

        return all_collisions

    # @profile
    # def check_collisions(self, cut_off_distances, collision_list_size=15):
    #     """
    #     :param cut_off_distances: (robot_link, body_b, link_b) -> cut off distance. Contacts between objects not in this
    #                                 dict or further away than the cut off distance will be ignored.
    #     :type cut_off_distances: dict
    #     :param self_collision_d: distances grater than this value will be ignored
    #     :type self_collision_d: float
    #     :type enable_self_collision: bool
    #     :return: (robot_link, body_b, link_b) -> Collision
    #     :rtype: Collisions
    #     """
    #     collisions = Collisions(self.world, collision_list_size)
    #     query = self.cut_off_distances_to_query(cut_off_distances)
    #
    #     result = self.kw.get_closest_filtered_POD_batch(query)
    #     for obj_a, contacts in result.items():
    #         # map_T_a = obj_a.np_transform
    #         link_a = self.object_name_to_id.inverse[obj_a]
    #         for contact in contacts:  # type: ClosestPair
    #             map_T_b = contact.obj_b.np_transform
    #             # b_T_map = contact.obj_b.np_inv_transform
    #             link_b = self.object_name_to_id.inverse[contact.obj_b]
    #             # b_T_map = self.get_fk_np(self.robot.get_link_path(link_b), 'map')
    #             for p in contact.points:  # type: ContactPoint
    #                 # map_P_a = map_T_a.dot(p.point_a)
    #                 map_P_b = map_T_b.dot(p.point_b)
    #                 body_b = RobotName if link_b in self.robot.link_names else ''
    #                 c = Collision(link_a=link_a,
    #                               body_b=body_b,
    #                               link_b=link_b,
    #                               contact_distance=p.distance,
    #                               map_V_n=p.normal_world_b,
    #                               a_P_pa=p.point_a,
    #                               b_P_pb=p.point_b)
    #                 collisions.add(c)
    #     return collisions

    def check_collision(self, link_a, link_b, distance):
        self.sync()
        query = defaultdict(set)
        query[self.object_name_to_id[link_a]].add((self.object_name_to_id[link_b], distance))
        return self.kw.get_closest_filtered_POD_batch(query)

    @profile
    def in_collision(self, link_a, link_b, distance):
        link_id_a = self.object_name_to_id[link_a]
        link_id_b = self.object_name_to_id[link_b]
        result = self.kw.get_distance(link_id_a, link_id_b)
        return len(result) > 0 and result[0].distance < distance

    @profile
    def sync(self):
        """
        :type world: giskardpy.model.world.WorldTree
        """
        if self.has_world_changed():
            self.reset_cache()
            logging.logdebug('hard sync')
            for o in self.kw.collision_objects:
                self.kw.remove_collision_object(o)
            self.object_name_to_id = BiDict()

            for link_name, link in self.world.links.items():
                if link.has_collisions():
                    self.add_object(link)
            self.objects_in_order = [self.object_name_to_id[link.name] for link in self.world.links.values() if
                                     link.has_collisions()]
            self.fks = self.world.compute_all_fks_matrix()
            bpb.batch_set_transforms(self.objects_in_order, self.fks())
            self.init_collision_matrix()
        try:
            bpb.batch_set_transforms(self.objects_in_order, self.fks())
        except Exception as e:
            logging.logerr('needed to recompute all matrices')
            self.fks = self.world.compute_all_fks_matrix()
            bpb.batch_set_transforms(self.objects_in_order, self.fks())

    def get_pose(self, link_name):
        collision_object = self.object_name_to_id[link_name]
        map_T_link = PoseStamped()
        map_T_link.header.frame_id = self.world.root_link_name
        map_T_link.pose.position.x = collision_object.transform.origin.x
        map_T_link.pose.position.y = collision_object.transform.origin.y
        map_T_link.pose.position.z = collision_object.transform.origin.z
        map_T_link.pose.orientation = Quaternion(*collision_object.transform.rotation)
        return map_T_link
