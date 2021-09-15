from collections import defaultdict
from itertools import combinations
from time import time

import numpy as np
from tf.transformations import quaternion_matrix, quaternion_from_matrix

import giskardpy.model.pybullet_wrapper as pbw
from giskardpy.data_types import BiDict
from giskardpy.model.world import SubWorldTree
from giskardpy.model.world import WorldTree
from giskardpy.utils import logging
from giskardpy.utils.tfwrapper import np_to_pose
from giskardpy.utils.utils import resolve_ros_iris


class PyBulletSyncer(object):
    def __init__(self, world, gui=False):
        pbw.start_pybullet(gui)
        self.object_name_to_bullet_id = BiDict()
        self.world = world # type: WorldTree
        self.collision_matrices = defaultdict(set)

    @property
    def god_map(self):
        """
        :rtype: giskardpy.god_map.GodMap
        """
        return self.world.god_map

    @profile
    def add_object(self, link):
        """
        :type link: giskardpy.model.world.Link
        """
        pose = self.fks[link.name]
        position = pose[:3]
        orientation = pose[4:]
        self.object_name_to_bullet_id[link.name] = pbw.load_urdf_string_into_bullet(link.as_urdf(),
                                                                                    position=position,
                                                                                    orientation=orientation)

    @profile
    def update_pose(self, link):
        pose = self.fks[link.name]
        position = pose[:3]
        orientation = pose[4:]
        pbw.resetBasePositionAndOrientation(self.object_name_to_bullet_id[link.name], position, orientation)

    def calc_collision_matrix(self, group_name, link_combinations=None, d=0.05, d2=0.0, num_rnd_tries=200):
        """
        :type group_name: str
        :param link_combinations: set with link name tuples
        :type link_combinations: set
        :param d: distance threshold to detect links that are always in collision
        :type d: float
        :param d2: distance threshold to find links that are sometimes in collision
        :type d2: float
        :param num_rnd_tries:
        :type num_rnd_tries: int
        :return: set of link name tuples which are sometimes in collision.
        :rtype: set
        """
        group = self.world.groups[group_name] # type: SubWorldTree
        if link_combinations is None:
            link_combinations = set(combinations(group.link_names_with_collisions, 2))
        # TODO computational expansive because of too many collision checks
        logging.loginfo(u'calculating self collision matrix')
        joint_state_tmp = group.state
        t = time()
        np.random.seed(1337)
        always = set()

        # find meaningless self-collisions
        for link_a, link_b in link_combinations:
            if group.are_linked(link_a, link_b) or \
                    link_a == link_b:
                    # link_a in self.ignored_pairs or \
                    # link_b in self.ignored_pairs or \
                    # (link_a, link_b) in self.ignored_pairs or \
                    # (link_b, link_a) in self.ignored_pairs:
                always.add((link_a, link_b))
        rest = link_combinations.difference(always)
        group.set_joint_state_to_zero()
        always = always.union(self.check_collisions(rest, d))
        rest = rest.difference(always)

        # find meaningful self-collisions
        group.set_min_joint_state()
        sometimes = self.check_collisions(rest, d2)
        rest = rest.difference(sometimes)
        group.set_max_joint_state()
        sometimes2 = self.check_collisions(rest, d2)
        rest = rest.difference(sometimes2)
        sometimes = sometimes.union(sometimes2)
        for i in range(num_rnd_tries):
            group.set_rnd_joint_state()
            sometimes2 = self.check_collisions(rest, d2)
            if len(sometimes2) > 0:
                rest = rest.difference(sometimes2)
                sometimes = sometimes.union(sometimes2)
        # sometimes = sometimes.union(self.added_pairs)
        logging.loginfo(u'calculated self collision matrix in {:.3f}s'.format(time() - t))
        group.state = joint_state_tmp

        self.collision_matrices[group_name] = sometimes
        return self.collision_matrices[group_name]

    def init_collision_matrix(self, group_name):
        self.sync()
        added_links = set(combinations(self.world.groups[group_name].link_names_with_collisions, 2))
        self.update_collision_matrix(group_name=group_name,
                                     added_links=added_links)

    def update_collision_matrix(self, group_name, added_links=None, removed_links=None):
        # if not self.load_self_collision_matrix(self.path_to_data_folder):
        if added_links is None:
            added_links = set()
        if removed_links is None:
            removed_links = set()
        # collision_matrix = {x for x in self.collision_matrices[group_name] if x[0] not in removed_links and
        #                                x[1] not in removed_links}
        collision_matrix = self.calc_collision_matrix(group_name, added_links)
        self.collision_matrices[group_name] = collision_matrix
        # self.safe_self_collision_matrix(self.path_to_data_folder)

    def check_collisions(self, link_combinations, distance):
        in_collision = set()
        self.sync_state()
        for link_a, link_b in link_combinations:
            if self.in_collision(link_a, link_b, distance):
                in_collision.add((link_a, link_b))
        return in_collision

    def in_collision(self, link_a, link_b, distance):
        link_id_a = self.object_name_to_bullet_id[link_a]
        link_id_b = self.object_name_to_bullet_id[link_b]
        return len(pbw.getClosestPoints(link_id_a, link_id_b, distance)) > 0

    @profile
    def sync_state(self):
        """
        :type world: giskardpy.model.world.WorldTree
        """
        pbw.deactivate_rendering()
        self.fks = self.world.compute_all_fks()
        for link_name, link in self.world.links.items():
            if link.has_collisions():
                self.update_pose(link)
        pbw.activate_rendering()

    def sync(self):
        """
        :type world: giskardpy.model.world.WorldTree
        """
        pbw.deactivate_rendering()
        self.object_name_to_bullet_id = BiDict()
        self.world.soft_reset()
        pbw.clear_pybullet()
        self.fks = self.world.compute_all_fks()
        for link_name, link in self.world.links.items():
            if link.has_collisions():
                self.add_object(link)
        pbw.activate_rendering()

    # def __add_ground_plane(self):
    #     """
    #     Adds a ground plane to the Bullet World.
    #     """
    #     path = resolve_ros_iris(u'package://giskardpy/urdfs/ground_plane.urdf')
    #     plane = WorldObject.from_urdf_file(path)
    #     plane.set_name(self.ground_plane_name)
    #     self.add_object(plane)
