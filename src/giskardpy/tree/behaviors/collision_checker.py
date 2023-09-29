from multiprocessing import Lock

from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.exceptions import SelfCollisionViolatedException
from giskardpy.god_map_interpreter import god_map
from giskardpy.model.collision_world_syncer import Collisions
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import catch_and_raise_to_blackboard, record_time
from giskardpy.utils.utils import raise_to_blackboard


class CollisionChecker(GiskardBehavior):
    @profile
    def __init__(self, name):
        super().__init__(name)
        self.lock = Lock()

    def add_added_checks(self, collision_matrix):
        try:
            added_checks = god_map.added_collision_checks
            god_map.added_collision_checks = {}
        except AttributeError:
            # no collision checks added
            added_checks = {}
        for key, distance in added_checks.items():
            if key in collision_matrix:
                collision_matrix[key] = max(distance, collision_matrix[key])
            else:
                collision_matrix[key] = distance
        return collision_matrix

    @record_time
    @profile
    def initialise(self):
        try:
            self.collision_matrix = god_map.collision_matrix
            self.collision_matrix = self.add_added_checks(self.collision_matrix)
            self.collision_list_size = sum([config.max_num_of_repeller()
                                            for config in god_map.collision_scene.collision_avoidance_configs.values()])
            god_map.collision_scene.sync()
            super().initialise()
        except Exception as e:
            raise_to_blackboard(e)

    def are_self_collisions_violated(self, collsions: Collisions):
        for key, self_collisions in collsions.self_collisions.items():
            for self_collision in self_collisions[:-1]: # the last collision is always some default crap
                if self_collision.link_b_hash == 0:
                    continue # Fixme figure out why there are sometimes two default collision entries
                distance = self_collision.contact_distance
                if distance < 0.0:
                    raise SelfCollisionViolatedException(f'{self_collision.original_link_a} and '
                                                         f'{self_collision.original_link_b} violate distance threshold:'
                                                         f'{self_collision.contact_distance} < {0}')

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        """
        Computes closest point info for all robot links and safes it to the god map.
        """
        god_map.collision_scene.sync()
        collisions = god_map.collision_scene.check_collisions(self.collision_matrix, self.collision_list_size)
        self.are_self_collisions_violated(collisions)
        god_map.closest_point = collisions
        return Status.RUNNING
