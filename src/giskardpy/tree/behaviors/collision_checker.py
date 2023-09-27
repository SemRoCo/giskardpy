from multiprocessing import Lock

from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.exceptions import SelfCollisionViolatedException
from giskardpy.god_map_user import GodMap
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
            added_checks = GodMap.god_map.get_data(identifier.added_collision_checks)
            GodMap.god_map.set_data(identifier.added_collision_checks, {})
        except KeyError:
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
            self.collision_matrix = GodMap.god_map.get_data(identifier.collision_matrix)
            self.collision_matrix = self.add_added_checks(self.collision_matrix)
            self.collision_list_size = sum([config.max_num_of_repeller()
                                            for config in self.collision_avoidance_configs.values()])
            GodMap.collision_scene.sync()
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
        GodMap.collision_scene.sync()
        collisions = GodMap.collision_scene.check_collisions(self.collision_matrix, self.collision_list_size)
        self.are_self_collisions_violated(collisions)
        GodMap.god_map.set_data(identifier.closest_point, collisions)
        return Status.RUNNING
