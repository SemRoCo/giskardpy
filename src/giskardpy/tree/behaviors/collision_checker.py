from multiprocessing import Lock

from py_trees import Status

from giskardpy.exceptions import SelfCollisionViolatedException
from giskardpy.god_map import god_map
from giskardpy.model.collision_world_syncer import Collisions
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import catch_and_raise_to_blackboard, record_time
from giskardpy.utils.utils import raise_to_blackboard


class CollisionChecker(GiskardBehavior):
    @profile
    def __init__(self, name):
        super().__init__(name)
        self.lock = Lock()

    def initialise(self):
        god_map.collision_scene.add_added_checks()
        super().initialise()

    def are_self_collisions_violated(self, collsions: Collisions):
        for key, self_collisions in collsions.self_collisions.items():
            for self_collision in self_collisions[:-1]:  # the last collision is always some default crap
                if self_collision.link_b_hash == 0:
                    continue  # Fixme figure out why there are sometimes two default collision entries
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
        collisions = god_map.collision_scene.check_collisions()
        self.are_self_collisions_violated(collisions)
        god_map.closest_point = collisions
        return Status.RUNNING
