from multiprocessing import Lock

from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.exceptions import SelfCollisionViolatedException
from giskardpy.model.collision_world_syncer import Collisions
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import catch_and_raise_to_blackboard, record_time
from giskardpy.utils.utils import raise_to_blackboard


class CollisionChecker(GiskardBehavior):
    @profile
    def __init__(self, name):
        super().__init__(name)
        self.lock = Lock()
        self.object_js_subs = {}  # JointState subscribers for articulated world objects
        self.object_joint_states = {}  # JointStates messages for articulated world objects

    def add_added_checks(self, collision_matrix):
        try:
            added_checks = self.get_god_map().get_data(identifier.added_collision_checks)
            self.god_map.set_data(identifier.added_collision_checks, {})
        except KeyError:
            # no collision checks added
            added_checks = {}
        for (link1, link2), distance in added_checks.items():
            key = self.world.sort_links(link1, link2)
            if key in collision_matrix:
                collision_matrix[key] = max(distance, collision_matrix[key])
            else:
                collision_matrix[key] = distance
        return collision_matrix

    @record_time
    @profile
    def initialise(self):
        try:
            self.collision_matrix = self.god_map.get_data(identifier.collision_matrix)
            self.collision_matrix = self.add_added_checks(self.collision_matrix)
            self.collision_list_size = sum([config.cal_max_param('number_of_repeller')
                                            for config in self.collision_avoidance_configs.values()])
            self.collision_scene.sync()
            super().initialise()
        except Exception as e:
            raise_to_blackboard(e)

    def are_self_collisions_violated(self, collsions: Collisions):
        for key, self_collisions in collsions.self_collisions.items():
            for self_collision in self_collisions[:-1]: # the last collision is always some default crap
                # link_a = self_collision.original_link_a
                # link_b = self_collision.original_link_b
                if self_collision.link_b_hash == 0:
                    continue # Fixme figure out why there are sometimes two default collision entries
                distance = self_collision.contact_distance
                # threshold_a = self.collision_avoidance_config._self_collision_avoidance[link_a].hard_threshold
                # threshold_b = self.collision_avoidance_config._self_collision_avoidance[link_b].hard_threshold
                # threshold = min(threshold_a, threshold_b)
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
        self.collision_scene.sync()
        collisions = self.collision_scene.check_collisions(self.collision_matrix, self.collision_list_size)
        self.are_self_collisions_violated(collisions)
        self.god_map.set_data(identifier.closest_point, collisions)
        return Status.RUNNING
