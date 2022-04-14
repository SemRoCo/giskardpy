from collections import defaultdict
from copy import deepcopy
from multiprocessing import Lock
from time import time

from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.tree.behaviors.plugin import GiskardBehavior


class CollisionChecker(GiskardBehavior):
    @profile
    def __init__(self, name):
        super(CollisionChecker, self).__init__(name)
        self.map_frame = self.get_god_map().get_data(identifier.map_frame)
        self.lock = Lock()
        self.object_js_subs = {}  # JointState subscribers for articulated world objects
        self.object_joint_states = {}  # JointStates messages for articulated world objects

    def _cal_max_param(self, parameter_name):
        external_distances = self.get_god_map().get_data(identifier.external_collision_avoidance)
        self_distances = self.get_god_map().get_data(identifier.self_collision_avoidance)
        default_distance = max(external_distances.default_factory()[parameter_name],
                               self_distances.default_factory()[parameter_name])
        for value in external_distances.values():
            default_distance = max(default_distance, value[parameter_name])
        for value in self_distances.values():
            default_distance = max(default_distance, value[parameter_name])
        return default_distance

    @profile
    def initialise(self):
        t = time()
        self.collision_scene.sync()
        collision_goals = self.get_god_map().get_data(identifier.collision_goal)
        max_distances = self.make_max_distances()
        try:
            added_checks = self.get_god_map().get_data(identifier.added_collision_checks)
            self.god_map.set_data(identifier.added_collision_checks, {})
        except KeyError:
            # no collision checks added
            added_checks = {}
        self.collision_matrix = self.collision_scene.collision_goals_to_collision_matrix(deepcopy(collision_goals),
                                                                                         max_distances,
                                                                                         added_checks)
        self.collision_list_size = self._cal_max_param('number_of_repeller')

        super(CollisionChecker, self).initialise()
        t2 = time() - t
        self.get_blackboard().runtime += t2

    def make_max_distances(self):
        external_distances = self.get_god_map().get_data(identifier.external_collision_avoidance)
        self_distances = self.get_god_map().get_data(identifier.self_collision_avoidance)
        # FIXME check all dict entries
        default_distance = self._cal_max_param('soft_threshold')

        max_distances = defaultdict(lambda: default_distance)
        # override max distances based on external distances dict
        for link_name in self.robot.link_names_with_collisions:
            controlled_parent_joint = self.get_robot().get_controlled_parent_joint_of_link(link_name)
            distance = external_distances[controlled_parent_joint]['soft_threshold']
            for child_link_name in self.get_robot().get_directly_controlled_child_links_with_collisions(
                    controlled_parent_joint):
                max_distances[child_link_name] = distance

        for link_name in self_distances:
            distance = self_distances[link_name]['soft_threshold']
            if link_name in max_distances:
                max_distances[link_name] = max(distance, max_distances[link_name])
            else:
                max_distances[link_name] = distance

        return max_distances

    @profile
    def update(self):
        """
        Computes closest point info for all robot links and safes it to the god map.
        """
        self.collision_scene.sync()
        collisions = self.collision_scene.check_collisions(self.collision_matrix, self.collision_list_size)
        self.god_map.set_data(identifier.closest_point, collisions)
        return Status.RUNNING
