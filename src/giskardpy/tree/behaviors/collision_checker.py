from collections import defaultdict
from copy import deepcopy
from multiprocessing import Lock
from time import time

from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.data_types import PrefixName
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
        try:
            default_distance = max(external_distances.default_factory(parameter_name.prefix)[parameter_name],
                                   self_distances.default_factory(parameter_name.prefix)[parameter_name])
        except KeyError:
            pass
        for key, value in external_distances.items():
            if key.prefix == parameter_name.prefix:
                default_distance = max(default_distance, value[parameter_name])
        for key, value in self_distances.items():
            if key.prefix == parameter_name.prefix:
                default_distance = max(default_distance, value[parameter_name])
        return default_distance

    @profile
    def initialise(self):
        try:
            self.collision_matrix = self.god_map.get_data(identifier.collision_matrix)
            self.collision_list_size = sum([self._cal_max_param(PrefixName('number_of_repeller', r_n))
                                             for r_n in self.robot_names()])
            self.collision_scene.sync()
            super(CollisionChecker, self).initialise()
        except Exception as e:
            self.raise_to_blackboard(e)


    @profile
    def update(self):
        """
        Computes closest point info for all robot links and safes it to the god map.
        """
        self.collision_scene.sync()
        collisions = self.collision_scene.check_collisions(self.collision_matrix, self.collision_list_size)
        self.god_map.set_data(identifier.closest_point, collisions)
        return Status.RUNNING
