from time import time

from py_trees import Behaviour, Blackboard

from giskardpy import identifier
from giskardpy.configs.data_types import CollisionAvoidanceConfig
from giskardpy.god_map import GodMap
from giskardpy.model.world import WorldTree
from giskardpy.utils.time_collector import TimeCollector
from giskardpy.utils.utils import has_blackboard_exception, get_blackboard_exception, clear_blackboard_exception
import giskardpy.utils.tfwrapper as tf


class GiskardBehavior(Behaviour):
    time_collector: TimeCollector

    def __init__(self, name):
        self.god_map: GodMap = Blackboard().god_map
        self.time_collector = self.god_map.unsafe_get_data(identifier.timer_collector)
        self.world: WorldTree = self.get_god_map().unsafe_get_data(identifier.world)
        super().__init__(name)

    def __str__(self):
        return f'{self.__class__.__name__}'

    @property
    def traj_time_in_sec(self):
        return self.god_map.unsafe_get_data(identifier.time) * self.god_map.unsafe_get_data(identifier.sample_period)

    @property
    def collision_avoidance_config(self) -> CollisionAvoidanceConfig:
        return self.god_map.unsafe_get_data(identifier.collision_avoidance_config)

    def get_god_map(self):
        """
        :rtype: giskardpy.god_map.GodMap
        """
        return self.god_map

    def get_runtime(self):
        return time() - self.get_blackboard().runtime

    @property
    def tree(self):
        """
        :rtype: giskardpy.tree.garden.TreeManager
        """
        return self.god_map.unsafe_get_data(identifier.tree_manager)

    @property
    def collision_scene(self):
        """
        :rtype: giskardpy.model.collision_world_syncer.CollisionWorldSynchronizer
        """
        return self.god_map.unsafe_get_data(identifier.collision_scene)

    @collision_scene.setter
    def collision_scene(self, value):
        self.god_map.unsafe_set_data(identifier.collision_scene, value)

    @property
    def robot(self):
        """
        :rtype: giskardpy.model.world.SubWorldTree
        """
        return self.world.groups[self.god_map.unsafe_get_data(identifier.robot_group_name)]

    def get_world(self):
        """
        :rtype: giskardpy.model.world.WorldTree
        """
        return self.world

    def unsafe_get_world(self):
        """
        :rtype: giskardpy.model.world.WorldTree
        """
        return self.world

    def get_robot(self):
        """
        :rtype: giskardpy.model.world.SubWorldTree
        """
        return self.robot

    def unsafe_get_robot(self):
        """
        :rtype: giskardpy.model.world.SubWorldTree
        """
        return self.robot

    def get_blackboard(self):
        return Blackboard()

    def has_blackboard_exception(self):
        return has_blackboard_exception()

    def get_blackboard_exception(self):
        return get_blackboard_exception()

    def clear_blackboard_exception(self):
        clear_blackboard_exception()

    def transform_msg(self, target_frame, msg, timeout=1):
        try:
            return self.world.transform_msg(target_frame, msg)
        except KeyError as e:
            return tf.transform_msg(target_frame, msg, timeout=timeout)
