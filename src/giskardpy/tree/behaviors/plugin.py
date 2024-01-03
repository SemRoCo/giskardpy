from time import time
from typing import Dict, Optional

from py_trees import Behaviour, Blackboard

import giskardpy.utils.tfwrapper as tf
from giskardpy import identifier
from giskardpy.configs.collision_avoidance_config import CollisionAvoidanceGroupThresholds
from giskardpy.god_map import GodMap
from giskardpy.god_map_user import GodMapWorshipper
from giskardpy.model.world import WorldTree
from giskardpy.utils.utils import has_blackboard_exception, get_blackboard_exception, clear_blackboard_exception


class GiskardBehavior(Behaviour, GodMapWorshipper):

    def __init__(self, name: Optional[str] = None):
        if name is None:
            name = self.__str__()
        super().__init__(name)

    def __str__(self):
        return f'{self.__class__.__name__}'

    def __copy__(self):
        return type(self)(self.name)

    @property
    def traj_time_in_sec(self):
        return self.god_map.unsafe_get_data(identifier.time) * self.god_map.unsafe_get_data(identifier.sample_period)

    @property
    def collision_avoidance_configs(self) -> Dict[str, CollisionAvoidanceGroupThresholds]:
        return self.god_map.unsafe_get_data(identifier.collision_avoidance_configs)

    def get_god_map(self):
        """
        :rtype: giskardpy.god_map.GodMap
        """
        return self.god_map

    def get_runtime(self):
        return time() - self.get_blackboard().runtime

    def robot(self, robot_name=''):
        """
        :rtype: giskardpy.model.world.WorldBranch
        """
        return self.collision_scene.robot(robot_name=robot_name)

    def robot_names(self):
        return self.collision_scene.robot_names

    def robot_namespaces(self):
        """
        :rtype: list of str
        """
        return self.collision_scene.robot_namespaces

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

    def raise_to_blackboard(self, exception):
        Blackboard().set('exception', exception)

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
