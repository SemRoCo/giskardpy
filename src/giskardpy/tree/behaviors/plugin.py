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
        if self.is_closed_loop:
            return self.god_map.unsafe_get_data(identifier.time)
        return self.god_map.unsafe_get_data(identifier.time) * self.god_map.unsafe_get_data(identifier.sample_period)

    @property
    def collision_avoidance_configs(self) -> Dict[str, CollisionAvoidanceGroupThresholds]:
        return self.god_map.unsafe_get_data(identifier.collision_avoidance_configs)

    def get_runtime(self):
        return time() - self.get_blackboard().runtime

    @staticmethod
    def raise_to_blackboard(exception):
        Blackboard().set('exception', exception)

    @staticmethod
    def get_blackboard():
        return Blackboard()

    @staticmethod
    def has_blackboard_exception():
        return has_blackboard_exception()

    @staticmethod
    def get_blackboard_exception():
        return get_blackboard_exception()

    @staticmethod
    def clear_blackboard_exception():
        clear_blackboard_exception()

    def transform_msg(self, target_frame, msg, timeout=1):
        try:
            return self.world.transform_msg(target_frame, msg)
        except KeyError as e:
            return tf.transform_msg(target_frame, msg, timeout=timeout)
