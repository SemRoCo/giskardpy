from time import time
from typing import Dict, Optional

from py_trees import Behaviour, Blackboard

import giskardpy.utils.tfwrapper as tf
from giskardpy.god_map import god_map
from giskardpy.utils.utils import has_blackboard_exception, get_blackboard_exception, clear_blackboard_exception


class GiskardBehavior(Behaviour):

    def __init__(self, name: Optional[str] = None):
        if name is None:
            name = self.__str__()
        super().__init__(name)

    def __str__(self):
        return f'{self.__class__.__name__}'

    def __copy__(self):
        return type(self)(self.name)

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
            return god_map.world.transform_msg(target_frame, msg)
        except KeyError as e:
            return tf.transform_msg(target_frame, msg, timeout=timeout)
