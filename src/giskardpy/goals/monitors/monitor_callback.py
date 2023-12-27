import abc
from abc import ABC
from typing import List, Dict, Tuple

from giskardpy.god_map import god_map
from giskardpy.utils import logging
from giskardpy.utils.logging import loginfo


class MonitorCallback(ABC):
    start_monitors: List[str]
    name: str

    def __init__(self, name: str, start_monitors: List[str]):
        self.start_monitors = start_monitors
        self.name = name

    @abc.abstractmethod
    def __call__(self):
        pass


class UpdateParentLinkOfGroup(MonitorCallback):
    def __init__(self, name: str, start_monitors: List[str], group_name: str, new_parent_link: str):
        self.group_name = group_name
        self.new_parent_link = new_parent_link
        super().__init__(name, start_monitors)

    def __call__(self):
        god_map.world.move_group(self.group_name, self.new_parent_link)


class Print(MonitorCallback):
    def __init__(self, name: str, start_monitors: List[str], message: str):
        self.message = message
        super().__init__(name, start_monitors)

    def __call__(self):
        logging.loginfo(self.message)


class ExpressionUpdater(MonitorCallback):

    def __call__(self):
        pass


class CollisionMatrixUpdater(MonitorCallback):
    collision_matrix: Dict[Tuple[str, str], float]

    def __init__(self, name: str, start_monitors: List[str], new_collision_matrix: Dict[Tuple[str, str], float]):
        super().__init__(name, start_monitors)
        self.collision_matrix = new_collision_matrix

    def __call__(self):
        god_map.collision_scene.set_collision_matrix(self.collision_matrix)
        god_map.collision_scene.reset_cache()
