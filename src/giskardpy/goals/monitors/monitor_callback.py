import abc
from abc import ABC
from typing import List, Dict, Tuple

from giskardpy.god_map import god_map


class MonitorCallback(ABC):
    trigger_monitors: List[str]

    def __init__(self, trigger_monitors: List[str]):
        self.trigger_monitors = trigger_monitors

    @abc.abstractmethod
    def __call__(self):
        pass


class UpdateParentLinkOfGroup(MonitorCallback):
    def __init__(self, trigger_monitors: List[str], group_name: str, new_parent_link: str):
        self.group_name = group_name
        self.new_parent_link = new_parent_link
        super().__init__(trigger_monitors)

    def __call__(self):
        print('asdfasdfasdf')


class ExpressionUpdater(MonitorCallback):

    def __call__(self):
        pass


class CollisionMatrixUpdater(MonitorCallback):
    collision_matrix: Dict[Tuple[str, str], float]

    def __init__(self, trigger_monitors: List[str], new_collision_matrix: Dict[Tuple[str, str], float]):
        super().__init__(trigger_monitors)
        self.collision_matrix = new_collision_matrix

    def __call__(self):
        god_map.collision_scene.set_collision_matrix(self.collision_matrix)
        god_map.collision_scene.reset_cache()
