from __future__ import annotations
from giskardpy import identifier
from giskardpy.god_map import GodMap
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from giskardpy.configs.behavior_tree_config import ControlModes
    from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer
    from giskardpy.tree.garden import TreeManager
    from giskardpy.model.world import WorldTree


class GodMapWorshipper:
    god_map = GodMap()

    @property
    def world(self) -> WorldTree:
        return self.god_map.get_data(identifier.world)

    @property
    def tree_manager(self) -> TreeManager:
        return self.god_map.get_data(identifier.tree_manager)

    @property
    def collision_scene(self) -> CollisionWorldSynchronizer:
        return self.god_map.get_data(identifier.collision_scene)

    @property
    def control_mode(self) -> ControlModes:
        return self.god_map.get_data(identifier.control_mode)
