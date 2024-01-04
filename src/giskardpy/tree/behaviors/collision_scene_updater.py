from py_trees import Status

from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class CollisionSceneUpdater(GiskardBehavior):
    @record_time
    @profile
    def update(self):
        god_map.collision_scene.sync()
        return Status.SUCCESS
