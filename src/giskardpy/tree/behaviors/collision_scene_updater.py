from py_trees import Status

from giskardpy.tree.behaviors.plugin import GiskardBehavior


class CollisionSceneUpdater(GiskardBehavior):
    @profile
    def update(self):
        self.collision_scene.sync()
        return Status.SUCCESS
