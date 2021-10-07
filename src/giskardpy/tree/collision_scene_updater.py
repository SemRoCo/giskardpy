from py_trees import Status

from giskardpy.tree.plugin import GiskardBehavior


class CollisionSceneUpdater(GiskardBehavior):
    def update(self):
        self.collision_scene.sync()
        return Status.SUCCESS
