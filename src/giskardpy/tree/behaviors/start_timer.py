from time import time

from py_trees import Status

from giskardpy.tree.behaviors.plugin import GiskardBehavior


class StartTimer(GiskardBehavior):
    def update(self):
        if not hasattr(self.get_blackboard(), 'runtime'):
            self.get_blackboard().runtime = time()
        return Status.SUCCESS