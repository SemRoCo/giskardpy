from time import time

from py_trees import Status

from giskardpy.tree.behaviors.plugin import GiskardBehavior


class StartTimer(GiskardBehavior):
    def __init__(self, name):
        super(StartTimer, self).__init__(name)

    def update(self):
        if not hasattr(self.get_blackboard(), 'runtime'):
            self.get_blackboard().runtime = time()
        return Status.SUCCESS