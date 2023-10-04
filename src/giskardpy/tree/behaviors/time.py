from py_trees import Status

from giskardpy.god_map_interpreter import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior


class TimePlugin(GiskardBehavior):

    @profile
    def __init__(self, name=None):
        if name is None:
            name = 'increase time'
        super().__init__(name)

    @profile
    def update(self):
        god_map.time += 1
        return Status.SUCCESS
