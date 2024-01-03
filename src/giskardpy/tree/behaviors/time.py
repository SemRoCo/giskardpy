from py_trees import Status

from giskardpy import identifier
from giskardpy.tree.behaviors.plugin import GiskardBehavior


class TimePlugin(GiskardBehavior):

    @profile
    def __init__(self, name=None):
        if name is None:
            name = 'increase time'
        super().__init__(name)

    @profile
    def update(self):
        with self.god_map:
            self.god_map.unsafe_set_data(identifier.time, self.god_map.unsafe_get_data(identifier.time) + 1)
        return Status.RUNNING
