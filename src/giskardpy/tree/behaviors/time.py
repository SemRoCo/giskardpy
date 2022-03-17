from py_trees import Status

from giskardpy import identifier
from giskardpy.tree.behaviors.plugin import GiskardBehavior


class TimePlugin(GiskardBehavior):
    @profile
    def update(self):
        with self.god_map:
            self.get_god_map().unsafe_set_data(identifier.time, self.get_god_map().unsafe_get_data(identifier.time) + 1)
        return Status.RUNNING
