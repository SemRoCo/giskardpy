from py_trees import Status

from giskardpy import identifier
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class CommandsRemaining(GiskardBehavior):
    @record_time
    @profile
    def update(self):
        if self.god_map.get_data(identifier.cmd_id) + 1 == self.god_map.get_data(identifier.number_of_move_cmds):
            return Status.SUCCESS
        return Status.FAILURE
