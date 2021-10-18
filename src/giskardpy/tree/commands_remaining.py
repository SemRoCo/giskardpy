from py_trees import Status

from giskardpy import identifier
from giskardpy.tree.plugin import GiskardBehavior


class CommandsRemaining(GiskardBehavior):
    def update(self):
        if self.god_map.get_data(identifier.cmd_id) + 1 == self.god_map.get_data(identifier.number_of_move_cmds):
            return Status.SUCCESS
        return Status.FAILURE