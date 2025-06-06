from py_trees import Status

from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class NotifyStateChange(GiskardBehavior):

    @record_time
    @profile
    def update(self):
        god_map.world.notify_state_change()
        return Status.SUCCESS


class NotifyModelChange(GiskardBehavior):

    @record_time
    @profile
    def update(self):
        god_map.world.notify_model_change()
        return Status.SUCCESS
