from py_trees import Status
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class NotifyStateChange(GiskardBehavior):

    @record_time
    @profile
    def update(self):
        self.world.notify_state_change()

        return Status.SUCCESS
