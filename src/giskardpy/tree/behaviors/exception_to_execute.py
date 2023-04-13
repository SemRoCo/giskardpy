from py_trees import Status

from giskardpy import identifier
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class ExceptionToExecute(GiskardBehavior):
    @record_time
    @profile
    def update(self):
        if self.get_blackboard_exception() is not None:
            self.clear_blackboard_exception()
            if self.god_map.get_data(identifier.skip_failures):
                return Status.FAILURE
            self.god_map.set_data(identifier.execute, False)
            return Status.SUCCESS
        return Status.FAILURE
