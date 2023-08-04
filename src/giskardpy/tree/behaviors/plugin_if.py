from py_trees import Status

from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class IF(GiskardBehavior):
    @profile
    def __init__(self, name, identifier):
        super().__init__(name)
        self.identifier = identifier

    @profile
    def update(self):
        if self.god_map.get_data(self.identifier):
            return Status.SUCCESS
        return Status.FAILURE


class IfFunction(GiskardBehavior):
    @profile
    def __init__(self, name, function):
        super().__init__(name)
        self.function = function

    @record_time
    @profile
    def update(self):
        if self.function():
            return Status.SUCCESS
        return Status.FAILURE
