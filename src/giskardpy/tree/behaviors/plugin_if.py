from py_trees import Status

from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class IF(GiskardBehavior):
    @profile
    def __init__(self, name, lambda_f):
        super().__init__(name)
        self.lambda_f = lambda_f

    @profile
    def update(self):
        # fixme
        if god_map.get_data(self.lambda_f()):
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
