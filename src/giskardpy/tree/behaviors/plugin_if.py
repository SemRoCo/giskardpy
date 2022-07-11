from py_trees import Status

from giskardpy.tree.behaviors.plugin import GiskardBehavior


class IF(GiskardBehavior):
    def __init__(self, name, identifier):
        super(IF, self).__init__(name)
        self.identifier = identifier

    def update(self):
        if self.get_god_map().get_data(self.identifier):
            return Status.SUCCESS
        return Status.FAILURE


class IF_NOT(GiskardBehavior):
    def __init__(self, name, identifier):
        super(IF_NOT, self).__init__(name)
        self.identifier = identifier

    def update(self):
        if self.get_god_map().get_data(self.identifier):
            return Status.FAILURE
        return Status.SUCCESS


class IfFunction(GiskardBehavior):
    def __init__(self, name, function):
        super(IfFunction, self).__init__(name)
        self.function = function

    def update(self):
        if self.function():
            return Status.SUCCESS
        return Status.FAILURE
