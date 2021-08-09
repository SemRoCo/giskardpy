from py_trees import Status

from giskardpy.tree.plugin import GiskardBehavior


class IF(GiskardBehavior):
    def __init__(self, name, identifier):
        super(IF, self).__init__(name)
        self.identifier = identifier

    def update(self):
        if self.get_god_map().get_data(self.identifier):
            return Status.SUCCESS
        return Status.FAILURE
