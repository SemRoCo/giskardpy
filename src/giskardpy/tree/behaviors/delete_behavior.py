import numpy as np
from py_trees import Status, Composite
from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class DeleteParent(GiskardBehavior):
    @profile
    def __init__(self, name, level: int):
        """
        :param name:
        :param level: >= 1, 1 will delete direct parent
        """
        super().__init__(name)
        self.level = level

    @record_time
    @profile
    def update(self):
        child = None
        parent: Composite = self
        for i in range(self.level):
            child = parent
            parent = parent.parent
        parent.remove_child(child)
        return Status.SUCCESS
