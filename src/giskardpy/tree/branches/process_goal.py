from py_trees import Selector

from giskardpy.tree.branches.control_loop import ControlLoop
from giskardpy.tree.decorators import success_is_failure


class ProcessGoal(Selector):
    control_loop_branch: ControlLoop

    def __init__(self, name: str = 'process goal', projection: bool = False):
        super().__init__(name)
        self.control_loop_branch = success_is_failure(ControlLoop)(projection=projection)
        self.add_child(self.control_loop_branch)
