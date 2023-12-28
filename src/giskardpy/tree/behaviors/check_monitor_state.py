import numpy as np
from py_trees import Status
from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class CheckMonitorState(GiskardBehavior):
    @profile
    def __init__(self, name: str, state_filter: np.ndarray):
        super().__init__(name)
        self.state_filter = state_filter

    @record_time
    @profile
    def update(self):
        if np.all(god_map.monitor_manager.state[self.state_filter]):
            return Status.SUCCESS
        return Status.RUNNING
