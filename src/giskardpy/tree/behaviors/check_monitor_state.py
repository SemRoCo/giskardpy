import numpy as np
from py_trees import Status

from giskardpy.data_types import TaskState
from giskardpy.god_map import god_map
from giskardpy.monitors.monitors import Monitor
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time, catch_and_raise_to_blackboard


class CheckMonitorState(GiskardBehavior):
    @profile
    def __init__(self, name: str, monitor: Monitor):
        super().__init__(name)
        self.monitor = monitor

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        if god_map.monitor_manager.life_cycle_state[self.monitor.id] == TaskState.running:
            return Status.SUCCESS
        return Status.RUNNING
