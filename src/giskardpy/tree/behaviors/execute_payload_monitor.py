import numpy as np
from py_trees import Status

from giskardpy.goals.monitors.payload_monitors import PayloadMonitor
from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class ExecutePayloadMonitor(GiskardBehavior):
    monitor: PayloadMonitor

    @profile
    def __init__(self, monitor: PayloadMonitor):
        super().__init__('execute\npayload')
        self.monitor = monitor

    @record_time
    @profile
    def update(self):
        self.monitor()
        if self.monitor.get_state():
            return Status.SUCCESS
        return Status.FAILURE
