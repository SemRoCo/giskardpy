from threading import Thread

from py_trees import Status

from giskardpy.monitors.payload_monitors import PayloadMonitor
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time, catch_and_raise_to_blackboard


class ExecutePayloadMonitor(GiskardBehavior):
    monitor: PayloadMonitor
    started: bool

    @profile
    def __init__(self, monitor: PayloadMonitor):
        super().__init__('execute\npayload')
        self.monitor = monitor
        self.started = False

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        if self.monitor.run_call_in_thread:
            if not self.started:
                self.payload_thread = Thread(target=self.monitor, name=self.monitor.name)
                self.payload_thread.start()
                self.started = True
            if self.payload_thread.is_alive():
                return Status.RUNNING
        else:
            self.monitor()
        if self.monitor.get_state():
            return Status.SUCCESS
        return Status.RUNNING
