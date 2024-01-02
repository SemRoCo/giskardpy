from py_trees import Sequence
from giskardpy.goals.monitors.payload_monitors import PayloadMonitor
from giskardpy.tree.behaviors.check_monitor_state import CheckMonitorState
from giskardpy.tree.behaviors.delete_monitors_behaviors import DeleteMonitor
from giskardpy.tree.behaviors.execute_payload_monitor import ExecutePayloadMonitor


class PayloadMonitorSequence(Sequence):
    monitor: PayloadMonitor

    def __init__(self, monitor: PayloadMonitor):
        super().__init__(monitor.name)
        self.monitor = monitor
        self.add_child(DeleteMonitor(name=f'delete\nparent?', parent=self))
        if self.monitor.start_monitors:
            self.add_child(CheckMonitorState(name='check\n'+'\n'.join(x.name for x in self.monitor.start_monitors),
                                             state_filter=self.monitor.state_filter))
        self.add_child(ExecutePayloadMonitor(self.monitor))