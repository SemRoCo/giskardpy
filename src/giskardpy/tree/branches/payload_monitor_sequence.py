from py_trees import Sequence
from giskardpy.motion_graph.monitors.monitors import PayloadMonitor
from giskardpy.tree.behaviors.check_monitor_state import CheckMonitorState
from giskardpy.tree.behaviors.delete_monitors_behaviors import DeleteMonitor
from giskardpy.tree.behaviors.execute_payload_monitor import ExecutePayloadMonitor
import giskardpy.casadi_wrapper as cas


class PayloadMonitorSequence(Sequence):
    monitor: PayloadMonitor

    def __init__(self, monitor: PayloadMonitor):
        super().__init__(str(monitor.name))
        self.monitor = monitor
        if not cas.is_false(self.monitor.end_condition):
            self.add_child(DeleteMonitor(name=f'delete\nparent?', parent=self))
        if self.monitor.start_condition:
            self.add_child(CheckMonitorState(monitor=self.monitor))
        self.add_child(ExecutePayloadMonitor(self.monitor))
