from giskardpy.motion_graph.monitors.monitors import EndMotion
from giskardpy.motion_graph.monitors.payload_monitors import PayloadMonitor
from giskardpy_ros.tree.branches.payload_monitor_sequence import PayloadMonitorSequence
from giskardpy_ros.tree.composites.running_selector import RunningSelector
from giskardpy_ros.tree.decorators import success_is_running


class CheckMonitors(RunningSelector):

    def __init__(self, name: str = 'check monitors'):
        super().__init__(name)

    def add_monitor(self, monitor: PayloadMonitor):
        if isinstance(monitor, EndMotion):
            self.add_child(PayloadMonitorSequence(monitor))
        else:
            self.add_child(success_is_running(PayloadMonitorSequence)(monitor))
