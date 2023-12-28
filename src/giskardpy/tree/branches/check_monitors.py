from giskardpy.goals.monitors.payload_monitors import PayloadMonitor, EndMotion
from giskardpy.tree.behaviors.max_trajectory_length import MaxTrajectoryLength
from giskardpy.tree.branches.payload_monitor_sequence import PayloadMonitorSequence
from giskardpy.tree.composites.running_selector import RunningSelector
from giskardpy.tree.decorators import success_is_running


class CheckMonitors(RunningSelector):

    def __init__(self, name: str = 'check monitors'):
        super().__init__(name)
        # self.add_child(CrucialMonitorsSatisfied())
        # self.add_child(LoopDetector('loop detector'))
        # self.add_child(LocalMinimum())
        self.add_child(MaxTrajectoryLength())
        # self.add_child(GoalDone('goal done check'))

    def add_monitor(self, monitor: PayloadMonitor):
        if isinstance(monitor, EndMotion):
            self.add_child(PayloadMonitorSequence(monitor))
        else:
            self.add_child(success_is_running(PayloadMonitorSequence)(monitor))
