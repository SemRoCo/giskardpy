from py_trees import Sequence

from giskardpy.goals.monitors.monitors import Monitor
from giskardpy.goals.monitors.payload_monitors import PayloadMonitor
from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.check_monitor_state import CheckMonitorState
from giskardpy.tree.behaviors.curcial_monitors_satisfied import CrucialMonitorsSatisfied
from giskardpy.tree.behaviors.delete_behavior import DeleteParent
from giskardpy.tree.behaviors.delete_monitors_behaviors import DeleteMonitor
from giskardpy.tree.behaviors.execute_payload_monitor import ExecutePayloadMonitor
from giskardpy.tree.behaviors.local_minimum import LocalMinimum
from giskardpy.tree.behaviors.max_trajectory_length import MaxTrajectoryLength
from giskardpy.tree.composites.running_selector import RunningSelector


class PayloadMonitorSequence(Sequence):
    monitor: PayloadMonitor

    def __init__(self, monitor: PayloadMonitor):
        super().__init__(monitor.name)
        self.monitor = monitor
        self.add_child(CheckMonitorState(name='check\n'+'\n'.join(x for x in self.monitor.start_monitors),
                                         state_filter=self.monitor.state_filter))
        self.add_child(ExecutePayloadMonitor(self.monitor))
        self.add_child(DeleteMonitor(name=f'delete\nparent', monitor=self))
