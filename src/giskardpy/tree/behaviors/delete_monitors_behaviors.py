from __future__ import annotations
from typing import TYPE_CHECKING

from py_trees import Status

from giskardpy.data_types import TaskState
from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior

if TYPE_CHECKING:
    from giskardpy.tree.branches.payload_monitor_sequence import PayloadMonitorSequence


class DeleteMonitors(GiskardBehavior):
    @profile
    def __init__(self, name: str = 'delete\nmonitors'):
        super().__init__(name)

    def update(self):
        god_map.tree.control_loop_branch.check_monitors.remove_all_children()
        return Status.SUCCESS


class DeleteMonitor(GiskardBehavior):
    @profile
    def __init__(self, parent: PayloadMonitorSequence, name: str = 'delete\nmonitor'):
        super().__init__(name)
        self.parent = parent

    def update(self):
        if self.parent.monitor.get_state():
            for monitor in god_map.tree.control_loop_branch.check_monitors.children:
                if monitor == self.parent or (hasattr(monitor, 'original') and monitor.original == self.parent):
                    god_map.tree.control_loop_branch.check_monitors.remove_child(monitor)
                    monitor_id = self.parent.monitor.id
                    current_life_cycle_state = god_map.monitor_manager.life_cycle_state[monitor_id]
                    if current_life_cycle_state not in [TaskState.succeeded, TaskState.failed]:
                        god_map.monitor_manager.life_cycle_state[monitor_id] = TaskState.succeeded
                    # return running because such that the following nodes are not called after the deleted comes
                    # into effect
                    return Status.RUNNING
        return Status.SUCCESS
