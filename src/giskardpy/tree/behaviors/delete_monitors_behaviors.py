from __future__ import annotations
from typing import TYPE_CHECKING

from py_trees import Status

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
                    # return running because such that the following nodes are not called after the deleted comes
                    # into effect
                    return Status.RUNNING
        return Status.SUCCESS
