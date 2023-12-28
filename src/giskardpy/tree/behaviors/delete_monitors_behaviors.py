from __future__ import annotations
from typing import TYPE_CHECKING

import rospy
from py_trees import Status, Sequence
from visualization_msgs.msg import MarkerArray, Marker

from giskardpy.debug_expression_manager import DebugExpressionManager
from giskardpy.goals.monitors.monitor_manager import MonitorManager
from giskardpy.goals.motion_goal_manager import MotionGoalManager
from giskardpy.god_map import god_map
from giskardpy.model.collision_world_syncer import Collisions
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time, catch_and_raise_to_blackboard

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
