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


class DeleteMonitors(GiskardBehavior):
    @profile
    def __init__(self, name: str = 'delete\nmonitors'):
        super().__init__(name)

    def update(self):
        god_map.tree.control_loop_branch.check_monitors.remove_all_children()
        return Status.SUCCESS


class DeleteMonitor(GiskardBehavior):
    @profile
    def __init__(self, monitor: Sequence, name: str = 'delete\nmonitor'):
        super().__init__(name)
        self.monitor = monitor

    def update(self):
        for monitor in god_map.tree.control_loop_branch.check_monitors.children:
            if monitor == self.monitor or (hasattr(monitor, 'original') and monitor.original == self.monitor):
                god_map.tree.control_loop_branch.check_monitors.remove_child(monitor)
        return Status.SUCCESS
