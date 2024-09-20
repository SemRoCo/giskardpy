import rospy
from line_profiler import profile
from py_trees import Status

from giskardpy.god_map import god_map
from giskardpy.motion_graph.monitors.monitors import CancelMotion
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time
from giskardpy_ros.tree.blackboard_utils import catch_and_raise_to_blackboard, GiskardBlackboard


class CompileMonitors(GiskardBehavior):
    def __init__(self, name: str = 'compile monitors', traj_tracking: bool = False):
        super().__init__(name)
        self.traj_tracking = traj_tracking

    @catch_and_raise_to_blackboard
    @record_time
    def update(self):
        god_map.motion_graph_manager.compile_monitors()
        self.add_payload_monitors_to_behavior_tree(traj_tracking=self.traj_tracking)
        return Status.SUCCESS

    @profile
    def add_payload_monitors_to_behavior_tree(self, traj_tracking: bool = False) -> None:
        payload_monitors = sorted(god_map.motion_graph_manager.payload_monitors, key=lambda x: isinstance(x, CancelMotion))
        for monitor in payload_monitors:
            if traj_tracking:
                GiskardBlackboard().tree.execute_traj.base_closed_loop.check_monitors.add_monitor(monitor)
            else:
                GiskardBlackboard().tree.control_loop_branch.check_monitors.add_monitor(monitor)