from py_trees import Status

from giskardpy.data_types.data_types import TaskState
from giskardpy.god_map import god_map
from giskardpy.motion_graph.monitors.monitors import Monitor
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time
from giskardpy_ros.tree.blackboard_utils import catch_and_raise_to_blackboard


class CheckMonitorState(GiskardBehavior):
    @profile
    def __init__(self, monitor: Monitor):
        self.monitor = monitor
        name = f'check\n{god_map.monitor_manager.format_condition(self.monitor.start_condition)}'
        super().__init__(name)

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        if god_map.monitor_manager.life_cycle_state[self.monitor.id] == TaskState.running:
            return Status.SUCCESS
        return Status.RUNNING
