from py_trees import Status

from giskardpy.god_map import god_map
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
        god_map.motion_statechart_manager.compile_node_state_updaters()
        return Status.SUCCESS
