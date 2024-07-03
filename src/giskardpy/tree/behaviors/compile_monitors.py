import rospy
from py_trees import Status

from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time, catch_and_raise_to_blackboard


class CompileMonitors(GiskardBehavior):
    def __init__(self, name: str = 'compile monitors', traj_tracking: bool = False):
        super().__init__(name)
        self.traj_tracking = traj_tracking

    @catch_and_raise_to_blackboard
    @record_time
    def update(self):
        god_map.monitor_manager.compile_monitors(traj_tracking=self.traj_tracking)
        return Status.SUCCESS
