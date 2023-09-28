from py_trees import Status

from giskardpy.god_map_interpreter import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time, catch_and_raise_to_blackboard


class CrucialMonitorsSatisfied(GiskardBehavior):
    @profile
    def __init__(self, name: str = 'crucial monitors satisfied'):
        super().__init__(name)

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        if god_map.monitor_manager.crucial_monitors_satisfied():
            return Status.SUCCESS
        return Status.RUNNING
