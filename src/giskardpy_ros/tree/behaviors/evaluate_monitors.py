from py_trees import Status

from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time, catch_and_raise_to_blackboard


class EvaluateMonitors(GiskardBehavior):

    def __init__(self, name: str = 'evaluate monitors'):
        super().__init__(name)

    @catch_and_raise_to_blackboard
    @record_time
    def update(self):
        god_map.monitor_manager.evaluate_monitors()
        return Status.SUCCESS
