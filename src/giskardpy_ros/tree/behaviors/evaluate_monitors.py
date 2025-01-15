from py_trees import Status

from giskardpy.god_map import god_map
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time
from giskardpy_ros.tree.blackboard_utils import catch_and_raise_to_blackboard


class EvaluateMonitors(GiskardBehavior):

    def __init__(self, name: str = 'evaluate monitors'):
        super().__init__(name)

    @catch_and_raise_to_blackboard
    @record_time
    def update(self):
        if god_map.motion_statechart_manager.evaluate_node_states():
            return Status.SUCCESS
        return Status.RUNNING
