import json

from py_trees import Status

from giskardpy.exceptions import PlanningException, LocalMinimumException
from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time, catch_and_raise_to_blackboard


class LocalMinimum(GiskardBehavior):
    @profile
    def __init__(self, name: str = 'local minimum reached?'):
        super().__init__(name)

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        if god_map.monitor_manager.is_local_minimum_reached():
            pretty_str = json.dumps(god_map.monitor_manager.get_state_dict(only_crucial=True), indent=4)
            raise LocalMinimumException(f'local min reached, but not all crucial monitors satisfied:\n'
                                        f'{pretty_str}')
        return Status.RUNNING
