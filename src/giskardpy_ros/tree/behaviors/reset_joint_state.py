from py_trees import Status

from giskardpy.god_map import god_map
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time
from giskardpy_ros.tree.blackboard_utils import catch_and_raise_to_blackboard


class ResetWorldState(GiskardBehavior):
    @record_time
    @profile
    def __init__(self, name: str = 'reset world state'):
        super().__init__(name)

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        js = god_map.trajectory.get_exact(0)
        god_map.world.state = js
        god_map.world.notify_state_change()
        return Status.SUCCESS

