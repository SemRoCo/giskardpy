from py_trees import Status

from giskardpy.god_map import god_map
from giskardpy.data_types.data_types import Derivatives
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class SetZeroVelocity(GiskardBehavior):
    @profile
    def __init__(self, name=None):
        if name is None:
            name = 'set velocity to zero'
        super().__init__(name)

    @record_time
    @profile
    def update(self):
        for free_variable, state in god_map.world.state.items():
            for derivative in Derivatives:
                if derivative == Derivatives.position:
                    continue
                god_map.world.state[free_variable][derivative] = 0
        god_map.world.notify_state_change()
        return Status.SUCCESS
