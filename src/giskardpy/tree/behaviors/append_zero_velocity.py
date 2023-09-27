from py_trees import Status

from giskardpy.god_map_user import GodMap
from giskardpy.my_types import Derivatives
from giskardpy.tree.behaviors.plugin import GiskardBehavior
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
        for free_variable, state in GodMap.get_world().state.items():
            for derivative in Derivatives:
                if derivative == Derivatives.position:
                    continue
                GodMap.get_world().state[free_variable][derivative] = 0
        GodMap.get_world().notify_state_change()
        return Status.SUCCESS
