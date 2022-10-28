from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.my_types import Derivatives
from giskardpy.tree.behaviors.plugin import GiskardBehavior


class SetZeroVelocity(GiskardBehavior):
    @profile
    def __init__(self, name=None):
        if name is None:
            name = 'set velocity to zero'
        super().__init__(name)

    @profile
    def update(self):
        for free_variable, state in self.world.state.items():
            for derivative in Derivatives:
                if derivative == Derivatives.position:
                    continue
                self.world.state[free_variable].set_derivative(derivative, 0)
        self.world.notify_state_change()
        return Status.SUCCESS
