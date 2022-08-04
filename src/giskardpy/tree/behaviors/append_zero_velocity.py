from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.data_types import JointStates, derivative_to_name
from giskardpy.tree.behaviors.plugin import GiskardBehavior


class SetZeroVelocity(GiskardBehavior):
    def __init__(self, name=None):
        if name is None:
            name = 'set velocity to zero'
        super().__init__(name)

    @profile
    def update(self):
        for free_variable, state in self.world.state.items():
            for derivative in derivative_to_name:
                if derivative == 0:
                    continue
                self.world.state[free_variable].set_derivative(derivative, 0)
        self.world.notify_state_change()
        return Status.SUCCESS
