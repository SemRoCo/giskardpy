from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.exceptions import PlanningException
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class MaxTrajectoryLength(GiskardBehavior):
    @profile
    def __init__(self, name, enabled, length):
        super().__init__(name)

    def initialise(self):
        self.length = self.god_map.get_data(identifier.MaxTrajectoryLength + ['length'])

    @record_time
    @profile
    def update(self):
        t = self.get_god_map().get_data(identifier.time)
        sample_period = self.get_god_map().get_data(identifier.sample_period)
        t = t * sample_period
        if t > self.length:
            raise PlanningException(f'Aborted because trajectory is longer than {self.length}')

        return Status.RUNNING
