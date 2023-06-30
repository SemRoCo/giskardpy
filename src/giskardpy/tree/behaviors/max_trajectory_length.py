from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.exceptions import PlanningException
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class MaxTrajectoryLength(GiskardBehavior):
    @profile
    def __init__(self, name, real_time=False):
        super().__init__(name)
        self.real_time = real_time
        self.endless_mode = False

    @profile
    def initialise(self):
        self.endless_mode = self.god_map.get_data(identifier.endless_mode)

    @record_time
    @profile
    def update(self):
        if self.endless_mode:
            return Status.RUNNING
        t = self.get_god_map().get_data(identifier.time)
        length = self.god_map.get_data(identifier.max_trajectory_length)
        if not self.real_time:
            sample_period = self.get_god_map().get_data(identifier.sample_period)
            t = t * sample_period
        if t > length:
            raise PlanningException(f'Aborted because trajectory is longer than {length}')

        return Status.RUNNING
