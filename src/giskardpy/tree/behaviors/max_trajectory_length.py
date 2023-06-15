from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.exceptions import PlanningException
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class MaxTrajectoryLength(GiskardBehavior):
    @record_time
    @profile
    def update(self):
        t = self.get_god_map().get_data(identifier.time)
        sample_period = self.get_god_map().get_data(identifier.sample_period)
        length = self.god_map.get_data(identifier.max_trajectory_length)
        t = t * sample_period
        if t > length:
            raise PlanningException(f'Aborted because trajectory is longer than {length}')

        return Status.RUNNING
