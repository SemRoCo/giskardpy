from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.exceptions import PlanningException
from giskardpy.plugins.plugin import GiskardBehavior


class MaxTrajectoryLength(GiskardBehavior):
    def __init__(self, name, enabled, length):
        super(MaxTrajectoryLength, self).__init__(name)
        self.length = length

    def update(self):
        t = self.get_god_map().get_data(identifier.time)
        sample_period = self.get_god_map().get_data(identifier.sample_period)
        t = t * sample_period
        if t > self.length:
            raise PlanningException(u'Aborted because trajectory is longer than {}'.format(self.length))

        return Status.RUNNING
