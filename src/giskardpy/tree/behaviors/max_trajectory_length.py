from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.exceptions import PlanningException
from giskardpy.god_map_interpreter import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time, catch_and_raise_to_blackboard


class MaxTrajectoryLength(GiskardBehavior):
    @profile
    def __init__(self, name, real_time=False):
        super().__init__(name)
        self.real_time = real_time
        self.endless_mode = False

    @profile
    def initialise(self):
        self.endless_mode = god_map.get_data(identifier.endless_mode)

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        if self.endless_mode:
            return Status.RUNNING
        t = god_map.get_data(identifier.time)
        length = god_map.get_data(identifier.max_trajectory_length)
        if not self.real_time:
            length = god_map.get_data(identifier.max_trajectory_length)
        else:
            sample_period = 1
        t = t * god_map.sample_period
        if t > length:
            raise PlanningException(f'Aborted because trajectory is longer than {length}')

        return Status.RUNNING
