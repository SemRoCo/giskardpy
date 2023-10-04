from py_trees import Status

from giskardpy.exceptions import PlanningException
from giskardpy.god_map_interpreter import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time, catch_and_raise_to_blackboard


class MaxTrajectoryLength(GiskardBehavior):
    @profile
    def __init__(self, name, real_time=False):
        super().__init__(name)
        self.real_time = real_time

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        t = god_map.time
        length = god_map.qp_controller_config.max_trajectory_length
        if not self.real_time:
            length = god_map.qp_controller_config.max_trajectory_length
        else:
            sample_period = 1
        t = t * god_map.qp_controller_config.sample_period
        if t > length:
            raise PlanningException(f'Aborted because trajectory is longer than {length}')

        return Status.RUNNING
