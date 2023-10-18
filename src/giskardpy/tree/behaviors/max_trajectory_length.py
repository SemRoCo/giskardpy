from typing import Optional

from py_trees import Status

from giskardpy.exceptions import PlanningException
from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time, catch_and_raise_to_blackboard


class MaxTrajectoryLength(GiskardBehavior):
    @profile
    def __init__(self, name: Optional[str] = 'max traj length'):
        super().__init__(name)

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        length = god_map.qp_controller_config.max_trajectory_length
        if god_map.time > length:
            raise PlanningException(f'Aborted because trajectory is longer than {length}. '
                                    f'Final monitor state: {god_map.monitor_manager.get_state_dict()}')

        return Status.RUNNING
