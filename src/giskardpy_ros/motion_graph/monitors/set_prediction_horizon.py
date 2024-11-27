from typing import Optional

import giskardpy.casadi_wrapper as cas
from giskardpy.data_types.exceptions import MonitorInitalizationException
from giskardpy.god_map import god_map
from giskardpy.motion_graph.monitors.monitors import PayloadMonitor


class EnableVelocityTrajectoryTracking(PayloadMonitor):
    def __init__(self, enabled: bool = True, name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
        """
        A hack for the PR2. This goal decides whether the velocity part of the trajectory message is filled,
        when they are send to the robot.
        :param enabled: If True, will the velocity part of the message.
        """
        if not cas.is_true_symbol(start_condition):
            raise MonitorInitalizationException(f'{self.__class__.__name__}: start_condition must be True.')
        if name is None:
            name = self.__class__.__name__
        super().__init__(run_call_in_thread=False, name=name,
                         start_condition=start_condition,
                         hold_condition=hold_condition,
                         end_condition=end_condition)
        god_map.fill_trajectory_velocity_values = enabled

    def __call__(self):
        self.state = True
