from typing import Union, Optional

import giskardpy.casadi_wrapper as cas
from giskardpy.configs.qp_controller_config import SupportedQPSolver
from giskardpy.exceptions import MonitorInitalizationException
from giskardpy.god_map import god_map
from giskardpy.motion_graph.monitors.monitors import PayloadMonitor
from giskardpy.utils import logging


class SetPredictionHorizon(PayloadMonitor):
    def __init__(self, prediction_horizon: int, name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
        """
        Will overwrite the prediction horizon for a single goal.
        Setting it to 1 will turn of acceleration and jerk limits.
        :param prediction_horizon: size of the prediction horizon, a number that should be 1 or above 5.
        """
        if not cas.is_true(start_condition):
            raise MonitorInitalizationException(f'{self.__class__.__name__}: start_condition must be True.')
        if name is None:
            name = self.__class__.__name__
        super().__init__(run_call_in_thread=False, name=name,
                         start_condition=start_condition,
                         hold_condition=hold_condition,
                         end_condition=end_condition)
        self.new_prediction_horizon = prediction_horizon

        if self.new_prediction_horizon < 7:
            logging.logwarn('Prediction horizon must be >= 7.')
        god_map.qp_controller_config.prediction_horizon = self.new_prediction_horizon

    def __call__(self):
        self.state = True


class SetQPSolver(PayloadMonitor):

    def __init__(self, qp_solver_id: Union[SupportedQPSolver, int], name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
        if not cas.is_true(start_condition):
            raise MonitorInitalizationException(f'{self.__class__.__name__}: start_condition must be True.')
        if name is None:
            name = self.__class__.__name__
        super().__init__(run_call_in_thread=False, name=name,
                         start_condition=start_condition,
                         hold_condition=hold_condition,
                         end_condition=end_condition)
        qp_solver_id = SupportedQPSolver(qp_solver_id)
        god_map.qp_controller_config.set_qp_solver(qp_solver_id)

    def __call__(self):
        self.state = True


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
        if not cas.is_true(start_condition):
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
