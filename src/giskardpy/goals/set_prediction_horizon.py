from typing import Union, Optional, List

from giskardpy.configs.qp_controller_config import SupportedQPSolver
from giskardpy.goals.goal import NonMotionGoal
from giskardpy.monitors.monitors import ExpressionMonitor
from giskardpy.god_map import god_map
from giskardpy.utils import logging
import giskardpy.casadi_wrapper as cas


class SetPredictionHorizon(NonMotionGoal):
    def __init__(self, prediction_horizon: int, name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.TrueSymbol):
        """
        Will overwrite the prediction horizon for a single goal.
        Setting it to 1 will turn of acceleration and jerk limits.
        :param prediction_horizon: size of the prediction horizon, a number that should be 1 or above 5.
        """
        if name is None:
            name = self.__class__.__name__
        super().__init__(name=name)
        self.new_prediction_horizon = prediction_horizon

        if self.new_prediction_horizon < 7:
            logging.logwarn('Prediction horizon must be >= 7.')
        god_map.qp_controller_config.prediction_horizon = self.new_prediction_horizon


class SetQPSolver(NonMotionGoal):

    def __init__(self, qp_solver_id: Union[SupportedQPSolver, int], name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.TrueSymbol):
        if name is None:
            name = self.__class__.__name__
        super().__init__(name=name)
        qp_solver_id = SupportedQPSolver(qp_solver_id)
        god_map.qp_controller_config.set_qp_solver(qp_solver_id)


class EnableVelocityTrajectoryTracking(NonMotionGoal):
    def __init__(self, enabled: bool = True, name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.TrueSymbol):
        """
        A hack for the PR2. This goal decides whether the velocity part of the trajectory message is filled,
        when they are send to the robot.
        :param enabled: If True, will the velocity part of the message.
        """
        if name is None:
            name = self.__class__.__name__
        super().__init__(name=name)
        god_map.fill_trajectory_velocity_values = enabled
