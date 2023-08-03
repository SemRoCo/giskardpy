from typing import Union

from giskardpy import identifier
from giskardpy.configs.qp_controller_config import SupportedQPSolver
from giskardpy.goals.goal import Goal, NonMotionGoal
from giskardpy.tree.behaviors.max_trajectory_length import MaxTrajectoryLength
from giskardpy.utils import logging


class SetPredictionHorizon(Goal):
    def __init__(self, prediction_horizon: int):
        """
        Will overwrite the prediction horizon for a single goal.
        Setting it to 1 will turn of acceleration and jerk limits.
        :param prediction_horizon: size of the prediction horizon, a number that should be 1 or above 5.
        """
        super().__init__()
        self.new_prediction_horizon = prediction_horizon

    def make_constraints(self):
        pass
        if self.new_prediction_horizon < 7:
            logging.logwarn('Prediction horizon must be >= 7.')
        self.god_map.set_data(identifier.prediction_horizon, self.new_prediction_horizon)

    def __str__(self) -> str:
        return str(self.__class__.__name__)


class SetQPSolver(NonMotionGoal):

    def __init__(self, qp_solver_id: Union[SupportedQPSolver, int]):
        super().__init__()
        qp_solver_id = SupportedQPSolver(qp_solver_id)
        self.god_map.set_data(identifier.qp_solver_name, qp_solver_id)

    def __str__(self) -> str:
        return str(self.__class__.__name__)


class SetMaxTrajLength(NonMotionGoal):
    def __init__(self,
                 new_length: int):
        """
        Overwrites Giskard trajectory length limit for planning.
        If the trajectory is longer than new_length, Giskard will preempt the goal.
        :param new_length: in seconds
        """
        super().__init__()
        assert new_length > 0
        self.god_map.set_data(identifier.max_trajectory_length, new_length)

    def __str__(self) -> str:
        return super().__str__()


class EndlessMode(NonMotionGoal):
    def __init__(self):
        super().__init__()
        self.god_map.set_data(identifier.endless_mode, True)

    def __str__(self) -> str:
        return super().__str__()


class EnableVelocityTrajectoryTracking(NonMotionGoal):
    def __init__(self, enabled: bool = True):
        """
        A hack for the PR2. This goal decides whether the velocity part of the trajectory message is filled,
        when they are send to the robot.
        :param enabled: If True, will the velocity part of the message.
        """
        super().__init__()
        self.god_map.set_data(identifier.fill_trajectory_velocity_values, enabled)

    def __str__(self) -> str:
        return super().__str__()
