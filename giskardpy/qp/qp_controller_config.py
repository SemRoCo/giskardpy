from __future__ import annotations

from typing import Optional

from giskardpy.data_types.data_types import Derivatives
from giskardpy.god_map import god_map
from giskardpy.qp.qp_controller import QPController, QPFormulation
from giskardpy.qp.qp_solver_ids import SupportedQPSolver


class QPControllerConfig:
    __max_derivative: Derivatives = Derivatives.jerk

    def __init__(self,
                 qp_solver: Optional[SupportedQPSolver] = None,
                 prediction_horizon: int = 7,
                 mpc_dt: float = 0.0125,
                 control_dt: Optional[float] = None,
                 max_trajectory_length: Optional[float] = 30,
                 retries_with_relaxed_constraints: int = 5,
                 qp_formulation: Optional[QPFormulation] = None,
                 added_slack: float = 100,
                 weight_factor: float = 100):
        """
        :param qp_solver: if not set, Giskard will search for the fasted installed solver.
        :param prediction_horizon: Giskard uses MPC and this is the length of the horizon. You usually don't need to change this.
        :param mpc_dt: time (s) difference between commands in the MPC horizon.
        :param max_trajectory_length: Giskard will stop planning/controlling the robot until this amount of s has passed.
                                      This is disabled if set to None.
        :param retries_with_relaxed_constraints: don't change, only for the pros.
        :param added_slack: don't change, only for the pros.
        :param weight_factor: don't change, only for the pros.
        """
        self.__qp_solver = qp_solver
        if prediction_horizon < 4:
            raise ValueError('prediction horizon must be >= 4.')
        self.__prediction_horizon = prediction_horizon
        self.__mpc_dt = mpc_dt
        self.control_dt = control_dt
        self.__max_trajectory_length = max_trajectory_length
        self.__retries_with_relaxed_constraints = retries_with_relaxed_constraints
        self.__added_slack = added_slack
        self.__qp_formulation = qp_formulation or QPFormulation()
        self.__weight_factor = weight_factor
        self.__endless_mode = self.__max_trajectory_length is None
        self.set_defaults()

    def set_defaults(self):
        god_map.qp_controller = QPController(mpc_dt=self.__mpc_dt,
                                             control_dt=self.control_dt,
                                             prediction_horizon=self.__prediction_horizon,
                                             qp_formulation=self.__qp_formulation,
                                             solver_id=self.__qp_solver,
                                             max_derivative=self.__max_derivative,
                                             retries_with_relaxed_constraints=self.__retries_with_relaxed_constraints,
                                             retry_added_slack=self.__added_slack,
                                             retry_weight_factor=self.__weight_factor)
