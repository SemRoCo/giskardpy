from __future__ import annotations
import abc
from abc import ABC
from collections import defaultdict
from enum import IntEnum
from typing import Dict, Optional, List, Union, DefaultDict

import numpy as np
import rospy
from numpy.typing import NDArray
from py_trees import Blackboard
from std_msgs.msg import ColorRGBA

from giskardpy import identifier
from giskardpy.exceptions import GiskardException, SetupException
from giskardpy.goals.goal import Goal
from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer
from giskardpy.model.joints import FixedJoint, OmniDrive, DiffDrive, Joint6DOF, OneDofJoint
from giskardpy.model.links import Link
from giskardpy.model.utils import robot_name_from_urdf_string
from giskardpy.model.world import WorldTree
from giskardpy.my_types import my_string, PrefixName, Derivatives, derivative_map
from giskardpy.tree.garden import OpenLoop, ClosedLoop, StandAlone, TreeManager
from giskardpy.utils import logging
from giskardpy.utils.utils import resolve_ros_iris, get_all_classes_in_package


class SupportedQPSolver(IntEnum):
    qpSWIFT = 1
    qpalm = 2
    gurobi = 3
    # clarabel = 4
    # qpOASES = 5
    # osqp = 6
    # quadprog = 7
    # cplex = 3
    # cvxopt = 7
    # qp_solvers = 8
    # mosek = 9
    # scs = 11
    # casadi = 12
    # super_csc = 14
    # cvxpy = 15

class QPControllerConfig:
    qp_solver: SupportedQPSolver
    prediction_horizon: int = 9
    sample_period: float = 0.05
    max_derivative: Derivatives = Derivatives.jerk
    action_server_name: str = '~command'
    max_trajectory_length: float = 30
    qp_solver: SupportedQPSolver = None,
    retries_with_relaxed_constraints: int = 5,
    added_slack: float = 100,
    weight_factor: float = 100

    def __init__(self,
                 qp_solver: Optional[SupportedQPSolver] = None,
                 prediction_horizon: int = 9,
                 sample_period: float = 0.05,
                 retries_with_relaxed_constraints: int = 5,
                 added_slack: float = 100,
                 weight_factor: float = 100,
                 endless_mode: bool = False):
        self.set_defaults()

    def set_defaults(self):
        self.qp_solver = None
        self.prediction_horizon = 9
        self.retries_with_relaxed_constraints = 5
        self.added_slack = 100
        self.sample_period = 0.05
        self.weight_factor = 100
        self.endless_mode = False
        self.default_weights = {d: defaultdict(float) for d in Derivatives}

    def set_prediction_horizon(self, new_prediction_horizon: int):
        """
        Set the prediction horizon for the MPC. If set to 1, it will turn off acceleration and jerk limits.
        :param new_prediction_horizon: should be >= 7
        """
        if new_prediction_horizon < 7:
            raise ValueError('prediction horizon must be >= 7.')
        self.prediction_horizon = new_prediction_horizon

    def set_qp_solver(self, new_solver: SupportedQPSolver):
        self.qp_solver = new_solver

    def set_max_trajectory_length(self, length: float = 30):
        self.max_trajectory_length = length

    def add_goal_package_name(self, package_name: str):
        new_goals = get_all_classes_in_package(package_name, Goal)
        if len(new_goals) == 0:
            raise GiskardException(f'No classes of type \'{Goal.__name__}\' found in {package_name}.')
        logging.loginfo(f'Made goal classes {new_goals} available Giskard.')
        self.goal_package_paths.add(package_name)
