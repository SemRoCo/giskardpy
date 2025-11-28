from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Dict, Type

import numpy as np

from giskardpy.middleware import get_middleware
from giskardpy.qp.exceptions import QPSolverException
from giskardpy.qp.qp_formulation import QPFormulation
from giskardpy.qp.solvers.qp_solver import QPSolver
from giskardpy.qp.solvers.qp_solver_ids import SupportedQPSolver
from giskardpy.utils.utils import get_all_classes_in_module
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.spatial_types.derivatives import Derivatives

available_solvers: Dict[SupportedQPSolver, Type[QPSolver]] = {}


def detect_solvers():
    global available_solvers
    qp_solver_class: Type[QPSolver]
    for qp_solver_name in SupportedQPSolver:
        module_name = f"giskardpy.qp.solvers.qp_solver_{qp_solver_name.name}"
        try:
            qp_solver_class = list(
                get_all_classes_in_module(module_name, QPSolver).items()
            )[0][1]
            available_solvers[qp_solver_name] = qp_solver_class
        except Exception:
            continue
    solver_names = [solver_name.name for solver_name in available_solvers.keys()]
    print(f"Found these qp solvers: {solver_names}")


detect_solvers()


@dataclass
class QPControllerConfig:
    control_dt: Optional[float]
    """
    if control_dt is None, then the controller will run as fast as possible, only recommended for testing.
    """

    dof_weights: Dict[PrefixedName, DerivativeMap[float]] = field(
        default_factory=lambda: defaultdict(
            lambda: DerivativeMap([None, 0.01, np.inf, None])
        )
    )
    max_derivative: Derivatives = field(default=Derivatives.jerk)
    qp_solver_id: Optional[SupportedQPSolver] = field(default=None)
    prediction_horizon: int = field(default=7)
    mpc_dt: float = field(default=0.0125)
    max_trajectory_length: Optional[float] = field(default=30)
    horizon_weight_gain_scalar: float = 0.1
    qp_formulation: Optional[QPFormulation] = field(default_factory=QPFormulation)
    retries_with_relaxed_constraints: int = field(default=5)
    added_slack: float = field(default=100)
    weight_factor: float = field(default=100)
    verbose: bool = field(default=True)

    # %% init false
    qp_solver_class: Type[QPSolver] = field(init=False)

    def __post_init__(self):
        if not self.qp_formulation.is_mpc:
            self.prediction_horizon = 1
            self.max_derivative = Derivatives.velocity

        if self.prediction_horizon < 4:
            raise ValueError("prediction horizon must be >= 4.")
        self.__endless_mode = self.max_trajectory_length is None
        self.set_qp_solver()

    @classmethod
    def create_default_with_50hz(cls):
        return cls(
            control_dt=0.02,
            mpc_dt=0.02,
            prediction_horizon=7,
        )

    def set_qp_solver(self) -> None:
        if self.qp_solver_id is not None:
            self.qp_solver_class = available_solvers[self.qp_solver_id]
        else:
            for qp_solver_id in SupportedQPSolver:
                if qp_solver_id in available_solvers:
                    self.qp_solver_class = available_solvers[qp_solver_id]
                    break
            else:
                raise QPSolverException(f"No qp solver found")
            self.qp_solver_id = self.qp_solver_class.solver_id
        get_middleware().loginfo(
            f'QP Solver set to "{self.qp_solver_class.solver_id.name}"'
        )

    def set_dof_weight(
        self, dof_name: PrefixedName, derivative: Derivatives, weight: float
    ):
        """Set weight for a specific DOF derivative."""
        self.dof_weights[dof_name].data[derivative] = weight

    def set_dof_weights(self, dof_name: PrefixedName, weight_map: DerivativeMap[float]):
        """Set multiple weights for a DOF."""
        self.dof_weights[dof_name] = weight_map

    def get_dof_weight(self, dof_name: PrefixedName, derivative: Derivatives) -> float:
        """Get weight for a specific DOF derivative."""
        return self.dof_weights[dof_name].data[derivative]
