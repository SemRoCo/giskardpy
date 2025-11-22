from __future__ import annotations
import abc
from abc import ABC
from dataclasses import dataclass
from typing import List, Union, Tuple, TYPE_CHECKING

import semantic_digital_twin.spatial_types.spatial_types as cas
from semantic_digital_twin.spatial_types.derivatives import Derivatives
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom

from giskardpy.qp.constraint import DerivativeInequalityConstraint, DerivativeEqualityConstraint
from giskardpy.qp.constraint_collection import ConstraintCollection

if TYPE_CHECKING:
    from giskardpy.qp.qp_controller_config import QPControllerConfig

@dataclass
class ProblemDataPart(ABC):
    """
    min_x 0.5*x^T*diag(w)*x + g^T*x
    s.t.  lb <= x <= ub
               Ex = b
        lbA <= Ax <= ubA
    """

    degrees_of_freedom: List[DegreeOfFreedom]
    constraint_collection: ConstraintCollection
    config: QPControllerConfig

    def __post_init__(self):
        self.control_horizon = (
            self.config.prediction_horizon - self.config.max_derivative + 1
        )

    @property
    def number_of_free_variables(self) -> int:
        return len(self.degrees_of_freedom)

    @property
    def number_of_non_slack_columns(self) -> int:
        return (
            self.number_of_free_variables
            * self.config.prediction_horizon
            * self.config.max_derivative
        )

    @property
    def number_ineq_slack_variables(self):
        return sum(self.control_horizon for c in self.velocity_constraints)

    def get_derivative_constraints(
        self, derivative: Derivatives
    ) -> List[DerivativeInequalityConstraint]:
        return [
            c
            for c in self.constraint_collection.derivative_constraints
            if c.derivative == derivative
        ]

    def get_eq_derivative_constraints(
        self, derivative: Derivatives
    ) -> List[DerivativeEqualityConstraint]:
        return [
            c
            for c in self.constraint_collection.eq_derivative_constraints
            if c.derivative == derivative
        ]

    @abc.abstractmethod
    def construct_expression(
        self,
    ) -> Union[cas.Expression, Tuple[cas.Expression, cas.Expression]]:
        pass

    @property
    def velocity_constraints(self) -> List[DerivativeInequalityConstraint]:
        return self.get_derivative_constraints(Derivatives.velocity)

    @property
    def velocity_eq_constraints(self) -> List[DerivativeEqualityConstraint]:
        return self.get_eq_derivative_constraints(Derivatives.velocity)

    @property
    def acceleration_constraints(self) -> List[DerivativeInequalityConstraint]:
        return self.get_derivative_constraints(Derivatives.acceleration)

    @property
    def jerk_constraints(self) -> List[DerivativeInequalityConstraint]:
        return self.get_derivative_constraints(Derivatives.jerk)

    def _sorter(self, *args: dict) -> Tuple[List[cas.SymbolicScalar], np.ndarray]:
        """
        Sorts every arg dict individually and then appends all of them.
        :arg args: a bunch of dicts
        :return: list
        """
        import numpy as np
        result = []
        result_names = []
        for arg in args:
            result.extend(self.__helper(arg))
            result_names.extend(self.__helper_names(arg))
        return result, np.array(result_names)

    def __helper(self, param: dict):
        return [x for _, x in sorted(param.items())]

    def __helper_names(self, param: dict):
        return [x for x, _ in sorted(param.items())]

    def _remove_columns_columns_where_variables_are_zero(
        self, free_variable_model: cas.Expression, max_derivative: Derivatives
    ) -> cas.Expression:
        import numpy as np
        if np.prod(free_variable_model.shape) == 0:
            return free_variable_model
        column_ids = []
        end = 0
        for derivative in Derivatives.range(Derivatives.velocity, max_derivative - 1):
            last_non_zero_variable = self.config.prediction_horizon - (
                max_derivative - derivative
            )
            start = end + self.number_of_free_variables * last_non_zero_variable
            end += self.number_of_free_variables * self.config.prediction_horizon
            column_ids.extend(range(start, end))
        free_variable_model.remove([], column_ids)
        return free_variable_model
