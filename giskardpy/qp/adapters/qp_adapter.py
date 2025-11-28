from __future__ import annotations

import abc
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Tuple, List, Union, Dict, TYPE_CHECKING, DefaultDict, Optional, Type

import numpy as np
from line_profiler import profile

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.qp.exceptions import (
    InfeasibleException,
    VelocityLimitUnreachableException,
)
from giskardpy.middleware import get_middleware
from giskardpy.qp.constraint import (
    DerivativeInequalityConstraint,
    DerivativeEqualityConstraint,
)
from giskardpy.qp.constraint_collection import ConstraintCollection
from giskardpy.qp.pos_in_vel_limits import b_profile
from giskardpy.qp.qp_data import QPData
from giskardpy.qp.solvers.qp_solver import QPSolver
from giskardpy.qp.weight_gain import QuadraticWeightGain, LinearWeightGain
from giskardpy.utils.decorators import memoize
from giskardpy.utils.math import mpc
from semantic_digital_twin.spatial_types.derivatives import Derivatives, DerivativeMap
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom

if TYPE_CHECKING:
    import scipy.sparse as sp
    from giskardpy.qp.qp_controller_config import QPControllerConfig


def max_velocity_from_horizon_and_jerk_qp(
    prediction_horizon: int,
    vel_limit: float,
    acc_limit: float,
    jerk_limit: float,
    dt: float,
    max_derivative: Derivatives,
    solver_class: Type[QPSolver],
):
    upper_limits = (
        (vel_limit,) * prediction_horizon,
        (acc_limit,) * prediction_horizon,
        (jerk_limit,) * prediction_horizon,
    )
    lower_limits = (
        (-vel_limit,) * prediction_horizon,
        (-acc_limit,) * prediction_horizon,
        (-jerk_limit,) * prediction_horizon,
    )
    return mpc(
        upper_limits=upper_limits,
        lower_limits=lower_limits,
        current_values=(0, 0),
        dt=dt,
        ph=prediction_horizon,
        q_weight=(0, 0, 0),
        lin_weight=(-1, 0, 0),
        solver_class=solver_class,
        link_to_current_vel=False,
    )


@memoize
def find_best_jerk_limit(
    prediction_horizon: int,
    dt: float,
    target_vel_limit: float,
    solver_class: Type[QPSolver],
    eps: float = 0.0001,
) -> float:
    jerk_limit = (4 * target_vel_limit) / dt**2
    upper_bound = jerk_limit
    lower_bound = 0
    best_vel_limit = 0
    best_jerk_limit = 0
    i = -1
    for i in range(100):
        vel_limit = max_velocity_from_horizon_and_jerk_qp(
            prediction_horizon=prediction_horizon,
            vel_limit=1000,
            acc_limit=np.inf,
            jerk_limit=jerk_limit,
            dt=dt,
            max_derivative=Derivatives.jerk,
            solver_class=solver_class,
        )[0]
        if abs(vel_limit - target_vel_limit) < abs(best_vel_limit - target_vel_limit):
            best_vel_limit = vel_limit
            best_jerk_limit = jerk_limit
        if abs(vel_limit - target_vel_limit) < eps:
            break
        if vel_limit > target_vel_limit:
            upper_bound = jerk_limit
            jerk_limit = round((jerk_limit + lower_bound) / 2, 4)
        else:
            lower_bound = jerk_limit
            jerk_limit = round((jerk_limit + upper_bound) / 2, 4)
    print(
        f"best velocity limit: {best_vel_limit} "
        f"(target = {target_vel_limit}) with jerk limit: {best_jerk_limit} after {i + 1} iterations"
    )
    return best_jerk_limit


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

    @profile
    def velocity_limit(
        self, v: DegreeOfFreedom, max_derivative: Derivatives
    ) -> Tuple[cas.Expression, cas.Expression]:
        lower_limits = DerivativeMap()
        upper_limits = DerivativeMap()

        # %% pos limits
        if not v.has_position_limits():
            lower_limits.position = upper_limits.position = None
        else:
            lower_limits.position = v.lower_limits.position
            upper_limits.position = v.upper_limits.position

        # %% vel limits
        lower_limits.velocity = v.lower_limits.velocity
        upper_limits.velocity = v.upper_limits.velocity
        if self.config.prediction_horizon == 1:
            return cas.Expression([lower_limits.velocity]), cas.Expression(
                [upper_limits.velocity]
            )

        # %% acc limits
        if v.lower_limits.acceleration is None:
            lower_limits.acceleration = -np.inf
        else:
            lower_limits.acceleration = v.lower_limits.acceleration
        if v.upper_limits.acceleration is None:
            upper_limits.acceleration = np.inf
        else:
            upper_limits.acceleration = v.upper_limits.acceleration

        # %% jerk limits
        if upper_limits.jerk is None:
            upper_limits.jerk = find_best_jerk_limit(
                self.config.prediction_horizon,
                self.config.mpc_dt,
                upper_limits.velocity,
                solver_class=self.config.qp_solver_class,
            )
            lower_limits.jerk = -upper_limits.jerk
        else:
            upper_limits.jerk = v.upper_limits.jerk
            lower_limits.jerk = v.lower_limits.jerk

        try:
            lb, ub = b_profile(
                dof_symbols=v.variables,
                lower_limits=lower_limits,
                upper_limits=upper_limits,
                solver_class=self.config.qp_solver_class,
                dt=self.config.mpc_dt,
                ph=self.config.prediction_horizon,
            )
        except InfeasibleException as e:
            max_reachable_vel = max_velocity_from_horizon_and_jerk_qp(
                prediction_horizon=self.config.prediction_horizon,
                vel_limit=100,
                acc_limit=upper_limits.acceleration,
                jerk_limit=upper_limits.jerk,
                dt=self.config.mpc_dt,
                max_derivative=max_derivative,
                solver_class=self.config.qp_solver_class,
            )[0]
            if max_reachable_vel < upper_limits.velocity:
                error_msg = (
                    f'Free variable "{v.name}" can\'t reach velocity limit of "{upper_limits.velocity}". '
                    f'Maximum reachable with prediction horizon = "{self.config.prediction_horizon}", '
                    f'jerk limit = "{upper_limits.jerk}" and dt = "{self.config.mpc_dt}" is "{max_reachable_vel}".'
                )
                get_middleware().logerr(error_msg)
                raise VelocityLimitUnreachableException(error_msg)
            else:
                raise
        return lb, ub


@dataclass
class Weights(ProblemDataPart):
    """
    order:
        free_variable_velocity
        free_variable_acceleration
        free_variable_jerk
        eq integral constraints
        eq vel constraints
        neq integral constraints
        neq vel constraints
    """

    evaluated: bool = field(default=True)

    def linear_f(
        self,
        current_position: cas.FloatVariable,
        limit: float,
        target_value: float,
        a: float = 10,
        exp: float = 2,
    ) -> Tuple[cas.Expression, float]:
        f = cas.abs(current_position * a) ** exp
        x_offset = cas.solve_for(f, target_value)
        return (cas.abs(current_position + x_offset - limit) * a) ** exp, x_offset

    @profile
    def construct_expression(
        self,
        quadratic_weight_gains: List[QuadraticWeightGain] = None,
        linear_weight_gains: List[LinearWeightGain] = None,
    ) -> Union[cas.Expression, Tuple[cas.Expression, cas.Expression]]:
        quadratic_weight_gains = quadratic_weight_gains or []
        linear_weight_gains = linear_weight_gains or []
        components = []
        components.extend(
            self.free_variable_weights_expression(
                quadratic_weight_gains=quadratic_weight_gains
            )
        )
        components.append(self.equality_weight_expressions())
        components.extend(self.eq_derivative_weight_expressions())
        components.append(self.inequality_weight_expressions())
        components.extend(self.derivative_weight_expressions())
        weights, _ = self._sorter(*components)
        weights = cas.Expression(weights)
        linear_weights = self.linear_weights_expression(
            linear_weight_gains=linear_weight_gains
        )
        if linear_weights is None:
            linear_weights = cas.Expression.zeros(*weights.shape)
        else:
            # as of now linear weights are only added for joints, therefore equality-, derivative- and inequality
            # weights are missing. Here the missing weights are filled in with zeroes.
            linear_weights, _ = self._sorter(*linear_weights)
            linear_weights = cas.Expression(linear_weights)
            linear_weights = cas.vstack(
                [linear_weights]
                + [cas.Expression(0)] * (weights.shape[0] - linear_weights.shape[0])
            )
        return cas.Expression(weights), linear_weights

    @profile
    def free_variable_weights_expression(
        self, quadratic_weight_gains: List[QuadraticWeightGain]
    ) -> List[defaultdict]:
        max_derivative = self.config.max_derivative
        params = []
        weights = defaultdict(dict)  # maps order to joints
        for t in range(self.config.prediction_horizon):
            for v in self.degrees_of_freedom:
                for derivative in Derivatives.range(
                    Derivatives.velocity, max_derivative
                ):
                    if t >= self.config.prediction_horizon - (
                        max_derivative - derivative
                    ):
                        continue
                    if (
                        derivative == Derivatives.acceleration
                        and not self.config.qp_formulation.has_explicit_acc_variables
                    ):
                        continue
                    if (
                        derivative == Derivatives.jerk
                        and not self.config.qp_formulation.has_explicit_jerk_variables
                    ):
                        continue
                    normalized_weight = self.normalize_dof_weight(
                        limit=v.upper_limits.data[derivative],
                        base_weight=self.config.get_dof_weight(v.name, derivative),
                        t=t,
                        derivative=derivative,
                        horizon=self.config.prediction_horizon - 3,
                        alpha=self.config.horizon_weight_gain_scalar,
                    )
                    weights[derivative][
                        f"t{t:03}/{v.variables.position.dof.name}/{derivative}"
                    ] = normalized_weight
                    for q_gain in quadratic_weight_gains:
                        if (
                            t < len(q_gain.gains)
                            and v in q_gain.gains[t][derivative].keys()
                        ):
                            weights[derivative][
                                f"t{t:03}/{v.variables.position.dof.name}/{derivative}"
                            ] *= q_gain.gains[t][derivative][v]
        for _, weight in sorted(weights.items()):
            params.append(weight)
        return params

    def normalize_dof_weight(self, limit, base_weight, t, derivative, horizon, alpha):
        def linear(x_in: float, weight: float, h: int, alpha: float) -> float:
            start = weight * alpha
            a = (weight - start) / h
            return a * x_in + start

        if limit is None:
            return 0.0
        weight = linear(t, base_weight, horizon, alpha)

        return weight * (1 / limit) ** 2

    def derivative_weight_expressions(self) -> List[Dict[str, cas.Expression]]:
        params = []
        for d in Derivatives.range(Derivatives.velocity, self.config.max_derivative):
            derivative_constr_weights = {}
            for t in range(self.config.prediction_horizon):
                d = Derivatives(d)
                for c in self.get_derivative_constraints(d):
                    if t < self.control_horizon:
                        derivative_constr_weights[f"t{t:03}/{c.name}"] = (
                            c.normalized_weight()
                        )
            params.append(derivative_constr_weights)
        return params

    def eq_derivative_weight_expressions(self) -> List[Dict[str, cas.Expression]]:
        params = []
        for d in Derivatives.range(Derivatives.velocity, self.config.max_derivative):
            derivative_constr_weights = {}
            for t in range(self.config.prediction_horizon):
                d = Derivatives(d)
                for c in self.get_eq_derivative_constraints(d):
                    if t < self.control_horizon:
                        derivative_constr_weights[f"t{t:03}/{c.name}"] = (
                            c.normalized_weight()
                        )
            params.append(derivative_constr_weights)
        return params

    def equality_weight_expressions(self) -> dict:
        error_slack_weights = {
            f"{c.name}/error": c.normalized_weight(self.control_horizon)
            for c in self.constraint_collection.eq_constraints
        }
        return error_slack_weights

    def inequality_weight_expressions(self) -> dict:
        error_slack_weights = {
            f"{c.name}/error": c.normalized_weight(self.control_horizon)
            for c in self.constraint_collection.neq_constraints
        }
        return error_slack_weights

    @profile
    def linear_weights_expression(
        self, linear_weight_gains: List[LinearWeightGain] = None
    ):
        if len(linear_weight_gains) > 0:
            params = []
            weights = defaultdict(dict)  # maps order to joints
            for t in range(self.config.prediction_horizon):
                for v in self.degrees_of_freedom:
                    for derivative in Derivatives.range(
                        Derivatives.velocity, self.config.max_derivative
                    ):
                        if t >= self.config.prediction_horizon - (
                            self.config.max_derivative - derivative
                        ):
                            continue
                        if (
                            derivative == Derivatives.acceleration
                            and not self.config.qp_formulation.has_explicit_acc_variables
                        ):
                            continue
                        if (
                            derivative == Derivatives.jerk
                            and not self.config.qp_formulation.has_explicit_jerk_variables
                        ):
                            continue
                        weights[derivative][
                            f"t{t:03}/{v.variables.position}/{derivative}"
                        ] = 0
                        for l_gain in linear_weight_gains:
                            if (
                                t < len(l_gain.gains)
                                and v in l_gain.gains[t][derivative].keys()
                            ):
                                weights[derivative][
                                    f"t{t:03}/{v.variables.position}/{derivative}"
                                ] += l_gain.gains[t][derivative][v]
            for _, weight in sorted(weights.items()):
                params.append(weight)
            return params
        return None

    def get_free_variable_symbols(self, order: Derivatives) -> List[cas.FloatVariable]:
        return self._sorter(
            {
                v.variables.position: v.variables.data[order]
                for v in self.degrees_of_freedom
            }
        )[0]


@dataclass
class FreeVariableBounds(ProblemDataPart):
    """
    order:
        free_variable_velocity
        free_variable_acceleration
        free_variable_jerk
        eq integral constraints
        eq vel constraints
        neq integral constraints
        neq vel constraints
    """

    names: np.ndarray = field(default=None)
    names_without_slack: np.ndarray = field(default=None)
    names_slack: np.ndarray = field(default=None)
    names_neq_slack: np.ndarray = field(default=None)
    names_derivative_slack: np.ndarray = field(default=None)
    names_eq_slack: np.ndarray = field(default=None)
    evaluated: bool = field(default=True)

    @profile
    def free_variable_bounds(
        self,
    ) -> Tuple[List[Dict[str, cas.ScalarData]], List[Dict[str, cas.ScalarData]]]:
        # if self.config.qp_formulation in [ControllerMode.explicit, ControllerMode.explicit_no_acc]:
        max_derivative = self.config.max_derivative
        lb: DefaultDict[Derivatives, Dict[str, cas.ScalarData]] = defaultdict(dict)
        ub: DefaultDict[Derivatives, Dict[str, cas.ScalarData]] = defaultdict(dict)
        for v in self.degrees_of_freedom:
            if self.config.qp_formulation.has_explicit_pos_limits:
                for t in range(self.config.prediction_horizon):
                    for derivative in Derivatives.range(
                        Derivatives.velocity, max_derivative
                    ):
                        if t >= self.config.prediction_horizon - (
                            max_derivative - derivative
                        ):
                            continue
                        if (
                            derivative == Derivatives.acceleration
                            and not self.config.qp_formulation.has_explicit_acc_variables
                        ):
                            continue
                        if (
                            derivative == Derivatives.jerk
                            and not self.config.qp_formulation.has_explicit_jerk_variables
                        ):
                            continue
                        index = t + self.config.prediction_horizon * (derivative - 1)
                        lb[derivative][f"t{t:03}/{v.name}/{derivative}"] = (
                            v.lower_limits.data[derivative]
                        )
                        ub[derivative][f"t{t:03}/{v.name}/{derivative}"] = (
                            v.upper_limits.data[derivative]
                        )
            else:
                lb_, ub_ = self.velocity_limit(v=v, max_derivative=max_derivative)
                for t in range(self.config.prediction_horizon):
                    for derivative in Derivatives.range(
                        Derivatives.velocity, max_derivative
                    ):
                        if t >= self.config.prediction_horizon - (
                            max_derivative - derivative
                        ):
                            continue
                        if (
                            derivative == Derivatives.acceleration
                            and not self.config.qp_formulation.has_explicit_acc_variables
                        ):
                            continue
                        if (
                            derivative == Derivatives.jerk
                            and not self.config.qp_formulation.has_explicit_jerk_variables
                        ):
                            continue
                        index = t + self.config.prediction_horizon * (derivative - 1)
                        lb[derivative][f"t{t:03}/{v.name}/{derivative}"] = lb_[index]
                        ub[derivative][f"t{t:03}/{v.name}/{derivative}"] = ub_[index]
        lb_params = []
        ub_params = []
        for derivative, name_to_bound_map in sorted(lb.items()):
            lb_params.append(name_to_bound_map)
        for derivative, name_to_bound_map in sorted(ub.items()):
            ub_params.append(name_to_bound_map)
        return lb_params, ub_params

    def derivative_slack_limits(
        self, derivative: Derivatives
    ) -> Tuple[Dict[str, cas.Expression], Dict[str, cas.Expression]]:
        lower_slack = {}
        upper_slack = {}
        for t in range(self.config.prediction_horizon):
            for c in self.get_derivative_constraints(derivative):
                if t < self.control_horizon:
                    lower_slack[f"t{t:03}/{c.name}"] = c.lower_slack_limit
                    upper_slack[f"t{t:03}/{c.name}"] = c.upper_slack_limit
        return lower_slack, upper_slack

    def eq_derivative_slack_limits(
        self, derivative: Derivatives
    ) -> Tuple[Dict[str, cas.Expression], Dict[str, cas.Expression]]:
        lower_slack = {}
        upper_slack = {}
        for t in range(self.config.prediction_horizon):
            for c in self.get_eq_derivative_constraints(derivative):
                if t < self.control_horizon:
                    lower_slack[f"t{t:03}/{c.name}"] = c.lower_slack_limit[t]
                    upper_slack[f"t{t:03}/{c.name}"] = c.upper_slack_limit[t]
        return lower_slack, upper_slack

    def equality_constraint_slack_lower_bound(self):
        return {
            f"{c.name}/error": c.lower_slack_limit
            for c in self.constraint_collection.eq_constraints
        }

    def equality_constraint_slack_upper_bound(self):
        return {
            f"{c.name}/error": c.upper_slack_limit
            for c in self.constraint_collection.eq_constraints
        }

    def inequality_constraint_slack_lower_bound(self):
        return {
            f"{c.name}/error": c.lower_slack_limit
            for c in self.constraint_collection.neq_constraints
        }

    def inequality_constraint_slack_upper_bound(self):
        return {
            f"{c.name}/error": c.upper_slack_limit
            for c in self.constraint_collection.neq_constraints
        }

    @profile
    def construct_expression(
        self,
    ) -> Union[cas.Expression, Tuple[cas.Expression, cas.Expression]]:
        # derivative model
        lb_params, ub_params = self.free_variable_bounds()
        num_free_variables = sum(len(x) for x in lb_params)

        # eq integral constraints
        equality_constraint_slack_lower_bounds = (
            self.equality_constraint_slack_lower_bound()
        )
        num_eq_slacks = len(equality_constraint_slack_lower_bounds)
        lb_params.append(equality_constraint_slack_lower_bounds)
        ub_params.append(self.equality_constraint_slack_upper_bound())

        # eq vel constraints
        num_eq_derivative_slack = 0
        for derivative in Derivatives.range(
            Derivatives.velocity, self.config.max_derivative
        ):
            lower_slack, upper_slack = self.eq_derivative_slack_limits(derivative)
            num_eq_derivative_slack += len(lower_slack)
            lb_params.append(lower_slack)
            ub_params.append(upper_slack)

        # neq integral constraints
        lb_params.append(self.inequality_constraint_slack_lower_bound())
        ub_params.append(self.inequality_constraint_slack_upper_bound())

        # neq vel constraints
        num_derivative_slack = 0
        for derivative in Derivatives.range(
            Derivatives.velocity, self.config.max_derivative
        ):
            lower_slack, upper_slack = self.derivative_slack_limits(derivative)
            num_derivative_slack += len(lower_slack)
            lb_params.append(lower_slack)
            ub_params.append(upper_slack)

        lb, self.names = self._sorter(*lb_params)
        ub, _ = self._sorter(*ub_params)
        self.names_without_slack = self.names[:num_free_variables]
        self.names_slack = self.names[num_free_variables:]

        derivative_slack_start = 0
        derivative_slack_stop = (
            derivative_slack_start + num_derivative_slack + num_eq_derivative_slack
        )
        self.names_derivative_slack = self.names_slack[
            derivative_slack_start:derivative_slack_stop
        ]

        eq_slack_start = derivative_slack_stop
        eq_slack_stop = eq_slack_start + num_eq_slacks
        self.names_eq_slack = self.names_slack[eq_slack_start:eq_slack_stop]

        neq_slack_start = eq_slack_stop
        self.names_neq_slack = self.names_slack[neq_slack_start:]
        return cas.Expression(lb), cas.Expression(ub)


@dataclass
class EqualityBounds(ProblemDataPart):
    """

    order:
        derivative model (optional)
        eq integral constraints
        eq vel constraints
    """

    names: np.ndarray = field(default=None)
    names_equality_constraints: np.ndarray = field(default=None)
    names_derivative_links: np.ndarray = field(default=None)
    evaluated: bool = field(default=True)

    def equality_constraint_bounds(self) -> Dict[str, cas.Expression]:
        return {
            f"{c.name}": c.capped_bound(self.config.mpc_dt, self.control_horizon)
            for c in self.constraint_collection.eq_constraints
        }

    def last_derivative_values(
        self, derivative: Derivatives
    ) -> Dict[str, cas.ScalarData]:
        last_values = {}
        for v in self.degrees_of_freedom:
            last_values[f"{v.name}/last_{derivative}"] = v.variables.data[derivative]
        return last_values

    def derivative_links(self, derivative: Derivatives) -> Dict[str, cas.ScalarData]:
        derivative_link = {}
        for t in range(self.config.prediction_horizon - 1):
            if t >= self.config.prediction_horizon - (
                self.config.max_derivative - derivative
            ):
                continue  # this row is all zero in the model, because the system has to stop at 0 vel
            for v in self.degrees_of_freedom:
                derivative_link[f"t{t:03}/{derivative}/{v.name}/link"] = 0
        return derivative_link

    def eq_derivative_constraint_bounds(
        self, derivative: Derivatives
    ) -> Dict[str, cas.Expression]:
        bound = {}
        for t in range(self.config.prediction_horizon):
            for c in self.get_eq_derivative_constraints(derivative):
                if t < self.control_horizon:
                    bound[f"t{t:03}/{c.name}"] = c.bound[t] * self.config.mpc_dt
        return bound

    @profile
    def construct_expression(
        self,
    ) -> Union[cas.Expression, Tuple[cas.Expression, cas.Expression]]:
        """
        explicit no acc
        -vc - ac*dt = -v0 + j0*dt**2
                 vc = -v1 + 2*v0 + j1*dt**2
                  0 = -vt + 2*vt-1 - vt-2 + jt*dt**2
                  0 =   0 + 2*vt-1 - vt-2 + jt*dt**2
                  0 =   0 +    0   - vt-2 + jt*dt**2
        :return:
        """
        bounds = []
        # derivative model
        if (
            not self.config.qp_formulation.has_explicit_acc_variables
            and self.config.qp_formulation.has_explicit_jerk_variables
        ):
            derivative_link = {}
            for t in range(self.config.prediction_horizon):
                for v in self.degrees_of_freedom:
                    name = f"t{t:03}/{Derivatives.jerk}/{v.name}/link"
                    if t == 0:
                        derivative_link[name] = (
                            -v.variables.velocity
                            - v.variables.acceleration * self.config.mpc_dt
                        )
                    elif t == 1:
                        derivative_link[name] = v.variables.velocity
                    else:
                        derivative_link[name] = 0
            bounds.append(derivative_link)
        else:
            if self.config.qp_formulation.is_implicit:
                max_derivative = Derivatives.velocity
            else:
                max_derivative = self.config.max_derivative

            for derivative in Derivatives.range(
                Derivatives.velocity, max_derivative - 1
            ):
                bounds.append(self.last_derivative_values(derivative))
                bounds.append(self.derivative_links(derivative))

        num_derivative_links = sum(len(x) for x in bounds)
        num_derivative_constraints = 0

        # eq integral constraints
        bounds.append(self.equality_constraint_bounds())

        # eq vel constraints
        for derivative in Derivatives.range(
            Derivatives.velocity, self.config.max_derivative
        ):
            bound = self.eq_derivative_constraint_bounds(derivative)
            num_derivative_constraints += len(bound)
            bounds.append(bound)

        bounds, self.names = self._sorter(*bounds)
        self.names_derivative_links = self.names[:num_derivative_links]
        # self.names_equality_constraints = self.names[num_derivative_links:]
        return cas.Expression(bounds)


@dataclass
class InequalityBounds(ProblemDataPart):
    """
    order:
        derivative model (optional)
        neq integral constraints
        neq vel constraints
    """

    default_limits: bool = field(default=False)
    names: np.ndarray = field(default=None)
    names_position_limits: np.ndarray = field(default=None)
    names_derivative_links: np.ndarray = field(default=None)
    names_neq_constraints: np.ndarray = field(default=None)
    names_non_position_limits: np.ndarray = field(default=None)
    evaluated: bool = field(default=True)

    def derivative_constraint_bounds(
        self, derivative: Derivatives
    ) -> Tuple[Dict[str, cas.Expression], Dict[str, cas.Expression]]:
        lower = {}
        upper = {}
        for t in range(self.config.prediction_horizon):
            for c in self.get_derivative_constraints(derivative):
                if t < self.control_horizon:
                    lower[f"t{t:03}/{c.name}"] = c.lower_limit * self.config.mpc_dt
                    upper[f"t{t:03}/{c.name}"] = c.upper_limit * self.config.mpc_dt
        return lower, upper

    def lower_inequality_constraint_bound(self):
        bounds = {}
        for constraint in self.constraint_collection.neq_constraints:
            if isinstance(constraint.lower_error, float) and np.isinf(
                constraint.lower_error
            ):
                bounds[f"{constraint.name}"] = constraint.lower_error
            else:
                bounds[f"{constraint.name}"] = constraint.capped_lower_error(
                    self.config.mpc_dt, self.control_horizon
                )
        return bounds

    def upper_inequality_constraint_bound(self):
        bounds = {}
        for constraint in self.constraint_collection.neq_constraints:
            if isinstance(constraint.upper_error, float) and np.isinf(
                constraint.upper_error
            ):
                bounds[f"{constraint.name}"] = constraint.upper_error
            else:
                bounds[f"{constraint.name}"] = constraint.capped_upper_error(
                    self.config.mpc_dt, self.control_horizon
                )
        return bounds

    def implicit_pos_model_limits(
        self,
    ) -> Tuple[List[Dict[str, cas.Expression]], List[Dict[str, cas.Expression]]]:
        lb_acc, ub_acc = {}, {}
        lb_jerk, ub_jerk = {}, {}
        for v in self.degrees_of_freedom:
            lb_, ub_ = v.lower_limits.position, v.upper_limits.position
            for t in range(self.config.prediction_horizon - 2):
                ptc = v.variables.position
                lb_jerk[f"t{t:03}/{v.name}/{Derivatives.position}"] = lb_ - ptc
                ub_jerk[f"t{t:03}/{v.name}/{Derivatives.position}"] = ub_ - ptc
        return [lb_acc, lb_jerk], [ub_acc, ub_jerk]

    def implicit_model_limits(
        self,
    ) -> Tuple[List[Dict[str, cas.Expression]], List[Dict[str, cas.Expression]]]:
        lb_acc, ub_acc = {}, {}
        lb_jerk, ub_jerk = {}, {}
        for v in self.degrees_of_freedom:
            if self.config.qp_formulation.has_explicit_pos_limits:
                lb_, ub_ = v.lower_limits.jerk, v.upper_limits.jerk
            else:
                lb_, ub_ = self.velocity_limit(v=v, max_derivative=Derivatives.jerk)
            for t in range(self.config.prediction_horizon):
                # if self.config.max_derivative >= Derivatives.acceleration:
                #     a_min = v.get_lower_limit(Derivatives.acceleration)
                #     a_max = v.get_upper_limit(Derivatives.acceleration)
                #     if not ((np.isinf(a_min) or cas.is_inf(a_min)) and (np.isinf(a_max) or cas.is_inf(a_max))):
                #         vtc = v.symbols.velocity
                #         if t == 0:
                #             # vtc/dt + a_min <= vt0/dt <= vtc/dt + a_max
                #             lb_acc[f't{t:03}/{v.name}/{Derivatives.acceleration}'] = vtc / self.config.mpc_dt + a_min
                #             ub_acc[f't{t:03}/{v.name}/{Derivatives.acceleration}'] = vtc / self.config.mpc_dt + a_max
                #         else:
                #             lb_acc[f't{t:03}/{v.name}/{Derivatives.acceleration}'] = a_min
                #             ub_acc[f't{t:03}/{v.name}/{Derivatives.acceleration}'] = a_max
                if self.config.max_derivative >= Derivatives.jerk:
                    if self.config.qp_formulation.has_explicit_pos_limits:
                        j_min = lb_
                        j_max = ub_
                    else:
                        j_min = lb_[self.config.prediction_horizon * 2 + t]
                        j_max = ub_[self.config.prediction_horizon * 2 + t]
                    vtc = v.variables.velocity
                    atc = v.variables.acceleration
                    if t == 0:
                        # vtc/dt**2 + atc/dt + j_min <=    vt0/dt**2     <= vtc/dt**2 + atc/dt + j_max
                        lb_jerk[f"t{t:03}/{v.name}/{Derivatives.jerk}"] = (
                            vtc / self.config.mpc_dt**2
                            + atc / self.config.mpc_dt
                            + j_min
                        )
                        ub_jerk[f"t{t:03}/{v.name}/{Derivatives.jerk}"] = (
                            vtc / self.config.mpc_dt**2
                            + atc / self.config.mpc_dt
                            + j_max
                        )
                    elif t == 1:
                        # (- vtc)/dt**2 + j_min <= (vt1 - 2vt0)/dt**2 <= (- vtc)/dt**2 + j_max
                        lb_jerk[f"t{t:03}/{v.name}/{Derivatives.jerk}"] = (
                            -vtc / self.config.mpc_dt**2 + j_min
                        )
                        ub_jerk[f"t{t:03}/{v.name}/{Derivatives.jerk}"] = (
                            -vtc / self.config.mpc_dt**2 + j_max
                        )
                    else:
                        lb_jerk[f"t{t:03}/{v.name}/{Derivatives.jerk}"] = j_min
                        ub_jerk[f"t{t:03}/{v.name}/{Derivatives.jerk}"] = j_max
        return [lb_acc, lb_jerk], [ub_acc, ub_jerk]

    @profile
    def construct_expression(
        self,
    ) -> Union[cas.Expression, Tuple[cas.Expression, cas.Expression]]:
        lb_params: List[Dict[str, cas.Expression]] = []
        ub_params: List[Dict[str, cas.Expression]] = []

        # derivative model
        if self.config.qp_formulation.has_explicit_pos_limits:
            lb, ub = self.implicit_pos_model_limits()
            lb_params.extend(lb)
            ub_params.extend(ub)

        if self.config.qp_formulation.is_implicit:
            lb, ub = self.implicit_model_limits()
            lb_params.extend(lb)
            ub_params.extend(ub)

        # neq integral constraints
        lower_inequality_constraint_bounds = self.lower_inequality_constraint_bound()
        lb_params.append(lower_inequality_constraint_bounds)
        ub_params.append(self.upper_inequality_constraint_bound())
        num_neq_constraints = len(lower_inequality_constraint_bounds)

        # neq vel constraints
        num_derivative_constraints = 0
        for derivative in Derivatives.range(
            Derivatives.velocity, self.config.max_derivative
        ):
            lower, upper = self.derivative_constraint_bounds(derivative)
            num_derivative_constraints += len(lower)
            lb_params.append(lower)
            ub_params.append(upper)

        lbA, self.names = self._sorter(*lb_params)
        ubA, _ = self._sorter(*ub_params)

        self.names_derivative_links = self.names[:num_derivative_constraints]
        self.names_neq_constraints = self.names[
            num_derivative_constraints + num_neq_constraints :
        ]

        return cas.Expression(lbA), cas.Expression(ubA)


@dataclass
class EqualityModel(ProblemDataPart):
    """
    Format:
        last free variable velocity
        0
        last free variable acceleration
        0
        equality_constraint_bounds
    """

    def equality_constraint_expressions(self) -> List[cas.Expression]:
        return self._sorter(
            {c.name: c.expression for c in self.constraint_collection.eq_constraints}
        )[0]

    def get_free_variable_symbols(
        self, derivative: Derivatives
    ) -> List[cas.FloatVariable]:
        return self._sorter(
            {
                v.variables.position.name: v.variables.data[derivative]
                for v in self.degrees_of_freedom
            }
        )[0]

    def get_eq_derivative_constraint_expressions(self, derivative: Derivatives):
        return self._sorter(
            {
                c.name: c.expression
                for c in self.constraint_collection.eq_derivative_constraints
                if c.derivative == derivative
            }
        )[0]

    def velocity_constraint_model(self) -> Tuple[cas.Expression, cas.Expression]:
        """
        model
        |   t1   |   t2   |   t3   |   t1   |   t2   |   t3   |   t1   |   t2   |   t3   | prediction horizon
        |v1 v2 v3|v1 v2 v3|v1 v2 v3|a1 a2 a3|a1 a2 a3|a1 a2 a3|j1 j2 j3|j1 j2 j3|j1 j2 j3| free variables
        |--------------------------------------------------------------------------------|
        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |        |
        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |        |
        |--------------------------------------------------------------------------------|
        |        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |
        |        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |
        |--------------------------------------------------------------------------------|

        slack model
        |   t1 |   t2 | prediction horizon
        |s1 s2 |s1 s2 | slack
        |------|------|
        |sp    |      | vel constr 1
        |   sp |      | vel constr 2
        |-------------|
        |      |sp    | vel constr 1
        |      |   sp | vel constr 2
        |-------------|
        """
        number_of_vel_rows = len(self.velocity_eq_constraints) * (
            self.config.prediction_horizon - 2
        )
        if number_of_vel_rows > 0:
            expressions = cas.Expression(
                self.get_eq_derivative_constraint_expressions(Derivatives.velocity)
            )
            parts = []
            for derivative in Derivatives.range(
                Derivatives.position, self.config.max_derivative - 1
            ):
                if (
                    derivative == Derivatives.velocity
                    and not self.config.qp_formulation.has_explicit_acc_variables
                ):
                    continue
                if (
                    derivative == Derivatives.acceleration
                    and not self.config.qp_formulation.has_explicit_jerk_variables
                ):
                    continue
                J_vel = (
                    expressions.jacobian(
                        variables=self.get_free_variable_symbols(derivative)
                    )
                    * self.config.mpc_dt
                )
                missing_variables = self.config.max_derivative - derivative - 1
                eye = cas.Expression.eye(self.config.prediction_horizon)[
                    :-2, : self.config.prediction_horizon - missing_variables
                ]
                J_vel_limit_block = eye.kron(J_vel)
                parts.append(J_vel_limit_block)

            # constraint slack
            model = cas.hstack(parts)
            num_slack_variables = sum(
                self.control_horizon for c in self.velocity_eq_constraints
            )
            slack_model = cas.Expression.eye(num_slack_variables) * self.config.mpc_dt
            return model, slack_model
        return cas.Expression(), cas.Expression()

    @property
    def number_of_non_slack_columns(self) -> int:
        vel_columns = self.number_of_free_variables * (
            self.config.prediction_horizon - 2
        )
        acc_columns = self.number_of_free_variables * (
            self.config.prediction_horizon - 1
        )
        jerk_columns = self.number_of_free_variables * self.config.prediction_horizon
        result = vel_columns
        if self.config.qp_formulation.has_explicit_acc_variables:
            result += acc_columns
        if self.config.qp_formulation.has_explicit_jerk_variables:
            result += jerk_columns
        return result

    @profile
    def derivative_link_model(self, max_derivative: Derivatives) -> cas.Expression:
        """
        Layout for prediction horizon 5
        Slots are matrices of |controlled variables| x |controlled variables|
        | vt0 | vt1 | vt2 | at0 | at1 | at2 | at3 | jt0 | jt1 | jt2 | jt3 | jt4 |
        |-----------------------------------------------------------------------|
        |  1  |     |     | -dt |     |     |     |     |     |     |     |     | last_v =  vt0 - at0*cdt
        | -1  |  1  |     |     | -dt |     |     |     |     |     |     |     |      0 = -vt0 + vt1 - at1 * mdt
        |     | -1  |  1  |     |     | -dt |     |     |     |     |     |     |      0 = -vt1 + vt2 - at2 * mdt
        |     |     | -1  |     |     |     | -dt |     |     |     |     |     |      0 = -vt2 - at3 * mdt
        |=======================================================================|
        |     |     |     |  1  |     |     |     | -dt |     |     |     |     | last_a =  at0 - jt0*cdt
        |     |     |     | -1  |  1  |     |     |     | -dt |     |     |     |      0 = -at0 + at1 - jt1 * mdt
        |     |     |     |     | -1  |  1  |     |     |     | -dt |     |     |      0 = -at1 + at2 - jt2 * mdt
        |     |     |     |     |     | -1  |  1  |     |     |     | -dt |     |      0 = -at2 + at3 - jt3 * mdt
        |     |     |     |     |     |     | -1  |     |     |     |     | -dt |      0 = -at3 - jt4 * mdt
        |-----------------------------------------------------------------------|
        """
        num_rows = (
            self.number_of_free_variables
            * self.config.prediction_horizon
            * (max_derivative - 1)
        )
        num_columns = (
            self.number_of_free_variables
            * self.config.prediction_horizon
            * max_derivative
        )
        derivative_link_model = cas.Expression.zeros(num_rows, num_columns)

        x_n = cas.Expression.eye(num_rows)
        derivative_link_model[:, : x_n.shape[0]] += x_n

        xd_n = -cas.Expression.eye(num_rows) * self.config.mpc_dt
        h_offset = self.number_of_free_variables * self.config.prediction_horizon
        derivative_link_model[:, h_offset:] += xd_n

        x_c_height = self.number_of_free_variables * (
            self.config.prediction_horizon - 1
        )
        x_c = -cas.Expression.eye(x_c_height)
        offset_v = 0
        offset_h = 0
        for derivative in Derivatives.range(Derivatives.velocity, max_derivative - 1):
            offset_v += self.number_of_free_variables
            derivative_link_model[
                offset_v : offset_v + x_c_height, offset_h : offset_h + x_c_height
            ] += x_c
            offset_v += x_c_height
            offset_h += self.config.prediction_horizon * self.number_of_free_variables
        derivative_link_model = self._remove_rows_columns_where_variables_are_zero(
            derivative_link_model
        )
        return derivative_link_model

    @profile
    def derivative_link_model_no_acc(
        self, max_derivative: Derivatives
    ) -> cas.Expression:
        """
        Layout for prediction horizon 5
        Slots are matrices of |controlled variables| x |controlled variables|
        | vt0 | vt1 | vt2 | jt0 | jt1 | jt2 | jt3 | jt4 |
        |-----------------------------------------------|
        | -1  |     |     |dt**2|     |     |     |     |       -vc - ac*dt = -v0 + j0*dt**2
        |  2  | -1  |     |     |dt**2|     |     |     |                vc = -v1 + 2*v0 + j1*dt**2
        | -1  |  2  | -1  |     |     |dt**2|     |     |                 0 = -vt + 2*vt-1 - vt-2 + jt*dt**2
        |     | -1  |  2  |     |     |     |dt**2|     |                 0 =   0 + 2*vt-1 - vt-2 + jt*dt**2
        |     |     | -1  |     |     |     |     |dt**2|                 0 =   0 +    0   - vt-2 + jt*dt**2
        |-----------------------------------------------|
        vt = vt-1 + at * dt     <=>  vt/dt - vt-1/dt = at
        at = at-1 + jt * dt     <=>  at/dt - at-1/dt = jt

        vt = vt-1 + (at-1 + jt * dt) * dt
        vt = vt-1 + at-1*dt + jt * dt**2
        vt = vt-1 + (vt-1/dt - vt-2/dt)*dt + jt * dt**2
        vt = vt-1 + vt-1 - vt-2 + jt*dt**2

        0 = -v1 + 2*v0 - vc + j1*dt**2
        vc = -v1 + 2*v0 + j1*dt**2

        a0/dt - ac/dt = j0
        (v0/dt - vc/dt)/dt - ac/dt = j0
        v0/dt**2 - vc/dt**2 - ac/dt = j0
        -vc/dt**2 - ac/dt = -v0/dt**2 + j0
        -vc - ac*dt = -v0 + j0*dt**2

        v0 = vc + ac*dt + j0*dt**2
        vc = -v1 + 2*v0 + j1*dt**2
        v1 = -vc + 2*v0 + j1*dt**2
        """
        n_vel = self.number_of_free_variables * (self.config.prediction_horizon - 2)
        n_jerk = self.number_of_free_variables * self.config.prediction_horizon
        model = cas.Expression.zeros(rows=n_jerk, columns=n_jerk + n_vel)
        pre_previous = -cas.Expression.eye(n_vel)
        same = pre_previous
        previous = -2 * pre_previous
        j_same = cas.Expression.eye(n_jerk) * self.config.mpc_dt**2
        model[: -self.number_of_free_variables * 2, :n_vel] += pre_previous
        model[
            self.number_of_free_variables : -self.number_of_free_variables, :n_vel
        ] += previous
        model[self.number_of_free_variables * 2 :, :n_vel] += same
        model[:, n_vel:] = j_same
        return model

    def _remove_rows_columns_where_variables_are_zero(
        self, derivative_link_model: cas.Expression
    ) -> cas.Expression:
        if np.prod(derivative_link_model.shape) == 0:
            return derivative_link_model
        row_ids = []
        end = 0
        for derivative in Derivatives.range(
            Derivatives.velocity, self.config.max_derivative - 1
        ):
            last_non_zero_variable = self.config.prediction_horizon - (
                self.config.max_derivative - derivative - 1
            )
            start = end + self.number_of_free_variables * last_non_zero_variable
            end += self.number_of_free_variables * self.config.prediction_horizon
            row_ids.extend(range(start, end))
        derivative_link_model.remove(row_ids, [])
        return derivative_link_model

    @profile
    def equality_constraint_model(self) -> Tuple[cas.Expression, cas.Expression]:
        """
        |   t1   |   t2   |   t1   |   t2   |   t1   |   t2   |   t1   |   t2   | prediction horizon
        |v1 v2 v3|v1 v2 v3|a1 a2 a3|a1 a2 a3|j1 j2 j3|j1 j2 j3|s1 s2 s3|s1 s2 s3| free variables / slack
        |-----------------------------------------------------------------------|
        |  J1*sp |  J1*sp |  J2*sp |  J2*sp |  J3*sp | J3*sp  | sp*ch  | sp*ch  |
        |-----------------------------------------------------------------------|

        explicit no acc
        |   t1   |   t2   |   t1   |   t2   |   t1   |   t2   | prediction horizon
        |v1 v2 v3|v1 v2 v3|j1 j2 j3|j1 j2 j3|s1 s2 s3|s1 s2 s3| free variables / slack
        |-----------------------------------------------------|
        |  J1*sp |  J1*sp |  J3*sp | J3*sp  | sp*ch  | sp*ch  |
        |-----------------------------------------------------|
        """
        if (
            not self.config.qp_formulation.has_explicit_acc_variables
            and self.config.qp_formulation.has_explicit_jerk_variables
        ):
            if len(self.constraint_collection.eq_constraints) > 0:
                model = cas.Expression.zeros(
                    len(self.constraint_collection.eq_constraints),
                    self.number_of_non_slack_columns,
                )
                J_eq = (
                    cas.Expression(self.equality_constraint_expressions()).jacobian(
                        variables=self.get_free_variable_symbols(Derivatives.position)
                    )
                    * self.config.mpc_dt
                )
                J_hstack = cas.hstack(
                    [J_eq for _ in range(self.config.prediction_horizon - 2)]
                )
                # set jacobian entry to 0 if control horizon shorter than prediction horizon
                horizontal_offset = J_hstack.shape[1]
                model[:, horizontal_offset * 0 : horizontal_offset * 1] = J_hstack

                # slack variable for total error
                slack_model = cas.diag(
                    cas.Expression(
                        [
                            self.config.mpc_dt
                            for c in self.constraint_collection.eq_constraints
                        ]
                    )
                )
                return model, slack_model
        else:
            if self.config.qp_formulation.is_implicit:
                max_derivative = Derivatives.velocity
            else:
                max_derivative = self.config.max_derivative
            if len(self.constraint_collection.eq_constraints) > 0:
                model = cas.Expression.zeros(
                    len(self.constraint_collection.eq_constraints),
                    self.number_of_non_slack_columns,
                )
                for derivative in Derivatives.range(
                    Derivatives.position, max_derivative - 1
                ):
                    J_eq = (
                        cas.Expression(self.equality_constraint_expressions()).jacobian(
                            variables=self.get_free_variable_symbols(derivative)
                        )
                        * self.config.mpc_dt
                    )

                    if (
                        self.config.qp_formulation.is_explicit
                        or not self.config.qp_formulation.is_mpc
                    ):
                        J_hstack = cas.hstack(
                            [J_eq for _ in range(self.config.prediction_horizon)]
                        )
                        horizontal_offset = J_hstack.shape[1]
                        model[
                            :,
                            horizontal_offset
                            * derivative : horizontal_offset
                            * (derivative + 1),
                        ] = J_hstack
                    else:
                        J_hstack = cas.hstack(
                            [J_eq for _ in range(self.config.prediction_horizon - 2)]
                        )
                        horizontal_offset = J_hstack.shape[1]
                        model[:, horizontal_offset * 0 : horizontal_offset * 1] = (
                            J_hstack
                        )

                # slack variable for total error
                slack_model = cas.diag(
                    cas.Expression(
                        [
                            self.config.mpc_dt
                            for c in self.constraint_collection.eq_constraints
                        ]
                    )
                )
                return model, slack_model
        return cas.Expression(), cas.Expression()

    @profile
    def construct_expression(
        self,
    ) -> Union[cas.Expression, Tuple[cas.Expression, cas.Expression]]:
        max_derivative = Derivatives.velocity
        derivative_link_model = cas.Expression()
        if self.config.qp_formulation.is_mpc:
            if self.config.qp_formulation.is_explicit:
                max_derivative = self.config.max_derivative
                derivative_link_model = self.derivative_link_model(max_derivative)
                derivative_link_model = (
                    self._remove_columns_columns_where_variables_are_zero(
                        derivative_link_model, max_derivative
                    )
                )
            elif (
                not self.config.qp_formulation.has_explicit_acc_variables
                and self.config.qp_formulation.has_explicit_jerk_variables
            ):
                max_derivative = Derivatives.velocity
                derivative_link_model = self.derivative_link_model_no_acc(
                    self.config.max_derivative
                )
        equality_constraint_model, equality_constraint_slack_model = (
            self.equality_constraint_model()
        )
        if self.config.qp_formulation.has_explicit_jerk_variables:
            equality_constraint_model = (
                self._remove_columns_columns_where_variables_are_zero(
                    equality_constraint_model, max_derivative
                )
            )
        vel_constr_model, vel_constr_slack_model = self.velocity_constraint_model()

        model_parts = []
        slack_model_parts = []
        if len(derivative_link_model) > 0:
            model_parts.append(derivative_link_model)
        if len(equality_constraint_model) > 0:
            model_parts.append(equality_constraint_model)
            slack_model_parts.append(equality_constraint_slack_model)
        if len(vel_constr_model) > 0:
            model_parts.append(vel_constr_model)
            slack_model_parts.append(vel_constr_slack_model)

        if len(model_parts) == 0:
            # if there are no eq constraints, make an empty matrix with the right columns to prevent stacking issues
            model = cas.Expression(np.empty((0, self.number_of_non_slack_columns)))
        else:
            model = cas.vstack(model_parts)
        slack_model = cas.diag_stack(slack_model_parts)

        slack_model = cas.vstack(
            [
                cas.Expression.zeros(
                    derivative_link_model.shape[0], slack_model.shape[1]
                ),
                slack_model,
            ]
        )
        # model = self._remove_columns_columns_where_variables_are_zero(model, max_derivative)
        return model, slack_model


@dataclass
class InequalityModel(ProblemDataPart):
    """
    order:
        derivative model (optional)
        neq integral constraints
        neq vel constraints
    """

    @property
    def number_of_free_variables(self):
        return len(self.degrees_of_freedom)

    @property
    def number_of_non_slack_columns(self) -> int:
        if self.config.qp_formulation.is_explicit:
            return (
                self.number_of_free_variables
                * self.config.prediction_horizon
                * self.config.max_derivative
            )
        elif self.config.qp_formulation.is_implicit:
            return self.number_of_free_variables * (self.config.prediction_horizon - 2)
        elif (
            not self.config.qp_formulation.has_explicit_acc_variables
            and self.config.qp_formulation.has_explicit_jerk_variables
        ):
            return (
                self.number_of_free_variables * (self.config.prediction_horizon - 2)
                + self.number_of_free_variables * self.config.prediction_horizon
            )
        return (
            self.number_of_free_variables
            * self.config.prediction_horizon
            * self.config.max_derivative
        )

    @memoize
    def num_position_limits(self):
        return self.number_of_free_variables - self.num_of_continuous_joints()

    @memoize
    def num_of_continuous_joints(self):
        return len([v for v in self.degrees_of_freedom if not v.has_position_limits()])

    def inequality_constraint_expressions(self) -> List[cas.Expression]:
        return self._sorter(
            {c.name: c.expression for c in self.constraint_collection.neq_constraints}
        )[0]

    def get_derivative_constraint_expressions(self, derivative: Derivatives):
        return self._sorter(
            {
                c.name: c.expression
                for c in self.constraint_collection.derivative_constraints
                if c.derivative == derivative
            }
        )[0]

    def get_free_variable_symbols(self, order: Derivatives):
        return self._sorter(
            {
                v.variables.position.name: v.variables.data[order]
                for v in self.degrees_of_freedom
            }
        )[0]

    def velocity_constraint_model(self) -> Tuple[cas.Expression, cas.Expression]:
        """
        model
        |   t1   |   t2   |   t3   |   t1   |   t2   |   t3   |   t1   |   t2   |   t3   | prediction horizon
        |v1 v2 v3|v1 v2 v3|v1 v2 v3|a1 a2 a3|a1 a2 a3|a1 a2 a3|j1 j2 j3|j1 j2 j3|j1 j2 j3| free variables
        |--------------------------------------------------------------------------------|
        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |        |
        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |        |
        |--------------------------------------------------------------------------------|
        |        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |
        |        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |
        |--------------------------------------------------------------------------------|

        slack model
        |   t1 |   t2 | prediction horizon
        |s1 s2 |s1 s2 | slack
        |------|------|
        |sp    |      | vel constr 1
        |   sp |      | vel constr 2
        |-------------|
        |      |sp    | vel constr 1
        |      |   sp | vel constr 2
        |-------------|
        """
        number_of_vel_rows = len(self.velocity_constraints) * (
            self.config.prediction_horizon - 2
        )
        if number_of_vel_rows > 0:
            expressions = cas.Expression(
                self.get_derivative_constraint_expressions(Derivatives.velocity)
            )
            parts = []
            for derivative in Derivatives.range(
                Derivatives.position, self.config.max_derivative - 1
            ):
                if (
                    derivative == Derivatives.velocity
                    and not self.config.qp_formulation.has_explicit_acc_variables
                ):
                    continue
                if (
                    derivative == Derivatives.acceleration
                    and not self.config.qp_formulation.has_explicit_jerk_variables
                ):
                    continue
                J_vel = (
                    expressions.jacobian(
                        variables=self.get_free_variable_symbols(derivative),
                    )
                    * self.config.mpc_dt
                )
                missing_variables = self.config.max_derivative - derivative - 1
                eye = cas.Expression.eye(self.config.prediction_horizon)[
                    :-2, : self.config.prediction_horizon - missing_variables
                ]
                J_vel_limit_block = eye.kron(J_vel)
                parts.append(J_vel_limit_block)

            # constraint slack
            model = cas.hstack(parts)
            num_slack_variables = sum(
                self.control_horizon for c in self.velocity_constraints
            )
            slack_model = cas.Expression.eye(num_slack_variables) * self.config.mpc_dt
            return model, slack_model
        return cas.Expression(), cas.Expression()

    def acceleration_constraint_model(self) -> Tuple[cas.Expression, cas.Expression]:
        """
        same structure as vel constraint model
        task acceleration = Jd_q * qd + (J_q + Jd_qd) * qdd + J_qd * qddd
        """
        # FIXME no test case for this so probably buggy
        number_of_acc_rows = (
            len(self.acceleration_constraints) * self.config.prediction_horizon
        )
        if number_of_acc_rows > 0:
            expressions = cas.Expression(
                self.get_derivative_constraint_expressions(Derivatives.acceleration)
            )
            assert self.config.max_derivative >= Derivatives.jerk
            model = cas.Expression.zeros(
                number_of_acc_rows, self.number_of_non_slack_columns
            )
            J_q = (
                expressions.jacobian(
                    variables=self.get_free_variable_symbols(Derivatives.position),
                )
                * self.config.mpc_dt
            )
            Jd_q = (
                expressions.jacobian_dot(
                    variables=self.get_free_variable_symbols(Derivatives.position),
                    variables_dot=self.get_free_variable_symbols(Derivatives.velocity),
                )
                * self.config.mpc_dt
            )
            J_qd = (
                expressions.jacobian(
                    variables=self.get_free_variable_symbols(Derivatives.velocity),
                )
                * self.config.mpc_dt
            )
            Jd_qd = (
                expressions.jacobian_dot(
                    variables=self.get_free_variable_symbols(Derivatives.velocity),
                    variables_dot=self.get_free_variable_symbols(
                        Derivatives.acceleration
                    ),
                )
                * self.config.mpc_dt
            )
            J_vel_block = cas.Expression.eye(self.config.prediction_horizon).kron(Jd_q)
            J_acc_block = cas.Expression.eye(self.config.prediction_horizon).kron(
                J_q + Jd_qd
            )
            J_jerk_block = cas.Expression.eye(self.config.prediction_horizon).kron(J_qd)
            horizontal_offset = (
                self.number_of_free_variables * self.config.prediction_horizon
            )
            model[:, :horizontal_offset] = J_vel_block
            model[:, horizontal_offset : horizontal_offset * 2] = J_acc_block
            model[:, horizontal_offset * 2 : horizontal_offset * 3] = J_jerk_block

            # delete rows if control horizon of constraint shorter than prediction horizon
            rows_to_delete = []
            for t in range(self.config.prediction_horizon):
                for i, c in enumerate(self.velocity_constraints):
                    v_index = i + (t * len(self.velocity_constraints))
                    if t + 1 > self.control_horizon:
                        rows_to_delete.append(v_index)
            model.remove(rows_to_delete, [])

            # slack model
            num_slack_variables = sum(
                self.control_horizon for c in self.acceleration_constraints
            )
            slack_model = cas.Expression.eye(num_slack_variables) * self.config.mpc_dt
            return model, slack_model
        return cas.Expression(), cas.Expression()

    @profile
    def inequality_constraint_model(
        self, max_derivative: Derivatives
    ) -> Tuple[cas.Expression, cas.Expression]:
        """
        |   t1   |   t2   |   t1   |   t2   |   t1   |   t2   |   t1   |   t2   | prediction horizon
        |v1 v2 v3|v1 v2 v3|a1 a2 a3|a1 a2 a3|j1 j2 j3|j1 j2 j3|s1 s2 s3|s1 s2 s3| free variables / slack
        |-----------------------------------------------------------------------|
        |  J1*sp |  J1*sp |  J2*sp |  J2*sp |  J3*sp | J3*sp  | sp*ch  | sp*ch  |
        |-----------------------------------------------------------------------|
        """
        if (
            not self.config.qp_formulation.has_explicit_acc_variables
            and self.config.qp_formulation.has_explicit_jerk_variables
        ):
            if len(self.constraint_collection.neq_constraints) > 0:
                model = cas.Expression.zeros(
                    len(self.constraint_collection.neq_constraints),
                    self.number_of_non_slack_columns,
                )
                J_neq = (
                    cas.Expression(self.inequality_constraint_expressions()).jacobian(
                        variables=self.get_free_variable_symbols(Derivatives.position),
                    )
                    * self.config.mpc_dt
                )
                J_hstack = cas.hstack(
                    [J_neq for _ in range(self.config.prediction_horizon - 2)]
                )
                # set jacobian entry to 0 if control horizon shorter than prediction horizon
                horizontal_offset = J_hstack.shape[1]
                model[:, horizontal_offset * 0 : horizontal_offset * 1] = J_hstack

                # slack variable for total error
                slack_model = cas.diag(
                    cas.Expression(
                        [
                            self.config.mpc_dt
                            for c in self.constraint_collection.neq_constraints
                        ]
                    )
                )
                return model, slack_model
        else:
            if self.config.qp_formulation.is_implicit:
                max_derivative = Derivatives.velocity
            else:
                max_derivative = self.config.max_derivative
        if len(self.constraint_collection.neq_constraints) > 0:
            model = cas.Expression.zeros(
                len(self.constraint_collection.neq_constraints),
                self.number_of_non_slack_columns,
            )
            for derivative in Derivatives.range(
                Derivatives.position, max_derivative - 1
            ):
                J_neq = (
                    cas.Expression(self.inequality_constraint_expressions()).jacobian(
                        variables=self.get_free_variable_symbols(derivative),
                    )
                    * self.config.mpc_dt
                )
                if (
                    self.config.qp_formulation.is_explicit
                    or not self.config.qp_formulation.is_mpc
                ):
                    J_hstack = cas.hstack(
                        [J_neq for _ in range(self.config.prediction_horizon)]
                    )
                    horizontal_offset = J_hstack.shape[1]
                    model[
                        :,
                        horizontal_offset
                        * derivative : horizontal_offset
                        * (derivative + 1),
                    ] = J_hstack
                else:
                    J_hstack = cas.hstack(
                        [J_neq for _ in range(self.config.prediction_horizon - 2)]
                    )
                    horizontal_offset = J_hstack.shape[1]
                    model[:, horizontal_offset * 0 : horizontal_offset * 1] = J_hstack

            # slack variable for total error
            slack_model = cas.diag(
                cas.Expression(
                    [
                        self.config.mpc_dt
                        for c in self.constraint_collection.neq_constraints
                    ]
                )
            )
            return model, slack_model
        return cas.Expression(), cas.Expression()

    def implicit_pos_limits(self) -> Tuple[cas.Expression, cas.Expression]:
        """
        pk = pk-1 + vk*dt
            p0 = pc + v0*dt
            pk/dt - pk-1/dt = vk
        vk = vk-1 + ak*dt
        ak = (vk - vk-1)/dt
        jk = (vk - 2vk-1 + vk-2)/dt**2

        Layout for prediction horizon 4
        Slots are matrices of |controlled variables| x |controlled variables|
        |  vt0   |  vt1   |  vt2   |  vt3   |
        |-----------------------------------|
        |  1*dt  |        |        |        |       pt0 - ptc = vt0*dt
        |  1*dt  |  1*dt  |        |        |       pt1 - ptc = vt0*dt + vt1*dt
        |  1*dt  |  1*dt  |  1*dt  |        |       pt2 - ptc = vt0*dt + vt1*dt + vt2*dt
        |  1*dt  |  1*dt  |  1*dt  |  1*dt  |       pt3 - ptc = vt0*dt + vt1*dt + vt2*dt + vt3*dt
        |-----------------------------------|

        :return:
        """
        n_vel = self.number_of_free_variables * (self.config.prediction_horizon - 2)
        model = cas.Expression.tri(n_vel) * self.config.mpc_dt
        slack_model = cas.Expression.zeros(
            model.shape[0], self.number_ineq_slack_variables
        )
        return model, slack_model

    def implicit_model(
        self, max_derivative: Derivatives
    ) -> Tuple[cas.Expression, cas.Expression]:
        """
        ak = (vk - vk-1)/dt
        jk = (vk - 2vk-1 + vk-2)/dt**2

        vt0 = vtc + at0 * cdt
        vt1 = vt0 + at1 * mdt
        vt = vt-1 + at * mdt

        at = vt-1 + jt * mdt

        Layout for prediction horizon 6
        Slots are matrices of |controlled variables| x |controlled variables|
        |  vt0   |  vt1   |  vt2   |  vt3   |
        |-----------------------------------|
        |  1/dt  |        |        |        |               vtc/cdt + at0 = vt0/cdt                   vtc/dt + a_min <= vt0/dt <= vtc/dt + a_max
        | -1/dt  |  1/dt  |        |        |                        at1 = (vt1 - vt0)/dt
        |        | -1/dt  |  1/dt  |        |                        at2 = (vt2 - vt1)/mdt
        |        |        | -1/dt  | 1/dt   |                        at3 = (vt3 - vt2)/mdt
        |===================================|
        | 1/dt**2|        |        |        |   vtc/dt**2 + atc/dt + jt0 = vt0/dt**2                vtc/dt**2 + atc/dt + j_min <=    vt0/dt**2     <= vtc/dt**2 + atc/dt + j_max
        |-2/dt**2| 1/dt**2|        |        |          - vtc/dt**2 + jt1 = (vt1 - 2vt0)/dt**2           (- vtc)/dt**2 + j_min <= (vt1 - 2vt0)/dt**2 <= (- vtc)/dt**2 + j_max
        | 1/dt**2|-2/dt**2| 1/dt**2|        |                        jt2 = (vt2 - 2vt1 + vt0)/dt**2
        |        | 1/dt**2|-2/dt**2| 1/dt**2|                        jt3 = (vt3 - 2vt2 + vt1)/dt**2
        |        |        | 1/dt**2|-2/dt**2|                        jt4 = (- 2vt3 + vt2)/dt**2
        |        |        |        | 1/dt**2|                        jt5 = (vt3)/dt**2
        |-----------------------------------|

        :param max_derivative:
        :return:
        """
        n_vel = self.number_of_free_variables * (self.config.prediction_horizon - 2)
        n_jerk = self.number_of_free_variables * (self.config.prediction_horizon)
        if max_derivative >= Derivatives.acceleration:
            # previous = cas.Expression.eye(self.number_of_free_variables * (self.config.prediction_horizon)) / self.config.mpc_dt
            # same = -cas.Expression.eye(self.number_of_free_variables * (self.config.prediction_horizon - 1)) / self.config.mpc_dt
            # A_acc = previous
            # A_acc[self.number_of_free_variables:, :-self.number_of_free_variables] += same
            # rows_to_delete = []
            # for i in range(self.config.prediction_horizon):
            #     for v_i, v in enumerate(self.free_variables):
            #         idx = i * len(self.free_variables) + v_i
            #         a_min = v.get_lower_limit(Derivatives.acceleration)
            #         a_max = v.get_upper_limit(Derivatives.acceleration)
            #         if (np.isinf(a_min) or cas.is_inf(a_min)) and (np.isinf(a_max) or cas.is_inf(a_max)):
            #             rows_to_delete.append(idx)
            # A_acc.remove(rows_to_delete, [])
            model = cas.Expression.zeros(rows=n_jerk, columns=n_vel)
            pre_previous = cas.Expression.eye(n_vel) / self.config.mpc_dt**2
            previous = -2 * cas.Expression.eye(n_vel) / self.config.mpc_dt**2
            same = pre_previous
            model[: -self.number_of_free_variables * 2, :] += pre_previous
            model[
                self.number_of_free_variables : -self.number_of_free_variables, :
            ] += previous
            model[self.number_of_free_variables * 2 :, :] += same
        else:
            model = cas.Expression()
        slack_model = cas.Expression.zeros(
            model.shape[0], self.number_ineq_slack_variables
        )
        return model, slack_model

    @profile
    def construct_expression(
        self,
    ) -> Union[cas.Expression, Tuple[cas.Expression, cas.Expression]]:
        model_parts = []
        slack_model_parts = []

        if self.config.qp_formulation.has_explicit_pos_limits:
            pos_model, pos_slack_model = self.implicit_pos_limits()
            if len(pos_model) > 0:
                model_parts.append(pos_model)
                slack_model_parts.append(pos_slack_model)
        if self.config.qp_formulation.is_implicit:
            max_derivative = Derivatives.velocity
            derivative_model, derivative_model_slack = self.implicit_model(
                self.config.max_derivative
            )
        else:
            max_derivative = self.config.max_derivative
            derivative_model, derivative_model_slack = (
                cas.Expression(),
                cas.Expression(),
            )

        inequality_model, inequality_slack_model = self.inequality_constraint_model(
            max_derivative
        )
        vel_constr_model, vel_constr_slack_model = self.velocity_constraint_model()

        # derivative model
        if len(derivative_model) > 0:
            model_parts.append(derivative_model)
            slack_model_parts.append(derivative_model_slack)
        # neq integral constraints
        if len(vel_constr_model) > 0:
            model_parts.append(vel_constr_model)
            slack_model_parts.append(vel_constr_slack_model)
        # neq vel constraints
        if len(inequality_model) > 0:
            model_parts.append(inequality_model)
            slack_model_parts.append(inequality_slack_model)

        combined_model = cas.vstack(model_parts)
        combined_slack_model = cas.diag_stack(slack_model_parts)
        return combined_model, combined_slack_model


@dataclass
class GiskardToQPAdapter:
    world_state_symbols: List[cas.FloatVariable]
    life_cycle_symbols: List[cas.FloatVariable]
    external_collision_symbols: List[cas.FloatVariable]
    self_collision_symbols: List[cas.FloatVariable]
    auxiliary_variables: List[cas.FloatVariable]

    degrees_of_freedom: List[DegreeOfFreedom]
    constraint_collection: ConstraintCollection
    config: QPControllerConfig
    sparse: bool = field(default=True)

    qp_data_raw: QPData = None

    compute_nI_I: bool = True
    _nAi_Ai_cache: dict = field(default_factory=dict)

    """
    Takes free variables and constraints and converts them to a QP problem in the following format, depending on the
    class attributes:
    min_x 0.5 x^T H x + g^T x
    s.t.  lb <= x <= ub     (box constraints)
          Edof x <= bE_dof          (equality constraints)
          Eslack x <= bE_slack        (equality constraints)
          lbA <= Adof x <= ubA_dof  (lower/upper inequality constraints)
          lbA <= Aslack x <= ubA_slack  (lower/upper inequality constraints)
    """

    def __post_init__(self):
        kwargs = {
            "degrees_of_freedom": self.degrees_of_freedom,
            "constraint_collection": self.constraint_collection,
            "config": self.config,
        }
        self.weights = Weights(**kwargs)
        self.free_variable_bounds = FreeVariableBounds(**kwargs)
        self.equality_model = EqualityModel(**kwargs)
        self.equality_bounds = EqualityBounds(**kwargs)
        self.inequality_model = InequalityModel(**kwargs)
        self.inequality_bounds = InequalityBounds(**kwargs)

        quadratic_weights, linear_weights = self.weights.construct_expression()
        box_lower_constraints, box_upper_constraints = (
            self.free_variable_bounds.construct_expression()
        )
        eq_matrix_dofs, self.eq_matrix_slack = (
            self.equality_model.construct_expression()
        )
        eq_bounds = self.equality_bounds.construct_expression()
        neq_matrix_dofs, self.neq_matrix_slack = (
            self.inequality_model.construct_expression()
        )
        neq_lower_bounds, neq_upper_bounds = (
            self.inequality_bounds.construct_expression()
        )
        self.general_qp_to_specific_qp(
            quadratic_weights=quadratic_weights,
            linear_weights=linear_weights,
            box_lower_constraints=box_lower_constraints,
            box_upper_constraints=box_upper_constraints,
            eq_matrix_dofs=eq_matrix_dofs,
            eq_matrix_slack=self.eq_matrix_slack,
            eq_bounds=eq_bounds,
            neq_matrix_dofs=neq_matrix_dofs,
            neq_matrix_slack=self.neq_matrix_slack,
            neq_lower_bounds=neq_lower_bounds,
            neq_upper_bounds=neq_upper_bounds,
        )

    def __hash__(self):
        return hash(id(self))

    @property
    def num_eq_constraints(self) -> int:
        return len(self.constraint_collection.eq_constraints)

    @property
    def num_neq_constraints(self) -> int:
        return len(self.constraint_collection.neq_constraints)

    @property
    def num_free_variable_constraints(self) -> int:
        return len(self.degrees_of_freedom)

    @property
    def num_eq_slack_variables(self) -> int:
        return self.eq_matrix_slack.shape[1]

    @property
    def num_neq_slack_variables(self) -> int:
        return self.neq_matrix_slack.shape[1]

    @property
    def num_slack_variables(self) -> int:
        return self.num_eq_slack_variables + self.num_neq_slack_variables

    @property
    def num_non_slack_variables(self) -> int:
        return self.num_free_variable_constraints - self.num_slack_variables

    def general_qp_to_specific_qp(
        self,
        quadratic_weights: cas.Expression,
        linear_weights: cas.Expression,
        box_lower_constraints: cas.Expression,
        box_upper_constraints: cas.Expression,
        eq_matrix_dofs: cas.Expression,
        eq_matrix_slack: cas.Expression,
        eq_bounds: cas.Expression,
        neq_matrix_dofs: cas.Expression,
        neq_matrix_slack: cas.Expression,
        neq_lower_bounds: cas.Expression,
        neq_upper_bounds: cas.Expression,
    ):
        raise NotImplementedError()

    def evaluate(
        self,
        world_state: np.ndarray,
        life_cycle_state: np.ndarray,
        external_collision_data: np.ndarray,
        self_collision_data: np.ndarray,
        auxiliary_variables: np.ndarray,
    ) -> QPData:
        raise NotImplementedError()

    @profile
    def _direct_limit_model(
        self,
        dimensions_after_zero_filter: int,
        Ai_inf_filter: Optional[np.ndarray] = None,
        two_sided: bool = False,
    ) -> Union[np.ndarray, sp.csc_matrix]:
        """
        These models are often identical, yet the computation is expensive. Caching to the rescue
        """
        if Ai_inf_filter is None:
            key = hash(dimensions_after_zero_filter)
        else:
            key = hash((dimensions_after_zero_filter, Ai_inf_filter.tobytes()))
        if key not in self._nAi_Ai_cache:
            nI_I = self._cached_eyes(dimensions_after_zero_filter, two_sided)
            if Ai_inf_filter is None:
                self._nAi_Ai_cache[key] = nI_I
            else:
                self._nAi_Ai_cache[key] = nI_I[Ai_inf_filter]
        return self._nAi_Ai_cache[key]

    @memoize
    def _cached_eyes(
        self, dimensions: int, two_sided: bool = False
    ) -> Union[np.ndarray, sp.csc_matrix]:
        if self.sparse:
            from scipy import sparse as sp

            if two_sided:
                data = np.ones(dimensions, dtype=float)
                row_indices = np.arange(dimensions)
                col_indices = np.arange(dimensions + 1)
                return sp.csc_matrix((data, row_indices, col_indices))
            else:
                d2 = dimensions * 2
                data = np.ones(d2, dtype=float)
                data[::2] *= -1
                r1 = np.arange(dimensions)
                r2 = np.arange(dimensions, d2)
                row_indices = np.empty((d2,), dtype=int)
                row_indices[0::2] = r1
                row_indices[1::2] = r2
                col_indices = np.arange(0, d2 + 1, 2)
                return sp.csc_matrix((data, row_indices, col_indices))
        else:
            I = np.eye(dimensions)
            if two_sided:
                return I
            else:
                return np.concatenate([-I, I])
