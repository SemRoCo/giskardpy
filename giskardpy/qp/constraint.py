from dataclasses import dataclass, field

import semantic_digital_twin.spatial_types.spatial_types as cas
from semantic_digital_twin.spatial_types.derivatives import Derivatives


@dataclass
class BaseConstraint:
    """
    Defines a (slack-relaxed) constraint on expression for a quadratic program.
    """

    name: str

    expression: cas.SymbolicScalar

    quadratic_weight: cas.ScalarData

    linear_weight: cas.ScalarData


@dataclass
class IntegralConstraint(BaseConstraint):
    normalization_factor: cas.ScalarData
    """
    This value is important to make constraints with different units comparable.
    The meaning depends on derivative.
    If the derivative is position, the normalization factor is rough velocity with which the expression can change.
    For example:
        - If you have a joint position constraint, the normalization factor should be the joint velocity limit.
        - If you have a cartesian position constraint, the normalization factor should be the cartesian velocity limit.
    In practice, use joint limits from the URDF for joint space constraints and define two values for cartesian constraints:
        - a m/s limit for translation
        - a rad/s value for rotation
    """

    def normalized_weight(self, control_horizon: int) -> cas.Expression:
        return self.quadratic_weight * (
            1 / (self.normalization_factor**2 * control_horizon)
        )

    def _apply_cap(
        self, value: cas.Expression, dt: float, control_horizon: int
    ) -> cas.Expression:
        return cas.limit(
            value,
            -self.normalization_factor * dt * control_horizon,
            self.normalization_factor * dt * control_horizon,
        )


@dataclass
class InequalityConstraint(IntegralConstraint):
    """
    Adds
    capped_lower_error(lower_error) <= expression * control_horizon + slack <= capped_upper_error(upper_error)
    lower_slack_limit <= slack <= upper_slack_limit
    """

    lower_error: cas.ScalarData
    upper_error: cas.ScalarData

    lower_slack_limit: cas.ScalarData
    upper_slack_limit: cas.ScalarData

    def capped_lower_error(self, dt: float, control_horizon: int) -> cas.Expression:
        return self._apply_cap(self.lower_error, dt, control_horizon)

    def capped_upper_error(self, dt: float, control_horizon: int) -> cas.Expression:
        return self._apply_cap(self.upper_error, dt, control_horizon)


@dataclass
class EqualityConstraint(IntegralConstraint):
    """ """

    bound: cas.ScalarData

    lower_slack_limit: cas.ScalarData
    upper_slack_limit: cas.ScalarData

    def capped_bound(self, dt: float, control_horizon: int) -> cas.Expression:
        return self._apply_cap(self.bound, dt, control_horizon)


@dataclass
class DerivativeConstraint(BaseConstraint):
    derivative: Derivatives = field(kw_only=True)
    """
    The constraint will be applied to the derivative of the expression.
    Position constraints are implemented by constraining the integral of the expressions' derivative over a prediction horizon.
    All other constraints are applied directly to that derivative of the expression.
    As a result, position constraints are cheaper, as they only require a single constraint.
    """

    normalization_factor: cas.ScalarData = field(kw_only=True)
    """
    This value is important to make constraints with different units comparable.
    The meaning depends on derivative.
    If the derivative is position, the normalization factor is rough velocity with which the expression can change.
    For example:
        - If you have a joint position constraint, the normalization factor should be the joint velocity limit.
        - If you have a cartesian position constraint, the normalization factor should be the cartesian velocity limit.
    For other derivatives, the normalization factor is the same unit as the expression.
    For example:
        - Joint velocity constraint -> joint velocity limit
        - Cartesian velocity constraint -> cartesian velocity limit
    .. Warning: This number is different from the bounds of the expression. 
                If you want to enforce a bound below the actual limit, the normalization factor should still be the true limit.
    In practice, use joint limits from the URDF for joint space constraints and define two values for cartesian constraints:
        - a m/s limit for translation
        - a rad/s value for rotation
    """

    def normalized_weight(self) -> float:
        return self.quadratic_weight * (1 / self.normalization_factor) ** 2


@dataclass
class DerivativeInequalityConstraint(DerivativeConstraint):
    lower_limit: cas.ScalarData
    upper_limit: cas.ScalarData

    lower_slack_limit: cas.ScalarData
    upper_slack_limit: cas.ScalarData


@dataclass
class DerivativeEqualityConstraint(DerivativeConstraint):
    bound: cas.ScalarData
    lower_slack_limit: cas.ScalarData
    upper_slack_limit: cas.ScalarData


@dataclass
class ODEConstraint(DerivativeConstraint):
    ode_function: cas.SymbolicScalar
    lower_slack_limit: cas.ScalarData
    upper_slack_limit: cas.ScalarData
