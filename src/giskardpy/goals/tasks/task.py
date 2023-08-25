from typing import Optional, List, Union, Dict
import giskardpy.casadi_wrapper as cas
from giskardpy.exceptions import GiskardException, ConstraintInitalizationException
from giskardpy.goals.monitors.monitors import AlwaysOne, AlwaysZero, Monitor
from giskardpy.qp.constraint import EqualityConstraint, InequalityConstraint, DerivativeInequalityConstraint


class Task:
    """
    Tasks are a set of constraints with the same predicates.
    """
    eq_constraints: Dict[str, EqualityConstraint]
    neq_constraints: Dict[str, InequalityConstraint]
    derivative_constraints: Dict[str, DerivativeInequalityConstraint]
    to_start: Monitor
    to_hold: Monitor
    to_end: Monitor
    name: Optional[str]

    def __init__(self, name: Optional[str],
                 to_start: Optional[Monitor] = None,
                 to_hold: Optional[Monitor] = None,
                 to_end: Optional[Monitor] = None):
        if name is None:
            self.name = str(self.__class__.__name__)
        else:
            self.name = name
        self.eq_constraints = {}
        self.neq_constraints = {}
        self.derivative_constraints = {}
        if to_start is None:
            self.to_start = AlwaysOne(crucial=False)
        else:
            self.to_start = to_start
        if to_hold is None:
            self.to_hold = AlwaysOne(crucial=False)
        else:
            self.to_hold = to_hold
        if to_end is None:
            self.to_end = AlwaysZero(crucial=False)
        else:
            self.to_end = to_end

    def __str__(self):
        return self.name

    def get_eq_constraints(self):
        constraints = []
        for constraint in self.eq_constraints.values():
            if self.to_start is not None:
                constraint.quadratic_weight *= self.to_start.get_state_expression()
            if self.to_hold is not None:
                constraint.quadratic_weight *= self.to_hold.get_state_expression()
            if self.to_end is not None:
                constraint.quadratic_weight *= (1 - self.to_end.get_state_expression())
            constraints.append(constraint)
        return constraints

    def add_equality_constraint(self,
                                reference_velocity: cas.symbol_expr_float,
                                equality_bound: cas.symbol_expr_float,
                                weight: cas.symbol_expr_float,
                                task_expression: cas.symbol_expr,
                                name: str = None,
                                lower_slack_limit: Optional[cas.symbol_expr_float] = None,
                                upper_slack_limit: Optional[cas.symbol_expr_float] = None,
                                control_horizon: Optional[int] = None):
        """
        Add a task constraint to the motion problem. This should be used for most constraints.
        It will not strictly stick to the reference velocity, but requires only a single constraint in the final
        optimization problem and is therefore faster.
        :param reference_velocity: used by Giskard to limit the error and normalize the weight, will not be strictly
                                    enforced.
        :param task_expression: defines the task function
        :param equality_bound: goal for the derivative of task_expression
        :param weight:
        :param name: give this constraint a name, required if you add more than one in the same goal
        :param lower_slack_limit: how much the lower error can be violated, don't use unless you know what you are doing
        :param upper_slack_limit: how much the upper error can be violated, don't use unless you know what you are doing
        """
        if task_expression.shape != (1, 1):
            raise GiskardException(f'expression must have shape (1, 1), has {task_expression.shape}')
        name = name if name else f'{len(self.eq_constraints)}'
        # name = str(self) + '/' + name
        # if name in self.eq_constraints:
        #     raise KeyError(f'A constraint with name \'{name}\' already exists. '
        #                    f'You need to set a name, if you add multiple constraints.')
        lower_slack_limit = lower_slack_limit if lower_slack_limit is not None else -float('inf')
        upper_slack_limit = upper_slack_limit if upper_slack_limit is not None else float('inf')
        self.eq_constraints[name] = EqualityConstraint(name=name,
                                                       expression=task_expression,
                                                       derivative_goal=equality_bound,
                                                       velocity_limit=reference_velocity,
                                                       quadratic_weight=weight,
                                                       lower_slack_limit=lower_slack_limit,
                                                       upper_slack_limit=upper_slack_limit,
                                                       control_horizon=control_horizon)

    def add_inequality_constraint(self,
                                  reference_velocity: cas.symbol_expr_float,
                                  lower_error: cas.symbol_expr_float,
                                  upper_error: cas.symbol_expr_float,
                                  weight: cas.symbol_expr_float,
                                  task_expression: cas.symbol_expr,
                                  name: Optional[str] = None,
                                  lower_slack_limit: Optional[cas.symbol_expr_float] = None,
                                  upper_slack_limit: Optional[cas.symbol_expr_float] = None,
                                  control_horizon: Optional[int] = None):
        """
        Add a task constraint to the motion problem. This should be used for most constraints.
        It will not strictly stick to the reference velocity, but requires only a single constraint in the final
        optimization problem and is therefore faster.
        :param reference_velocity: used by Giskard to limit the error and normalize the weight, will not be strictly
                                    enforced.
        :param lower_error: lower bound for the error of expression
        :param upper_error: upper bound for the error of expression
        :param weight:
        :param task_expression: defines the task function
        :param name: give this constraint a name, required if you add more than one in the same goal
        :param lower_slack_limit: how much the lower error can be violated, don't use unless you know what you are doing
        :param upper_slack_limit: how much the upper error can be violated, don't use unless you know what you are doing
        """
        if task_expression.shape != (1, 1):
            raise GiskardException(f'expression must have shape (1,1), has {task_expression.shape}')
        name = name if name else ''
        name = str(self) + "/" + name
        if name in self._inequality_constraints:
            raise KeyError(f'A constraint with name \'{name}\' already exists. '
                           f'You need to set a name, if you add multiple constraints.')
        lower_slack_limit = lower_slack_limit if lower_slack_limit is not None else -float('inf')
        upper_slack_limit = upper_slack_limit if upper_slack_limit is not None else float('inf')
        self._inequality_constraints[name] = InequalityConstraint(name=name,
                                                                  expression=task_expression,
                                                                  lower_error=lower_error,
                                                                  upper_error=upper_error,
                                                                  velocity_limit=reference_velocity,
                                                                  quadratic_weight=weight,
                                                                  lower_slack_limit=lower_slack_limit,
                                                                  upper_slack_limit=upper_slack_limit,
                                                                  control_horizon=control_horizon)

    def add_inequality_constraint_vector(self,
                                         reference_velocities: Union[
                                             cas.Expression, cas.Vector3, cas.Point3, List[cas.symbol_expr_float]],
                                         lower_errors: Union[
                                             cas.Expression, cas.Vector3, cas.Point3, List[cas.symbol_expr_float]],
                                         upper_errors: Union[
                                             cas.Expression, cas.Vector3, cas.Point3, List[cas.symbol_expr_float]],
                                         weights: Union[
                                             cas.Expression, cas.Vector3, cas.Point3, List[cas.symbol_expr_float]],
                                         task_expression: Union[
                                             cas.Expression, cas.Vector3, cas.Point3, List[cas.symbol_expr]],
                                         names: List[str],
                                         lower_slack_limits: Optional[List[cas.symbol_expr_float]] = None,
                                         upper_slack_limits: Optional[List[cas.symbol_expr_float]] = None):
        """
        Calls add_constraint for a list of expressions.
        """
        if len(lower_errors) != len(upper_errors) \
                or len(lower_errors) != len(task_expression) \
                or len(lower_errors) != len(reference_velocities) \
                or len(lower_errors) != len(weights) \
                or (names is not None and len(lower_errors) != len(names)) \
                or (lower_slack_limits is not None and len(lower_errors) != len(lower_slack_limits)) \
                or (upper_slack_limits is not None and len(lower_errors) != len(upper_slack_limits)):
            raise ConstraintInitalizationException('All parameters must have the same length.')
        for i in range(len(lower_errors)):
            name_suffix = names[i] if names else None
            lower_slack_limit = lower_slack_limits[i] if lower_slack_limits else None
            upper_slack_limit = upper_slack_limits[i] if upper_slack_limits else None
            self.add_inequality_constraint(reference_velocity=reference_velocities[i],
                                           lower_error=lower_errors[i],
                                           upper_error=upper_errors[i],
                                           weight=weights[i],
                                           task_expression=task_expression[i],
                                           name=name_suffix,
                                           lower_slack_limit=lower_slack_limit,
                                           upper_slack_limit=upper_slack_limit)

    def add_equality_constraint_vector(self,
                                       reference_velocities: Union[
                                           cas.Expression, cas.Vector3, cas.Point3, List[cas.symbol_expr_float]],
                                       equality_bounds: Union[
                                           cas.Expression, cas.Vector3, cas.Point3, List[cas.symbol_expr_float]],
                                       weights: Union[
                                           cas.Expression, cas.Vector3, cas.Point3, List[cas.symbol_expr_float]],
                                       task_expression: Union[
                                           cas.Expression, cas.Vector3, cas.Point3, List[cas.symbol_expr]],
                                       names: List[str],
                                       lower_slack_limits: Optional[List[cas.symbol_expr_float]] = None,
                                       upper_slack_limits: Optional[List[cas.symbol_expr_float]] = None):
        """
        Calls add_constraint for a list of expressions.
        """
        for i in range(len(equality_bounds)):
            name_suffix = names[i] if names else None
            lower_slack_limit = lower_slack_limits[i] if lower_slack_limits else None
            upper_slack_limit = upper_slack_limits[i] if upper_slack_limits else None
            self.add_equality_constraint(reference_velocity=reference_velocities[i],
                                         equality_bound=equality_bounds[i],
                                         weight=weights[i],
                                         task_expression=task_expression[i],
                                         name=name_suffix,
                                         lower_slack_limit=lower_slack_limit,
                                         upper_slack_limit=upper_slack_limit)
