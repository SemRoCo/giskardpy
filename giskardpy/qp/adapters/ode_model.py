from dataclasses import dataclass
from typing import Tuple, List, Union

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.qp.adapters.problem_data_part import ProblemDataPart
from semantic_digital_twin.spatial_types.derivatives import Derivatives

@dataclass
class ODEModel(ProblemDataPart):
    """
    Handles ODE constraints of the form dot(h) = f(h, t).
    Linearized as: (J_target - J_ode * dt) * v = f(h)
    """

    def ode_constraint_expressions(self) -> List[cas.Expression]:
        return self._sorter(
            {c.name: c.ode_function for c in self.constraint_collection.ode_constraints}
        )[0]

    def target_variable_expressions(self) -> List[cas.Expression]:
        return self._sorter(
            {c.name: c.expression for c in self.constraint_collection.ode_constraints}
        )[0]

    def get_free_variable_symbols(self, derivative: Derivatives) -> List[cas.FloatVariable]:
        return self._sorter(
            {
                v.variables.position.name: v.variables.data[derivative]
                for v in self.degrees_of_freedom
            }
        )[0]
        
    def ode_constraint_slack_lower_bound(self):
        return {
            f"{c.name}/error": c.lower_slack_limit
            for c in self.constraint_collection.ode_constraints
        }

    def ode_constraint_slack_upper_bound(self):
        return {
            f"{c.name}/error": c.upper_slack_limit
            for c in self.constraint_collection.ode_constraints
        }
        
    def ode_weight_expressions(self) -> dict:
        # Using normalized_weight from DerivativeConstraint (inherited by ODEConstraint)
        # Assuming normalization_factor is handled in the constraint
        error_slack_weights = {
            f"{c.name}/error": c.normalized_weight()
            for c in self.constraint_collection.ode_constraints
        }
        return error_slack_weights

    def construct_expression(self) -> Union[cas.Expression, Tuple[cas.Expression, cas.Expression]]:
        if len(self.constraint_collection.ode_constraints) == 0:
            return cas.Expression(), cas.Expression()

        # 1. Compute A matrix
        # A = J_target * dt
        # We assume f(h) is constant over the step (no linearization of f(h) w.r.t h)
        
        target_exprs = cas.Expression(self.target_variable_expressions())
        
        positions = self.get_free_variable_symbols(Derivatives.position)
        
        J_target = target_exprs.jacobian(variables=positions)
        
        # Linearize ODE function: f(h) + J_ode * delta_h
        # delta_h = v * dt
        # Constraint: J_target * v = f(h) + J_ode * v * dt
        # => (J_target - J_ode * dt) * v = f(h)
        ode_exprs = cas.Expression(self.ode_constraint_expressions())
        J_ode = ode_exprs.jacobian(variables=positions)
        
        dt = self.config.mpc_dt
        
        A_matrix = (J_target - J_ode * dt) * dt
        
        # Stack for prediction horizon using Kronecker product
        # This creates a block diagonal matrix where each block is A_matrix
        # Build the Jacobian matrix for all time steps
        # J_full will be block diagonal with A_matrix on the diagonal
        n_dofs = self.number_of_free_variables
        n_horizon = self.config.prediction_horizon
        
        # Determine the column structure based on QP formulation
        # For "no acc" formulation: has_explicit_acc_variables=False, has_explicit_jerk_variables=True
        # Column structure: [velocity for first (horizon-2) steps] [jerk for all horizon steps]
        # For standard formulation: [all velocity] [all acceleration] [all jerk]
        
        if not self.config.qp_formulation.has_explicit_acc_variables and self.config.qp_formulation.has_explicit_jerk_variables:
            # "No acc" formulation
            # Velocity: only first (horizon - 2) time steps
            n_vel_steps = n_horizon - 2
            n_vel_cols = n_dofs * n_vel_steps
            n_jerk_cols = n_dofs * n_horizon
            total_cols = n_vel_cols + n_jerk_cols
            
            # Create model with correct number of columns
            model = cas.Expression.zeros(
                len(self.constraint_collection.ode_constraints) * n_horizon,
                total_cols
            )
            
            # Place Jacobian for velocity time steps (0 to n_vel_steps-1)
            # We can only enforce ODE for time steps that have velocity variables
            for t in range(n_vel_steps):
                row_start = t * len(self.constraint_collection.ode_constraints)
                row_end = (t + 1) * len(self.constraint_collection.ode_constraints)
                col_start = t * n_dofs
                col_end = (t + 1) * n_dofs
                model[row_start:row_end, col_start:col_end] = A_matrix
            
            # For remaining time steps (n_vel_steps to n_horizon-1), we can't enforce the ODE
            # using velocity variables, so those rows will be zero (or we could use jerk)
            # For now, leave them as zero
            
        else:
            # Standard formulation with all derivatives
            eye_horizon = cas.Expression.eye(n_horizon)
            J_full = eye_horizon.kron(A_matrix)
            
            model = cas.Expression.zeros(
                len(self.constraint_collection.ode_constraints) * n_horizon,
                self.number_of_non_slack_columns
            )
            
            # Calculate the column offset for velocity variables
            # In the explicit formulation, velocity is derivative index 1
            horizontal_offset = n_dofs * n_horizon
            velocity_start = horizontal_offset * Derivatives.velocity
            velocity_end = horizontal_offset * (Derivatives.velocity + 1)
            
            # Place J_full in the velocity columns
            model[:, velocity_start : velocity_end] = J_full
        
        # Slack model
        # We need slacks for each step.
        num_slacks = len(self.constraint_collection.ode_constraints) * self.config.prediction_horizon
        slack_model = cas.Expression.eye(num_slacks) * dt # Scale slack by dt like equality constraints
        
        return model, slack_model


@dataclass
class ODEBounds(ProblemDataPart):
    def ode_constraint_expressions(self) -> List[cas.Expression]:
        return self._sorter(
            {c.name: c.ode_function for c in self.constraint_collection.ode_constraints}
        )[0]

    def construct_expression(self) -> cas.Expression:
        if len(self.constraint_collection.ode_constraints) == 0:
            return cas.Expression()
            
        # Bounds are simply the ode_function evaluated at current state
        # scaled by dt (to match the A matrix scaling)
        
        ode_exprs = cas.Expression(self.ode_constraint_expressions())
        
        # Stack for prediction horizon
        # We need to repeat the bounds for each step
        # Order must match the rows of the matrix (t0, t1, ...)
        
        dt = self.config.mpc_dt
        
        # ode_exprs is a column vector (n_constraints x 1)
        # We want (n_constraints * horizon x 1)
        # [c1..cn]_t0, [c1..cn]_t1, ...
        
        # cas.repmat? Or just vstack repeated
        
        bound_block = ode_exprs * dt
        
        bounds = cas.vstack([bound_block for _ in range(self.config.prediction_horizon)])
        
        return bounds
