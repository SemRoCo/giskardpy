from __future__ import annotations

from typing import Tuple, TYPE_CHECKING

import numpy as np
from line_profiler import profile

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.qp.adapters.qp_adapter import GiskardToQPAdapter
from giskardpy.qp.qp_data import QPData

if TYPE_CHECKING:
    import scipy.sparse as sp


class GiskardToTwoSidedNeqQPAdapter(GiskardToQPAdapter):
    """
    Takes free variables and constraints and converts them to a QP problem in the following format, depending on the
    class attributes:

    min_x 0.5 x^T H x + g^T x
    s.t.  lbA <= Ax <= ubA
    """

    b_bE_bA_filter: np.ndarray
    b_zero_inf_filter_view: np.ndarray
    bE_filter_view: np.ndarray
    bA_filter_view: np.ndarray
    bE_bA_filter: np.ndarray

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
        if len(neq_matrix_dofs) == 0:
            constraint_matrix = cas.hstack([eq_matrix_dofs, eq_matrix_slack])
        else:
            eq_matrix = cas.hstack(
                [
                    eq_matrix_dofs,
                    eq_matrix_slack,
                    cas.zeros(eq_matrix_dofs.shape[0], neq_matrix_slack.shape[1]),
                ]
            )
            neq_matrix = cas.hstack(
                [
                    neq_matrix_dofs,
                    cas.zeros(neq_matrix_dofs.shape[0], eq_matrix_slack.shape[1]),
                    neq_matrix_slack,
                ]
            )
            constraint_matrix = cas.vstack([eq_matrix, neq_matrix])

        self.free_symbols = [
            self.world_state_symbols,
            self.life_cycle_symbols,
            self.external_collision_symbols,
            self.self_collision_symbols,
            self.auxiliary_variables,
        ]

        len_lb_be_lba_end = (
            quadratic_weights.shape[0]
            + box_lower_constraints.shape[0]
            + eq_bounds.shape[0]
            + neq_lower_bounds.shape[0]
        )
        len_ub_be_uba_end = (
            len_lb_be_lba_end
            + box_upper_constraints.shape[0]
            + eq_bounds.shape[0]
            + neq_upper_bounds.shape[0]
        )

        self.combined_vector_f = cas.CompiledFunctionWithViews(
            expressions=[
                quadratic_weights,
                box_lower_constraints,
                eq_bounds,
                neq_lower_bounds,
                box_upper_constraints,
                eq_bounds,
                neq_upper_bounds,
                linear_weights,
            ],
            variable_parameters=self.free_symbols,
            additional_views=[
                slice(quadratic_weights.shape[0], len_lb_be_lba_end),
                slice(len_lb_be_lba_end, len_ub_be_uba_end),
            ],
        )

        self.neq_matrix_compiled = constraint_matrix.compile(
            parameters=self.free_symbols, sparse=self.sparse
        )

        self.b_bE_bA_filter = np.ones(
            box_lower_constraints.shape[0]
            + eq_bounds.shape[0]
            + neq_lower_bounds.shape[0],
            dtype=bool,
        )
        self.b_zero_inf_filter_view = self.b_bE_bA_filter[
            : box_lower_constraints.shape[0]
        ]
        self.bE_filter_view = self.b_bE_bA_filter[
            box_lower_constraints.shape[0] : box_lower_constraints.shape[0]
            + eq_bounds.shape[0]
        ]
        self.bA_filter_view = self.b_bE_bA_filter[
            box_lower_constraints.shape[0] + eq_bounds.shape[0] :
        ]
        self.bE_bA_filter = self.b_bE_bA_filter[box_lower_constraints.shape[0] :]

        if self.compute_nI_I:
            self._nAi_Ai_cache = {}

    def create_filters(
        self,
        quadratic_weights_np_raw: np.ndarray,
        box_lower_constraints_np_raw: np.ndarray,
        box_upper_constraints_np_raw: np.ndarray,
        num_slack_variables: int,
        num_eq_slack_variables: int,
        num_neq_slack_variables: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.b_bE_bA_filter.fill(True)

        zero_quadratic_weight_filter: np.ndarray = quadratic_weights_np_raw != 0
        zero_quadratic_weight_filter[:-num_slack_variables] = True

        slack_part = zero_quadratic_weight_filter[
            -(num_eq_slack_variables + num_neq_slack_variables) :
        ]
        bE_part = slack_part[:num_eq_slack_variables]
        if len(bE_part) > 0:
            self.bE_filter_view[-len(bE_part) :] = bE_part

        bA_part = slack_part[num_eq_slack_variables:]
        if len(bA_part) > 0:
            self.bA_filter_view[-len(bA_part) :] = bA_part

        b_finite_filter = np.isfinite(box_lower_constraints_np_raw) | np.isfinite(
            box_upper_constraints_np_raw
        )
        self.b_zero_inf_filter_view[::] = zero_quadratic_weight_filter & b_finite_filter
        Ai_inf_filter = b_finite_filter[zero_quadratic_weight_filter]
        return (
            zero_quadratic_weight_filter,
            Ai_inf_filter,
            self.bE_bA_filter,
            self.b_bE_bA_filter,
        )

    @profile
    def apply_filters(
        self,
        qp_data_raw: QPData,
        zero_quadratic_weight_filter: np.ndarray,
        Ai_inf_filter: np.ndarray,
        bE_bA_filter: np.ndarray,
        b_bE_bA_filter: np.ndarray,
    ) -> QPData:
        from scipy import sparse as sp

        qp_data_filtered = QPData()
        qp_data_filtered.quadratic_weights = qp_data_raw.quadratic_weights[
            zero_quadratic_weight_filter
        ]
        qp_data_filtered.linear_weights = qp_data_raw.linear_weights[
            zero_quadratic_weight_filter
        ]
        qp_data_filtered.neq_lower_bounds = qp_data_raw.neq_lower_bounds[b_bE_bA_filter]
        qp_data_filtered.neq_upper_bounds = qp_data_raw.neq_upper_bounds[b_bE_bA_filter]
        qp_data_filtered.neq_matrix = qp_data_raw.neq_matrix[
            :, zero_quadratic_weight_filter
        ][bE_bA_filter, :]

        box_matrix = self._direct_limit_model(
            qp_data_filtered.quadratic_weights.shape[0], Ai_inf_filter, two_sided=True
        )
        qp_data_filtered.neq_matrix = sp.vstack(
            (box_matrix, qp_data_filtered.neq_matrix)
        )

        return qp_data_filtered

    def evaluate(
        self,
        world_state: np.ndarray,
        life_cycle_state: np.ndarray,
        external_collision_data: np.ndarray,
        self_collision_data: np.ndarray,
        auxiliary_variables: np.ndarray,
    ) -> QPData:
        args = [
            world_state,
            life_cycle_state,
            external_collision_data,
            self_collision_data,
            auxiliary_variables,
        ]
        neq_matrix = self.neq_matrix_compiled(*args)
        (
            quadratic_weights_np_raw,
            box_lower_constraints_np_raw,
            _,
            _,
            box_upper_constraints_np_raw,
            _,
            _,
            linear_weights_np_raw,
            box_eq_neq_lower_bounds_np_raw,
            box_eq_neq_upper_bounds_np_raw,
        ) = self.combined_vector_f(*args)
        self.qp_data_raw = QPData(
            quadratic_weights=quadratic_weights_np_raw,
            linear_weights=linear_weights_np_raw,
            neq_matrix=neq_matrix,
            neq_lower_bounds=box_eq_neq_lower_bounds_np_raw,
            neq_upper_bounds=box_eq_neq_upper_bounds_np_raw,
        )

        zero_quadratic_weight_filter, Ai_inf_filter, bE_bA_filter, b_bE_bA_filter = (
            self.create_filters(
                quadratic_weights_np_raw=quadratic_weights_np_raw,
                box_lower_constraints_np_raw=box_lower_constraints_np_raw,
                box_upper_constraints_np_raw=box_upper_constraints_np_raw,
                num_slack_variables=self.num_slack_variables,
                num_eq_slack_variables=self.num_eq_slack_variables,
                num_neq_slack_variables=self.num_neq_slack_variables,
            )
        )

        self.qp_data_raw.filtered = self.apply_filters(
            qp_data_raw=self.qp_data_raw,
            zero_quadratic_weight_filter=zero_quadratic_weight_filter,
            Ai_inf_filter=Ai_inf_filter,
            bE_bA_filter=bE_bA_filter,
            b_bE_bA_filter=b_bE_bA_filter,
        )

        return self.qp_data_raw
