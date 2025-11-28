from __future__ import annotations

import datetime
from copy import deepcopy
from dataclasses import dataclass, field, InitVar
from typing import List, Optional, TYPE_CHECKING

import numpy as np
import pandas
import pandas as pd
from line_profiler import profile

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.middleware import get_middleware
from giskardpy.qp.adapters.qp_adapter import GiskardToQPAdapter
from giskardpy.qp.constraint_collection import ConstraintCollection
from giskardpy.qp.exceptions import (
    HardConstraintsViolatedException,
    InfeasibleException,
)
from giskardpy.qp.solvers.qp_solver import QPSolver
from giskardpy.utils.utils import create_path
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom

if TYPE_CHECKING:
    from giskardpy.qp.qp_controller_config import QPControllerConfig

# used for saving pandas in the same folder every time within a run
date_str = datetime.datetime.now().strftime("%Yy-%mm-%dd--%Hh-%Mm-%Ss")


@dataclass
class QPControllerDebugger:
    qp_controller: QPController
    xdot_full: pandas.DataFrame = field(init=False)
    weights: pandas.DataFrame = field(init=False)
    A: pandas.DataFrame = field(init=False)
    b: pandas.DataFrame = field(init=False)
    E: pandas.DataFrame = field(init=False)
    bE: pandas.DataFrame = field(init=False)
    lbA: pandas.DataFrame = field(init=False)
    ubA: pandas.DataFrame = field(init=False)
    xdot: pandas.DataFrame = field(init=False)
    debug: pandas.DataFrame = field(init=False)

    def _has_nan(self):
        nan_entries = self.p_A[0].isnull().stack()
        row_col_names = nan_entries[nan_entries].index.tolist()
        pass

    def are_hard_limits_violated(self, error_message):
        try:
            lower_violations = self.p_lb[self.qp_controller.qp_solver.lb_filter]
            upper_violations = self.p_ub[self.qp_controller.qp_solver.ub_filter]
            if len(upper_violations) > 0 or len(lower_violations) > 0:
                error_message += "\n"
                if len(upper_violations) > 0:
                    error_message += "upper slack bounds of following constraints might be too low: {}\n".format(
                        list(upper_violations.index)
                    )
                if len(lower_violations) > 0:
                    error_message += "lower slack bounds of following constraints might be too high: {}".format(
                        list(lower_violations.index)
                    )
                raise HardConstraintsViolatedException(error_message)
        except AttributeError:
            pass
        get_middleware().loginfo("No slack limit violation detected.")

    def _print_pandas_array(self, array):
        import pandas as pd

        if len(array) > 0:
            with pd.option_context(
                "display.max_rows", None, "display.max_columns", None
            ):
                print(array)

    def save_all_pandas(self, folder_name: Optional[str] = None):
        raise NotImplementedError()
        # self.save_pandas(
        #     [
        #         self.p_weights,
        #         self.p_b,
        #         self.p_E,
        #         self.p_bE,
        #         self.p_A,
        #         self.p_lbA,
        #         self.p_ubA,
        #         None,
        #         self.p_xdot,
        #     ],
        #     ["weights", "b", "E", "bE", "A", "lbA", "ubA", "debug"],
        #     god_map.tmp_folder,
        #     None,
        #     folder_name,
        # )

    def save_pandas(
        self, dfs, names, path, time: float, folder_name: Optional[str] = None
    ):

        if folder_name is None:
            folder_name = ""
        folder_name = f"{path}/pandas/{folder_name}_{date_str}/{time}/"
        create_path(folder_name)
        for df, name in zip(dfs, names):
            csv_string = "name\n"
            with pd.option_context(
                "display.max_rows", None, "display.max_columns", None
            ):
                if df.shape[1] > 1:
                    for column_name, column in df.T.items():
                        zero_filtered_column = (
                            column.replace(0, np.nan)
                            .dropna(how="all")
                            .replace(np.nan, 0)
                        )
                        csv_string += zero_filtered_column.add_prefix(
                            column_name + "||"
                        ).to_csv(float_format="%.6f")
                else:
                    csv_string += df.to_csv(float_format="%.6f")
            file_name2 = f"{folder_name}{name}.csv"
            with open(file_name2, "w") as f:
                f.write(csv_string)

    def _update_quadratic_weights(self):
        self.p_weights = pd.DataFrame(
            self.qp_data.quadratic_weights,
            self.free_variable_names,
            ["data"],
            dtype=float,
        )

    def _update_linear_weights(self):
        self.p_g = pd.DataFrame(
            self.qp_data.linear_weights,
            self.free_variable_names,
            ["data"],
            dtype=float,
        )

    def _update_box_constraints(self):
        self.p_lb = pd.DataFrame(
            self.qp_data.box_lower_constraints,
            self.free_variable_names,
            ["data"],
            dtype=float,
        )
        self.p_ub = pd.DataFrame(
            self.qp_data.box_upper_constraints,
            self.free_variable_names,
            ["data"],
            dtype=float,
        )
        self.p_b = pd.DataFrame(
            {
                "lb": self.qp_data.box_lower_constraints,
                "ub": self.qp_data.box_upper_constraints,
            },
            self.free_variable_names,
            dtype=float,
        )

    def _update_eq_constraints(self):
        if len(self.qp_data.eq_bounds) > 0:
            self.p_bE_raw = pd.DataFrame(
                self.qp_data.eq_bounds,
                self.equality_constr_names,
                ["data"],
                dtype=float,
            )
            self.p_bE = deepcopy(self.p_bE_raw)
            self.p_bE[
                len(
                    self.qp_controller.qp_adapter.equality_bounds.names_derivative_links
                ) :
            ] /= self.qp_controller.config.mpc_dt
        else:
            self.p_bE = pd.DataFrame()

    def _update_inequality_constraints(self):
        if len(self.qp_data.neq_lower_bounds) > 0:
            self.p_lbA_raw = pd.DataFrame(
                self.qp_data.neq_lower_bounds,
                self.inequality_constr_names,
                ["data"],
                dtype=float,
            )
            self.p_lbA = deepcopy(self.p_lbA_raw)
            self.p_lbA /= self.qp_controller.config.mpc_dt

            self.p_ubA_raw = pd.DataFrame(
                self.qp_data.neq_upper_bounds,
                self.inequality_constr_names,
                ["data"],
                dtype=float,
            )
            self.p_ubA = deepcopy(self.p_ubA_raw)
            self.p_ubA /= self.qp_controller.config.mpc_dt

            self.p_bA_raw = pd.DataFrame(
                {
                    "lbA": self.qp_data.neq_lower_bounds,
                    "ubA": self.qp_data.neq_upper_bounds,
                },
                self.inequality_constr_names,
                dtype=float,
            )
            self.p_bA = deepcopy(self.p_bA_raw)
            self.p_bA /= self.qp_controller.config.mpc_dt
        else:
            self.p_lbA = pd.DataFrame()
            self.p_ubA = pd.DataFrame()

    def _update_equality_matrix(self):
        if len(self.qp_data.dense_eq_matrix) > 0:
            self.p_E = pd.DataFrame(
                self.qp_data.dense_eq_matrix,
                self.equality_constr_names,
                self.free_variable_names,
                dtype=float,
            )
        else:
            self.p_E = pd.DataFrame()

    def _update_inequality_matrix(self):
        if len(self.qp_data.dense_neq_matrix) > 0:
            self.p_A = pd.DataFrame(
                self.qp_data.dense_neq_matrix,
                self.inequality_constr_names,
                self.free_variable_names,
                dtype=float,
            )
        else:
            self.p_A = pd.DataFrame()

    def _update_xdot(self, new_xdot_full: Optional[np.ndarray]):
        self.p_xdot = None
        if new_xdot_full is None:
            return

        num_vel_constr = len(
            self.qp_controller.qp_adapter.constraint_collection.derivative_constraints
        ) * (self.qp_controller.config.prediction_horizon - 2)
        num_eq_vel_constr = len(
            self.qp_controller.qp_adapter.constraint_collection.eq_derivative_constraints
        ) * (self.qp_controller.config.prediction_horizon - 2)
        num_neq_constr = len(
            self.qp_controller.qp_adapter.constraint_collection.neq_constraints
        )
        num_eq_constr = len(
            self.qp_controller.qp_adapter.constraint_collection.eq_constraints
        )
        num_constr = num_vel_constr + num_neq_constr + num_eq_constr + num_eq_vel_constr

        xdot_full = np.ones(self.qp_data.zero_quadratic_weight_filter.shape) * np.nan
        xdot_full[self.qp_data.zero_quadratic_weight_filter] = new_xdot_full
        self.p_xdot = pd.DataFrame(
            xdot_full, self.free_variable_names, ["data"], dtype=float
        )
        self.p_b["xdot"] = self.p_xdot
        self.p_b = self.p_b[["lb", "xdot", "ub"]]
        self.p_pure_xdot = deepcopy(self.p_xdot)
        self.p_pure_xdot[-num_constr:] = 0
        if len(self.p_A) > 0:
            self.p_Ax = pd.DataFrame(
                self.p_A.dot(self.p_pure_xdot),
                self.inequality_constr_names,
                ["data"],
                dtype=float,
            )
            bA_slack = self.p_xdot[len(self.p_xdot) - num_neq_constr - num_vel_constr :]
            self.p_bA_raw.insert(1, "Ax", self.p_Ax)
            self.p_bA_raw.insert(2, "slack", bA_slack.values)
        else:
            self.p_Ax = pd.DataFrame()
        if len(self.p_E) > 0:
            self.p_Ex = pd.DataFrame(
                self.p_E.dot(self.p_pure_xdot),
                self.equality_constr_names,
                ["data"],
                dtype=float,
            )
        else:
            self.p_Ex = pd.DataFrame()

    @property
    def qp_data(self):
        return self.qp_controller.qp_adapter.qp_data_raw

    @property
    def free_variable_names(self):
        return self.qp_controller.qp_adapter.free_variable_bounds.names.tolist()

    @property
    def equality_constr_names(self):
        return self.qp_controller.qp_adapter.equality_bounds.names

    @property
    def inequality_constr_names(self):
        return self.qp_controller.qp_adapter.inequality_bounds.names

    @profile
    def update(self, new_xdot_full: Optional[np.ndarray]) -> None:
        self._update_quadratic_weights()
        self._update_linear_weights()
        self._update_box_constraints()
        self._update_eq_constraints()
        self._update_inequality_constraints()
        self._update_equality_matrix()
        self._update_inequality_matrix()
        self._update_xdot(new_xdot_full)

    def _print_iis(self):
        import pandas as pd

        def print_iis_matrix(
            row_filter: np.ndarray,
            column_filter: np.ndarray,
            matrix: pd.DataFrame,
            bounds: pd.DataFrame,
        ):
            if len(row_filter) == 0:
                return
            filtered_matrix = matrix.loc[row_filter, column_filter]
            filtered_matrix["bounds"] = bounds.loc[row_filter]
            print(filtered_matrix)

        result = self.qp_controller.qp_solver.analyze_infeasibility()
        if result is None:
            get_middleware().loginfo(
                f"Can only compute possible causes with gurobi, "
                f"but current solver is {self.qp_controller.config.qp_solver_id.name}."
            )
            return
        lb_ids, ub_ids, eq_ids, lbA_ids, ubA_ids = result
        b_ids = lb_ids | ub_ids
        with pd.option_context(
            "display.max_rows", None, "display.max_columns", None, "display.width", None
        ):
            get_middleware().loginfo("Irreducible Infeasible Subsystem:")
            get_middleware().loginfo("  Free variable bounds")
            free_variables = self.p_lb[b_ids]
            free_variables["ub"] = self.p_ub[b_ids]
            free_variables = free_variables.rename(columns={"data": "lb"})
            print(free_variables)
            get_middleware().loginfo("  Equality constraints:")
            print_iis_matrix(eq_ids, b_ids, self.p_E, self.p_bE)
            get_middleware().loginfo("  Inequality constraint lower bounds:")
            print_iis_matrix(lbA_ids, b_ids, self.p_A, self.p_lbA)
            get_middleware().loginfo("  Inequality constraint upper bounds:")
            print_iis_matrix(ubA_ids, b_ids, self.p_A, self.p_ubA)


@dataclass
class QPController:
    """
    Wraps around QP Solver. Builds the required matrices from constraints.
    """

    config: QPControllerConfig
    degrees_of_freedom: InitVar[List[DegreeOfFreedom]]
    active_dofs: List[DegreeOfFreedom] = field(init=False)
    constraint_collection: ConstraintCollection
    world_state_symbols: List[cas.FloatVariable]
    life_cycle_variables: List[cas.FloatVariable]
    external_collision_avoidance_variables: List[cas.FloatVariable]
    self_collision_avoidance_variables: List[cas.FloatVariable]
    auxiliary_variables: List[cas.FloatVariable]

    qp_adapter: GiskardToQPAdapter = field(default=None, init=False)
    qp_solver: QPSolver = field(default=None, init=False)
    debugger: QPControllerDebugger = field(default=None, init=False)

    @profile
    def __post_init__(self, degrees_of_freedom: List[DegreeOfFreedom]):
        self.qp_solver = self.config.qp_solver_class()
        if self.config.verbose:
            get_middleware().loginfo(
                f"Initialized QP Controller:\n"
                f'sample period: "{self.config.mpc_dt}"s\n'
                f'max derivative: "{self.config.max_derivative.name}"\n'
                f'prediction horizon: "{self.config.prediction_horizon}"\n'
                f'QP solver: "{self.config.qp_solver_id.name}"'
            )
        self.debugger = QPControllerDebugger(self)
        self._set_active_dofs(degrees_of_freedom)

        self.qp_adapter = self.qp_solver.required_adapter_type(
            world_state_symbols=self.world_state_symbols,
            life_cycle_symbols=self.life_cycle_variables,
            auxiliary_variables=self.auxiliary_variables,
            external_collision_symbols=self.external_collision_avoidance_variables,
            self_collision_symbols=self.self_collision_avoidance_variables,
            degrees_of_freedom=self.active_dofs,
            constraint_collection=self.constraint_collection,
            config=self.config,
        )

        get_middleware().loginfo("Done compiling controller:")
        # get_middleware().loginfo(f'  #free variables: {self.num_free_variables}')
        # get_middleware().loginfo(f'  #equality constraints: {self.num_eq_constraints}')
        # get_middleware().loginfo(f'  #inequality constraints: {self.num_ineq_constraints}')

    def _set_active_dofs(self, degrees_of_freedom: List[DegreeOfFreedom]):
        all_active_float_variables = set().union(
            *[
                {
                    dof.variables.position.name,
                    dof.variables.velocity.name,
                    dof.variables.acceleration.name,
                    dof.variables.jerk.name,
                }
                for dof in degrees_of_freedom
            ]
        )
        float_variable_names = self.constraint_collection.get_all_float_variable_names()
        active_float_variables = all_active_float_variables & float_variable_names

        def dof_used(dof: DegreeOfFreedom) -> bool:
            vars_ = dof.variables
            return (
                vars_.position.name in float_variable_names
                or vars_.velocity.name in float_variable_names
                or vars_.acceleration.name in float_variable_names
                or vars_.jerk.name in float_variable_names
            )

        self.dof_filter = np.array(
            [
                i
                for i, v in sorted(
                    enumerate(self.world_state_symbols), key=lambda x: x[1].name
                )
                if v.name in active_float_variables
            ]
        )
        self.active_dofs = [dof for dof in degrees_of_freedom if dof_used(dof)]

    def has_not_free_variables(self) -> bool:
        return len(self.active_dofs) == 0

    @profile
    def get_cmd(
        self,
        world_state: np.ndarray,
        life_cycle_state: np.ndarray,
        external_collisions: np.ndarray,
        self_collisions: np.ndarray,
        auxiliary_variables: np.ndarray,
    ) -> np.ndarray:
        """
        Uses substitutions for each symbol to compute the next commands for each joint.
        """
        try:
            qp_data = self.qp_adapter.evaluate(
                world_state,
                life_cycle_state,
                external_collisions,
                self_collisions,
                auxiliary_variables,
            )
            try:
                self.xdot_full = self.qp_solver.solver_call(qp_data.filtered)
            except InfeasibleException as e:
                self.config.retries_with_relaxed_constraints -= 1
                relaxed_solution = self.qp_solver.solver_call(qp_data.relaxed())
                if self.config.retries_with_relaxed_constraints < 0:
                    raise HardConstraintsViolatedException(
                        f"Hard constraints were violated too often."
                    )
                self.xdot_full = self.qp_solver.solver_call(
                    qp_data.partially_relaxed(relaxed_solution)
                )
            return self.xdot_to_control_commands(self.xdot_full)

        except InfeasibleException as e_original:
            self.xdot_full = None
            self.debugger.update(self.xdot_full)
            # self._has_nan()
            # self._print_iis()
            if isinstance(e_original, HardConstraintsViolatedException):
                raise
            self.xdot_full = None
            self.debugger.are_hard_limits_violated(str(e_original))
            raise

    def xdot_to_control_commands(self, xdot: np.ndarray) -> np.ndarray:
        offset = len(self.active_dofs) * (self.config.prediction_horizon - 2)
        offset_end = offset + len(self.active_dofs)
        control_cmds = xdot[offset:offset_end]
        # divide by 4 because the world state has pos/vel/acc/jerk variables
        full_control_cmds = np.zeros(len(self.world_state_symbols) // 4)
        full_control_cmds[self.dof_filter] = control_cmds
        return full_control_cmds
