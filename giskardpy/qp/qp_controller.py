import datetime
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Dict, Type, Optional

import numpy as np
from giskardpy.data_types.data_types import Derivatives
from giskardpy.data_types.exceptions import HardConstraintsViolatedException, QPSolverException, InfeasibleException
from giskardpy.god_map import god_map
from giskardpy.middleware import get_middleware
from giskardpy.qp.constraint import DerivativeEqualityConstraint
from giskardpy.qp.constraint import InequalityConstraint, EqualityConstraint, DerivativeInequalityConstraint
from giskardpy.qp.free_variable import FreeVariable
from giskardpy.qp.next_command import NextCommands
from giskardpy.qp.qp_adapter import GiskardToExplicitQPAdapter, GiskardToQPAdapter
from giskardpy.qp.qp_formulation import QPFormulation
from giskardpy.qp.qp_solver import QPSolver
from giskardpy.qp.qp_solver_ids import SupportedQPSolver
from giskardpy.qp.weight_gain import QuadraticWeightGain, LinearWeightGain
from giskardpy.symbol_manager import symbol_manager, SymbolManager
from giskardpy.utils.utils import create_path, get_all_classes_in_module
from line_profiler import profile

# used for saving pandas in the same folder every time within a run
date_str = datetime.datetime.now().strftime('%Yy-%mm-%dd--%Hh-%Mm-%Ss')


def save_pandas(dfs, names, path, time: float, folder_name: Optional[str] = None):
    import pandas as pd
    if folder_name is None:
        folder_name = ''
    folder_name = f'{path}/pandas/{folder_name}_{date_str}/{time}/'
    create_path(folder_name)
    for df, name in zip(dfs, names):
        csv_string = 'name\n'
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            if df.shape[1] > 1:
                for column_name, column in df.T.items():
                    zero_filtered_column = column.replace(0, np.nan).dropna(how='all').replace(np.nan, 0)
                    csv_string += zero_filtered_column.add_prefix(column_name + '||').to_csv(float_format='%.6f')
            else:
                csv_string += df.to_csv(float_format='%.6f')
        file_name2 = f'{folder_name}{name}.csv'
        with open(file_name2, 'w') as f:
            f.write(csv_string)


available_solvers: Dict[SupportedQPSolver, Type[QPSolver]] = {}


def detect_solvers():
    global available_solvers
    solver_name: str
    qp_solver_class: Type[QPSolver]
    for qp_solver_name in SupportedQPSolver:
        module_name = f'giskardpy.qp.qp_solver_{qp_solver_name.name}'
        try:
            qp_solver_class = list(get_all_classes_in_module(module_name, QPSolver).items())[0][1]
            available_solvers[qp_solver_name] = qp_solver_class
        except Exception:
            continue
    solver_names = [solver_name.name for solver_name in available_solvers.keys()]
    print(f'Found these qp solvers: {solver_names}')


detect_solvers()


class QPController:
    """
    Wraps around QP Solver. Builds the required matrices from constraints.
    """
    qp_adapters: List[GiskardToQPAdapter]

    @profile
    def __init__(self,
                 mpc_dt: float,
                 prediction_horizon: int,
                 control_dt: Optional[float] = None,
                 max_derivative: Derivatives = Derivatives.jerk,
                 solver_id: Optional[SupportedQPSolver] = None,
                 retries_with_relaxed_constraints: int = 5,
                 retry_added_slack: float = 100,
                 retry_weight_factor: float = 100,
                 qp_formulation: Optional[QPFormulation] = None,
                 horizon_weight_gain_scalar: float = 0.1,
                 verbose: bool = True):
        # todo
        # base_symbols = []
        # non_base_free_variables = []
        # for v in free_variables:
        #     if v.is_base:
        #         base_symbols.append(v.get_symbol(Derivatives.position))
        #     else:
        #         non_base_free_variables.append(v)
        #
        # god_map.qp_controller2.init(
        #     free_variables=non_base_free_variables,
        #     equality_constraints=eq_constraints,
        #     inequality_constraints=neq_constraints,
        #     eq_derivative_constraints=eq_derivative_constraints,
        #     derivative_constraints=derivative_constraints,
        #     quadratic_weight_gains=quadratic_weight_gains,
        #     linear_weight_gains=linear_weight_gains,
        # )
        # god_map.qp_controller2.compile()
        if control_dt is None:
            control_dt = mpc_dt
        self.control_dt = control_dt
        self.horizon_weight_gain_scalar = horizon_weight_gain_scalar
        self.qp_formulation = qp_formulation or QPFormulation()
        self.mpc_dt = mpc_dt
        self.max_derivative = max_derivative
        self.prediction_horizon = prediction_horizon
        self.retries_with_relaxed_constraints = retries_with_relaxed_constraints
        self.retry_added_slack = retry_added_slack
        self.retry_weight_factor = retry_weight_factor
        self.verbose = verbose
        self.set_qp_solver(solver_id)
        if not self.qp_formulation.is_mpc:
            self.prediction_horizon = 1
            self.max_derivative = Derivatives.velocity

        if self.verbose:
            get_middleware().loginfo(f'Initialized QP Controller:\n'
                                     f'sample period: "{self.mpc_dt}"s\n'
                                     f'max derivative: "{self.max_derivative.name}"\n'
                                     f'prediction horizon: "{self.prediction_horizon}"\n'
                                     f'QP solver: "{self.qp_solver_class.solver_id.name}"')
        self.reset()

    def set_qp_solver(self, solver_id: Optional[SupportedQPSolver] = None) -> None:
        print_later = hasattr(self, 'qp_solver_class')
        if solver_id is not None:
            self.qp_solver_class = available_solvers[solver_id]
        else:
            for solver_id in SupportedQPSolver:
                if solver_id in available_solvers:
                    self.qp_solver_class = available_solvers[solver_id]
                    break
            else:
                raise QPSolverException(f'No qp solver found')
        self.qp_solver = self.qp_solver_class()
        if print_later:
            get_middleware().loginfo(f'QP Solver set to "{self.qp_solver_class.solver_id.name}"')

    def reset(self):
        self.free_variables = []
        self.equality_constraints = []
        self.inequality_constraints = []
        self.derivative_constraints = []
        self.eq_derivative_constraints = []
        self.quadratic_weight_gains = []
        self.linear_weight_gains = []
        self.xdot_full = None

    def init(self,
             free_variables: List[FreeVariable] = None,
             equality_constraints: List[EqualityConstraint] = None,
             inequality_constraints: List[InequalityConstraint] = None,
             derivative_constraints: List[DerivativeInequalityConstraint] = None,
             eq_derivative_constraints: List[DerivativeEqualityConstraint] = None,
             quadratic_weight_gains: List[QuadraticWeightGain] = None,
             linear_weight_gains: List[LinearWeightGain] = None):
        self.free_variables = free_variables
        if self.qp_formulation.double_qp:
            num_adapters = 2
        else:
            num_adapters = 1
        self.qp_adapters = []
        for _ in range(num_adapters):
            self.qp_adapters.append(self.qp_solver.required_adapter_type(
                world_state_symbols=god_map.world.get_state_symbols(),
                free_variables=free_variables,
                equality_constraints=equality_constraints,
                inequality_constraints=inequality_constraints,
                derivative_constraints=derivative_constraints,
                eq_derivative_constraints=eq_derivative_constraints,
                mpc_dt=self.mpc_dt,
                prediction_horizon=self.prediction_horizon,
                max_derivative=self.max_derivative,
                horizon_weight_gain_scalar=self.horizon_weight_gain_scalar,
                qp_formulation=self.qp_formulation))

        get_middleware().loginfo('Done compiling controller:')
        # get_middleware().loginfo(f'  #free variables: {self.num_free_variables}')
        # get_middleware().loginfo(f'  #equality constraints: {self.num_eq_constraints}')
        # get_middleware().loginfo(f'  #inequality constraints: {self.num_ineq_constraints}')

    def save_all_pandas(self, folder_name: Optional[str] = None):
        self._create_debug_pandas(self.qp_solver)
        save_pandas(
            [self.p_weights, self.p_b,
             self.p_E, self.p_bE,
             self.p_A, self.p_lbA, self.p_ubA,
             god_map.debug_expression_manager.to_pandas(), self.p_xdot],
            ['weights', 'b', 'E', 'bE', 'A', 'lbA', 'ubA', 'debug'],
            god_map.tmp_folder,
            god_map.time,
            folder_name)

    def _print_pandas_array(self, array):
        import pandas as pd
        if len(array) > 0:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(array)

    @profile
    def get_cmd(self, symbol_manager: SymbolManager) -> NextCommands:
        """
        Uses substitutions for each symbol to compute the next commands for each joint.
        """
        try:
            # todo
            # 1. get qp data(s)
            # 2. solve qp(s)
            # 3. decide solution
            # 4. return
            if self.qp_formulation.double_qp:
                for adapter in self.qp_adapters:
                    qp_data = adapter.evaluate(god_map.world.state.data, symbol_manager)
            else:
                qp_data = self.qp_adapters[0].evaluate(god_map.world.state.data, symbol_manager)
            self.xdot_full = self.qp_solver.solver_call(qp_data)
            # self._create_debug_pandas(self.qp_solver)
            if self.qp_formulation.is_implicit:
                return NextCommands.from_xdot_implicit(self.free_variables, self.xdot_full, self.max_derivative,
                                                       self.prediction_horizon, god_map.world, self.mpc_dt)
            elif self.qp_formulation.is_explicit or not self.qp_formulation.is_mpc:
                return NextCommands.from_xdot(self.free_variables, self.xdot_full, self.max_derivative,
                                              self.prediction_horizon)
            else:
                return NextCommands.from_xdot_explicit_no_acc(self.free_variables, self.xdot_full, self.max_derivative,
                                                              self.prediction_horizon, god_map.world,
                                                              self.mpc_dt)
        except InfeasibleException as e_original:
            self.xdot_full = None
            self._create_debug_pandas(self.qp_solver)
            self._has_nan()
            self._print_iis()
            if isinstance(e_original, HardConstraintsViolatedException):
                raise
            self.xdot_full = None
            self._are_hard_limits_violated(str(e_original))
            raise

    def _has_nan(self):
        nan_entries = self.p_A.isnull().stack()
        row_col_names = nan_entries[nan_entries].index.tolist()
        pass

    def _are_hard_limits_violated(self, error_message):
        self._create_debug_pandas(self.qp_solver)
        try:
            lower_violations = self.p_lb[self.qp_solver.lb_filter]
            upper_violations = self.p_ub[self.qp_solver.ub_filter]
            if len(upper_violations) > 0 or len(lower_violations) > 0:
                error_message += '\n'
                if len(upper_violations) > 0:
                    error_message += 'upper slack bounds of following constraints might be too low: {}\n'.format(
                        list(upper_violations.index))
                if len(lower_violations) > 0:
                    error_message += 'lower slack bounds of following constraints might be too high: {}'.format(
                        list(lower_violations.index))
                raise HardConstraintsViolatedException(error_message)
        except AttributeError:
            pass
        get_middleware().loginfo('No slack limit violation detected.')

    def _viz_mpc(self, joint_name):
        import matplotlib.pyplot as plt
        def pad(a, desired_length, pad_value):
            tmp = np.ones(desired_length) * pad_value
            tmp[:len(a)] = a
            return tmp

        free_variable: FreeVariable = [x for x in self.free_variables if x.name == joint_name][0]
        try:
            start_pos = god_map.world.state[joint_name].position
        except KeyError:
            get_middleware().loginfo('start position not found in state')
            start_pos = 0
        ts = np.array([(i + 1) * self.mpc_dt for i in range(self.prediction_horizon)])
        filtered_x = self.p_xdot.filter(like=f'/{joint_name}/', axis=0)
        vel_end = self.prediction_horizon - self.order + 1
        acc_end = vel_end + self.prediction_horizon - self.order + 2
        velocities = filtered_x[:vel_end].values
        positions = [start_pos]
        for x_ in velocities:
            positions.append(positions[-1] + x_ * self.mpc_dt)

        positions = np.array(positions[1:])
        positions = pad(positions.T[0], len(ts), pad_value=positions[-1])
        velocities = pad(velocities.T[0], len(ts), pad_value=0)

        if joint_name in god_map.world.state:
            accelerations = filtered_x[vel_end:acc_end].values
            jerks = filtered_x[acc_end:].values
            accelerations = pad(accelerations.T[0], len(ts), pad_value=0)

        f, axs = plt.subplots(4, sharex=True, figsize=(2 + self.prediction_horizon, 16))
        axs[0].set_title('position')
        axs[0].plot(ts, positions, 'b')
        axs[0].grid()
        axs[1].set_title('velocity')
        axs[1].plot(ts, velocities, 'b')
        axs[1].grid()
        if joint_name in god_map.world.state:
            axs[2].set_title('acceleration')
            axs[2].plot(ts, accelerations, 'b')
            axs[2].grid()
            axs[3].set_title('jerk')
            axs[3].plot(ts, jerks, 'b')
            axs[3].grid()
        for i, ax in enumerate(axs):
            derivative = Derivatives(i)
            if not free_variable.has_position_limits():
                continue
            upper_limit = free_variable.get_upper_limit(derivative, evaluated=True)
            if not np.isinf(upper_limit):
                ax.axhline(y=upper_limit, color='k', linestyle='--')
            lower_limit = free_variable.get_lower_limit(derivative, evaluated=True)
            if not np.isinf(lower_limit):
                ax.axhline(y=lower_limit, color='k', linestyle='--')
        # Example: Set x-ticks for each subplot
        tick_labels = [f'{x}/{x * self.mpc_dt:.3f}' for x in range(self.prediction_horizon)]

        axs[-1].set_xticks(ts)  # Set tick locations
        axs[-1].set_xticklabels(tick_labels)  # Set custom tick labels

        plt.tight_layout()
        path, dirs, files = next(os.walk('tmp_data/mpc'))
        file_count = len(files)
        file_name = f'{god_map.tmp_folder}/mpc/mpc_{joint_name}_{file_count}.png'
        create_path(file_name)
        plt.savefig(file_name)

    @profile
    def _create_debug_pandas(self, qp_solver: QPSolver):
        import pandas as pd
        weights, g, lb, ub, E, bE, A, lbA, ubA, weight_filter, bE_filter, bA_filter = qp_solver.get_problem_data()
        self.free_variable_names = self.free_variable_bounds.names[weight_filter]
        self.equality_constr_names = self.equality_bounds.names[bE_filter]
        self.inequality_constr_names = self.inequality_bounds.names[bA_filter]
        num_vel_constr = len(self.derivative_constraints) * (self.prediction_horizon - 2)
        num_eq_vel_constr = len(self.eq_derivative_constraints) * (self.prediction_horizon - 2)
        num_neq_constr = len(self.inequality_constraints)
        num_eq_constr = len(self.equality_constraints)
        num_constr = num_vel_constr + num_neq_constr + num_eq_constr + num_eq_vel_constr

        self.p_weights = pd.DataFrame(weights, self.free_variable_names, ['data'], dtype=float)
        self.p_g = pd.DataFrame(g, self.free_variable_names, ['data'], dtype=float)
        self.p_lb = pd.DataFrame(lb, self.free_variable_names, ['data'], dtype=float)
        self.p_ub = pd.DataFrame(ub, self.free_variable_names, ['data'], dtype=float)
        self.p_b = pd.DataFrame({'lb': lb, 'ub': ub}, self.free_variable_names, dtype=float)
        if len(bE) > 0:
            self.p_bE_raw = pd.DataFrame(bE, self.equality_constr_names, ['data'], dtype=float)
            self.p_bE = deepcopy(self.p_bE_raw)
            self.p_bE[len(self.equality_bounds.names_derivative_links):] /= self.mpc_dt
        else:
            self.p_bE = pd.DataFrame()
        if len(lbA) > 0:
            self.p_lbA_raw = pd.DataFrame(lbA, self.inequality_constr_names, ['data'], dtype=float)
            self.p_lbA = deepcopy(self.p_lbA_raw)
            self.p_lbA /= self.mpc_dt

            self.p_ubA_raw = pd.DataFrame(ubA, self.inequality_constr_names, ['data'], dtype=float)
            self.p_ubA = deepcopy(self.p_ubA_raw)
            self.p_ubA /= self.mpc_dt

            self.p_bA_raw = pd.DataFrame({'lbA': lbA, 'ubA': ubA}, self.inequality_constr_names, dtype=float)
            self.p_bA = deepcopy(self.p_bA_raw)
            self.p_bA /= self.mpc_dt
        else:
            self.p_lbA = pd.DataFrame()
            self.p_ubA = pd.DataFrame()
        # remove sample period factor
        if len(E) > 0:
            self.p_E = pd.DataFrame(E, self.equality_constr_names, self.free_variable_names, dtype=float)
        else:
            self.p_E = pd.DataFrame()
        if len(A) > 0:
            self.p_A = pd.DataFrame(A, self.inequality_constr_names, self.free_variable_names, dtype=float)
        else:
            self.p_A = pd.DataFrame()
        self.p_xdot = None
        if self.xdot_full is not None:
            self.p_xdot = pd.DataFrame(self.xdot_full, self.free_variable_names, ['data'], dtype=float)
            self.p_b['xdot'] = self.p_xdot
            self.p_b = self.p_b[['lb', 'xdot', 'ub']]
            self.p_pure_xdot = deepcopy(self.p_xdot)
            self.p_pure_xdot[-num_constr:] = 0
            # self.p_Ax = pd.DataFrame(self.p_A.dot(self.p_xdot), self.inequality_constr_names, ['data'], dtype=float)
            if len(self.p_A) > 0:
                self.p_Ax = pd.DataFrame(self.p_A.dot(self.p_pure_xdot), self.inequality_constr_names,
                                         ['data'], dtype=float)
            else:
                self.p_Ax = pd.DataFrame()
            # self.p_Ax_without_slack = deepcopy(self.p_Ax_without_slack_raw)
            # self.p_Ax_without_slack[-num_constr:] /= self.sample_period
            if len(self.p_E) > 0:
                self.p_Ex = pd.DataFrame(self.p_E.dot(self.p_pure_xdot), self.equality_constr_names,
                                         ['data'], dtype=float)
            else:
                self.p_Ex = pd.DataFrame()

        else:
            self.p_xdot = None
        self.p_debug = god_map.debug_expression_manager.to_pandas()

    def _print_iis(self):
        import pandas as pd

        def print_iis_matrix(row_filter: np.ndarray, column_filter: np.ndarray, matrix: pd.DataFrame,
                             bounds: pd.DataFrame):
            if len(row_filter) == 0:
                return
            filtered_matrix = matrix.loc[row_filter, column_filter]
            filtered_matrix['bounds'] = bounds.loc[row_filter]
            print(filtered_matrix)

        result = self.qp_solver.analyze_infeasibility()
        if result is None:
            get_middleware().loginfo(f'Can only compute possible causes with gurobi, '
                                     f'but current solver is {self.qp_solver_class.solver_id.name}.')
            return
        lb_ids, ub_ids, eq_ids, lbA_ids, ubA_ids = result
        b_ids = lb_ids | ub_ids
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
            get_middleware().loginfo('Irreducible Infeasible Subsystem:')
            get_middleware().loginfo('  Free variable bounds')
            free_variables = self.p_lb[b_ids]
            free_variables['ub'] = self.p_ub[b_ids]
            free_variables = free_variables.rename(columns={'data': 'lb'})
            print(free_variables)
            get_middleware().loginfo('  Equality constraints:')
            print_iis_matrix(eq_ids, b_ids, self.p_E, self.p_bE)
            get_middleware().loginfo('  Inequality constraint lower bounds:')
            print_iis_matrix(lbA_ids, b_ids, self.p_A, self.p_lbA)
            get_middleware().loginfo('  Inequality constraint upper bounds:')
            print_iis_matrix(ubA_ids, b_ids, self.p_A, self.p_ubA)
