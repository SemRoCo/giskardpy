from enum import IntEnum

import numpy as np
import qpoases
from qpoases import PyReturnValue

from giskardpy.configs.data_types import SupportedQPSolver
from giskardpy.exceptions import QPSolverException, InfeasibleException, HardConstraintsViolatedException
from giskardpy.qp.qp_solver import QPSolver
from giskardpy.utils import logging
from giskardpy.utils.utils import record_time


class QPoasesModes(IntEnum):
    MPC = 1
    Fast = 2
    Default = 3
    Reliable = 4


class QPSolverQPOases(QPSolver):
    solver_id = SupportedQPSolver.qpOASES
    STATUS_VALUE_DICT = {value: name for name, value in vars(PyReturnValue).items()}

    def __init__(self, num_non_slack: int, retry_added_slack: float, retry_weight_factor: float,
                 retries_with_relaxed_constraints: int):
        super().__init__(num_non_slack, retry_added_slack, retry_weight_factor, retries_with_relaxed_constraints)
        self.started = False
        self.mode = QPoasesModes.MPC
        self.shape = (0, 0)

    def init(self, dim_a, dim_b):
        self.qpProblem = qpoases.PySQProblem(dim_a, dim_b)
        options = qpoases.PyOptions()
        if self.mode == QPoasesModes.MPC:
            options.setToMPC()
        elif self.mode == QPoasesModes.Fast:
            options.setToFast()
        elif self.mode == QPoasesModes.Reliable:
            options.setToReliable()
        else:
            options.setToDefault()
        options.printLevel = qpoases.PyPrintLevel.NONE
        self.qpProblem.setOptions(options)
        self._xdot_full = np.zeros(dim_a)
        self.started = False

    def did_problem_change(self, A):
        return A.shape != self.shape

    @profile
    @record_time
    def solve(self, weights: np.ndarray, g: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray, lbA: np.ndarray,
              ubA: np.ndarray) -> np.ndarray:
        H = np.diag(weights).copy()
        A = A.copy()
        lbA = lbA.copy()
        ubA = ubA.copy()
        lb = lb.copy()
        ub = ub.copy()
        nWSR = np.array([sum(A.shape) * 2])

        if self.did_problem_change(A):
            self.started = False
            self.mode = QPoasesModes.MPC
            self.shape = A.shape

        if not self.started:
            self.init(A.shape[1], A.shape[0])
            success = self.qpProblem.init(H, g, A, lb, ub, lbA, ubA, nWSR)
        else:
            success = self.qpProblem.hotstart(H, g, A, lb, ub, lbA, ubA, nWSR)
        if success == PyReturnValue.SUCCESSFUL_RETURN:
            self.started = True
            self.qpProblem.getPrimalSolution(self._xdot_full)
            return self._xdot_full
        self.started = False
        if success in [PyReturnValue.INIT_FAILED_INFEASIBILITY,
                       PyReturnValue.QP_INFEASIBLE,
                       PyReturnValue.HOTSTART_STOPPED_INFEASIBILITY,
                       PyReturnValue.ADDBOUND_FAILED_INFEASIBILITY,
                       PyReturnValue.ADDCONSTRAINT_FAILED_INFEASIBILITY]:
            raise InfeasibleException(self.STATUS_VALUE_DICT[success], success)
        raise QPSolverException(self.STATUS_VALUE_DICT[success], success)

    @profile
    def solve_and_retry(self, weights, g, A, lb, ub, lbA, ubA):
        relaxed = False
        for number_of_retries in range(2 + len(QPoasesModes)):
            try:
                return self.solve(weights, g, A, lb, ub, lbA, ubA)
            except QPSolverException as e:
                if number_of_retries == 0:
                    logging.loginfo(f'{e}; retrying with A rounded to {self.on_fail_round_to} decimal places')
                    weights = np.round(weights, self.on_fail_round_to)
                    A = np.round(A, self.on_fail_round_to)
                    lb = np.round(lb, self.on_fail_round_to)
                    ub = np.round(ub, self.on_fail_round_to)
                    lbA = np.round(lbA, self.on_fail_round_to)
                    ubA = np.round(ubA, self.on_fail_round_to)
                elif isinstance(e, InfeasibleException) and not relaxed:
                    logging.loginfo(f'{e}; retrying with relaxed hard constraints')
                    try:
                        weights, lb, ub = self.compute_relaxed_hard_constraints(weights, g, A, lb, ub, lbA, ubA)
                        relaxed = True
                    except InfeasibleException as e2:
                        if isinstance(e2, HardConstraintsViolatedException):
                            raise e2
                        raise e
                else:
                    self.mode = QPoasesModes(self.mode + 1)
                    logging.loginfo(f'{e}; retrying with {repr(self.mode)} mode.')
        raise e
