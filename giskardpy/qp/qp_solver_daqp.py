from ctypes import c_int
from typing import Tuple

import daqp
import numpy as np
from giskardpy.data_types.exceptions import InfeasibleException
from giskardpy.qp.qp_solver_ids import SupportedQPSolver
from line_profiler import profile
from qp.qp_solver_qpalm import QPSolverQPalm
from scipy import sparse as sp


class QPSolverDAQP(QPSolverQPalm):
    """
    min_x 0.5 x^T H x + f^T x
    s.t.  lb <= x <= ub
                Ex = b
          lbA <= Ax <= ubA
    """

    solver_id = SupportedQPSolver.daqp
    sparse = True
    compute_nI_I = False

    @profile
    def update_filters(self):
        self.weight_filter = self.weights != 0
        self.weight_filter[:-self.num_slack_variables] = True
        self.slack_part = self.weight_filter[-(self.num_eq_slack_variables + self.num_neq_slack_variables):]
        self.bE_part = self.slack_part[:self.num_eq_slack_variables]
        self.bA_part = self.slack_part[self.num_eq_slack_variables:]

        self.b_bE_bA_filter = np.ones(self.lb.shape[0] + self.bE.shape[0] + self.lbA.shape[0], dtype=bool)
        self.b_zero_inf_filter_view = self.b_bE_bA_filter[:self.lb.shape[0]]
        self.bE_filter_view = self.b_bE_bA_filter[self.lb.shape[0]:self.lb.shape[0] + self.bE.shape[0]]
        self.bA_filter_view = self.b_bE_bA_filter[self.lb.shape[0] + self.bE.shape[0]:]
        self.bE_bA_filter = self.b_bE_bA_filter[self.lb.shape[0]:]

        self.b_zero_filter = self.weight_filter.copy()
        self.b_zero_inf_filter_view[::] = self.b_zero_filter

        if len(self.bE_part) > 0:
            self.bE_filter_view[-len(self.bE_part):] = self.bE_part

        if len(self.bA_part) > 0:
            self.bA_filter_view[-len(self.bA_part):] = self.bA_part

    @profile
    def problem_data_to_qp_format(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        H = np.diag(self.weights + 0.00001)
        A = self.A.toarray()
        return H, self.g, A, self.lb_bE_lbA, self.ub_bE_ubA

    @profile
    def solver_call(self, H: np.ndarray, g: np.ndarray, A: sp.csc_matrix, lbA: np.ndarray, ubA: np.ndarray) -> np.ndarray:
        sense = np.zeros(lbA.shape, dtype=c_int)
        sense[len(self.b_zero_inf_filter_view):len(self.b_zero_inf_filter_view)+len(self.bE_filter_view)] = 5
        (xstar, fval, exitflag, info) = daqp.solve(H, g, A, ubA, lbA, sense)
        if exitflag != 1:
            raise InfeasibleException(f'failed to solve qp {exitflag}')
        return xstar
