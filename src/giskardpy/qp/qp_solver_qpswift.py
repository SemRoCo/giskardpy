import abc
from collections import defaultdict
from enum import IntEnum
from typing import Tuple, Iterable, List, Union, Optional, Dict

import numpy as np

from giskardpy.configs.qp_controller_config import SupportedQPSolver
from giskardpy.exceptions import QPSolverException, InfeasibleException, HardConstraintsViolatedException
from giskardpy.qp.qp_solver import QPSolver, record_solver_call_time, QPSWIFTFormatter
import qpSWIFT
import giskardpy.casadi_wrapper as cas
import scipy.sparse as sp

from giskardpy.utils.decorators import record_time


class QPSWIFTExitFlags(IntEnum):
    Optimal = 0  # Solution Found
    KKT_Failure = 1  # Failure in factorising KKT matrix
    MAX_ITER_REACHED = 2  # Maximum Number of Iterations Reached
    ERROR = 3  # Unknown Problem in Solver



class QPSolverQPSwift(QPSWIFTFormatter):
    """
    min_x 0.5 x^T P x + c^T x
    s.t.  Ax = b
          Gx <= h
    """
    solver_id = SupportedQPSolver.qpSWIFT
    _times: Dict[Tuple[int, int, int], list] = defaultdict(list)

    opts = {
        'OUTPUT': 1,  # 0 = sol; 1 = sol + basicInfo; 2 = sol + basicInfo + advInfo
        # 'MAXITER': 100,  # 0 < MAXITER < 200; default 100. maximum number of iterations needed
        # 'ABSTOL': 9e-4,  # 0 < ABSTOL < 1; default 1e-6. absolute tolerance
        'RELTOL': 2e-5,  # 0 < RELTOL < 1; default 1e-6. relative tolerance
        # 'SIGMA': 0.01,  # default 100. maximum centering allowed
        # 'VERBOSE': 1  # 0 = no print; 1 = print
    }

    @profile
    def solver_call(self, H: np.ndarray, g: np.ndarray, E: sp.csc_matrix, b: np.ndarray, A: sp.csc_matrix,
                    h: np.ndarray) -> np.ndarray:
        A = A.toarray()
        E = E.toarray()
        result = qpSWIFT.run(c=g, h=h, P=H, G=A, A=E, b=b, opts=self.opts)
        exit_flag = result['basicInfo']['ExitFlag']
        if exit_flag != 0:
            error_code = QPSWIFTExitFlags(exit_flag)
            if error_code == QPSWIFTExitFlags.MAX_ITER_REACHED:
                raise InfeasibleException(f'Failed to solve qp: {str(error_code)}')
            raise QPSolverException(f'Failed to solve qp: {str(error_code)}')
        return result['sol']

    def default_interface_solver_call(self, H, g, lb, ub, E, bE, A, lbA, ubA) -> np.ndarray:
        A_lb_ub = np.eye(len(ub))
        if len(A) > 0:
            A_lb_ub = np.vstack((-A_lb_ub, A_lb_ub, -A, A))
            h = np.concatenate((-lb, ub, -lbA, ubA))
        else:
            A_lb_ub = np.vstack((-A_lb_ub, A_lb_ub))
            h = np.concatenate((-lb, ub))
        h_filter = np.isfinite(h)
        h = h[h_filter]
        A_lb_ub = A_lb_ub[h_filter, :]
        bE_filter = np.isfinite(bE)
        E = E[bE_filter, :]
        if len(E) == 0:
            result = qpSWIFT.run(c=g, h=h, P=H, G=A_lb_ub, opts=self.opts)
        else:
            result = qpSWIFT.run(c=g, h=h, P=H, G=A_lb_ub, A=E, b=bE, opts=self.opts)
        exit_flag = result['basicInfo']['ExitFlag']
        if exit_flag != 0:
            error_code = QPSWIFTExitFlags(exit_flag)
            if error_code == QPSWIFTExitFlags.MAX_ITER_REACHED:
                raise InfeasibleException(f'Failed to solve qp: {str(error_code)}')
            raise QPSolverException(f'Failed to solve qp: {str(error_code)}')
        return result['sol']

