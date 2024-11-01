from typing import Union

import cvxopt
import numpy as np
from giskardpy.qp.qp_solver_ids import SupportedQPSolver
from qp.qp_solver_qpswift import QPSolverQPSwift
from scipy import sparse as sp
from line_profiler import profile

from giskardpy.qp.qp_solver import QPSolver

__infty__ = 1e20  # 1e20 tends to yield division-by-zero errors


def _to_cvxopt(
        M: Union[np.ndarray, sp.csc_matrix]
) -> Union[cvxopt.matrix, cvxopt.spmatrix]:
    """Convert matrix to CVXOPT format.

    Parameters
    ----------
    M :
        Matrix in NumPy or CVXOPT format.

    Returns
    -------
    :
        Matrix in CVXOPT format.
    """
    if isinstance(M, np.ndarray):
        M_noinf = np.nan_to_num(M, posinf=__infty__, neginf=-__infty__)
        return cvxopt.matrix(M_noinf)
    coo = M.tocoo()
    return cvxopt.spmatrix(
        coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=M.shape
    )


cvxopt.solvers.options["show_progress"] = False  # disable verbose output


class QPSolverCVXOPT(QPSolverQPSwift):
    """
    min_x 0.5 x^T P x + q^T x
    s.t.  Ax = b
          Gx <= h
    """

    solver_id = SupportedQPSolver.cvxopt

    @profile
    def solver_call(self, H: np.ndarray, g: np.ndarray, E: sp.csc_matrix, b: np.ndarray, A: sp.csc_matrix,
                    h: np.ndarray) -> np.ndarray:
        H = _to_cvxopt(H)
        A = _to_cvxopt(A)
        E = _to_cvxopt(E)
        g = _to_cvxopt(g)
        h = _to_cvxopt(h)
        b = _to_cvxopt(b)
        return np.array(cvxopt.solvers.qp(P=H, q=g, G=A, h=h, A=E, b=b, verbose=False)['x']).flatten()

