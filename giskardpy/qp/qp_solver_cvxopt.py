from typing import Union

import cvxopt
import mosek
import numpy as np
from cvxopt import solvers
from scipy import sparse

from giskardpy.qp.qp_solver import QPSolver

__infty__ = 1e20  # 1e20 tends to yield division-by-zero errors


def _to_cvxopt(
        M: Union[np.ndarray, sparse.csc_matrix]
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


solvers.options['mosek'] = {mosek.iparam.log: 0}
cvxopt.solvers.options["show_progress"] = False  # disable verbose output


class QPSolverCVXOPT(QPSolver):
    """
    min_x 0.5 x^T P x + q^T x
    s.t.  Ax = b
          Gx <= h
    """

    @profile
    def solve(self, weights: np.ndarray, g: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray, lbA: np.ndarray,
              ubA: np.ndarray) -> np.ndarray:
        A_b = np.eye(lb.shape[0])
        G = _to_cvxopt(np.vstack([-A_b, A_b, -A, A]))
        P = _to_cvxopt(np.diag(weights))
        h = np.concatenate([-lb, ub, -lbA, ubA])
        g = _to_cvxopt(g)
        h = _to_cvxopt(h)
        # return np.array(cvxopt.solvers.qp(P=P, q=g, G=G, h=h, solver='mosek')['x']).reshape(weights.shape[0])
        return np.array(cvxopt.solvers.qp(P=P, q=g, G=G, h=h, verbose=False)['x']).reshape(weights.shape[0])

    # @profile
    def solve_and_retry(self, weights, g, A, lb, ub, lbA, ubA):
        return self.solve(weights, g, A, lb, ub, lbA, ubA)
