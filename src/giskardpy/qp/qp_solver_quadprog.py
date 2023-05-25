import numpy as np
import quadprog

from giskardpy.configs.data_types import SupportedQPSolver
from giskardpy.exceptions import InfeasibleException
from giskardpy.qp.qp_solver import QPSolver


class QPSolverQuadprog(QPSolver):
    solver_id = SupportedQPSolver.quadprog
    """
    min_x 0.5 x^T G x + a^T x
    s.t.  Cx <= b
    """

    def solve(self, weights: np.ndarray, g: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray, lbA: np.ndarray,
              ubA: np.ndarray) -> np.ndarray:
        G, a, C_ueq, b_ueq, C_eq, b_eq = self.transform_qpSWIFT(weights, g, A, lb, ub, lbA, ubA)
        C = np.vstack([C_eq, C_ueq])
        b = np.concatenate([b_eq, b_ueq])
        try:
            return quadprog.solve_qp(G=G, a=a, C=C.T, b=b, meq=b_eq.shape[0])[0]
        except ValueError as e:
            raise InfeasibleException(str(e))

    def solve_and_retry(self, weights: np.ndarray, g: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray,
                        lbA: np.ndarray, ubA: np.ndarray) -> np.ndarray:
        return self.solve(weights, g, A, lb, ub, lbA, ubA)
