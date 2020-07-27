import numpy as np
from giskardpy.qp_solver import QPSolver


def test_simple_problem():
    H = np.eye(2) * 2
    A = np.ones((1,2))
    g = np.zeros(2)
    lba = np.array([10.])
    lb = np.array([-10., -10.])
    ub = np.array([10., 10.])

    qp = QPSolver(1,2,0)
    x = qp.solve(H, g, A, lb, ub, lba, lba)
    np.testing.assert_array_almost_equal(x, np.array([5,5]), decimal=4)

def test_non_diagonal_H():
    H = np.array([[1,-0.5],
                  [-1,-1]])
    A = np.ones((1,2))
    g = np.zeros(2)
    lba = np.array([-10.])
    lb = np.array([-10., -10.])
    ub = np.array([10., 10.])

    qp = QPSolver(1,2,0)
    x = qp.solve(H, g, A, lb, ub, lba, lba)
    print(x)
    print(qp.qpProblem.getObjVal())
    # np.testing.assert_array_almost_equal(x, np.array([5,5]), decimal=4)
