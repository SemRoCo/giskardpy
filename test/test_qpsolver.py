import numpy as np

from giskardpy.qp_solver import QPSolver


def test_simple_problem():
    H = np.eye(2) * 2
    A = np.ones((1, 2))
    g = np.zeros(2)
    lba = np.array([10.])
    lb = np.array([-10., -10.])
    ub = np.array([10., 10.])

    qp = QPSolver()
    x = qp.solve(H, g, A, lb, ub, lba, lba)
    np.testing.assert_array_almost_equal(x, np.array([5, 5]), decimal=4)


def test_simple_problem_split():
    cw1 = 10.
    cw2 = 1000.
    H = np.diag([1, 1, cw1, cw2])
    A = np.array([[1., 1, 1, 0],
                  [1, 1, 0, 1]])
    g = np.zeros(4)
    lba = np.array([10., 5.])
    lb = np.array([-10., -10., -1e9, -1e9])
    ub = np.array([10., 10., 1e9, 1e9])

    qp = QPSolver()
    x1 = qp.solve(H, g, A, lb, ub, lba, lba)
    print(x1)

    H = np.diag([1, 1, cw1])
    A = np.array([[1., 1, 1]])
    g = np.zeros(3)
    lba = np.array([10.])
    lb = np.array([-10., -10., -1e9])
    ub = np.array([10., 10., 1e9])

    qp = QPSolver()
    x2 = qp.solve(H, g, A, lb, ub, lba, lba)
    print(x2)

    H = np.diag([1, 1, cw2])
    A = np.array([[1., 1, 1]])
    g = np.zeros(3)
    lba = np.array([5.])
    lb = np.array([-10., -10., -1e9])
    ub = np.array([10., 10., 1e9])

    qp = QPSolver()
    x3 = qp.solve(H, g, A, lb, ub, lba, lba)
    print(x3)

    H = np.diag([1., 1, cw1, cw1, cw2, cw2])
    A = np.array([[1., 0, 1, 0, 0, 0],
                  [0., 1, 0, 1, 0, 0],
                  [1., 0, 0, 0, 1, 0],
                  [0., 1, 0, 0, 0, 1]])
    g = np.zeros(H.shape[0])
    lba = np.array([x2[0], x2[1],
                    x3[0], x3[1], ])
    lb = np.array([-10., -10., -1e9, -1e9, -1e9, -1e9])
    ub = np.array([10., 10., 1e9, 1e9, 1e9, 1e9])

    qp = QPSolver()
    x4 = qp.solve(H, g, A, lb, ub, lba, lba)
    print(x4)

    H = np.diag([1., 1, cw1, cw2, cw1, cw2])
    A = np.array([[1., 0, 1, 0, 0, 0],
                  [0., 1, 1, 0, 0, 0],
                  [1., 0, 0, 1, 0, 0],
                  [0., 1, 0, 1, 0, 0]])
    g = np.zeros(H.shape[0])
    lba = np.array([x2[0], x2[1],
                    x3[0], x3[1], ])
    lb = np.array([-10., -10., -1e9, -1e9, -1e9, -1e9])
    ub = np.array([10., 10., 1e9, 1e9, 1e9, 1e9])

    qp = QPSolver()
    x5 = qp.solve(H, g, A, lb, ub, lba, lba)
    print(x5)
    print((x2[:2] + x3[:2]) / 2)
    print(np.sqrt((x1[:2] ** 2 + x2[:2] ** 2)) / 2)
    # np.testing.assert_array_almost_equal(x, np.array([5.,5.]), decimal=4)


def test_non_diagonal_H():
    H = np.array([[1, -0.5],
                  [-1, -1]])
    A = np.ones((1, 2))
    g = np.zeros(2)
    lba = np.array([-10.])
    lb = np.array([-10., -10.])
    ub = np.array([10., 10.])

    qp = QPSolver()
    x = qp.solve(H, g, A, lb, ub, lba, lba)
    print(x)
    print(qp.qpProblem.getObjVal())
    # np.testing.assert_array_almost_equal(x, np.array([5,5]), decimal=4)
