from time import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

from giskardpy.qp_solver import QPSolver
from giskardpy.qp_solver_gurobi import QPSolverGurobi


def limit(l,v):
    return max(-l,min(l,v))

def viz(ts, positions, velocities, accelerations, jerks):
    f, axs = plt.subplots(4, sharex=True)
    axs[0].set_title('position')
    axs[0].plot(ts, positions, 'b')
    axs[0].grid()
    axs[1].set_title('velocity')
    axs[1].plot(ts, velocities, 'b')
    axs[1].grid()
    axs[2].set_title('acceleration')
    axs[2].plot(ts, accelerations, 'b')
    axs[2].grid()
    axs[3].set_title('jerk')
    axs[3].plot(ts, jerks, 'b')
    plt.grid()
    plt.show()


def viz2(x, h, sample_period, start_pos):
    ts = np.array([i * sample_period for i in range(h)])
    velocities = x[:h]
    accelerations = x[h:h * 2]
    jerks = x[h * 2:h * 3]
    positions = [start_pos]
    for x_ in velocities[:-1]:
        positions.append(positions[-1] + x_ * sample_period)

    f, axs = plt.subplots(4, sharex=True)
    axs[0].set_title('position')
    axs[0].plot(ts, positions, 'b')
    axs[0].grid()
    axs[1].set_title('velocity')
    axs[1].plot(ts, velocities, 'b')
    axs[1].grid()
    axs[2].set_title('acceleration')
    axs[2].plot(ts, accelerations, 'b')
    axs[2].grid()
    axs[3].set_title('jerk')
    axs[3].plot(ts, jerks, 'b')
    plt.grid()
    plt.show()


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


def test_mpc():
    h = 100
    sample_period = 0.05
    goal = 1
    lpos_limt = -100
    upos_limt = 100
    velocity_limit = 0.5
    v0 = 0.0
    acceleration_limit = 999
    jerk_limit = 1
    H = np.zeros((h * 3, h * 3))
    H[:h,:h] = np.identity(h) * np.diag([i for i in range(h)])
    A = np.zeros((h * 3, h * 3))
    # A[:h,:h] = np.tril(np.ones((h,h))*sample_period)
    A[h - 1] = np.concatenate((np.ones(h) * sample_period, np.zeros(h), np.zeros(h)))
    # A[h] = np.concatenate((np.zeros(h), np.ones(h), np.zeros(h)))
    # A[h + 1] = np.concatenate((np.zeros(h), np.zeros(h), np.ones(h)))
    link1 = np.identity(h)
    link1[1:, :-1] += -np.identity(h - 1)
    link2 = -np.identity(h) * sample_period
    A[h:h * 2, :h] = link1
    A[h:h * 2, h:h * 2] = link2
    A[h + h:h * 3, h:h * 2] = link1
    A[h + h:h * 3, h * 2:h * 3] = link2

    g = np.zeros(h * 4)
    lba = np.zeros(2 + h * 3)
    lba[:h - 1] = lpos_limt  # velocity integral
    lba[h - 1] = goal  # velocity integral
    lba[h] = v0  # start velocity
    uba = np.copy(lba)
    uba[:h - 1] = upos_limt  # velocity integral
    ub = np.concatenate((np.ones(h) * velocity_limit,
                         np.ones(h) * acceleration_limit,
                         np.ones(h) * jerk_limit))
    lb = -ub

    qp = QPSolverGurobi()
    t = time()
    asdf = 1
    for i in range(asdf):
        print(i)
        x = qp.solve(H, g, A, lb, ub, lba, uba)
    t = (time() - t) / asdf
    print(t)
    viz2(x, h, sample_period, start_pos=0)

def test_mpc_osqp():
    import osqp
    h = 100
    sample_period = 0.05
    goal = 1
    lpos_limt = -100
    upos_limt = 100
    velocity_limit = 0.5
    v0 = 0.0
    acceleration_limit = 999
    jerk_limit = 1
    H = np.zeros((h * 3, h * 3))
    H[:h, :h] = np.identity(h) * np.diag([100*(i + 1) for i in range(h)])
    A = np.zeros((2 + h * 3, h * 3))
    # A[:h,:h] = np.tril(np.ones((h,h))*sample_period)
    A[h - 1] = np.concatenate((np.ones(h) * sample_period, np.zeros(h), np.zeros(h)))
    A[h] = np.concatenate((np.zeros(h), np.ones(h) * sample_period, np.zeros(h)))
    A[h + 1] = np.concatenate((np.zeros(h), np.zeros(h), np.ones(h)))
    link1 = np.identity(h)
    link1[1:, :-1] += -np.identity(h - 1)
    link2 = -np.identity(h) * sample_period
    A[2 + h:2 + h * 2, :h] = link1
    A[2 + h:2 + h * 2, h:h * 2] = link2
    A[2 + h + h:2 + h * 3, h:h * 2] = link1
    A[2 + h + h:2 + h * 3, h * 2:h * 3] = link2

    g = np.zeros(h * 3)
    lba = np.zeros(2 + h * 3)
    lba[:h - 1] = -lpos_limt  # velocity integral
    lba[h - 1] = goal  # velocity integral
    lba[h] = -v0  # acceleration integral
    lba[h + 1] = 0  # jerk integral
    lba[h + 2] = v0  # start velocity
    uba = np.copy(lba)
    uba[:h - 1] = upos_limt  # velocity integral
    ub = np.concatenate((np.ones(h) * velocity_limit,
                         np.ones(h) * acceleration_limit,
                         np.ones(h) * jerk_limit))
    lb = -ub

    qp = osqp.OSQP()

    I = np.identity(len(lb))
    A = np.concatenate((I, A))
    lba = np.concatenate((lb, lba))
    uba = np.concatenate((ub, uba))
    Hs = sparse.csc_matrix(H)
    AIs = sparse.csc_matrix(A)
    qp.setup(P=Hs, q=g, A=AIs, l=lba, u=g,
             # rho=10
             adaptive_rho=False,
             polish=True,
             verbose=False)

    t = time()
    asdf = 1
    for i in range(asdf):
        print(i)
        t2 = time()
        Hs = sparse.csc_matrix(H)
        AIs = sparse.csc_matrix(A)
        t2 = (time() - t2) / asdf
        print(t2)
        qp.update(Px=Hs.data, Ax=AIs.data, l=lba, u=uba)
        x = qp.solve().x
        # x = qp.solve(H, g, A, lb, ub, lba, uba)
    t = (time() - t) / asdf
    print(t)

    ts = np.array([i * sample_period for i in range(h)])

    velocities = x[:h]
    accelerations = x[h:h * 2]
    jerk = x[h * 2:]
    positions = [0]
    for x_ in velocities[:-1]:
        positions.append(positions[-1] + x_ * sample_period)

    print('final position: {}'.format(positions[-1]))

    f, axs = plt.subplots(4, sharex=True)
    axs[0].set_title('position')
    axs[0].plot(ts, positions, 'b')
    axs[0].grid()
    axs[1].set_title('velocity')
    axs[1].plot(ts, velocities, 'b')
    axs[1].grid()
    axs[2].set_title('acceleration')
    axs[2].plot(ts, accelerations, 'b')
    axs[2].grid()
    axs[3].set_title('jerk')
    axs[3].plot(ts, jerk, 'b')
    plt.title('osqp')
    plt.grid()
    plt.show()


def test_mpc2():
    h = 100
    ch = 1
    sample_period = 0.05
    goal = 5
    start_pose = -0.5
    upos_limit = 0.5
    lpos_limit = -0.5
    velocity_limit = 0.5
    v0 = 0
    a0 = 0
    acceleration_limit = 999
    jerk_limit = 1
    H = np.zeros((h * 3 + 1, h * 3 + 1))
    # H[:h,:h] = np.identity(h) * np.diag([(np.log(x+1)) for x in range(1,h+1)])
    # H[:h, :h] = np.identity(h) * np.diag([1.11**x for x in range(1, h+1)])
    H[:h, :h] = np.identity(h) * np.diag([x for x in range(h)])
    # H[:h, :h] = np.identity(h) * np.diag([int(x/10)*10 for x in range(h)])
    # H[h:h*2, h:h*2] = np.identity(h) * np.diag([x for x in range(h)])
    # H[h:h*2, h:h*2] = np.identity(h) * np.diag([0 if x < h/4 else 1 for x in range(h)])
    # H[:h,:h] = np.identity(h) * np.diag([x+(np.log(x+1)) for x in range(1,h+1)])
    H[-1, -1] = 100
    A = np.zeros((2 + h * 3 + 1, h * 3 + 1))
    # A[:h,:h] = np.tril(np.ones((h,h))*sample_period)
    A[0] = np.concatenate(([sample_period], np.zeros(h - 1), np.zeros(h), np.zeros(h), [0]))
    A[h - 1] = np.concatenate((np.ones(h) * sample_period, np.zeros(h), np.zeros(h), [0]))
    # A[h] = np.concatenate((np.zeros(h), np.ones(h)*sample_period, np.zeros(h), [0]))
    # A[h+1] = np.concatenate((np.zeros(h), np.zeros(h), np.ones(h), [0]))
    link1 = np.identity(h)
    link1[1:, :-1] += -np.identity(h - 1)
    link2 = -np.identity(h) * sample_period
    A[2 + h:2 + h * 2, :h] = link1
    A[2 + h:2 + h * 2, h:h * 2] = link2
    A[2 + h + h:2 + h * 3, h:h * 2] = link1
    A[2 + h + h:2 + h * 3, h * 2:h * 3] = link2

    A[-1] = np.concatenate((np.ones(ch)*sample_period, np.zeros(h+h-ch), np.zeros(h), [sample_period]))
    # A[-1, 0] = sample_period
    # A[-1, -1] = ch*sample_period

    g = np.zeros(h * 4 + 1)
    lba = np.zeros(2 + h * 3 + 1)
    lba[:h] = lpos_limit - start_pose  # velocity integral
    # lba[h] = -v0 # acceleration integral
    # lba[h+1] = 0 # jerk integral
    lba[h + 2] = v0  # start velocity
    lba[h * 2 + 2] = a0  # start acc
    lba[-1] = limit(velocity_limit*sample_period*ch, goal - start_pose)#*sample_period
    uba = np.copy(lba)
    uba[:h] = upos_limit - start_pose
    ub = np.concatenate((
        # np.ones(h - 1) * velocity_limit, [0],
        # np.ones(h - 1) * acceleration_limit, [0],
        np.ones(h) * velocity_limit,
        np.ones(h) * acceleration_limit,
        np.ones(h) * jerk_limit,
        [9999]))
    lb = -ub

    qp = QPSolver()
    x = qp.solve(H, g, A, lb, ub, lba, uba)
    # viz2(x, h, sample_period, start_pose)
    # assert False

    # ts = np.array([i*sample_period for i in range(h)])

    velocities = [v0]
    accelerations = [a0]
    jerks = [0]
    ts = [0]
    positions = [start_pose]
    try:
        for i in range(150):
            lba[-1] = limit(velocity_limit*sample_period*ch, goal - positions[-1])#*sample_period
            uba[-1] = limit(velocity_limit*sample_period*ch, goal - positions[-1])#*sample_period
            lba[:h] = lpos_limit - positions[-1]
            uba[:h] = upos_limit - positions[-1]
            # lba[h] = -velocities[-1]
            # uba[h] = -velocities[-1]
            # lba[h +1] = -accelerations[-1]  # jerk integral
            # uba[h +1] = -accelerations[-1]  # jerk integral
            lba[h + 2] = velocities[-1]  # start velocity
            uba[h + 2] = velocities[-1]  # start velocity
            lba[h * 2 + 2] = accelerations[-1]  # start acc
            uba[h * 2 + 2] = accelerations[-1]  # start acc
            x = qp.solve(H, g, A, lb, ub, lba, uba)
            jerks.append(x[h * 2])
            accelerations.append(x[h])
            velocities.append(x[0])
            # accelerations.append(accelerations[-1]+jerks[-1]*sample_period)
            # velocities.append(velocities[-1] + accelerations[-1]*sample_period)
            positions.append(positions[-1] + velocities[-1] * sample_period)
            ts.append(i * sample_period)
    except Exception as e:
        print(e)

    print('final position: {}'.format(positions[-1]))
    viz(ts, positions, velocities, accelerations, jerks)

def test_mpc3():
    h = 100
    sample_period = 0.05
    goal = 0.5
    start_pose = -0.5
    upos_limit = 0.5
    lpos_limit = -0.5
    velocity_limit = 0.5
    v0 = 0
    a0 = 0
    acceleration_limit = 999
    jerk_limit = 1
    H = np.zeros((h * 2 + 1, h * 2 + 1))
    # H[:h,:h] = np.identity(h) * np.diag([(np.log(x+1)) for x in range(1,h+1)])
    # H[:h, :h] = np.identity(h) * np.diag([1.11**x for x in range(1, h+1)])
    H[:h, :h] = np.identity(h) * np.diag([x for x in range(h)])
    # H[:h,:h] = np.identity(h) * np.diag([x+(np.log(x+1)) for x in range(1,h+1)])
    H[-1, -1] = 3000
    A = np.zeros((2 + h + 1, h * 2 + 1))
    # A[:h,:h] = np.tril(np.ones((h,h))*sample_period)
    A[0] = np.concatenate(([sample_period], np.zeros(h - 1), np.zeros(h), [0]))
    A[1] = np.concatenate((np.ones(h) * sample_period, np.zeros(h), [0]))
    # A[h] = np.concatenate((np.zeros(h), np.ones(h)*sample_period, np.zeros(h), [0]))
    # A[h+1] = np.concatenate((np.zeros(h), np.zeros(h), np.ones(h), [0]))
    link1 = -np.identity(h)
    link1[1:, :-1] += np.identity(h - 1)
    # link2 = -np.identity(h) * sample_period
    link3 = np.tril(np.ones((h,h))*sample_period)
    A[2:2 + h, :h] = link1
    # A[2 + h:2 + h * 2, h:h * 2] = link2
    # A[2 + h + h:2 + h * 3, h:h * 2] = link1
    A[2 :2 + h, h:h * 2] = link3

    # A[-1] = np.concatenate((np.ones(h)*sample_period, np.zeros(h), np.zeros(h), [1]))
    A[-1, 0] = 1
    A[-1, -1] = 1

    g = np.zeros(h * 2 + 1)
    lba = np.zeros(2 + h + 1)
    lba[0] = lpos_limit - start_pose  # velocity integral
    lba[1] = lpos_limit - start_pose  # velocity integral
    lba[2] = -v0-a0  # start velocity
    lba[3:-1] = -a0  # start acc
    lba[-1] = goal - start_pose
    uba = np.copy(lba)
    uba[0] = upos_limit - start_pose
    uba[1] = upos_limit - start_pose
    ub = np.concatenate((
        # np.ones(h - 1) * velocity_limit, [0],
        # np.ones(h - 1) * acceleration_limit, [0],
        np.ones(h) * velocity_limit,
        # np.ones(h) * acceleration_limit,
        np.ones(h) * jerk_limit,
        [9999]))
    lb = -ub

    qp = QPSolver()
    x = qp.solve(H, g, A, lb, ub, lba, uba)
    # viz2(x, h, sample_period, start_pose)
    # assert False

    # ts = np.array([i*sample_period for i in range(h)])

    velocities = [v0]
    accelerations = [a0]
    jerks = [0]
    ts = [0]
    positions = [start_pose]
    try:
        for i in range(150):
            lba[-1] = goal - positions[-1]
            uba[-1] = goal - positions[-1]
            lba[:2] = lpos_limit - positions[-1]
            uba[:2] = upos_limit - positions[-1]
            # lba[h] = -velocities[-1]
            # uba[h] = -velocities[-1]
            # lba[h +1] = -accelerations[-1]  # jerk integral
            # uba[h +1] = -accelerations[-1]  # jerk integral
            lba[2] = -velocities[-1]-accelerations[-1]  # start velocity
            uba[2] = -velocities[-1]-accelerations[-1]  # start velocity
            lba[3:-1] = -accelerations[-1]  # start acc
            uba[3:-1] = -accelerations[-1]  # start acc
            x = qp.solve(H, g, A, lb, ub, lba, uba)
            jerks.append(x[h])
            accelerations.append(accelerations[-1] + jerks[-1]*sample_period)
            velocities.append(x[0])
            # accelerations.append(accelerations[-1]+jerks[-1]*sample_period)
            # velocities.append(velocities[-1] + accelerations[-1]*sample_period)
            positions.append(positions[-1] + velocities[-1] * sample_period)
            ts.append(i * sample_period)
    except Exception as e:
        print(e)

    print('final position: {}'.format(positions[-1]))
    viz(ts, positions, velocities, accelerations, jerks)


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
