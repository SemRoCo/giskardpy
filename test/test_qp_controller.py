import traceback
from copy import deepcopy
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

import giskardpy.casadi_wrapper as ca
from giskardpy.data_types import FreeVariable, Constraint, VelocityConstraint
from giskardpy.qp_controller import QPController


def simulate(start_state, qp_controller, sample_period, print_traj=False, time_limit=6., name='',
             save=False):
    num_free_variables = len(qp_controller.free_variables)
    state = deepcopy(start_state)
    traj = [[[] for __ in range(num_free_variables)] for _ in range(4)]
    time = []
    for t in range(int(1 / sample_period * time_limit)):
        try:
            state['time'] = t
            subs = [state[x] for x in qp_controller.compiled_big_ass_M.str_params]
            [cmd_vel, cmd_acc, cmd_jerk], _ = qp_controller.get_cmd(subs)
            for i, (free_variable, cmd) in enumerate(cmd_vel.items()):
                state[free_variable] += cmd * sample_period
                traj[0][i].append(state[free_variable])
            all_zero = True
            for i, (free_variable, cmd) in enumerate(cmd_vel.items()):
                state[free_variable + '_v'] = cmd
                traj[1][i].append(cmd)
                all_zero &= abs(cmd) < 1e-2
            for i, (free_variable, cmd) in enumerate(cmd_acc.items()):
                state[free_variable + '_a'] = cmd
                traj[2][i].append(cmd)
            for i, (free_variable, cmd) in enumerate(cmd_jerk.items()):
                state[free_variable + '_j'] = cmd
                traj[3][i].append(cmd)
            time.append(t * sample_period)
            if all_zero:
                break
        except Exception as e:
            traceback.print_exc()

    if print_traj:
        f, axs = plt.subplots(4 * num_free_variables, sharex=True)
        f.set_size_inches(w=7, h=9)
        for j in range(num_free_variables):
            for i in range(4):
                index = i + (j * 4)
                axs[index].set_ylabel('{}:{}'.format(qp_controller.free_variables[j].name, i))
                axs[index].plot(time, traj[i][j])
                ticks = [0]
                min_ = np.round(min(traj[i][j]), 4)
                max_ = np.round(max(traj[i][j]), 4)
                range_ = max_ - min_
                if min_ < -range_ / 8:
                    ticks.append(min_)
                    if min_ < -range_ / 3:
                        ticks.append(min_ / 2)
                if max_ > range_ / 8:
                    ticks.append(max_)
                    if max_ > range_ / 3:
                        ticks.append(max_ / 2)
                axs[index].set_yticks(ticks)
                axs[index].grid()
        plt.title(name)
        plt.tight_layout()
        if save:
            plt.savefig('tmp_data/results/l_{}_{}.png'.format(len(traj[0][0]), name))
        else:
            f.show()

    return state, traj


def two_joint_setup(sample_period=0.05, prediction_horizon=10, j_start=0, j2_start=0, upos_limit=1.5, lpos_limit=-1.5,
                    vel_limit=1, acc_limit=4, jerk_limit=30, hf=None, joint_weight=None):
    j, j_v, j_a, j_j, j2, j2_v, j2_a, j2_j, sample_period_symbol = ca.var(
        'j j_v j_a j_j j2 j2_v j2_a j2_j sample_period')

    state = {
        'j': j_start,
        'j_v': 0,
        'j_a': 0,
        'j_j': 0,
        'j2': j2_start,
        'j2_v': 0,
        'j2_a': 0,
        'j2_j': 0,
        'sample_period': sample_period
    }
    if joint_weight is None:
        joint_weight = {
            1: 0.01,
            2: 0,
            3: 0,
        }

    if hf is None:
        def f(w, t):
            return w + w * 1 * t

        hf = {
            1: f
        }

    jc = FreeVariable(
        symbols={
            0: j,
            1: j_v,
            2: j_a,
            3: j_j,
        },
        lower_limits={
            0: lpos_limit,
            1: -vel_limit,
            2: -acc_limit,
            3: -jerk_limit
        },
        upper_limits={
            0: upos_limit,
            1: vel_limit,
            2: acc_limit,
            3: jerk_limit
        },
        quadratic_weights=joint_weight,
        horizon_functions=hf,
    )

    jc2 = FreeVariable(
        symbols={
            0: j2,
            1: j2_v,
            2: j2_a,
            3: j2_j,
        },
        lower_limits={
            0: lpos_limit,
            1: -vel_limit,
            2: - acc_limit,
            3: -jerk_limit
        },
        upper_limits={
            0: upos_limit,
            1: vel_limit,
            2: acc_limit,
            3: jerk_limit
        },
        quadratic_weights=joint_weight,
        horizon_functions=hf,
    )

    qp = QPController(sample_period_symbol, prediction_horizon, 'gurobi', [jc, jc2])
    return qp, j, j2, state


def test_joint_goal():
    ph = 10
    sample_period = 0.05
    qp, j, j2, state = two_joint_setup(sample_period, ph)

    goal_s, goal2_s = ca.var('goal goal2')
    goal1 = -0.5
    goal2 = 1.2
    state['goal'] = goal1
    state['goal2'] = goal2

    error = goal_s - j
    error2 = goal2_s - j2

    constraints = [
        Constraint('j1 goal',
                   expression=j,
                   lower_error=error,
                   upper_error=error,
                   velocity_limit=1,
                   quadratic_weight=1,
                   control_horizon=ph - 2),
        Constraint('j2 goal',
                   expression=j2,
                   lower_error=error2,
                   upper_error=error2,
                   velocity_limit=1,
                   quadratic_weight=1,
                   control_horizon=ph - 2),
    ]
    qp.add_constraints(constraints)
    qp.compile()

    final_state, _ = simulate(state, qp, sample_period, True, time_limit=2.5)
    np.testing.assert_almost_equal(final_state['j'], goal1, decimal=4)
    np.testing.assert_almost_equal(final_state['j2'], goal2, decimal=4)


def test_joint_goal2():
    j1_vel = 1
    j2_vel = 1
    # asdf = [0, 0.0001, 0.001, 0.01, 0.1, 1]
    # asdf = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    # asdf_v = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    resolution = 10
    v_min = 0.0001
    v_max = 0.02
    # asdf_v = np.arange(v_min, v_max, (v_max - v_min) / resolution)
    asdf_v = [1e-3]

    resolution = 5
    h_min = 0.1
    h_max = 1.5
    # asdf_h = np.arange(h_min, h_max, (h_max - h_min) / resolution)
    asdf_h = [1e-1]

    resolution = 10
    j_min = 0.0001
    j_max = 0.02
    # asdf_j = np.arange(j_min, j_max, (j_max - j_min) / resolution)
    asdf_j = [1e-3]

    results = []
    final = len(asdf_h) * len(asdf_v) * len(asdf_j)
    counter = 0
    for h_counter, h_i in enumerate(asdf_h):
        for v_counter, v_i in enumerate(asdf_v):
            for j_counter, j_i in enumerate(asdf_j):
                counter += 1
                print('{}/{}'.format(counter, final))
                ph = 10
                ch = ph - 2
                if h_i == 0 and v_i == 0 and j_i == 0:
                    continue
                # hf = lambda w, t: w
                # start_v = 0.005
                # a_v = (v_i - start_v) / (ch)
                # hf = lambda w, t: start_v + a_v * t

                # start = j_i * h_i
                # a = (j_i - start) / (ch)
                # hfj = lambda w, t: start + a * t
                # hfj = lambda w, t: w
                sample_period = 0.05
                qp, j, j2, state = two_joint_setup(sample_period, ph,
                                                   hf={
                                                       1: h_i,
                                                       3: 1,
                                                   },
                                                   joint_weight={
                                                       1: v_i,
                                                       2: 0.0,
                                                       3: j_i,
                                                   })

                goal_s, goal2_s = ca.var('goal goal2')
                goal1 = -0.5
                goal2 = 1.2
                state['goal'] = goal1
                state['goal2'] = goal2

                error = goal_s - j
                error2 = goal2_s - j2

                constraints = [
                    Constraint('j1 goal',
                               expression=j,
                               lower_error=error,
                               upper_error=error,
                               velocity_limit=1,
                               quadratic_weight=1,
                               control_horizon=ph - 2),
                    Constraint('j2 goal',
                               expression=j2,
                               lower_error=error2,
                               upper_error=error2,
                               velocity_limit=1,
                               quadratic_weight=1,
                               control_horizon=ph - 2),
                ]
                vel_constraints = [
                    VelocityConstraint('j1 goal vel',
                                       expression=j,
                                       lower_velocity_limit=-j1_vel,
                                       upper_velocity_limit=j1_vel,
                                       quadratic_weight=100,
                                       control_horizon=ph - 2),
                    VelocityConstraint('j2 goal vel',
                                       expression=j2,
                                       lower_velocity_limit=-j2_vel,
                                       upper_velocity_limit=j2_vel,
                                       quadratic_weight=100,
                                       lower_slack_limit=0,
                                       upper_slack_limit=0,
                                       control_horizon=ph - 2),
                ]
                qp.add_constraints(constraints)
                # qp.add_velocity_constraints(vel_constraints)
                qp.compile()

                final_state, traj = simulate(state, qp, sample_period,
                                             time_limit=8,
                                             name='h_{}-v_{}-j_{}'.format(h_i, v_i, j_i),
                                             # print_traj=False,
                                             print_traj=True,
                                             save=False)
                l = len(traj[0][0])
                results.append([h_i, v_i, j_i, l])
                pass
                # np.testing.assert_almost_equal(final_state['j'], goal1, decimal=4)
                # np.testing.assert_almost_equal(final_state['j2'], goal2, decimal=4)
    fig = plt.figure(figsize=[16, 8])
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    results = np.array(results)
    ax.scatter(results[:, 0], results[:, 1], results[:, 2], c=results[:, 3], s=20)
    minmax = np.array(list(sorted(results, key=lambda x: x[3])))
    print(minmax)
    ax.set_xlabel('h')
    ax.set_ylabel('v')
    ax.set_zlabel('j')
    plt.show()
    pass


def test_joint_goal_vel_limit():
    ph = 10
    sample_period = 0.05
    j1_vel = 0.5
    j2_vel = 0.8
    qp, j, j2, state = two_joint_setup(sample_period, ph)

    goal_s, goal2_s = ca.var('goal goal2')
    goal1 = -0.5
    goal2 = 1.5
    state['goal'] = goal1
    state['goal2'] = goal2

    error = goal_s - j
    error2 = goal2_s - j2
    weight = 10

    constraints = [
        Constraint('j1 goal',
                   expression=j,
                   lower_error=error,
                   upper_error=error,
                   velocity_limit=j1_vel,
                   quadratic_weight=weight,
                   control_horizon=ph - 2),
        Constraint('j2 goal',
                   expression=j2,
                   lower_error=error2,
                   upper_error=error2,
                   velocity_limit=j2_vel,
                   quadratic_weight=weight,
                   control_horizon=ph - 2),
    ]
    vel_constraints = [
        VelocityConstraint('j1 goal vel',
                           expression=j,
                           lower_velocity_limit=-j1_vel,
                           upper_velocity_limit=j1_vel,
                           quadratic_weight=100,
                           control_horizon=ph - 2),
        VelocityConstraint('j2 goal vel',
                           expression=j2,
                           lower_velocity_limit=-j2_vel,
                           upper_velocity_limit=j2_vel,
                           quadratic_weight=100,
                           lower_slack_limit=0,
                           upper_slack_limit=0,
                           control_horizon=ph - 2),
    ]
    qp.add_constraints(constraints)
    qp.add_velocity_constraints(vel_constraints)
    qp.compile()

    final_state, traj = simulate(state, qp, sample_period, True, time_limit=2.5)
    np.testing.assert_almost_equal(final_state['j'], goal1, decimal=4)
    np.testing.assert_almost_equal(final_state['j2'], goal2, decimal=4)
    assert max(np.abs(traj[1][0])) <= j1_vel + 0.04
    assert max(np.abs(traj[1][1])) <= j2_vel


def test_joint_continuous_goal():
    ph = 10
    sample_period = 0.05
    qp, j, j2, state = two_joint_setup(sample_period, ph,
                                       lpos_limit=None,
                                       upos_limit=None
                                       )

    goal_s, goal2_s = ca.var('goal goal2')
    goal1 = -0.5
    goal2 = 1.5
    state['goal'] = goal1
    state['goal2'] = goal2

    error = goal_s - j
    error2 = goal2_s - j2

    def horizon_function(w, t):
        return w

    constraints = [
        Constraint('j1 goal',
                   expression=j,
                   lower_error=error,
                   upper_error=error,
                   velocity_limit=0.5,
                   quadratic_weight=1,
                   control_horizon=10),
        Constraint('j2 goal',
                   expression=j2,
                   lower_error=error2,
                   upper_error=error2,
                   velocity_limit=0.8,
                   quadratic_weight=1,
                   control_horizon=10),
    ]
    qp.add_constraints(constraints)
    qp.compile()

    final_state, _ = simulate(state, qp, sample_period, True, time_limit=2.5)
    np.testing.assert_almost_equal(final_state['j'], goal1, decimal=4)
    np.testing.assert_almost_equal(final_state['j2'], goal2, decimal=4)


def test_joint_goal_close_to_limits():
    ph = 10
    sample_period = 0.05
    qp, j, j2, state = two_joint_setup(sample_period, ph)

    goal_s, goal2_s = ca.var('goal goal2')
    goal1 = -1.6
    goal2 = 1.6
    state['goal'] = goal1
    state['goal2'] = goal2

    error = goal_s - j
    error2 = goal2_s - j2

    constraints = [
        Constraint('j1 goal',
                   expression=j,
                   lower_error=error,
                   upper_error=error,
                   velocity_limit=0.9,
                   quadratic_weight=1,
                   control_horizon=10),
        Constraint('j2 goal',
                   expression=j2,
                   lower_error=error2,
                   upper_error=error2,
                   velocity_limit=0.8,
                   quadratic_weight=1,
                   control_horizon=10),
    ]
    qp.add_constraints(constraints)
    qp.compile()

    final_state, _ = simulate(state, qp, sample_period, True, time_limit=2.5)
    np.testing.assert_almost_equal(final_state['j'], qp.get_free_variable('j2').lower_position_limit, decimal=4)
    np.testing.assert_almost_equal(final_state['j2'], qp.get_free_variable('j2').upper_position_limit, decimal=4)


def test_joint_goal_control_horizon_1():
    ph = 10
    sample_period = 0.05
    qp, j, j2, state = two_joint_setup(sample_period, ph)

    goal_s, goal2_s = ca.var('goal goal2')
    goal1 = -0.5
    goal2 = 2
    state['goal'] = goal1
    state['goal2'] = goal2

    error = goal_s - j
    error2 = goal2_s - j2

    constraints = [
        Constraint('j1 goal',
                   expression=j,
                   lower_error=error,
                   upper_error=error,
                   velocity_limit=0.3,
                   quadratic_weight=10,
                   control_horizon=1),
        Constraint('j2 goal',
                   expression=j2,
                   lower_error=error2,
                   upper_error=error2,
                   velocity_limit=0.8,
                   quadratic_weight=10,
                   control_horizon=int(ph / 2)),
    ]
    qp.add_constraints(constraints)
    qp.compile()

    final_state = simulate(state, qp, sample_period, True, time_limit=2.5)[0]
    # np.testing.assert_almost_equal(final_state['j'], goal1, decimal=4)
    # np.testing.assert_almost_equal(final_state['j2'], qp.get_free_variable('j2').upper_position_limit, decimal=3)


def test_fk():
    ph = 10
    ch = 10
    time_limit = 8
    sample_period = 0.05
    qp, j, j2, state = two_joint_setup(sample_period, ph, upos_limit=None, lpos_limit=None,
                                       j2_start=0.01, j_start=0.01, jerk_limit=50)

    fk = ca.Matrix([
        ca.sin(j) + ca.sin(j + j2),
        ca.cos(j) + ca.cos(j + j2)
    ])
    gx, gy = ca.var('gx gy')

    ex = gx - fk[0]
    ey = gy - fk[1]
    state['gx'] = 1.5
    state['gy'] = -1
    c_max_vel = 0.5
    vel = 0.5
    error_weight = 1
    constraints = [
        Constraint('x',
                   expression=fk[0],
                   lower_error=ex,
                   upper_error=ex,
                   velocity_limit=c_max_vel,
                   quadratic_weight=error_weight,
                   control_horizon=ch),
        Constraint('y',
                   expression=fk[1],
                   lower_error=ey,
                   upper_error=ey,
                   velocity_limit=c_max_vel,
                   quadratic_weight=1000,
                   control_horizon=ch),
    ]
    vel_constraints = [
        VelocityConstraint('trans vel',
                           expression=j,
                           lower_velocity_limit=-vel,
                           upper_velocity_limit=vel,
                           quadratic_weight=1,
                           control_horizon=ph - 2,
                           lower_slack_limit=0,
                           upper_slack_limit=0),
    ]
    qp.add_constraints(constraints)
    qp.add_velocity_constraints(vel_constraints)
    qp.compile()

    final_state, traj = simulate(state, qp, sample_period, True, time_limit=time_limit)

    def compute_fk(joint1, joint2):
        final_state = ca.ca.substitute(ca.ca.substitute(fk, j, joint1), j2, joint2)
        return float(final_state[0]), float(final_state[1])

    cart_x = np.array([compute_fk(j1, j2)[0] for [j1, j2] in zip(*traj[0])])
    cart_x_vel = np.diff(cart_x) / sample_period
    cart_y = np.array([compute_fk(j1, j2)[1] for [j1, j2] in zip(*traj[0])])
    cart_y_vel = np.diff(cart_y) / sample_period
    fig, ax = plt.subplots(4, 1, figsize=(5, 5), sharex=True)
    ax[0].plot(cart_x)
    ax[0].grid()
    ax[1].plot(cart_x_vel)
    ax[1].set_yticks([-c_max_vel, 0, c_max_vel])
    ax[1].grid()
    ax[2].plot(cart_y)
    ax[2].grid()
    ax[3].plot(cart_y_vel)
    ax[3].set_yticks([-c_max_vel, 0, c_max_vel])
    ax[3].grid()
    plt.tight_layout()
    plt.show()

    fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))

    plt.plot(cart_x, cart_y, 'C3', lw=3)
    # plt.scatter(cart_x, cart_y, s=120)
    plt.title('cart space')
    plt.grid()
    plt.xlim(-2.1, 2.1)
    plt.ylim(-2.1, 2.1)

    plt.tight_layout()
    plt.show()

    x, y = compute_fk(final_state['j'], final_state['j2'])
    np.testing.assert_almost_equal(x, final_state['gx'], decimal=2)
    np.testing.assert_almost_equal(y, final_state['gy'], decimal=2)
    assert max(abs(cart_x_vel)) <= c_max_vel + 0.04
    assert max(abs(cart_y_vel)) <= c_max_vel + 0.04


def test_fk2():
    ph = 10
    ch = ph - 2
    time_limit = 8
    sample_period = 0.05
    qp, j, j2, state = two_joint_setup(sample_period, ph, upos_limit=None, lpos_limit=None,
                                       j2_start=0.01, j_start=0.01, jerk_limit=50)

    fk = ca.Matrix([
        j,
        j2
    ])
    gx, gy = ca.var('gx gy')
    g = ca.Matrix([gx, gy])

    e = g - fk
    state['gx'] = 2
    state['gy'] = 1
    c_max_vel = 0.5
    vel = 0.5
    constraints = [
        Constraint('x',
                   expression=fk[0],
                   lower_error=e[0],
                   upper_error=e[0],
                   velocity_limit=vel,
                   quadratic_weight=1,
                   control_horizon=ch),
        Constraint('y',
                   expression=fk[1],
                   lower_error=e[1],
                   upper_error=e[1],
                   velocity_limit=vel,
                   quadratic_weight=1,
                   control_horizon=ch),
    ]
    vel_constraints = [
        VelocityConstraint('trans vel x',
                           expression=fk[0],
                           lower_velocity_limit=-vel,
                           upper_velocity_limit=vel,
                           quadratic_weight=100000,
                           control_horizon=ph - 2,
                           lower_slack_limit=None,
                           upper_slack_limit=None),
        VelocityConstraint('trans vel y',
                           expression=fk[1],
                           lower_velocity_limit=-vel,
                           upper_velocity_limit=vel,
                           quadratic_weight=100000,
                           control_horizon=ph - 2,
                           lower_slack_limit=None,
                           upper_slack_limit=None),
    ]
    qp.add_constraints(constraints)
    qp.add_velocity_constraints(vel_constraints)
    qp.compile()

    final_state, traj = simulate(state, qp, sample_period, True, time_limit=time_limit)

    def compute_fk(joint1, joint2):
        final_state = ca.ca.substitute(ca.ca.substitute(fk, j, joint1), j2, joint2)
        return float(final_state[0]), float(final_state[1])

    cart_x = np.array([compute_fk(j1, j2)[0] for [j1, j2] in zip(*traj[0])])
    cart_x_vel = np.diff(cart_x) / sample_period
    cart_y = np.array([compute_fk(j1, j2)[1] for [j1, j2] in zip(*traj[0])])
    cart_y_vel = np.diff(cart_y) / sample_period
    fig, ax = plt.subplots(4, 1, figsize=(5, 5), sharex=True)
    ax[0].plot(cart_x)
    ax[0].grid()
    ax[1].plot(cart_x_vel)
    ax[1].set_yticks([-c_max_vel, 0, c_max_vel])
    ax[1].grid()
    ax[2].plot(cart_y)
    ax[2].grid()
    ax[3].plot(cart_y_vel)
    ax[3].set_yticks([-c_max_vel, 0, c_max_vel])
    ax[3].grid()
    plt.tight_layout()
    plt.show()

    fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))

    plt.plot(cart_x, cart_y, 'C3', lw=3)
    # plt.scatter(cart_x, cart_y, s=120)
    plt.title('cart space')
    plt.grid()
    plt.xlim(-2.1, 2.1)
    plt.ylim(-2.1, 2.1)

    plt.tight_layout()
    plt.show()

    x, y = compute_fk(final_state['j'], final_state['j2'])
    np.testing.assert_almost_equal(x, final_state['gx'], decimal=2)
    np.testing.assert_almost_equal(y, final_state['gy'], decimal=2)
    assert max(abs(cart_x_vel)) <= c_max_vel + 0.04
    assert max(abs(cart_y_vel)) <= c_max_vel + 0.04

# TODO test continuous joint
# TODO test with non square J
