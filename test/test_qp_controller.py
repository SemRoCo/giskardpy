import hypothesis.strategies as st
# from mpl_toolkits import mplot3d
import hypothesis.strategies
from hypothesis import given, assume
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

import giskardpy.casadi_wrapper as w
from giskardpy.constraints import WEIGHT_COLLISION_AVOIDANCE, WEIGHT_ABOVE_CA, WEIGHT_BELOW_CA
from giskardpy.data_types import FreeVariable, Constraint, VelocityConstraint
from giskardpy.god_map import GodMap
from giskardpy.qp_controller import QPController
from utils_for_tests import float_no_nan_no_inf


class TwoJointSetup(object):
    def __init__(self,
                 sample_period=0.05,
                 prediction_horizon=10,
                 j_start=0.,
                 j2_start=0.,
                 j_upos_limit=1.5,
                 j_lpos_limit=-1.5,
                 j2_upos_limit=1.5,
                 j2_lpos_limit=-1.5,
                 j_vel_limit=0.9,
                 j_acc_limit=4.,
                 j_jerk_limit=30.,
                 j2_vel_limit=0.9,
                 j2_acc_limit=4.,
                 j2_jerk_limit=30.,
                 hf=None,
                 j_joint_weight=None,
                 j2_joint_weight=None):
        self.sample_period = sample_period
        self.prediction_horizon = prediction_horizon
        self.j_start = j_start
        self.j2_start = j2_start
        self.j_upos_limit = j_upos_limit
        self.j_lpos_limit = j_lpos_limit
        self.j2_upos_limit = j2_upos_limit
        self.j2_lpos_limit = j2_lpos_limit
        self.j_vel_limit = j_vel_limit
        self.j_acc_limit = j_acc_limit
        self.j_jerk_limit = j_jerk_limit
        self.j2_vel_limit = j2_vel_limit
        self.j2_acc_limit = j2_acc_limit
        self.j2_jerk_limit = j2_jerk_limit
        self.hf = hf
        self.j_joint_weight = j_joint_weight
        self.j2_joint_weight = j2_joint_weight
        self.control_horizon = self.prediction_horizon - 2
        self.control_horizon = max(1, self.control_horizon)

        self.god_map = GodMap()
        self.god_map.set_data(['j'], self.j_start)
        self.god_map.set_data(['j_v'], 0)
        self.god_map.set_data(['j_a'], 0)
        self.god_map.set_data(['j_j'], 0)
        self.god_map.set_data(['j2'], self.j2_start)
        self.god_map.set_data(['j2_v'], 0)
        self.god_map.set_data(['j2_a'], 0)
        self.god_map.set_data(['j2_j'], 0)
        self.god_map.set_data(['sample_period'], self.sample_period)
        self.god_map.set_data(['time'], 0)

        if self.j_joint_weight is None:
            self.j_joint_weight = {
                1: 0.01,
                2: 0,
                3: 0,
            }
        if self.j2_joint_weight is None:
            self.j2_joint_weight = {
                1: 0.01,
                2: 0,
                3: 0,
            }

        self.hf = {
            1: 0.1
        }

        self.jc = FreeVariable(
            symbols={
                0: self.god_map.to_symbol(['j']),
                1: self.god_map.to_symbol(['j_v']),
                2: self.god_map.to_symbol(['j_a']),
                3: self.god_map.to_symbol(['j_j']),
            },
            lower_limits={
                0: self.j_lpos_limit,
                1: -self.j_vel_limit,
                2: -self.j_acc_limit,
                3: -self.j_jerk_limit
            },
            upper_limits={
                0: self.j_upos_limit,
                1: self.j_vel_limit,
                2: self.j_acc_limit,
                3: self.j_jerk_limit
            },
            quadratic_weights=self.j_joint_weight,
            horizon_functions=self.hf,
        )

        self.jc2 = FreeVariable(
            symbols={
                0: self.god_map.to_symbol(['j2']),
                1: self.god_map.to_symbol(['j2_v']),
                2: self.god_map.to_symbol(['j2_a']),
                3: self.god_map.to_symbol(['j2_j']),
            },
            lower_limits={
                0: self.j2_lpos_limit,
                1: -self.j2_vel_limit,
                2: - self.j2_acc_limit,
                3: -self.j2_jerk_limit
            },
            upper_limits={
                0: self.j2_upos_limit,
                1: self.j2_vel_limit,
                2: self.j2_acc_limit,
                3: self.j2_jerk_limit
            },
            quadratic_weights=self.j2_joint_weight,
            horizon_functions=self.hf,
        )

        self.qp = QPController(self.god_map.to_symbol(['sample_period']), self.prediction_horizon, 'gurobi',
                               [self.jc, self.jc2])

    def simulate(self, symbols_to_plot=None, time_limit=6., min_time=1., file_name='', save=False):
        self.qp.compile()
        trajectories = {}
        if symbols_to_plot is not None:
            for s in symbols_to_plot:
                trajectories[s] = []
            trajectories['time'] = []

        for t in range(int(1 / self.sample_period * time_limit)):
            self.god_map.set_data(['time'], t)
            subs = self.god_map.get_values(self.qp.compiled_big_ass_M.str_params)
            [cmd_vel, cmd_acc, cmd_jerk], _ = self.qp.get_cmd(subs)
            for i, (free_variable, cmd) in enumerate(cmd_vel.items()):
                current = self.god_map.get_data([free_variable])
                self.god_map.set_data([free_variable], current + cmd * self.god_map.get_data(['sample_period']))
            all_zero = True
            for i, (free_variable, cmd) in enumerate(cmd_vel.items()):
                free_variable += '_v'
                self.god_map.set_data([free_variable], cmd)
                all_zero &= abs(cmd) < 1e-3
            for i, (free_variable, cmd) in enumerate(cmd_acc.items()):
                free_variable += '_a'
                self.god_map.set_data([free_variable], cmd)
            for i, (free_variable, cmd) in enumerate(cmd_jerk.items()):
                free_variable += '_j'
                self.god_map.set_data([free_variable], cmd)
            for s in trajectories:
                trajectories[s].append(self.god_map.get_data([s]))
            if all_zero and t > min_time:
                break

        if symbols_to_plot:
            f, axs = plt.subplots(len(symbols_to_plot), sharex=True)
            f.set_size_inches(w=7, h=9)
            time = trajectories['time']
            for i, (s, traj) in enumerate(trajectories.items()):
                if s == 'time':
                    continue
                axs[i].set_ylabel(s)
                axs[i].plot(np.array(time) * self.sample_period, traj)
                ticks = [0]
                min_ = np.round(min(traj), 4)
                max_ = np.round(max(traj), 4)
                range_ = max_ - min_
                if min_ < -range_ / 8:
                    ticks.append(min_)
                    if min_ < -range_ / 3:
                        ticks.append(min_ / 2)
                if max_ > range_ / 8:
                    ticks.append(max_)
                    if max_ > range_ / 3:
                        ticks.append(max_ / 2)
                axs[i].set_yticks(ticks)
                axs[i].grid()
            plt.title(file_name)
            plt.tight_layout()
            if save:
                plt.savefig('tmp_data/results/l_{}_{}.png'.format(len(trajectories[0][0]), file_name))
            else:
                f.show()

        return trajectories


@given(float_no_nan_no_inf(outer_limit=1.5),
       float_no_nan_no_inf(outer_limit=1.5),
       hypothesis.strategies.booleans())
def test_collision_avoidance(j_start, j_goal, above):
    tjs = TwoJointSetup(j_start=j_start, j_vel_limit=0.7)
    if above:
        weight = WEIGHT_ABOVE_CA
    else:
        weight = WEIGHT_BELOW_CA
    ca_vel_limit = 0.2

    tjs.god_map.set_data(['goal'], j_goal)
    goal_s = tjs.god_map.to_symbol(['goal'])
    j = tjs.god_map.to_symbol(['j'])

    hard_threshold = 0.1
    soft_threshold = hard_threshold + 0.1
    qp_limits_for_lba = ca_vel_limit * tjs.sample_period * tjs.control_horizon

    actual_distance = j
    penetration_distance = soft_threshold - actual_distance
    lower_limit = penetration_distance

    lower_limit_limited = w.limit(lower_limit,
                                  -qp_limits_for_lba,
                                  qp_limits_for_lba)

    upper_slack = w.if_greater(actual_distance, hard_threshold,
                               w.limit(soft_threshold - hard_threshold,
                                       -qp_limits_for_lba,
                                       qp_limits_for_lba),
                               lower_limit_limited)

    # undo factor in A
    upper_slack /= (tjs.sample_period * tjs.prediction_horizon)

    upper_slack = w.if_greater(actual_distance, 50,  # assuming that distance of unchecked closest points is 100
                               1e4,
                               # 1e4,
                               w.max(0, upper_slack))

    error = goal_s - j

    constraints = [
        Constraint('j1 goal',
                   expression=j,
                   lower_error=error,
                   upper_error=error,
                   velocity_limit=tjs.j_vel_limit,
                   quadratic_weight=weight,
                   control_horizon=tjs.control_horizon),
        Constraint('ca',
                   expression=actual_distance,
                   lower_error=lower_limit,
                   upper_error=1e4,
                   velocity_limit=ca_vel_limit,
                   quadratic_weight=WEIGHT_COLLISION_AVOIDANCE,
                   control_horizon=tjs.control_horizon,
                   upper_slack_limit=upper_slack),
    ]
    tjs.qp.add_constraints(constraints)

    tjs.simulate(
        # symbols_to_plot=['j', 'j_v'],
        time_limit=5
    )
    if above:
        if j_goal < soft_threshold:
            assert tjs.god_map.get_data(['j']) < soft_threshold
        if j_start < hard_threshold:
            assert tjs.god_map.get_data(['j']) >= j_start - 1e-4
        else:
            assert tjs.god_map.get_data(['j']) >= hard_threshold
            if abs(j_goal - hard_threshold) > 0.05 and j_goal > hard_threshold:
                np.testing.assert_almost_equal(tjs.god_map.get_data(['j']),
                                               tjs.god_map.get_data(['goal']), decimal=2)
    else:
        assert tjs.god_map.get_data(['j']) > soft_threshold * 0.95

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

                goal_s, goal2_s = w.var('goal goal2')
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
    tjs = TwoJointSetup()

    constraint_vel_limit = 0.5

    tjs.god_map.set_data(['goal'], 1)
    goal_s = tjs.god_map.to_symbol(['goal'])
    j = tjs.god_map.to_symbol(['j'])

    error = goal_s - j

    constraints = [
        Constraint('j1 goal',
                   expression=j,
                   lower_error=error,
                   upper_error=error,
                   velocity_limit=constraint_vel_limit,
                   quadratic_weight=WEIGHT_BELOW_CA,
                   control_horizon=tjs.control_horizon),
    ]
    vel_constraints = [
        VelocityConstraint('j1 goal vel',
                           expression=j,
                           lower_velocity_limit=-constraint_vel_limit,
                           upper_velocity_limit=constraint_vel_limit,
                           quadratic_weight=WEIGHT_BELOW_CA,
                           control_horizon=tjs.control_horizon,
                           upper_slack_limit=0,
                           lower_slack_limit=0
                           ),
    ]
    tjs.qp.add_constraints(constraints)
    tjs.qp.add_velocity_constraints(vel_constraints)

    traj = tjs.simulate(
        symbols_to_plot=['j', 'j_v', 'j_a', 'j_j'],
        time_limit=10
    )
    current = tjs.god_map.get_data(['j'])
    goal = tjs.god_map.get_data(['goal'])

    np.testing.assert_almost_equal(current, goal, decimal=3)

    assert max(traj['j_v']) <= constraint_vel_limit
    assert min(traj['j_v']) >= -constraint_vel_limit


@given(st.floats(max_value=5, min_value=-5),
       st.floats(max_value=5, min_value=-5),
       st.floats(max_value=5, min_value=-5))
def test_joint_goal(j_upos_limit, j_lpos_limit, goal):
    assume((j_upos_limit - j_lpos_limit) > 0.01)
    j_start = (j_upos_limit + j_lpos_limit) / 2
    tjs = TwoJointSetup(j_start=j_start,
                        j_upos_limit=j_upos_limit,
                        j_lpos_limit=j_lpos_limit)

    tjs.god_map.set_data(['goal'], goal)
    goal_s = tjs.god_map.to_symbol(['goal'])
    j = tjs.god_map.to_symbol(['j'])

    error = goal_s - j

    constraints = [
        Constraint('j1 goal',
                   expression=j,
                   lower_error=error,
                   upper_error=error,
                   velocity_limit=tjs.j_vel_limit,
                   quadratic_weight=WEIGHT_BELOW_CA,
                   control_horizon=tjs.control_horizon),
    ]
    tjs.qp.add_constraints(constraints)

    tjs.simulate(
        # symbols_to_plot=['j', 'j_v', 'j_a', 'j_j'],
        time_limit=10
    )
    current = tjs.god_map.get_data(['j'])
    assert current < j_upos_limit
    assert current > j_lpos_limit
    if goal < j_upos_limit and goal > j_lpos_limit:
        np.testing.assert_almost_equal(current, goal, decimal=3)

def test_joint_goal_continuous():
    tjs = TwoJointSetup(j_upos_limit=None,
                        j_lpos_limit=None)

    tjs.god_map.set_data(['goal'], 4)
    goal_s = tjs.god_map.to_symbol(['goal'])
    j = tjs.god_map.to_symbol(['j'])

    error = goal_s - j

    constraints = [
        Constraint('j1 goal',
                   expression=j,
                   lower_error=error,
                   upper_error=error,
                   velocity_limit=tjs.j_vel_limit,
                   quadratic_weight=WEIGHT_BELOW_CA,
                   control_horizon=tjs.control_horizon),
    ]
    tjs.qp.add_constraints(constraints)

    tjs.simulate(
        # symbols_to_plot=['j', 'j_v', 'j_a', 'j_j'],
        time_limit=10
    )
    current = tjs.god_map.get_data(['j'])
    goal = tjs.god_map.get_data(['goal'])
    np.testing.assert_almost_equal(current, goal, decimal=3)


@given(st.integers(max_value=20, min_value=1))
def test_joint_goal_control_horizon(prediction_horizon):
    tjs = TwoJointSetup(prediction_horizon=prediction_horizon,
                        j_acc_limit=999,
                        j_jerk_limit=999)

    tjs.god_map.set_data(['goal'], 1)
    goal_s = tjs.god_map.to_symbol(['goal'])
    j = tjs.god_map.to_symbol(['j'])

    error = goal_s - j

    constraints = [
        Constraint('j1 goal',
                   expression=j,
                   lower_error=error,
                   upper_error=error,
                   velocity_limit=tjs.j_vel_limit,
                   quadratic_weight=WEIGHT_BELOW_CA,
                   control_horizon=tjs.control_horizon),
    ]
    tjs.qp.add_constraints(constraints)

    tjs.simulate(
        # symbols_to_plot=['j', 'j_v', 'j_a', 'j_j'],
        time_limit=10
    )
    current = tjs.god_map.get_data(['j'])
    goal = tjs.god_map.get_data(['goal'])
    np.testing.assert_almost_equal(current, goal, decimal=3)


def test_fk():
    ph = 10
    ch = 10
    time_limit = 8
    sample_period = 0.05
    qp, j, j2, state = two_joint_setup(sample_period, ph, upos_limit=None, lpos_limit=None,
                                       j2_start=0.01, j_start=0.01, jerk_limit=50)

    fk = w.Matrix([
        w.sin(j) + w.sin(j + j2),
        w.cos(j) + w.cos(j + j2)
    ])
    gx, gy = w.var('gx gy')

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
        final_state = w.ca.substitute(w.ca.substitute(fk, j, joint1), j2, joint2)
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

    fk = w.Matrix([
        j,
        j2
    ])
    gx, gy = w.var('gx gy')
    g = w.Matrix([gx, gy])

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
        final_state = w.ca.substitute(w.ca.substitute(fk, j, joint1), j2, joint2)
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
