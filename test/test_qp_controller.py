# from mpl_toolkits import mplot3d
import hypothesis.strategies
import hypothesis.strategies as st
import matplotlib.pyplot as plt
import numpy as np
from hypothesis import given, assume

import giskardpy.casadi_wrapper as w
from giskardpy.goals.open_close import WEIGHT_COLLISION_AVOIDANCE, WEIGHT_ABOVE_CA, WEIGHT_BELOW_CA
from giskardpy.data_types import FreeVariable, Constraint, VelocityConstraint
from giskardpy.god_map import GodMap
from giskardpy.qp.qp_controller import QPController
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

    def simulate(self, things_to_plot=None, time_limit=6., min_time=1., file_name='', save=False,
                 hlines=None):
        self.qp.compile()
        trajectories = {}
        if things_to_plot is not None:
            for s in things_to_plot:
                trajectories[s] = []
            trajectories['time'] = []

        for t in range(int(1 / self.sample_period * time_limit)):
            self.god_map.set_data(['time'], t)
            subs = self.god_map.get_values(self.qp.compiled_big_ass_M.str_params)
            [cmd_vel, cmd_acc, cmd_jerk], debug_expr = self.qp.get_cmd(subs)
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
                try:
                    trajectories[s].append(self.god_map.get_data([s]))
                except KeyError:
                    trajectories[s].append(debug_expr[s])
            if all_zero and t > min_time:
                break

        if things_to_plot:
            f, axs = plt.subplots(len(things_to_plot), sharex=True)
            f.set_size_inches(w=7, h=9)
            time = trajectories['time']
            for i, (s) in enumerate(things_to_plot):
                traj = trajectories[s]
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
            if hlines:
                for h in hlines:
                    axs[0].axhline(h)
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
        # things_to_plot=['j', 'j_v'],
        # hlines=[hard_threshold, soft_threshold],
        time_limit=5,
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


def test_collision_avoidance2():
    j_goal = 0
    j_start = 1
    tjs = TwoJointSetup(j_start=j_start, j_vel_limit=0.7)
    weight = WEIGHT_ABOVE_CA
    ca_vel_limit = 0.2

    tjs.god_map.set_data(['goal'], j_goal)
    goal_s = tjs.god_map.to_symbol(['goal'])
    t = tjs.god_map.to_symbol(['time']) * 0.05
    j = tjs.god_map.to_symbol(['j'])
    j_v = tjs.god_map.to_symbol(['j_v'])

    hard_threshold1 = 0.
    hard_threshold2 = 0.5
    hard_threshold = w.if_greater(t, 1, hard_threshold2, hard_threshold1)
    # hard_threshold = hard_threshold2
    soft_threshold = hard_threshold + 0.05
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
    tjs.qp.add_debug_expressions(
        {
            'hard_threshold': hard_threshold,
        }
    )

    tjs.simulate(
        # things_to_plot=['j', 'j_v', 'j_j', 'hard_threshold'],
        # hlines=[j_start, hard_threshold2],
        time_limit=5,
    )
    assert tjs.god_map.get_data(['j']) >= hard_threshold2 / 2
    if abs(j_goal - hard_threshold2) > 0.05 and j_goal > hard_threshold2:
        np.testing.assert_almost_equal(tjs.god_map.get_data(['j']),
                                       tjs.god_map.get_data(['goal']), decimal=2)


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
        things_to_plot=['j', 'j_v', 'j_a', 'j_j'],
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


def test_joint_goal_opposing():
    tjs = TwoJointSetup(j_acc_limit=999,
                        j_jerk_limit=999)

    tjs.god_map.set_data(['goal'], 1)
    goal_s = tjs.god_map.to_symbol(['goal'])
    tjs.god_map.set_data(['goal2'], -1)
    goal2_s = tjs.god_map.to_symbol(['goal2'])
    j = tjs.god_map.to_symbol(['j'])

    error = goal_s - j
    error2 = goal2_s - j

    constraints = [
        Constraint('j1 goal',
                   expression=j,
                   lower_error=error,
                   upper_error=error,
                   velocity_limit=tjs.j_vel_limit,
                   quadratic_weight=WEIGHT_BELOW_CA,
                   control_horizon=tjs.control_horizon),
        Constraint('j2 goal',
                   expression=j,
                   lower_error=error2,
                   upper_error=error2,
                   velocity_limit=tjs.j_vel_limit,
                   quadratic_weight=WEIGHT_BELOW_CA,
                   control_horizon=1),
    ]
    tjs.qp.add_constraints(constraints)

    tjs.simulate(
        # things_to_plot=['j', 'j_v', 'j_a', 'j_j'],
        time_limit=10
    )
    current = tjs.god_map.get_data(['j'])
    goal = tjs.god_map.get_data(['goal'])
    goal2 = tjs.god_map.get_data(['goal2'])
    np.testing.assert_almost_equal(current, (goal + goal2) / 2, decimal=3)


def test_fk():
    tjs = TwoJointSetup(j_upos_limit=None,
                        j_lpos_limit=None,
                        j2_upos_limit=None,
                        j2_lpos_limit=None)

    tjs.god_map.set_data(['gx'], 1)
    gx = tjs.god_map.to_symbol(['gx'])

    tjs.god_map.set_data(['gy'], -1)
    gy = tjs.god_map.to_symbol(['gy'])

    j = tjs.god_map.to_symbol(['j'])
    j2 = tjs.god_map.to_symbol(['j2'])
    j_v = tjs.god_map.to_symbol(['j_v'])
    j2_v = tjs.god_map.to_symbol(['j2_v'])

    fk = w.Matrix([
        w.sin(j) + w.sin(j + j2),
        w.cos(j) + w.cos(j + j2)
    ])

    ex = gx - fk[0]
    ey = gy - fk[1]
    c_max_vel = 0.5
    vel = 0.5
    trans_vel = w.norm(fk)
    constraints = [
        Constraint('x',
                   expression=fk[0],
                   lower_error=ex,
                   upper_error=ex,
                   velocity_limit=c_max_vel,
                   quadratic_weight=WEIGHT_BELOW_CA,
                   control_horizon=tjs.control_horizon),
        Constraint('y',
                   expression=fk[1],
                   lower_error=ey,
                   upper_error=ey,
                   velocity_limit=c_max_vel,
                   quadratic_weight=WEIGHT_BELOW_CA,
                   control_horizon=tjs.control_horizon),
    ]
    vel_constraints = [
        VelocityConstraint('trans vel',
                           expression=trans_vel,
                           lower_velocity_limit=-vel,
                           upper_velocity_limit=vel,
                           quadratic_weight=1,
                           control_horizon=tjs.control_horizon,
                           lower_slack_limit=0,
                           upper_slack_limit=0),
    ]
    tjs.qp.add_constraints(constraints)
    tjs.qp.add_velocity_constraints(vel_constraints)
    tjs.qp.add_debug_expressions(
        {
            'fk/x': fk[0],
            'fk/y': fk[1],
            'fk/x/dot': w.total_derivative(fk[0], [j, j2], [j_v, j2_v]),
            'fk/y/dot': w.total_derivative(fk[1], [j, j2], [j_v, j2_v]),
            'cart_vel': w.total_derivative(trans_vel, [j, j2], [j_v, j2_v]),
        }
    )

    traj = tjs.simulate(
        things_to_plot=['j', 'j_v', 'j2', 'j2_v', 'fk/x', 'fk/x/dot', 'fk/y', 'fk/y/dot', 'cart_vel'],
        time_limit=10
    )

    cart_x = traj['fk/x']
    cart_y = traj['fk/y']

    plt.plot(cart_x, cart_y, 'C3', lw=3)
    # plt.scatter(cart_x, cart_y, s=120)
    plt.title('cart space')
    plt.grid()
    plt.xlim(-2.1, 2.1)
    plt.ylim(-2.1, 2.1)

    plt.tight_layout()
    plt.show()

    x = traj['fk/x'][-1]
    y = traj['fk/y'][-1]
    gx = tjs.god_map.get_data(['gx'])
    gy = tjs.god_map.get_data(['gy'])
    np.testing.assert_almost_equal(x, gx, decimal=2)
    np.testing.assert_almost_equal(y, gy, decimal=2)
    assert np.max(np.abs(traj['cart_vel'])) <= c_max_vel * 1.04

# TODO test with non square J
