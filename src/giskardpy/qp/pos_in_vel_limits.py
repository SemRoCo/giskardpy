import numpy as np
import giskardpy.casadi_wrapper as cas
import giskardpy.utils.math as gm


def shifted_velocity_profile(vel_profile, acc_profile, distance, dt):
    distance = cas.Expression(distance)
    vel_profile = vel_profile.copy()
    vel_profile[vel_profile < 0] = 0
    vel_if_cases = []
    acc_if_cases = []
    for x in range(len(vel_profile) - 1, -1, -1):
        condition = dt * sum(vel_profile[x:])
        vel_result = np.concatenate([vel_profile[x + 1:], np.zeros(x + 1)])
        acc_result = np.concatenate([acc_profile[x + 1:], np.zeros(x + 1)])
        vel_if_cases.append((condition, vel_result))
        acc_if_cases.append((condition, acc_result))
    shifted_vel_profile = cas.if_less_eq_cases(distance, vel_if_cases, vel_profile)
    shifted_acc_profile = cas.if_less_eq_cases(distance, acc_if_cases, acc_profile)
    return shifted_vel_profile, shifted_acc_profile


def acc_cap(current_vel, jerk_limit, dt):
    acc_integral = cas.abs(current_vel) / dt
    jerk_step = jerk_limit * dt
    n = cas.floor(cas.r_gauss(cas.abs(acc_integral / jerk_step)))
    x = (- cas.gauss(n) * jerk_limit * dt + acc_integral) / (n + 1)
    return cas.abs(n * jerk_limit * dt + x)


def compute_next_vel_and_acc(current_vel, current_acc, vel_limit, jerk_limit, dt, remaining_ph, no_cap):
    acc_cap1 = acc_cap(current_vel, jerk_limit, dt)
    acc_cap2 = remaining_ph * jerk_limit * dt
    next_acc_min = current_acc - jerk_limit * dt
    next_acc_max = current_acc + jerk_limit * dt
    acc_to_vel = (vel_limit - current_vel) / dt
    acc_ph_max = cas.min(acc_cap1, acc_cap2)
    acc_ph_min = - acc_ph_max
    target_acc = cas.max(next_acc_min, acc_to_vel)
    target_acc = cas.if_else(no_cap, target_acc, cas.limit(target_acc, acc_ph_min, acc_ph_max))
    next_acc = cas.limit(target_acc, next_acc_min, next_acc_max)
    next_vel = current_vel + next_acc * dt
    return next_vel, next_acc


def compute_projected_vel_profile(current_vel, current_acc, target_vel_profile, jerk_limit, dt, ph, skip_first):
    vel_profile = []
    acc_profile = []
    next_vel, next_acc = current_vel, current_acc
    for i in range(ph):
        next_vel, next_acc = compute_next_vel_and_acc(next_vel, next_acc, target_vel_profile[i], jerk_limit, dt,
                                                      ph - i - 1,
                                                      cas.logic_and(skip_first, cas.equal(i, 0)))
        vel_profile.append(next_vel)
        acc_profile.append(next_acc)
    acc_profile = cas.Expression(acc_profile)
    acc_profile2 = cas.Expression(acc_profile)
    acc_profile2[1:] = acc_profile[:-1]
    acc_profile2[0] = current_acc
    jerk_profile = (acc_profile - acc_profile2) / dt

    return cas.Expression(vel_profile), acc_profile, jerk_profile


def b_profile(current_pos, current_vel, current_acc,
              pos_limits, vel_limits, acc_limits, jerk_limits, dt, ph, eps=0.00001):
    vel_limit = vel_limits[1]
    acc_limit = acc_limits[1]
    jerk_limit = jerk_limits[1]
    pos_range = pos_limits[1] - pos_limits[0]
    pos_limit_lb = pos_limits[0]
    pos_limit_ub = pos_limits[1]
    vel_limit = min(vel_limit * dt, pos_range / 2) / dt
    profile = gm.simple_mpc(vel_limit, acc_limit, jerk_limit, vel_limit, 0, dt, ph, (0, 0, 0), (-1, 0, 0))
    vel_profile_mpc = profile[:ph]
    acc_profile_mpc = profile[ph:ph * 2]
    pos_error_lb = pos_limit_lb - current_pos
    pos_error_ub = pos_limit_ub - current_pos
    shifted_vel_profile_lb, shifted_acc_profile_lb = shifted_velocity_profile(vel_profile_mpc,
                                                                              acc_profile_mpc, -pos_error_lb, dt)
    shifted_vel_profile_lb *= -1
    shifted_acc_profile_lb *= -1
    shifted_vel_profile_ub, shifted_acc_profile_ub = shifted_velocity_profile(vel_profile_mpc,
                                                                              acc_profile_mpc, pos_error_ub, dt)
    one_step_change_ = jerk_limit * dt ** 2
    one_step_change_lb = cas.min(cas.max(0, pos_error_lb), one_step_change_)
    one_step_change_lb = cas.limit(one_step_change_lb, -vel_limit, vel_limit)
    one_step_change_ub = cas.max(cas.min(0, pos_error_ub), -one_step_change_)
    one_step_change_ub = cas.limit(one_step_change_ub, -vel_limit, vel_limit)
    shifted_vel_profile_lb[0] = cas.if_greater(pos_error_lb, 0, one_step_change_lb, shifted_vel_profile_lb[0])
    shifted_vel_profile_ub[0] = cas.if_less(pos_error_ub, 0, one_step_change_ub, shifted_vel_profile_ub[0])

    acc_profile = cas.ones(*shifted_vel_profile_ub.shape) * acc_limit
    jerk_profile = cas.ones(*shifted_vel_profile_ub.shape) * jerk_limit

    goal_profile = cas.max(shifted_vel_profile_lb, 0) + cas.min(shifted_vel_profile_ub, 0)
    skip_first = cas.logic_or(shifted_vel_profile_lb[0] >= 0, shifted_vel_profile_ub[0] <= 0)
    proj_vel_profile, proj_acc_profile, proj_jerk_profile = compute_projected_vel_profile(current_vel,
                                                                                          current_acc,
                                                                                          goal_profile,
                                                                                          jerk_limit,
                                                                                          dt, ph,
                                                                                          skip_first)
    _, _, proj_jerk_profile_violated = compute_projected_vel_profile(current_vel,
                                                                     current_acc,
                                                                     goal_profile,
                                                                     np.inf,
                                                                     dt, ph,
                                                                     skip_first)
    vel_lb_violated = cas.logic_any(proj_vel_profile < shifted_vel_profile_lb - eps)
    vel_ub_violated = cas.logic_any(proj_vel_profile > shifted_vel_profile_ub + eps)

    special_jerk_limits = cas.logic_or(vel_lb_violated, vel_ub_violated)
    jerk_profile[0] = cas.if_else(special_jerk_limits, cas.max(jerk_limit, cas.abs(proj_jerk_profile_violated[0])),
                                  jerk_limit)
    jerk_profile[1] = cas.if_else(special_jerk_limits, cas.max(jerk_limit, cas.abs(proj_jerk_profile_violated[1])),
                                  jerk_limit)
    jerk_profile[2] = cas.if_else(special_jerk_limits, cas.max(jerk_limit, cas.abs(proj_jerk_profile_violated[2])),
                                  jerk_limit)

    lb_profile = cas.vstack([shifted_vel_profile_lb, -acc_profile, -jerk_profile])
    ub_profile = cas.vstack([shifted_vel_profile_ub, acc_profile, jerk_profile])
    lb_profile = cas.min(lb_profile, ub_profile)
    ub_profile = cas.max(lb_profile, ub_profile)
    return lb_profile, ub_profile
