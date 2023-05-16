from scipy.optimize import curve_fit
import numpy as np
import giskardpy.casadi_wrapper as cas
import giskardpy.utils.math as gm


def one_step_change(current_acceleration, jerk_limit, dt):
    return current_acceleration * dt + jerk_limit * dt ** 2


def desired_velocity(current_position, goal_position, dt, ph):
    e = goal_position - current_position
    a = e / (cas.gauss(ph) * dt)
    # a = e / ((gauss(ph-1) + ph - 1)*dt)
    return a * ph
    # return a * (ph-2)


def shifted_velocity_profile(vel_profile, distance, dt):
    distance = cas.Expression(distance)
    vel_profile = vel_profile.copy()
    vel_profile[vel_profile < 0] = 0
    if_cases = [(dt * sum(vel_profile[x:]), np.concatenate([vel_profile[x + 1:], np.zeros(x + 1)])) for x in
                range(len(vel_profile) - 1, -1, -1)]
    return cas.if_less_eq_cases(distance, if_cases, np.ones(vel_profile.shape) * vel_profile[0])


def ub_profile(current_pos, current_vel, current_acc,
               pos_limit, vel_limit, acc_limit, jerk_limit, dt, ph):
    profile = gm.simple_mpc(vel_limit, acc_limit, jerk_limit, vel_limit, 0, dt, ph, 1)
    vel_profile = profile[:ph]
    acc_profile = cas.Expression(profile[ph:ph * 2])
    jerk_profile = cas.Expression(profile[-ph:])
    pos_error = pos_limit - current_pos
    vel_profile = shifted_velocity_profile(vel_profile, pos_error, dt)
    ph -= 1
    one_step_change_ = -jerk_limit * dt ** 2
    one_step_change_ = cas.limit(one_step_change_, -vel_limit, vel_limit)
    vel_profile[0] = cas.if_less(pos_error, 0, one_step_change_, vel_profile[0])

    target_velocity = cas.Expression(vel_profile[0])
    acc_profile = cas.ones(*vel_profile.shape) * acc_limit
    jerk_profile = cas.ones(*vel_profile.shape) * jerk_limit

    vel_error = target_velocity - current_vel
    position_limit_active = cas.less(target_velocity, vel_limit)
    min_next_vel = current_vel + current_acc * dt - jerk_limit * dt**2
    max_next_vel = current_vel + current_acc * dt + jerk_limit * dt**2
    above_req_vel = cas.less(target_velocity - min_next_vel, 0)
    acc_would_violate_vel_limit = cas.logic_or(cas.less(max_next_vel, -vel_limit),
                                               cas.greater(min_next_vel, target_velocity),
                                               cas.less((target_velocity - current_vel)/dt, current_acc),
                                               cas.greater((-vel_limit - current_vel)/dt, current_acc))
    cant_decc_within_horizon = cas.greater(cas.abs(current_acc), jerk_limit * dt * (ph-1))
    special_jerk_limits = cas.logic_or(cas.logic_and(position_limit_active, above_req_vel),
                                       acc_would_violate_vel_limit,
                                       cant_decc_within_horizon)
    # vel_error = cas.if_else(special_jerk_limits, vel_error, 0)
    target_acc = vel_error / dt - current_acc
    target_jerk = target_acc / dt

    target_velocity2 = cas.Expression(vel_profile[1])
    next_acc = current_acc + target_acc
    # next_vel = target_acc * dt
    vel_error2 = target_velocity2 - (target_velocity)

    # vel_error2 = cas.if_less(target_velocity, vel_limit, vel_error2, 0)
    target_acc2 = vel_error2 / dt - next_acc
    target_jerk2 = target_acc2 / dt
    jerk_profile[0] = cas.if_else(special_jerk_limits, target_jerk, jerk_limit)
    jerk_profile[1] = cas.if_else(special_jerk_limits, target_jerk2, jerk_limit)

    return cas.vstack([vel_profile, acc_profile, jerk_profile])


def b_profile(current_position, current_velocity, current_acceleration,
              position_limits, velocity_limits, acceleration_limits, jerk_limits, dt, ph):
    lb = ub_profile(-current_position, -current_velocity, -current_acceleration,
                             -position_limits[0], -velocity_limits[0], -acceleration_limits[0], -jerk_limits[0],
                             dt, ph)
    lb = -lb
    ub = ub_profile(current_position, current_velocity, current_acceleration,
                             position_limits[1], velocity_limits[1], acceleration_limits[1], jerk_limits[1], dt,
                             ph)
    jerk_lb = cas.abs(lb[ph*2])
    jerk_ub = cas.abs(ub[ph*2])
    jerk_b = cas.max(jerk_lb, jerk_ub)
    jerk_b = cas.max(jerk_b, jerk_limits[1])
    lb[ph * 2] = -jerk_b
    ub[ph * 2] = jerk_b

    jerk2_lb = cas.abs(lb[ph*2+1])
    jerk2_ub = cas.abs(ub[ph*2+1])
    jerk2_b = cas.max(jerk2_lb, jerk2_ub)
    jerk2_b = cas.max(jerk2_b, jerk_limits[1])
    lb[ph * 2 + 1] = -jerk2_b
    ub[ph * 2 + 1] = jerk2_b
    # jerk_lb = min(jerk_limits[0], jerk_lb)
    # jerk_lb = min(jerk_ub, jerk_lb)
    # jerk = max(jerk_lb, jerk_ub)
    return lb, ub
