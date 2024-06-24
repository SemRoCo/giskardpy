from typing import Tuple, Union, Dict, Type, Optional, List

import numpy as np

from giskardpy.data_types.data_types import Derivatives
from giskardpy.qp.qp_solver import QPSolver
from giskardpy.qp.qp_solver_qpalm import QPSolverQPalm
from giskardpy.utils.decorators import memoize


def quaternion_multiply(quaternion1: np.ndarray, quaternion0: np.ndarray) -> np.ndarray:
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array((
        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0), dtype=np.float64)


def quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    return np.array((-quaternion[0], -quaternion[1],
                     -quaternion[2], quaternion[3]), dtype=np.float64)


def qv_mult(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Transforms a vector by a quaternion
    :param quaternion: Quaternion
    :param vector: vector
    :return: transformed vector
    """
    q = quaternion
    v = [vector[0], vector[1], vector[2], 0]
    return quaternion_multiply(quaternion_multiply(q, v), quaternion_conjugate(q))[:-1]


def quaternion_from_axis_angle(axis: Union[List[float], Tuple[float, float, float], np.ndarray], angle: float) -> np.ndarray:
    half_angle = angle / 2
    return np.array([axis[0] * np.sin(half_angle),
                     axis[1] * np.sin(half_angle),
                     axis[2] * np.sin(half_angle),
                     np.cos(half_angle)])


_EPS = np.finfo(float).eps * 4.0


def rpy_from_matrix(rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
    """
    :param rotation_matrix: 4x4 Matrix
    :type rotation_matrix: Matrix
    :return: roll, pitch, yaw
    :rtype: (Union[float, Symbol], Union[float, Symbol], Union[float, Symbol])
    """
    i = 0
    j = 1
    k = 2

    cy = np.sqrt(rotation_matrix[i, i] * rotation_matrix[i, i] + rotation_matrix[j, i] * rotation_matrix[j, i])
    if cy - _EPS > 0:
        roll = np.arctan2(rotation_matrix[k, j], rotation_matrix[k, k])
        pitch = np.arctan2(-rotation_matrix[k, i], cy)
        yaw = np.arctan2(rotation_matrix[j, i], rotation_matrix[i, i])
    else:
        roll = np.arctan2(-rotation_matrix[j, k], rotation_matrix[j, j])
        pitch = np.arctan2(-rotation_matrix[k, i], cy)
        yaw = 0
    return roll, pitch, yaw


def rpy_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float]:
    return rpy_from_matrix(rotation_matrix_from_quaternion(qx, qy, qz, qw))


def rotation_matrix_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Conversion of roll, pitch, yaw to 4x4 rotation matrix according to:
    https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/src/frames.cpp#L167
    :return: 4x4 Matrix
    """
    rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(roll), -np.sin(roll), 0],
                   [0, np.sin(roll), np.cos(roll), 0],
                   [0, 0, 0, 1]])
    ry = np.array([[np.cos(pitch), 0, np.sin(pitch), 0],
                   [0, 1, 0, 0],
                   [-np.sin(pitch), 0, np.cos(pitch), 0],
                   [0, 0, 0, 1]])
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0],
                   [np.sin(yaw), np.cos(yaw), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    return np.dot(np.dot(rz, ry), rx)


def rotation_matrix_from_quaternion(x: float, y: float, z: float, w: float) -> np.ndarray:
    """
    Unit quaternion to 4x4 rotation matrix according to:
    https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/src/frames.cpp#L167
    :return: 4x4 Matrix
    """
    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w
    return np.array([[w2 + x2 - y2 - z2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y, 0],
                     [2 * x * y + 2 * w * z, w2 - x2 + y2 - z2, 2 * y * z - 2 * w * x, 0],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, w2 - x2 - y2 + z2, 0],
                     [0, 0, 0, 1]])


def rotation_matrix_from_axis_angle(axis: Union[List[float], Tuple[float, float, float], np.ndarray], angle: float) \
        -> np.ndarray:
    return rotation_matrix_from_quaternion(*quaternion_from_axis_angle(axis, angle))


def quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return quaternion_from_rotation_matrix(rotation_matrix_from_rpy(roll, pitch, yaw))


def quaternion_from_rotation_matrix(matrix: Union[List[List[float]], np.ndarray]) -> np.ndarray:
    """
    :param matrix: 4x4 Matrix
    :return: array length 4
    """
    q = np.empty((4,), dtype=np.float64)
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    t = np.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / np.sqrt(t * M[3, 3])
    return q


def axis_angle_from_quaternion(x: float, y: float, z: float, w: float) -> Tuple[np.ndarray, float]:
    l = np.linalg.norm(np.array([x, y, z, w]))
    x, y, z, w = x / l, y / l, z / l, w / l
    w2 = np.sqrt(1 - w ** 2)
    if w2 == 0:
        m = 1
    else:
        m = w2
    if w2 == 0:
        angle = 0
    else:
        angle = (2 * np.arccos(min(max(-1, w), 1)))
    if w2 == 0:
        x = 0
        y = 0
        z = 1
    else:
        x = x / m
        y = y / m
        z = z / m
    return np.array([x, y, z]), angle


def gauss(n: float) -> float:
    return (n ** 2 + n) / 2


def max_velocity_from_horizon_and_jerk(prediction_horizon, jerk_limit, sample_period):
    n2 = int((prediction_horizon) / 2)
    (prediction_horizon ** 2 + prediction_horizon) / 2
    return (gauss(n2) + gauss(n2 - 1)) * jerk_limit * sample_period ** 2


@memoize
def mpc(upper_limits: Tuple[Tuple[float, ...], ...],
        lower_limits: Tuple[Tuple[float, ...], ...],
        current_values: Tuple[float, ...],
        dt: float,
        ph: int,
        q_weight: Tuple[float, ...],
        lin_weight: Tuple[float, ...],
        solver_class: Optional[Type[QPSolver]] = None,
        link_to_current_vel: bool = True) -> np.ndarray:
    if solver_class is None:
        solver = QPSolverQPalm.empty()
        # solver = QPSolverQPSwift.empty()
        # solver = QPSolverGurobi.empty()
    else:
        solver = solver_class.empty()
    max_d = len(upper_limits)
    lb = []
    ub = []
    for derivative in range(max_d):
        lb_ = lower_limits[derivative]
        ub_ = upper_limits[derivative]
        lb.extend(lb_)
        ub.extend(ub_)
        if (derivative + 1) != max_d:  # because we start at velocity
            lb[-1] = 0
            ub[-1] = 0
    model = derivative_link_model(dt, ph, max_d)
    bE = np.zeros(model.shape[0])
    for i, current_value in enumerate(current_values):
        derivative = i + 1
        bE[ph * (derivative - 1)] = current_value
    w = np.zeros(len(lb))
    w[:ph] = q_weight[0]
    w[ph:ph * 2] = q_weight[1]
    w[-ph:] = q_weight[2]
    H = np.diag(w)
    g = np.zeros(len(lb))
    g[:ph] = lin_weight[0]
    g[ph:ph * 2] = lin_weight[1]
    g[-ph:] = lin_weight[2]
    empty = np.eye(0)
    lb = np.array(lb)
    ub = np.array(ub)
    if not link_to_current_vel:
        model = np.delete(model, [0, ph], axis=0)
        bE = np.delete(bE, [0, ph])
    result = solver.default_interface_solver_call(H=H, g=g, lb=lb, ub=ub,
                                                  E=model, bE=bE,
                                                  A=empty, lbA=np.array([]), ubA=np.array([]))
    return result


def simple_mpc(vel_limit, acc_limit, jerk_limit, current_vel, current_acc, dt, ph, q_weight, lin_weight,
               solver_class=None, link_to_current_vel: bool = True):
    upper_limits = ((vel_limit,) * ph,
                    (acc_limit,) * ph,
                    (jerk_limit,) * ph)
    lower_limits = ((-vel_limit,) * ph,
                    (-acc_limit,) * ph,
                    (-jerk_limit,) * ph)
    return mpc(upper_limits=upper_limits, lower_limits=lower_limits,
               current_values=(current_vel, current_acc), dt=dt, ph=ph, q_weight=q_weight, lin_weight=lin_weight,
               solver_class=solver_class, link_to_current_vel=link_to_current_vel)


def mpc_velocities(upper_limits: Tuple[Tuple[float, ...], ...],
                   lower_limits: Tuple[Tuple[float, ...], ...],
                   current_values: Tuple[float, ...],
                   dt: float,
                   ph: int,
                   solver_class=None):
    return mpc(upper_limits, lower_limits, current_values, dt, ph, (1, 1, 1), (0, 0, 0), solver_class)


def derivative_link_model(dt, ph, max_derivative):
    num_rows = ph * (max_derivative - 1)
    num_columns = ph * max_derivative
    derivative_link_model = np.zeros((num_rows, num_columns))

    x_n = np.eye(num_rows)
    derivative_link_model[:, :x_n.shape[0]] += x_n

    xd_n = -np.eye(num_rows) * dt
    h_offset = ph
    derivative_link_model[:, h_offset:] += xd_n

    x_c_height = ph - 1
    x_c = -np.eye(x_c_height)
    offset_v = 0
    offset_h = 0
    for derivative in Derivatives.range(Derivatives.velocity, max_derivative - 1):
        offset_v += 1
        derivative_link_model[offset_v:offset_v + x_c_height, offset_h:offset_h + x_c_height] += x_c
        offset_v += x_c_height
        offset_h += ph
    return derivative_link_model


def mpc_velocity_integral(limits: Dict[Derivatives, float], dt: float, ph: int) -> float:
    upper_limits = {
        Derivatives.velocity: np.ones(ph) * limits[Derivatives.velocity],
        Derivatives.acceleration: np.ones(ph) * limits[Derivatives.acceleration],
        Derivatives.jerk: np.ones(ph) * limits[Derivatives.jerk]
    }
    lower_limits = {
        Derivatives.velocity: np.ones(ph) * -limits[Derivatives.velocity],
        Derivatives.acceleration: np.ones(ph) * -limits[Derivatives.acceleration],
        Derivatives.jerk: np.ones(ph) * -limits[Derivatives.jerk]
    }
    return np.sum(mpc_velocities(upper_limits, lower_limits,
                                 {Derivatives.velocity: limits[Derivatives.velocity] + limits[
                                     Derivatives.jerk] * dt ** 2,
                                  Derivatives.acceleration: 0}, dt, ph)) * dt


def mpc_velocity_integral2(limits: Dict[Derivatives, float], dt: float, ph: int) -> float:
    ph -= 2
    i1 = gauss(ph) * (limits[Derivatives.velocity] / (ph)) * dt
    ph -= 1
    i2 = gauss(ph) * (limits[Derivatives.velocity] / (ph)) * dt
    return (i1 + i2) / 2


def mpc_velocity_integral3(limits: Dict[Derivatives, float], dt: float, ph: int) -> float:
    ph -= 1
    v = limits[Derivatives.velocity]
    i1 = (v * dt * ph) / 2
    ph -= 1
    i2 = (v * dt * ph) / 2
    return (i1 + i2) / 2


def limit(a: float, lower_limit: float, upper_limit: float) -> float:
    return max(lower_limit, min(upper_limit, a))


def inverse_frame(f1_T_f2: np.ndarray) -> np.ndarray:
    """
    :param f1_T_f2: 4x4 Matrix
    :return: f2_T_f1
    """
    f2_T_f1 = np.eye(4)
    f2_T_f1[:3, :3] = f1_T_f2[:3, :3].T
    f2_T_f1[:3, 3] = np.dot(-f2_T_f1[:3, :3], f1_T_f2[:3, 3])
    return f2_T_f1


def angle_between_vector(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    :param v1: vector length 3
    :param v2: vector length 3
    """
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def my_cross(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    :param v1: vector length 3 or 4
    :param v2: vector length 3 or 4
    :return: cross product vector length 3
    """
    return np.cross(v1[:-1], v2[:-1])


def point_to_single_caster_angle(px, py, caster_v, forward_velocity):
    max_angular_velocity = 1
    z = np.array([0, 0, 1, 0])
    x = np.array([1, 0, 0, 0])
    center_P_p = np.array([px, py, 0, 1])
    c_V_p = center_P_p - caster_v
    c_V_goal = my_cross(c_V_p, z)
    angle = angle_between_vector(x[:-1], c_V_goal)
    radius = np.linalg.norm(c_V_p)
    if radius > 0.01:
        circumference = 2 * np.pi * radius
        number_of_revolutions = forward_velocity / circumference
        angular_velocity = number_of_revolutions / (2 * np.pi)
        angular_velocity = min(max(angular_velocity, -max_angular_velocity), max_angular_velocity)
    else:
        angular_velocity = max_angular_velocity
    print(f'angular velocity {angular_velocity}')
    return angle, c_V_goal


def point_to_caster_angles(px, py):
    forward_velocity = 2
    # center_P_fl = np.array([1, 1, 0, 1])
    # print(f'fl {point_to_single_caster_angle(px, py, center_P_fl, forward_velocity)}')
    # center_P_fr = np.array([1, -1, 0, 1])
    # print(f'fr {point_to_single_caster_angle(px, py, center_P_fr, forward_velocity)}')
    # center_P_bl = np.array([-1, 1, 0, 1])
    # print(f'bl {point_to_single_caster_angle(px, py, center_P_bl, forward_velocity)}')
    # center_P_br = np.array([-1, -1, 0, 1])
    # print(f'br {point_to_single_caster_angle(px, py, center_P_br, forward_velocity)}')
    center = np.array([0, 0, 0, 1])
    print(f'center {point_to_single_caster_angle(px, py, center, forward_velocity)}')
