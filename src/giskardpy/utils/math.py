from typing import Tuple, Union

import numpy as np
from geometry_msgs.msg import Quaternion, Point
from tf.transformations import quaternion_multiply, quaternion_conjugate, quaternion_matrix, quaternion_from_matrix


def qv_mult(quaternion, vector):
    """
    Transforms a vector by a quaternion
    :param quaternion: Quaternion
    :type quaternion: list
    :param vector: vector
    :type vector: list
    :return: transformed vector
    :type: list
    """
    q = quaternion
    v = [vector[0], vector[1], vector[2], 0]
    return quaternion_multiply(quaternion_multiply(q, v), quaternion_conjugate(q))[:-1]


def quaternion_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    half_angle = angle / 2
    return np.array([axis[0] * np.sin(half_angle),
                     axis[1] * np.sin(half_angle),
                     axis[2] * np.sin(half_angle),
                     np.cos(half_angle)])


_EPS = np.finfo(float).eps * 4.0


def rpy_from_matrix(rotation_matrix):
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


def rpy_from_quaternion(qx, qy, qz, qw):
    return rpy_from_matrix(quaternion_matrix([qx, qy, qz, qw]))


def rotation_matrix_from_rpy(roll, pitch, yaw):
    """
    Conversion of roll, pitch, yaw to 4x4 rotation matrix according to:
    https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/src/frames.cpp#L167
    :type roll: Union[float, Symbol]
    :type pitch: Union[float, Symbol]
    :type yaw: Union[float, Symbol]
    :return: 4x4 Matrix
    :rtype: Matrix
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
    return np.dot(rz, ry, rx)


def rotation_matrix_from_quaternion(x, y, z, w):
    """
    Unit quaternion to 4x4 rotation matrix according to:
    https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/src/frames.cpp#L167
    :type x: Union[float, Symbol]
    :type y: Union[float, Symbol]
    :type z: Union[float, Symbol]
    :type w: Union[float, Symbol]
    :return: 4x4 Matrix
    :rtype: Matrix
    """
    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w
    return np.array([[w2 + x2 - y2 - z2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y, 0],
                     [2 * x * y + 2 * w * z, w2 - x2 + y2 - z2, 2 * y * z - 2 * w * x, 0],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, w2 - x2 - y2 + z2, 0],
                     [0, 0, 0, 1]])


def quaternion_from_rpy(roll, pitch, yaw):
    return quaternion_from_matrix(rotation_matrix_from_rpy(roll, pitch, yaw))


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


def max_velocity_from_horizon_and_jerk(prediction_horizon, jerk_limit, sample_period):
    def gauss(n):
        return (n ** 2 + n) / 2

    n2 = int((prediction_horizon) / 2)
    (prediction_horizon ** 2 + prediction_horizon) / 2
    return (gauss(n2) + gauss(n2 - 1)) * jerk_limit * sample_period ** 2


def inverse_frame(f1_T_f2):
    """
    :param f1_T_f2: 4x4 Matrix
    :type f1_T_f2: Matrix
    :return: f2_T_f1
    :rtype: Matrix
    """
    f2_T_f1 = np.eye(4)
    f2_T_f1[:3, :3] = f1_T_f2[:3, :3].T
    f2_T_f1[:3, 3] = np.dot(-f2_T_f1[:3, :3], f1_T_f2[:3, 3])
    return f2_T_f1


def angle_between_vector(v1, v2):
    """
    :type v1: Vector3
    :type v2: Vector3
    :rtype: float
    """
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def normalize(v):
    return v / np.linalg.norm(v)


def compare_poses(actual_pose, desired_pose, decimal=2):
    """
    :type actual_pose: Pose
    :type desired_pose: Pose
    """
    compare_points(actual_point=actual_pose.position,
                   desired_point=desired_pose.position,
                   decimal=decimal)
    compare_orientations(actual_orientation=actual_pose.orientation,
                         desired_orientation=desired_pose.orientation,
                         decimal=decimal)


def compare_points(actual_point: Point, desired_point: Point, decimal: float = 2):
    np.testing.assert_almost_equal(actual_point.x, desired_point.x, decimal=decimal)
    np.testing.assert_almost_equal(actual_point.y, desired_point.y, decimal=decimal)
    np.testing.assert_almost_equal(actual_point.z, desired_point.z, decimal=decimal)


def compare_orientations(actual_orientation: Union[Quaternion, np.ndarray],
                         desired_orientation: Union[Quaternion, np.ndarray],
                         decimal: float = 2):
    if isinstance(actual_orientation, Quaternion):
        q1 = np.array([actual_orientation.x,
                       actual_orientation.y,
                       actual_orientation.z,
                       actual_orientation.w])
    else:
        q1 = actual_orientation
    if isinstance(desired_orientation, Quaternion):
        q2 = np.array([desired_orientation.x,
                       desired_orientation.y,
                       desired_orientation.z,
                       desired_orientation.w])
    else:
        q2 = desired_orientation
    try:
        np.testing.assert_almost_equal(q1[0], q2[0], decimal=decimal)
        np.testing.assert_almost_equal(q1[1], q2[1], decimal=decimal)
        np.testing.assert_almost_equal(q1[2], q2[2], decimal=decimal)
        np.testing.assert_almost_equal(q1[3], q2[3], decimal=decimal)
    except:
        np.testing.assert_almost_equal(q1[0], -q2[0], decimal=decimal)
        np.testing.assert_almost_equal(q1[1], -q2[1], decimal=decimal)
        np.testing.assert_almost_equal(q1[2], -q2[2], decimal=decimal)
        np.testing.assert_almost_equal(q1[3], -q2[3], decimal=decimal)


def my_cross(v1, v2):
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
