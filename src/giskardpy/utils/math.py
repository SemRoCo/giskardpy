import numpy as np
from geometry_msgs.msg import Quaternion, Point
from tf.transformations import quaternion_multiply, quaternion_conjugate


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


def max_velocity_from_horizon_and_jerk(prediction_horizon, jerk_limit, sample_period):
    def gauss(n):
        return (n ** 2 + n) / 2

    n2 = int((prediction_horizon) / 2)
    (prediction_horizon**2+prediction_horizon)/2
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


def compare_orientations(actual_orientation: Quaternion, desired_orientation: Quaternion, decimal: float = 2):
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
