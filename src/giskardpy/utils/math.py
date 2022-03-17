import numpy as np
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