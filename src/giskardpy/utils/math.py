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

