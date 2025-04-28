import keyword

import hypothesis.strategies as st
import numpy as np
from hypothesis import assume
from hypothesis.strategies import composite
from numpy import pi

from giskardpy.middleware import get_middleware
from giskardpy.utils.math import shortest_angular_distance

BIG_NUMBER = 1e100
SMALL_NUMBER = 1e-100


def vector(x):
    return st.lists(float_no_nan_no_inf(), min_size=x, max_size=x)


def angle_positive():
    return st.floats(0, 2 * np.pi)


def random_angle():
    return st.floats(-np.pi, np.pi)


def compare_axis_angle(actual_angle, actual_axis, expected_angle, expected_axis, decimal=3):
    try:
        np.testing.assert_array_almost_equal(actual_axis, expected_axis, decimal=decimal)
        np.testing.assert_almost_equal(shortest_angular_distance(actual_angle, expected_angle), 0, decimal=decimal)
    except AssertionError:
        try:
            np.testing.assert_array_almost_equal(actual_axis, -expected_axis, decimal=decimal)
            np.testing.assert_almost_equal(shortest_angular_distance(actual_angle, abs(expected_angle - 2 * pi)), 0,
                                           decimal=decimal)
        except AssertionError:
            np.testing.assert_almost_equal(shortest_angular_distance(actual_angle, 0), 0, decimal=decimal)
            np.testing.assert_almost_equal(shortest_angular_distance(0, expected_angle), 0, decimal=decimal)
            assert not np.any(np.isnan(actual_axis))
            assert not np.any(np.isnan(expected_axis))

def compare_orientations(actual_orientation: np.ndarray,
                         desired_orientation: np.ndarray,
                         decimal: int = 2) -> None:
    q1 = actual_orientation
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

@composite
def variable_name(draw):
    variable = draw(st.text('qwertyuiopasdfghjklzxcvbnm', min_size=1))
    assume(variable not in keyword.kwlist)
    return variable


@composite
def lists_of_same_length(draw, data_types=(), min_length=1, max_length=10, unique=False):
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    lists = []
    for elements in data_types:
        lists.append(draw(st.lists(elements, min_size=length, max_size=length, unique=unique)))
    return lists


@composite
def rnd_joint_state(draw, joint_limits):
    return {jn: draw(st.floats(ll, ul, allow_nan=False, allow_infinity=False)) for jn, (ll, ul) in joint_limits.items()}


@composite
def rnd_joint_state2(draw, joint_limits):
    muh = draw(joint_limits)
    muh = {jn: ((ll if ll is not None else pi * 2), (ul if ul is not None else pi * 2))
           for (jn, (ll, ul)) in muh.items()}
    return {jn: draw(st.floats(ll, ul, allow_nan=False, allow_infinity=False)) for jn, (ll, ul) in muh.items()}


def pr2_without_base_urdf():
    with open('../../../../../PycharmProjects/giskardpy/test/urdfs/pr2.urdf', 'r') as f:
        urdf_string = f.read()
    return urdf_string


def base_bot_urdf():
    with open('../../../../../PycharmProjects/giskardpy/test/urdfs/2d_base_bot.urdf', 'r') as f:
        urdf_string = f.read()
    return urdf_string


def donbot_urdf():
    with open('../../../../../PycharmProjects/giskardpy/test/urdfs/iai_donbot.urdf', 'r') as f:
        urdf_string = f.read()
    return urdf_string


def boxy_urdf():
    with open('../../../../../PycharmProjects/giskardpy/test/urdfs/boxy.urdf', 'r') as f:
        urdf_string = f.read()
    return urdf_string


def hsr_urdf():
    with open('urdfs/hsr.urdf', 'r') as f:
        urdf_string = f.read()
    return urdf_string


def float_no_nan_no_inf(outer_limit=1e5):
    return float_no_nan_no_inf_min_max(-outer_limit, outer_limit)


def float_no_nan_no_inf_min_max(min_value=-1e5, max_value=1e5):
    return st.floats(allow_nan=False, allow_infinity=False, max_value=max_value, min_value=min_value,
                     allow_subnormal=False)


@composite
def sq_matrix(draw):
    i = draw(st.integers(min_value=1, max_value=10))
    i_sq = i ** 2
    l = draw(st.lists(float_no_nan_no_inf(outer_limit=1000), min_size=i_sq, max_size=i_sq))
    return np.array(l).reshape((i, i))


def unit_vector(length, elements=None):
    if elements is None:
        elements = float_no_nan_no_inf()
    vector = st.lists(elements,
                      min_size=length,
                      max_size=length).filter(lambda x: SMALL_NUMBER < np.linalg.norm(x) < BIG_NUMBER)

    def normalize(v):
        v = [round(x, 4) for x in v]
        l = np.linalg.norm(v)
        if l == 0:
            return np.array([0] * (length - 1) + [1])
        return np.array([x / l for x in v])

    return st.builds(normalize, vector)


def quaternion():
    return unit_vector(4, float_no_nan_no_inf(outer_limit=1))


def pykdl_frame_to_numpy(pykdl_frame):
    return np.array([[pykdl_frame.M[0, 0], pykdl_frame.M[0, 1], pykdl_frame.M[0, 2], pykdl_frame.p[0]],
                     [pykdl_frame.M[1, 0], pykdl_frame.M[1, 1], pykdl_frame.M[1, 2], pykdl_frame.p[1]],
                     [pykdl_frame.M[2, 0], pykdl_frame.M[2, 1], pykdl_frame.M[2, 2], pykdl_frame.p[2]],
                     [0, 0, 0, 1]])
