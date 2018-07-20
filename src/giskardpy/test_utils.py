from angles import normalize_angle
from hypothesis import given, reproduce_failure, assume
import hypothesis.strategies as st
from hypothesis.strategies import composite
import keyword
import numpy as np
from numpy import pi

from giskardpy.symengine_robot import Robot

BIG_NUMBER = 1e100
SMALL_NUMBER = 1e-100

vector = lambda x: st.lists(limited_float(), min_size=x, max_size=x)

def robot_urdfs():
    return st.sampled_from([u'pr2.urdf', u'boxy.urdf', u'iai_donbot.urdf'])
    # return st.sampled_from([u'pr2.urdf'])

def angle(*args, **kwargs):
    return st.builds(normalize_angle, limited_float(*args, **kwargs))

def keys_values(max_length=10, value_type=st.floats(allow_nan=False)):
    return lists_of_same_length([variable_name(), value_type], max_length=max_length, unique=True)

@composite
def variable_name(draw):
    variable = draw(st.text(u'qwertyuiopasdfghjklzxcvbnm', min_size=1))
    assume(variable not in keyword.kwlist)
    return variable



@composite
def lists_of_same_length(draw, data_types=(), max_length=10, unique=False):
    length = draw(st.integers(min_value=1, max_value=max_length))
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
    muh = {jn: ((ll if ll is not None else pi*2), (ul if ul is not None else pi*2))
                    for (jn, (ll, ul)) in muh.items()}
    return {jn: draw(st.floats(ll, ul, allow_nan=False, allow_infinity=False)) for jn, (ll, ul) in muh.items()}


@composite
def pr2_joint_state(draw):
    pr2 = Robot.from_urdf_file(u'../test/pr2.urdf')
    return draw(rnd_joint_state(*pr2.get_joint_limits()))


def limited_float(outer_limit=BIG_NUMBER, min_dist_to_zero=None):
    # f = st.floats(allow_nan=False, allow_infinity=False, max_value=outer_limit, min_value=-outer_limit)
    f = st.floats(allow_nan=False, allow_infinity=False)
    if min_dist_to_zero is not None:
        f = f.filter(lambda x: (outer_limit > abs(x) and abs(x) > min_dist_to_zero) or x == 0)
    else:
        f = f.filter(lambda x: abs(x) < outer_limit)
    return f


def unit_vector(length, elements=None):
    if elements is None:
        elements = limited_float(min_dist_to_zero=1e-20)
    vector = st.lists(elements,
                      min_size=length,
                      max_size=length).filter(lambda x: np.linalg.norm(x) > SMALL_NUMBER and
                                                        np.linalg.norm(x) < BIG_NUMBER)

    def normalize(v):
        l = np.linalg.norm(v)
        return [round(x / l, 10) for x in v]

    return st.builds(normalize, vector)


def quaternion(elements=None):
    return unit_vector(4, elements)


def pykdl_frame_to_numpy(pykdl_frame):
    return np.array([[pykdl_frame.M[0, 0], pykdl_frame.M[0, 1], pykdl_frame.M[0, 2], pykdl_frame.p[0]],
                     [pykdl_frame.M[1, 0], pykdl_frame.M[1, 1], pykdl_frame.M[1, 2], pykdl_frame.p[1]],
                     [pykdl_frame.M[2, 0], pykdl_frame.M[2, 1], pykdl_frame.M[2, 2], pykdl_frame.p[2]],
                     [0, 0, 0, 1]])


