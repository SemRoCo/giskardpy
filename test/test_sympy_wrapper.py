import unittest

from hypothesis import given, reproduce_failure, assume
import hypothesis.strategies as st
from collections import OrderedDict

import numpy as np

import PyKDL
import itertools
from tf.transformations import quaternion_matrix, quaternion_about_axis, quaternion_from_euler, euler_matrix, \
    rotation_matrix, quaternion_multiply, quaternion_conjugate, random_quaternion, quaternion_from_matrix, \
    quaternion_slerp

from numpy import pi

import giskardpy.symengine_wrappers as spw
from giskardpy import BACKEND
from giskardpy.utils import georg_slerp

PKG = 'giskardpy'

BIG_NUMBER = 1e100
SMALL_NUMBER = 1e-100
float_nonan_noinf = st.floats(allow_nan=False, allow_infinity=False)


class TestSympyWrapper(unittest.TestCase):

    # fails if numbers too small or big
    @given(float_nonan_noinf.filter(lambda x: abs(x) < BIG_NUMBER))
    def test_abs(self, f1):
        self.assertAlmostEqual(spw.diffable_abs(f1), abs(f1), places=7)

    # fails if numbers too small or big
    @given(float_nonan_noinf.filter(lambda x: abs(x) < 1e10),
           float_nonan_noinf.filter(lambda x: abs(x) < 1e10))
    def test_max(self, f1, f2):
        r1 = np.float(spw.diffable_Max(f1, f2))
        self.assertTrue(np.isclose(r1, max(f1, f2)), msg='max({},{})={}'.format(f1, f2, r1))

    # fails if numbers too big
    @given(float_nonan_noinf.filter(lambda x: abs(x) < 1e10),
           float_nonan_noinf.filter(lambda x: abs(x) < 1e10))
    def test_min(self, f1, f2):
        r1 = np.float(spw.diffable_Min(f1, f2))
        self.assertTrue(np.isclose(r1, min(f1, f2)), msg='min({},{})={}'.format(f1, f2, r1))

    # fails if numbers too big
    @given(float_nonan_noinf.filter(lambda x: abs(x) < BIG_NUMBER and abs(x) > SMALL_NUMBER))
    def test_sign(self, f1):
        r1 = np.float(spw.diffable_sign(f1))
        r2 = np.sign(f1)
        self.assertTrue(np.isclose(r1, r2), msg='spw.sign({})={} != np.sign({})={}'.format(f1, r1, f1, r2))

    # fails if condition is to close too 0 or too big or too small
    @given(float_nonan_noinf.filter(lambda x: abs(x) < BIG_NUMBER and abs(x) > SMALL_NUMBER),
           float_nonan_noinf.filter(lambda x: abs(x) < BIG_NUMBER),
           float_nonan_noinf.filter(lambda x: abs(x) < BIG_NUMBER))
    def test_if_greater_zero(self, condition, if_result, else_result):
        r1 = np.float(spw.if_greater_zero(condition, if_result, else_result))
        r2 = np.float(if_result if condition > 0 else else_result)
        self.assertTrue(np.isclose(r1, r2), msg='{} if {} > 0 else {} => {}'.format(if_result, condition, else_result,
                                                                                    r1))

    # fails if condition is to close too 0 or too big or too small
    # fails if if_result is too big or too small
    @given(float_nonan_noinf.filter(lambda x: abs(x) < BIG_NUMBER and abs(x) > SMALL_NUMBER),
           float_nonan_noinf.filter(lambda x: abs(x) < 1e10),
           float_nonan_noinf.filter(lambda x: abs(x) < BIG_NUMBER))
    def test_if_eq_zero(self, condition, if_result, else_result):
        r1 = np.float(spw.if_eq_zero(condition, if_result, else_result))
        r2 = np.float(if_result if condition == 0 else else_result)
        self.assertTrue(np.isclose(r1, r2, atol=1.e-7), msg='{} if {} == 0 else {} => {}'.format(if_result, condition,
                                                                                                 else_result,
                                                                                                 r1))

    # TODO test save compiled function
    # TODO test load compiled function
    # TODO test compiled function class
    # TODO test speedup

    # fails if numbers too big
    @given(st.lists(float_nonan_noinf.filter(lambda x: abs(x) < BIG_NUMBER), min_size=3, max_size=3),
           st.lists(float_nonan_noinf.filter(lambda x: abs(x) < BIG_NUMBER), min_size=3, max_size=3))
    def test_cross(self, u, v):

        r1 = np.array(spw.cross(u,v)).astype(float).T[0]
        r2 = np.cross(u,v)
        self.assertTrue(np.isclose(r1, r2).all(), msg='{}x{}=\n{} != {}'.format(u,v,r1,r2))

    @given(st.lists(float_nonan_noinf, min_size=3, max_size=3))
    def test_vector3(self, v):
        r1 = spw.vector3(*v)
        self.assertEqual(r1[0], v[0])
        self.assertEqual(r1[1], v[1])
        self.assertEqual(r1[2], v[2])
        self.assertEqual(r1[3], 0)

    @given(st.lists(float_nonan_noinf, min_size=3, max_size=3))
    def test_point3(self, v):
        r1 = spw.point3(*v)
        self.assertEqual(r1[0], v[0])
        self.assertEqual(r1[1], v[1])
        self.assertEqual(r1[2], v[2])
        self.assertEqual(r1[3], 1)

    # fails if numbers too big
    @given(st.lists(float_nonan_noinf.filter(lambda x: abs(x) < BIG_NUMBER)))
    def test_norm(self, v):
        r1 = np.float(spw.norm(v))
        r2 = np.linalg.norm(v)
        self.assertTrue(np.isclose(r1, r2), msg='|{}|2=\n{} != {}'.format(v,r1,r2))

    # fails if numbers too big
    @given(st.lists(float_nonan_noinf.filter(lambda x: abs(x) < BIG_NUMBER)),
           float_nonan_noinf.filter(lambda x: abs(x) < BIG_NUMBER))
    def test_scale(self, v, a):
        assume(np.linalg.norm(v) != 0)
        r1 = np.array(spw.scale(spw.Matrix(v), a)).astype(float).T[0]
        r2 = v / np.linalg.norm(v) * a
        self.assertTrue(np.isclose(r1, r2).all(), msg='v={} a={}\n{} != {}'.format(v, a, r1, r2))


if __name__ == '__main__':
    import rosunit

    rosunit.unitrun(package=PKG,
                    test_name='TestSympyWrapper',
                    test=TestSympyWrapper)
