import unittest

import PyKDL
import hypothesis.strategies as st
import numpy as np
from angles import shortest_angular_distance, normalize_angle_positive, normalize_angle
from hypothesis import given, assume
from tf.transformations import quaternion_matrix, quaternion_about_axis, quaternion_from_euler, euler_matrix, \
    rotation_matrix, quaternion_multiply, quaternion_conjugate, quaternion_from_matrix, \
    quaternion_slerp, rotation_from_matrix
from transforms3d.euler import euler2axangle
from transforms3d.quaternions import quat2mat, quat2axangle

from giskardpy import symbolic_wrapper as w
from utils_for_tests import limited_float, SMALL_NUMBER, unit_vector, quaternion, vector, \
    pykdl_frame_to_numpy, lists_of_same_length, angle, compare_axis_angle, angle_positive


class TestSympyWrapper(unittest.TestCase):

    #TODO test free symbols

    def test_is_matrix(self):
        self.assertFalse(w.is_matrix(w.Symbol('a')))
        self.assertTrue(w.is_matrix(w.Matrix([[0,0]])))

    def test_jacobian(self):
        a = w.Symbol('a')
        b = w.Symbol('b')
        m = w.Matrix([a+b, a**2, b**2])
        jac = w.jacobian(m, [a,b])
        expected = w.Matrix([[1,1],[2*a,0],[0,2*b]])
        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                assert w.equivalent(jac[i,j], expected[i,j])

    # fails if numbers too small or big
    @given(limited_float(outer_limit=1e10))
    def test_abs(self, f1):
        self.assertAlmostEqual(w.compile_and_execute(w.diffable_abs, [f1]), abs(f1), places=7)

    # fails if numbers too small or big
    # TODO decide if h(0)==0.5 or not
    # @given(limited_float(min_dist_to_zero=SMALL_NUMBER))
    # def test_heaviside(self, f1):
    #     # r1 = float(w.diffable_heaviside(f1))
    #     # r2 = 0 if f1 < 0 else 1
    #     np.heaviside()
    #     self.assertAlmostEqual(w.compile_and_execute(w.diffable_heaviside, [f1]),
    #                            0 if f1 < 0 else 1, places=7)
        # self.assertTrue(np.isclose(r1, r2), msg='0 if {} < 0 else 1 => {} != {}'.format(f1, r1, r2))

    # fails if numbers too small or big
    @given(limited_float(outer_limit=1e7),
           limited_float(outer_limit=1e7))
    def test_max(self, f1, f2):
        self.assertAlmostEqual(w.compile_and_execute(w.diffable_max, [f1, f2]),
                               max(f1, f2), places=7)
        self.assertAlmostEqual(w.compile_and_execute(w.diffable_max_fast, [f1, f2]),
                               max(f1, f2), places=7)
        self.assertAlmostEqual(w.compile_and_execute(w.Max, [f1, f2]),
                               max(f1, f2), places=7)

    # fails if numbers too small or big
    @given(limited_float(outer_limit=1e7),
           limited_float(outer_limit=1e7))
    def test_save_division(self, f1, f2):
        self.assertTrue(np.isclose(w.compile_and_execute(w.save_division, [f1, f2]),
                                   f1 / f2 if f2 != 0 else 0))
        # r1 = compile_and_execute(w.save_division, (f1, f2))
        # r2 = f1 / f2 if f2 != 0 else 0
        # self.assertTrue(np.isclose(r1, r2), msg='{}/{}={}'.format(f1, f2, r1))

    # fails if numbers too big
    @given(limited_float(outer_limit=1e7),
           limited_float(outer_limit=1e7))
    def test_min(self, f1, f2):
        self.assertAlmostEqual(w.compile_and_execute(w.diffable_min, [f1, f2]),
                               min(f1, f2), places=7)
        self.assertAlmostEqual(w.compile_and_execute(w.diffable_min_fast, [f1, f2]),
                               min(f1, f2), places=7)
        self.assertAlmostEqual(w.compile_and_execute(w.Min, [f1, f2]),
                               min(f1, f2), places=7)

    @given(st.integers(min_value=1, max_value=10))
    def test_matrix(self, x_dim):
        data = list(range(x_dim))
        m = w.Matrix(data)
        self.assertEqual(m[0], 0)
        self.assertEqual(m[-1], x_dim -1)

    @given(st.integers(min_value=1, max_value=10),
           st.integers(min_value=1, max_value=10))
    def test_matrix2(self, x_dim, y_dim):
        data = [[(i)+(j*x_dim) for j in range(y_dim)] for i in range(x_dim)]
        m = w.Matrix(data)
        self.assertEqual(float(m[0,0]), 0)
        self.assertEqual(float(m[x_dim-1,y_dim-1]), (x_dim*y_dim)-1)

    @given(st.integers(min_value=1, max_value=10),
           st.integers(min_value=1, max_value=10))
    def test_matrix3(self, x_dim, y_dim):
        data = [[(i)+(j*x_dim) for j in range(y_dim)] for i in range(x_dim)]
        m = w.Matrix(data)
        m2 = w.Matrix(m)
        # self.assertEqual(float(m[0,0]), 0)
        # self.assertEqual(float(m[x_dim-1,y_dim-1]), (x_dim*y_dim)-1)


    # fails if numbers too big
    @given(limited_float(min_dist_to_zero=SMALL_NUMBER))
    def test_sign(self, f1):
        self.assertAlmostEqual(w.compile_and_execute(w.sign, [f1]),
                               np.sign(f1), places=7)
        self.assertAlmostEqual(w.compile_and_execute(w.diffable_sign, [f1]),
                               np.sign(f1), places=7)

    # fails if condition is to close too 0 or too big or too small
    @given(limited_float(min_dist_to_zero=SMALL_NUMBER),
           limited_float(),
           limited_float())
    def test_if_greater_zero(self, condition, if_result, else_result):
        self.assertAlmostEqual(w.compile_and_execute(w.diffable_if_greater_zero, [condition, if_result, else_result]),
                               np.float(if_result if condition > 0 else else_result), places=5)
        self.assertAlmostEqual(w.compile_and_execute(w.if_greater_zero, [condition, if_result, else_result]),
                               np.float(if_result if condition > 0 else else_result), places=7)

    # fails if condition is to close too 0 or too big or too small
    @given(limited_float(min_dist_to_zero=SMALL_NUMBER),
           limited_float(),
           limited_float())
    def test_if_greater_eq_zero(self, condition, if_result, else_result):
        result = w.compile_and_execute(w.diffable_if_greater_eq_zero, [condition, if_result, else_result])
        expected = np.float(if_result if condition >= 0 else else_result)
        self.assertAlmostEqual(result, expected,
            msg='{} >= 0: {} instead of {}'.format(condition, result, expected))
        self.assertAlmostEqual(w.compile_and_execute(w.if_greater_eq_zero, [condition, if_result, else_result]),
                               np.float(if_result if condition >= 0 else else_result), places=7)

    # fails if condition is to close too 0 or too big or too small
    @given(limited_float(min_dist_to_zero=SMALL_NUMBER),
           limited_float(min_dist_to_zero=SMALL_NUMBER),
           limited_float(),
           limited_float())
    def test_if_greater_eq(self, a, b, if_result, else_result):
        self.assertAlmostEqual(
            w.compile_and_execute(w.diffable_if_greater_eq, [a, b, if_result, else_result]),
            np.float(if_result if a >= b else else_result), places=7)
        self.assertAlmostEqual(
            w.compile_and_execute(w.if_greater_eq, [a, b, if_result, else_result]),
            np.float(if_result if a >= b else else_result), places=7)

    # fails if condition is to close too 0 or too big or too small
    # fails if if_result is too big or too small
    @given(limited_float(min_dist_to_zero=SMALL_NUMBER),
           limited_float(outer_limit=1e8),
           limited_float())
    def test_if_eq_zero(self, condition, if_result, else_result):
        self.assertAlmostEqual(
            w.compile_and_execute(w.diffable_if_eq_zero, [condition, if_result, else_result]),
            np.float(if_result if condition == 0 else else_result), places=7)
        self.assertAlmostEqual(
            w.compile_and_execute(w.if_eq_zero, [condition, if_result, else_result]),
            np.float(if_result if condition == 0 else else_result), places=7)

    # fails if condition is to close too 0 or too big or too small
    # fails if if_result is too big or too small
    @given(limited_float(min_dist_to_zero=SMALL_NUMBER),
           limited_float(min_dist_to_zero=SMALL_NUMBER),
           limited_float(outer_limit=1e8),
           limited_float())
    def test_if_eq(self, a, b, if_result, else_result):
        r1 = w.compile_and_execute(w.diffable_if_eq, [a, b, if_result, else_result])
        r2 = np.float(if_result if a == b else else_result)
        self.assertTrue(np.isclose(r1, r2), msg='{} != {}'.format(r1, r2))
        self.assertTrue(np.isclose(
            w.compile_and_execute(w.if_eq, [a, b, if_result, else_result]),
            np.float(if_result if a == b else else_result)))

    #
    # @given(limited_float(),
    #        limited_float(),
    #        limited_float())
    # def test_if_greater_zero(self, condition, if_result, else_result):
    #     self.assertAlmostEqual(
    #         w.compile_and_execute(w.diffable_if_greater_zero, [condition, if_result, else_result]),
    #         np.float(if_result if condition > 0 else else_result), places=7)

    # @given(limited_float(),
    #        limited_float(),
    #        limited_float())
    # def test_if_greater_eq_zero(self, condition, if_result, else_result):
    #     self.assertAlmostEqual(
    #         w.compile_and_execute(w.diffable_if_greater_eq_zero, [condition, if_result, else_result]),
    #         np.float(if_result if condition >= 0 else else_result), places=7)
    #
    # @given(limited_float(),
    #        limited_float(),
    #        limited_float(),
    #        limited_float())
    # def test_if_greater_eq(self, a, b, if_result, else_result):
    #     r2 = np.float(if_result if a >= b else else_result)
    #     self.assertAlmostEqual(compile_and_execute(w.if_greater_eq, [a, b, if_result, else_result]),
    #                            r2, places=7)
    #
    # @given(limited_float(),
    #        limited_float(),
    #        limited_float())
    # def test_if_eq_zero(self, condition, if_result, else_result):
    #     r1 = np.float(w.if_eq_zero(condition, if_result, else_result))
    #     r2 = np.float(if_result if condition == 0 else else_result)
    #     self.assertTrue(np.isclose(r1, r2, atol=1.e-7), msg='{} if {} == 0 else {} => {}'.format(if_result, condition,
    #                                                                                              else_result,
    #                                                                                              r1))
    #     self.assertAlmostEqual(compile_and_execute(w.if_eq_zero, [condition, if_result, else_result]),
    #                            r1, places=7)

    # TODO test save compiled function
    # TODO test load compiled function
    # TODO test compiled function class

    @given(limited_float(min_dist_to_zero=1e-5),
           limited_float(min_dist_to_zero=1e-5),
           limited_float(),
           limited_float())
    def test_if_greater(self, a, b, if_result, else_result):
        self.assertAlmostEqual(
            w.compile_and_execute(w.diffable_if_greater, [a, b, if_result, else_result]),
            np.float(if_result if a > b else else_result), places=7)

    # fails if numbers too big or too small
    @given(unit_vector(length=3),
           angle())
    def test_speed_up_matrix_from_axis_angle(self, axis, angle):
        np.testing.assert_array_almost_equal(
            w.compile_and_execute(w.rotation_matrix_from_axis_angle, [axis, angle]),
            rotation_matrix(angle, axis))

    # @given(quaternion(),
    #        quaternion(),
    #        st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1))
    # def test_speed_up_diffable_slerp(self, q1, q2, t):
    #     q1 = np.array(q1)
    #     q2 = np.array(q2)
    #     r = compile_and_execute(w.diffable_slerp, [q1, q2, t])
    #     r_ref = quaternion_slerp(q1, q2, t)
    #     try:
    #         np.testing.assert_almost_equal(r, r_ref, decimal=3)
    #     except:
    #         np.testing.assert_almost_equal(r, -r_ref, decimal=3)

    # @given(quaternion(),
    #        quaternion(),
    #        st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1))
    # @reproduce_failure('4.0.2', 'AXicY2CAAkYGBixM7IBotSwAAU0ACQ==')
    # FIXME
    # def test_speed_up_slerp(self, q1, q2, t):
    #     q1 = np.array(q1)
    #     q2 = np.array(q2)
    #     r = speed_up_and_execute(spw.slerp, [q1, q2, t])
    #     r_ref = quaternion_slerp(q1, q2, t)
    #     try:
    #         np.testing.assert_almost_equal(r, r_ref, decimal=4)
    #     except:
    #         np.testing.assert_almost_equal(r, -r_ref, decimal=4)

    # fails if numbers too big
    @given(vector(3),
           vector(3))
    def test_cross(self, u, v):
        np.testing.assert_array_almost_equal(
            w.compile_and_execute(w.cross, [u, v]),
            np.cross(u, v))

    @given(vector(3))
    def test_vector3(self, v):
        r1 = w.vector3(*v)
        self.assertEqual(r1[0], v[0])
        self.assertEqual(r1[1], v[1])
        self.assertEqual(r1[2], v[2])
        self.assertEqual(r1[3], 0)

    @given(vector(3))
    def test_point3(self, v):
        r1 = w.point3(*v)
        self.assertEqual(r1[0], v[0])
        self.assertEqual(r1[1], v[1])
        self.assertEqual(r1[2], v[2])
        self.assertEqual(r1[3], 1)

    # fails if numbers too big
    @given(st.lists(limited_float()))
    def test_norm(self, v):
        self.assertTrue(np.isclose(
            w.compile_and_execute(w.norm, [v]),
            np.linalg.norm(v)))

    # fails if numbers too big
    @given(vector(3),
           limited_float(outer_limit=SMALL_NUMBER))
    def test_scale(self, v, a):
        if np.linalg.norm(v) == 0:
            r2 = [0, 0, 0]
        else:
            r2 = v / np.linalg.norm(v) * a
        np.testing.assert_array_almost_equal(
            w.compile_and_execute(w.scale, [v, a]),
            r2)

    # fails if numbers too big
    @given(lists_of_same_length([limited_float(), limited_float()], max_length=50))
    def test_dot(self, vectors):
        u, v = vectors
        u = np.array(u, ndmin=2)
        v = np.array(v, ndmin=2)
        self.assertTrue(np.isclose(w.compile_and_execute(w.dot, [u, v.T]),
                                   np.dot(u, v.T)))

    # fails if numbers too big
    @given(limited_float(),
           limited_float(),
           limited_float())
    def test_translation3(self, x, y, z):
        r1 = w.compile_and_execute(w.translation3, [x, y, z])
        r2 = np.identity(4)
        r2[0, 3] = x
        r2[1, 3] = y
        r2[2, 3] = z
        self.assertTrue(np.isclose(r1, r2).all(), msg='{} != {}'.format(r1, r2))

    # fails if numbers too big
    @given(angle(),
           angle(),
           angle())
    def test_rotation_matrix_from_rpy(self, roll, pitch, yaw):
        m1 = w.compile_and_execute(w.rotation_matrix_from_rpy, [roll, pitch, yaw])
        m2 = euler_matrix(roll, pitch, yaw)
        np.testing.assert_array_almost_equal(m1, m2)

    # fails if numbers too big or too small
    @given(unit_vector(length=3),
           angle())
    def test_rotation3_axis_angle(self, axis, angle):
        np.testing.assert_array_almost_equal(w.compile_and_execute(w.rotation_matrix_from_axis_angle, [axis, angle]),
                                             rotation_matrix(angle, np.array(axis)))

    # fails if numbers too big or too small
    @given(quaternion())
    def test_rotation3_quaternion(self, q):
        np.testing.assert_array_almost_equal(w.compile_and_execute(w.rotation_matrix_from_quaternion, q),
                                             quaternion_matrix(q))

    # fails if numbers too big or too small
    @given(limited_float(),
           limited_float(),
           limited_float(),
           unit_vector(length=3),
           angle())
    def test_frame3_axis_angle(self, x, y, z, axis, angle):
        r2 = rotation_matrix(angle, np.array(axis))
        r2[0, 3] = x
        r2[1, 3] = y
        r2[2, 3] = z
        np.testing.assert_array_almost_equal(w.compile_and_execute(w.frame_axis_angle, [x, y, z, axis, angle]),
                                             r2)

    # fails if numbers too big or too small
    @given(limited_float(),
           limited_float(),
           limited_float(),
           angle(),
           angle(),
           angle())
    def test_frame3_rpy(self, x, y, z, roll, pitch, yaw):
        r2 = euler_matrix(roll, pitch, yaw)
        r2[0, 3] = x
        r2[1, 3] = y
        r2[2, 3] = z
        np.testing.assert_array_almost_equal(w.compile_and_execute(w.frame_rpy, [x, y, z, roll, pitch, yaw]),
                                             r2)

    # fails if numbers too big or too small
    @given(limited_float(),
           limited_float(),
           limited_float(),
           unit_vector(4))
    def test_frame3_quaternion(self, x, y, z, q):
        r2 = quaternion_matrix(q)
        r2[0, 3] = x
        r2[1, 3] = y
        r2[2, 3] = z
        np.testing.assert_array_almost_equal(w.compile_and_execute(w.frame_quaternion, [x, y, z] + q.tolist()),
                                             r2)

    # fails if numbers too big or too small
    @given(limited_float(outer_limit=SMALL_NUMBER),
           limited_float(outer_limit=SMALL_NUMBER),
           limited_float(outer_limit=SMALL_NUMBER),
           quaternion())
    def test_inverse_frame(self, x, y, z, q):
        f = quaternion_matrix(q)
        f[0, 3] = x
        f[1, 3] = y
        f[2, 3] = z

        r2 = PyKDL.Frame()
        r2.M = PyKDL.Rotation.Quaternion(q[0], q[1], q[2], q[3])
        r2.p[0] = x
        r2.p[1] = y
        r2.p[2] = z
        r2 = r2.Inverse()
        r2 = pykdl_frame_to_numpy(r2)
        self.assertTrue(np.isclose(w.compile_and_execute(w.inverse_frame, [f]),
                                   r2, atol=1.e-4, rtol=1.e-4).all())

    # fails if numbers too big or too small
    @given(limited_float(),
           limited_float(),
           limited_float(),
           unit_vector(4))
    def test_pos_of(self, x, y, z, q):
        r1 = w.position_of(w.frame_quaternion(x, y, z, q[0], q[1], q[2], q[3]))
        r2 = [x, y, z, 1]
        for i, e in enumerate(r2):
            self.assertAlmostEqual(r1[i],e)

    # fails if numbers too big or too small
    @given(limited_float(),
           limited_float(),
           limited_float(),
           unit_vector(4))
    def test_trans_of(self, x, y, z, q):
        r1 = w.compile_and_execute(lambda *args: w.translation_of(w.frame_quaternion(*args)), [x, y, z, q[0], q[1], q[2], q[3]])
        r2 = np.identity(4)
        r2[0, 3] = x
        r2[1, 3] = y
        r2[2, 3] = z
        for i in range(r2.shape[0]):
            for j in range(r2.shape[1]):
                self.assertAlmostEqual(float(r1[i,j]), r2[i,j])

    # fails if numbers too big or too small
    @given(limited_float(),
           limited_float(),
           limited_float(),
           unit_vector(4))
    def test_rot_of(self, x, y, z, q):
        r1 = w.compile_and_execute(lambda *args: w.rotation_of(w.frame_quaternion(*args)),
                                   [x, y, z, q[0], q[1], q[2], q[3]])
        r2 = quaternion_matrix(q)
        self.assertTrue(np.isclose(r1, r2).all(), msg='\n{} != \n{}'.format(r1, r2))

    # fails if numbers too big or too small
    @given(unit_vector(4))
    def test_trace(self, q):
        m = quaternion_matrix(q)
        np.testing.assert_array_almost_equal(w.compile_and_execute(w.trace, [m]),
                                             np.trace(m))

    # TODO test rotation_dist

    # fails if numbers too big or too small
    # TODO use 'if' to make angle always positive?
    @given(unit_vector(length=3),
           angle_positive())
    def test_axis_angle_from_matrix(self, axis, angle):
        assume(angle < np.pi - 0.0001)
        m = rotation_matrix(angle, axis)
        axis2 = w.compile_and_execute(lambda x: w.diffable_axis_angle_from_matrix(x)[0], [m])
        angle2 = w.compile_and_execute(lambda x: w.diffable_axis_angle_from_matrix(x)[1], [m])
        if angle < 0:
            angle = -angle
            axis = [-x for x in axis]
        if angle2 < 0:
            angle2 = -angle2
            axis2 *= -1
        self.assertTrue(np.isclose(angle, angle2), msg='{} != {}'.format(angle, angle2))
        self.assertTrue(np.isclose(axis, axis2).all(), msg='{} != {}'.format(axis, axis2))

    # fails if numbers too big or too small
    @given(unit_vector(length=3),
           angle_positive())
    def test_axis_angle_from_matrix_stable(self, axis, angle):
        assume(angle < np.pi - 0.0001)
        m = rotation_matrix(angle, axis)
        axis2 = w.compile_and_execute(lambda x: w.axis_angle_from_matrix(x)[0], [m])
        angle2 = w.compile_and_execute(lambda x: w.axis_angle_from_matrix(x)[1], [m])
        angle, axis, _ = rotation_from_matrix(m)
        if angle < 0:
            angle = -angle
            axis = [-x for x in axis]
        if angle2 < 0:
            angle2 = -angle2
            axis2 *= -1
        self.assertTrue(np.isclose(angle, angle2), msg='{} != {}'.format(angle, angle2))
        self.assertTrue(np.isclose(axis, axis2).all(), msg='{} != {}'.format(axis, axis2))

    # TODO what is this test for?
    # @given(quaternion())
    # def test_axis_angle_from_matrix2(self, q):
    #     m = quat2mat(q)
    #     axis_reference, angle_reference = mat2axangle(m)
    #     assume(angle_reference < np.pi - 0.001)
    #     assume(angle_reference > -np.pi + 0.001)
    #     axis, angle = w.diffable_axis_angle_from_matrix_stable(m)
    #     self.assertGreaterEqual(angle, -1.e-10)
    #     my_m = w.to_numpy(angle_axis2mat(angle, axis))
    #     angle_diff = mat2axangle(m.T.dot(my_m))[1]
    #     self.assertAlmostEqual(angle_diff, 0.0, places=6)

    # fails if numbers too big or too small
    # TODO nan if angle 0
    # TODO use 'if' to make angle always positive?
    # FIXME my implementation is fine, but rotation_from_matrix is buggy...
    # @given(quaternion(),
    #        quaternion())
    # def test_rotation_distance(self, q1, q2):
    #     m1 = quaternion_matrix(q1)
    #     m2 = quaternion_matrix(q2)
    #     angle = w.compile_and_execute(w.rotation_distance, [m1, m2])
    #     ref_angle, _, _ = rotation_from_matrix(m1.T.dot(m2))
    #     ref_angle = abs(ref_angle)
    #     self.assertAlmostEqual(angle, ref_angle, msg='{} != {}'.format(angle, ref_angle))

    # fails if numbers too big or too small
    # TODO buggy
    # @given(unit_vector(length=4, elements=float_nonan_noinf_nobig_nosmall))
    # def test_axis_angle_from_quaternion1(self, q):
    #     assume(not np.isclose(q,[0,0,0,1], atol=1.e-4).all())
    #     assume(not np.isclose(q,[0,0,0,-1], atol=1.e-4).all())
    #     axis, angle = spw.axis_angle_from_quaternion(*q)
    #     angle = float(angle)
    #     axis = np.array(axis).astype(float).T[0]
    #     angle2, axis2, _ = rotation_from_matrix(quaternion_matrix(q))
    #     if angle < 0:
    #         angle = -angle
    #         axis = [-x for x in axis]
    #     if angle2 < 0:
    #         angle2 = -angle2
    #         axis2 *= -1
    #     self.assertTrue(np.isclose(angle, angle2), msg='{} != {}'.format(angle, angle2))
    #     self.assertTrue(np.isclose(axis, axis2).all(), msg='{} != {}'.format(axis, axis2))

    # fails if numbers too big or too small
    @given(unit_vector(length=3),
           angle())
    def test_quaternion_from_axis_angle1(self, axis, angle):
        r2 = quaternion_about_axis(angle, axis)
        self.assertTrue(np.isclose(w.compile_and_execute(w.quaternion_from_axis_angle, [axis, angle]),
                                   r2).all())

    # fails if numbers too big or too small
    @given(angle(),
           angle(),
           angle())
    def test_axis_angle_from_rpy(self, roll, pitch, yaw):
        axis2, angle2 = euler2axangle(roll, pitch, yaw)
        assume(abs(angle2) > SMALL_NUMBER)
        axis = w.compile_and_execute(lambda r, p, y: w.axis_angle_from_rpy(r, p, y)[0], [roll, pitch, yaw])
        angle = w.compile_and_execute(lambda r, p, y: w.axis_angle_from_rpy(r, p, y)[1], [roll, pitch, yaw])
        if angle < 0:
            angle = -angle
            axis = [-x for x in axis]
        if angle2 < 0:
            angle2 = -angle2
            axis2 *= -1
        compare_axis_angle(angle, axis, angle2, axis2)

    # fails if numbers too big or too small
    @given(quaternion())
    def test_axis_angle_from_quaternion(self, q):
        axis2, angle2 = quat2axangle([q[-1], q[0], q[1], q[2]])
        axis = w.compile_and_execute(lambda x,y,z,w_: w.axis_angle_from_quaternion(x,y,z,w_)[0], q)
        angle = w.compile_and_execute(lambda x,y,z,w_: w.axis_angle_from_quaternion(x,y,z,w_)[1], q)
        # axis = [x, y, z]
        # angle = float(angle)
        compare_axis_angle(angle, axis, angle2, axis2, 2)

    def test_axis_angle_from_quaternion2(self):
        q = [0, 0, 0, 1.0000001]
        axis2, angle2 = quat2axangle([q[-1], q[0], q[1], q[2]])
        axis = w.compile_and_execute(lambda x, y, z, w_: w.axis_angle_from_quaternion(x, y, z, w_)[0], q)
        angle = w.compile_and_execute(lambda x, y, z, w_: w.axis_angle_from_quaternion(x, y, z, w_)[1], q)
        compare_axis_angle(angle, axis, angle2, axis2, 2)

    # fails if numbers too big or too small
    # TODO rpy does not follow some conventions I guess
    # @given(unit_vector(4))
    # @reproduce_failure('3.57.0', 'AXicY2BABgUXRSZL1n9jYMAi+pkBAG0ABuo=')
    # def test_rpy_from_matrix(self, q):
    #     matrix = quaternion_matrix(q)
    #     roll, pitch, yaw = spw.rpy_from_matrix(matrix)
    #     roll = float(roll.evalf(real=True))
    #     pitch = float(pitch.evalf(real=True))
    #     yaw = float(yaw.evalf(real=True))
    #     roll2, pitch2, yaw2 = euler_from_matrix(matrix)
    #     self.assertTrue(np.isclose(roll, roll2), msg='{} != {}'.format(roll, roll2))
    #     self.assertTrue(np.isclose(pitch, pitch2), msg='{} != {}'.format(pitch, pitch2))
    #     self.assertTrue(np.isclose(yaw, yaw2), msg='{} != {}'.format(yaw, yaw2))

    # fails if numbers too big or too small
    @given(unit_vector(4))
    def test_rpy_from_matrix(self, q):
        matrix = quaternion_matrix(q)
        roll = w.compile_and_execute(lambda m: w.rpy_from_matrix(m)[0], [matrix])
        pitch = w.compile_and_execute(lambda m: w.rpy_from_matrix(m)[1], [matrix])
        yaw = w.compile_and_execute(lambda m: w.rpy_from_matrix(m)[2], [matrix])
        try:
            roll = float(roll.evalf(real=True))
        except AttributeError:
            pass
        try:
            pitch = float(pitch.evalf(real=True))
        except AttributeError:
            pass
        try:
            yaw = float(yaw.evalf(real=True))
        except AttributeError:
            pass
        r1 = w.compile_and_execute(w.rotation_matrix_from_rpy, [roll, pitch, yaw])
        self.assertTrue(np.isclose(r1, matrix).all(), msg='{} != {}'.format(r1, matrix))

    # fails if numbers too big or too small
    @given(angle(),
           angle(),
           angle())
    def test_quaternion_from_rpy(self, roll, pitch, yaw):
        q = w.compile_and_execute(w.quaternion_from_rpy, [roll, pitch, yaw])
        q2 = quaternion_from_euler(roll, pitch, yaw)
        self.assertTrue(np.isclose(q, q2).all(), msg='{} != {}'.format(q, q2))

    # fails if numbers too big or too small
    @given(quaternion())
    def test_quaternion_from_matrix(self, q):
        matrix = quaternion_matrix(q)
        q2 = quaternion_from_matrix(matrix)
        q1_2 = w.compile_and_execute(w.quaternion_from_matrix, [matrix])
        self.assertTrue(np.isclose(q1_2, q2).all() or np.isclose(q1_2, -q2).all(), msg='{} != {}'.format(q, q1_2))

    # fails if numbers too big or too small
    @given(quaternion(),
           quaternion())
    def test_quaternion_multiply(self, q, p):
        r1 = w.compile_and_execute(w.quaternion_multiply, [q, p])
        r2 = quaternion_multiply(q, p)
        self.assertTrue(np.isclose(r1, r2).all() or np.isclose(r1, -r2).all(), msg='{} != {}'.format(r1, r2))

    # fails if numbers too big or too small
    @given(quaternion())
    def test_quaternion_conjugate(self, q):
        r1 = w.compile_and_execute(w.quaternion_conjugate, [q])
        r2 = quaternion_conjugate(q)
        self.assertTrue(np.isclose(r1, r2).all() or np.isclose(r1, -r2).all(), msg='{} != {}'.format(r1, r2))

    # fails if numbers too big or too small
    @given(quaternion(),
           quaternion())
    def test_quaternion_diff(self, q1, q2):
        q3 = quaternion_multiply(quaternion_conjugate(q1), q2)
        q4 = w.compile_and_execute(w.quaternion_diff, [q1, q2])
        self.assertTrue(np.isclose(q3, q4).all() or np.isclose(q3, -q4).all(), msg='{} != {}'.format(q1, q4))

    # TODO cosine distance

    # fails if numbers too big or too small
    # TODO moved to utils
    # @given(quaternion(),
    #        quaternion(),
    #        st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1))
    # def test_slerp(self, q1, q2, t):
    #     r1 = np.array([float(x.evalf(real=True)) for x in spw.slerp(spw.Matrix(q1), spw.Matrix(q2), t)])
    #     r2 = quaternion_slerp(q1, q2, t)
    #     self.assertTrue(np.isclose(r1, r2, atol=1e-3).all() or
    #                     np.isclose(r1, -r2, atol=1e-3).all(),
    #                     msg='q1={} q2={} t={}\n{} != {}'.format(q1, q2, t, r1, r2))

    # fails if numbers too big or too small
    @given(quaternion(),
           quaternion(),
           st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1))
    def test_slerp(self, q1, q2, t):
        r1 = w.compile_and_execute(w.diffable_slerp, [q1, q2, t])
        r2 = quaternion_slerp(q1, q2, t)
        self.assertTrue(np.isclose(r1, r2, atol=1e-3).all() or
                        np.isclose(r1, -r2, atol=1e-3).all(),
                        msg='q1={} q2={} t={}\n{} != {}'.format(q1, q2, t, r1, r2))

    # fails if numbers too big or too small
    @given(limited_float(outer_limit=1e5, min_dist_to_zero=1e-4),
           limited_float(outer_limit=1e5, min_dist_to_zero=1e-4))
    def test_fmod(self, a, b):
        assume(b != 0)
        ref_r = np.fmod(a, b)
        self.assertAlmostEqual(w.compile_and_execute(w.fmod, [a, b]), ref_r, places=4)

    # fails if numbers too big or too small
    @given(limited_float(outer_limit=1e10))
    def test_normalize_angle_positive(self, a):
        a = a * np.pi
        ref_r = normalize_angle_positive(a)
        sw_r = w.compile_and_execute(w.normalize_angle_positive, [a])

        self.assertAlmostEqual(shortest_angular_distance(ref_r, sw_r), 0.0, places=5)

    # fails if numbers too big or too small
    @given(limited_float(outer_limit=1e5))
    def test_normalize_angle(self, a):
        a = a * np.pi
        ref_r = normalize_angle(a)
        self.assertAlmostEqual(w.compile_and_execute(w.normalize_angle, [a]), ref_r, places=5)

    # fails if numbers too big or too small
    @given(limited_float(outer_limit=1e3),
           limited_float(outer_limit=1e3))
    def test_shorted_angular_distance(self, f1, f2):
        angle1 = np.pi * f1
        angle2 = np.pi * f2
        ref_distance = shortest_angular_distance(angle1, angle2)
        distance = w.compile_and_execute(w.shortest_angular_distance, [angle1, angle2])
        self.assertAlmostEqual(distance, ref_distance, places=7)
        assert abs(distance) <= np.pi

    @given(unit_vector(4),
           unit_vector(4))
    def test_entrywise_product(self, q1, q2):
        # TODO use real matrices
        m1 = quat2mat(q1)
        m2 = quat2mat(q2)
        r1 = w.compile_and_execute(w.entrywise_product, [m1, m2])
        r2 = m1 * m2
        np.testing.assert_array_almost_equal(r1, r2)
