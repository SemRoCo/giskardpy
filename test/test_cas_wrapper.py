import unittest

import PyKDL
import hypothesis.strategies as st
import numpy as np
from angles import shortest_angular_distance, normalize_angle_positive, normalize_angle
from hypothesis import given, assume
from tf.transformations import quaternion_matrix, quaternion_about_axis, quaternion_from_euler, euler_matrix, \
    rotation_matrix, quaternion_multiply, quaternion_conjugate, quaternion_from_matrix, \
    quaternion_slerp, rotation_from_matrix, euler_from_matrix

from giskardpy import casadi_wrapper as w
from giskardpy.casadi_wrapper import rotation_matrix_from_quaternion
from giskardpy.utils.math import compare_orientations, axis_angle_from_quaternion, rotation_matrix_from_quaternion
from utils_for_tests import float_no_nan_no_inf, unit_vector, quaternion, vector, \
    pykdl_frame_to_numpy, lists_of_same_length, random_angle, compare_axis_angle, angle_positive, sq_matrix


class TestCASWrapper(unittest.TestCase):

    def test_free_symbols(self):
        m = w.Matrix(w.var('a b c d'))
        assert len(w.free_symbols(m)) == 4
        a = w.Symbol('a')
        assert w.equivalent(a, w.free_symbols(a)[0])

    def test_is_matrix(self):
        self.assertFalse(w.is_matrix(w.Symbol('a')))
        self.assertTrue(w.is_matrix(w.Matrix([[0, 0]])))

    def test_jacobian(self):
        a = w.Symbol('a')
        b = w.Symbol('b')
        m = w.Matrix([a + b, a ** 2, b ** 2])
        jac = w.jacobian(m, [a, b])
        expected = w.Matrix([[1, 1], [2 * a, 0], [0, 2 * b]])
        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                assert w.equivalent(jac[i, j], expected[i, j])

    def test_jacobian_order2(self):
        a = w.Symbol('a')
        b = w.Symbol('b')
        m = w.Matrix([a + b, a ** 2 + b, a ** 3 + b ** 2])
        jac = w.jacobian(m, [a, b], order=2)
        expected = w.Matrix([[0, 0], [2, 0], [6 * a, 2]])
        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                assert w.equivalent(jac[i, j], expected[i, j])

    def test_var(self):
        result = w.var('a b c')
        assert str(result[0]) == 'a'
        assert str(result[1]) == 'b'
        assert str(result[2]) == 'c'

    def test_diag(self):
        result = w.diag([1, 2, 3])
        assert result[0, 0] == 1
        assert result[0, 1] == 0
        assert result[0, 2] == 0

        assert result[1, 0] == 0
        assert result[1, 1] == 2
        assert result[1, 2] == 0

        assert result[2, 0] == 0
        assert result[2, 1] == 0
        assert result[2, 2] == 3
        assert w.equivalent(w.diag(w.Matrix([1, 2, 3])), w.diag([1, 2, 3]))

    @given(float_no_nan_no_inf())
    def test_abs(self, f1):
        self.assertAlmostEqual(w.compile_and_execute(w.abs, [f1]), abs(f1), places=7)

    @given(float_no_nan_no_inf(),
           float_no_nan_no_inf())
    def test_max(self, f1, f2):
        self.assertAlmostEqual(w.compile_and_execute(w.max, [f1, f2]),
                               max(f1, f2), places=7)

    @given(float_no_nan_no_inf(),
           float_no_nan_no_inf())
    def test_save_division(self, f1, f2):
        self.assertTrue(np.isclose(w.compile_and_execute(w.save_division, [f1, f2]),
                                   f1 / f2 if f2 != 0 else 0))

    @given(float_no_nan_no_inf(),
           float_no_nan_no_inf())
    def test_min(self, f1, f2):
        self.assertAlmostEqual(w.compile_and_execute(w.min, [f1, f2]),
                               min(f1, f2), places=7)

    @given(st.integers(min_value=1, max_value=10))
    def test_matrix(self, x_dim):
        data = list(range(x_dim))
        m = w.Matrix(data)
        self.assertEqual(m[0], 0)
        self.assertEqual(m[-1], x_dim - 1)

    @given(st.integers(min_value=1, max_value=10),
           st.integers(min_value=1, max_value=10))
    def test_matrix2(self, x_dim, y_dim):
        data = [[i + (j * x_dim) for j in range(y_dim)] for i in range(x_dim)]
        m = w.Matrix(data)
        self.assertEqual(float(m[0, 0]), 0)
        self.assertEqual(float(m[x_dim - 1, y_dim - 1]), (x_dim * y_dim) - 1)

    @given(float_no_nan_no_inf())
    def test_sign(self, f1):
        self.assertAlmostEqual(w.compile_and_execute(w.sign, [f1]),
                               np.sign(f1), places=7)

    @given(float_no_nan_no_inf(),
           float_no_nan_no_inf(),
           float_no_nan_no_inf())
    def test_if_greater_zero(self, condition, if_result, else_result):
        self.assertAlmostEqual(w.compile_and_execute(w.if_greater_zero, [condition, if_result, else_result]),
                               np.float(if_result if condition > 0 else else_result), places=7)

    @given(float_no_nan_no_inf(),
           float_no_nan_no_inf(),
           float_no_nan_no_inf())
    def test_if_greater_eq_zero(self, condition, if_result, else_result):
        self.assertAlmostEqual(w.compile_and_execute(w.if_greater_eq_zero, [condition, if_result, else_result]),
                               np.float(if_result if condition >= 0 else else_result), places=7)

    @given(float_no_nan_no_inf(),
           float_no_nan_no_inf(),
           float_no_nan_no_inf(),
           float_no_nan_no_inf())
    def test_if_greater_eq(self, a, b, if_result, else_result):
        self.assertAlmostEqual(
            w.compile_and_execute(w.if_greater_eq, [a, b, if_result, else_result]),
            np.float(if_result if a >= b else else_result), places=7)

    @given(float_no_nan_no_inf(),
           float_no_nan_no_inf(),
           float_no_nan_no_inf(),
           float_no_nan_no_inf())
    def test_if_less_eq(self, a, b, if_result, else_result):
        self.assertAlmostEqual(
            w.compile_and_execute(w.if_less_eq, [a, b, if_result, else_result]),
            np.float(if_result if a <= b else else_result), places=7)

    @given(float_no_nan_no_inf(),
           float_no_nan_no_inf(),
           float_no_nan_no_inf())
    def test_if_eq_zero(self, condition, if_result, else_result):
        self.assertAlmostEqual(
            w.compile_and_execute(w.if_eq_zero, [condition, if_result, else_result]),
            np.float(if_result if condition == 0 else else_result), places=7)

    @given(float_no_nan_no_inf(),
           float_no_nan_no_inf(),
           float_no_nan_no_inf(),
           float_no_nan_no_inf())
    def test_if_eq(self, a, b, if_result, else_result):
        self.assertTrue(np.isclose(
            w.compile_and_execute(w.if_eq, [a, b, if_result, else_result]),
            np.float(if_result if a == b else else_result)))

    @given(float_no_nan_no_inf())
    def test_if_eq_cases(self, a):
        b_result_cases = [(1, 1),
                          (3, 3),
                          (4, 4),
                          (-1, -1),
                          (0.5, 0.5),
                          (-0.5, -0.5)]

        def reference(a_, b_result_cases_, else_result):
            for b, if_result in b_result_cases_:
                if a_ == b:
                    return if_result
            return else_result

        self.assertTrue(np.isclose(
            w.compile_and_execute(w.if_eq_cases, [a, b_result_cases, 0]),
            np.float(reference(a, b_result_cases, 0))))

    @given(float_no_nan_no_inf())
    def test_if_less_eq_cases(self, a):
        b_result_cases = [
            (-1, -1),
            (-0.5, -0.5),
            (0.5, 0.5),
            (1, 1),
            (3, 3),
            (4, 4),
        ]

        def reference(a_, b_result_cases_, else_result):
            for b, if_result in b_result_cases_:
                if a_ <= b:
                    return if_result
            return else_result

        self.assertAlmostEqual(w.compile_and_execute(w.if_less_eq_cases, [a, b_result_cases, 0]),
                               np.float(reference(a, b_result_cases, 0)))

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

    @given(float_no_nan_no_inf(),
           float_no_nan_no_inf(),
           float_no_nan_no_inf(),
           float_no_nan_no_inf())
    def test_if_greater(self, a, b, if_result, else_result):
        self.assertAlmostEqual(
            w.compile_and_execute(w.if_greater, [a, b, if_result, else_result]),
            np.float(if_result if a > b else else_result), places=7)

    @given(float_no_nan_no_inf(),
           float_no_nan_no_inf(),
           float_no_nan_no_inf(),
           float_no_nan_no_inf())
    def test_if_less(self, a, b, if_result, else_result):
        self.assertAlmostEqual(
            w.compile_and_execute(w.if_less, [a, b, if_result, else_result]),
            np.float(if_result if a < b else else_result), places=7)

    # fails if numbers too big or too small
    @given(unit_vector(length=3),
           random_angle())
    def test_speed_up_matrix_from_axis_angle(self, axis, angle):
        np.testing.assert_array_almost_equal(
            w.compile_and_execute(w.rotation_matrix_from_axis_angle, [axis, angle]),
            rotation_matrix(angle, axis))

    @given(vector(3),
           vector(3))
    def test_cross(self, u, v):
        np.testing.assert_array_almost_equal(
            w.compile_and_execute(w.cross, [u, v])[:3],
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

    @given(st.lists(float_no_nan_no_inf()))
    def test_norm(self, v):
        actual = w.compile_and_execute(w.norm, [v])
        expected = np.linalg.norm(v)
        assume(not np.isinf(expected))
        self.assertTrue(np.isclose(actual, expected, equal_nan=True))

    @given(vector(3),
           float_no_nan_no_inf())
    def test_scale(self, v, a):
        if np.linalg.norm(v) == 0:
            r2 = [0, 0, 0]
        else:
            r2 = v / np.linalg.norm(v) * a
        np.testing.assert_array_almost_equal(
            w.compile_and_execute(w.scale, [v, a]),
            r2)

    @given(lists_of_same_length([float_no_nan_no_inf(), float_no_nan_no_inf()], max_length=50))
    def test_dot(self, vectors):
        u, v = vectors
        u = np.array(u, ndmin=2)
        v = np.array(v, ndmin=2)
        result = w.compile_and_execute(w.dot, [u, v.T])
        if not np.isnan(result):
            self.assertTrue(np.isclose(result, np.dot(u, v.T)))

    @given(float_no_nan_no_inf(),
           float_no_nan_no_inf(),
           float_no_nan_no_inf())
    def test_translation3(self, x, y, z):
        r1 = w.compile_and_execute(w.translation3, [x, y, z])
        r2 = np.identity(4)
        r2[0, 3] = x
        r2[1, 3] = y
        r2[2, 3] = z
        self.assertTrue(np.isclose(r1, r2).all(), msg='{} != {}'.format(r1, r2))

    @given(random_angle(),
           random_angle(),
           random_angle())
    def test_rotation_matrix_from_rpy(self, roll, pitch, yaw):
        m1 = w.compile_and_execute(w.rotation_matrix_from_rpy, [roll, pitch, yaw])
        m2 = euler_matrix(roll, pitch, yaw)
        np.testing.assert_array_almost_equal(m1, m2)

    @given(unit_vector(length=3),
           random_angle())
    def test_rotation3_axis_angle(self, axis, angle):
        np.testing.assert_array_almost_equal(w.compile_and_execute(w.rotation_matrix_from_axis_angle, [axis, angle]),
                                             rotation_matrix(angle, np.array(axis)))

    @given(quaternion())
    def test_rotation3_quaternion(self, q):
        np.testing.assert_array_almost_equal(w.compile_and_execute(w.rotation_matrix_from_quaternion, q),
                                             quaternion_matrix(q))

    @given(float_no_nan_no_inf(),
           float_no_nan_no_inf(),
           float_no_nan_no_inf(),
           unit_vector(length=3),
           random_angle())
    def test_frame3_axis_angle(self, x, y, z, axis, angle):
        r2 = rotation_matrix(angle, np.array(axis))
        r2[0, 3] = x
        r2[1, 3] = y
        r2[2, 3] = z
        np.testing.assert_array_almost_equal(w.compile_and_execute(w.frame_axis_angle, [x, y, z, axis, angle]),
                                             r2)

    @given(float_no_nan_no_inf(),
           float_no_nan_no_inf(),
           float_no_nan_no_inf(),
           random_angle(),
           random_angle(),
           random_angle())
    def test_frame3_rpy(self, x, y, z, roll, pitch, yaw):
        r2 = euler_matrix(roll, pitch, yaw)
        r2[0, 3] = x
        r2[1, 3] = y
        r2[2, 3] = z
        np.testing.assert_array_almost_equal(w.compile_and_execute(w.frame_rpy, [x, y, z, roll, pitch, yaw]),
                                             r2)

    @given(float_no_nan_no_inf(),
           float_no_nan_no_inf(),
           float_no_nan_no_inf(),
           unit_vector(4))
    def test_frame3_quaternion(self, x, y, z, q):
        r2 = quaternion_matrix(q)
        r2[0, 3] = x
        r2[1, 3] = y
        r2[2, 3] = z
        np.testing.assert_array_almost_equal(w.compile_and_execute(w.frame_quaternion, [x, y, z] + q.tolist()),
                                             r2)

    @given(float_no_nan_no_inf(outer_limit=1000),
           float_no_nan_no_inf(outer_limit=1000),
           float_no_nan_no_inf(outer_limit=1000),
           quaternion())
    def test_inverse_frame(self, x, y, z, q):
        f = quaternion_matrix(q)
        f[0, 3] = x
        f[1, 3] = y
        f[2, 3] = z
        r = w.compile_and_execute(w.inverse_frame, [f])

        r2 = PyKDL.Frame()
        r2.M = PyKDL.Rotation.Quaternion(q[0], q[1], q[2], q[3])
        r2.p[0] = x
        r2.p[1] = y
        r2.p[2] = z
        r2 = r2.Inverse()
        r2 = pykdl_frame_to_numpy(r2)
        self.assertTrue(np.isclose(r, r2, atol=1.e-4, rtol=1.e-4).all())

    @given(float_no_nan_no_inf(),
           float_no_nan_no_inf(),
           float_no_nan_no_inf(),
           unit_vector(4))
    def test_pos_of(self, x, y, z, q):
        r1 = w.position_of(w.frame_quaternion(x, y, z, q[0], q[1], q[2], q[3]))
        r2 = [x, y, z, 1]
        for i, e in enumerate(r2):
            self.assertAlmostEqual(r1[i], e)

    @given(float_no_nan_no_inf(),
           float_no_nan_no_inf(),
           float_no_nan_no_inf(),
           unit_vector(4))
    def test_trans_of(self, x, y, z, q):
        r1 = w.compile_and_execute(lambda *args: w.translation_of(w.frame_quaternion(*args)),
                                   [x, y, z, q[0], q[1], q[2], q[3]])
        r2 = np.identity(4)
        r2[0, 3] = x
        r2[1, 3] = y
        r2[2, 3] = z
        for i in range(r2.shape[0]):
            for j in range(r2.shape[1]):
                self.assertAlmostEqual(float(r1[i, j]), r2[i, j])

    @given(float_no_nan_no_inf(),
           float_no_nan_no_inf(),
           float_no_nan_no_inf(),
           unit_vector(4))
    def test_rot_of(self, x, y, z, q):
        r1 = w.compile_and_execute(lambda *args: w.rotation_of(w.frame_quaternion(*args)),
                                   [x, y, z, q[0], q[1], q[2], q[3]])
        r2 = quaternion_matrix(q)
        self.assertTrue(np.isclose(r1, r2).all(), msg='\n{} != \n{}'.format(r1, r2))

    def test_rot_of2(self):
        """
        Test to make sure the function doesn't alter the original
        """
        f = w.translation3(1, 2, 3)
        r = w.rotation_of(f)
        self.assertTrue(f[0, 3], 1)
        self.assertTrue(f[0, 3], 2)
        self.assertTrue(f[0, 3], 3)
        self.assertTrue(r[0, 0], 1)
        self.assertTrue(r[1, 1], 1)
        self.assertTrue(r[2, 2], 1)

    @given(unit_vector(4))
    def test_trace(self, q):
        m = quaternion_matrix(q)
        np.testing.assert_array_almost_equal(w.compile_and_execute(w.trace, [m]), np.trace(m))

    @given(quaternion(),
           quaternion())
    def test_rotation_distance(self, q1, q2):
        m1 = quaternion_matrix(q1)
        m2 = quaternion_matrix(q2)
        actual_angle = w.compile_and_execute(w.rotation_distance, [m1, m2])
        _, expected_angle = axis_angle_from_quaternion(*quaternion_from_matrix(m1.T.dot(m2)))
        expected_angle = expected_angle
        try:
            self.assertAlmostEqual(shortest_angular_distance(actual_angle, expected_angle), 0, places=3)
        except AssertionError:
            self.assertAlmostEqual(shortest_angular_distance(actual_angle, -expected_angle), 0, places=3)

    @given(quaternion())
    def test_axis_angle_from_matrix(self, q):
        m = quaternion_matrix(q)
        actual_axis = w.compile_and_execute(lambda x: w.axis_angle_from_matrix(x)[0], [m])
        actual_angle = w.compile_and_execute(lambda x: w.axis_angle_from_matrix(x)[1], [m])
        expected_angle, expected_axis, _ = rotation_from_matrix(m)
        compare_axis_angle(actual_angle, actual_axis, expected_angle, expected_axis)

    @given(unit_vector(length=3),
           angle_positive())
    def test_axis_angle_from_matrix2(self, expected_axis, expected_angle):
        m = rotation_matrix(expected_angle, expected_axis)
        actual_axis = w.compile_and_execute(lambda x: w.axis_angle_from_matrix(x)[0], [m])
        actual_angle = w.compile_and_execute(lambda x: w.axis_angle_from_matrix(x)[1], [m])
        expected_angle, expected_axis, _ = rotation_from_matrix(m)
        compare_axis_angle(actual_angle, actual_axis, expected_angle, expected_axis)

    @given(unit_vector(length=3),
           random_angle())
    def test_quaternion_from_axis_angle1(self, axis, angle):
        r2 = quaternion_about_axis(angle, axis)
        self.assertTrue(np.isclose(w.compile_and_execute(w.quaternion_from_axis_angle, [axis, angle]),
                                   r2).all())

    @given(random_angle(),
           random_angle(),
           random_angle())
    def test_axis_angle_from_rpy(self, roll, pitch, yaw):
        angle2, axis2, _ = rotation_from_matrix(euler_matrix(roll, pitch, yaw))
        axis = w.compile_and_execute(lambda r, p, y: w.axis_angle_from_rpy(r, p, y)[0], [roll, pitch, yaw])
        angle = w.compile_and_execute(lambda r, p, y: w.axis_angle_from_rpy(r, p, y)[1], [roll, pitch, yaw])
        if angle < 0:
            angle = -angle
            axis = [-x for x in axis]
        if angle2 < 0:
            angle2 = -angle2
            axis2 *= -1
        compare_axis_angle(angle, axis, angle2, axis2)

    @given(quaternion())
    def test_axis_angle_from_quaternion(self, q):
        axis2, angle2 = axis_angle_from_quaternion(q[0], q[1], q[2], q[3])
        axis = w.compile_and_execute(lambda x, y, z, w_: w.axis_angle_from_quaternion(x, y, z, w_)[0], q)
        angle = w.compile_and_execute(lambda x, y, z, w_: w.axis_angle_from_quaternion(x, y, z, w_)[1], q)
        compare_axis_angle(angle, axis, angle2, axis2, 2)

    def test_axis_angle_from_quaternion2(self):
        q = [0, 0, 0, 1.0000001]
        axis2, angle2 = axis_angle_from_quaternion(q[0], q[1], q[2], q[3])
        axis = w.compile_and_execute(lambda x, y, z, w_: w.axis_angle_from_quaternion(x, y, z, w_)[0], q)
        angle = w.compile_and_execute(lambda x, y, z, w_: w.axis_angle_from_quaternion(x, y, z, w_)[1], q)
        compare_axis_angle(angle, axis, angle2, axis2, 2)

    @given(unit_vector(4))
    def test_rpy_from_matrix(self, q):
        matrix = quaternion_matrix(q)
        roll = w.compile_and_execute(lambda m: w.rpy_from_matrix(m)[0], [matrix])
        pitch = w.compile_and_execute(lambda m: w.rpy_from_matrix(m)[1], [matrix])
        yaw = w.compile_and_execute(lambda m: w.rpy_from_matrix(m)[2], [matrix])
        roll2, pitch2, yaw2 = euler_from_matrix(matrix)
        self.assertTrue(np.isclose(roll, roll2), msg='{} != {}'.format(roll, roll2))
        self.assertTrue(np.isclose(pitch, pitch2), msg='{} != {}'.format(pitch, pitch2))
        self.assertTrue(np.isclose(yaw, yaw2), msg='{} != {}'.format(yaw, yaw2))

    @given(unit_vector(4))
    def test_rpy_from_matrix2(self, q):
        matrix = quaternion_matrix(q)
        roll = w.compile_and_execute(lambda m: w.rpy_from_matrix(m)[0], [matrix])
        pitch = w.compile_and_execute(lambda m: w.rpy_from_matrix(m)[1], [matrix])
        yaw = w.compile_and_execute(lambda m: w.rpy_from_matrix(m)[2], [matrix])
        r1 = w.compile_and_execute(w.rotation_matrix_from_rpy, [roll, pitch, yaw])
        self.assertTrue(np.isclose(r1, matrix, atol=1.e-4).all(), msg='{} != {}'.format(r1, matrix))

    @given(random_angle(),
           random_angle(),
           random_angle())
    def test_quaternion_from_rpy(self, roll, pitch, yaw):
        q = w.compile_and_execute(w.quaternion_from_rpy, [roll, pitch, yaw])
        q2 = quaternion_from_euler(roll, pitch, yaw)
        self.assertTrue(np.isclose(q, q2).all(), msg='{} != {}'.format(q, q2))

    @given(quaternion())
    def test_quaternion_from_matrix(self, q):
        matrix = quaternion_matrix(q)
        q2 = quaternion_from_matrix(matrix)
        q1_2 = w.compile_and_execute(w.quaternion_from_matrix, [matrix])
        self.assertTrue(np.isclose(q1_2, q2).all() or np.isclose(q1_2, -q2).all(), msg='{} != {}'.format(q, q1_2))

    @given(quaternion(),
           quaternion())
    def test_quaternion_multiply(self, q, p):
        r1 = w.compile_and_execute(w.quaternion_multiply, [q, p])
        r2 = quaternion_multiply(q, p)
        self.assertTrue(np.isclose(r1, r2).all() or np.isclose(r1, -r2).all(), msg='{} != {}'.format(r1, r2))

    @given(quaternion())
    def test_quaternion_conjugate(self, q):
        r1 = w.compile_and_execute(w.quaternion_conjugate, [q])
        r2 = quaternion_conjugate(q)
        self.assertTrue(np.isclose(r1, r2).all() or np.isclose(r1, -r2).all(), msg='{} != {}'.format(r1, r2))

    @given(quaternion(),
           quaternion())
    def test_quaternion_diff(self, q1, q2):
        q3 = quaternion_multiply(quaternion_conjugate(q1), q2)
        q4 = w.compile_and_execute(w.quaternion_diff, [q1, q2])
        self.assertTrue(np.isclose(q3, q4).all() or np.isclose(q3, -q4).all(), msg='{} != {}'.format(q1, q4))

    @given(quaternion(),
           quaternion(),
           st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1))
    def test_slerp(self, q1, q2, t):
        r1 = w.compile_and_execute(w.quaternion_slerp, [q1, q2, t])
        r2 = quaternion_slerp(q1, q2, t)
        self.assertTrue(np.isclose(r1, r2, atol=1e-3).all() or
                        np.isclose(r1, -r2, atol=1e-3).all(),
                        msg='q1={} q2={} t={}\n{} != {}'.format(q1, q2, t, r1, r2))

    @given(quaternion(),
           quaternion())
    def test_slerp123(self, q1, q2):
        step = 0.1
        q_d = w.compile_and_execute(w.quaternion_diff, [q1, q2])
        axis = w.compile_and_execute(lambda x, y, z, w_: w.axis_angle_from_quaternion(x, y, z, w_)[0], q_d)
        angle = w.compile_and_execute(lambda x, y, z, w_: w.axis_angle_from_quaternion(x, y, z, w_)[1], q_d)
        assume(angle != np.pi)
        if np.abs(angle) > np.pi:
            angle = angle - np.pi * 2
        elif np.abs(angle) < -np.pi:
            angle = angle + np.pi * 2
        r1s = []
        r2s = []
        for t in np.arange(0, 1.001, step):
            r1 = w.compile_and_execute(w.quaternion_slerp, [q1, q2, t])
            r1 = w.compile_and_execute(w.quaternion_diff, [q1, r1])
            axis2 = w.compile_and_execute(lambda x, y, z, w_: w.axis_angle_from_quaternion(x, y, z, w_)[0], r1)
            angle2 = w.compile_and_execute(lambda x, y, z, w_: w.axis_angle_from_quaternion(x, y, z, w_)[1], r1)
            r2 = w.compile_and_execute(w.quaternion_from_axis_angle, [axis, angle * t])
            r1s.append(r1)
            r2s.append(r2)
        aa1 = []
        aa2 = []
        for r1, r2 in zip(r1s, r2s):
            axisr1 = w.compile_and_execute(lambda x, y, z, w_: w.axis_angle_from_quaternion(x, y, z, w_)[0], r1)
            angler1 = w.compile_and_execute(lambda x, y, z, w_: w.axis_angle_from_quaternion(x, y, z, w_)[1], r1)
            aa1.append([axisr1, angler1])
            axisr2 = w.compile_and_execute(lambda x, y, z, w_: w.axis_angle_from_quaternion(x, y, z, w_)[0], r2)
            angler2 = w.compile_and_execute(lambda x, y, z, w_: w.axis_angle_from_quaternion(x, y, z, w_)[1], r2)
            aa2.append([axisr2, angler2])
        aa1 = np.array(aa1)
        aa2 = np.array(aa2)
        r1snp = np.array(r1s)
        r2snp = np.array(r2s)
        qds = []
        for i in range(len(r1s) - 1):
            q1t = r1s[i]
            q2t = r1s[i + 1]
            qds.append(w.compile_and_execute(w.quaternion_diff, [q1t, q2t]))
        qds = np.array(qds)
        for r1, r2 in zip(r1s, r2s):
            compare_orientations(r1, r2)

    # fails if numbers too big or too small
    @given(unit_vector(3),
           unit_vector(3),
           st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1))
    def test_slerp2(self, q1, q2, t):
        r1 = w.compile_and_execute(w.quaternion_slerp, [q1, q2, t])
        r2 = quaternion_slerp(q1, q2, t)
        self.assertTrue(np.isclose(r1, r2, atol=1e-3).all() or
                        np.isclose(r1, -r2, atol=1e-3).all(),
                        msg='q1={} q2={} t={}\n{} != {}'.format(q1, q2, t, r1, r2))

    @given(float_no_nan_no_inf(),
           float_no_nan_no_inf())
    def test_fmod(self, a, b):
        ref_r = np.fmod(a, b)
        self.assertTrue(np.isclose(w.compile_and_execute(w.fmod, [a, b]), ref_r, equal_nan=True))

    @given(float_no_nan_no_inf())
    def test_normalize_angle_positive(self, a):
        expected = normalize_angle_positive(a)
        actual = w.compile_and_execute(w.normalize_angle_positive, [a])
        self.assertAlmostEqual(shortest_angular_distance(expected, actual), 0.0, places=5)

    @given(float_no_nan_no_inf())
    def test_normalize_angle(self, a):
        ref_r = normalize_angle(a)
        self.assertAlmostEqual(w.compile_and_execute(w.normalize_angle, [a]), ref_r, places=5)

    @given(float_no_nan_no_inf(),
           float_no_nan_no_inf())
    def test_shorted_angular_distance(self, angle1, angle2):
        try:
            expected = shortest_angular_distance(angle1, angle2)
        except ValueError:
            expected = np.nan
        actual = w.compile_and_execute(w.shortest_angular_distance, [angle1, angle2])
        self.assertTrue(np.isclose(actual, expected, equal_nan=True))

    @given(unit_vector(4),
           unit_vector(4))
    def test_entrywise_product(self, q1, q2):
        m1 = rotation_matrix_from_quaternion(q1[0], q1[1], q1[2], q1[3])
        m2 = rotation_matrix_from_quaternion(q2[0], q2[1], q2[2], q2[3])
        r1 = w.compile_and_execute(w.entrywise_product, [m1, m2])
        r2 = m1 * m2
        np.testing.assert_array_almost_equal(r1, r2)

    @given(sq_matrix())
    def test_sum(self, m):
        actual_sum = w.compile_and_execute(w.sum, [m])
        expected_sum = np.sum(m)
        self.assertTrue(np.isclose(actual_sum, expected_sum, rtol=1.e-4))

    @given(st.integers(max_value=10000, min_value=1),
           st.integers(max_value=5000, min_value=-5000),
           st.integers(max_value=5000, min_value=-5000),
           st.integers(max_value=1000, min_value=1))
    def test_r_gauss(self, acceleration, desired_result, j, step_size):
        step_size /= 1000
        acceleration /= 1000
        desired_result /= 1000
        j /= 1000
        # set current position to 0 such that the desired result is already the difference
        velocity = w.compile_and_execute(w.velocity_limit_from_position_limit,
                                         [acceleration, desired_result, j, step_size])
        position = j
        i = 0
        start_sign = np.sign(velocity)
        while np.sign(velocity) == start_sign and i < 100000:
            position += velocity * step_size
            velocity -= np.sign(desired_result - j) * acceleration * step_size
            i += 1
        np.testing.assert_almost_equal(position, desired_result)

    @given(sq_matrix())
    def test_sum_row(self, m):
        actual_sum = w.compile_and_execute(w.sum_row, [m])
        expected_sum = np.sum(m, axis=0)
        self.assertTrue(np.all(np.isclose(actual_sum, expected_sum)))

    @given(sq_matrix())
    def test_sum_column(self, m):
        actual_sum = w.compile_and_execute(w.sum_column, [m])
        expected_sum = np.sum(m, axis=1)
        self.assertTrue(np.all(np.isclose(actual_sum, expected_sum)))

    def test_distance_point_to_line_segment1(self):
        p = np.array([0, 0, 0])
        start = np.array([0, 0, 0])
        end = np.array([0, 0, 1])
        distance = w.compile_and_execute(lambda a, b, c: w.distance_point_to_line_segment(a, b, c)[0], [p, start, end])
        nearest = w.compile_and_execute(lambda a, b, c: w.distance_point_to_line_segment(a, b, c)[1], [p, start, end])
        assert distance == 0
        assert nearest[0] == 0
        assert nearest[1] == 0
        assert nearest[2] == 0

    def test_distance_point_to_line_segment2(self):
        p = np.array([0, 1, 0.5])
        start = np.array([0, 0, 0])
        end = np.array([0, 0, 1])
        distance = w.compile_and_execute(lambda a, b, c: w.distance_point_to_line_segment(a, b, c)[0], [p, start, end])
        nearest = w.compile_and_execute(lambda a, b, c: w.distance_point_to_line_segment(a, b, c)[1], [p, start, end])
        assert distance == 1
        assert nearest[0] == 0
        assert nearest[1] == 0
        assert nearest[2] == 0.5

    def test_distance_point_to_line_segment3(self):
        p = np.array([0, 1, 2])
        start = np.array([0, 0, 0])
        end = np.array([0, 0, 1])
        distance = w.compile_and_execute(lambda a, b, c: w.distance_point_to_line_segment(a, b, c)[0], [p, start, end])
        nearest = w.compile_and_execute(lambda a, b, c: w.distance_point_to_line_segment(a, b, c)[1], [p, start, end])
        assert distance == 1.4142135623730951
        assert nearest[0] == 0
        assert nearest[1] == 0
        assert nearest[2] == 1

    def test_to_str(self):
        expr = w.norm(w.quaternion_from_axis_angle(w.Matrix(w.create_symbols(['v1', 'v2', 'v3'])),
                                                   w.Symbol('alpha')))
        assert w.to_str(expr) == 'sqrt((((sq((v1*sin((alpha/2))))' \
                                 '+sq((v2*sin((alpha/2)))))' \
                                 '+sq((v3*sin((alpha/2)))))' \
                                 '+sq(cos((alpha/2)))))'
