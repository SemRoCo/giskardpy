import unittest

from angles import normalize_angle
from hypothesis import given, reproduce_failure, assume
import hypothesis.strategies as st

import numpy as np

import PyKDL

from tf.transformations import quaternion_matrix, quaternion_about_axis, quaternion_from_euler, euler_matrix, \
    rotation_matrix, quaternion_multiply, quaternion_conjugate, random_quaternion, quaternion_from_matrix, \
    quaternion_slerp, rotation_from_matrix, euler_from_matrix

from numpy import pi

from transforms3d.axangles import mat2axangle
from transforms3d.euler import euler2mat, axangle2euler, euler2axangle

import giskardpy.symengine_wrappers as spw
from giskardpy import BACKEND
from giskardpy.test_utils import limited_float, SMALL_NUMBER, unit_vector, quaternion, vector, \
    pykdl_frame_to_numpy, lists_of_same_length, angle, compare_axis_angle

PKG = 'giskardpy'


class TestSympyWrapper(unittest.TestCase):

    # fails if numbers too small or big
    @given(limited_float(outer_limit=1e10))
    def test_abs(self, f1):
        self.assertAlmostEqual(spw.diffable_abs(f1), abs(f1), places=7)

    # fails if numbers too small or big
    @given(limited_float(min_dist_to_zero=SMALL_NUMBER))
    def test_heaviside(self, f1):
        r1 = float(spw.diffable_heaviside(f1))
        r2 = 0 if f1 < 0 else 1
        self.assertTrue(np.isclose(r1, r2), msg='0 if {} < 0 else 1 => {} != {}'.format(f1, r1, r2))

    # fails if numbers too small or big
    @given(limited_float(outer_limit=1e7),
           limited_float(outer_limit=1e7))
    def test_max(self, f1, f2):
        r1 = np.float(spw.diffable_max_fast(f1, f2))
        self.assertTrue(np.isclose(r1, max(f1, f2)), msg='max({},{})={}'.format(f1, f2, r1))

    # fails if numbers too small or big
    @given(limited_float(),
           limited_float())
    def test_max(self, f1, f2):
        r1 = np.float(spw.diffable_max(f1, f2))
        self.assertTrue(np.isclose(r1, max(f1, f2)), msg='max({},{})={}'.format(f1, f2, r1))

    # fails if numbers too big
    @given(limited_float(outer_limit=1e7),
           limited_float(outer_limit=1e7))
    def test_min(self, f1, f2):
        r1 = np.float(spw.diffable_min_fast(f1, f2))
        self.assertTrue(np.isclose(r1, min(f1, f2)), msg='min({},{})={}'.format(f1, f2, r1))

    # fails if numbers too big
    @given(limited_float(),
           limited_float())
    def test_min(self, f1, f2):
        r1 = np.float(spw.diffable_min(f1, f2))
        self.assertTrue(np.isclose(r1, min(f1, f2)), msg='min({},{})={}'.format(f1, f2, r1))

    # fails if numbers too big
    @given(limited_float(min_dist_to_zero=SMALL_NUMBER))
    def test_sign(self, f1):
        r1 = np.float(spw.diffable_sign(f1))
        r2 = np.sign(f1)
        # r2 = 0.5 if f1 == 0 else r2
        self.assertTrue(np.isclose(r1, r2), msg='spw.sign({})={} != np.sign({})={}'.format(f1, r1, f1, r2))

    # fails if condition is to close too 0 or too big or too small
    @given(limited_float(min_dist_to_zero=SMALL_NUMBER),
           limited_float(),
           limited_float())
    def test_if_greater_zero(self, condition, if_result, else_result):
        r1 = np.float(spw.if_greater_zero(condition, if_result, else_result))
        r2 = np.float(if_result if condition > 0 else else_result)
        self.assertTrue(np.isclose(r1, r2), msg='{} if {} > 0 else {} => {}'.format(if_result, condition, else_result,
                                                                                    r1))

    # fails if condition is to close too 0 or too big or too small
    @given(limited_float(min_dist_to_zero=SMALL_NUMBER),
           limited_float(),
           limited_float())
    def test_if_greater_eq_zero(self, condition, if_result, else_result):
        r1 = np.float(spw.if_greater_eq_zero(condition, if_result, else_result))
        r2 = np.float(if_result if condition >= 0 else else_result)
        self.assertTrue(np.isclose(r1, r2), msg='{} if {} >= 0 else {} => {}'.format(if_result, condition, else_result,
                                                                                     r1))

    # fails if condition is to close too 0 or too big or too small
    # fails if if_result is too big or too small
    @given(limited_float(min_dist_to_zero=SMALL_NUMBER),
           limited_float(outer_limit=1e8),
           limited_float())
    def test_if_eq_zero(self, condition, if_result, else_result):
        r1 = np.float(spw.if_eq_zero(condition, if_result, else_result))
        r2 = np.float(if_result if condition == 0 else else_result)
        self.assertTrue(np.isclose(r1, r2, atol=1.e-7), msg='{} if {} == 0 else {} => {}'.format(if_result, condition,
                                                                                                 else_result,
                                                                                                 r1))

    # TODO test save compiled function
    # TODO test load compiled function
    # TODO test compiled function class

    # fails if numbers too small or big
    @given(limited_float(outer_limit=1e7),
           limited_float(outer_limit=1e7))
    def test_speed_up_max(self, f1, f2):
        f1_s = spw.Symbol('f1')
        f2_s = spw.Symbol('f2')
        expr = spw.diffable_max_fast(f1_s, f2_s)
        llvm = spw.speed_up(spw.Matrix([expr]), expr.free_symbols)
        kwargs = {'f1': f1, 'f2': f2}
        r1_llvm = llvm(**kwargs)[0]
        # r1_expr = np.float(expr.subs())
        r1 = np.float(spw.diffable_max_fast(f1, f2))
        self.assertTrue(np.isclose(r1, r1_llvm), msg='max({},{})={}, max_expr({},{})={}'.format(f1, f2, r1,
                                                                                                f1, f2, r1_llvm))

    @given(limited_float(),
           limited_float(),
           limited_float())
    def test_speed_up_if_greater_zero(self, condition, if_result, else_result):
        condition_s = spw.Symbol('condition')
        if_s = spw.Symbol('if')
        else_s = spw.Symbol('else')
        expr = spw.if_greater_zero(condition_s, if_s, else_s)
        llvm = spw.speed_up(spw.Matrix([expr]), expr.free_symbols)
        kwargs = {'condition': condition,
                  'if': if_result,
                  'else': else_result}

        # r1_expr = float(expr.subs(kwargs))
        r1_llvm = llvm(**kwargs)[0][0]
        r1 = float(spw.if_greater_zero(condition, if_result, else_result))

        self.assertTrue(np.isclose(r1, r1_llvm), msg='{} if {} > 0 else {} => {} != {}'.format(if_result, condition,
                                                                                               else_result,
                                                                                               r1_llvm, r1))

    # fails if numbers too big or too small
    @given(unit_vector(length=3),
           angle())
    def test_speed_up_matrix_from_axis_angle(self, axis, angle):
        axis_s = spw.var('x y z')
        angle_s = spw.Symbol('angle')
        kwargs = {'x': axis[0],
                  'y': axis[1],
                  'z': axis[2],
                  'angle': angle}

        expr = spw.rotation_matrix_from_axis_angle(spw.Matrix(axis_s), angle_s)
        llvm = spw.speed_up(spw.Matrix([expr]), expr.free_symbols)
        r1_llvm = llvm(**kwargs)

        r1 = np.array(spw.rotation_matrix_from_axis_angle(axis, angle)).astype(float)
        self.assertTrue(np.isclose(r1, r1_llvm).all(), msg='{} {}\n{} != \n{}'.format(axis, angle, r1, r1_llvm))

    # fails if numbers too big
    @given(limited_float(min_dist_to_zero=SMALL_NUMBER))
    def test_speed_sign(self, f1):
        f1_s = spw.Symbol('f1')
        expr = spw.diffable_sign(f1_s)
        llvm = spw.speed_up(spw.Matrix([expr]), list(expr.free_symbols))
        kwargs = {'f1': f1}

        r1_llvm = llvm(**kwargs)
        r1 = float(spw.diffable_sign(f1))
        self.assertTrue(np.isclose(r1, r1_llvm), msg='spw.sign({})={} != np.sign({})={}'.format(f1, r1, f1, r1_llvm))

    @given(quaternion(),
           quaternion(),
           st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1))
    def test_speed_up_slerp(self, q1, q2, t):
        q1_s = spw.var('q1x q1y q1z q1w')
        q2_s = spw.var('q2x q2y q2z q2w')
        t_s = spw.Symbol('t')
        q1_expr = spw.Matrix(q1_s)
        q2_expr = spw.Matrix(q2_s)
        expr = spw.diffable_slerp(q1_expr, q2_expr, t_s)

        slerp = spw.speed_up(expr, expr.free_symbols)
        kwargs = {'q1x': q1[0], 'q1y': q1[1], 'q1z': q1[2], 'q1w': q1[3],
                  'q2x': q2[0], 'q2y': q2[1], 'q2z': q2[2], 'q2w': q2[3],
                  't': t}
        r1 = slerp(**kwargs).T[0]
        r2 = np.array([float(x.evalf(real=True)) for x in spw.diffable_slerp(spw.Matrix(q1), spw.Matrix(q2), t)])
        self.assertTrue(np.isclose(r1, r2, atol=1e-3).all() or
                        np.isclose(r1, -r2, atol=1e-3).all(),
                        msg='q1={} q2={} t={}\n{} != {}'.format(q1, q2, t, r1, r2))

    # fails if numbers too big
    @given(vector(3),
           vector(3))
    def test_cross(self, u, v):
        r1 = np.array(spw.cross(u, v)).astype(float).T[0]
        r2 = np.cross(u, v)
        self.assertTrue(np.isclose(r1, r2).all(), msg='{}x{}=\n{} != {}'.format(u, v, r1, r2))

    @given(vector(3))
    def test_vector3(self, v):
        r1 = spw.vector3(*v)
        self.assertEqual(r1[0], v[0])
        self.assertEqual(r1[1], v[1])
        self.assertEqual(r1[2], v[2])
        self.assertEqual(r1[3], 0)

    @given(vector(3))
    def test_point3(self, v):
        r1 = spw.point3(*v)
        self.assertEqual(r1[0], v[0])
        self.assertEqual(r1[1], v[1])
        self.assertEqual(r1[2], v[2])
        self.assertEqual(r1[3], 1)

    # fails if numbers too big
    @given(st.lists(limited_float()))
    def test_norm(self, v):
        r1 = np.float(spw.norm(v))
        r2 = np.linalg.norm(v)
        self.assertTrue(np.isclose(r1, r2), msg='|{}|2=\n{} != {}'.format(v, r1, r2))

    # fails if numbers too big
    @given(st.lists(limited_float()),
           limited_float())
    def test_scale(self, v, a):
        assume(np.linalg.norm(v) != 0)
        r1 = np.array(spw.scale(spw.Matrix(v), a)).astype(float).T[0]
        r2 = v / np.linalg.norm(v) * a
        self.assertTrue(np.isclose(r1, r2).all(), msg='v={} a={}\n{} != {}'.format(v, a, r1, r2))

    # fails if numbers too big
    @given(lists_of_same_length([limited_float(), limited_float()], max_length=50))
    def test_dot(self, vectors):
        u, v = vectors
        r1 = np.float(spw.dot(spw.Matrix(u), spw.Matrix(v)))
        r2 = np.dot(u, v)
        self.assertTrue(np.isclose(r1, r2), msg='{} * {}={} != {}'.format(u, v, r1, r2))

    # fails if numbers too big
    @given(limited_float(),
           limited_float(),
           limited_float())
    def test_translation3(self, x, y, z):
        r1 = np.array(spw.translation3(x, y, z)).astype(float)
        r2 = np.identity(4)
        r2[0, 3] = x
        r2[1, 3] = y
        r2[2, 3] = z
        self.assertTrue(np.isclose(r1, r2).all(), msg='{} != {}'.format(r1, r2))

    # fails if numbers too big
    @given(angle(),
           angle(),
           angle())
    def test_rotation3_rpy(self, roll, pitch, yaw):
        r1 = np.array(spw.rotation_matrix_from_rpy(roll, pitch, yaw)).astype(float)
        r2 = euler_matrix(roll, pitch, yaw)
        self.assertTrue(np.isclose(r1, r2).all(), msg='{} != {}'.format(r1, r2))

    # fails if numbers too big or too small
    @given(unit_vector(length=3),
           angle())
    def test_rotation3_axis_angle(self, axis, angle):
        r1 = np.array(spw.rotation_matrix_from_axis_angle(axis, angle)).astype(float)
        r2 = rotation_matrix(angle, np.array(axis))
        self.assertTrue(np.isclose(r1, r2).all(), msg='{} {}\n{} != \n{}'.format(axis, angle, r1, r2))

    # fails if numbers too big or too small
    @given(unit_vector(length=4))
    def test_rotation3_quaternion(self, q):
        r1 = np.array(spw.rotation_matrix_from_quaternion(*q)).astype(float)
        r2 = quaternion_matrix(q)
        self.assertTrue(np.isclose(r1, r2).all(), msg='{} \n{} != \n{}'.format(q, r1, r2))

    # fails if numbers too big or too small
    @given(limited_float(),
           limited_float(),
           limited_float(),
           unit_vector(length=3),
           angle())
    def test_frame3_axis_angle(self, x, y, z, axis, angle):
        r1 = np.array(spw.frame_axis_angle(x, y, z, axis, angle)).astype(float)
        r2 = rotation_matrix(angle, np.array(axis))
        r2[0, 3] = x
        r2[1, 3] = y
        r2[2, 3] = z
        self.assertTrue(np.isclose(r1, r2).all(), msg='\n{} != \n{}'.format(r1, r2))

    # fails if numbers too big or too small
    @given(limited_float(),
           limited_float(),
           limited_float(),
           angle(),
           angle(),
           angle())
    def test_frame3_rpy(self, x, y, z, roll, pitch, yaw):
        r1 = np.array(spw.frame_rpy(x, y, z, roll, pitch, yaw)).astype(float)
        r2 = euler_matrix(roll, pitch, yaw)
        r2[0, 3] = x
        r2[1, 3] = y
        r2[2, 3] = z
        self.assertTrue(np.isclose(r1, r2).all(), msg='\n{} != \n{}'.format(r1, r2))

    # fails if numbers too big or too small
    @given(limited_float(),
           limited_float(),
           limited_float(),
           unit_vector(4))
    def test_frame3_quaternion(self, x, y, z, q):
        r1 = np.array(spw.frame_quaternion(x, y, z, q[0], q[1], q[2], q[3])).astype(float)
        r2 = quaternion_matrix(q)
        r2[0, 3] = x
        r2[1, 3] = y
        r2[2, 3] = z
        self.assertTrue(np.isclose(r1, r2).all(), msg='\n{} != \n{}'.format(r1, r2))

    # fails if numbers too big or too small
    @given(limited_float(),
           limited_float(),
           limited_float(),
           unit_vector(4))
    def test_inverse_frame(self, x, y, z, q):
        r1 = np.array(spw.inverse_frame(spw.frame_quaternion(x, y, z, q[0], q[1], q[2], q[3]))).astype(float)
        r2 = PyKDL.Frame()
        r2.M = PyKDL.Rotation.Quaternion(q[0], q[1], q[2], q[3])
        r2.p[0] = x
        r2.p[1] = y
        r2.p[2] = z
        r2 = r2.Inverse()
        r2 = pykdl_frame_to_numpy(r2)
        self.assertTrue(np.isclose(r1, r2).all(), msg='\n{} != \n{}'.format(r1, r2))

    # fails if numbers too big or too small
    @given(limited_float(),
           limited_float(),
           limited_float(),
           unit_vector(4))
    def test_pos_of(self, x, y, z, q):
        r1 = np.array(spw.position_of(spw.frame_quaternion(x, y, z, q[0], q[1], q[2], q[3]))).astype(float).T[0]
        r2 = np.array([x, y, z, 1])
        self.assertTrue(np.isclose(r1, r2).all(), msg='\n{} != \n{}'.format(r1, r2))

    # fails if numbers too big or too small
    @given(limited_float(),
           limited_float(),
           limited_float(),
           unit_vector(4))
    def test_trans_of(self, x, y, z, q):
        r1 = np.array(spw.translation_of(spw.frame_quaternion(x, y, z, q[0], q[1], q[2], q[3]))).astype(float)
        r2 = np.identity(4)
        r2[0, 3] = x
        r2[1, 3] = y
        r2[2, 3] = z
        self.assertTrue(np.isclose(r1, r2).all(), msg='\n{} != \n{}'.format(r1, r2))

    # fails if numbers too big or too small
    @given(limited_float(),
           limited_float(),
           limited_float(),
           unit_vector(4))
    def test_rot_of(self, x, y, z, q):
        r1 = np.array(spw.rotation_of(spw.frame_quaternion(x, y, z, q[0], q[1], q[2], q[3]))).astype(float)
        r2 = quaternion_matrix(q)
        self.assertTrue(np.isclose(r1, r2).all(), msg='\n{} != \n{}'.format(r1, r2))

    # fails if numbers too big or too small
    @given(unit_vector(4))
    def test_trace(self, q):
        r1 = np.array(spw.trace(spw.rotation_matrix_from_quaternion(q[0], q[1], q[2], q[3]))).astype(float)
        r2 = quaternion_matrix(q)
        r2 = np.trace(r2)
        self.assertTrue(np.isclose(r1, r2).all(), msg='\n{} != \n{}'.format(r1, r2))

    # fails if numbers too big or too small
    @given(unit_vector(4))
    def test_trace(self, q):
        r1 = np.array(spw.trace(spw.rotation_matrix_from_quaternion(q[0], q[1], q[2], q[3]))).astype(float)
        r2 = quaternion_matrix(q)
        r2 = np.trace(r2)
        self.assertTrue(np.isclose(r1, r2).all(), msg='\n{} != \n{}'.format(r1, r2))

    # TODO test rotation_dist

    # fails if numbers too big or too small
    # TODO nan if angle 0
    # TODO use 'if' to make angle always positive?
    @given(unit_vector(length=3),
           angle())
    def test_axis_angle_from_matrix(self, axis, angle):
        assume(angle != 0)
        axis2, angle2 = spw.axis_angle_from_matrix(spw.rotation_matrix_from_axis_angle(axis, angle))
        angle2 = float(angle2)
        axis2 = np.array(axis2).astype(float).T[0]
        if angle < 0:
            angle = -angle
            axis = [-x for x in axis]
        if angle2 < 0:
            angle2 = -angle2
            axis2 *= -1
        self.assertTrue(np.isclose(angle, angle2), msg='{} != {}'.format(angle, angle2))
        self.assertTrue(np.isclose(axis, axis2).all(), msg='{} != {}'.format(axis, axis2))

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
        r1 = np.array(spw.quaternion_from_axis_angle(axis, angle)).astype(float).T[0]
        r2 = quaternion_about_axis(angle, axis)
        self.assertTrue(np.isclose(r1, r2).all(), msg='{} != {}'.format(r1, r2))

    # fails if numbers too big or too small
    @given(angle(),
           angle(),
           angle())
    def test_axis_angle_from_rpy(self, roll, pitch, yaw):
        axis2, angle2 = euler2axangle(roll, pitch, yaw)
        assume(abs(angle2) > SMALL_NUMBER)
        axis, angle = spw.axis_angle_from_rpy(roll, pitch, yaw)
        angle = float(angle)
        axis = np.array(axis).astype(float).T[0]
        if angle < 0:
            angle = -angle
            axis = [-x for x in axis]
        if angle2 < 0:
            angle2 = -angle2
            axis2 *= -1
        compare_axis_angle(angle, axis, angle2, axis2)
        # self.assertTrue(np.isclose(angle, angle2), msg='{} != {}'.format(angle, angle2))
        # self.assertTrue(np.isclose(axis, axis2).all(), msg='{} != {}'.format(axis, axis2))

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
        roll, pitch, yaw = spw.rpy_from_matrix(matrix)
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
        r1 = np.array(spw.rotation_matrix_from_rpy(roll, pitch, yaw)).astype(float)
        self.assertTrue(np.isclose(r1, matrix).all(), msg='{} != {}'.format(r1, matrix))

    # fails if numbers too big or too small
    @given(angle(),
           angle(),
           angle())
    def test_quaternion_from_rpy(self, roll, pitch, yaw):
        q = np.array(spw.quaternion_from_rpy(roll, pitch, yaw)).astype(float).T
        q2 = quaternion_from_euler(roll, pitch, yaw)
        self.assertTrue(np.isclose(q, q2).all(), msg='{} != {}'.format(q, q2))

    # fails if numbers too big or too small
    @given(quaternion())
    def test_quaternion_from_matrix(self, q):
        matrix = quaternion_matrix(q)
        q2 = quaternion_from_matrix(matrix)
        q1 = np.array(spw.quaternion_from_matrix(matrix.tolist())).astype(float).T
        self.assertTrue(np.isclose(q1, q2).all() or np.isclose(q1, -q2).all(), msg='{} != {}'.format(q, q2))

    # fails if numbers too big or too small
    @given(quaternion(),
           quaternion())
    def test_quaternion_multiply(self, q, p):
        r1 = np.array(spw.quaternion_multiply(q, p)).astype(float).T
        r2 = quaternion_multiply(q, p)
        self.assertTrue(np.isclose(r1, r2).all() or np.isclose(r1, -r2).all(), msg='{} != {}'.format(r1, r2))

    # fails if numbers too big or too small
    @given(quaternion())
    def test_quaternion_conjugate(self, q):
        r1 = np.array(spw.quaternion_conjugate(q)).astype(float).T
        r2 = quaternion_conjugate(q)
        self.assertTrue(np.isclose(r1, r2).all() or np.isclose(r1, -r2).all(), msg='{} != {}'.format(r1, r2))

    # fails if numbers too big or too small
    @given(quaternion(),
           quaternion())
    def test_quaternion_diff(self, q1, q2):
        q3 = spw.quaternion_diff(q1, q2)
        q4 = np.array(spw.quaternion_multiply(q1, q3)).astype(float).T
        self.assertTrue(np.isclose(q2, q4).all() or np.isclose(q2, -q4).all(), msg='{} != {}'.format(q1, q4))

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
    def test_slerp2(self, q1, q2, t):
        r1 = np.array([float(x.evalf(real=True)) for x in spw.diffable_slerp(spw.Matrix(q1), spw.Matrix(q2), t)])
        r2 = quaternion_slerp(q1, q2, t)
        self.assertTrue(np.isclose(r1, r2, atol=1e-3).all() or
                        np.isclose(r1, -r2, atol=1e-3).all(),
                        msg='q1={} q2={} t={}\n{} != {}'.format(q1, q2, t, r1, r2))


if __name__ == '__main__':
    import rosunit

    rosunit.unitrun(package=PKG,
                    test_name='TestSympyWrapper',
                    test=TestSympyWrapper)
