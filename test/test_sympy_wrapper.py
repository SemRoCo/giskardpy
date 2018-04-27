import unittest
from collections import OrderedDict

import numpy as np

import PyKDL
import itertools
from tf.transformations import quaternion_matrix, quaternion_about_axis, quaternion_from_euler, euler_matrix, \
    rotation_matrix, quaternion_multiply, quaternion_conjugate, random_quaternion

from numpy import pi

import giskardpy.symengine_wrappers as spw

PKG = 'giskardpy'


class TestSympyWrapper(unittest.TestCase):
    def test_q_to_m(self):
        q = spw.Matrix([0., 0., 0.14943813, 0.98877108])
        r1 = spw.rotation3_quaternion(*q)
        r1 = np.asarray(r1.tolist()).reshape(r1.shape).astype(float)[:3, :3]

        r_goal = quaternion_matrix(q)[:3, :3]

        r_goal2 = PyKDL.Rotation.Quaternion(*q)
        r_goal2 = np.array([r_goal2[x, y] for x, y in (itertools.product(range(3), repeat=2))]).reshape(3, 3)

        np.testing.assert_array_almost_equal(r1, r_goal)
        np.testing.assert_array_almost_equal(r1, r_goal2)

    def test_2(self):
        angle = .42
        axis = [0, 0, 1]
        r1 = spw.rotation3_axis_angle(axis, angle)
        r1 = np.asarray(r1.tolist()).reshape(r1.shape).astype(float)[:3, :3]

        r_goal = quaternion_matrix(quaternion_about_axis(angle, axis))[:3, :3]

        r_goal2 = PyKDL.Rotation.Rot(PyKDL.Vector(*axis), angle)
        r_goal2 = np.array([r_goal2[x, y] for x, y in (itertools.product(range(3), repeat=2))]).reshape(3, 3)

        np.testing.assert_array_almost_equal(r1, r_goal)
        np.testing.assert_array_almost_equal(r1, r_goal2)

    def test_3(self):
        r, p, y = .2, .7, -.3

        r1 = spw.rotation3_rpy(r, p, y)
        r1 = np.asarray(r1.tolist()).reshape(r1.shape)[:3, :3]

        r_goal = quaternion_matrix(quaternion_from_euler(r, p, y))[:3, :3]

        r_goal2 = PyKDL.Rotation.RPY(r, p, y)
        r_goal2 = np.array([r_goal2[x, y] for x, y in (itertools.product(range(3), repeat=2))]).reshape(3, 3)

        np.testing.assert_array_almost_equal(r1, r_goal)
        np.testing.assert_array_almost_equal(r1, r_goal2)

    def test_matrix_rpy(self):
        rpy = [[0, 0, 1],
               [0, 1, 1],
               [-1, 0, 1],
               [-0.2, 0, 0]]
        for r, p, y in rpy:
            m1 = np.array([x.evalf(real=True) for x in spw.rotation3_rpy(r, p, y)]).astype(float).reshape(4, 4)
            m2 = euler_matrix(r, p, y)
            np.testing.assert_array_almost_equal(m1, m2)

    def test_axis_angle(self):
        tests = [([0, 0, 1], np.pi / 2),
                 ([1, 0, 0], np.pi / 4),
                 ([1, 1, 1], np.pi / 1.2)]
        for axis, angle in tests:
            n_axis = np.array(axis) / np.linalg.norm(axis)
            m = spw.rotation3_axis_angle(n_axis, angle)
            new_axis, new_angle = spw.axis_angle_from_matrix(m)
            self.assertAlmostEqual(new_angle, angle)
            np.testing.assert_array_almost_equal(np.array(new_axis.T).astype(float)[0], n_axis)

    def test_axis_angle2(self):
        tests = [([0, 0, 1], np.pi / 2),
                 ([1, 0, 0], np.pi / 4),
                 ([1, 1, 1], np.pi / 1.2)]
        for axis, angle in tests:
            n_axis = np.array(axis) / np.linalg.norm(axis)
            q = spw.quaterntion_from_axis_angle(n_axis, angle)
            m = spw.rotation3_quaternion(*q)
            new_axis, new_angle = spw.axis_angle_from_matrix(m)
            self.assertAlmostEqual(new_angle, angle)
            np.testing.assert_array_almost_equal(np.array(new_axis.T).astype(float)[0], n_axis)

    def test_quaternion_from_axis_angle(self):
        np.random.seed(23)

        for i in range(50):
            q = random_quaternion()
            spw_axis, spw_angle = spw.axis_angle_from_quaternion(q)
            spw_q = spw.quaterntion_from_axis_angle(spw_axis, spw_angle)
            spw_q = np.array(spw_q).astype(float).T[0]
            np.testing.assert_array_almost_equal(q, spw_q)
            tf_q = quaternion_about_axis(spw_angle, spw_axis)
            np.testing.assert_array_almost_equal(q, tf_q)
            kdl_q = np.array(PyKDL.Rotation.Rot(PyKDL.Vector(*spw_axis), spw_angle).GetQuaternion())
            try:
                np.testing.assert_array_almost_equal(q, kdl_q)
            except AssertionError as e:
                np.testing.assert_array_almost_equal(q, -kdl_q)

    def test_axis_angle3(self):
        rpy = [[0, 0, 1],
               [0, 1, 1],
               [-1, 0, 1],
               [-0.2, 0, 0]]
        for r, p, y in rpy:
            kdl_angle, kdl_axis = PyKDL.Rotation.RPY(r, p, y).GetRotAngle()

            spw_m = spw.rotation3_rpy(r, p, y)
            spw_axis, spw_angle = spw.axis_angle_from_matrix(spw_m)
            spw_axis = np.array([x.evalf(real=True) for x in spw_axis]).astype(float)
            spw_angle = spw_angle.evalf(real=True)
            self.assertAlmostEqual(kdl_angle, spw_angle)
            np.testing.assert_array_almost_equal([x for x in kdl_axis], spw_axis)

    def test_quaterntion_from_rpy1(self):
        rpy = [[0, 0, 1],
               [0, 1, 1],
               [-1, 0, 1],
               [-0.2, 0, 0],
               [0, 0, pi],
               [0.0, 1.57079632679, 0.0]]
        for r, p, y in rpy:
            q1 = np.array(spw.quaternion_from_rpy(r, p, y)).astype(float).T[0]
            q2 = quaternion_from_euler(r, p, y)
            q3 = PyKDL.Rotation.RPY(r, p, y).GetQuaternion()
            np.testing.assert_array_almost_equal(q1, q2)
            np.testing.assert_array_almost_equal(q1, q3)
            np.testing.assert_array_almost_equal(q2, q3)

    def test_quaterntion_from_rpy2(self):
        rpy = [[0, 0, 1],
               [0, 1, 1],
               [-1, 0, 1],
               [-0.2, 0, 0],
               [0, 0, pi]]
        for r, p, y in rpy:
            m1 = spw.rotation3_quaternion(*spw.quaternion_from_rpy(r, p, y))
            m1 = np.array([x.evalf(real=True) for x in m1]).astype(float).reshape(4, 4)
            m2 = spw.rotation3_rpy(r, p, y)
            m2 = np.array([x.evalf(real=True) for x in m2]).astype(float).reshape(4, 4)
            np.testing.assert_array_almost_equal(m1, m2)

    def test_quaternion_conjugate(self):
        np.random.seed(23)

        for i in range(50):
            q = random_quaternion()
            q1_inv = np.array(spw.quaternion_conjugate(q)).astype(float).T[0]
            q1_inv2 = quaternion_conjugate(q)
            np.testing.assert_array_almost_equal(q1_inv, q1_inv2)

    def test_quaternion_diff(self):
        np.random.seed(23)

        for i in range(50):
            q1 = random_quaternion()
            q2 = random_quaternion()
            m1 = spw.rotation3_quaternion(*q1)
            m2 = spw.rotation3_quaternion(*q2)
            m_diff = m1.T*m2
            q_diff = spw.quaternion_diff(q1,q2)
            m_q_diff = spw.rotation3_quaternion(*q_diff)

            m_diff = np.array(m_diff).astype(float)
            m_q_diff = np.array(m_q_diff).astype(float)
            np.testing.assert_array_almost_equal(m_diff, m_q_diff)



if __name__ == '__main__':
    import rosunit

    rosunit.unitrun(package=PKG,
                    test_name='TestSympyWrapper',
                    test=TestSympyWrapper)
