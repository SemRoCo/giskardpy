import unittest
from collections import OrderedDict

import numpy as np

import PyKDL
import itertools
from tf.transformations import quaternion_matrix, quaternion_about_axis, quaternion_from_euler

from giskardpy import USE_SYMENGINE
from giskardpy.pointy_bot import PointyBot
from giskardpy.joint_space_control import JointSpaceControl

if USE_SYMENGINE:
    import giskardpy.symengine_wrappers as spw
else:
    import giskardpy.sympy_wrappers as spw

PKG = 'giskardpy'


class TestSympyWrapper(unittest.TestCase):
    def test_q_to_m(self):
        q = spw.Matrix([0., 0., 0.14943813, 0.98877108])
        r1 = spw.rotation3_quaternion(*q)
        r1 = np.asarray(r1.tolist()).reshape(r1.shape).astype(float)[:3,:3]

        r_goal = quaternion_matrix(q)[:3,:3]

        r_goal2 = PyKDL.Rotation.Quaternion(*q)
        r_goal2 = np.array([r_goal2[x, y] for x, y in (itertools.product(range(3), repeat=2))]).reshape(3, 3)

        np.testing.assert_array_almost_equal(r1, r_goal)
        np.testing.assert_array_almost_equal(r1, r_goal2)

    def test_2(self):
        angle = .42
        axis = [0, 0, 1]
        r1 = spw.rotation3_axis_angle(axis, angle)
        r1 = np.asarray(r1.tolist()).reshape(r1.shape).astype(float)[:3,:3]

        r_goal = quaternion_matrix(quaternion_about_axis(angle, axis))[:3,:3]

        r_goal2 = PyKDL.Rotation.Rot(PyKDL.Vector(*axis), angle)
        r_goal2 = np.array([r_goal2[x, y] for x, y in (itertools.product(range(3), repeat=2))]).reshape(3, 3)

        np.testing.assert_array_almost_equal(r1, r_goal)
        np.testing.assert_array_almost_equal(r1, r_goal2)

    def test_3(self):
        r, p, y = .2, .7, -.3

        r1 = spw.rotation3_rpy(r, p, y)
        r1 = np.asarray(r1.tolist()).reshape(r1.shape)[:3, :3]

        r_goal = quaternion_matrix(quaternion_from_euler(r, p, y))[:3, :3]

        r_goal2 = PyKDL.Rotation.RPY(r,p,y)
        r_goal2 = np.array([r_goal2[x,y] for x,y in (itertools.product(range(3), repeat=2))]).reshape(3,3)

        np.testing.assert_array_almost_equal(r1, r_goal)
        np.testing.assert_array_almost_equal(r1, r_goal2)


if __name__ == '__main__':
    import rosunit

    rosunit.unitrun(package=PKG,
                    test_name='TestSympyWrapper',
                    test=TestSympyWrapper)
