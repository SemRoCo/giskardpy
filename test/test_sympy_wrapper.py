import unittest
from collections import OrderedDict

import numpy as np

import PyKDL
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
    def test_1(self):
        q = spw.Matrix([0., 0., 0.14943813, 0.98877108])
        r1 = spw.rotation3_quaternion(*q)
        r1 = np.asarray(r1.tolist()).reshape(r1.shape)

        r_goal = quaternion_matrix(q)
        np.testing.assert_array_almost_equal(r1, r_goal)

    def test_2(self):
        angle = .42
        axis = [0, 0, 1]
        r1 = spw.rotation3_axis_angle(axis, angle)
        r1 = np.asarray(r1.tolist()).reshape(r1.shape)

        r_goal = quaternion_matrix(quaternion_about_axis(angle, axis))
        np.testing.assert_array_almost_equal(r1, r_goal)

    def test_3(self):
        r, p, y = .2, .7, -.3

        r1 = spw.rotation3_rpy(r, p, y)
        r1 = np.asarray(r1.tolist()).reshape(r1.shape)[:3,:3]

        r_goal = quaternion_matrix(quaternion_from_euler(r, p, y, axes='rxyz')).T[:3,:3]

        np.testing.assert_array_almost_equal(r1, r_goal)


if __name__ == '__main__':
    import rosunit

    rosunit.unitrun(package=PKG,
                    test_name='TestSympyWrapper',
                    test=TestSympyWrapper)
