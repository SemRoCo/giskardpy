import unittest
from collections import OrderedDict

import numpy as np
from tf.transformations import quaternion_matrix, quaternion_about_axis, quaternion_from_euler

from giskardpy.pointy_bot import PointyBot
from giskardpy.joint_space_control import JointSpaceControl
from giskardpy.sympy_wrappers import rotation3_quaternion, rotation3_axis_angle, rotation3_rpy
import sympy as sp

PKG = 'giskardpy'


class TestSympyWrapper(unittest.TestCase):
    def test_1(self):
        q = sp.Matrix([0., 0., 0.14943813, 0.98877108])
        r1 = rotation3_quaternion(*q)

        r_goal = quaternion_matrix(q).T
        np.testing.assert_array_almost_equal(r1, r_goal)

    def test_2(self):
        angle = .42
        axis = [0, 0, 1]
        r1 = rotation3_axis_angle(axis, angle)

        r_goal = quaternion_matrix(quaternion_about_axis(angle, axis)).T
        np.testing.assert_array_almost_equal(r1, r_goal)

    def test_3(self):
        # r, p, y = 0,np.pi/2,np.pi/2
        r, p, y = .2, .7, -.3

        r1 = rotation3_rpy(r,p,y)
        r_goal = quaternion_matrix(quaternion_from_euler(r,p,y, axes='rxyz')).T
        np.testing.assert_array_almost_equal(r1, r_goal)


if __name__ == '__main__':
    import rosunit

    rosunit.unitrun(package=PKG,
                    test_name='TestSympyWrapper',
                    test=TestSympyWrapper)
