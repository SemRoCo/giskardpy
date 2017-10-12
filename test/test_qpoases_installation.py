#!/usr/bin/env python
import unittest

from giskardpy.dummy import check_qpoaes
import numpy as np

PKG = 'giskardpy'


class TestQPOasesInstallation(unittest.TestCase):
    def test_1(self):
        x_goal = np.array([.7, .8, .9])
        x_start = np.array([.5, 1.05, .35])

        hard_constraints_l = np.array([1.1, 1.3, 1.5])
        hard_constraints_u = np.array([1.2, 1.4, 1.6])

        control_constraints_l = np.array([-.1, -.3, -.5])
        control_constraints_u = np.array([.2, .4, .6])

        w_joints = np.ones(3) * .9
        w_tasks = np.ones(3) * 42
        a = check_qpoaes(x_start, x_goal, control_constraints_l, control_constraints_u,
                         hard_constraints_l, hard_constraints_u, w_joints, w_tasks)[:3]
        print(a)
        self.assertTrue(0. <= a[0], 'dx[0]={} is not positive'.format(a[0]))
        self.assertTrue(a[1] <= 0., 'dx[1]={} is not negative'.format(a[1]))
        self.assertTrue(0. <= a[2], 'dx[2]={} is not positive'.format(a[2]))
        # self.assertTrue(np.all(a <= np.array([0.3, -0.4, 0.7])))


if __name__ == '__main__':
    import rosunit

    rosunit.unitrun(package=PKG,
                    test_name='TestQPOasesInstallation',
                    test=TestQPOasesInstallation)
