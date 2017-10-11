#!/usr/bin/env python
import unittest

from giskardpy.dummy import check_sympy_installation

PKG = 'giskardpy'


class TestSympyInstallation(unittest.TestCase):
    def test_1(self):
        a = check_sympy_installation(2, 2)
        self.assertEqual(a, 4, '{}!=4'.format(a))


if __name__ == '__main__':
    import rosunit

    rosunit.unitrun(package=PKG,
                    test_name='TestSympyInstallation',
                    test=TestSympyInstallation)
