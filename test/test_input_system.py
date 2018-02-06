import unittest
from collections import OrderedDict

import numpy as np

from giskardpy.input_system import Point3Input
from giskardpy.pointy_bot import PointyBot
from giskardpy.joint_space_control import JointSpaceControl
import sympy as sp

from giskardpy.sympy_wrappers import point3

PKG = 'giskardpy'


class TestInputSystem(unittest.TestCase):
    def test_point3(self):
        state = {}
        point_input = Point3Input(prefix='p')
        p = [1,2,3]
        state.update(point_input.get_update_dict(*p))
        self.assertEqual(point_input.get_x(), sp.Symbol('p__x'))
        self.assertEqual(point_input.get_y(), sp.Symbol('p__y'))
        self.assertEqual(point_input.get_z(), sp.Symbol('p__z'))
        self.assertEqual(point_input.get_expression(), point3(sp.Symbol('p__x'),sp.Symbol('p__y'),sp.Symbol('p__z')))
        self.assertEqual(state['p__x'], 1)
        self.assertEqual(state['p__y'], 2)
        self.assertEqual(state['p__z'], 3)


if __name__ == '__main__':
    import rosunit

    rosunit.unitrun(package=PKG,
                    test_name='TestInputSystem',
                    test=TestInputSystem)
