import unittest
import numpy as np
import giskardpy.utils.math as giskard_math
from giskardpy.my_types import Derivatives


class TestMath(unittest.TestCase):
    def test_velocity_integral_jerk(self):
        limits = {
            Derivatives.velocity: 1,
            Derivatives.acceleration: np.inf,
            Derivatives.jerk: 30
        }
        actual = giskard_math.mpc_velocity_integral(limits, 0.05, 9)
        self.assertAlmostEqual(actual, 0.185)

    def test_velocity_integral_acc(self):
        limits = {
            Derivatives.velocity: 0.5,
            Derivatives.acceleration: 1.5,
        }
        actual = giskard_math.mpc_velocity_integral(limits, 0.05, 9)
        self.assertAlmostEqual(actual, 0.095)

    def test_velocity_integral_acc2(self):
        limits = {
            Derivatives.velocity: 0.6,
            Derivatives.acceleration: 1.5,
        }
        actual = giskard_math.mpc_velocity_integral(limits, 0.05, 9)
        self.assertAlmostEqual(actual, 0.135)