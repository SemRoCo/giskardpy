import unittest
import numpy as np
import giskardpy.utils.math as giskard_math
from giskardpy.data_types import Derivatives


class TestMath(unittest.TestCase):
    def test_velocity_integral_jerk(self):
        limits = {
            Derivatives.velocity: 1,
            Derivatives.acceleration: np.inf,
            Derivatives.jerk: 21.1
        }
        actual = giskard_math.mpc_velocity_integral(limits, 0.05, 9)
        expected = giskard_math.mpc_velocity_integral3(limits, 0.05, 9)
        self.assertAlmostEqual(actual, expected)

    def test_velocity_integral_jerk2(self):
        limits = {
            Derivatives.velocity: 0.8,
            Derivatives.acceleration: np.inf,
            Derivatives.jerk: 30
        }
        actual = giskard_math.mpc_velocity_integral(limits, 0.05, 9)
        expected = giskard_math.mpc_velocity_integral2(limits, 0.05, 9)
        self.assertAlmostEqual(actual, expected)

    def test_velocity_integral_jerk3(self):
        limits = {
            Derivatives.velocity: 0.8,
            Derivatives.acceleration: np.inf,
            Derivatives.jerk: 60
        }
        actual = giskard_math.mpc_velocity_integral(limits, 0.05, 9)
        expected = giskard_math.mpc_velocity_integral2(limits, 0.05, 9)
        self.assertAlmostEqual(actual, expected)
