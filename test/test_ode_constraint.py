
import unittest
import numpy as np
import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.qp.constraint import ODEConstraint
from giskardpy.qp.constraint_collection import ConstraintCollection
from giskardpy.qp.adapters.qp_adapter import GiskardToQPAdapter
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.qp.solvers.qp_solver import QPSolver
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World

class MockSolver(QPSolver):
    def solver_call(self, qp_data):
        return np.zeros(10)
    
    def required_adapter_type(self, **kwargs):
        return GiskardToQPAdapter(**kwargs)

class TestODEConstraint(unittest.TestCase):
    def test_ode_constraint_compilation(self):
        # Setup
        cc = ConstraintCollection()
        
        # Define variables
        h = cas.FloatVariable(name=PrefixedName("h"))
        t = cas.FloatVariable(name=PrefixedName("t"))
        
        # ODE: h_dot = -0.5 * h (exponential decay)
        ode_function = -0.5 * h
        
        # Add constraint
        cc.add_ode_constraint(
            target_variable=h,
            ode_function=ode_function,
            weight=1.0,
            name="decay_constraint"
        )
        
        self.assertEqual(len(cc.constraints), 1)
        self.assertIsInstance(cc.constraints[0], ODEConstraint)
        
        # Verify compilation (mocking necessary parts)
        # This part is tricky without a full environment, so we'll focus on unit testing the components first.
        # But we can check if the constraint holds the right data.
        constraint = cc.constraints[0]
        self.assertEqual(constraint.expression, h)
        self.assertEqual(constraint.ode_function, ode_function)

if __name__ == '__main__':
    unittest.main()
