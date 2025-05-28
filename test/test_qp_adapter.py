from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from giskardpy.data_types.data_types import JointStates, Derivatives, PrefixName
from giskardpy.god_map import god_map
from giskardpy.motion_statechart.tasks.task import WEIGHT_BELOW_CA
from giskardpy.qp.constraint import EqualityConstraint
from giskardpy.qp.free_variable import FreeVariable
from giskardpy.qp.qp_controller import QPFormulation
from giskardpy.qp.qp_formatter import GiskardToQPAdapter
from giskardpy.symbol_manager import symbol_manager
import giskardpy.casadi_wrapper as cas

@dataclass
class FakeWorld:
    state: JointStates = field(default_factory=JointStates)


def test_joint_goal():
    god_map.world = FakeWorld()

    dof = FreeVariable(PrefixName('a'),
                       lower_limits={Derivatives.velocity: -1,
                                     Derivatives.acceleration: -np.inf,
                                     Derivatives.jerk: -30},
                       upper_limits={Derivatives.velocity: 1,
                                     Derivatives.acceleration: np.inf,
                                     Derivatives.jerk: 30},
                       quadratic_weights=defaultdict(float, {Derivatives.velocity: 0.01},))
    god_map.world.state[dof.name].position = 1

    eq_constraint = EqualityConstraint('eq1', PrefixName(''),
                                      expression=dof.get_symbol(Derivatives.position),
                                      derivative_goal=0.1,
                                      velocity_limit=1,
                                      quadratic_weight=WEIGHT_BELOW_CA)

    adapter = GiskardToQPAdapter(free_variables=[dof],
                                 equality_constraints=[eq_constraint],
                                 inequality_constraints=[],
                                 derivative_constraints=[],
                                 eq_derivative_constraints=[],
                                 mpc_dt=0.05,
                                 prediction_horizon=10,
                                 max_derivative=Derivatives.jerk,
                                 horizon_weight_gain_scalar=0.1,
                                 qp_formulation=QPFormulation.explicit_no_acc,
                                 sparse=True)

    adapter.compile_format()
    data = symbol_manager.resolve_symbols(adapter.free_symbols_str)
    adapter.evaluate(data)
    adapter.filter()
    adapter.apply_filters()
    adapter.problem_data_to_qp_format()
    adapter.pretty_print_problem()

