from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

import numpy as np
import pytest

from giskardpy.data_types.data_types import JointStates, Derivatives, PrefixName
from giskardpy.god_map import god_map
from giskardpy.motion_statechart.tasks.task import WEIGHT_BELOW_CA
from giskardpy.qp.constraint import EqualityConstraint, InequalityConstraint
from giskardpy.qp.free_variable import FreeVariable
from giskardpy.qp.qp_controller import QPFormulation
from giskardpy.qp.qp_adapter import GiskardToQPAdapter, GiskardToExplicitQPAdapter
from giskardpy.qp.qp_solver_qpSWIFT import QPSolverQPSwift
from giskardpy.qp.qp_solver_qpalm import QPSolverQPalm
from giskardpy.symbol_manager import symbol_manager
import giskardpy.casadi_wrapper as cas


@dataclass
class FakeWorld:
    dofs: List[FreeVariable]
    state: JointStates = field(default_factory=JointStates)


@pytest.fixture
def fake_world():
    upper_limits = {Derivatives.velocity: 1,
                    Derivatives.acceleration: np.inf,
                    Derivatives.jerk: 30}
    lower_limits = {d: -v for d, v in upper_limits.items()}
    quadratic_weights = defaultdict(float, {Derivatives.velocity: 0.01})

    dofs = []
    dofs.append(FreeVariable(PrefixName('a'),
                             lower_limits=lower_limits,
                             upper_limits=upper_limits,
                             quadratic_weights=quadratic_weights))
    dofs.append(FreeVariable(PrefixName('b'),
                             lower_limits=lower_limits,
                             upper_limits=upper_limits,
                             quadratic_weights=quadratic_weights))
    dofs.append(FreeVariable(PrefixName('c'),
                             lower_limits=lower_limits,
                             upper_limits=upper_limits,
                             quadratic_weights=quadratic_weights))

    upper_limits = deepcopy(upper_limits)
    lower_limits = deepcopy(lower_limits)
    upper_limits[Derivatives.position] = 1
    lower_limits[Derivatives.position] = -0.5
    dofs.append(FreeVariable(PrefixName('d'),
                             lower_limits=lower_limits,
                             upper_limits=upper_limits,
                             quadratic_weights=quadratic_weights))

    god_map.world = FakeWorld(dofs)
    return god_map.world


def test_explicit_qp_format(fake_world: FakeWorld):
    prediction_horizon = 10
    eq_constraints = []
    eq_constraints.append(EqualityConstraint('eq1', PrefixName(''),
                                             expression=fake_world.dofs[0].get_symbol(Derivatives.position),
                                             derivative_goal=0.1,
                                             velocity_limit=1,
                                             quadratic_weight=WEIGHT_BELOW_CA))
    eq_constraints.append(EqualityConstraint('eq2', PrefixName(''),
                                             expression=fake_world.dofs[1].get_symbol(Derivatives.position),
                                             derivative_goal=-0.1,
                                             velocity_limit=1,
                                             quadratic_weight=WEIGHT_BELOW_CA))

    eq_constraints.append(EqualityConstraint('eq3', PrefixName(''),
                                             expression=fake_world.dofs[2].get_symbol(Derivatives.position),
                                             derivative_goal=0.1,
                                             velocity_limit=1,
                                             quadratic_weight=0))

    solvers = [QPSolverQPSwift(), QPSolverQPalm()]
    for qp_solver in solvers:
        adapter = qp_solver.required_adapter_type(free_variables=fake_world.dofs,
                                                  equality_constraints=eq_constraints,
                                                  inequality_constraints=[],
                                                  derivative_constraints=[],
                                                  eq_derivative_constraints=[],
                                                  mpc_dt=0.05,
                                                  prediction_horizon=prediction_horizon,
                                                  max_derivative=Derivatives.jerk,
                                                  horizon_weight_gain_scalar=0.1,
                                                  qp_formulation=QPFormulation.explicit_no_acc,
                                                  sparse=True)

        qp_data = adapter.evaluate(symbol_manager)
        assert len(qp_data.quadratic_weights) == ((prediction_horizon - 2) * len(fake_world.dofs)
                                                  + (prediction_horizon) * len(fake_world.dofs) + len(
                    eq_constraints) - 1)
        result = qp_solver.solver_call(qp_data)
        assert np.isclose(result[0], 0.075, atol=1e-3)
        assert np.isclose(result[1], -0.075, atol=1e-3)
        assert result[-2] > 0
        assert result[-1] < 0


def test_explicit_qp_format_neq(fake_world: FakeWorld):
    prediction_horizon = 10
    neq_constraints = []
    neq_constraints.append(InequalityConstraint('neq1', PrefixName(''),
                                                expression=fake_world.dofs[0].get_symbol(Derivatives.position),
                                                lower_error=-0.2,
                                                upper_error=-0.1,
                                                velocity_limit=1,
                                                quadratic_weight=WEIGHT_BELOW_CA))
    neq_constraints.append(InequalityConstraint('neq2', PrefixName(''),
                                                expression=fake_world.dofs[1].get_symbol(Derivatives.position),
                                                lower_error=-np.inf,
                                                upper_error=-0.1,
                                                velocity_limit=1,
                                                quadratic_weight=WEIGHT_BELOW_CA))

    neq_constraints.append(InequalityConstraint('neq3', PrefixName(''),
                                                expression=fake_world.dofs[2].get_symbol(Derivatives.position),
                                                lower_error=0.1,
                                                upper_error=np.inf,
                                                velocity_limit=1,
                                                quadratic_weight=WEIGHT_BELOW_CA))

    neq_constraints.append(InequalityConstraint('neq4', PrefixName(''),
                                                expression=fake_world.dofs[2].get_symbol(Derivatives.position),
                                                lower_error=-0.1,
                                                upper_error=np.inf,
                                                velocity_limit=1,
                                                quadratic_weight=0))

    solvers = [QPSolverQPSwift(), QPSolverQPalm()]
    for qp_solver in solvers:
        adapter = qp_solver.required_adapter_type(free_variables=fake_world.dofs,
                                                  equality_constraints=[],
                                                  inequality_constraints=neq_constraints,
                                                  derivative_constraints=[],
                                                  eq_derivative_constraints=[],
                                                  mpc_dt=0.05,
                                                  prediction_horizon=prediction_horizon,
                                                  max_derivative=Derivatives.jerk,
                                                  horizon_weight_gain_scalar=0.1,
                                                  qp_formulation=QPFormulation.explicit_no_acc,
                                                  sparse=True)

        qp_data = adapter.evaluate(symbol_manager)
        assert len(qp_data.quadratic_weights) == ((prediction_horizon - 2) * len(fake_world.dofs)
                                                  + (prediction_horizon) * len(fake_world.dofs)
                                                  + len(neq_constraints) - 1)
        result = qp_solver.solver_call(qp_data)
        assert np.isclose(result[0], -0.075, atol=1.e-3)
        assert np.isclose(result[1], -0.075, atol=1.e-3)
        assert np.isclose(result[2], 0.075, atol=1.e-3)
        assert result[-3] < 0
        assert result[-2] < 0
        assert result[-1] > 0
