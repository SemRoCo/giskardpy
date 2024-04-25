from itertools import chain
from typing import Dict, List

from py_trees import Status

import giskardpy.casadi_wrapper as w
from giskardpy.data_types.exceptions import EmptyProblemException
from giskardpy.god_map import god_map
from giskardpy.qp.constraint import EqualityConstraint, InequalityConstraint, DerivativeInequalityConstraint
from giskardpy.qp.qp_controller import QPProblemBuilder
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time
from giskardpy.tree.blackboard_utils import catch_and_raise_to_blackboard


class InitQPController(GiskardBehavior):
    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        eq_constraints, neq_constraints, derivative_constraints, quadratic_weight_gains, linear_weight_gains = god_map.motion_goal_manager.get_constraints_from_goals()
        free_variables = self.get_active_free_symbols(eq_constraints, neq_constraints, derivative_constraints)

        qp_controller = QPProblemBuilder(
            free_variables=free_variables,
            equality_constraints=eq_constraints,
            inequality_constraints=neq_constraints,
            derivative_constraints=derivative_constraints,
            quadratic_weight_gains=quadratic_weight_gains,
            linear_weight_gains=linear_weight_gains,
            sample_period=god_map.qp_controller_config.sample_period,
            prediction_horizon=god_map.qp_controller_config.prediction_horizon,
            solver_id=god_map.qp_controller_config.qp_solver,
            retries_with_relaxed_constraints=god_map.qp_controller_config.retries_with_relaxed_constraints,
            retry_added_slack=god_map.qp_controller_config.added_slack,
            retry_weight_factor=god_map.qp_controller_config.weight_factor,
        )
        god_map.qp_controller = qp_controller

        return Status.SUCCESS

    def get_active_free_symbols(self,
                                eq_constraints: List[EqualityConstraint],
                                neq_constraints: List[InequalityConstraint],
                                derivative_constraints: List[DerivativeInequalityConstraint]):
        symbols = set()
        for c in chain(eq_constraints, neq_constraints, derivative_constraints):
            symbols.update(str(s) for s in w.free_symbols(c.expression))
        free_variables = list(sorted([v for v in god_map.world.free_variables.values() if v.position_name in symbols],
                                     key=lambda x: x.position_name))
        if len(free_variables) == 0:
            raise EmptyProblemException('Goal parsing resulted in no free variables.')
        god_map.free_variables = free_variables
        return free_variables
