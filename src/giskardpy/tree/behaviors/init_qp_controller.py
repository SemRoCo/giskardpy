from itertools import chain
from typing import Dict

from py_trees import Status

import giskardpy.casadi_wrapper as w
from giskardpy.exceptions import EmptyProblemException, ConstraintInitalizationException
from giskardpy.god_map import god_map
from giskardpy.qp.constraint import EqualityConstraint, InequalityConstraint, DerivativeInequalityConstraint
from giskardpy.qp.qp_controller import QPProblemBuilder
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import catch_and_raise_to_blackboard, record_time


class InitQPController(GiskardBehavior):
    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        eq_constraints, neq_constraints, derivative_constraints, manip_constraints = god_map.motion_goal_manager.get_constraints_from_goals()
        free_variables = self.get_active_free_symbols(eq_constraints, neq_constraints, derivative_constraints)

        qp_controller = QPProblemBuilder(
            free_variables=free_variables,
            equality_constraints=list(eq_constraints.values()),
            inequality_constraints=list(neq_constraints.values()),
            derivative_constraints=list(derivative_constraints.values()),
            manipulability_constraints=list(manip_constraints.values()),
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
                                eq_constraints: Dict[str, EqualityConstraint],
                                neq_constraints: Dict[str, InequalityConstraint],
                                derivative_constraints: Dict[str, DerivativeInequalityConstraint]):
        symbols = set()
        for c in chain(eq_constraints.values(), neq_constraints.values(), derivative_constraints.values()):
            symbols.update(str(s) for s in w.free_symbols(c.expression))
        free_variables = list(sorted([v for v in god_map.world.free_variables.values() if v.position_name in symbols],
                                     key=lambda x: x.position_name))
        if len(free_variables) == 0:
            raise EmptyProblemException('Goal parsing resulted in no free variables.')
        god_map.free_variables = free_variables
        return free_variables
