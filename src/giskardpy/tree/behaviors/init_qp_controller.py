from itertools import chain
from typing import Dict

from py_trees import Status

import giskardpy.casadi_wrapper as w
import giskardpy.identifier as identifier
from giskardpy.exceptions import EmptyProblemException, ConstraintInitalizationException
from giskardpy.goals.goal import Goal
from giskardpy.qp.qp_controller import QPController
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.utils import catch_and_raise_to_blackboard


class InitQPController(GiskardBehavior):
    @catch_and_raise_to_blackboard
    @profile
    def update(self):
        constraints, vel_constraints, debug_expressions = self.get_constraints_from_goals()
        free_variables = self.get_active_free_symbols(constraints, vel_constraints)

        qp_controller = QPController(
            free_variables=free_variables,
            constraints=list(constraints.values()),
            velocity_constraints=list(vel_constraints.values()),
            sample_period=self.get_god_map().unsafe_get_data(identifier.sample_period),
            prediction_horizon=self.get_god_map().unsafe_get_data(identifier.prediction_horizon),
            debug_expressions=debug_expressions,
            solver_name=self.get_god_map().unsafe_get_data(identifier.qp_solver_name),
            retries_with_relaxed_constraints=self.get_god_map().unsafe_get_data(
                identifier.retries_with_relaxed_constraints),
            retry_added_slack=self.get_god_map().unsafe_get_data(identifier.retry_added_slack),
            retry_weight_factor=self.get_god_map().unsafe_get_data(identifier.retry_weight_factor),
            time_collector=self.time_collector
        )
        qp_controller.compile()
        self.god_map.set_data(identifier.qp_controller, qp_controller)

        return Status.SUCCESS

    @profile
    def get_constraints_from_goals(self):
        constraints = {}
        vel_constraints = {}
        debug_expressions = {}
        goals: Dict[str, Goal] = self.god_map.get_data(identifier.goals)
        for goal_name, goal in list(goals.items()):
            try:
                _constraints, _vel_constraints, _debug_expressions = goal.get_constraints()
            except Exception as e:
                raise ConstraintInitalizationException(str(e))
            constraints.update(_constraints)
            vel_constraints.update(_vel_constraints)
            debug_expressions.update(_debug_expressions)
            # logging.loginfo(f'{goal_name} added {len(_constraints)+len(_vel_constraints)} constraints.')
        self.get_god_map().set_data(identifier.constraints, constraints)
        self.get_god_map().set_data(identifier.vel_constraints, vel_constraints)
        self.get_god_map().set_data(identifier.debug_expressions, debug_expressions)
        return constraints, vel_constraints, debug_expressions

    def get_active_free_symbols(self, constraints, vel_constraints):
        symbols = set()
        for c in chain(constraints.values(), vel_constraints.values()):
            symbols.update(str(s) for s in w.free_symbols(c.expression))
        free_variables = list(sorted([v for v in self.world.free_variables.values() if v.position_name in symbols],
                                     key=lambda x: x.position_name))
        if len(free_variables) == 0:
            raise EmptyProblemException('Goal parsing resulted in no free variables.')
        self.get_god_map().set_data(identifier.free_variables, free_variables)
        return free_variables
