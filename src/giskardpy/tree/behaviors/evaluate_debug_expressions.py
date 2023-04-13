from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.qp.qp_controller import QPProblemBuilder
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import catch_and_raise_to_blackboard, record_time


class EvaluateDebugExpressions(GiskardBehavior):
    controller: QPProblemBuilder = None

    @catch_and_raise_to_blackboard
    @profile
    def initialise(self):
        self.controller = self.god_map.get_data(identifier.qp_controller)

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        self.controller.eval_debug_exprs()
        return Status.RUNNING

