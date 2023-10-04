from py_trees import Status

from giskardpy.god_map_interpreter import god_map
from giskardpy.qp.qp_controller import QPProblemBuilder
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import catch_and_raise_to_blackboard, record_time


class EvaluateDebugExpressions(GiskardBehavior):
    controller: QPProblemBuilder = None

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        god_map.debug_expression_manager.eval_debug_exprs()
        return Status.RUNNING

