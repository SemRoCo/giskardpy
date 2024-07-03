from typing import Optional

from py_trees import Status

from giskardpy.god_map import god_map
from giskardpy.qp.qp_controller import QPProblemBuilder
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import catch_and_raise_to_blackboard, record_time


class EvaluateDebugExpressions(GiskardBehavior):
    controller: QPProblemBuilder = None

    def __init__(self, name: str = 'eval debug expressions', log_traj: bool = True):
        super().__init__(name)
        self.log_traj = log_traj

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        god_map.debug_expression_manager.eval_debug_expressions(self.log_traj)
        return Status.RUNNING

