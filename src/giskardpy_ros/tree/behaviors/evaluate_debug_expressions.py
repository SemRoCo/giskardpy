from typing import Optional

from py_trees import Status

from giskardpy.god_map import god_map
from giskardpy.qp.qp_controller import QPController
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time
from giskardpy_ros.tree.blackboard_utils import catch_and_raise_to_blackboard


class EvaluateDebugExpressions(GiskardBehavior):
    controller: QPController = None

    def __init__(self, name: str = 'eval debug expressions', log_traj: bool = True):
        super().__init__(name)
        self.log_traj = log_traj

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        god_map.debug_expression_manager.eval_debug_expressions(self.log_traj)
        return Status.RUNNING

