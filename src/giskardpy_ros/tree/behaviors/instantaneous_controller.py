from py_trees import Status

from giskardpy.god_map import god_map
from giskardpy.qp.qp_controller import QPController
from giskardpy.symbol_manager import symbol_manager
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time
from giskardpy_ros.tree.blackboard_utils import catch_and_raise_to_blackboard


class ControllerPlugin(GiskardBehavior):
    controller: QPController = None

    @catch_and_raise_to_blackboard
    @profile
    def initialise(self):
        self.controller = god_map.qp_controller

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        parameters = self.controller.get_parameter_names()
        substitutions = symbol_manager.resolve_symbols(parameters)

        next_cmds = self.controller.get_cmd(substitutions)
        god_map.qp_solver_solution = next_cmds
        # non_negative_entries = goal_reached_panda['data'] >= 0
        # if (goal_reached_panda.loc[non_negative_entries]['data'] == 0).any():
        #     return Status.RUNNING
        return Status.RUNNING

