from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.qp.qp_controller import QPController
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.utils import catch_and_raise_to_blackboard


class ControllerPlugin(GiskardBehavior):
    controller: QPController = None

    @profile
    @catch_and_raise_to_blackboard
    def initialise(self):
        self.controller = self.god_map.get_data(identifier.qp_controller)

    @profile
    @catch_and_raise_to_blackboard
    def update(self):
        parameters = self.controller.get_parameter_names()
        substitutions = self.god_map.get_values(parameters)

        next_cmds, debug_expressions = self.controller.get_cmd(substitutions)
        self.get_god_map().set_data(identifier.qp_solver_solution, next_cmds)
        self.get_god_map().set_data(identifier.debug_expressions_evaluated, debug_expressions)

        return Status.RUNNING
