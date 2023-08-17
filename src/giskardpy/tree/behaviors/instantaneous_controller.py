from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.qp.qp_controller import QPProblemBuilder
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import catch_and_raise_to_blackboard, record_time


class ControllerPlugin(GiskardBehavior):
    controller: QPProblemBuilder = None

    @catch_and_raise_to_blackboard
    @profile
    def initialise(self):
        self.controller = self.god_map.get_data(identifier.qp_controller)

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        parameters = self.controller.get_parameter_names()
        substitutions = self.god_map.get_values(parameters)

        next_cmds = self.controller.get_cmd(substitutions)
        self.god_map.set_data(identifier.qp_solver_solution, next_cmds)

        return Status.RUNNING

