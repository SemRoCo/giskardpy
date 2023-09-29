from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.god_map_interpreter import god_map
from giskardpy.qp.qp_controller import QPProblemBuilder
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import catch_and_raise_to_blackboard, record_time


class ControllerPluginBase(GiskardBehavior):
    controller: QPProblemBuilder = None

    @catch_and_raise_to_blackboard
    @profile
    def initialise(self):
        self.controller = god_map.qp_controller

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        parameters = self.controller.get_parameter_names()
        substitutions = god_map.get_values(parameters)

        next_cmds = self.controller.get_cmd(substitutions)
        god_map.qp_solver_solution = next_cmds

        return Status.RUNNING

