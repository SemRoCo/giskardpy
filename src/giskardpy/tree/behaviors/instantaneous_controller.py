from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.qp.qp_controller import QPController


class ControllerPlugin(GiskardBehavior):
    def __init__(self, name):
        super(ControllerPlugin, self).__init__(name)
        self.path_to_functions = self.get_god_map().get_data(identifier.data_folder)
        self.qp_data = {}
        self.rc_prismatic_velocity = self.get_god_map().get_data(identifier.rc_prismatic_velocity)
        self.rc_continuous_velocity = self.get_god_map().get_data(identifier.rc_continuous_velocity)
        self.rc_revolute_velocity = self.get_god_map().get_data(identifier.rc_revolute_velocity)
        self.rc_other_velocity = self.get_god_map().get_data(identifier.rc_other_velocity)
        self.controller = None

    @profile
    def initialise(self):
        super(ControllerPlugin, self).initialise()
        constraints = self.get_god_map().get_data(identifier.constraints)
        vel_constraints = self.get_god_map().get_data(identifier.vel_constraints)
        free_variables = self.get_god_map().get_data(identifier.free_variables)
        debug_expressions = self.get_god_map().get_data(identifier.debug_expressions)

        self.controller = QPController(
            free_variables=free_variables,
            constraints=list(constraints.values()),
            velocity_constraints=list(vel_constraints.values()),
            sample_period=self.get_god_map().to_symbol(identifier.sample_period),
            prediction_horizon=self.get_god_map().unsafe_get_data(identifier.prediction_horizon),
            debug_expressions=debug_expressions,
            solver_name=self.get_god_map().unsafe_get_data(identifier.qp_solver_name),
            retries_with_relaxed_constraints=self.get_god_map().unsafe_get_data(identifier.retries_with_relaxed_constraints),
            retry_added_slack=self.get_god_map().unsafe_get_data(identifier.retry_added_slack),
            retry_weight_factor=self.get_god_map().unsafe_get_data(identifier.retry_weight_factor)
        )

        self.controller.compile()

    @profile
    def update(self):
        parameters = self.controller.get_parameter_names()
        substitutions = self.god_map.get_values(parameters)

        next_cmds, debug_expressions = self.controller.get_cmd(substitutions)
        self.get_god_map().set_data(identifier.qp_solver_solution, next_cmds)
        self.get_god_map().set_data(identifier.debug_expressions_evaluated, debug_expressions)

        return Status.RUNNING
