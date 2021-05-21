from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.plugin import GiskardBehavior
from giskardpy.qp_controller import QPController


class ControllerPlugin(GiskardBehavior):
    def __init__(self, name):
        super(ControllerPlugin, self).__init__(name)
        self.path_to_functions = self.get_god_map().get_data(identifier.data_folder)
        self.qp_data = {}
        self.get_god_map().set_data(identifier.qp_data, self.qp_data)  # safe dict on godmap and work on ref
        self.rc_prismatic_velocity = self.get_god_map().get_data(identifier.rc_prismatic_velocity)
        self.rc_continuous_velocity = self.get_god_map().get_data(identifier.rc_continuous_velocity)
        self.rc_revolute_velocity = self.get_god_map().get_data(identifier.rc_revolute_velocity)
        self.rc_other_velocity = self.get_god_map().get_data(identifier.rc_other_velocity)
        self.controller = None

    def initialise(self):
        super(ControllerPlugin, self).initialise()
        self.init_controller()

    def setup(self, timeout=0.0):
        return super(ControllerPlugin, self).setup(5.0)

    def init_controller(self):
        constraints = self.get_god_map().get_data(identifier.constraints)
        free_variables = self.get_god_map().get_data(identifier.free_variables)
        debug_expressions = self.get_god_map().get_data(identifier.debug_expressions)

        self.controller = QPController(
            free_variables=list(free_variables.values()),
            constraints=list(constraints.values()),
            sample_period=self.get_god_map().unsafe_get_data(identifier.sample_period),
            prediciton_horizon=self.get_god_map().unsafe_get_data(identifier.prediction_horizon),
            control_horizon=self.get_god_map().unsafe_get_data(identifier.control_horizon),
            debug_expressions=debug_expressions,
            solver_name=self.get_god_map().unsafe_get_data(identifier.qp_solver_name)
        )

        self.controller.compile()

    @profile
    def update(self):
        expr = self.controller.get_parameter_names()
        expr = self.god_map.get_values(expr)

        (next_velocity, next_acceleration, next_jerk), debug_expressions = self.controller.get_cmd(expr)
        self.get_god_map().set_data(identifier.qp_solver_solution, [next_velocity, next_acceleration, next_jerk])
        self.get_god_map().set_data(identifier.debug_expressions_evaluated, debug_expressions)

        return Status.RUNNING
