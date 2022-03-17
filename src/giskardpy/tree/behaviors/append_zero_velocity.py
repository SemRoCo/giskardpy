from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.data_types import JointStates
from giskardpy.tree.behaviors.plugin import GiskardBehavior


class AppendZeroVelocity(GiskardBehavior):
    def initialise(self):
        self.sample_period = self.get_god_map().get_data(identifier.sample_period)
        super(AppendZeroVelocity, self).initialise()

    @profile
    def update(self):
        # FIXME we do we need this plugin?
        motor_commands = self.get_god_map().get_data(identifier.qp_solver_solution)
        current_js = self.get_god_map().get_data(identifier.joint_states)
        next_js = None
        if motor_commands:
            next_js = JointStates()
            for joint_name, sjs in current_js.items():
                next_js[joint_name].position = sjs.position
        if next_js is not None:
            self.get_god_map().set_data(identifier.joint_states, next_js)
        else:
            self.get_god_map().set_data(identifier.joint_states, current_js)
        return Status.SUCCESS
