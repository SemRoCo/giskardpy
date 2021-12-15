from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.data_types import JointStates
from giskardpy.tree.behaviors.plugin import GiskardBehavior


class AppendZeroVelocity(GiskardBehavior):
    def __init__(self, name):
        """
        :type js_identifier: str
        :type next_cmd_identifier: str
        :type time_identifier: str
        :param sample_period: the time difference in s between each step.
        :type sample_period: float
        """
        super(AppendZeroVelocity, self).__init__(name)

    def initialise(self):
        self.sample_period = self.get_god_map().get_data(identifier.sample_period)
        super(AppendZeroVelocity, self).initialise()

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
