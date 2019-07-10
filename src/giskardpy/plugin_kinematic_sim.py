from collections import OrderedDict

from py_trees import Status

from giskardpy.data_types import SingleJointState
import giskardpy.identifier as identifier
from giskardpy.plugin import GiskardBehavior


class KinSimPlugin(GiskardBehavior):
    def __init__(self, name):
        """
        :type js_identifier: str
        :type next_cmd_identifier: str
        :type time_identifier: str
        :param sample_period: the time difference in s between each step.
        :type sample_period: float
        """
        super(KinSimPlugin, self).__init__(name)
        self.frequency = self.get_god_map().safe_get_data(identifier.sample_period)

    def initialise(self):
        self.next_js = None
        self.time = -1
        super(KinSimPlugin, self).initialise()

    def update(self):
        self.time += 1
        motor_commands = self.get_god_map().safe_get_data(identifier.cmd)
        current_js = self.get_god_map().safe_get_data(identifier.joint_states)
        if motor_commands:
            self.next_js = OrderedDict()
            for joint_name, sjs in current_js.items():
                if joint_name in motor_commands:
                    cmd = motor_commands[joint_name]
                else:
                    cmd = 0.0
                self.next_js[joint_name] = SingleJointState(sjs.name, sjs.position + cmd, velocity=cmd / self.frequency)
        if self.next_js is not None:
            self.get_god_map().safe_set_data(identifier.joint_states, self.next_js)
        else:
            self.get_god_map().safe_set_data(identifier.joint_states, current_js)
        self.get_god_map().safe_set_data(identifier.time_identifier, self.time)
        return Status.RUNNING
