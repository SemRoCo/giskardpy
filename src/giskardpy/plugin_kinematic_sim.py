from collections import OrderedDict

from py_trees import Status

from giskardpy.data_types import SingleJointState
from giskardpy.identifier import time_identifier, js_identifier, next_cmd_identifier
from giskardpy.plugin import GiskardBehavior


class KinSimPlugin(GiskardBehavior):
    def __init__(self, name, sample_period):
        """
        :type js_identifier: str
        :type next_cmd_identifier: str
        :type time_identifier: str
        :param sample_period: the time difference in s between each step.
        :type sample_period: float
        """
        self.frequency = sample_period
        super(KinSimPlugin, self).__init__(name)

    def initialise(self):
        self.next_js = None
        self.time = -self.frequency
        super(KinSimPlugin, self).initialise()

    def update(self):
        self.time += self.frequency
        motor_commands = self.get_god_map().safe_get_data(next_cmd_identifier)
        current_js = self.get_god_map().safe_get_data(js_identifier)
        if motor_commands is not None:
            self.next_js = OrderedDict()
            for joint_name, sjs in current_js.items():
                if joint_name in motor_commands:
                    cmd = motor_commands[joint_name]
                else:
                    cmd = 0.0
                self.next_js[joint_name] = SingleJointState(sjs.name, sjs.position + cmd * self.frequency, velocity=cmd)
        if self.next_js is not None:
            self.get_god_map().safe_set_data(js_identifier, self.next_js)
        else:
            self.get_god_map().safe_set_data(js_identifier, current_js)
        self.get_god_map().safe_set_data(time_identifier, self.time)
        return Status.RUNNING