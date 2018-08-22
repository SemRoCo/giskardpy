from collections import OrderedDict

from giskardpy.data_types import SingleJointState
from giskardpy.plugin import Plugin


class KinematicSimPlugin(Plugin):
    def __init__(self, js_identifier, next_cmd_identifier, time_identifier, sample_period):
        self.js_identifier = js_identifier
        self.next_cmd_identifier = next_cmd_identifier
        self.time_identifier = time_identifier
        self.frequency = sample_period
        self.time = -self.frequency
        super(KinematicSimPlugin, self).__init__()

    def update(self):
        self.time += self.frequency
        motor_commands = self.god_map.get_data([self.next_cmd_identifier])
        current_js = self.god_map.get_data([self.js_identifier])
        if motor_commands is not None:
            self.next_js = OrderedDict()
            for joint_name, sjs in current_js.items():
                if joint_name in motor_commands:
                    cmd = motor_commands[joint_name]
                else:
                    cmd = 0.0
                self.next_js[joint_name] = SingleJointState(sjs.name, sjs.position + cmd * self.frequency, velocity=cmd)
        if self.next_js is not None:
            self.god_map.set_data([self.js_identifier], self.next_js)
        else:
            self.god_map.set_data([self.js_identifier], current_js)
        self.god_map.set_data([self.time_identifier], self.time)

    def start_always(self):
        self.next_js = None

    def copy(self):
        c = self.__class__(self.js_identifier, self.next_cmd_identifier, self.time_identifier, self.frequency)
        return c
