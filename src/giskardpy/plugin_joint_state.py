from Queue import Empty, Queue
from collections import OrderedDict

import rospy

from sensor_msgs.msg import JointState

from giskardpy.plugin import Plugin
from giskardpy.trajectory import SingleJointState
from giskardpy.utils import to_joint_state_dict


class JointStatePlugin(Plugin):
    def __init__(self, js_identifier, time_identifier, next_cmd_identifier, sample_period):
        super(JointStatePlugin, self).__init__()
        self.js_identifier = js_identifier
        self.time_identifier = time_identifier
        self.next_cmd_identifier = next_cmd_identifier
        self.sample_period = sample_period
        self.js = None
        self.mjs = None
        self.lock = Queue(maxsize=1)

    def cb(self, data):
        try:
            self.lock.get_nowait()
        except Empty:
            pass
        self.lock.put(data)

    def update(self):
        try:
            js = self.lock.get_nowait()
            self.mjs = to_joint_state_dict(js)
        except Empty:
            pass
        self.god_map.set_data([self.js_identifier], self.mjs)

    def start_always(self):
        self.joint_state_sub = rospy.Subscriber('joint_states', JointState, self.cb, queue_size=1)

    def stop(self):
        self.joint_state_sub.unregister()

    def copy(self):
        return KinematicSimPlugin(js_identifier=self.js_identifier, next_cmd_identifier=self.next_cmd_identifier,
                                  time_identifier=self.time_identifier, sample_period=self.sample_period)


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
        if motor_commands is not None:
            current_js = self.god_map.get_data([self.js_identifier])
            self.next_js = OrderedDict()
            for joint_name, sjs in current_js.items():
                if joint_name in motor_commands:
                    cmd = motor_commands[joint_name]
                else:
                    cmd = 0.0
                self.next_js[joint_name] = SingleJointState(sjs.name, sjs.position + cmd * self.frequency, velocity=cmd)
        if self.next_js is not None:
            self.god_map.set_data([self.js_identifier], self.next_js)
        self.god_map.set_data([self.time_identifier], self.time)

    def start_always(self):
        self.next_js = None

    def copy(self):
        # TODO return copy instead of self
        return self
