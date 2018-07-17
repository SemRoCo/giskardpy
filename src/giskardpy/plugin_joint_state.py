from Queue import Empty, Queue

import rospy

from sensor_msgs.msg import JointState

from giskardpy.plugin import Plugin
from giskardpy.plugin_kinematic_sim import KinematicSimPlugin
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

