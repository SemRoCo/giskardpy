from Queue import Empty
from collections import OrderedDict

import rospy
from multiprocessing import Lock, Queue

from copy import deepcopy
from sensor_msgs.msg import JointState

from giskardpy.plugin import IOPlugin
from giskardpy.trajectory import MultiJointState, SingleJointState


class JointStatePlugin(IOPlugin):
    def __init__(self):
        super(JointStatePlugin, self).__init__()
        self.js = None
        self.lock = Queue(maxsize=1)

    def cb(self, data):
        # TODO might still be buggy
        try:
            self.lock.get_nowait()
        except Empty:
            pass
        self.lock.put(data)

    def get_readings(self):
        # TODO probably bug when we dont have a joint state before first get readings call

        js = self.lock.get()
        mjs = OrderedDict()
        for i, joint_name in enumerate(js.name):
            sjs = SingleJointState()
            sjs.name = joint_name
            sjs.position = js.position[i]
            sjs.velocity = js.velocity[i]
            sjs.effort = js.effort[i]
            mjs[joint_name] = sjs
        return {'js': mjs}

    def start(self, god_map):
        self.joint_state_sub = rospy.Subscriber('joint_states', JointState, self.cb)
        super(JointStatePlugin, self).start(god_map)

    def stop(self):
        self.joint_state_sub.unregister()

    def update(self):
        pass

    def copy(self):
        return KinematicSimPlugin()


class KinematicSimPlugin(IOPlugin):
    def __init__(self):
        self.js_identifier = 'js'
        self.next_cmd_identifier = 'motor'
        self.time_identifier = 'time'
        self.frequency = 0.1
        self.time = -self.frequency
        super(KinematicSimPlugin, self).__init__()

    def get_readings(self):
        return {self.js_identifier: self.next_js,
                self.time_identifier: self.time}

    def update(self):
        self.time += self.frequency
        motor_commands = self.god_map.get_data(self.next_cmd_identifier)
        if motor_commands is not None:
            current_js = self.god_map.get_data(self.js_identifier)
            self.next_js = OrderedDict()
            for joint_name, sjs in current_js.items():
                if joint_name in motor_commands:
                    cmd = motor_commands[joint_name]
                else:
                    cmd = 0.0
                self.next_js[joint_name] = SingleJointState(sjs.name, sjs.position + cmd * self.frequency, velocity=cmd)

    def start(self, god_map):
        super(KinematicSimPlugin, self).start(god_map)

    def stop(self):
        pass

    def create_parallel_universe(self):
        return super(KinematicSimPlugin, self).create_parallel_universe()

    def end_parallel_universe(self):
        return super(KinematicSimPlugin, self).end_parallel_universe()

    def copy(self):
        return self
