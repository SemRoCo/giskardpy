from collections import OrderedDict

import rospy
from multiprocessing import Lock

from copy import deepcopy
from sensor_msgs.msg import JointState

from giskardpy.plugin import IOPlugin
from giskardpy.trajectory import MultiJointState, SingleJointState


class JointStateInput(IOPlugin):
    def __init__(self):
        super(JointStateInput, self).__init__()
        self.js = None
        self.lock = Lock() #TODO not sure if locks are really neccessary

    def cb(self, data):
        with self.lock:
            self.js = data

    def get_readings(self):
        with self.lock:
            mjs = OrderedDict()
            for i, joint_name in enumerate(self.js.name):
                sjs = SingleJointState()
                sjs.name = joint_name
                sjs.position = self.js.position[i]
                sjs.velocity = self.js.velocity[i]
                sjs.effort = self.js.effort[i]
                mjs[joint_name] = sjs
        return {'js': mjs}

    def start(self, databus):
        self.joint_state_sub = rospy.Subscriber('joint_states', JointState, self.cb)
        super(JointStateInput, self).start(databus)

    def stop(self):
        self.joint_state_sub.unregister()

    def update(self):
        pass
