import rospy
from multiprocessing import Lock

from copy import deepcopy
from sensor_msgs.msg import JointState

from giskardpy.plugin import InputPlugin


class JointStateInput(InputPlugin):
    def __init__(self):
        super(JointStateInput, self).__init__()
        self.js = None
        self.lock = Lock() #TODO not sure if locks are really neccessary

    def cb(self, data):
        with self.lock:
            self.js = data

    def get_readings(self):
        with self.lock:
            return {'js': deepcopy(self.js)}

    def start(self):
        self.joint_state_sub = rospy.Subscriber('joint_states', JointState, self.cb)

    def stop(self):
        self.joint_state_sub.unregister()
