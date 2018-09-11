from Queue import Empty, Queue
from time import time

import rospy
from iai_wsg_50_msgs.msg import Status
from py_trees import Status, Blackboard

from sensor_msgs.msg import JointState
from smach import State

from giskardpy.plugin import PluginBase, GiskardBehavior, NewPluginBase
from giskardpy.plugin_kinematic_sim import KinematicSimPlugin
from giskardpy.utils import to_joint_state_dict


class JointStatePlugin(PluginBase):
    """
    Listens to a joint state topic, transforms it into a dict and writes it to the got map.
    Gets replace with a kinematic sim plugin during a parallel universe.
    """

    def __init__(self, js_identifier, time_identifier, next_cmd_identifier, sample_period):
        """
        :type js_identifier: str
        :type time_identifier: str
        :type next_cmd_identifier: str
        :param sample_period: gets passed to KinematicSimPlugin
        :type: int
        """
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
        self.god_map.safe_set_data([self.js_identifier], self.mjs)

    def initialize(self):
        self.joint_state_sub = rospy.Subscriber(u'joint_states', JointState, self.cb, queue_size=1)

    def stop(self):
        self.joint_state_sub.unregister()

    def copy(self):
        return KinematicSimPlugin(js_identifier=self.js_identifier, next_cmd_identifier=self.next_cmd_identifier,
                                  time_identifier=self.time_identifier, sample_period=self.sample_period)


class JointStatePlugin2(NewPluginBase):
    """
    Listens to a joint state topic, transforms it into a dict and writes it to the got map.
    Gets replace with a kinematic sim plugin during a parallel universe.
    """

    def __init__(self, js_identifier):
        """
        :type js_identifier: str
        """
        super(JointStatePlugin2, self).__init__()
        self.js_identifier = js_identifier
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
        self.god_map.safe_set_data([self.js_identifier], self.mjs)
        return Status.RUNNING

    def setup(self):
        self.joint_state_sub = rospy.Subscriber(u'joint_states', JointState, self.cb, queue_size=1)



class JSBehavior(GiskardBehavior):
    def __init__(self, name, js_identifier):
        self.sub = rospy.Subscriber('joint_states', JointState, self.cb)
        self.lock = Queue(maxsize=1)
        self.js_identifier = js_identifier
        self.mjs = None
        super(JSBehavior, self).__init__(name)

    def setup(self, timeout):
        self.sub = rospy.Subscriber('joint_states', JointState, self.cb)
        self.lock = Queue(maxsize=1)
        # self.js_identifier = js_identifier
        self.js_identifier = 'js'
        self.mjs = None
        return super(JSBehavior, self).setup(timeout)

    def cb(self, data):
        try:
            self.lock.get_nowait()
        except Empty:
            pass
        self.lock.put(data)

    def update(self):
        # Blackboard().time = time()
        try:
            js = self.lock.get_nowait()
            self.mjs = to_joint_state_dict(js)
        except Empty:
            pass
        self.god_map.safe_set_data([self.js_identifier], self.mjs)
        return Status.SUCCESS