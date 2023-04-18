from queue import Queue, Empty, LifoQueue
from threading import Lock, Semaphore
from typing import Optional

import rospy
from py_trees import Status
from sensor_msgs.msg import JointState

import giskardpy.utils.tfwrapper as tf
from giskardpy.data_types import JointStates
from giskardpy.model.world import WorldBranch
from giskardpy.tree.behaviors.plugin import GiskardBehavior


class SyncConfiguration2(GiskardBehavior):
    """
    Listens to a joint state topic, transforms it into a dict and writes it to the got map.
    Gets replace with a kinematic sim plugin during a parallel universe.
    """

    @profile
    def __init__(self, group_name: str, joint_state_topic='joint_states'):
        """
        :type js_identifier: str
        """
        super().__init__(str(self))
        self.pub = rospy.Publisher('asdfasfd', JointState, queue_size=10)
        self.joint_state_topic = joint_state_topic
        if not self.joint_state_topic.startswith('/'):
            self.joint_state_topic = '/' + self.joint_state_topic
        super().__init__(str(self))
        self.mjs: Optional[JointStates] = None
        self.group_name = group_name
        self.lock = Lock()
        # self.lock = LifoQueue(maxsize=0)

    @profile
    def setup(self, timeout=0.0):
        self.joint_state_sub = rospy.Subscriber(self.joint_state_topic, JointState, self.cb, queue_size=1)
        return super().setup(timeout)

    def cb(self, data):
        # self.pub.publish(data)
        # try:
        #     self.lock.get_nowait()
        # except Empty:
        #     pass
        self.msg = data
        # try:
        #     self.lock.release()
        # except Exception:
        #     pass
        # self.lock.put(data)

    @profile
    def initialise(self):
        self.last_time = rospy.get_rostime()
        super().initialise()

    @profile
    def update(self):
        # self.lock.acquire()
        # try:
        # if self.mjs is None:
        # js = self.lock.get()
        # else:
        #     js = self.lock.get_nowait()
        # dt = (js.header.stamp - self.last_time).to_sec()
        self.mjs = JointStates.from_msg(self.msg, self.group_name)
        # self.last_time = js.header.stamp
        # self.world.state.update(self.mjs)
        for joint_name, next_state in self.mjs.items():
            # self.world.state[joint_name].acceleration = (next_state.velocity - self.world.state[joint_name].velocity)/dt
            # self.world.state[joint_name].velocity = next_state.velocity
            self.world.state[joint_name].position = next_state.position
        self.world.notify_state_change()
        # except Empty:
        #     pass

        return Status.RUNNING
