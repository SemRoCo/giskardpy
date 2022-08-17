from typing import Optional

from rospy import ROSException

from giskardpy.data_types import JointStates
from giskardpy.model.world import SubWorldTree

from queue import Queue, Empty

import rospy
from py_trees import Status
from sensor_msgs.msg import JointState

import giskardpy.identifier as identifier
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging
import giskardpy.utils.tfwrapper as tf


class SyncConfiguration(GiskardBehavior):

    @profile
    def __init__(self, joint_state_topic: str):
        self.joint_state_topic: str = joint_state_topic
        super().__init__(str(self))
        self.mjs: Optional[JointStates] = None
        self.lock = Queue(maxsize=1)

    @profile
    def setup(self, timeout=0.0):
        msg = None
        while msg is None and not rospy.is_shutdown():
            try:
                msg = rospy.wait_for_message(self.joint_state_topic, JointState, rospy.Duration(1))
                self.lock.put(msg)
            except ROSException as e:
                logging.logwarn(f'Waiting for topic \'/{self.joint_state_topic}\' to appear.')
        self.joint_state_sub = rospy.Subscriber(self.joint_state_topic, JointState, self.cb, queue_size=1)
        return super().setup(timeout)

    def cb(self, data):
        try:
            self.lock.get_nowait()
        except Empty:
            pass
        self.lock.put(data)

    @profile
    def update(self):
        try:
            if self.mjs is None:
                js = self.lock.get()
            else:
                js = self.lock.get_nowait()
            self.mjs = JointStates.from_msg(js, None)
        except Empty:
            pass

        self.get_world().state.update(self.mjs)
        self.world.notify_state_change()
        return Status.RUNNING

    def __str__(self):
        return f'{super().__str__()} ({self.joint_state_topic})'

