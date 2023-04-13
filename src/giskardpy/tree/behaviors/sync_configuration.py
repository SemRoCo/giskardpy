from queue import Queue, Empty
from typing import Optional

import rospy
from py_trees import Status
from rospy import ROSException
from sensor_msgs.msg import JointState

from giskardpy.data_types import JointStates
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging
from giskardpy.utils.decorators import record_time


class SyncConfiguration(GiskardBehavior):

    @record_time
    @profile
    def __init__(self, group_name: str, joint_state_topic: str = 'joint_states'):
        self.joint_state_topic = joint_state_topic
        if not self.joint_state_topic.startswith('/'):
            self.joint_state_topic = '/' + self.joint_state_topic
        super().__init__(str(self))
        self.mjs: Optional[JointStates] = None
        self.group_name = group_name
        self.lock = Queue(maxsize=1)

    @record_time
    @profile
    def setup(self, timeout=0.0):
        msg = None
        while msg is None and not rospy.is_shutdown():
            try:
                msg = rospy.wait_for_message(self.joint_state_topic, JointState, rospy.Duration(1))
                self.lock.put(msg)
            except ROSException as e:
                logging.logwarn(f'Waiting for topic \'{self.joint_state_topic}\' to appear.')
        self.joint_state_sub = rospy.Subscriber(self.joint_state_topic, JointState, self.cb, queue_size=1)
        return super().setup(timeout)

    def cb(self, data):
        try:
            self.lock.get_nowait()
        except Empty:
            pass
        self.lock.put(data)

    @record_time
    @profile
    def update(self):
        try:
            if self.mjs is None:
                js = self.lock.get()
            else:
                js = self.lock.get_nowait()
            self.mjs = JointStates.from_msg(js, self.group_name)
        except Empty:
            pass

        self.get_world().state.update(self.mjs)
        self.world.notify_state_change()
        return Status.RUNNING

    def __str__(self):
        return f'{super().__str__()} ({self.joint_state_topic})'

