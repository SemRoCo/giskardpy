from queue import Queue, Empty, LifoQueue
from threading import Lock, Semaphore
from typing import Optional

import rospy
from py_trees import Status
from geometry_msgs.msg import WrenchStamped

from giskardpy import identifier
from giskardpy.tree.behaviors.plugin import GiskardBehavior


class SyncFTSensor(GiskardBehavior):

    @profile
    def __init__(self, group_name: str, ft_topic='hsrb/wrist_wrench/compensated'):
        """
        :type js_identifier: str
        """
        super().__init__(str(self))
        self.ft_topic = ft_topic
        if not self.ft_topic.startswith('/'):
            self.ft_topic = '/' + self.ft_topic
        super().__init__(str(self))
        self.msg = None
        self.group_name = group_name

    @profile
    def setup(self, timeout=0.0):
        self.sub = rospy.Subscriber(self.ft_topic, WrenchStamped, self.cb, queue_size=1)
        return super().setup(timeout)

    def cb(self, data):
        self.msg = data

    @profile
    def update(self):
        self.god_map.set_data(identifier.ft_msg, self.msg)
        return Status.RUNNING
