from typing import Optional

import rospy
from py_trees import Status

from giskardpy.tree.behaviors.plugin import GiskardBehavior


class Sleep(GiskardBehavior):
    def __init__(self, name: Optional[str] = None, time: float = 0.1):
        super().__init__(name)
        self.time = time

    def update(self):
        rospy.sleep(self.time)
        return Status.SUCCESS
