from queue import Queue, Empty
from typing import Any

import actionlib
import rospy
from py_trees import Blackboard

from giskard_msgs.msg import MoveGoal
from giskard_msgs.msg import MoveResult
from giskardpy.exceptions import GiskardException
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class ActionServerHandler:
    """
    Interface to action server which is more useful for behaviors.
    """
    goal_id: int
    name: str

    @record_time
    def __init__(self, action_name: str, action_type: Any):
        self.name = action_name
        self.goal_id = -1
        self.goal_msg = None
        self._result_msg = None
        self.goal_queue = Queue(1)
        self.result_queue = Queue(1)
        self._as = actionlib.SimpleActionServer(self.name, action_type,
                                                execute_cb=self.execute_cb, auto_start=False)
        self._as.start()

    def execute_cb(self, goal) -> None:
        self.goal_queue.put(goal)
        result_cb = self.result_queue.get()
        result_cb()
        self.goal_msg = None
        self.result_msg = None

    def accept_goal(self) -> None:
        try:
            self.goal_msg = self.goal_queue.get_nowait()
            self.goal_id += 1
        except Empty:
            return None

    @property
    def result_msg(self):
        if self._result_msg is None:
            raise GiskardException('no result message set.')
        return self._result_msg

    @result_msg.setter
    def result_msg(self, value):
        self._result_msg = value

    def has_goal(self):
        return not self.goal_queue.empty()

    def send_feedback(self, message):
        self._as.publish_feedback(message)

    def send_preempted(self):
        def call_me_now():
            self._as.set_preempted(self.result_msg)

        self.result_queue.put(call_me_now)

    def send_aborted(self):
        def call_me_now():
            self._as.set_aborted(self.result_msg)

        self.result_queue.put(call_me_now)

    def send_result(self):
        def call_me_now():
            self._as.set_succeeded(self.result_msg)

        self.result_queue.put(call_me_now)

    def is_preempt_requested(self):
        return self._as.is_preempt_requested()
