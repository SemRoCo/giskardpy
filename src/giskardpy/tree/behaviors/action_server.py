from queue import Queue, Empty

import actionlib
from py_trees import Blackboard

from giskard_msgs.msg._MoveGoal import MoveGoal
from giskard_msgs.msg._MoveResult import MoveResult
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time

ERROR_CODE_TO_NAME = {getattr(MoveResult, x): x for x in dir(MoveResult) if x.isupper()}


class ActionServerHandler(object):
    """
    Interface to action server which is more useful for behaviors.
    """

    @record_time
    def __init__(self, action_name, action_type):
        self.goal_queue = Queue(1)
        self.result_queue = Queue(1)
        self._as = actionlib.SimpleActionServer(action_name, action_type,
                                                execute_cb=self.execute_cb, auto_start=False)
        self._as.start()

    def execute_cb(self, goal):
        """
        :type goal: MoveGoal
        """
        self.goal_queue.put(goal)
        result_cb = self.result_queue.get()
        result_cb()

    def pop_goal(self):
        try:
            goal = self.goal_queue.get_nowait()
            # self.canceled = False
            return goal
        except Empty:
            return None

    def has_goal(self):
        return not self.goal_queue.empty()

    def send_feedback(self, message):
        self._as.publish_feedback(message)

    def send_preempted(self, result=None):
        def call_me_now():
            self._as.set_preempted(result)

        self.result_queue.put(call_me_now)

    def send_aborted(self, result=None):
        def call_me_now():
            self._as.set_aborted(result)

        self.result_queue.put(call_me_now)

    def send_result(self, result=None):
        """
        :type result: MoveResult
        """

        def call_me_now():
            self._as.set_succeeded(result)

        self.result_queue.put(call_me_now)

    def is_preempt_requested(self):
        return self._as.is_preempt_requested()


class ActionServerBehavior(GiskardBehavior):
    as_handler: ActionServerHandler

    @record_time
    def __init__(self, name: str, as_name: str, action_type=None):
        self.as_name = as_name
        self.action_type = action_type
        self.as_handler = Blackboard().get(self.as_name)
        if self.as_handler is None:
            self.as_handler = ActionServerHandler(self.as_name, self.action_type)
            Blackboard().set(self.as_name, self.as_handler)
        super().__init__(name)

    def get_as(self) -> ActionServerHandler:
        return self.as_handler
