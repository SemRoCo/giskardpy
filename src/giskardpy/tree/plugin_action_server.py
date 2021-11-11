from giskardpy.utils import logging

try:
    # Python 2
    from Queue import Empty, Queue
except ImportError:
    # Python 3
    from queue import Queue, Empty

import actionlib
import rospy
from giskard_msgs.msg._MoveGoal import MoveGoal
from giskard_msgs.msg._MoveResult import MoveResult
from py_trees import Blackboard, Status

import giskardpy.identifier as identifier
from giskardpy.exceptions import PreemptedException
from giskardpy.tree.plugin import GiskardBehavior

ERROR_CODE_TO_NAME = {getattr(MoveResult, x): x for x in dir(MoveResult) if x.isupper()}


class ActionServerHandler(object):
    """
    Interface to action server which is more useful for behaviors.
    """

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

    def send_feedback(self):
        # TODO
        pass

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
    def __init__(self, name, as_name, action_type=None):
        self.as_handler = None
        self.as_name = as_name
        self.action_type = action_type
        super(ActionServerBehavior, self).__init__(name)

    def setup(self, timeout):
        # TODO handle timeout
        self.as_handler = Blackboard().get(self.as_name)
        if self.as_handler is None:
            self.as_handler = ActionServerHandler(self.as_name, self.action_type)
            Blackboard().set(self.as_name, self.as_handler)
        return super(ActionServerBehavior, self).setup(timeout)

    def get_as(self):
        """
        :rtype: ActionServerHandler
        """
        return self.as_handler


class GoalReceived(ActionServerBehavior):
    def update(self):
        if self.get_as().has_goal():
            rospy.sleep(.5)
            logging.loginfo(u'Received new goal.')
            return Status.SUCCESS
        return Status.FAILURE


class GetGoal(ActionServerBehavior):
    def __init__(self, name, as_name):
        super(GetGoal, self).__init__(name, as_name)

    def pop_goal(self):
        return self.get_as().pop_goal()


class GoalCanceled(ActionServerBehavior):
    def update(self):
        if self.get_as().is_preempt_requested() and self.get_blackboard_exception() is None:
            logging.logerr('preempted')
            self.raise_to_blackboard(PreemptedException(u''))
        if self.get_blackboard_exception() is not None:
            return Status.SUCCESS
        else:
            return Status.FAILURE


class SendResult(ActionServerBehavior):
    def __init__(self, name, as_name, action_type=None):
        super(SendResult, self).__init__(name, as_name, action_type)

    def update(self):
        skip_failures = self.get_god_map().get_data(identifier.skip_failures)
        Blackboard().set('exception', None) # FIXME move this to reset?
        result = self.get_god_map().get_data(identifier.result_message)

        trajectory = self.get_god_map().get_data(identifier.trajectory)
        sample_period = self.get_god_map().get_data(identifier.sample_period)
        controlled_joints = self.get_god_map().get_data(identifier.controlled_joints)['/pr2_a']
        result.trajectory = trajectory.to_msg(sample_period, controlled_joints, True)

        if result.error_codes[-1] == MoveResult.PREEMPTED:
            logging.logerr('Goal preempted')
            self.get_as().send_preempted(result)
            return Status.SUCCESS
        if skip_failures:
            if not self.any_goal_succeeded(result):
                self.get_as().send_aborted(result)
                return Status.SUCCESS

        else:
            if not self.all_goals_succeeded(result):
                logging.logwarn(u'Failed to execute goal.')
                self.get_as().send_aborted(result)
                return Status.SUCCESS
            else:
                logging.loginfo(u'----------------Successfully executed goal.----------------')
        self.get_as().send_result(result)
        return Status.SUCCESS

    def any_goal_succeeded(self, result):
        """
        :type result: MoveResult
        :rtype: bool
        """
        return MoveResult.SUCCESS in result.error_codes

    def all_goals_succeeded(self, result):
        """
        :type result: MoveResult
        :rtype: bool
        """
        return len([x for x in result.error_codes if x != MoveResult.SUCCESS]) == 0


