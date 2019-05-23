from Queue import Empty, Queue

import actionlib
import rospy
from giskard_msgs.msg._MoveGoal import MoveGoal
from giskard_msgs.msg._MoveResult import MoveResult
from py_trees import Blackboard, Status

from giskardpy.exceptions import MAX_NWSR_REACHEDException, QPSolverException, SolverTimeoutError, InsolvableException, \
    SymengineException, PathCollisionException, UnknownBodyException, ImplementationException
from giskardpy.identifier import trajectory_identifier
from giskardpy.logging import loginfo
from giskardpy.plugin import GiskardBehavior
from giskardpy.utils import plot_trajectory

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
        self.result_queue.get()()

    def get_goal(self):
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
        # TODO put shit in queue
        def call_me_now():
            self._as.set_preempted(result)

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
            return Status.SUCCESS
        return Status.FAILURE


class GetGoal(ActionServerBehavior):
    def __init__(self, name, as_name):
        super(GetGoal, self).__init__(name, as_name)

    def get_goal(self):
        return self.get_as().get_goal()


class GoalCanceled(ActionServerBehavior):
    def update(self):
        if self.get_as().is_preempt_requested() or Blackboard().get('exception') is not None:
            return Status.SUCCESS
        else:
            return Status.FAILURE


class SendResult(ActionServerBehavior):
    def __init__(self, name, as_name, path_to_data_folder, action_type=None, plot_trajectory=False):
        self.path_to_data_folder = path_to_data_folder
        self.plot_trajectory = plot_trajectory
        super(SendResult, self).__init__(name, as_name, action_type)

    def update(self):
        # TODO get result from god map or blackboard
        e = Blackboard().get('exception')
        Blackboard().set('exception', None)
        result = MoveResult()
        result.error_code = self.exception_to_error_code(e)
        if self.get_as().is_preempt_requested() or not result.error_code == MoveResult.SUCCESS:
            self.plot_traj()
            self.get_as().send_preempted(result)
        else:
            self.plot_traj()
            self.get_as().send_result(result)
        return Status.SUCCESS

    def plot_traj(self):
        if self.plot_trajectory:
            trajectory = self.get_god_map().safe_get_data(trajectory_identifier)
            controlled_joints = self.get_robot().controlled_joints
            plot_trajectory(trajectory, controlled_joints, self.path_to_data_folder)

    def exception_to_error_code(self, exception):
        """
        :type exception: Exception
        :rtype: int
        """
        error_code = MoveResult.SUCCESS
        if isinstance(exception, MAX_NWSR_REACHEDException):
            error_code = MoveResult.MAX_NWSR_REACHED
        elif isinstance(exception, QPSolverException):
            error_code = MoveResult.QP_SOLVER_ERROR
        elif isinstance(exception, UnknownBodyException):
            error_code = MoveResult.UNKNOWN_OBJECT
        elif isinstance(exception, SolverTimeoutError):
            error_code = MoveResult.SOLVER_TIMEOUT
        elif isinstance(exception, InsolvableException):
            error_code = MoveResult.INSOLVABLE
        elif isinstance(exception, SymengineException):
            error_code = MoveResult.SYMENGINE_ERROR
        elif isinstance(exception, PathCollisionException):
            error_code = MoveResult.PATH_COLLISION
        elif isinstance(exception, ImplementationException):
            print(exception)
            error_code = MoveResult.INSOLVABLE
        elif exception is not None:
            error_code = MoveResult.INSOLVABLE
        return error_code
