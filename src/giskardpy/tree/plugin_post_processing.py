import numpy as np
from giskard_msgs.msg import MoveResult
from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.exceptions import UnreachableException, MAX_NWSR_REACHEDException, \
    QPSolverException, UnknownBodyException, ImplementationException, OutOfJointLimitsException, \
    HardConstraintsViolatedException, PhysicsWorldException, ConstraintException, UnknownConstraintException, \
    ConstraintInitalizationException, PlanningException, ShakingException, ExecutionException, InvalidGoalException, \
    PreemptedException
from giskardpy.tree.plugin import GiskardBehavior
from giskardpy.utils.utils import logging


class PostProcessing(GiskardBehavior):

    def __init__(self, name):
        self.reachability_threshold = 0.001
        super(PostProcessing, self).__init__(name)

    def setup(self, timeout):
        # return True
        return super(PostProcessing, self).setup(timeout)

    def initialise(self):
        super(PostProcessing, self).initialise()

    @profile
    def update(self):
        if self.get_god_map().get_data(identifier.check_reachability):
            raise NotImplementedError()
        e = self.get_blackboard_exception()

        cmd_id = self.get_god_map().get_data(identifier.cmd_id)

        result = self.get_god_map().get_data(identifier.result_message)
        error_code, error_message = self.exception_to_error_code(e)
        result.error_codes[cmd_id] = error_code
        result.error_messages[cmd_id] = error_message
        if error_code == MoveResult.PREEMPTED:
            for i in range(len(result.error_codes)-cmd_id):
                result.error_codes[cmd_id + i] = error_code
                result.error_messages[cmd_id + i] = error_message
            logging.logwarn(u'Planning preempted: {}.'.format(error_message))
        else:
            if error_code == MoveResult.SUCCESS:
                logging.loginfo(u'Planning succeeded.')
            else:
                logging.logwarn(u'Planning failed: {}.'.format(error_message))
        self.get_god_map().set_data(identifier.result_message, result)
        return Status.SUCCESS

    def exception_to_error_code(self, exception):
        """
        :type exception: Exception
        :rtype: int
        """
        try:
            error_message = str(exception)
        except:
            error_message = u''
        error_code = MoveResult.SUCCESS

        # qp exceptions
        if isinstance(exception, QPSolverException):
            error_code = MoveResult.QP_SOLVER_ERROR
            if isinstance(exception, MAX_NWSR_REACHEDException):
                error_code = MoveResult.MAX_NWSR_REACHED
            elif isinstance(exception, OutOfJointLimitsException):
                error_code = MoveResult.OUT_OF_JOINT_LIMITS
            elif isinstance(exception, HardConstraintsViolatedException):
                error_code = MoveResult.HARD_CONSTRAINTS_VIOLATED
        # world exceptions
        elif isinstance(exception, PhysicsWorldException):
            error_code = MoveResult.WORLD_ERROR
            if isinstance(exception, UnknownBodyException):
                error_code = MoveResult.UNKNOWN_OBJECT
        # problem building exceptions
        elif isinstance(exception, ConstraintException):
            error_code = MoveResult.CONSTRAINT_ERROR
            if isinstance(exception, UnknownConstraintException):
                error_code = MoveResult.UNKNOWN_CONSTRAINT
            elif isinstance(exception, ConstraintInitalizationException):
                error_code = MoveResult.CONSTRAINT_INITIALIZATION_ERROR
            elif isinstance(exception, InvalidGoalException):
                error_code = MoveResult.INVALID_GOAL
        # planning exceptions
        elif isinstance(exception, PlanningException):
            error_code = MoveResult.PLANNING_ERROR
            if isinstance(exception, ShakingException):
                error_code = MoveResult.SHAKING
            elif isinstance(exception, UnreachableException):
                if self.get_god_map().get_data(identifier.check_reachability):
                    error_code = MoveResult.UNREACHABLE
                else:
                    error_code = MoveResult.ERROR
        # execution exceptions
        elif isinstance(exception, ExecutionException):
            error_code = MoveResult.EXECUTION_ERROR
            if isinstance(exception, PreemptedException):
                error_code = MoveResult.PREEMPTED

        elif isinstance(exception, ImplementationException):
            print(exception)
            error_code = MoveResult.ERROR
        elif exception is not None:
            error_code = MoveResult.ERROR
        return error_code, error_message
