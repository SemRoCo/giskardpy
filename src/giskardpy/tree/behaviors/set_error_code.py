from py_trees import Status

import giskardpy.identifier as identifier
from giskard_msgs.msg import MoveResult
from giskardpy.exceptions import *
from giskardpy.goals.goal import NonMotionGoal
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging
from giskardpy.utils.decorators import record_time


class SetErrorCode(GiskardBehavior):

    @profile
    def __init__(self, name, context, print=True):
        self.reachability_threshold = 0.001
        self.print = print
        self.context = context
        super().__init__(name)

    @record_time
    @profile
    def update(self):
        e = self.get_blackboard_exception()

        cmd_id = self.get_god_map().get_data(identifier.cmd_id)

        result = self.get_god_map().get_data(identifier.result_message)
        error_code, error_message = self.exception_to_error_code(e)
        result.error_codes[cmd_id] = error_code
        result.error_messages[cmd_id] = error_message
        trajectory = self.god_map.get_data(identifier.trajectory)
        joints = [self.world.joints[joint_name] for joint_name in self.world.movable_joint_names]
        sample_period = self.god_map.get_data(identifier.sample_period)
        result.trajectory = trajectory.to_msg(sample_period=sample_period, start_time=0, joints=joints)
        if error_code == MoveResult.PREEMPTED:
            for i in range(len(result.error_codes) - cmd_id):
                result.error_codes[cmd_id + i] = error_code
                result.error_messages[cmd_id + i] = error_message
            logging.logwarn(f'Goal preempted: \'{error_message}\'.')
        else:
            if self.print:
                if error_code == MoveResult.SUCCESS:
                    logging.loginfo(f'{self.context} succeeded.')
                else:
                    logging.logwarn(f'{self.context} failed: {error_message}.')
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
            error_message = ''
        error_code = MoveResult.SUCCESS

        # qp exceptions
        if isinstance(exception, QPSolverException):
            error_code = MoveResult.QP_SOLVER_ERROR
            if isinstance(exception, OutOfJointLimitsException):
                error_code = MoveResult.OUT_OF_JOINT_LIMITS
            elif isinstance(exception, HardConstraintsViolatedException):
                error_code = MoveResult.HARD_CONSTRAINTS_VIOLATED
            elif isinstance(exception, EmptyProblemException):
                goals = list(self.god_map.get_data(identifier.goals).values())
                non_motion_goals = [x for x in goals if isinstance(x, NonMotionGoal)]
                if len(non_motion_goals) == 0:
                    error_code = MoveResult.EMPTY_PROBLEM
                else:
                    error_code = MoveResult.SUCCESS
                    error_message = ''
        # world exceptions
        elif isinstance(exception, PhysicsWorldException):
            error_code = MoveResult.WORLD_ERROR
            if isinstance(exception, UnknownGroupException):
                error_code = MoveResult.UNKNOWN_GROUP
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
            elif isinstance(exception, SelfCollisionViolatedException):
                error_code = MoveResult.SELF_COLLISION_VIOLATED
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
            elif isinstance(exception, ExecutionPreemptedException):
                error_code = MoveResult.EXECUTION_PREEMPTED
            elif isinstance(exception, ExecutionTimeoutException):
                error_code = MoveResult.EXECUTION_TIMEOUT
            elif isinstance(exception, ExecutionSucceededPrematurely):
                error_code = MoveResult.EXECUTION_SUCCEEDED_PREMATURELY
            elif isinstance(exception, FollowJointTrajectory_INVALID_GOAL):
                error_code = MoveResult.FollowJointTrajectory_INVALID_GOAL
            elif isinstance(exception, FollowJointTrajectory_INVALID_JOINTS):
                error_code = MoveResult.FollowJointTrajectory_INVALID_JOINTS
            elif isinstance(exception, FollowJointTrajectory_OLD_HEADER_TIMESTAMP):
                error_code = MoveResult.FollowJointTrajectory_OLD_HEADER_TIMESTAMP
            elif isinstance(exception, FollowJointTrajectory_PATH_TOLERANCE_VIOLATED):
                error_code = MoveResult.FollowJointTrajectory_PATH_TOLERANCE_VIOLATED
            elif isinstance(exception, FollowJointTrajectory_GOAL_TOLERANCE_VIOLATED):
                error_code = MoveResult.FollowJointTrajectory_GOAL_TOLERANCE_VIOLATED

        elif isinstance(exception, ImplementationException):
            print(exception)
            error_code = MoveResult.ERROR
        elif exception is not None:
            error_code = MoveResult.ERROR
        return error_code, error_message
