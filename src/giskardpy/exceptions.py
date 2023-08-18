from giskard_msgs.msg import MoveResult


class DontPrintStackTrace:
    pass


class GiskardException(Exception):
    error_code: int = MoveResult.ERROR
    error_message: str = ''

    def __init__(self, error_message: str):
        super().__init__(error_message)
        self.error_message = error_message

    def to_move_result(self) -> MoveResult:
        move_result = MoveResult()
        move_result.error_code = self.error_code
        move_result.error_message = self.error_message
        return move_result


class SetupException(GiskardException):
    pass


# %% solver exceptions
class QPSolverException(GiskardException):
    error_code = MoveResult.QP_SOLVER_ERROR


class InfeasibleException(QPSolverException):
    error_code = MoveResult.INFEASIBLE


class VelocityLimitUnreachableException(QPSolverException):
    error_code = MoveResult.VELOCITYLIMITUNREACHABLE


class OutOfJointLimitsException(InfeasibleException):
    error_code = MoveResult.OUT_OF_JOINT_LIMITS


class HardConstraintsViolatedException(InfeasibleException):
    error_code = MoveResult.HARD_CONSTRAINTS_VIOLATED


class EmptyProblemException(InfeasibleException, DontPrintStackTrace):
    error_code = MoveResult.EMPTY_PROBLEM


# %% world state exceptions
class WorldException(GiskardException):
    error_code = MoveResult.WORLD_ERROR


class UnknownGroupException(WorldException, KeyError):
    error_code = MoveResult.UNKNOWN_GROUP


class UnknownLinkException(WorldException, KeyError):
    pass


class RobotExistsException(WorldException):
    pass


class DuplicateNameException(WorldException):
    pass


class UnsupportedOptionException(WorldException):
    pass


class CorruptShapeException(WorldException):
    pass


# %% error during motion problem building phase
class ConstraintException(GiskardException):
    error_code = MoveResult.CONSTRAINT_ERROR


class UnknownConstraintException(ConstraintException, KeyError):
    error_code = MoveResult.UNKNOWN_CONSTRAINT


class ConstraintInitalizationException(ConstraintException):
    error_code = MoveResult.CONSTRAINT_INITIALIZATION_ERROR


class InvalidGoalException(ConstraintException):
    error_code = MoveResult.INVALID_GOAL


# %% errors during planning
class PlanningException(GiskardException):
    error_code = MoveResult.CONTROL_ERROR


class ShakingException(PlanningException):
    error_code = MoveResult.SHAKING


class LocalMinimumException(PlanningException):
    error_code = MoveResult.LOCAL_MINIMUM


class SelfCollisionViolatedException(PlanningException):
    error_code = MoveResult.SELF_COLLISION_VIOLATED


# errors during execution
class ExecutionException(GiskardException):
    error_code = MoveResult.EXECUTION_ERROR


class FollowJointTrajectory_INVALID_GOAL(ExecutionException):
    error_code = MoveResult.FollowJointTrajectory_INVALID_GOAL


class FollowJointTrajectory_INVALID_JOINTS(ExecutionException):
    error_code = MoveResult.FollowJointTrajectory_INVALID_JOINTS


class FollowJointTrajectory_OLD_HEADER_TIMESTAMP(ExecutionException):
    error_code = MoveResult.FollowJointTrajectory_OLD_HEADER_TIMESTAMP


class FollowJointTrajectory_PATH_TOLERANCE_VIOLATED(ExecutionException):
    error_code = MoveResult.FollowJointTrajectory_PATH_TOLERANCE_VIOLATED


class FollowJointTrajectory_GOAL_TOLERANCE_VIOLATED(ExecutionException):
    error_code = MoveResult.FollowJointTrajectory_GOAL_TOLERANCE_VIOLATED


class PreemptedException(ExecutionException):
    error_code = MoveResult.PREEMPTED


class ExecutionPreemptedException(ExecutionException):
    error_code = MoveResult.EXECUTION_PREEMPTED


class ExecutionTimeoutException(ExecutionException):
    error_code = MoveResult.EXECUTION_TIMEOUT


class ExecutionSucceededPrematurely(ExecutionException):
    error_code = MoveResult.EXECUTION_SUCCEEDED_PREMATURELY


# %% behavior tree exceptions

class BehaviorTreeException(GiskardException):
    error_code = MoveResult.BEHAVIOR_TREE_ERROR
