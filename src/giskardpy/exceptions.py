from giskard_msgs.msg import MoveResult


class DontPrintStackTrace:
    pass


class GiskardException(Exception):
    error_code: int = MoveResult.ERROR
    error_message: str = ''
    _error_code_map = {}

    def __init__(self, error_message: str):
        super().__init__(error_message)
        self.error_message = error_message

    def to_move_result(self) -> MoveResult:
        move_result = MoveResult()
        move_result.error_code = self.error_code
        move_result.error_message = self.error_message
        return move_result

    @classmethod
    def register_error_code(cls, error_code):
        def decorator(subclass):
            cls._error_code_map[error_code] = subclass
            subclass.error_code = error_code
            return subclass

        return decorator

    @classmethod
    def from_error_code(cls, error_code: int, error_message: str):
        subclass = cls._error_code_map.get(error_code, cls)
        return subclass(error_message=error_message)


GiskardException._error_code_map[MoveResult.ERROR] = GiskardException


@GiskardException.register_error_code(MoveResult.SETUP_ERROR)
class SetupException(GiskardException):
    pass


@GiskardException.register_error_code(MoveResult.DUPLICATE_NAME)
class DuplicateNameException(GiskardException):
    pass


# %% solver exceptions
@GiskardException.register_error_code(MoveResult.QP_SOLVER_ERROR)
class QPSolverException(GiskardException):
    pass


@GiskardException.register_error_code(MoveResult.INFEASIBLE)
class InfeasibleException(QPSolverException):
    pass


@GiskardException.register_error_code(MoveResult.VELOCITY_LIMIT_UNREACHABLE)
class VelocityLimitUnreachableException(QPSolverException):
    pass


@GiskardException.register_error_code(MoveResult.OUT_OF_JOINT_LIMITS)
class OutOfJointLimitsException(InfeasibleException):
    pass


@GiskardException.register_error_code(MoveResult.HARD_CONSTRAINTS_VIOLATED)
class HardConstraintsViolatedException(InfeasibleException):
    pass


@GiskardException.register_error_code(MoveResult.EMPTY_PROBLEM)
class EmptyProblemException(InfeasibleException, DontPrintStackTrace):
    pass


# %% world state exceptions
@GiskardException.register_error_code(MoveResult.WORLD_ERROR)
class WorldException(GiskardException):
    pass


@GiskardException.register_error_code(MoveResult.UNKNOWN_GROUP)
class UnknownGroupException(WorldException, KeyError):
    pass


@GiskardException.register_error_code(MoveResult.UNKNOWN_LINK)
class UnknownLinkException(WorldException, KeyError):
    pass


@GiskardException.register_error_code(MoveResult.UNKNOWN_JOINT)
class UnknownJointException(WorldException, KeyError):
    pass


class UnsupportedOptionException(WorldException):
    pass


class CorruptShapeException(WorldException):
    pass


# %% error during motion problem building phase
@GiskardException.register_error_code(MoveResult.MOTION_PROBLEM_BUILDING_ERROR)
class MotionBuildingException(GiskardException):
    pass


@GiskardException.register_error_code(MoveResult.INVALID_GOAL)
class InvalidGoalException(MotionBuildingException):
    pass


@GiskardException.register_error_code(MoveResult.UNKNOWN_GOAL)
class UnknownGoalException(MotionBuildingException, KeyError):
    pass


@GiskardException.register_error_code(MoveResult.GOAL_INITIALIZATION_ERROR)
class GoalInitalizationException(MotionBuildingException):
    pass


@GiskardException.register_error_code(MoveResult.UNKNOWN_MONITOR)
class UnknownMonitorException(MotionBuildingException, KeyError):
    pass


@GiskardException.register_error_code(MoveResult.MONITOR_INITIALIZATION_ERROR)
class MonitorInitalizationException(MotionBuildingException):
    pass


# %% errors during planning
@GiskardException.register_error_code(MoveResult.CONTROL_ERROR)
class PlanningException(GiskardException):
    pass


@GiskardException.register_error_code(MoveResult.SHAKING)
class ShakingException(PlanningException):
    pass


@GiskardException.register_error_code(MoveResult.LOCAL_MINIMUM)
class LocalMinimumException(PlanningException):
    pass


@GiskardException.register_error_code(MoveResult.MAX_TRAJECTORY_LENGTH)
class MaxTrajectoryLengthException(PlanningException):
    pass


@GiskardException.register_error_code(MoveResult.SELF_COLLISION_VIOLATED)
class SelfCollisionViolatedException(PlanningException):
    pass


# errors during execution
@GiskardException.register_error_code(MoveResult.EXECUTION_ERROR)
class ExecutionException(GiskardException):
    pass


@GiskardException.register_error_code(MoveResult.FollowJointTrajectory_INVALID_GOAL)
class FollowJointTrajectory_INVALID_GOAL(ExecutionException):
    pass


@GiskardException.register_error_code(MoveResult.FollowJointTrajectory_INVALID_JOINTS)
class FollowJointTrajectory_INVALID_JOINTS(ExecutionException):
    pass


@GiskardException.register_error_code(MoveResult.FollowJointTrajectory_OLD_HEADER_TIMESTAMP)
class FollowJointTrajectory_OLD_HEADER_TIMESTAMP(ExecutionException):
    pass


@GiskardException.register_error_code(MoveResult.FollowJointTrajectory_PATH_TOLERANCE_VIOLATED)
class FollowJointTrajectory_PATH_TOLERANCE_VIOLATED(ExecutionException):
    pass


@GiskardException.register_error_code(MoveResult.FollowJointTrajectory_GOAL_TOLERANCE_VIOLATED)
class FollowJointTrajectory_GOAL_TOLERANCE_VIOLATED(ExecutionException):
    pass


@GiskardException.register_error_code(MoveResult.PREEMPTED)
class PreemptedException(ExecutionException):
    pass


@GiskardException.register_error_code(MoveResult.EXECUTION_PREEMPTED)
class ExecutionPreemptedException(ExecutionException):
    pass


@GiskardException.register_error_code(MoveResult.EXECUTION_TIMEOUT)
class ExecutionTimeoutException(ExecutionException):
    pass


@GiskardException.register_error_code(MoveResult.EXECUTION_SUCCEEDED_PREMATURELY)
class ExecutionSucceededPrematurely(ExecutionException):
    pass


# %% behavior tree exceptions
@GiskardException.register_error_code(MoveResult.BEHAVIOR_TREE_ERROR)
class BehaviorTreeException(GiskardException):
    pass
