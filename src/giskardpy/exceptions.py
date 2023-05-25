from typing import Optional


class DontPrintStackTrace:
    pass

class GiskardException(Exception):
    pass


# solver exceptions-----------------------------------------------------------------------------------------------------
# int64 QP_SOLVER_ERROR=5 # if no solver code fits
class QPSolverException(GiskardException):

    def __init__(self, error_message: str, error_code: Optional[int] = None):
        super().__init__(error_message)
        self.error_code = error_code


class InfeasibleException(QPSolverException):
    pass


class VelocityLimitUnreachableException(QPSolverException):
    pass


class OutOfJointLimitsException(InfeasibleException):
    pass


class HardConstraintsViolatedException(InfeasibleException):
    pass


class EmptyProblemException(InfeasibleException, DontPrintStackTrace):
    pass


# world state exceptions------------------------------------------------------------------------------------------------
# int64 WORLD_ERROR=7 # if no world error fits
class PhysicsWorldException(GiskardException):
    pass


# int64 UNKNOWN_OBJECT=6
class UnknownGroupException(PhysicsWorldException, KeyError):
    pass


class UnknownLinkException(PhysicsWorldException, KeyError):
    pass


class RobotExistsException(PhysicsWorldException):
    pass


class DuplicateNameException(PhysicsWorldException):
    pass


class UnsupportedOptionException(PhysicsWorldException):
    pass


class CorruptShapeException(PhysicsWorldException):
    pass


# error during motion problem building phase----------------------------------------------------------------------------
# int64 CONSTRAINT_ERROR # if no constraint code fits
class ConstraintException(GiskardException):
    pass


# int64 UNKNOWN_CONSTRAINT
class UnknownConstraintException(ConstraintException, KeyError):
    pass


# int64 CONSTRAINT_INITIALIZATION_ERROR
class ConstraintInitalizationException(ConstraintException):
    pass


# int64 INVALID_GOAL
class InvalidGoalException(ConstraintException):
    pass


# errors during planning------------------------------------------------------------------------------------------------
# int64 PLANNING_ERROR=13 # if no planning code fits
class PlanningException(GiskardException):
    pass


# int64 SHAKING # Planning was stopped because the trajectory contains a shaky velocity profile. Detection parameters can be tuned in config file
class ShakingException(PlanningException):
    pass


# int64 UNREACHABLE # if reachability check fails
class UnreachableException(PlanningException):
    pass


class SelfCollisionViolatedException(PlanningException):
    pass

# errors during execution
# int64 EXECUTION_ERROR # if no execution code fits
# int64 PREEMPTED # goal got canceled via action server interface
class ExecutionException(GiskardException):
    pass


class FollowJointTrajectory_INVALID_GOAL(ExecutionException):
    pass


class FollowJointTrajectory_INVALID_JOINTS(ExecutionException):
    pass


class FollowJointTrajectory_OLD_HEADER_TIMESTAMP(ExecutionException):
    pass


class FollowJointTrajectory_PATH_TOLERANCE_VIOLATED(ExecutionException):
    pass


class FollowJointTrajectory_GOAL_TOLERANCE_VIOLATED(ExecutionException):
    pass


class PreemptedException(ExecutionException):
    pass


class ExecutionPreemptedException(ExecutionException):
    pass


class ExecutionTimeoutException(ExecutionException):
    pass


class ExecutionSucceededPrematurely(ExecutionException):
    pass


# -----------------------------------------------------------------------------------------------------------------------
class ImplementationException(GiskardException):
    pass
