class DontPrintStackTrace:
    pass


class GiskardException(Exception):
    pass


class SetupException(GiskardException):
    pass


class DuplicateNameException(GiskardException):
    pass


# %% solver exceptions
class QPSolverException(GiskardException):
    pass


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


# %% world exceptions
class WorldException(GiskardException):
    pass


class UnknownGroupException(WorldException, KeyError):
    pass


class UnknownLinkException(WorldException, KeyError):
    pass


class UnknownJointException(WorldException, KeyError):
    pass


class InvalidWorldOperationException(WorldException, KeyError):
    pass


class CorruptShapeException(WorldException):
    pass


class CorruptMeshException(CorruptShapeException):
    pass


class CorruptURDFException(CorruptShapeException):
    pass


class TransformException(WorldException):
    pass


# %% error during motion problem building phase
class MotionBuildingException(GiskardException):
    pass


class InvalidGoalException(MotionBuildingException):
    pass


class UnknownGoalException(MotionBuildingException, KeyError):
    pass


class GoalInitalizationException(MotionBuildingException):
    pass


class UnknownMonitorException(MotionBuildingException, KeyError):
    pass


class MonitorInitalizationException(MotionBuildingException):
    pass


class UnknownTaskException(MotionBuildingException, KeyError):
    pass


class TaskInitalizationException(MotionBuildingException):
    pass


# %% errors during planning
class PlanningException(GiskardException):
    pass


class ShakingException(PlanningException):
    pass


class LocalMinimumException(PlanningException):
    pass


class MaxTrajectoryLengthException(PlanningException):
    pass


class SelfCollisionViolatedException(PlanningException):
    pass


# errors during execution
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


# %% behavior tree exceptions
class BehaviorTreeException(GiskardException):
    pass
