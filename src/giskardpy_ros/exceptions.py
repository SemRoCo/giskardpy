from giskard_msgs.msg import GiskardError


class DontPrintStackTrace:
    pass


class GiskardException(Exception):
    error_code: int = GiskardError.ERROR
    error_message: str = ''
    _error_code_map = {}

    def __init__(self, error_message: str):
        super().__init__(error_message)
        self.error_message = error_message

    def to_error_msg(self) -> GiskardError:
        error = GiskardError()
        error.code = self.error_code
        error.msg = self.error_message
        return error

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


GiskardException._error_code_map[GiskardError.ERROR] = GiskardException


@GiskardException.register_error_code(GiskardError.SETUP_ERROR)
class SetupException(GiskardException):
    pass


@GiskardException.register_error_code(GiskardError.DUPLICATE_NAME)
class DuplicateNameException(GiskardException):
    pass


# %% solver exceptions
@GiskardException.register_error_code(GiskardError.QP_SOLVER_ERROR)
class QPSolverException(GiskardException):
    pass


@GiskardException.register_error_code(GiskardError.INFEASIBLE)
class InfeasibleException(QPSolverException):
    pass


@GiskardException.register_error_code(GiskardError.VELOCITY_LIMIT_UNREACHABLE)
class VelocityLimitUnreachableException(QPSolverException):
    pass


@GiskardException.register_error_code(GiskardError.OUT_OF_JOINT_LIMITS)
class OutOfJointLimitsException(InfeasibleException):
    pass


@GiskardException.register_error_code(GiskardError.HARD_CONSTRAINTS_VIOLATED)
class HardConstraintsViolatedException(InfeasibleException):
    pass


@GiskardException.register_error_code(GiskardError.EMPTY_PROBLEM)
class EmptyProblemException(InfeasibleException, DontPrintStackTrace):
    pass


# %% world exceptions
@GiskardException.register_error_code(GiskardError.WORLD_ERROR)
class WorldException(GiskardException):
    pass


@GiskardException.register_error_code(GiskardError.UNKNOWN_GROUP)
class UnknownGroupException(WorldException, KeyError):
    pass


@GiskardException.register_error_code(GiskardError.UNKNOWN_LINK)
class UnknownLinkException(WorldException, KeyError):
    pass


@GiskardException.register_error_code(GiskardError.UNKNOWN_JOINT)
class UnknownJointException(WorldException, KeyError):
    pass


@GiskardException.register_error_code(GiskardError.INVALID_WORLD_OPERATION)
class InvalidWorldOperationException(WorldException, KeyError):
    pass


@GiskardException.register_error_code(GiskardError.CORRUPT_SHAPE)
class CorruptShapeException(WorldException):
    pass


@GiskardException.register_error_code(GiskardError.CORRUPT_MESH)
class CorruptMeshException(CorruptShapeException):
    pass


@GiskardException.register_error_code(GiskardError.CORRUPT_URDF)
class CorruptURDFException(CorruptShapeException):
    pass


@GiskardException.register_error_code(GiskardError.TRANSFORM_ERROR)
class TransformException(WorldException):
    pass


# %% error during motion problem building phase
@GiskardException.register_error_code(GiskardError.MOTION_PROBLEM_BUILDING_ERROR)
class MotionBuildingException(GiskardException):
    pass


@GiskardException.register_error_code(GiskardError.INVALID_GOAL)
class InvalidGoalException(MotionBuildingException):
    pass


@GiskardException.register_error_code(GiskardError.UNKNOWN_GOAL)
class UnknownGoalException(MotionBuildingException, KeyError):
    pass


@GiskardException.register_error_code(GiskardError.GOAL_INITIALIZATION_ERROR)
class GoalInitalizationException(MotionBuildingException):
    pass


@GiskardException.register_error_code(GiskardError.UNKNOWN_MONITOR)
class UnknownMonitorException(MotionBuildingException, KeyError):
    pass


@GiskardException.register_error_code(GiskardError.MONITOR_INITIALIZATION_ERROR)
class MonitorInitalizationException(MotionBuildingException):
    pass


# %% errors during planning
@GiskardException.register_error_code(GiskardError.CONTROL_ERROR)
class PlanningException(GiskardException):
    pass


@GiskardException.register_error_code(GiskardError.SHAKING)
class ShakingException(PlanningException):
    pass


@GiskardException.register_error_code(GiskardError.LOCAL_MINIMUM)
class LocalMinimumException(PlanningException):
    pass


@GiskardException.register_error_code(GiskardError.MAX_TRAJECTORY_LENGTH)
class MaxTrajectoryLengthException(PlanningException):
    pass


@GiskardException.register_error_code(GiskardError.SELF_COLLISION_VIOLATED)
class SelfCollisionViolatedException(PlanningException):
    pass


# errors during execution
@GiskardException.register_error_code(GiskardError.EXECUTION_ERROR)
class ExecutionException(GiskardException):
    pass


@GiskardException.register_error_code(GiskardError.FollowJointTrajectory_INVALID_GOAL)
class FollowJointTrajectory_INVALID_GOAL(ExecutionException):
    pass


@GiskardException.register_error_code(GiskardError.FollowJointTrajectory_INVALID_JOINTS)
class FollowJointTrajectory_INVALID_JOINTS(ExecutionException):
    pass


@GiskardException.register_error_code(GiskardError.FollowJointTrajectory_OLD_HEADER_TIMESTAMP)
class FollowJointTrajectory_OLD_HEADER_TIMESTAMP(ExecutionException):
    pass


@GiskardException.register_error_code(GiskardError.FollowJointTrajectory_PATH_TOLERANCE_VIOLATED)
class FollowJointTrajectory_PATH_TOLERANCE_VIOLATED(ExecutionException):
    pass


@GiskardException.register_error_code(GiskardError.FollowJointTrajectory_GOAL_TOLERANCE_VIOLATED)
class FollowJointTrajectory_GOAL_TOLERANCE_VIOLATED(ExecutionException):
    pass


@GiskardException.register_error_code(GiskardError.PREEMPTED)
class PreemptedException(ExecutionException):
    pass


@GiskardException.register_error_code(GiskardError.EXECUTION_PREEMPTED)
class ExecutionPreemptedException(ExecutionException):
    pass


@GiskardException.register_error_code(GiskardError.EXECUTION_TIMEOUT)
class ExecutionTimeoutException(ExecutionException):
    pass


@GiskardException.register_error_code(GiskardError.EXECUTION_SUCCEEDED_PREMATURELY)
class ExecutionSucceededPrematurely(ExecutionException):
    pass


# %% behavior tree exceptions
@GiskardException.register_error_code(GiskardError.BEHAVIOR_TREE_ERROR)
class BehaviorTreeException(GiskardException):
    pass


# %% force torque exceptions
@GiskardException.register_error_code(GiskardError.FORCE_TORQUE_MONITOR_ERROR)
class ForceTorqueExceptions(GiskardException):
    pass


@GiskardException.register_error_code(GiskardError.FORCE_TORQUE_MONITOR_GRASPING_MISSED_OBJECT)
class ForceTorqueMonitorGraspsingMissedObjectExceptions(GiskardException):
    pass


@GiskardException.register_error_code(GiskardError.FORCE_TORQUE_MONITOR_TRANSPORTING_LOST_OBJECT)
class ForceTorqueTransportingLostObjectExceptions(GiskardException):
    pass


@GiskardException.register_error_code(GiskardError.FORCE_TORQUE_MONITOR_PLACING_MISSED_PLACING_LOCATION)
class ForceTorquePlacingMissedPlacingLocationExceptions(GiskardException):
    pass
