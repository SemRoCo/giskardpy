class DontPrintStackTrace:
    pass


class GiskardException(Exception):
    pass


class SetupException(GiskardException):
    pass


class DuplicateNameException(GiskardException):
    pass


class NoQPControllerConfigException(SetupException):
    def __init__(
        self,
        message: str = "Motion Statechart has constraints, no QP controller config is provided.",
    ):
        super().__init__(message)


# %% errors during planning
class PlanningException(GiskardException):
    pass


class MaxTrajectoryLengthException(PlanningException):
    pass


class SelfCollisionViolatedException(PlanningException):
    pass
