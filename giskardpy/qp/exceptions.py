from giskardpy.data_types.exceptions import GiskardException, DontPrintStackTrace


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
    def __init__(self):
        super().__init__("Empty QP problem.")
