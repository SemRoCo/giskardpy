class GiskardException(Exception):
    pass

class ImplementationException(GiskardException):
    pass

# -----------------------------------------------------------------------------------------------------------------------
class InsolvableException(GiskardException):
    pass

class UnreachableException(GiskardException):
    pass

# -----------------------------------------------------------------------------------------------------------------------
class PhysicsWorldException(GiskardException):
    pass


class UnknownBodyException(PhysicsWorldException):
    pass


class RobotExistsException(PhysicsWorldException):
    pass


class DuplicateNameException(PhysicsWorldException):
    pass


class UnsupportedOptionException(PhysicsWorldException):
    pass


class StartStateCollisionException(PhysicsWorldException):
    pass


class PathCollisionException(PhysicsWorldException):
    pass


class CorruptShapeException(PhysicsWorldException):
    pass


# -----------------------------------------------------------------------------------------------------------------------
class SymengineException(GiskardException):
    pass


# -----------------------------------------------------------------------------------------------------------------------
class QPSolverException(GiskardException):
    pass


class SolverTimeoutError(QPSolverException):
    pass


class MAX_NWSR_REACHEDException(QPSolverException):
    pass

# -----------------------------------------------------------------------------------------------------------------------
class ConstraintException(GiskardException):
    pass