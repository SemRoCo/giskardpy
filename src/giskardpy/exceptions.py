class GiskardException(Exception):
    pass

class NameConflictException(GiskardException):
    pass

class QPSolverException(GiskardException):
    pass

class MAX_NWSR_REACHEDException(QPSolverException):
    pass

class WorldException(GiskardException):
    pass

class UpdateWorldException(GiskardException):
    pass

class CorruptShapeException(GiskardException):
    pass

class UnknownBodyException(UpdateWorldException):
    pass

class DuplicateBodyNameException(UpdateWorldException):
    pass

class DuplicateRobotNameException(DuplicateBodyNameException):
    pass

class DuplicateObjectNameException(DuplicateBodyNameException):
    pass

