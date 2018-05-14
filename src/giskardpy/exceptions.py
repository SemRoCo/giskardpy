class GiskardException(Exception):
    pass

class NameConflictException(GiskardException):
    pass

class QPSolverException(GiskardException):
    pass

class MAX_NWSR_REACHEDException(QPSolverException):
    pass

