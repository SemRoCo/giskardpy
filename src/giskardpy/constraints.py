from collections import namedtuple

ControllableConstraint = namedtuple('ControllableConstraint', ['lb', 'ub', 'expression'])
HardConstraint = namedtuple('HardConstraint', ['lbA', 'ubA', 'expression'])
SoftConstraint = namedtuple('SoftConstraint', ['lb', 'ub', 'lbA', 'ubA', 'expression'])
