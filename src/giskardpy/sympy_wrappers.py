from sympy import ImmutableMatrix
from sympy import Matrix


def Vector(x,y,z):
    return ImmutableMatrix([x,y,z])


unitX = Vector(1,0,0)
unitY = Vector(0,1,0)
unitZ = Vector(0,0,1)
