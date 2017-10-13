from sympy.vector import *

giskard_base_Frame = CoordSys3D('N')
unitX = giskard_base_Frame.i
unitY = giskard_base_Frame.j
unitZ = giskard_base_Frame.k

def vec3(x, y, z):
    return unitX * x + unitY * y + unitZ * z


