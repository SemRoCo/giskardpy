DEBUG = 0
NORMAL = 1
ERROR = 2
# TODO figure out how to do this print level shit properly
PRINT_LEVEL = NORMAL



# BACKEND = None
# BACKEND = 'cse'
# BACKEND = 'numpy'
# BACKEND = 'cython'
from giskardpy.pybullet_world_object import PyBulletWorldObj

BACKEND = 'llvm'
# BACKEND = 'lambda'

WorldObjImpl = PyBulletWorldObj

