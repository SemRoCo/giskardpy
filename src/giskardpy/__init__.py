# DEBUG = 0
# NORMAL = 1
# ERROR = 2
# # TODO figure out how to do this print level shit properly
# PRINT_LEVEL = NORMAL


from giskardpy.pybullet_world import PyBulletWorldObj

# BACKEND = None
# BACKEND = 'cse'
# BACKEND = 'numpy'
# BACKEND = 'cython'

BACKEND = 'llvm'
# BACKEND = 'lambda'

WorldObjImpl = PyBulletWorldObj

