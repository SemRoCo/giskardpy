try:
    # Python 2
    import __builtin__ as builtins
except ImportError:
    # Python 3
    import builtins

try:
    builtins.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func
    builtins.profile = profile

DEBUG = 0
NORMAL = 1
ERROR = 2
PRINT_LEVEL = NORMAL

MAP = u'map'
ROBOTNAME = 'robot'

# import casadi_wrapper as cas_wrapper


WORLD_IMPLEMENTATION = u'pybullet'