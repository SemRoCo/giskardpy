import builtins

try:
    builtins.profile  #type: ignore
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func
    builtins.profile = profile #type: ignore

DEBUG = 0
NORMAL = 1
ERROR = 2
PRINT_LEVEL = NORMAL

RobotPrefix = None