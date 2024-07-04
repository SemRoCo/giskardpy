import builtins

import giskardpy
from giskardpy_ros.ros1.interface import ROS1Wrapper

try:
    builtins.profile  # type: ignore
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func):
        return func


    builtins.profile = profile  # type: ignore

giskardpy.middleware.middleware = ROS1Wrapper()