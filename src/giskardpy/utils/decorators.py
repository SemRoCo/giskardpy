from __future__ import division

# I only do this, because otherwise test/test_integration_pr2.py::TestWorldManipulation::test_unsupported_options
# fails on github actions
import urdf_parser_py.urdf as up

import errno
import inspect
import json
import os
import pkgutil
import sys
import traceback
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from time import time
from typing import Type, Optional, Dict, TypeVar, Callable

import numpy as np
import roslaunch
import rospkg
import rospy
import trimesh
from genpy import Message
from geometry_msgs.msg import PointStamped, Point, Vector3Stamped, Vector3, Pose, PoseStamped, QuaternionStamped, \
    Quaternion
from py_trees import Status, Blackboard

from giskardpy.exceptions import DontPrintStackTrace
from giskardpy.utils.utils import has_blackboard_exception, raise_to_blackboard

from functools import wraps
from typing import Any, TypeVar

T = TypeVar("T", bound=Callable)


def memoize(function: T) -> T:
    memo = function.memo = {}

    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        key = (args, frozenset(kwargs.items()))
        try:
            return memo[key]
        except KeyError:
            rv = function(*args, **kwargs)
            memo[key] = rv
            return rv

    return wrapper  # type: ignore


def memoize_with_counter(reset_after: int):
    def memoize(function: T) -> T:
        memo = function.memo = {}
        function.__counter = 0

        @wraps(function)
        def wrapper(*args, **kwargs):
            key = (args, frozenset(kwargs.items()))
            try:
                hit = memo[key]
                if function.__counter >= reset_after:
                    raise KeyError
                else:
                    function.__counter += 1
                    return hit
            except KeyError:
                function.__counter = 1
                rv = function(*args, **kwargs)
                memo[key] = rv
                return rv

        return wrapper

    return memoize


def record_time(function: T) -> T:
    # return function
    function_name = function.__name__

    @wraps(function)
    def wrapper(*args, **kwargs):
        self = args[0]
        if not hasattr(self, '__times'):
            setattr(self, '__times', defaultdict(list))
        start_time = time()
        result = function(*args, **kwargs)
        time_delta = time() - start_time
        self.__times[function_name].append(time_delta)
        return result

    return wrapper


def clear_memo(f):
    if hasattr(f, 'memo'):
        f.memo.clear()


def copy_memoize(function: T) -> T:
    memo = function.memo = {}

    @wraps(function)
    def wrapper(*args, **kwargs):
        key = (args, frozenset(kwargs.items()))
        try:
            return deepcopy(memo[key])
        except KeyError:
            rv = function(*args, **kwargs)
            memo[key] = rv
            return deepcopy(rv)

    return wrapper


def catch_and_raise_to_blackboard(function: T) -> T:
    @wraps(function)
    def wrapper(*args, **kwargs):
        if has_blackboard_exception():
            return Status.FAILURE
        try:
            r = function(*args, **kwargs)
        except Exception as e:
            if not isinstance(e, DontPrintStackTrace):
                traceback.print_exc()
            raise_to_blackboard(e)
            return Status.FAILURE
        return r

    return wrapper


# %% these two decorators automatically add a state variable to an object that prevents multiple calls for off on pairs
def toggle_on(state_var: str):
    def decorator(func: T) -> T:
        def wrapper(self, *args, **kwargs) -> T:
            if getattr(self, state_var, False):
                return
            setattr(self, state_var, True)
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


def toggle_off(state_var: str):
    def decorator(func: T) -> T:
        def wrapper(self, *args, **kwargs) -> T:
            if not getattr(self, state_var, True):
                return
            setattr(self, state_var, False)
            return func(self, *args, **kwargs)

        return wrapper

    return decorator
