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
from typing import Type, Optional, Dict

import numpy as np
import roslaunch
import rospkg
import rospy
import trimesh
from genpy import Message
from geometry_msgs.msg import PointStamped, Point, Vector3Stamped, Vector3, Pose, PoseStamped, QuaternionStamped, \
    Quaternion
from py_trees import Status, Blackboard
from rospy_message_converter.message_converter import \
    convert_ros_message_to_dictionary as original_convert_ros_message_to_dictionary, \
    convert_dictionary_to_ros_message as original_convert_dictionary_to_ros_message
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray

from giskardpy import identifier
from giskardpy.exceptions import DontPrintStackTrace
from giskardpy.god_map import GodMap
from giskardpy.my_types import PrefixName
from giskardpy.utils import logging
from giskardpy.utils.time_collector import TimeCollector
from giskardpy.utils.utils import has_blackboard_exception, raise_to_blackboard


def memoize(function):
    memo = function.memo = {}

    @wraps(function)
    def wrapper(*args, **kwargs):
        key = (args, frozenset(kwargs.items()))
        try:
            return memo[key]
        except KeyError:
            rv = function(*args, **kwargs)
            memo[key] = rv
            return rv

    return wrapper


def memoize_with_counter(reset_after: int):
    def memoize(function):
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


def record_time(function):
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


def copy_memoize(function):
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


def catch_and_raise_to_blackboard(function):
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
