from __future__ import division

# I only do this, because otherwise test/test_integration_pr2.py::TestWorldManipulation::test_unsupported_options
# fails on github actions

import traceback
from collections import defaultdict
from copy import deepcopy
from time import time
from typing import Callable

from py_trees import Status

from giskardpy.data_types.exceptions import DontPrintStackTrace

from functools import wraps
from typing import Any, TypeVar

from giskardpy.tree.blackboard_utils import has_blackboard_exception, raise_to_blackboard

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
