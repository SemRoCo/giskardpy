from __future__ import annotations
import traceback
from functools import wraps
from typing import TypeVar, Callable, TYPE_CHECKING

from py_trees import Blackboard, Status

from giskardpy.data_types.exceptions import DontPrintStackTrace

if TYPE_CHECKING:
    from giskardpy.tree.branches.giskard_bt import GiskardBT

blackboard_exception_name = 'exception'


class GiskardBlackboard(Blackboard):
    tree: GiskardBT
    runtime: float


def raise_to_blackboard(exception):
    GiskardBlackboard().set(blackboard_exception_name, exception)


def has_blackboard_exception():
    return hasattr(GiskardBlackboard(), blackboard_exception_name) \
        and getattr(GiskardBlackboard(), blackboard_exception_name) is not None


def get_blackboard_exception():
    return GiskardBlackboard().get(blackboard_exception_name)


def clear_blackboard_exception():
    raise_to_blackboard(None)


T = TypeVar("T", bound=Callable)


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
