from __future__ import annotations
from typing import TypeVar, Union, Type, TYPE_CHECKING

import py_trees
from py_trees import Composite

from giskardpy.tree.behaviors.plugin import GiskardBehavior

T = TypeVar('T', bound=Union[Type[GiskardBehavior], Type[Composite]])


def running_is_success(cls: T) -> T:
    return py_trees.meta.running_is_success(cls)


def success_is_failure(cls: T) -> T:
    return py_trees.meta.success_is_failure(cls)


def failure_is_success(cls: T) -> T:
    return py_trees.meta.failure_is_success(cls)


def running_is_failure(cls: T) -> T:
    return py_trees.meta.running_is_failure(cls)


def failure_is_running(cls: T) -> T:
    return py_trees.meta.failure_is_running(cls)


def success_is_running(cls: T) -> T:
    return py_trees.meta.success_is_running(cls)


def anything_is_success(cls: T) -> T:
    return running_is_success(failure_is_success(cls))


def anything_is_failure(cls: T) -> T:
    return running_is_failure(success_is_failure(cls))