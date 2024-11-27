from __future__ import annotations
import traceback
from functools import wraps
from typing import TypeVar, Callable, TYPE_CHECKING

from py_trees import Blackboard, Status

from giskardpy.data_types.exceptions import DontPrintStackTrace

if TYPE_CHECKING:
    from giskardpy_ros.configs.giskard import Giskard
    from giskardpy_ros.ros1.ros_msg_visualization import ROSMsgVisualization
    from giskardpy_ros.tree.behaviors.action_server import ActionServerHandler
    from giskardpy_ros.tree.branches.giskard_bt import GiskardBT

blackboard_exception_name = 'exception'


class GiskardBlackboard(Blackboard):
    giskard: Giskard
    tree: GiskardBT
    runtime: float
    move_action_server: ActionServerHandler
    world_action_server: ActionServerHandler
    ros_visualizer: ROSMsgVisualization
    fill_trajectory_velocity_values: bool
    control_loop_max_hz: float
    simulation_max_hz: float


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
