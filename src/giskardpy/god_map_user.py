from __future__ import annotations

from giskard_msgs.msg import MoveGoal
from giskardpy import identifier
from giskardpy.god_map import _GodMap
from typing import TYPE_CHECKING, List, Dict

from giskardpy.utils.utils import int_to_bit_list

if TYPE_CHECKING:
    from giskardpy.qp.qp_controller import QPProblemBuilder
    from giskardpy.configs.qp_controller_config import QPControllerConfig
    from giskardpy.my_types import Derivatives
    from giskardpy.goals.goal import Goal
    from giskardpy.configs.giskard import Giskard
    from giskardpy.goals.motion_goal_manager import MotionGoalManager
    from giskardpy.debug_expression_manager import DebugExpressionManager
    from giskardpy.goals.monitors.monitor_manager import MonitorManager
    from giskardpy.goals.monitors.monitors import Monitor
    from giskardpy.configs.collision_avoidance_config import CollisionAvoidanceConfig
    from giskardpy.configs.world_config import WorldConfig
    from giskardpy.tree.control_modes import ControlModes
    from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer, CollisionCheckerLib, \
        CollisionAvoidanceGroupThresholds
    from giskardpy.tree.garden import TreeManager
    from giskardpy.model.world import WorldTree


god_map = _GodMap()


class GodMap:
    god_map = _GodMap()

    @classmethod
    def get_goal_id(cls) -> int:
        if god_map.has_data(identifier.goal_id):
            return god_map.get_data(identifier.goal_id)
        else:
            return -1

    @classmethod
    def get_world(cls) -> WorldTree:
        return god_map.get_data(identifier.world)

    @classmethod
    def get_monitors(cls) -> List[Monitor]:
        return god_map.get_data(identifier.monitors, [])

    @classmethod
    def get_monitor_manager(cls) -> MonitorManager:
        return god_map.get_data(identifier.monitor_manager)

    @classmethod
    def get_motion_goal_manager(cls) -> MotionGoalManager:
        return god_map.get_data(identifier.motion_goal_manager)

    @classmethod
    def get_giskard(cls) -> Giskard:
        return god_map.get_data(identifier.giskard)

    @classmethod
    def get_debug_expression_manager(cls) -> DebugExpressionManager:
        return god_map.get_data(identifier.debug_expression_manager)

    @classmethod
    def get_tree_manager(cls) -> TreeManager:
        return god_map.get_data(identifier.tree_manager)

    @classmethod
    def get_collision_scene(cls) -> CollisionWorldSynchronizer:
        return god_map.get_data(identifier.collision_scene)

    @classmethod
    def get_prediction_horizon(cls) -> int:
        return god_map.get_data(identifier.prediction_horizon)

    @classmethod
    def get_max_derivative(cls) -> Derivatives:
        return god_map.get_data(identifier.max_derivative)

    @classmethod
    def get_qp_controller(cls) -> QPProblemBuilder:
        return god_map.get_data(identifier.qp_controller)

    @classmethod
    def get_qp_controller_config(cls) -> QPControllerConfig:
        return god_map.get_data(identifier.qp_controller_config)

    @classmethod
    def get_control_mode(cls) -> ControlModes:
        return god_map.get_data(identifier.control_mode)

    @classmethod
    def is_closed_loop(cls):
        return cls.get_control_mode() == cls.get_control_mode().close_loop

    @classmethod
    def is_standalone(cls):
        return cls.get_control_mode() == cls.get_control_mode().standalone

    @classmethod
    def is_open_loop(cls):
        return cls.get_control_mode() == cls.get_control_mode().open_loop

    @classmethod
    def get_collision_checker_id(cls) -> CollisionCheckerLib:
        return god_map.get_data(identifier.collision_checker)

    @classmethod
    def get_world_config(cls) -> WorldConfig:
        return god_map.get_data(identifier.world_config)

    @classmethod
    def get_collision_avoidance_config(cls) -> CollisionAvoidanceConfig:
        return god_map.get_data(identifier.collision_avoidance_config)

    @classmethod
    def get_collision_avoidance_configs(cls) -> Dict[str, CollisionAvoidanceGroupThresholds]:
        return god_map.unsafe_get_data(identifier.collision_avoidance_configs)

    @classmethod
    def get_trajectory_time_in_seconds(cls):
        time = god_map.get_data(identifier.time)
        if cls.is_closed_loop():
            return time
        return time * cls.get_sample_period()

    @classmethod
    def get_goal_msg_type(cls) -> int:
        return cls.get_goal_msg().type

    @classmethod
    def get_sample_period(cls) -> float:
        return god_map.get_data(identifier.sample_period)

    @classmethod
    def get_goal_msg(cls) -> MoveGoal:
        return god_map.get_data(identifier.goal_msg)

    @classmethod
    def get_motion_goals(cls) -> Dict[str, Goal]:
        return god_map.get_data(identifier.motion_goals)

    @classmethod
    def is_goal_msg_type_execute(cls):
        return MoveGoal.EXECUTE in int_to_bit_list(cls.get_goal_msg_type())

    @classmethod
    def is_goal_msg_type_projection(cls):
        return MoveGoal.PROJECTION in int_to_bit_list(cls.get_goal_msg_type())

    @classmethod
    def is_goal_msg_local_minimum_is_success(cls):
        return cls.get_goal_msg().local_minimum_is_success

    @classmethod
    def is_goal_msg_type_undefined(cls):
        return MoveGoal.UNDEFINED in int_to_bit_list(cls.get_goal_msg_type())
