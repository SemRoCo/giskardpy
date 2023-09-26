from __future__ import annotations

from giskard_msgs.msg import MoveGoal
from giskardpy import identifier
from giskardpy.god_map import GodMap
from typing import TYPE_CHECKING, List, Dict

from giskardpy.utils.utils import int_to_bit_list

if TYPE_CHECKING:
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


class GodMapWorshipper:
    god_map = GodMap()

    @property
    def goal_id(self) -> int:
        if self.god_map.has_data(identifier.goal_id):
            return self.god_map.get_data(identifier.goal_id)
        else:
            return -1

    @property
    def world(self) -> WorldTree:
        return self.god_map.get_data(identifier.world)

    @property
    def monitors(self) -> List[Monitor]:
        return self.god_map.get_data(identifier.monitors, [])

    @property
    def monitor_manager(self) -> MonitorManager:
        return self.god_map.get_data(identifier.monitor_manager)

    @property
    def motion_goal_manager(self) -> MotionGoalManager:
        return self.god_map.get_data(identifier.motion_goal_manager)

    @property
    def debug_expression_manager(self) -> DebugExpressionManager:
        return self.god_map.get_data(identifier.debug_expression_manager, [])

    @property
    def tree_manager(self) -> TreeManager:
        return self.god_map.get_data(identifier.tree_manager)

    @property
    def collision_scene(self) -> CollisionWorldSynchronizer:
        return self.god_map.get_data(identifier.collision_scene)

    @property
    def control_mode(self) -> ControlModes:
        return self.god_map.get_data(identifier.control_mode)

    @property
    def is_closed_loop(self):
        return self.control_mode == self.control_mode.close_loop

    @property
    def is_standalone(self):
        return self.control_mode == self.control_mode.standalone

    @property
    def is_open_loop(self):
        return self.control_mode == self.control_mode.open_loop

    @property
    def collision_checker_id(self) -> CollisionCheckerLib:
        return self.god_map.get_data(identifier.collision_checker)

    @property
    def world_config(self) -> WorldConfig:
        return self.god_map.get_data(identifier.world_config)

    @property
    def collision_avoidance_config(self) -> CollisionAvoidanceConfig:
        return self.god_map.get_data(identifier.collision_avoidance_config)

    @property
    def collision_avoidance_configs(self) -> Dict[str, CollisionAvoidanceGroupThresholds]:
        return self.god_map.unsafe_get_data(identifier.collision_avoidance_configs)

    @property
    def trajectory_time_in_seconds(self):
        time = self.god_map.get_data(identifier.time)
        if self.is_closed_loop:
            return time
        return time * self.sample_period

    @property
    def goal_msg_type(self) -> int:
        return self.goal_msg.type

    @property
    def sample_period(self) -> float:
        return self.god_map.get_data(identifier.sample_period)

    @property
    def goal_msg(self) -> MoveGoal:
        return self.god_map.get_data(identifier.goal_msg)

    def is_goal_msg_type_execute(self):
        return MoveGoal.EXECUTE in int_to_bit_list(self.goal_msg_type)

    def is_goal_msg_type_projection(self):
        return MoveGoal.PROJECTION in int_to_bit_list(self.goal_msg_type)

    def is_goal_msg_local_minimum_is_success(self):
        return self.goal_msg.local_minimum_is_success

    def is_goal_msg_type_undefined(self):
        return MoveGoal.UNDEFINED in int_to_bit_list(self.goal_msg_type)
