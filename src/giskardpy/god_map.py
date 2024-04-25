from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Dict, Tuple

if TYPE_CHECKING:
    from giskardpy.configs.behavior_tree_config import BehaviorTreeConfig
    from giskardpy.qp.free_variable import FreeVariable
    from giskardpy.qp.next_command import NextCommands
    from giskardpy.model.trajectory import Trajectory
    from giskardpy.qp.qp_controller import QPProblemBuilder
    from giskardpy.configs.qp_controller_config import QPControllerConfig
    from giskardpy.data_types.data_types import PrefixName
    from giskardpy.configs.giskard import Giskard
    from giskardpy.goals.motion_goal_manager import MotionGoalManager
    from giskardpy.debug_expression_manager import DebugExpressionManager
    from giskardpy.monitors.monitor_manager import MonitorManager
    from giskardpy.configs.collision_avoidance_config import CollisionAvoidanceConfig
    from giskardpy.configs.world_config import WorldConfig
    from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer, CollisionCheckerLib, \
        CollisionAvoidanceGroupThresholds, Collisions
    from giskardpy.model.world import WorldTree


class GodMap:
    monitor_manager: MonitorManager
    giskard: Giskard
    time: float  # real/planning time in s
    control_cycle_counter: int
    world: WorldTree
    motion_goal_manager: MotionGoalManager
    debug_expression_manager: DebugExpressionManager
    collision_scene: CollisionWorldSynchronizer
    collision_checker_id: CollisionCheckerLib
    qp_controller: QPProblemBuilder
    world_config: WorldConfig
    qp_controller_config: QPControllerConfig
    behavior_tree_config: BehaviorTreeConfig
    collision_avoidance_config: CollisionAvoidanceConfig
    collision_avoidance_configs: Dict[str, CollisionAvoidanceGroupThresholds]
    trajectory: Trajectory
    qp_solver_solution: NextCommands
    added_collision_checks: Dict[Tuple[PrefixName, PrefixName], float]
    closest_point: Collisions
    motion_start_time: float
    hack: float
    free_variables: List[FreeVariable]

    def is_collision_checking_enabled(self):
        return self.collision_scene.collision_checker_id != self.collision_scene.collision_checker_id.none

    @staticmethod
    def is_in_github_workflow():
        return 'GITHUB_WORKFLOW' in os.environ


god_map = GodMap()
