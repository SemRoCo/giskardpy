from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Dict, Tuple

if TYPE_CHECKING:
    from giskardpy.model.world import WorldTree
    from giskardpy.qp.free_variable import FreeVariable
    from giskardpy.qp.next_command import NextCommands
    from giskardpy.model.trajectory import Trajectory
    from giskardpy.qp.qp_controller import QPController
    from giskardpy.data_types.data_types import PrefixName
    from giskardpy.configs.giskard import Giskard
    from giskardpy.monitors.monitor_manager import MonitorManager
    from giskardpy.goals.motion_goal_manager import MotionGoalManager
    from giskardpy.debug_expression_manager import DebugExpressionManager
    from giskardpy.configs.collision_avoidance_config import CollisionAvoidanceConfig
    from giskardpy.configs.world_config import WorldConfig
    from giskardpy.model.collision_world_syncer import (CollisionWorldSynchronizer, CollisionAvoidanceGroupThresholds,
                                                        Collisions)


class GodMap:
    # %% important objects
    world: WorldTree
    collision_scene: CollisionWorldSynchronizer
    qp_controller: QPController

    # %% managers
    monitor_manager: MonitorManager
    motion_goal_manager: MotionGoalManager
    debug_expression_manager: DebugExpressionManager

    # %% controller datatypes
    time: float  # real/planning time in s
    control_cycle_counter: int
    trajectory: Trajectory
    qp_solver_solution: NextCommands
    added_collision_checks: Dict[Tuple[PrefixName, PrefixName], float]
    closest_point: Collisions
    motion_start_time: float
    hack: float
    free_variables: List[FreeVariable]

    # %% configs
    giskard: Giskard
    collision_avoidance_config: CollisionAvoidanceConfig
    collision_avoidance_configs: Dict[str, CollisionAvoidanceGroupThresholds]

    __initialized = False

    def initialize(self):
        # can't use __init__ because it creates a circular import exception
        if not self.__initialized:
            from giskardpy.model.world import WorldTree
            self.world = WorldTree()

            from giskardpy.goals.motion_goal_manager import MotionGoalManager
            self.motion_goal_manager = MotionGoalManager()

            from giskardpy.monitors.monitor_manager import MonitorManager
            self.monitor_manager = MonitorManager()

            from giskardpy.debug_expression_manager import DebugExpressionManager
            self.debug_expression_manager = DebugExpressionManager()

            self.__initialized = True

    def __getattr__(self, item):
        # automatically initialize self, when an attribute isn't found
        self.initialize()
        return super().__getattribute__(item)

    def is_collision_checking_enabled(self):
        return self.collision_scene.collision_checker_id != self.collision_scene.collision_checker_id.none

    @staticmethod
    def is_in_github_workflow():
        return 'GITHUB_WORKFLOW' in os.environ


god_map = GodMap()
