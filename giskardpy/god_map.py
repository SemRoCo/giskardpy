from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Dict, Tuple

from giskardpy.middleware import get_middleware
from giskardpy.utils.utils import create_path

if TYPE_CHECKING:
    from giskardpy.model.world import WorldTree
    from giskardpy.qp.free_variable import FreeVariable
    from giskardpy.qp.next_command import NextCommands
    from giskardpy.model.trajectory import Trajectory
    from giskardpy.qp.qp_controller import QPController
    from giskardpy.data_types.data_types import PrefixName
    from giskardpy.motion_graph.monitors.monitor_manager import MonitorManager
    from giskardpy.goals.motion_goal_manager import MotionGoalManager
    from giskardpy.debug_expression_manager import DebugExpressionManager
    from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer, Collisions


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

    # %% other
    tmp_folder: str

    def __getattr__(self, item):
        # automatically initialize certain attributes
        if item == 'world':
            from giskardpy.model.world import WorldTree
            self.world = WorldTree()
        elif item == 'motion_goal_manager':
            from giskardpy.goals.motion_goal_manager import MotionGoalManager
            self.motion_goal_manager = MotionGoalManager()
        elif item == 'monitor_manager':
            from giskardpy.motion_graph.monitors.monitor_manager import MonitorManager
            self.monitor_manager = MonitorManager()
        elif item == 'debug_expression_manager':
            from giskardpy.debug_expression_manager import DebugExpressionManager
            self.debug_expression_manager = DebugExpressionManager()
        return super().__getattribute__(item)

    def is_collision_checking_enabled(self):
        return self.collision_scene.collision_checker_id != self.collision_scene.collision_checker_id.none

    @staticmethod
    def is_in_github_workflow():
        return 'GITHUB_WORKFLOW' in os.environ

    def to_tmp_path(self, file_name: str) -> str:
        path = god_map.tmp_folder
        return get_middleware().resolve_iri(f'{path}{file_name}')

    def write_to_tmp(self, file_name: str, file_str: str) -> str:
        """
        Writes a URDF string into a temporary file on disc. Used to deliver URDFs to PyBullet that only loads file.
        :param file_name: Name of the temporary file without any path information, e.g. 'pr2.urdfs'
        :param file_str: URDF as an XML string that shall be written to disc.
        :return: Complete path to where the urdfs was written, e.g. '/tmp/pr2.urdfs'
        """
        new_path = self.to_tmp_path(file_name)
        create_path(new_path)
        with open(new_path, 'w') as f:
            f.write(file_str)
        return new_path

    def load_from_tmp(self, file_name: str):
        new_path = self.to_tmp_path(file_name)
        create_path(new_path)
        with open(new_path, 'r') as f:
            loaded_file = f.read()
        return loaded_file


god_map = GodMap()
