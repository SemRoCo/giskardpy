from __future__ import annotations

import rospy

from giskard_msgs.msg import MoveGoal, MoveResult
from typing import TYPE_CHECKING, List, Dict, Tuple
import os

if TYPE_CHECKING:
    from giskardpy.tree.branches.giskard_bt import GiskardBT
    from giskardpy.model.joints import Joint
    from giskardpy.model.ros_msg_visualization import ROSMsgVisualization
    from giskardpy.qp.constraint import EqualityConstraint, InequalityConstraint, DerivativeInequalityConstraint, \
        ManipulabilityConstraint
    from giskardpy.qp.free_variable import FreeVariable
    from giskardpy.qp.next_command import NextCommands
    from giskardpy.model.trajectory import Trajectory
    from giskardpy.qp.qp_controller import QPProblemBuilder
    from giskardpy.configs.qp_controller_config import QPControllerConfig
    from giskardpy.data_types import Derivatives, PrefixName
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
    goal_id: int = -1
    monitor_manager: MonitorManager
    giskard: Giskard
    time: float  # real/planning time in s
    control_cycle_counter: int
    world: WorldTree
    motion_goal_manager: MotionGoalManager
    debug_expression_manager: DebugExpressionManager
    tree: GiskardBT
    collision_scene: CollisionWorldSynchronizer
    prediction_horizon: int
    max_derivative: Derivatives
    qp_controller: QPProblemBuilder
    qp_controller_config: QPControllerConfig
    collision_checker_id: CollisionCheckerLib
    world_config: WorldConfig
    collision_avoidance_config: CollisionAvoidanceConfig
    collision_avoidance_configs: Dict[str, CollisionAvoidanceGroupThresholds]
    goal_msg: MoveGoal
    trajectory: Trajectory
    qp_solver_solution: NextCommands
    added_collision_checks: Dict[Tuple[PrefixName, PrefixName], float]
    closest_point: Collisions
    # collision_matrix: Dict[Tuple[PrefixName, PrefixName], float]
    time_delay: rospy.Duration
    tracking_start_time: rospy.Time
    result_message: MoveResult
    eq_constraints: Dict[str, EqualityConstraint]
    neq_constraints: Dict[str, InequalityConstraint]
    derivative_constraints: Dict[str, DerivativeInequalityConstraint]
    manip_constraints: Dict[str, ManipulabilityConstraint]
    hack: float
    fill_trajectory_velocity_values: bool
    ros_visualizer: ROSMsgVisualization
    free_variables: List[FreeVariable]
    controlled_joints: List[Joint]

    def is_goal_msg_type_execute(self):
        return self.goal_msg.type in [MoveGoal.EXECUTE]

    def is_goal_msg_type_projection(self):
        return MoveGoal.PROJECTION == self.goal_msg.type

    def is_goal_msg_type_undefined(self):
        return MoveGoal.UNDEFINED == self.goal_msg.type

    def is_closed_loop(self):
        return self.tree.is_closed_loop()

    def is_standalone(self):
        return self.tree.is_standalone()

    def is_planning(self):
        return self.tree.is_planning()

    def is_collision_checking_enabled(self):
        return self.collision_scene.collision_checker_id != self.collision_scene.collision_checker_id.none

    @staticmethod
    def is_in_github_workflow():
        return 'GITHUB_WORKFLOW' in os.environ

    def is_tree_alive(self):
        return self.tree.count > 1


god_map = GodMap()
