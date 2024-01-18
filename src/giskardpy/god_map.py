from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Dict, Tuple

import rospy

from giskard_msgs.msg import MoveGoal, MoveResult

if TYPE_CHECKING:
    from giskardpy.tree.behaviors.action_server import ActionServerHandler
    from giskardpy.configs.behavior_tree_config import BehaviorTreeConfig
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
    behavior_tree_config: BehaviorTreeConfig
    collision_checker_id: CollisionCheckerLib
    world_config: WorldConfig
    collision_avoidance_config: CollisionAvoidanceConfig
    collision_avoidance_configs: Dict[str, CollisionAvoidanceGroupThresholds]
    move_action_server: ActionServerHandler
    world_action_server: ActionServerHandler
    trajectory: Trajectory
    qp_solver_solution: NextCommands
    added_collision_checks: Dict[Tuple[PrefixName, PrefixName], float]
    closest_point: Collisions
    # collision_matrix: Dict[Tuple[PrefixName, PrefixName], float]
    time_delay: rospy.Duration
    tracking_start_time: rospy.Time
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
        return self.move_action_server.goal_msg.type in [MoveGoal.EXECUTE]

    def is_goal_msg_type_projection(self):
        return MoveGoal.PROJECTION == self.move_action_server.goal_msg.type

    def is_goal_msg_type_undefined(self):
        return MoveGoal.UNDEFINED == self.move_action_server.goal_msg.type

    def is_closed_loop(self):
        return self.tree.is_closed_loop()

    def is_standalone(self):
        return self.tree.is_standalone()

    def is_open_loop(self):
        return self.tree.is_open_loop()

    def is_collision_checking_enabled(self):
        return self.collision_scene.collision_checker_id != self.collision_scene.collision_checker_id.none

    @staticmethod
    def is_in_github_workflow():
        return 'GITHUB_WORKFLOW' in os.environ

    def is_tree_alive(self):
        return self.tree.count > 1


god_map = GodMap()
