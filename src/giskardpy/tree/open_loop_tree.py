import pydot
from py_trees import Selector, Sequence
from py_trees.meta import running_is_success, failure_is_success, success_is_failure, running_is_failure, \
    failure_is_running
from py_trees_ros.trees import BehaviourTree

import giskardpy
from giskard_msgs.msg import MoveAction, MoveFeedback
from giskardpy import identifier, RobotName
from giskardpy.god_map import GodMap
from giskardpy.tree.append_zero_velocity import AppendZeroVelocity
from giskardpy.tree.async_composite import PluginBehavior
from giskardpy.tree.better_parallel import ParallelPolicy, Parallel
from giskardpy.tree.cleanup import CleanUp
from giskardpy.tree.collision_checker import CollisionChecker
from giskardpy.tree.collision_marker import CollisionMarker
from giskardpy.tree.collision_scene_updater import CollisionSceneUpdater
from giskardpy.tree.commands_remaining import CommandsRemaining
from giskardpy.tree.exception_to_execute import ExceptionToExecute
from giskardpy.tree.goal_canceled import GoalCanceled
from giskardpy.tree.goal_reached import GoalReachedPlugin
from giskardpy.tree.goal_received import GoalReceived
from giskardpy.tree.instantaneous_controller import ControllerPlugin
from giskardpy.tree.kinematic_sim import KinSimPlugin
from giskardpy.tree.log_debug_expressions import LogDebugExpressionsPlugin
from giskardpy.tree.log_trajectory import LogTrajPlugin
from giskardpy.tree.loop_detector import LoopDetector
from giskardpy.tree.max_trajectory_length import MaxTrajectoryLength
from giskardpy.tree.plot_debug_expressions import PlotDebugExpressions
from giskardpy.tree.plot_trajectory import PlotTrajectory
from giskardpy.tree.plugin_if import IF
from giskardpy.tree.publish_feedback import PublishFeedback
from giskardpy.tree.send_result import SendResult
from giskardpy.tree.set_cmd import SetCmd
from giskardpy.tree.set_error_code import SetErrorCode
from giskardpy.tree.shaking_detector import WiggleCancel
from giskardpy.tree.start_timer import StartTimer
from giskardpy.tree.sync_configuration import SyncConfiguration
from giskardpy.tree.sync_localization import SyncLocalization
from giskardpy.tree.tf_publisher import TFPublisher
from giskardpy.tree.time import TimePlugin
from giskardpy.tree.tree_manager import TreeManager
from giskardpy.tree.update_constraints import GoalToConstraints
from giskardpy.tree.visualization import VisualizationBehavior
from giskardpy.tree.world_updater import WorldUpdater
from giskardpy.utils.utils import get_all_classes_in_package


class OpenLoopTree(TreeManager):
    def __init__(self, god_map: GodMap):
        super().__init__(god_map)

    def grow_giskard(self):
        root = Sequence('Giskard')
        root.add_child(self.grow_wait_for_goal())
        root.add_child(CleanUp('cleanup'))
        root.add_child(self.grow_process_goal())
        root.add_child(self.grow_follow_joint_trajectory_execution())
        root.add_child(SendResult('send result', self.action_server_name, MoveAction))
        return root

    def grow_wait_for_goal(self):
        wait_for_goal = Sequence('wait for goal')
        wait_for_goal.add_child(self.grow_sync_branch())
        wait_for_goal.add_child(GoalReceived('has goal',
                                             self.action_server_name,
                                             MoveAction))
        return wait_for_goal

    def grow_sync_branch(self):
        sync = Sequence('Synchronize')
        sync.add_child(WorldUpdater('update world'))
        sync.add_child(running_is_success(SyncConfiguration)('update robot configuration', RobotName))
        sync.add_child(SyncLocalization('update robot localization', RobotName))
        sync.add_child(TFPublisher('publish tf', **self.god_map.get_data(identifier.TFPublisher)))
        sync.add_child(CollisionSceneUpdater('update collision scene'))
        sync.add_child(running_is_success(VisualizationBehavior)('visualize collision scene'))
        return sync

    def grow_process_goal(self):
        process_move_cmd = success_is_failure(Sequence)('Process move commands')
        process_move_cmd.add_child(SetCmd('set move cmd', self.action_server_name))
        process_move_cmd.add_child(self.grow_planning())
        process_move_cmd.add_child(SetErrorCode('set error code', 'Planning'))
        process_move_goal = failure_is_success(Selector)('Process goal')
        process_move_goal.add_child(success_is_failure(PublishFeedback)('publish feedback',
                                                                        self.action_server_name,
                                                                        MoveFeedback.PLANNING))
        process_move_goal.add_child(process_move_cmd)
        process_move_goal.add_child(ExceptionToExecute('clear exception'))
        process_move_goal.add_child(failure_is_running(CommandsRemaining)('commands remaining?'))
        return process_move_goal

    def grow_planning(self):
        planning = failure_is_success(Sequence)('planning')
        planning.add_child(IF('command set?', identifier.next_move_goal))
        planning.add_child(GoalToConstraints('update constraints', self.action_server_name))
        planning.add_child(self.grow_planning2())
        # planning.add_child(planning_1)
        # planning.add_child(SetErrorCode('set error code'))
        if self.god_map.get_data(identifier.PlotTrajectory_enabled):
            kwargs = self.god_map.get_data(identifier.PlotTrajectory)
            planning.add_child(PlotTrajectory('plot trajectory', **kwargs))
        if self.god_map.get_data(identifier.PlotDebugTrajectory_enabled):
            kwargs = self.god_map.get_data(identifier.PlotDebugTrajectory)
            planning.add_child(PlotDebugExpressions('plot debug expressions', **kwargs))
        return planning

    def grow_planning2(self):
        planning_2 = failure_is_success(Selector)('planning II')
        planning_2.add_child(GoalCanceled('goal canceled', self.action_server_name))
        planning_2.add_child(success_is_failure(PublishFeedback)('publish feedback',
                                                                 self.action_server_name,
                                                                 MoveFeedback.PLANNING))
        if self.god_map.get_data(identifier.enable_VisualizationBehavior):
            planning_2.add_child(running_is_failure(VisualizationBehavior)('visualization'))
        if self.god_map.get_data(identifier.enable_CPIMarker) and self.god_map.get_data(
                identifier.collision_checker) is not None:
            planning_2.add_child(running_is_failure(CollisionMarker)('cpi marker'))
        planning_2.add_child(success_is_failure(StartTimer)('start runtime timer'))
        planning_2.add_child(self.grow_planning3())
        return planning_2

    def grow_planning3(self):
        planning_3 = Sequence('planning III', sleep=0)
        planning_3.add_child(self.grow_planning4())
        planning_3.add_child(running_is_success(TimePlugin)('time for zero velocity'))
        planning_3.add_child(AppendZeroVelocity('append zero velocity'))
        planning_3.add_child(running_is_success(LogTrajPlugin)('log zero velocity'))
        if self.god_map.get_data(identifier.enable_VisualizationBehavior):
            planning_3.add_child(running_is_success(VisualizationBehavior)('visualization', ensure_publish=True))
        if self.god_map.get_data(identifier.enable_CPIMarker) and self.god_map.get_data(
                identifier.collision_checker) is not None:
            planning_3.add_child(running_is_success(CollisionMarker)('collision marker'))
        return planning_3

    def grow_planning4(self):
        planning_4 = PluginBehavior('planning IIII', sleep=0)
        if self.god_map.get_data(identifier.collision_checker) is not None:
            planning_4.add_plugin(CollisionChecker('collision checker'))
        # planning_4.add_plugin(VisualizationBehavior('visualization'))
        # planning_4.add_plugin(CollisionMarker('cpi marker'))
        planning_4.add_plugin(ControllerPlugin('controller'))
        planning_4.add_plugin(KinSimPlugin('kin sim'))
        planning_4.add_plugin(LogTrajPlugin('log'))
        if self.god_map.get_data(identifier.PlotDebugTrajectory_enabled):
            planning_4.add_plugin(LogDebugExpressionsPlugin('log lba'))
        planning_4.add_plugin(WiggleCancel('wiggle'))
        planning_4.add_plugin(LoopDetector('loop detector'))
        planning_4.add_plugin(GoalReachedPlugin('goal reached'))
        planning_4.add_plugin(TimePlugin('time'))
        if self.god_map.get_data(identifier.MaxTrajectoryLength_enabled):
            kwargs = self.god_map.get_data(identifier.MaxTrajectoryLength)
            planning_4.add_plugin(MaxTrajectoryLength('traj length check', **kwargs))
        return planning_4

    def grow_follow_joint_trajectory_execution(self):
        execution_action_server = Parallel('execution action servers',
                                           policy=ParallelPolicy.SuccessOnAll(synchronise=True))
        action_servers = self.god_map.get_data(identifier.robot_interface)
        behaviors = get_all_classes_in_package(giskardpy.tree)
        for i, (execution_action_server_name, params) in enumerate(action_servers.items()):
            C = behaviors[params['plugin']]
            del params['plugin']
            execution_action_server.add_child(C(execution_action_server_name, **params))

        execute_canceled = Sequence('execute canceled')
        execute_canceled.add_child(GoalCanceled('goal canceled', self.action_server_name))
        execute_canceled.add_child(SetErrorCode('set error code', 'Execution'))

        publish_result = failure_is_success(Selector)('monitor execution')
        publish_result.add_child(success_is_failure(PublishFeedback)('publish feedback',
                                                                     self.god_map.get_data(
                                                                         identifier.action_server_name),
                                                                     MoveFeedback.EXECUTION))
        publish_result.add_child(execute_canceled)
        publish_result.add_child(execution_action_server)
        publish_result.add_child(SetErrorCode('set error code', 'Execution'))

        move_robot = failure_is_success(Sequence)('move robot')
        move_robot.add_child(IF('execute?', identifier.execute))
        move_robot.add_child(publish_result)
        return move_robot
