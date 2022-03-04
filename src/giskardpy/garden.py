from collections import defaultdict

import rospy
from giskard_msgs.msg import MoveAction
from py_trees import Sequence, Selector, BehaviourTree, Blackboard
from py_trees.meta import failure_is_success, success_is_failure, running_is_success, running_is_failure, \
    failure_is_running
from py_trees_ros.trees import BehaviourTree

import giskardpy.identifier as identifier
from giskardpy import RobotPrefix, RobotName
from giskardpy.data_types import order_map, KeyDefaultDict
from giskardpy.global_planner_needed import GlobalPlannerNeeded
from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer
from giskardpy.tree.AsyncComposite import PluginBehavior
from giskardpy.global_planner import GlobalPlanner, PreGraspSampler
from giskardpy.tree.commands_remaining import CommandsRemaining
from giskardpy.tree.exception_to_execute import ExceptionToExecute
from giskardpy.tree.goal_canceled import GoalCanceled
from giskardpy.tree.goal_received import GoalReceived
from giskardpy.tree.retry_planning import RetryPlanning
from giskardpy.tree.send_result import SendResult
from giskardpy.tree.start_timer import StartTimer
from giskardpy.tree.sync_localization import SyncLocalization
from giskardpy.utils.config_loader import ros_load_robot_config
from giskardpy.god_map import GodMap
from giskardpy.model.world import WorldTree
from giskardpy.tree.collision_scene_updater import CollisionSceneUpdater
from giskardpy.tree.append_zero_velocity import AppendZeroVelocity
from giskardpy.tree.cleanup import CleanUp
from giskardpy.tree.collision_checker import CollisionChecker
from giskardpy.tree.collision_marker import CollisionMarker
from giskardpy.tree.sync_configuration import SyncConfiguration
from giskardpy.tree.goal_reached import GoalReachedPlugin
from giskardpy.tree.plugin_if import IF, IfFunction, IF_NOT
from giskardpy.tree.instantaneous_controller import ControllerPlugin
from giskardpy.tree.kinematic_sim import KinSimPlugin
from giskardpy.tree.log_debug_expressions import LogDebugExpressionsPlugin
from giskardpy.tree.log_trajectory import LogTrajPlugin
from giskardpy.tree.loop_detector import LoopDetector
from giskardpy.tree.max_trajectory_length import MaxTrajectoryLength
from giskardpy.tree.plot_debug_expressions import PlotDebugExpressions
from giskardpy.tree.plot_trajectory import PlotTrajectory
from giskardpy.tree.set_error_code import SetErrorCode
from giskardpy.tree.send_trajectory import SendTrajectory
from giskardpy.tree.set_cmd import SetCmd, SetCollisionGoal
from giskardpy.tree.tf_publisher import TFPublisher
from giskardpy.tree.time import TimePlugin
from giskardpy.tree.update_constraints import GoalToConstraints
from giskardpy.tree.visualization import VisualizationBehavior
from giskardpy.tree.shaking_detector import WiggleCancel
from giskardpy.tree.tree_manager import TreeManager, render_dot_tree
from giskardpy.tree.world_updater import WorldUpdater
from giskardpy.utils import logging
from giskardpy.utils.math import max_velocity_from_horizon_and_jerk
from giskardpy.utils.utils import create_path


def upload_config_file_to_paramserver():
    old_params = rospy.get_param('~')
    if rospy.has_param('~test'):
        test = rospy.get_param('~test')
    else:
        test = False
    config_file_name = rospy.get_param('~{}'.format(u'config'))
    ros_load_robot_config(config_file_name, old_data=old_params, test=test)


def initialize_god_map():
    upload_config_file_to_paramserver()
    god_map = GodMap.init_from_paramserver(rospy.get_name())
    blackboard = Blackboard
    blackboard.god_map = god_map

    world = WorldTree(god_map)
    world.delete_all_but_robot()

    collision_checker = god_map.get_data(identifier.collision_checker)
    if collision_checker == 'bpb':
        logging.loginfo('Using bpb for collision checking.')
        from giskardpy.model.better_pybullet_syncer import BetterPyBulletSyncer
        collision_scene = BetterPyBulletSyncer(world)
    elif collision_checker == 'pybullet':
        logging.loginfo('Using pybullet for collision checking.')
        from giskardpy.model.pybullet_syncer import PyBulletSyncer
        collision_scene = PyBulletSyncer(world)
    else:
        logging.logwarn('Unknown collision checker {}. Collision avoidance is disabled'.format(collision_checker))
        collision_scene = CollisionWorldSynchronizer(world)
        god_map.set_data(identifier.collision_checker, None)
    god_map.set_data(identifier.collision_scene, collision_scene)

    # sanity_check_derivatives(god_map)
    # sanity_check(god_map)
    return god_map


def sanity_check(god_map):
    check_velocity_limits_reachable(god_map)


def sanity_check_derivatives(god_map):
    weights = god_map.get_data(identifier.joint_weights)
    limits = god_map.get_data(identifier.joint_limits)
    check_derivatives(weights, u'Weights')
    check_derivatives(limits, u'Limits')
    if len(weights) != len(limits):
        raise AttributeError(u'Weights and limits are not defined for the same number of derivatives')


def check_derivatives(entries, name):
    """
    :type entries: dict
    """
    allowed_derivates = list(order_map.values())[1:]
    for weight in entries:
        if weight not in allowed_derivates:
            raise AttributeError(
                u'{} set for unknown derivative: {} not in {}'.format(name, weight, list(allowed_derivates)))
    weight_ids = [order_map.inverse[x] for x in entries]
    if max(weight_ids) != len(weight_ids):
        raise AttributeError(
            u'{} for {} set, but some of the previous derivatives are missing'.format(name, order_map[max(weight_ids)]))


def check_velocity_limits_reachable(god_map):
    # TODO a more general version of this
    robot = god_map.get_data(identifier.robot)
    sample_period = god_map.get_data(identifier.sample_period)
    prediction_horizon = god_map.get_data(identifier.prediction_horizon)
    print_help = False
    for joint_name in robot.get_joint_names():
        velocity_limit = robot.get_joint_limit_expr_evaluated(joint_name, 1, god_map)
        jerk_limit = robot.get_joint_limit_expr_evaluated(joint_name, 3, god_map)
        velocity_limit_horizon = max_velocity_from_horizon_and_jerk(prediction_horizon, jerk_limit, sample_period)
        if velocity_limit_horizon < velocity_limit:
            logging.logwarn(u'Joint \'{}\' '
                            u'can reach at most \'{:.4}\' '
                            u'with to prediction horizon of \'{}\' '
                            u'and jerk limit of \'{}\', '
                            u'but limit in urdf/config is \'{}\''.format(
                joint_name,
                velocity_limit_horizon,
                prediction_horizon,
                jerk_limit,
                velocity_limit
            ))
            print_help = True
    if print_help:
        logging.logwarn(u'Check utils.py/max_velocity_from_horizon_and_jerk for help.')


def process_joint_specific_params(identifier_, default, override, god_map):
    default_value = god_map.unsafe_get_data(default)
    d = defaultdict(lambda: default_value)
    override = god_map.get_data(override)
    if isinstance(override, dict):
        d.update(override)
    god_map.set_data(identifier_, d)
    return KeyDefaultDict(lambda key: god_map.to_symbol(identifier_ + [key]))


def grow_tree():
    action_server_name = u'~command'

    god_map = initialize_god_map()
    # ----------------------------------------------
    sync = Sequence(u'Synchronize')
    sync.add_child(WorldUpdater(u'update world'))
    sync.add_child(SyncConfiguration(u'update robot configuration', RobotName))
    sync.add_child(SyncLocalization(u'update robot localization', RobotName))
    sync.add_child(TFPublisher(u'publish tf', **god_map.get_data(identifier.TFPublisher)))
    sync.add_child(CollisionSceneUpdater(u'update collision scene'))
    sync.add_child(running_is_success(VisualizationBehavior)(u'visualize collision scene'))
    # ----------------------------------------------
    wait_for_goal = Sequence(u'wait for goal')
    wait_for_goal.add_child(sync)
    wait_for_goal.add_child(GoalReceived(u'has goal', action_server_name, MoveAction))
    # ----------------------------------------------
    planning_4 = PluginBehavior(u'planning IIII', sleep=0)
    if god_map.get_data(identifier.collision_checker) is not None:
        planning_4.add_plugin(CollisionChecker(u'collision checker'))
    planning_4.add_plugin(VisualizationBehavior(u'visualization')) # visualization during planning
    # planning_4.add_plugin(CollisionMarker(u'cpi marker'))
    planning_4.add_plugin(ControllerPlugin(u'controller'))
    planning_4.add_plugin(KinSimPlugin(u'kin sim'))
    planning_4.add_plugin(LogTrajPlugin(u'log'))
    if god_map.get_data(identifier.PlotDebugTrajectory_enabled):
        planning_4.add_plugin(LogDebugExpressionsPlugin(u'log lba'))
    planning_4.add_plugin(WiggleCancel(u'wiggle'))
    planning_4.add_plugin(LoopDetector(u'loop detector'))
    planning_4.add_plugin(GoalReachedPlugin(u'goal reached'))
    planning_4.add_plugin(TimePlugin(u'time'))
    if god_map.get_data(identifier.MaxTrajectoryLength_enabled):
        kwargs = god_map.get_data(identifier.MaxTrajectoryLength)
        planning_4.add_plugin(MaxTrajectoryLength(u'traj length check', **kwargs))
    # ---------------------------------------------
    global_planning = Sequence(u'global planning')
    global_planning.add_child(PreGraspSampler())
    global_planning.add_child(running_is_success(GlobalPlannerNeeded)(u'GlobalPlannerNeeded', action_server_name, planning_4))
    if god_map.get_data(identifier.enable_VisualizationBehavior):
        global_planning.add_child(running_is_success(VisualizationBehavior)(u'visualization'))
    global_planning.add_child(GlobalPlanner(u'global planner', action_server_name))
    #global_planning.add_child(success_is_failure(GoalToConstraints)(u'update constraints', action_server_name))
    #trajectory_planning = Selector(u'trajectory planning', sleep=0)
    #trajectory_planning.add_child(global_planning)
    #trajectory_planning.add_child(planning_4)
    # ----------------------------------------------
    # ----------------------------------------------
    planning_3 = Sequence(u'planning III', sleep=0)
    planning_3.add_child(planning_4)
    planning_3.add_child(running_is_success(TimePlugin)(u'time for zero velocity'))
    planning_3.add_child(AppendZeroVelocity(u'append zero velocity'))
    planning_3.add_child(running_is_success(LogTrajPlugin)(u'log zero velocity'))
    if god_map.get_data(identifier.enable_VisualizationBehavior):
        planning_3.add_child(running_is_success(VisualizationBehavior)(u'visualization', ensure_publish=True))
    if god_map.get_data(identifier.enable_CPIMarker) and god_map.get_data(identifier.collision_checker) is not None:
        planning_3.add_child(running_is_success(CollisionMarker)(u'collision marker'))
    # ----------------------------------------------
    # ----------------------------------------------
    execute_canceled = Sequence(u'execute canceled')
    execute_canceled.add_child(GoalCanceled(u'goal canceled', action_server_name))
    execute_canceled.add_child(SetErrorCode(u'set error code'))
    publish_result = failure_is_success(Selector)(u'monitor execution')
    publish_result.add_child(execute_canceled)
    publish_result.add_child(SendTrajectory(u'send traj'))
    # ----------------------------------------------
    # ----------------------------------------------
    planning_2 = Selector(u'planning II')
    planning_2.add_child(GoalCanceled(u'goal canceled', action_server_name))
    if god_map.get_data(identifier.enable_VisualizationBehavior):
        planning_2.add_child(running_is_failure(VisualizationBehavior)(u'visualization'))
    # if god_map.get_data(identifier.enable_WorldVisualizationBehavior):
    #     planning_2.add_child(success_is_failure(WorldVisualizationBehavior)(u'world_visualization'))
    if god_map.get_data(identifier.enable_CPIMarker) and god_map.get_data(identifier.collision_checker) is not None:
        planning_2.add_child(running_is_failure(CollisionMarker)(u'cpi marker'))
    planning_2.add_child(success_is_failure(StartTimer)('start runtime timer'))
    planning_2.add_child(planning_3)
    # ----------------------------------------------
    move_robot = failure_is_success(Sequence)(u'move robot')
    move_robot.add_child(IF(u'execute?', identifier.execute))
    move_robot.add_child(publish_result)
    # ----------------------------------------------
    # ----------------------------------------------
    # planning_1 = Sequence(u'planning I')
    # ----------------------------------------------
    planning = success_is_failure(Sequence)(u'planning')
    planning.add_child(IF(u'command set?', identifier.next_move_goal))
    planning.add_child(SetCollisionGoal(u'set collision goal', action_server_name))
    planning.add_child(global_planning)
    planning.add_child(GoalToConstraints(u'update constraints', action_server_name))
    planning.add_child(planning_2)
    # planning.add_child(planning_1)
    # planning.add_child(SetErrorCode(u'set error code'))
    if god_map.get_data(identifier.PlotTrajectory_enabled):
        kwargs = god_map.get_data(identifier.PlotTrajectory)
        planning.add_child(PlotTrajectory(u'plot trajectory', **kwargs))
    if god_map.get_data(identifier.PlotDebugTrajectory_enabled):
        kwargs = god_map.get_data(identifier.PlotDebugTrajectory)
        planning.add_child(PlotDebugExpressions(u'plot debug expressions', **kwargs))

    process_move_cmd = Selector(u'Process move commands')
    process_move_cmd.add_child(success_is_failure(SetCmd)(u'set move cmd', action_server_name))
    process_move_cmd.add_child(planning)
    process_move_cmd.add_child(success_is_failure(RetryPlanning)(u'done planning?'))
    process_move_cmd.add_child(SetErrorCode(u'set error code'))

    process_move_goal = failure_is_success(Selector)(u'Process goal')
    process_move_goal.add_child(process_move_cmd)
    process_move_goal.add_child(ExceptionToExecute('clear exception'))
    process_move_goal.add_child(failure_is_running(CommandsRemaining)('commands remaining?'))


    # ----------------------------------------------
    # ----------------------------------------------
    root = Sequence(u'Giskard')
    root.add_child(wait_for_goal)
    root.add_child(CleanUp(u'cleanup'))
    root.add_child(process_move_goal)
    root.add_child(move_robot)
    root.add_child(SendResult(u'send result', action_server_name, MoveAction))

    tree = BehaviourTree(root)

    # if god_map.get_data(identifier.debug):
    #     def post_tick(snapshot_visitor, behaviour_tree):
    #         logging.logdebug(u'\n' + py_trees.display.ascii_tree(behaviour_tree.root,
    #                                                              snapshot_information=snapshot_visitor))
    #
    #     snapshot_visitor = py_trees_ros.visitors.SnapshotVisitor()
    #     tree.add_post_tick_handler(functools.partial(post_tick, snapshot_visitor))
    #     tree.visitors.append(snapshot_visitor)
    path = god_map.get_data(identifier.data_folder) + u'tree'
    create_path(path)
    render_dot_tree(root, name=path)

    tree.setup(30)
    tree_m = TreeManager(tree)
    god_map.set_data(identifier.tree_manager, tree_m)
    return tree
