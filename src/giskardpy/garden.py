import functools
from collections import defaultdict

import py_trees
import py_trees_ros
import rospy
from control_msgs.msg import JointTrajectoryControllerState
from giskard_msgs.msg import MoveAction
from py_trees import Sequence, Selector, BehaviourTree, Blackboard
from py_trees.meta import failure_is_success, success_is_failure, running_is_success
from py_trees_ros.trees import BehaviourTree
from rospy import ROSException

import giskardpy.identifier as identifier
import giskardpy.pybullet_wrapper as pbw
from giskardpy import logging
from giskardpy.god_map import GodMap
from giskardpy.input_system import JointStatesInput
from giskardpy.plugin import PluginBehavior
from giskardpy.plugin_action_server import GoalReceived, SendResult, GoalCanceled
from giskardpy.plugin_append_zero_velocity import AppendZeroVelocity
from giskardpy.plugin_cleanup import CleanUp
from giskardpy.plugin_collision_checker import CollisionChecker
from giskardpy.plugin_collision_marker import CollisionMarker
from giskardpy.plugin_configuration import ConfigurationPlugin
from giskardpy.plugin_goal_reached import GoalReachedPlugin
from giskardpy.plugin_if import IF
from giskardpy.plugin_instantaneous_controller import ControllerPlugin
from giskardpy.plugin_interrupts import WiggleCancel, MaxTrajLength
from giskardpy.plugin_kinematic_sim import KinSimPlugin
from giskardpy.plugin_log_debug_expressions import LogDebugExpressionsPlugin
from giskardpy.plugin_log_trajectory import LogTrajPlugin
from giskardpy.plugin_loop_detector import LoopDetector
from giskardpy.plugin_plot_debug_expressions import PlotDebugExpressions
from giskardpy.plugin_plot_trajectory import PlotTrajectory
from giskardpy.plugin_post_processing import PostProcessing
from giskardpy.plugin_pybullet import WorldUpdatePlugin
from giskardpy.plugin_send_trajectory import SendTrajectory
from giskardpy.plugin_set_cmd import SetCmd
from giskardpy.plugin_tf_publisher import TFPlugin
from giskardpy.plugin_time import TimePlugin
from giskardpy.plugin_update_constraints import GoalToConstraints
from giskardpy.plugin_visualization import VisualizationBehavior
from giskardpy.plugin_world_visualization import WorldVisualizationBehavior
from giskardpy.pybullet_world import PyBulletWorld
from giskardpy.tree_manager import TreeManager
from giskardpy.utils import create_path, render_dot_tree, KeyDefaultDict, max_velocity_from_horizon_and_jerk
from giskardpy.world_object import WorldObject


def initialize_god_map():
    god_map = GodMap()
    blackboard = Blackboard
    blackboard.god_map = god_map
    god_map.set_data(identifier.rosparam, rospy.get_param(rospy.get_name()))
    god_map.set_data(identifier.robot_description, rospy.get_param(u'robot_description'))
    path_to_data_folder = god_map.get_data(identifier.data_folder)
    # fix path to data folder
    if not path_to_data_folder.endswith(u'/'):
        path_to_data_folder += u'/'
    god_map.set_data(identifier.data_folder, path_to_data_folder)

    pbw.start_pybullet(god_map.get_data(identifier.gui))
    while not rospy.is_shutdown():
        try:
            controlled_joints = rospy.wait_for_message(u'/whole_body_controller/state',
                                                       JointTrajectoryControllerState,
                                                       timeout=5.0).joint_names
        except ROSException as e:
            logging.logerr(u'state topic not available')
            logging.logerr(str(e))
        else:
            break
        rospy.sleep(0.5)

    process_joint_specific_params(identifier.self_collision_avoidance_distance,
                                  identifier.self_collision_avoidance_default_threshold,
                                  identifier.self_collision_avoidance_default_override,
                                  god_map)

    process_joint_specific_params(identifier.external_collision_avoidance_distance,
                                  identifier.external_collision_avoidance_default_threshold,
                                  identifier.external_collision_avoidance_default_override,
                                  god_map)

    world = PyBulletWorld(False, blackboard.god_map.get_data(identifier.data_folder))
    god_map.set_data(identifier.world, world)
    robot = WorldObject(god_map.get_data(identifier.robot_description),
                        None,
                        controlled_joints)
    world.add_robot(robot, None, controlled_joints,
                    ignored_pairs=god_map.get_data(identifier.ignored_self_collisions),
                    added_pairs=god_map.get_data(identifier.added_self_collisions))

    d = set_default_in_override_block(identifier.joint_velocity_weight, god_map)
    world.robot.set_joint_velocity_weight_symbols(d)

    d = set_default_in_override_block(identifier.joint_acceleration_weight, god_map)
    world.robot.set_joint_acceleration_weight_symbols(d)

    d = set_default_in_override_block(identifier.joint_jerk_weight, god_map)
    world.robot.set_joint_jerk_weight_symbols(d)

    d_linear = set_default_in_override_block(identifier.joint_velocity_linear_limit, god_map)
    d_angular = set_default_in_override_block(identifier.joint_velocity_angular_limit, god_map)
    world.robot.set_joint_velocity_limit_symbols(d_linear, d_angular)

    d_linear = set_default_in_override_block(identifier.joint_acceleration_linear_limit, god_map)
    d_angular = set_default_in_override_block(identifier.joint_acceleration_angular_limit, god_map)
    world.robot.set_joint_acceleration_limit_symbols(d_linear, d_angular)

    d_linear = set_default_in_override_block(identifier.joint_jerk_linear_limit, god_map)
    d_angular = set_default_in_override_block(identifier.joint_jerk_angular_limit, god_map)
    world.robot.set_joint_jerk_limit_symbols(d_linear, d_angular)

    joint_position_symbols = JointStatesInput(blackboard.god_map.to_symbol, world.robot.get_movable_joints(),
                                              identifier.joint_states,
                                              suffix=[u'position'])
    world.robot.set_joint_position_symbols(joint_position_symbols.joint_map)
    joint_vel_symbols = JointStatesInput(blackboard.god_map.to_symbol, world.robot.get_movable_joints(),
                                         identifier.joint_states,
                                         suffix=[u'velocity'])
    world.robot.set_joint_velocity_symbols(joint_vel_symbols.joint_map)
    joint_acc_symbols = JointStatesInput(blackboard.god_map.to_symbol, world.robot.get_movable_joints(),
                                         identifier.joint_states,
                                         suffix=[u'acceleration'])
    world.robot.set_joint_acceleration_symbols(joint_acc_symbols.joint_map)
    world.robot.reinitialize()

    world.robot.init_self_collision_matrix()
    sanity_check(god_map)
    return god_map


def sanity_check(god_map):
    check_velocity_limits_reachable(god_map)


def check_velocity_limits_reachable(god_map):
    robot = god_map.get_data(identifier.robot)
    sample_period = god_map.get_data(identifier.sample_period)
    prediction_horizon = god_map.get_data(identifier.prediction_horizon)
    print_help = False
    for joint_name in robot.get_joint_names():
        velocity_limit = robot.get_joint_velocity_limit_expr_evaluated(joint_name, god_map)
        jerk_limit = robot.get_joint_jerk_limit_expr_evaluated(joint_name, god_map)
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


def set_default_in_override_block(block_identifier, god_map):
    default_value = god_map.get_data(block_identifier[:-1] + [u'default'])
    override = god_map.get_data(block_identifier)
    d = defaultdict(lambda: default_value)
    if isinstance(override, dict):
        d.update(override)
    god_map.set_data(block_identifier, d)
    return KeyDefaultDict(lambda key: god_map.to_symbol(block_identifier + [key]))


def grow_tree():
    action_server_name = u'~command'

    god_map = initialize_god_map()
    # ----------------------------------------------
    wait_for_goal = Sequence(u'wait for goal')
    wait_for_goal.add_child(TFPlugin(u'tf'))
    wait_for_goal.add_child(ConfigurationPlugin(u'js1'))
    wait_for_goal.add_child(WorldUpdatePlugin(u'pybullet updater'))
    wait_for_goal.add_child(GoalReceived(u'has goal', action_server_name, MoveAction))
    wait_for_goal.add_child(ConfigurationPlugin(u'js2'))
    # ----------------------------------------------
    planning_4 = PluginBehavior(u'planning IIII', sleep=0)
    planning_4.add_plugin(CollisionChecker(u'coll'))
    # if god_map.safe_get_data(identifier.enable_collision_marker):
    #     planning_3.add_plugin(success_is_running(CPIMarker)(u'cpi marker'))
    planning_4.add_plugin(ControllerPlugin(u'controller'))
    planning_4.add_plugin(KinSimPlugin(u'kin sim'))
    planning_4.add_plugin(LogTrajPlugin(u'log'))
    planning_4.add_plugin(LogDebugExpressionsPlugin(u'log lba'))
    planning_4.add_plugin(WiggleCancel(u'wiggle'))
    planning_4.add_plugin(LoopDetector(u'loop detector'))
    planning_4.add_plugin(GoalReachedPlugin(u'goal reached'))
    planning_4.add_plugin(TimePlugin(u'time'))
    planning_4.add_plugin(MaxTrajLength(u'traj length check', 15))
    # ----------------------------------------------
    # ----------------------------------------------
    planning_3 = Sequence(u'planning III', sleep=0)
    planning_3.add_child(planning_4)
    planning_3.add_child(running_is_success(TimePlugin)(u'time for zero velocity'))
    planning_3.add_child(AppendZeroVelocity(u'append zero velocity'))
    planning_3.add_child(running_is_success(LogTrajPlugin)(u'log zero velocity'))
    if god_map.get_data(identifier.enable_VisualizationBehavior):
        planning_3.add_child(VisualizationBehavior(u'visualization', ensure_publish=True))
    if god_map.get_data(identifier.enable_WorldVisualizationBehavior):
        planning_3.add_child(WorldVisualizationBehavior(u'world_visualization', ensure_publish=True))
    if god_map.get_data(identifier.enable_CPIMarker):
        planning_3.add_child(CollisionMarker(u'cpi marker'))
    # ----------------------------------------------
    # ----------------------------------------------
    publish_result = failure_is_success(Selector)(u'monitor execution')
    publish_result.add_child(GoalCanceled(u'goal canceled', action_server_name))
    publish_result.add_child(SendTrajectory(u'send traj'))
    # ----------------------------------------------
    # ----------------------------------------------
    planning_2 = failure_is_success(Selector)(u'planning II')
    planning_2.add_child(GoalCanceled(u'goal canceled', action_server_name))
    if god_map.get_data(identifier.enable_VisualizationBehavior):
        planning_2.add_child(success_is_failure(VisualizationBehavior)(u'visualization'))
    if god_map.get_data(identifier.enable_WorldVisualizationBehavior):
        planning_2.add_child(success_is_failure(WorldVisualizationBehavior)(u'world_visualization'))
    if god_map.get_data(identifier.enable_CPIMarker):
        planning_2.add_child(success_is_failure(CollisionMarker)(u'cpi marker'))
    planning_2.add_child(planning_3)
    # ----------------------------------------------
    move_robot = failure_is_success(Sequence)(u'move robot')
    move_robot.add_child(IF(u'execute?', identifier.execute))
    move_robot.add_child(publish_result)
    # ----------------------------------------------
    # ----------------------------------------------
    planning_1 = Sequence(u'planning I')
    planning_1.add_child(GoalToConstraints(u'update constraints', action_server_name))
    planning_1.add_child(planning_2)
    # ----------------------------------------------
    post_processing = failure_is_success(Sequence)(u'post planning')
    # post_processing.add_child(WiggleCancel(u'final wiggle detection', final_detection=True))
    if god_map.get_data(identifier.enable_PlotTrajectory):
        post_processing.add_child(PlotTrajectory(u'plot trajectory', order=4))
        post_processing.add_child(PlotDebugExpressions(u'plot lba', order=2))
    post_processing.add_child(PostProcessing(u'evaluate result'))
    # post_processing.add_child(PostProcessing(u'check reachability'))
    # ----------------------------------------------
    planning = success_is_failure(Sequence)(u'planning')
    planning.add_child(IF(u'goal_set?', identifier.next_move_goal))
    planning.add_child(planning_1)
    planning.add_child(post_processing)

    process_move_goal = failure_is_success(Selector)(u'process move goal')
    process_move_goal.add_child(planning)
    # process_move_goal.add_child(planning_1)
    # process_move_goal.add_child(post_processing)
    process_move_goal.add_child(SetCmd(u'set move goal', action_server_name))
    # ----------------------------------------------
    #
    # post_processing = failure_is_success(Sequence)(u'post processing')
    # post_processing.add_child(PostProcessing(u'post_processing'))

    # ----------------------------------------------
    # ----------------------------------------------
    root = Sequence(u'root')
    root.add_child(wait_for_goal)
    root.add_child(CleanUp(u'cleanup'))
    root.add_child(process_move_goal)
    root.add_child(move_robot)
    root.add_child(SendResult(u'send result', action_server_name, MoveAction))

    tree = BehaviourTree(root)

    if god_map.get_data(identifier.debug):
        def post_tick(snapshot_visitor, behaviour_tree):
            logging.logdebug(u'\n' + py_trees.display.ascii_tree(behaviour_tree.root,
                                                                 snapshot_information=snapshot_visitor))

        snapshot_visitor = py_trees_ros.visitors.SnapshotVisitor()
        tree.add_post_tick_handler(functools.partial(post_tick, snapshot_visitor))
        tree.visitors.append(snapshot_visitor)
    path = god_map.get_data(identifier.data_folder) + u'tree'
    create_path(path)
    render_dot_tree(root, name=path)

    tree.setup(30)
    tree_m = TreeManager(tree)
    god_map.set_data(identifier.tree_manager, tree_m)
    return tree
