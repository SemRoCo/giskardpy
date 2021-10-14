import functools
from collections import defaultdict
from copy import deepcopy

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
from giskardpy.global_planner import GlobalPlanner
import giskardpy.model.pybullet_wrapper as pbw
from giskardpy.data_types import BiDict, KeyDefaultDict
from giskardpy.god_map import GodMap
from giskardpy.tree.plugin import PluginBehavior
from giskardpy.tree.plugin_action_server import GoalReceived, SendResult, GoalCanceled
from giskardpy.tree.plugin_append_zero_velocity import AppendZeroVelocity
from giskardpy.tree.plugin_cleanup import CleanUp
from giskardpy.tree.plugin_collision_checker import CollisionChecker
from giskardpy.tree.plugin_collision_marker import CollisionMarker
from giskardpy.tree.plugin_configuration import ConfigurationPlugin
from giskardpy.tree.plugin_goal_reached import GoalReachedPlugin
from giskardpy.tree.plugin_if import IF
from giskardpy.tree.plugin_instantaneous_controller import ControllerPlugin
from giskardpy.tree.plugin_max_trajectory_length import MaxTrajectoryLength
from giskardpy.tree.plugin_tip_stuck_checker import TipStuckChecker
from giskardpy.tree.plugin_wiggle_cancel import WiggleCancel
from giskardpy.tree.plugin_kinematic_sim import KinSimPlugin
from giskardpy.tree.plugin_log_debug_expressions import LogDebugExpressionsPlugin
from giskardpy.tree.plugin_log_trajectory import LogTrajPlugin
from giskardpy.tree.plugin_loop_detector import LoopDetector
from giskardpy.tree.plugin_plot_debug_expressions import PlotDebugExpressions
from giskardpy.tree.plugin_plot_trajectory import PlotTrajectory
from giskardpy.tree.plugin_post_processing import PostProcessing
from giskardpy.tree.plugin_pybullet import WorldUpdatePlugin
from giskardpy.tree.plugin_send_trajectory import SendTrajectory
from giskardpy.tree.plugin_set_cmd import SetCmd
from giskardpy.tree.plugin_tf_publisher import TFPublisher
from giskardpy.tree.plugin_time import TimePlugin
from giskardpy.tree.plugin_update_constraints import GoalToConstraints
from giskardpy.tree.plugin_visualization import VisualizationBehavior
from giskardpy.tree.plugin_world_visualization import WorldVisualizationBehavior
from giskardpy.model.pybullet_world import PyBulletWorld
from giskardpy.tree.tree_manager import TreeManager
from giskardpy.utils import logging
from giskardpy.utils.math import max_velocity_from_horizon_and_jerk
from giskardpy.utils.utils import create_path, render_dot_tree
from giskardpy.model.world_object import WorldObject

# TODO hardcode this somewhere else
order_map = BiDict({
    0: u'position',
    1: u'velocity',
    2: u'acceleration',
    3: u'jerk',
    4: u'snap',
    5: u'crackle',
    6: u'pop'
})


def initialize_god_map():
    # FIXME i hate this function
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

    set_default_in_override_block(identifier.external_collision_avoidance, god_map)
    set_default_in_override_block(identifier.self_collision_avoidance, god_map)

    world = PyBulletWorld(False, blackboard.god_map.get_data(identifier.data_folder))
    god_map.set_data(identifier.world, world)
    robot = WorldObject(god_map.get_data(identifier.robot_description),
                        None,
                        controlled_joints)
    world.add_robot(robot, None, controlled_joints,
                    ignored_pairs=god_map.get_data(identifier.ignored_self_collisions),
                    added_pairs=god_map.get_data(identifier.added_self_collisions))

    sanity_check_derivatives(god_map)

    # weights
    for i, key in enumerate(god_map.get_data(identifier.joint_weights), start=1):
        d = set_default_in_override_block(identifier.joint_weights + [order_map[i], u'override'], god_map)
        world.robot.set_joint_weight_symbols(d, i)


    # limits
    for i, key in enumerate(god_map.get_data(identifier.joint_limits), start=1):
        d_linear = set_default_in_override_block(identifier.joint_limits + [order_map[i], u'linear', u'override'],
                                                 god_map)
        d_angular = set_default_in_override_block(identifier.joint_limits + [order_map[i], u'angular', u'override'],
                                                  god_map)
        world.robot.set_joint_limit_symbols(d_linear, d_angular, i)

    order = len(god_map.get_data(identifier.joint_weights))+1
    god_map.set_data(identifier.order, order)

    # joint symbols
    for o in range(order):
        key = order_map[o]
        joint_position_symbols = {}
        for joint_name in world.robot.get_movable_joints():
            joint_position_symbols[joint_name] = god_map.to_symbol(identifier.joint_states + [joint_name, key])
        world.robot.set_joint_symbols(joint_position_symbols, o)

    world.robot.reinitialize()

    world.robot.init_self_collision_matrix()
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
            raise AttributeError(u'{} set for unknown derivative: {} not in {}'.format(name, weight, list(allowed_derivates)))
    weight_ids = [order_map.inverse[x] for x in entries]
    if max(weight_ids) != len(weight_ids):
        raise AttributeError(u'{} for {} set, but some of the previous derivatives are missing'.format(name, order_map[max(weight_ids)]))

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


def set_default_in_override_block(block_identifier, god_map):
    default_value = god_map.get_data(block_identifier[:-1] + [u'default'])
    override = god_map.get_data(block_identifier)
    d = defaultdict(lambda: default_value)
    if isinstance(override, dict):
        if isinstance(default_value, dict):
            for key, value in override.items():
                o = deepcopy(default_value)
                o.update(value)
                override[key] = o
        d.update(override)
    god_map.set_data(block_identifier, d)
    return KeyDefaultDict(lambda key: god_map.to_symbol(block_identifier + [key]))


def grow_tree():
    action_server_name = u'~command'

    god_map = initialize_god_map()
    # ----------------------------------------------
    wait_for_goal = Sequence(u'wait for goal')
    wait_for_goal.add_child(TFPublisher(u'tf', **god_map.get_data(identifier.TFPublisher)))
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
    planning_4.add_plugin(TipStuckChecker(u'tip stuck checker'))
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
    planning_1.add_child(GlobalPlanner(u'global planner', action_server_name))
    planning_1.add_child(GoalToConstraints(u'update constraints', action_server_name))
    planning_1.add_child(planning_2)
    # ----------------------------------------------
    post_processing = failure_is_success(Sequence)(u'post planning')
    # post_processing.add_child(WiggleCancel(u'final wiggle detection', final_detection=True))
    if god_map.get_data(identifier.PlotTrajectory_enabled):
        kwargs = god_map.get_data(identifier.PlotTrajectory)
        post_processing.add_child(PlotTrajectory(u'plot trajectory', **kwargs))
    if god_map.get_data(identifier.PlotDebugTrajectory_enabled):
        kwargs = god_map.get_data(identifier.PlotDebugTrajectory)
        post_processing.add_child(PlotDebugExpressions(u'plot debug expressions', **kwargs))
    post_processing.add_child(PostProcessing(u'evaluate result'))
    # ----------------------------------------------
    planning = success_is_failure(Sequence)(u'planning')
    planning.add_child(IF(u'goal_set?', identifier.next_move_goal))
    planning.add_child(planning_1)
    planning.add_child(post_processing)

    process_move_goal = failure_is_success(Selector)(u'process move goal')
    process_move_goal.add_child(planning)
    process_move_goal.add_child(SetCmd(u'set move goal', action_server_name))

    # ----------------------------------------------
    # ----------------------------------------------
    root = Sequence(u'root')
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
