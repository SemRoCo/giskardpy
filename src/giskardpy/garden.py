import functools
from itertools import combinations

import py_trees
import py_trees_ros
import rospy
from control_msgs.msg import JointTrajectoryControllerState
from giskard_msgs.msg import MoveAction
from py_trees import Sequence, Selector, BehaviourTree, Blackboard
from py_trees.meta import failure_is_success, success_is_failure, failure_is_running, running_is_success
from py_trees_ros.trees import BehaviourTree
from rospy import ROSException

import giskardpy.identifier as identifier
import giskardpy.pybullet_wrapper as pbw
from giskardpy import logging
from giskardpy.god_map import GodMap
from giskardpy.input_system import JointStatesInput
from giskardpy.plugin import PluginBehavior, SuccessPlugin
from giskardpy.plugin_action_server import GoalReceived, SendResult, GoalCanceled
from giskardpy.plugin_append_zero_velocity import AppendZeroVelocity
from giskardpy.plugin_attached_tf_publicher import TFPlugin
from giskardpy.plugin_cleanup import CleanUp
from giskardpy.plugin_collision_checker import CollisionChecker
from giskardpy.plugin_configuration import ConfigurationPlugin
from giskardpy.plugin_collision_marker import CollisionMarker
from giskardpy.plugin_goal_reached import GoalReachedPlugin
from giskardpy.plugin_if import IF
from giskardpy.plugin_instantaneous_controller import ControllerPlugin
from giskardpy.plugin_interrupts import WiggleCancel
from giskardpy.plugin_kinematic_sim import KinSimPlugin
from giskardpy.plugin_log_trajectory import LogTrajPlugin
from giskardpy.plugin_loop_detector import LoopDetector
from giskardpy.plugin_plot_trajectory import PlotTrajectory
#from giskardpy.plugin_plot_trajectory_fft import PlotTrajectoryFFT
from giskardpy.plugin_pybullet import WorldUpdatePlugin
from giskardpy.plugin_send_trajectory import SendTrajectory
from giskardpy.plugin_set_cmd import SetCmd
from giskardpy.plugin_time import TimePlugin
from giskardpy.plugin_update_constraints import GoalToConstraints
from giskardpy.plugin_visualization import VisualizationBehavior
from giskardpy.plugin_post_processing import PostProcessing
# from giskardpy.pybullet_world import PyBulletWorld
from giskardpy.pybullet_world import PyBulletWorld
from giskardpy.utils import create_path, render_dot_tree, KeyDefaultDict
from giskardpy.world import World
from giskardpy.world_object import WorldObject
from collections import defaultdict
from giskardpy.tree_manager import TreeManager


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

    # fix nWSR
    nWSR = god_map.get_data(identifier.nWSR)
    if nWSR == u'None':
        nWSR = None
    god_map.set_data(identifier.nWSR, nWSR)

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

    joint_weight_symbols = process_joint_specific_params(identifier.joint_weight,
                                                         identifier.joint_weight_default,
                                                         identifier.joint_weight_override,
                                                         god_map)

    process_joint_specific_params(identifier.self_collision_avoidance_distance,
                                  identifier.self_collision_avoidance_default_threshold,
                                  identifier.self_collision_avoidance_default_override,
                                  god_map)

    process_joint_specific_params(identifier.external_collision_avoidance_distance,
                                  identifier.external_collision_avoidance_default_threshold,
                                  identifier.external_collision_avoidance_default_override,
                                  god_map)

    #TODO add checks to test if joints listed as linear are actually linear
    joint_velocity_linear_limit_symbols = process_joint_specific_params(identifier.joint_velocity_linear_limit,
                                                                        identifier.joint_velocity_linear_limit_default,
                                                                        identifier.joint_velocity_linear_limit_override,
                                                                        god_map)
    joint_velocity_angular_limit_symbols = process_joint_specific_params(identifier.joint_velocity_angular_limit,
                                                                         identifier.joint_velocity_angular_limit_default,
                                                                         identifier.joint_velocity_angular_limit_override,
                                                                         god_map)

    joint_acceleration_linear_limit_symbols = process_joint_specific_params(identifier.joint_acceleration_linear_limit,
                                                                            identifier.joint_acceleration_linear_limit_default,
                                                                            identifier.joint_acceleration_linear_limit_override,
                                                                            god_map)
    joint_acceleration_angular_limit_symbols = process_joint_specific_params(identifier.joint_acceleration_angular_limit,
                                                                             identifier.joint_acceleration_angular_limit_default,
                                                                             identifier.joint_acceleration_angular_limit_override,
                                                                             god_map)

    world = PyBulletWorld(False, blackboard.god_map.get_data(identifier.data_folder))
    god_map.set_data(identifier.world, world)
    robot = WorldObject(god_map.get_data(identifier.robot_description),
                        None,
                        controlled_joints)
    world.add_robot(robot, None, controlled_joints,
                    ignored_pairs=god_map.get_data(identifier.ignored_self_collisions),
                    added_pairs=god_map.get_data(identifier.added_self_collisions))

    joint_position_symbols = JointStatesInput(blackboard.god_map.to_symbol, world.robot.get_movable_joints(),
                                              identifier.joint_states,
                                              suffix=[u'position'])
    joint_vel_symbols = JointStatesInput(blackboard.god_map.to_symbol, world.robot.get_movable_joints(),
                                         identifier.joint_states,
                                         suffix=[u'velocity'])
    world.robot.update_joint_symbols(joint_position_symbols.joint_map, joint_vel_symbols.joint_map,
                                     joint_weight_symbols,
                                     joint_velocity_linear_limit_symbols, joint_velocity_angular_limit_symbols,
                                     joint_acceleration_linear_limit_symbols, joint_acceleration_angular_limit_symbols)
    world.robot.init_self_collision_matrix()
    return god_map

def process_joint_specific_params(identifier_, default, override, god_map):
    default_value = god_map.unsafe_get_data(default)
    d = defaultdict(lambda: default_value)
    override = god_map.get_data(override)
    if isinstance(override, dict):
        d.update(override)
    god_map.set_data(identifier_, d)
    return KeyDefaultDict(lambda key: god_map.to_symbol(identifier_ + [key]))


def grow_tree():
    action_server_name = u'giskardpy/command'

    god_map = initialize_god_map()
    # ----------------------------------------------
    wait_for_goal = Sequence(u'wait for goal')
    wait_for_goal.add_child(TFPlugin(u'tf'))
    wait_for_goal.add_child(ConfigurationPlugin(u'js1'))
    wait_for_goal.add_child(WorldUpdatePlugin(u'pybullet updater'))
    wait_for_goal.add_child(GoalReceived(u'has goal', action_server_name, MoveAction))
    wait_for_goal.add_child(ConfigurationPlugin(u'js2'))
    # ----------------------------------------------
    planning_3 = PluginBehavior(u'planning III', sleep=0)
    planning_3.add_plugin(CollisionChecker(u'coll'))
    # if god_map.safe_get_data(identifier.enable_collision_marker):
    #     planning_3.add_plugin(success_is_running(CPIMarker)(u'cpi marker'))
    planning_3.add_plugin(ControllerPlugin(u'controller'))
    planning_3.add_plugin(KinSimPlugin(u'kin sim'))
    planning_3.add_plugin(LogTrajPlugin(u'log'))
    planning_3.add_plugin(WiggleCancel(u'wiggle'))
    planning_3.add_plugin(LoopDetector(u'loop detector'))
    planning_3.add_plugin(GoalReachedPlugin(u'goal reached'))
    planning_3.add_plugin(TimePlugin(u'time'))
    # planning_3.add_plugin(MaxTrajLength(u'traj length check'))
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
    planning_1.add_child(running_is_success(TimePlugin)(u'time for zero velocity'))
    planning_1.add_child(AppendZeroVelocity(u'append zero velocity'))
    planning_1.add_child(running_is_success(LogTrajPlugin)(u'log zero velocity'))
    # planning_1.add_child(running_is_success(TimePlugin)(u'time for zero velocity'))
    # planning_1.add_child(AppendZeroVelocity(u'append zero velocity'))
    # planning_1.add_child(running_is_success(LogTrajPlugin)(u'log zero velocity'))
    if god_map.get_data(identifier.enable_VisualizationBehavior):
        planning_1.add_child(VisualizationBehavior(u'visualization', ensure_publish=True))
    if god_map.get_data(identifier.enable_CPIMarker):
        planning_1.add_child(CollisionMarker(u'cpi marker'))
    # ----------------------------------------------
    post_processing = failure_is_success(Sequence)(u'post planning')
    # post_processing.add_child(WiggleCancel(u'final wiggle detection', final_detection=True))
    if god_map.get_data(identifier.enable_PlotTrajectory):
        post_processing.add_child(PlotTrajectory(u'plot trajectory', order=3))
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
