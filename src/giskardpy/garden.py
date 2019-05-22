import functools

import py_trees
import py_trees_ros
import rospy
from control_msgs.msg import JointTrajectoryControllerState
from giskard_msgs.msg import MoveAction
from py_trees import Sequence, Selector, BehaviourTree, Blackboard
from py_trees.meta import success_is_running, failure_is_success, success_is_failure
from py_trees_ros.trees import BehaviourTree

import giskardpy.pybullet_wrapper as pbw
from giskardpy.god_map import GodMap
from giskardpy.identifier import world_identifier, js_identifier, default_joint_weight_identifier, \
    default_joint_vel_identifier
from giskardpy.input_system import JointStatesInput
from giskardpy.plugin import PluginBehavior, SuccessPlugin
from giskardpy.plugin_action_server import GoalReceived, SendResult, GoalCanceled
from giskardpy.plugin_attached_tf_publicher import TFPlugin
from giskardpy.plugin_cleanup import CleanUp
from giskardpy.plugin_configuration import ConfigurationPlugin
from giskardpy.plugin_goal_reached import GoalReachedPlugin
from giskardpy.plugin_instantaneous_controller import GoalToConstraints, ControllerPlugin
from giskardpy.plugin_interrupts import CollisionCancel, WiggleCancel
from giskardpy.plugin_kinematic_sim import KinSimPlugin
from giskardpy.plugin_log_trajectory import LogTrajPlugin
from giskardpy.plugin_pybullet import WorldUpdatePlugin, CollisionChecker
from giskardpy.plugin_send_trajectory import SendTrajectory
from giskardpy.pybullet_world import PyBulletWorld
from giskardpy.utils import create_path, render_dot_tree
from giskardpy.plugin_visualization import VisualizationBehavior
from giskardpy.world_object import WorldObject
from giskardpy import logging

def initialize_blackboard(urdf, default_joint_vel_limit, default_joint_weight, path_to_data_folder, gui):
    pbw.start_pybullet(gui)
    controlled_joints = rospy.wait_for_message(u'/joint_goal_splitter/state',
                                               JointTrajectoryControllerState).joint_names

    blackboard = Blackboard
    blackboard.god_map = GodMap()

    blackboard.god_map.safe_set_data(default_joint_weight_identifier, default_joint_weight)
    blackboard.god_map.safe_set_data(default_joint_vel_identifier, default_joint_vel_limit)

    default_joint_weight = blackboard.god_map.to_symbol(default_joint_weight_identifier)
    default_joint_vel = blackboard.god_map.to_symbol(default_joint_vel_identifier)

    world = PyBulletWorld(gui, path_to_data_folder)
    robot = WorldObject(urdf, None, controlled_joints)
    world.add_robot(robot, None, controlled_joints, default_joint_vel, default_joint_weight, True)
    js_input = JointStatesInput(blackboard.god_map.to_symbol, world.robot.get_controllable_joints(), js_identifier,
                                suffix=[u'position'])
    world.robot.reinitialize(js_input.joint_map)
    blackboard.god_map.safe_set_data(world_identifier, world)

def grow_tree():
    gui = rospy.get_param(u'~enable_gui')
    map_frame = rospy.get_param(u'~map_frame')
    enable_visualization = rospy.get_param(u'~enable_visualization')
    debug = rospy.get_param(u'~debug')
    # if debug:
    #     giskardpy.PRINT_LEVEL = DEBUG

    # robot related params
    urdf = rospy.get_param(u'robot_description')
    default_joint_vel_limit = rospy.get_param(u'~default_joint_vel_limit')
    default_joint_weight = rospy.get_param(u'~default_joint_weight')
    root_link = rospy.get_param(u'~root_link')

    # planning related params
    nWSR = rospy.get_param(u'~nWSR')
    if nWSR == u'None':
        nWSR = None
    sample_period = rospy.get_param(u'~sample_period')
    joint_convergence_threshold = rospy.get_param(u'~joint_convergence_threshold')
    wiggle_precision_threshold = rospy.get_param(u'~wiggle_precision_threshold')
    default_collision_avoidance_distance = rospy.get_param(u'~default_collision_avoidance_distance')
    collision_time_threshold = rospy.get_param(u'~collision_time_threshold')
    slerp = rospy.get_param(u'~slerp')
    plot_trajectory = rospy.get_param(u'~plot_trajectory')
    # max_traj_length = rospy.get_param(u'~max_traj_length')

    # output related params
    fill_velocity_values = rospy.get_param(u'~fill_velocity_values')
    action_server_name = u'giskardpy/command'

    # other
    path_to_data_folder = rospy.get_param(u'~path_to_data_folder')
    if not path_to_data_folder.endswith(u'/'):
        path_to_data_folder += u'/'

    marker = rospy.get_param(u'~enable_collision_marker')

    initialize_blackboard(urdf, default_joint_vel_limit, default_joint_weight, path_to_data_folder, gui)

    # ----------------------------------------------
    wait_for_goal = Sequence(u'wait for goal')
    wait_for_goal.add_child(TFPlugin(u'tf'))
    wait_for_goal.add_child(ConfigurationPlugin(u'js', map_frame))
    wait_for_goal.add_child(WorldUpdatePlugin(u'pybullet updater'))
    wait_for_goal.add_child(GoalReceived(u'has goal', action_server_name, MoveAction))
    wait_for_goal.add_child(ConfigurationPlugin(u'js', map_frame))
    # ----------------------------------------------
    planning = failure_is_success(Selector)(u'planning')
    planning.add_child(GoalCanceled(u'goal canceled', action_server_name))
    planning.add_child(CollisionCancel(u'in collision', collision_time_threshold))
    planning.add_child(success_is_failure(VisualizationBehavior)(u'visualization', enable_visualization))

    actual_planning = PluginBehavior(u'planning', sleep=0)
    actual_planning.add_plugin(KinSimPlugin(u'kin sim', sample_period))
    actual_planning.add_plugin(
        CollisionChecker(u'coll', default_collision_avoidance_distance, map_frame, root_link, marker))
    actual_planning.add_plugin(ControllerPlugin(u'controller', path_to_data_folder, nWSR))
    actual_planning.add_plugin(LogTrajPlugin(u'log'))
    actual_planning.add_plugin(GoalReachedPlugin(u'goal reached', joint_convergence_threshold))
    actual_planning.add_plugin(WiggleCancel(u'wiggle', wiggle_precision_threshold))
    planning.add_child(actual_planning)
    # ----------------------------------------------
    publish_result = failure_is_success(Selector)(u'move robot')
    publish_result.add_child(GoalCanceled(u'goal canceled', action_server_name))
    publish_result.add_child(SendTrajectory(u'send traj', fill_velocity_values))
    # ----------------------------------------------
    root = Sequence(u'root')
    root.add_child(wait_for_goal)
    root.add_child(GoalToConstraints(u'update constraints', action_server_name, slerp))
    root.add_child(planning)
    root.add_child(CleanUp(u'cleanup'))
    root.add_child(publish_result)
    root.add_child(SendResult(u'send result', action_server_name, path_to_data_folder, plot_trajectory))

    tree = BehaviourTree(root)

    if debug:
        def post_tick(snapshot_visitor, behaviour_tree):
            logging.logdebug(u'\n' + py_trees.display.ascii_tree(behaviour_tree.root,
                                                      snapshot_information=snapshot_visitor))

        snapshot_visitor = py_trees_ros.visitors.SnapshotVisitor()
        tree.add_post_tick_handler(functools.partial(post_tick, snapshot_visitor))
        tree.visitors.append(snapshot_visitor)
    path = path_to_data_folder + u'/tree'
    create_path(path)
    render_dot_tree(root, name=path)

    tree.setup(30)
    return tree
