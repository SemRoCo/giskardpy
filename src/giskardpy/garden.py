import functools

import py_trees
import py_trees_ros
import rospy
from control_msgs.msg import JointTrajectoryControllerState
from giskard_msgs.msg import MoveAction
from py_trees import Sequence, Selector, BehaviourTree, Blackboard
from py_trees.meta import failure_is_success, success_is_failure
from py_trees_ros.trees import BehaviourTree
from rospy import ROSException

import giskardpy.identifier as identifier
import giskardpy.pybullet_wrapper as pbw
from giskardpy import logging
from giskardpy.god_map import GodMap
from giskardpy.input_system import JointStatesInput
from giskardpy.plugin import PluginBehavior
from giskardpy.plugin_action_server import GoalReceived, SendResult, GoalCanceled
from giskardpy.plugin_attached_tf_publicher import TFPlugin
from giskardpy.plugin_cleanup import CleanUp
from giskardpy.plugin_configuration import ConfigurationPlugin
from giskardpy.plugin_goal_reached import GoalReachedPlugin
from giskardpy.plugin_instantaneous_controller import GoalToConstraints, ControllerPlugin
from giskardpy.plugin_interrupts import WiggleCancel
from giskardpy.plugin_kinematic_sim import KinSimPlugin
from giskardpy.plugin_log_trajectory import LogTrajPlugin
from giskardpy.plugin_pybullet import WorldUpdatePlugin, CollisionChecker
from giskardpy.plugin_send_trajectory import SendTrajectory
from giskardpy.plugin_visualization import VisualizationBehavior
from giskardpy.pybullet_world import PyBulletWorld
from giskardpy.utils import create_path, render_dot_tree, KeyDefaultDict
from giskardpy.world_object import WorldObject


def initialize_god_map():
    god_map = GodMap()
    blackboard = Blackboard
    blackboard.god_map = god_map
    god_map.safe_set_data(identifier.rosparam, rospy.get_param(rospy.get_name()))
    god_map.safe_set_data(identifier.robot_description, rospy.get_param(u'robot_description'))
    path_to_data_folder = god_map.safe_get_data(identifier.data_folder)
    # fix path to data folder
    if not path_to_data_folder.endswith(u'/'):
        path_to_data_folder += u'/'
    god_map.safe_set_data(identifier.data_folder, path_to_data_folder)

    # fix nWSR
    nWSR = god_map.safe_get_data(identifier.nWSR)
    if nWSR == u'None':
        nWSR = None
    god_map.safe_set_data(identifier.nWSR, nWSR)

    pbw.start_pybullet(god_map.safe_get_data(identifier.gui))
    while True:
        try:
            controlled_joints = rospy.wait_for_message(u'/whole_body_controller/state',
                                                       JointTrajectoryControllerState,
                                                       timeout=5.0).joint_names
        except ROSException as e:
            logging.logerr(u'state topic not available')
            logging.logerr(e)
        else:
            break

    joint_weight_symbols = KeyDefaultDict(lambda key: god_map.to_symbol(identifier.joint_weights + [key]))


    joint_weights = KeyDefaultDict(lambda key: god_map.get_data(identifier.default_joint_weight_identifier))
    joint_weights.update(god_map.safe_get_data(identifier.joint_weights))
    god_map.safe_set_data(identifier.joint_weights, joint_weights)

    default_joint_vel = god_map.to_symbol(identifier.default_joint_vel_identifier)

    world = PyBulletWorld(god_map.safe_get_data(identifier.gui),
                          blackboard.god_map.safe_get_data(identifier.data_folder))
    robot = WorldObject(god_map.safe_get_data(identifier.robot_description),
                        None,
                        controlled_joints)
    world.add_robot(robot, None, controlled_joints, default_joint_vel, joint_weight_symbols, True)
    js_input = JointStatesInput(blackboard.god_map.to_symbol, world.robot.get_controllable_joints(),
                                identifier.joint_states,
                                suffix=[u'position'])
    world.robot.reinitialize(js_input.joint_map)
    god_map.safe_set_data(identifier.world, world)
    return god_map


def grow_tree():
    action_server_name = u'giskardpy/command'

    god_map = initialize_god_map()
    # ----------------------------------------------
    wait_for_goal = Sequence(u'wait for goal')
    wait_for_goal.add_child(TFPlugin(u'tf'))
    wait_for_goal.add_child(ConfigurationPlugin(u'js'))
    wait_for_goal.add_child(WorldUpdatePlugin(u'pybullet updater'))
    wait_for_goal.add_child(GoalReceived(u'has goal', action_server_name, MoveAction))
    wait_for_goal.add_child(ConfigurationPlugin(u'js'))
    # ----------------------------------------------
    planning = failure_is_success(Selector)(u'planning')
    planning.add_child(GoalCanceled(u'goal canceled', action_server_name))
    # planning.add_child(CollisionCancel(u'in collision', collision_time_threshold))
    planning.add_child(success_is_failure(VisualizationBehavior)(u'visualization'))

    actual_planning = PluginBehavior(u'planning', sleep=0)
    actual_planning.add_plugin(KinSimPlugin(u'kin sim'))
    actual_planning.add_plugin(CollisionChecker(u'coll'))
    # actual_planning.add_plugin(success_is_running(VisualizationBehavior)(u'visualization', enable_visualization))
    actual_planning.add_plugin(ControllerPlugin(u'controller'))
    actual_planning.add_plugin(LogTrajPlugin(u'log'))
    actual_planning.add_plugin(GoalReachedPlugin(u'goal reached'))
    actual_planning.add_plugin(WiggleCancel(u'wiggle'))
    planning.add_child(actual_planning)
    # ----------------------------------------------
    publish_result = failure_is_success(Selector)(u'move robot')
    publish_result.add_child(GoalCanceled(u'goal canceled', action_server_name))
    publish_result.add_child(SendTrajectory(u'send traj'))
    # ----------------------------------------------
    root = Sequence(u'root')
    root.add_child(wait_for_goal)
    root.add_child(GoalToConstraints(u'update constraints', action_server_name))
    root.add_child(planning)
    root.add_child(CleanUp(u'cleanup'))
    root.add_child(publish_result)
    root.add_child(SendResult(u'send result', action_server_name, MoveAction))

    tree = BehaviourTree(root)

    if god_map.safe_get_data(identifier.debug):
        def post_tick(snapshot_visitor, behaviour_tree):
            logging.logdebug(u'\n' + py_trees.display.ascii_tree(behaviour_tree.root,
                                                                 snapshot_information=snapshot_visitor))

        snapshot_visitor = py_trees_ros.visitors.SnapshotVisitor()
        tree.add_post_tick_handler(functools.partial(post_tick, snapshot_visitor))
        tree.visitors.append(snapshot_visitor)
    path = god_map.safe_get_data(identifier.data_folder) + u'tree'
    create_path(path)
    render_dot_tree(root, name=path)

    tree.setup(30)
    return tree
