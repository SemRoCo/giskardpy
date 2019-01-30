#!/usr/bin/env python
import functools
from time import time

import rospy
from control_msgs.msg import JointTrajectoryControllerState
from giskard_msgs.msg import MoveAction
from py_trees import Sequence, Selector, BehaviourTree, Blackboard
import py_trees
from py_trees.behaviours import SuccessEveryN
from py_trees.display import render_dot_tree
from py_trees.meta import running_is_failure, success_is_running, failure_is_success

import giskardpy
from giskardpy import DEBUG
from giskardpy.god_map import GodMap
from giskardpy.plugin import PluginBehavior, SuccessPlugin
from giskardpy.plugin_action_server import GoalReceived, SendResult, GoalCanceled
from giskardpy.plugin_cleanup import CleanUp
from giskardpy.plugin_fk import FkPlugin
from giskardpy.plugin_goal_reached import GoalReachedPlugin
from giskardpy.plugin_instantaneous_controller import GoalToConstraints, ControllerPlugin
from giskardpy.plugin_interrupts import CollisionCancel, WiggleCancel
from giskardpy.plugin_joint_state import JointStatePlugin
from giskardpy.plugin_kinematic_sim import NewKinSimPlugin
from giskardpy.plugin_log_trajectory import NewLogTrajPlugin
from giskardpy.plugin_pybullet import PyBulletMonitor, PyBulletUpdatePlugin, CollisionChecker
from giskardpy.plugin_send_trajectory import SendTrajectory
from giskardpy.utils import create_path, resolve_ros_iris


# TODO add transform3d to package xml
# TODO add pytest to package xml
# TODO move to src folder


def ini(param_name, robot_description_identifier, controlled_joints_identifier):
    # TODO this should be part of sync
    urdf = rospy.get_param(param_name)
    urdf = resolve_ros_iris(urdf)
    Blackboard().god_map.safe_set_data([robot_description_identifier], urdf)

    msg = rospy.wait_for_message(u'/whole_body_controller/state',
                                 JointTrajectoryControllerState)  # type: JointTrajectoryControllerState
    Blackboard().god_map.safe_set_data([controlled_joints_identifier], msg.joint_names)


def grow_tree():
    blackboard = Blackboard
    blackboard.god_map = GodMap()

    gui = rospy.get_param(u'~enable_gui')
    map_frame = rospy.get_param(u'~map_frame')
    debug = rospy.get_param(u'~debug')
    if debug:
        giskardpy.PRINT_LEVEL = DEBUG
    # tree_tick_rate = rospy.get_param(u'~tree_tick_rate')
    joint_convergence_threshold = rospy.get_param(u'~joint_convergence_threshold')
    wiggle_precision_threshold = rospy.get_param(u'~wiggle_precision_threshold')
    sample_period = rospy.get_param(u'~sample_period')
    default_joint_vel_limit = rospy.get_param(u'~default_joint_vel_limit')
    default_joint_weight = rospy.get_param(u'~default_joint_weight')
    default_collision_avoidance_distance = rospy.get_param(u'~default_collision_avoidance_distance')
    fill_velocity_values = rospy.get_param(u'~fill_velocity_values')
    nWSR = rospy.get_param(u'~nWSR')
    root_link = rospy.get_param(u'~root_link')
    marker = rospy.get_param(u'~enable_collision_marker')
    # enable_self_collision = rospy.get_param(u'~enable_self_collision')
    if nWSR == u'None':
        nWSR = None
    path_to_data_folder = rospy.get_param(u'~path_to_data_folder')
    collision_time_threshold = rospy.get_param(u'~collision_time_threshold')
    max_traj_length = rospy.get_param(u'~max_traj_length')
    # path_to_data_folder = '/home/ichumuh/giskardpy_ws/src/giskardpy/data/pr2'
    if not path_to_data_folder.endswith(u'/'):
        path_to_data_folder += u'/'

    fk_identifier = u'fk'
    cartesian_goal_identifier = u'goal'
    js_identifier = u'js'
    controlled_joints_identifier = u'controlled_joints'
    trajectory_identifier = u'traj'
    time_identifier = u'time'
    next_cmd_identifier = u'motor'
    closest_point_identifier = u'cpi'
    collision_goal_identifier = u'collision_goal'
    pyfunction_identifier = u'pyfunctions'
    controllable_links_identifier = u'controllable_links'
    robot_description_identifier = u'robot_description'
    pybullet_identifier = u'pybullet_world'
    soft_constraint_identifier = u'soft_constraints'
    action_server_name = u'giskardpy/command'

    ini(u'robot_description', robot_description_identifier, controlled_joints_identifier)

    # ----------------------------------------------
    sync = PluginBehavior(u'sync')
    sync.add_plugin(u'js', JointStatePlugin(js_identifier))
    sync.add_plugin(u'fk', FkPlugin(fk_identifier, js_identifier, robot_description_identifier))
    sync.add_plugin(u'pw', PyBulletMonitor(js_identifier, pybullet_identifier, controlled_joints_identifier,
                                           map_frame, root_link, path_to_data_folder, gui))
    sync.add_plugin(u'in sync', SuccessPlugin())
    # ----------------------------------------------
    wait_for_goal = Selector(u'wait for goal')
    wait_for_goal.add_child(GoalReceived(u'has goal', action_server_name, MoveAction))
    monitor = PluginBehavior(u'monitor')
    monitor.add_plugin(u'js', JointStatePlugin(js_identifier))
    monitor.add_plugin(u'fk', FkPlugin(fk_identifier, js_identifier, robot_description_identifier))
    monitor.add_plugin(u'pw', PyBulletMonitor(js_identifier, pybullet_identifier, controlled_joints_identifier,
                                              map_frame, root_link, path_to_data_folder, gui))
    monitor.add_plugin(u'pybullet updater', PyBulletUpdatePlugin(pybullet_identifier, controlled_joints_identifier,
                                                                 robot_description_identifier, path_to_data_folder,
                                                                 gui))
    wait_for_goal.add_child(monitor)
    # ----------------------------------------------
    planning = failure_is_success(Selector)(u'planning')
    planning.add_child(GoalCanceled(u'goal canceled', action_server_name))
    planning.add_child(CollisionCancel(u'in collision', collision_time_threshold, time_identifier,
                                       closest_point_identifier))

    actual_planning = PluginBehavior(u'planning', sleep=0)
    actual_planning.add_plugin(u'kin sim', NewKinSimPlugin(js_identifier, next_cmd_identifier,
                                                           time_identifier, sample_period))
    actual_planning.add_plugin(u'fk', FkPlugin(fk_identifier, js_identifier, robot_description_identifier))
    actual_planning.add_plugin(u'pw', PyBulletMonitor(js_identifier, pybullet_identifier, controlled_joints_identifier,
                                                      map_frame, root_link, path_to_data_folder, gui))
    actual_planning.add_plugin(u'coll', CollisionChecker(collision_goal_identifier, controllable_links_identifier,
                                                         pybullet_identifier, controlled_joints_identifier,
                                                         closest_point_identifier, default_collision_avoidance_distance,
                                                         map_frame, root_link,
                                                         path_to_data_folder, gui))
    actual_planning.add_plugin(u'controller', ControllerPlugin(robot_description_identifier, js_identifier,
                                                               path_to_data_folder, next_cmd_identifier,
                                                               soft_constraint_identifier, controlled_joints_identifier,
                                                               default_joint_vel_limit, default_joint_weight, nWSR))
    actual_planning.add_plugin(u'log', NewLogTrajPlugin(trajectory_identifier, js_identifier, time_identifier))
    actual_planning.add_plugin(u'goal reached', GoalReachedPlugin(js_identifier, time_identifier,
                                                                  joint_convergence_threshold))
    actual_planning.add_plugin(u'wiggle', WiggleCancel(wiggle_precision_threshold, js_identifier, time_identifier))
    planning.add_child(actual_planning)
    # ----------------------------------------------


    # ----------------------------------------------
    publish_result = failure_is_success(Selector)(u'move robot')
    publish_result.add_child(GoalCanceled(u'goal canceled', action_server_name))
    publish_result.add_child(SendTrajectory(u'send traj',
                                            trajectory_identifier,
                                            fill_velocity_values))
    # ----------------------------------------------
    root = Sequence(u'root')
    root.add_child(sync)
    root.add_child(wait_for_goal)
    root.add_child(GoalToConstraints(u'update constraints', action_server_name, root_link,
                                     robot_description_identifier, js_identifier, cartesian_goal_identifier,
                                     controlled_joints_identifier, controllable_links_identifier,
                                     fk_identifier, pyfunction_identifier, closest_point_identifier,
                                     soft_constraint_identifier, collision_goal_identifier, default_joint_vel_limit,
                                     default_joint_weight))
    root.add_child(planning)
    root.add_child(CleanUp(u'cleanup', closest_point_identifier))
    root.add_child(publish_result)
    root.add_child(SendResult(u'send result', action_server_name, trajectory_identifier, controlled_joints_identifier,
                              path_to_data_folder))

    tree = BehaviourTree(root)

    if debug:
        def post_tick(snapshot_visitor, behaviour_tree):
            print(u'\n' + py_trees.display.ascii_tree(behaviour_tree.root,
                                                      snapshot_information=snapshot_visitor))

        snapshot_visitor = py_trees.visitors.SnapshotVisitor()
        tree.add_post_tick_handler(functools.partial(post_tick, snapshot_visitor))
        tree.visitors.append(snapshot_visitor)
        path = path_to_data_folder + u'/tree'
        create_path(path)
        render_dot_tree(root, name=path)

    blackboard.time = time()

    # TODO fail if monitor is not called once
    tree.setup(30)
    return tree


if __name__ == u'__main__':
    rospy.init_node(u'giskard')
    tree_tick_rate = rospy.get_param(u'~tree_tick_rate')
    tree = grow_tree()
    while not rospy.is_shutdown():
        try:
            tree.tick()
            rospy.sleep(tree_tick_rate)
        except KeyboardInterrupt:
            break
    print(u'\n')
