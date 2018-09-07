import functools
from random import randint
from time import time

import rospy
from control_msgs.msg import JointTrajectoryControllerState
from giskard_msgs.msg import MoveAction
from py_trees import Sequence, Selector, BehaviourTree, Blackboard, Behaviour, Status, Chooser
from py_trees.behaviours import SuccessEveryN, Success, Count, Failure
import py_trees
from py_trees.display import render_dot_tree

from giskardpy.god_map import GodMap
from giskardpy.plugin import PluginBehavior
from giskardpy.plugin_action_server import GoalReceived, GetGoal, SendResult, GoalCanceled
from giskardpy.plugin_fk import NewFkPlugin
from giskardpy.plugin_goal_reached import GoalReachedPlugin
from giskardpy.plugin_instantaneous_controller import GoalToConstraints, ControllerPlugin
from giskardpy.plugin_joint_state import JSBehavior, JointStatePlugin, JointStatePlugin2
from giskardpy.plugin_kinematic_sim import NewKinSimPlugin
from giskardpy.plugin_log_trajectory import NewLogTrajPlugin
from giskardpy.plugin_pybullet import PyBulletMonitor, PyBulletUpdatePlugin, CollisionChecker
from giskardpy.plugin_send_trajectory import SendTrajectory


class MySuccess(Success):
    def update(self):
        print('success')
        return Status.SUCCESS

class Rnd(Behaviour):
    def __init__(self, name, p):
        self.p = p
        super(Rnd, self).__init__(name)

    def update(self):
        if randint(0,self.p) == 0:
            return Status.SUCCESS
        return Status.FAILURE

def ini(param_name, robot_description_identifier, controlled_joints_identifier):
    urdf = rospy.get_param(param_name)
    Blackboard().god_map.set_data([robot_description_identifier], urdf)

    msg = rospy.wait_for_message(u'/whole_body_controller/state',
                                 JointTrajectoryControllerState)  # type: JointTrajectoryControllerState
    Blackboard().god_map.set_data([controlled_joints_identifier], msg.joint_names)

def grow_tree():
    blackboard = Blackboard
    blackboard.god_map = GodMap()

    # gui = rospy.get_param(u'~enable_gui')
    # gui = True
    gui = False
    map_frame = rospy.get_param(u'~map_frame')
    joint_convergence_threshold = rospy.get_param(u'~joint_convergence_threshold')
    wiggle_precision_threshold = rospy.get_param(u'~wiggle_precision_threshold')
    sample_period = rospy.get_param(u'~sample_period')
    default_joint_vel_limit = rospy.get_param(u'~default_joint_vel_limit')
    default_collision_avoidance_distance = rospy.get_param(u'~default_collision_avoidance_distance')
    fill_velocity_values = rospy.get_param(u'~fill_velocity_values')
    nWSR = rospy.get_param(u'~nWSR')
    root_link = rospy.get_param(u'~root_link')
    marker = rospy.get_param(u'~enable_collision_marker')
    enable_self_collision = rospy.get_param(u'~enable_self_collision')
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
    collision_identifier = u'collision'
    closest_point_identifier = u'cpi'
    collision_goal_identifier = u'collision_goal'
    pyfunction_identifier = u'pyfunctions'
    controllable_links_identifier = u'controllable_links'
    robot_description_identifier = u'robot_description'
    pybullet_identifier = u'pybullet_world'
    soft_constraint_identifier = u'soft_constraints'
    action_server_name =  u'giskardpy/command'

    ini(u'robot_description', robot_description_identifier, controlled_joints_identifier)

    #----------------------------------------------
    wait_for_goal = Selector('wait for goal')
    wait_for_goal.add_child(GoalReceived(u'has_goal', action_server_name, MoveAction))
    monitor = PluginBehavior('monitor')
    monitor.add_plugin('js', JointStatePlugin2(js_identifier))
    monitor.add_plugin('fk', NewFkPlugin(fk_identifier, js_identifier, robot_description_identifier))
    monitor.add_plugin('pw', PyBulletMonitor(js_identifier, pybullet_identifier, map_frame, root_link,
                                             path_to_data_folder, gui))
    monitor.add_plugin('pybullet updater', PyBulletUpdatePlugin(pybullet_identifier, robot_description_identifier,
                                                                path_to_data_folder, gui))
    wait_for_goal.add_child(monitor)
    #----------------------------------------------
    planning = Selector('planning')
    planning.add_child(GoalCanceled(u'goal canceled', action_server_name))

    actual_planning = PluginBehavior('actual planning', sleep=0)
    actual_planning.add_plugin('kin sim', NewKinSimPlugin(js_identifier, next_cmd_identifier,
                                                          time_identifier, sample_period))
    actual_planning.add_plugin('fk', NewFkPlugin(fk_identifier, js_identifier, robot_description_identifier))
    actual_planning.add_plugin('pw', PyBulletMonitor(js_identifier, pybullet_identifier, map_frame, root_link,
                                             path_to_data_folder, gui))
    actual_planning.add_plugin('coll', CollisionChecker(collision_goal_identifier, controllable_links_identifier,
                                                        pybullet_identifier, closest_point_identifier,
                                                        default_collision_avoidance_distance,
                                                        enable_self_collision, map_frame, root_link,
                                                        path_to_data_folder, gui))
    actual_planning.add_plugin('controller', ControllerPlugin(robot_description_identifier, js_identifier,
                                                              path_to_data_folder, next_cmd_identifier,
                                                              soft_constraint_identifier, controlled_joints_identifier,
                                                              nWSR))
    actual_planning.add_plugin('log', NewLogTrajPlugin(trajectory_identifier, js_identifier, time_identifier))
    actual_planning.add_plugin('goal reached', GoalReachedPlugin(js_identifier, time_identifier,
                                                                 joint_convergence_threshold))
    planning.add_child(actual_planning)
    # ----------------------------------------------
    parse_goal = Sequence('parse goal')
    parse_goal.add_child(GoalToConstraints(u'update constraints', action_server_name, root_link,
                                           robot_description_identifier, js_identifier, cartesian_goal_identifier,
                                           controlled_joints_identifier, controllable_links_identifier,
                                           fk_identifier, pyfunction_identifier, closest_point_identifier,
                                           soft_constraint_identifier, collision_goal_identifier))

    #----------------------------------------------
    publish_result = Selector('pub result')
    publish_result.add_child(GoalCanceled(u'goal_canceled2', action_server_name))
    publish_result.add_child(SendTrajectory(u'send traj', trajectory_identifier, fill_velocity_values))
    #----------------------------------------------
    main = Sequence('main')
    main.add_child(wait_for_goal)
    main.add_child(parse_goal)
    main.add_child(planning)
    main.add_child(publish_result)
    main.add_child(SendResult(u'send result', action_server_name))
    #----------------------------------------------
    root = Sequence('root')
    root.add_child(main)

    tree = BehaviourTree(root)


    def post_tick(snapshot_visitor, behaviour_tree):
        print("\n" + py_trees.display.ascii_tree(behaviour_tree.root,
                                                 snapshot_information=snapshot_visitor))


    snapshot_visitor = py_trees.visitors.SnapshotVisitor()
    tree.add_post_tick_handler(functools.partial(post_tick, snapshot_visitor))
    tree.visitors.append(snapshot_visitor)

    render_dot_tree(root)

    blackboard.time = time()

    # TODO fail if monitor is not called once
    tree.setup(30)
    return tree

if __name__ == u'__main__':
    rospy.init_node(u'giskard')
    tree = grow_tree()
    while not rospy.is_shutdown():
        try:
            tree.tick()
            rospy.sleep(1)
        except KeyboardInterrupt:
            break
    print("\n")