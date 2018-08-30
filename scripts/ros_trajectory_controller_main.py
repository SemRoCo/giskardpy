#!/usr/bin/env python
import os
from rospkg import RosPack

import rospy

from giskardpy.plugin import PluginParallelUniverseOnly
from giskardpy.plugin_action_server import ActionServerPlugin
from giskardpy.application import ROSApplication
from giskardpy.plugin_instantaneous_controller import CartesianBulletControllerPlugin
from giskardpy.plugin_fk import FKPlugin
from giskardpy.plugin_interactive_marker import InteractiveMarkerPlugin
from giskardpy.plugin_joint_state import JointStatePlugin
from giskardpy.plugin_pybullet import PyBulletPlugin
from giskardpy.plugin_set_controlled_joints import SetControlledJointsPlugin, UploadRobotDescriptionPlugin
from giskardpy.process_manager import ProcessManager

def giskard_pm():
    # TODO 0 0 0 in base footprint as goal results in /0
    # TODO bug if first goal is joint
    # TODO you should specify here which plugins get replaced with which during a parallel universe
    # root_link = rospy.get_param('~root_link', 'odom')
    root_tips = rospy.get_param(u'~interactive_marker_chains')
    gui = rospy.get_param(u'~enable_gui')
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

    pm = ProcessManager()
    pm.register_plugin(u'js',
                       JointStatePlugin(js_identifier=js_identifier,
                                        time_identifier=time_identifier,
                                        next_cmd_identifier=next_cmd_identifier,
                                        sample_period=sample_period))
    pm.register_plugin(u'controlled joints',
                       SetControlledJointsPlugin(controlled_joints_identifier=controlled_joints_identifier))
    pm.register_plugin(u'upload robot description',
                       UploadRobotDescriptionPlugin(robot_description_identifier=robot_description_identifier))
    pm.register_plugin(u'action server',
                       ActionServerPlugin(js_identifier=js_identifier,
                                          trajectory_identifier=trajectory_identifier,
                                          cartesian_goal_identifier=cartesian_goal_identifier,
                                          time_identifier=time_identifier,
                                          closest_point_identifier=closest_point_identifier,
                                          controlled_joints_identifier=controlled_joints_identifier,
                                          collision_goal_identifier=collision_goal_identifier,
                                          joint_convergence_threshold=joint_convergence_threshold,
                                          wiggle_precision_threshold=wiggle_precision_threshold,
                                          pyfunction_identifier=pyfunction_identifier,
                                          plot_trajectory=False,
                                          fill_velocity_values=fill_velocity_values,
                                          collision_time_threshold=collision_time_threshold,
                                          max_traj_length=max_traj_length))
    pm.register_plugin(u'bullet',
                       PyBulletPlugin(js_identifier=js_identifier,
                                      collision_identifier=collision_identifier,
                                      closest_point_identifier=closest_point_identifier,
                                      collision_goal_identifier=collision_goal_identifier,
                                      controllable_links_identifier=controllable_links_identifier,
                                      map_frame=map_frame,
                                      root_link=root_link,
                                      path_to_data_folder=path_to_data_folder,
                                      gui=gui,
                                      marker=marker,
                                      default_collision_avoidance_distance=default_collision_avoidance_distance,
                                      enable_self_collision=enable_self_collision,
                                      robot_description_identifier=robot_description_identifier))
    pm.register_plugin(u'fk', FKPlugin(js_identifier=js_identifier,
                                      fk_identifier=fk_identifier,
                                      robot_description_identifier=robot_description_identifier))
    pm.register_plugin(u'cart bullet controller',
                       PluginParallelUniverseOnly(
                           CartesianBulletControllerPlugin(root_link=root_link,
                                                           fk_identifier=fk_identifier,
                                                           goal_identifier=cartesian_goal_identifier,
                                                           js_identifier=js_identifier,
                                                           next_cmd_identifier=next_cmd_identifier,
                                                           collision_identifier=collision_identifier,
                                                           pyfunction_identifier=pyfunction_identifier,
                                                           closest_point_identifier=closest_point_identifier,
                                                           controlled_joints_identifier=controlled_joints_identifier,
                                                           controllable_links_identifier=controllable_links_identifier,
                                                           collision_goal_identifier=collision_goal_identifier,
                                                           path_to_functions=path_to_data_folder,
                                                           nWSR=nWSR,
                                                           default_joint_vel_limit=default_joint_vel_limit,
                                                           robot_description_identifier=robot_description_identifier)))
    pm.register_plugin(u'interactive marker',
                       InteractiveMarkerPlugin(root_tips=root_tips))
    return pm

if __name__ == u'__main__':
    rospy.init_node(u'giskard')
    app = ROSApplication(giskard_pm())
    app.run()

