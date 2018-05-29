#!/usr/bin/env python
import os
from rospkg import RosPack

import rospy

from giskardpy.plugin_action_server import ActionServerPlugin
from giskardpy.application import ROSApplication
from giskardpy.plugin_instantaneous_controller import CartesianBulletControllerPlugin
from giskardpy.plugin_fk import FKPlugin
from giskardpy.plugin_interactive_marker import InteractiveMarkerPlugin
from giskardpy.plugin_joint_state import JointStatePlugin
from giskardpy.plugin_pybullet import PyBulletPlugin
from giskardpy.plugin_set_controlled_joints import SetControlledJointsPlugin
from giskardpy.process_manager import ProcessManager

if __name__ == '__main__':
    rospy.init_node('muh')
    print(os.getcwd())

    # TODO do we need a solution where we have a different root for some links?

    fk_identifier = 'fk'
    cartesian_goal_identifier = 'goal'
    js_identifier = 'js'
    controlled_joints_identifier = 'controlled_joints'
    trajectory_identifier = 'traj'
    time_identifier = 'time'
    next_cmd_identifier = 'motor'
    collision_identifier = 'collision'
    closest_point_identifier = 'cpi'
    collision_goal_identifier = 'collision_goal'

    root_link = rospy.get_param('~root_link', 'base_footprint')

    pm = ProcessManager()
    pm.register_plugin('js',
                       JointStatePlugin(js_identifier=js_identifier,
                                        time_identifier=time_identifier,
                                        next_cmd_identifier=next_cmd_identifier))
    pm.register_plugin('controlled joints',
                       SetControlledJointsPlugin(controlled_joints_identifier=controlled_joints_identifier))
    pm.register_plugin('action server',
                       ActionServerPlugin(js_identifier=js_identifier,
                                          trajectory_identifier=trajectory_identifier,
                                          cartesian_goal_identifier=cartesian_goal_identifier,
                                          time_identifier=time_identifier,
                                          closest_point_identifier=closest_point_identifier,
                                          controlled_joints_identifier=controlled_joints_identifier,
                                          collision_goal_identifier=collision_goal_identifier,
                                          plot_trajectory=True))
    pm.register_plugin('bullet',
                       PyBulletPlugin(js_identifier=js_identifier,
                                      collision_identifier=collision_identifier,
                                      closest_point_identifier=closest_point_identifier,
                                      collision_goal_identifier=collision_goal_identifier,
                                      map_frame=rospy.get_param('~map_frame', 'map'),
                                      root_link=root_link,
                                      gui=rospy.get_param('~enable_gui', True),
                                      marker=rospy.get_param('~enable_collision_marker', True)))
    pm.register_plugin('fk', FKPlugin(js_identifier=js_identifier, fk_identifier=fk_identifier))
    pm.register_plugin('cart bullet controller',
                       CartesianBulletControllerPlugin(root_link=root_link,
                                                       fk_identifier=fk_identifier,
                                                       goal_identifier=cartesian_goal_identifier,
                                                       js_identifier=js_identifier,
                                                       next_cmd_identifier=next_cmd_identifier,
                                                       collision_identifier=collision_identifier,
                                                       closest_point_identifier=closest_point_identifier,
                                                       controlled_joints_identifier=controlled_joints_identifier,
                                                       collision_goal_identifier=collision_goal_identifier,
                                                       path_to_functions=RosPack().get_path('giskardpy') + '/data/'))
    pm.register_plugin('interactive marker',
                       InteractiveMarkerPlugin(rospy.get_param('~interactive_marker_chains',
                                                               [('base_link', 'r_gripper_tool_frame'),
                                                                ('base_link', 'l_gripper_tool_frame')])))

    app = ROSApplication(pm)
    app.run()
