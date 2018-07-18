#!/usr/bin/env python
import unittest

from giskard_msgs.msg import MoveActionGoal, MoveGoal
from hypothesis.strategies import composite

from giskardpy.exceptions import DuplicateObjectNameException, UnknownBodyException, RobotExistsException
from giskardpy.object import WorldObject, Box, Sphere, Cylinder
from giskardpy.plugin import PluginContainer
from giskardpy.pybullet_world import PyBulletWorld
import pybullet as p
import hypothesis.strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, rule, invariant, initialize
from giskardpy.data_types import MultiJointState, SingleJointState, Transform, Point, Quaternion
from giskardpy.test_utils import variable_name, robot_urdfs
import os
from rospkg import RosPack

import rospy

from giskardpy.plugin_action_server import ActionServerPlugin
from giskardpy.application import ROSApplication
from giskardpy.plugin_instantaneous_controller import CartesianBulletControllerPlugin
from giskardpy.plugin_fk import FKPlugin
from giskardpy.plugin_interactive_marker import InteractiveMarkerPlugin
from giskardpy.plugin_joint_state import JointStatePlugin
from giskardpy.plugin_kinematic_sim import KinematicSimPlugin
from giskardpy.plugin_pybullet import PyBulletPlugin
from giskardpy.plugin_set_controlled_joints import SetControlledJointsPlugin, UploadRobotDescriptionPlugin, \
    UploadUrdfPlugin
from giskardpy.process_manager import ProcessManager


class TestPyBulletWorld(RuleBasedStateMachine):
    def __init__(self):
        super(TestPyBulletWorld, self).__init__()
        rospy.init_node('test_integration')
        self.pm = None

    @initialize(urdf=robot_urdfs())
    def init(self, urdf):
        with open(urdf, u'r') as f:
            urdf = f.read()
        gui = False
        map_frame = u'map'
        joint_convergence_threshold = 0.002
        wiggle_precision_threshold = 7
        sample_period = 0.1
        default_joint_vel_limit = 0.5
        default_collision_avoidance_distance = 0.05
        fill_velocity_values = False
        nWSR = None
        root_link = u'base_footprint'
        marker = True
        enable_self_collision = True
        path_to_data_folder = u'../data/pr2/'
        collision_time_threshold = 15
        max_traj_length = 30

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
        pyfunction_identifier = 'pyfunctions'
        controllable_links_identifier = 'controllable_links'
        robot_description_identifier = 'robot_description'
        self.pm = ProcessManager()
        self.pm.register_plugin('js',
                           JointStatePlugin(js_identifier=js_identifier,
                                            time_identifier=time_identifier,
                                            next_cmd_identifier=next_cmd_identifier,
                                            sample_period=sample_period))
        self.pm.register_plugin('upload robot description',
                           UploadUrdfPlugin(robot_description_identifier=robot_description_identifier,
                                            urdf=urdf))
        self.pm.register_plugin('action server',
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
        self.pm.register_plugin('bullet',
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
        self.pm.register_plugin('fk', FKPlugin(js_identifier=js_identifier,
                                          fk_identifier=fk_identifier,
                                          robot_description_identifier=robot_description_identifier))
        self.pm.register_plugin('cart bullet controller',
                           PluginContainer(
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
        self.pm.start_plugins()

    def loop_once(self):
        self.pm.update()

    # @invariant()
    # def loop(self):
    #     if self.pm is not None:
    #     self.loop_once()

    @rule()
    def set_goal(self):
        self.loop_once()
        goal = MoveGoal()
        self.pm._plugins['action server'].action_server_cb(goal)
        self.loop_once()

    def teardown(self):
        self.pm.stop()

TestTrees = TestPyBulletWorld.TestCase

if __name__ == '__main__':
    # unittest.main()
    state = TestPyBulletWorld()
    state.init(urdf=u'pr2.urdf')
    state.set_goal()
    state.teardown()