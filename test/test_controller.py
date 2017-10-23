#!/usr/bin/env python
import unittest
from collections import OrderedDict

import sympy as sp
from urdf_parser_py.urdf import URDF
import numpy as np

from giskardpy.eef_position_controller import EEFPositionControl
from giskardpy.pointy_bot import PointyBot
from giskardpy.joint_space_control import JointSpaceControl
from giskardpy.pr2 import PR2
from giskardpy.qp_problem_builder import QProblemBuilder
from giskardpy.robot import Robot

PKG = 'giskardpy'


class TestController(unittest.TestCase):
    def default_pr2(self):
        pr2 = PR2()
        js = {
            'torso_lift_joint': 0.300026266574,
            'l_elbow_flex_joint': -1.00213547438,
            'l_forearm_roll_joint': 0.834058592757,
            'l_shoulder_lift_joint': 0.103903217692,
            'l_shoulder_pan_joint': 0.3688738798,
            'l_upper_arm_roll_joint': 0.730572260662,
            'l_wrist_flex_joint': -1.34841376457,
            'l_wrist_roll_joint': 7.00870758722,
            'r_elbow_flex_joint': -1.64225215677,
            'r_forearm_roll_joint': 5.26860285279,
            'r_shoulder_lift_joint': 0.573400943385,
            'r_shoulder_pan_joint': -0.976376687461,
            'r_upper_arm_roll_joint': -0.954544248502,
            'r_wrist_flex_joint': -1.50746408471,
            'r_wrist_roll_joint': 1.90604009753,
        }

        pr2.update_observables(js)
        return pr2

    def test_jointcontroller_1(self):
        r = PointyBot(.9)
        c = JointSpaceControl(r, 42)

        start = np.array([.5, 1.05, .35])
        start_dict = {'joint_x': start[0],
                      'joint_y': start[1],
                      'joint_z': start[2]}

        goal = np.array([.7, .8, .9])
        goal_dict = {'joint_x': goal[0],
                     'joint_y': goal[1],
                     'joint_z': goal[2]}

        r.update_observables(start_dict)
        c.set_goal(goal_dict)

        for i in range(20):
            cmd_dict = c.get_next_command()
            self.assertIsNotNone(cmd_dict)
            next_state = OrderedDict()
            robot_state = r.get_state()
            for j, (joint_name, joint_change) in enumerate(cmd_dict.items()):
                next_state[joint_name] = robot_state[joint_name] + joint_change
            r.update_observables(next_state)
            print('iteration #{}: {}'.format(i + 1, r))
        np.testing.assert_array_almost_equal(goal, r.get_state().values())

    def test_jointcontroller_2(self):
        r = PointyBot(weight=.01, urdf='pointy_adrian.urdf')
        c = JointSpaceControl(r, weight=1)

        start = np.array([0, -1., .5])
        start_dict = {'joint_x': start[0],
                      'joint_y': start[1],
                      'joint_z': start[2]}
        goal = np.array([1, 1, -1])
        goal_dict = {'joint_x_goal': goal[0],
                     'joint_y_goal': goal[1],
                     'joint_z_goal': goal[2]}

        r.update_observables(start_dict)
        c.set_goal(goal_dict)

        for i in range(20):
            cmd_dict = c.get_next_command()
            self.assertIsNotNone(cmd_dict)
            next_state = OrderedDict()
            robot_state = r.get_state()
            for j, (joint_name, joint_change) in enumerate(cmd_dict.items()):
                next_state[joint_name] = robot_state[joint_name] + joint_change
            r.update_observables(next_state)
            print('iteration #{}: {}'.format(i + 1, r))
        np.testing.assert_array_almost_equal(goal, r.get_state().values())

    def test_default_pr2(self):
        pr2 = self.default_pr2()
        eef_pose = pr2.get_eef_position()
        np.testing.assert_array_almost_equal(eef_pose['l_gripper_tool_frame'][0], [0.641, 0.054, 1.173], decimal=3)
        np.testing.assert_array_almost_equal(eef_pose['l_gripper_tool_frame'][1], [-0.673, 0.731, 0.090, -0.059],
                                             decimal=3)
        np.testing.assert_array_almost_equal(eef_pose['r_gripper_tool_frame'][0], [0.450, -0.297, 0.940], decimal=3)
        np.testing.assert_array_almost_equal(eef_pose['r_gripper_tool_frame'][1], [-0.192, 0.004, 0.718, 0.669],
                                             decimal=3)

    def test_joint_controller_pr2(self):
        r = self.default_pr2()
        c = JointSpaceControl(r, weight=1)
        joints = ['torso_lift_joint', 'l_elbow_flex_joint', 'l_forearm_roll_joint',
                  'l_shoulder_lift_joint', 'l_shoulder_pan_joint', 'l_upper_arm_roll_joint', 'l_wrist_flex_joint',
                  'l_wrist_roll_joint', 'r_elbow_flex_joint', 'r_forearm_roll_joint', 'r_shoulder_lift_joint',
                  'r_shoulder_pan_joint', 'r_upper_arm_roll_joint', 'r_wrist_flex_joint', 'r_wrist_roll_joint']
        goal = [0.0] * len(joints)
        goal_dict = {'{}_goal'.format(joint): goal[i] for i,joint in enumerate(joints)}
        c.set_goal(goal_dict)

        for i in range(30):
            cmd_dict = c.get_next_command()
            self.assertIsNotNone(cmd_dict)
            next_state = OrderedDict()
            robot_state = r.get_state()
            for j, (joint_name, joint_change) in enumerate(cmd_dict.items()):
                next_state[joint_name] = robot_state[joint_name] + joint_change
            r.update_observables(next_state)
            print('iteration #{}: {}'.format(i + 1, r))
        np.testing.assert_array_almost_equal(goal, r.get_state().values())

        # def test_eef_controller_pr2_1(self):
        #     pr2 = self.default_pr2()
        #     eef_controller = EEFPositionControl(pr2)
        #     eef_controller.set_goal()


if __name__ == '__main__':
    import rosunit

    rosunit.unitrun(package=PKG,
                    test_name='TestController',
                    test=TestController)
