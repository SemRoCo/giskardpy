#!/usr/bin/env python
import unittest
from collections import OrderedDict
from time import time

import numpy as np

from tf.transformations import quaternion_inverse, quaternion_multiply

from giskardpy.cartesian_controller import CartesianController
from giskardpy.cartesian_controller_old import CartesianControllerOld
from giskardpy.cartesian_line_controller import CartesianLineController
from giskardpy.donbot import DonBot
from giskardpy.eef_dist_controller import EEFDiffController
from giskardpy.eef_position_controller import EEFPositionControl
from giskardpy.giskardpy_controller import trajectory_rollout
from giskardpy.pointy_bot import PointyBot
from giskardpy.joint_space_control import JointSpaceControl
from giskardpy.pr2 import PR2
from giskardpy.robot import Robot
from giskardpy.symengine_wrappers import trace, rotation_distance

PKG = 'giskardpy'


class TestController(unittest.TestCase):
    def default_pr2(self):
        pr2 = PR2()
        js = {
            'torso_lift_joint': 0.0,
            'l_elbow_flex_joint': 0.0,
            'l_forearm_roll_joint': 0.0,
            'l_shoulder_lift_joint': 0.0,
            'l_shoulder_pan_joint': 0.0,
            'l_upper_arm_roll_joint': 0.0,
            'l_wrist_flex_joint': 0.0,
            'l_wrist_roll_joint': 0.0,
            'r_elbow_flex_joint': 0.0,
            'r_forearm_roll_joint': 0.0,
            'r_shoulder_lift_joint': 0.0,
            'r_shoulder_pan_joint': 0.0,
            'r_upper_arm_roll_joint': 0.0,
            'r_wrist_flex_joint': 0.0,
            'r_wrist_roll_joint': 0.0,
        }

        pr2.set_joint_state(js)
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

        r.set_joint_state(start_dict)
        c.set_goal(goal_dict)

        for i in range(20):
            cmd_dict = c.get_next_command()
            self.assertIsNotNone(cmd_dict)
            next_state = OrderedDict()
            robot_state = r.get_state()
            for j, (joint_name, joint_change) in enumerate(cmd_dict.items()):
                next_state[joint_name] = robot_state[joint_name] + joint_change
            r.set_joint_state(next_state)
            print('iteration #{}: {}'.format(i + 1, r))
        for k, v in goal_dict.items():
            self.assertAlmostEqual(v, r.get_state()[k])

    def test_jointcontroller_2(self):
        r = PointyBot(weight=.01, urdf='2d_base_bot.urdf')
        c = JointSpaceControl(r, weight=1)

        start = np.array([0, -1., .5])
        start_dict = {'joint_x': start[0],
                      'joint_y': start[1],
                      'rot_z': start[2]}
        goal = np.array([1, 1, -1])
        goal_dict = {'joint_x': goal[0],
                     'joint_y': goal[1],
                     'rot_z': goal[2]}

        r.set_joint_state(start_dict)
        c.set_goal(goal_dict)

        for i in range(20):
            cmd_dict = c.get_next_command()
            self.assertIsNotNone(cmd_dict)
            next_state = OrderedDict()
            robot_state = r.get_state()
            for j, (joint_name, joint_change) in enumerate(cmd_dict.items()):
                next_state[joint_name] = robot_state[joint_name] + joint_change
            r.set_joint_state(next_state)
            print('iteration #{}: {}'.format(i + 1, r))
        for k, v in goal_dict.items():
            self.assertAlmostEqual(v, r.get_state()[k])

    def test_jointcontroller_donbot(self):
        r = DonBot()
        c = JointSpaceControl(r, weight=1)

        goal_dict = {'ur5_shoulder_pan_joint': 0.2,
                     'ur5_shoulder_lift_joint': 0.2,
                     'ur5_elbow_joint': 0.2,
                     'ur5_wrist_1_joint': 0.2,
                     'ur5_wrist_2_joint': 0.2,
                     'ur5_wrist_3_joint': 0.2,}

        c.set_goal(goal_dict)

        for i in range(20):
            cmd_dict = c.get_next_command()
            self.assertIsNotNone(cmd_dict)
            next_state = OrderedDict()
            robot_state = r.get_state()
            for j, (joint_name, joint_change) in enumerate(cmd_dict.items()):
                next_state[joint_name] = robot_state[joint_name] + joint_change
            r.set_joint_state(next_state)
            print('iteration #{}: {}'.format(i + 1, r))
        for k, v in goal_dict.items():
            self.assertAlmostEqual(v, r.get_state()[k])

    # @profile
    def test_joint_controller_pr2(self):
        t = time()
        r = self.default_pr2()
        c = JointSpaceControl(r, weight=1)
        print('time spent on init: {}'.format(time() - t))
        joints = ['torso_lift_joint', 'l_elbow_flex_joint', 'l_forearm_roll_joint',
                  'l_shoulder_lift_joint', 'l_shoulder_pan_joint', 'l_upper_arm_roll_joint', 'l_wrist_flex_joint',
                  'l_wrist_roll_joint', 'r_elbow_flex_joint', 'r_forearm_roll_joint', 'r_shoulder_lift_joint',
                  'r_shoulder_pan_joint', 'r_upper_arm_roll_joint', 'r_wrist_flex_joint', 'r_wrist_roll_joint']
        goal = [0.0] * len(joints)
        goal_dict = {joint: goal[i] for i, joint in enumerate(joints)}
        c.set_goal(goal_dict)
        ts = []
        for i in range(200):
            t = time()
            cmd_dict = c.get_next_command()
            ts.append(time() - t)
            self.assertIsNotNone(cmd_dict)
            next_state = OrderedDict()
            robot_state = r.get_state()
            for j, (joint_name, joint_change) in enumerate(cmd_dict.items()):
                next_state[joint_name] = robot_state[joint_name] + joint_change
            r.set_joint_state(next_state)
        for k, v in goal_dict.items():
            self.assertAlmostEqual(v, r.get_state()[k], places=5)
        print('next cmd took {} mean'.format(np.mean(ts)))
        print('next cmd took {} std'.format(np.std(ts)))

    def test_eef_controller_pointy(self):
        r = PointyBot(1)
        c = EEFPositionControl(r)

        start = np.array([0, -1., .5])
        start_dict = {'joint_x': start[0],
                      'joint_y': start[1],
                      'joint_z': start[2]}
        goal = {'eef': [1, 1, .1]}

        r.set_joint_state(start_dict)
        c.set_goal(goal)

        for i in range(30):
            cmd_dict = c.get_next_command()
            self.assertIsNotNone(cmd_dict)
            next_state = OrderedDict()
            robot_state = r.get_state()
            for j, (joint_name, joint_change) in enumerate(cmd_dict.items()):
                next_state[joint_name] = robot_state[joint_name] + joint_change
            r.set_joint_state(next_state)
            print('iteration #{}: {}'.format(i + 1, r.get_eef_position2()['eef'][4:]))
        np.testing.assert_array_almost_equal(r.get_eef_position2()['eef'][4:], goal['eef'])

    def test_eef_controller_base_bot(self):
        r = PointyBot(1, urdf='2d_base_bot.urdf')
        c = EEFPositionControl(r)

        start = np.array([0, -1., 0])
        start_dict = {'joint_x': start[0],
                      'joint_y': start[1],
                      'rot_z': start[2]}
        goal = {'eef': [1, 0, 0]}

        r.set_joint_state(start_dict)
        c.set_goal(goal)

        for i in range(30):
            cmd_dict = c.get_next_command()
            self.assertIsNotNone(cmd_dict)
            next_state = OrderedDict()
            robot_state = r.get_state()
            for j, (joint_name, joint_change) in enumerate(cmd_dict.items()):
                next_state[joint_name] = robot_state[joint_name] + joint_change
            r.set_joint_state(next_state)
            print('iteration #{}: {}'.format(i + 1, r.get_eef_position2()['eef'][4:]))
        np.testing.assert_array_almost_equal(r.get_eef_position2()['eef'][4:], goal['eef'])

    # @profile
    def test_eef_controller_pr2(self):

        r = self.default_pr2()
        c = EEFPositionControl(r)

        goal = {'l_gripper_tool_frame': [.68, 0.01, 1.19]}

        c.set_goal(goal)

        for i in range(200):
            cmd_dict = c.get_next_command()
            self.assertIsNotNone(cmd_dict)
            next_state = OrderedDict()
            robot_state = r.get_state()
            for j, (joint_name, joint_change) in enumerate(cmd_dict.items()):
                next_state[joint_name] = robot_state[joint_name] + joint_change
            r.set_joint_state(next_state)
            # print('iteration #{}: {}'.format(i + 1, r.get_eef_position()['l_gripper_tool_frame'][:3, 3]))
        np.testing.assert_array_almost_equal(r.get_eef_position2()['l_gripper_tool_frame'][4:],
                                             goal['l_gripper_tool_frame'], decimal=4)

    def test_cart_controller_base_bot(self):
        r = PointyBot(1, urdf='2d_base_bot.urdf')

        c = CartesianController(r)

        start = np.array([0, -1., 0])
        start_dict = {'joint_x': start[0],
                      'joint_y': start[1],
                      'rot_z': start[2]}
        # goal = {'eef': [0.,  0.,  0.,  1, -1, 1.1, 0]}
        # goal = {'eef': [0., 0., -0.09983342, 0.99500417, -1, 1.1, 0]}
        goal = {'eef': [0., 0., 0.52268723, 0.85252452, -1, 1.1, 0]}

        r.set_joint_state(start_dict)
        c.set_goal(goal)

        for i in range(20):
            cmd_dict = c.get_next_command()
            self.assertIsNotNone(cmd_dict)
            next_state = OrderedDict()
            robot_state = r.get_state()
            for j, (joint_name, joint_change) in enumerate(cmd_dict.items()):
                next_state[joint_name] = robot_state[joint_name] + joint_change
            r.set_joint_state(next_state)
            print('iteration #{}: {}'.format(i + 1, r.get_eef_position2()['eef']))
        np.testing.assert_array_almost_equal(r.get_eef_position2()['eef'],
                                             goal['eef'], decimal=4)

    # @profile
    def test_cart_controller_pr2(self):
        t = time()
        # r = self.default_pr2()
        r = PR2()
        # r = Robot()
        c = CartesianControllerOld(r, gain=1)
        # c = CartesianController(r, gain=1)
        eef = 'l_gripper_tool_frame'
        print('init took {}'.format(time() - t))

        # goal = {'l_gripper_tool_frame': [-0.0339, -0.1344, -0.0228,  0.9901, .68, 0.01, 1.19]}
        goal = {eef: [0,0,0,1, .9, 0.2, 0.6]}

        goal_rotation = np.eye(3)

        c.set_goal(goal)

        ts = []
        print('iteration #{}: {}'.format(-1, r.get_eef_position2()[eef]))
        for i in range(1000):
            t = time()
            cmd_dict = c.get_next_command()
            ts.append(time() - t)
            self.assertIsNotNone(cmd_dict)
            next_state = OrderedDict()
            robot_state = r.get_state()
            for j, (joint_name, joint_change) in enumerate(cmd_dict.items()):
                next_state[joint_name] = robot_state[joint_name] + joint_change
            r.set_joint_state(next_state)


        # current_pose = np.array(r.get_eef_position()[eef]).astype(float)
        # current_rotation = current_pose[:3,:3]
        # print(current_rotation)
        # print(np.arccos(-rotation_distance(goal_rotation, current_rotation)+1))

        print('iteration #{}: {}'.format(i + 1, r.get_eef_position2()[eef]))
        np.testing.assert_array_almost_equal(r.get_eef_position2()[eef],
                                             goal[eef], decimal=2)
        print('next cmd took {} mean'.format(np.mean(ts)))
        print('next cmd took {} std'.format(np.std(ts)))

    def test_cart_controller_donbot(self):
        t = time()
        r = DonBot()
        c = CartesianController(r)
        eef = 'gripper_tool_frame'
        print('init took {}'.format(time() - t))

        goal = {eef: [0.5, 0.5, 0.5, -0.5, .5, 0.25, 0.7]}

        c.set_goal(goal)
        print('iteration #{}: {}'.format(-1, r.get_eef_position2()[eef]))

        t = time()
        goal_msg = trajectory_rollout(c, c.get_robot().get_joint_names())
        print('traj rollout took {}s'.format(time()-t))
        end_joint_state = goal_msg.trajectory.points[-1]
        r.set_joint_state(dict(zip(goal_msg.trajectory.joint_names, end_joint_state.positions)))
        np.testing.assert_array_almost_equal(r.get_eef_position2()[eef],
                                             goal[eef], decimal=2)

    def test_cart_line_controller_pr2(self):
        t = time()
        r = self.default_pr2()
        c = CartesianLineController(r)
        print('init took {}'.format(time() - t))

        goal = {'l_gripper_tool_frame': [0,0,0,1, .9, 0.2, 0.6]}

        c.set_goal(goal)

        x1 = r.get_eef_position2()['l_gripper_tool_frame'][4:]
        x2 = goal['l_gripper_tool_frame'][4:]
        x0 = r.get_eef_position2()['l_gripper_tool_frame'][4:]
        print(x1)

        ts = []
        for i in range(200):
            t = time()
            cmd_dict = c.get_next_command()
            ts.append(time() - t)
            self.assertIsNotNone(cmd_dict)
            next_state = OrderedDict()
            robot_state = r.get_state()
            for j, (joint_name, joint_change) in enumerate(cmd_dict.items()):
                next_state[joint_name] = robot_state[joint_name] + joint_change
            r.set_joint_state(next_state)

            x0 = r.get_eef_position2()['l_gripper_tool_frame'][4:]
            d = np.linalg.norm(np.cross((x0 - x1), (x0 - x2))) / np.linalg.norm((x2 - x1))
            # print(d)
            self.assertGreaterEqual(0.15, d)

        print('iteration #{}: {}'.format(i + 1, r.get_eef_position2()['l_gripper_tool_frame']))
        np.testing.assert_array_almost_equal(r.get_eef_position2()['l_gripper_tool_frame'],
                                             goal['l_gripper_tool_frame'], decimal=2)
        print('next cmd took {} mean'.format(np.mean(ts)))
        print('next cmd took {} std'.format(np.std(ts)))

    # @profile
    def test_diff_controller_pr2(self):
        t = time()
        r = self.default_pr2()
        c = EEFDiffController(r)
        print('init took {}'.format(time() - t))
        goal = [0,0,1,0, 0,0,0.1]
        q_goal = goal[:4]
        p_goal = goal[4:]

        c.set_goal(goal)

        eef_r_start = r.get_eef_position2()['r_gripper_tool_frame']
        q1_r_start = eef_r_start[:4]
        p1_r_start = eef_r_start[4:]
        eef_l_start = r.get_eef_position2()['l_gripper_tool_frame']
        q1_l_start = eef_l_start[:4]
        p1_l_start = eef_l_start[4:]

        ts = []
        for i in range(200):
            t = time()
            cmd_dict = c.get_next_command()
            ts.append(time()-t)
            self.assertIsNotNone(cmd_dict)
            next_state = OrderedDict()
            robot_state = r.get_state()
            for j, (joint_name, joint_change) in enumerate(cmd_dict.items()):
                next_state[joint_name] = robot_state[joint_name] + joint_change
            r.set_joint_state(next_state)
        eef_pose = r.get_eef_position2()['l_gripper_tool_frame']
        q1 = eef_pose[:4]
        p1 = eef_pose[4:]
        eef_pose = r.get_eef_position2()['r_gripper_tool_frame']
        q2 = eef_pose[:4]
        p2 = eef_pose[4:]
        np.testing.assert_array_almost_equal(p1 - p2, p_goal)
        np.testing.assert_array_almost_equal(quaternion_multiply(q1, quaternion_inverse(q2)), q_goal)
        self.assertTrue(np.linalg.norm(p1_l_start - p1) > 0.1)
        self.assertTrue(np.linalg.norm(p1_r_start - p2) > 0.1)
        print('next cmd took {} mean'.format(np.mean(ts)))
        print('next cmd took {} std'.format(np.std(ts)))

if __name__ == '__main__':
    import rosunit

    rosunit.unitrun(package=PKG,
                    test_name='TestController',
                    test=TestController)
