#!/usr/bin/env python
import unittest
from collections import OrderedDict

import sympy as sp
from urdf_parser_py.urdf import URDF
import numpy as np
from giskardpy.pointy_bot import PointyBot
from giskardpy.joint_space_control import JointSpaceControl
from giskardpy.qp_controller import QPController
from giskardpy.qp_problem_builder import QProblemBuilder
from giskardpy.robot import Robot
from giskardpy.urdf_import import tree_joints_from_urdf_object

PKG = 'giskardpy'


class TestQPController(unittest.TestCase):
    def set_robot_js(self, robot, next_state):
        next_state_dict = OrderedDict()
        for i, joint_symbol in enumerate(robot.joints_observables):
            next_state_dict[joint_symbol] = next_state[i]
        robot.set_joint_state(next_state_dict)

    def set_controller_goal(self, controller, goal):
        goal_dict = OrderedDict()
        for i, goal_symbol in enumerate(controller.get_controller_observables()):
            goal_dict[goal_symbol] = goal[i]
        controller.set_goal(goal_dict)

    def test_qpbuilder_1(self):
        robot_weight = .9
        r = PointyBot(robot_weight)
        control_weight = 42
        c = JointSpaceControl(r, control_weight)

        qpbuilder = QProblemBuilder(c)

        start = np.array([.5, 1.05, .35])
        goal = np.array([.7, .8, .9])

        self.set_robot_js(r, start)
        self.set_controller_goal(c, goal)

        qpbuilder.update()
        A = qpbuilder.get_A()
        expected_A = np.array([[1, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0],
                               [1, 0, 0, 1, 0, 0],
                               [0, 1, 0, 0, 1, 0],
                               [0, 0, 1, 0, 0, 1]])
        np.testing.assert_array_almost_equal(A, expected_A)

        H = qpbuilder.get_H()
        expected_H = np.identity(6) * np.array([robot_weight,
                                                robot_weight,
                                                robot_weight,
                                                control_weight,
                                                control_weight,
                                                control_weight])
        np.testing.assert_array_almost_equal(H, expected_H)

        lb = qpbuilder.get_lb()
        expected_lb = np.array([-0.1, -.3, -.5, -10e6, -10e6, -10e6])
        np.testing.assert_array_equal(lb, expected_lb)

        ub = qpbuilder.get_ub()
        expected_ub = np.array([0.1, .3, .5, 10e6, 10e6, 10e6])
        np.testing.assert_array_equal(ub, expected_ub)

        lbA = qpbuilder.get_lbA()
        expected_lbA = np.array([-1.1, -1.3, -1.5])
        expected_lbA = np.concatenate((expected_lbA - start, goal - start))
        np.testing.assert_array_almost_equal(lbA, expected_lbA)

        ubA = qpbuilder.get_ubA()
        expected_ubA = np.array([1.2, 1.4, 1.6])
        expected_ubA = np.concatenate((expected_ubA - start, goal - start))
        np.testing.assert_array_almost_equal(ubA, expected_ubA)

    def test_jointcontroller_1(self):
        r = PointyBot(.9)
        c = JointSpaceControl(r, 42)
        qpcontroller = QPController(c)

        start = np.array([.5, 1.05, .35])
        goal = np.array([.7, .8, .9])

        self.set_robot_js(r, start)
        self.set_controller_goal(c, goal)

        nWSR = np.array([1000])
        self.assertTrue(qpcontroller.start(nWSR))

        for i in range(20):
            nWSR = np.array([1000])
            self.assertTrue(qpcontroller.update(nWSR))
            cmd = qpcontroller.get_command_vector()
            self.assertTrue(cmd[0] >= -1e-5)
            self.assertTrue(cmd[1] <= 1e-5)
            self.assertTrue(cmd[2] >= -1e-5)
            cmd_dict = OrderedDict()
            for j, (joint_name, joint_position) in enumerate(r.get_updates().items()):
                cmd_dict[joint_name] = joint_position + cmd[j]
            r.set_joint_state(cmd_dict)
            print('iteration #{}: {}'.format(i + 1, r))
        np.testing.assert_array_almost_equal(goal, r.get_updates().values())

    def test_jointcontroller_2(self):
        r = PointyBot(weight=.01, urdf='pointy_adrian.urdf')
        c = JointSpaceControl(r, weights=1)
        qpcontroller = QPController(c)

        start = np.array([0, -1., .5])
        goal = np.array([1, 1, -1])

        self.set_robot_js(r, start)
        self.set_controller_goal(c, goal)

        nWSR = np.array([1000])
        self.assertTrue(qpcontroller.start(nWSR))

        for i in range(20):
            nWSR = np.array([1000])
            self.assertTrue(qpcontroller.update(nWSR))
            cmd = qpcontroller.get_command_vector()
            self.assertTrue(cmd[0] > 0)
            self.assertTrue(cmd[1] > 0)
            self.assertTrue(cmd[2] < 0)
            cmd_dict = OrderedDict()
            for j, (joint_name, joint_position) in enumerate(r.get_updates().items()):
                cmd_dict[joint_name] = joint_position + cmd[j]
            r.set_joint_state(cmd_dict)
            print('iteration #{}: {}'.format(i + 1, r))
        np.testing.assert_array_almost_equal(goal, r.get_updates().values())


if __name__ == '__main__':
    import rosunit

    rosunit.unitrun(package=PKG,
                    test_name='TestQPController',
                    test=TestQPController)
