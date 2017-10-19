import unittest
from collections import OrderedDict

import numpy as np

from giskardpy.pointy_bot import PointyBot
from giskardpy.joint_space_control import JointSpaceControl

PKG = 'giskardpy'


class TestQPBuilder(unittest.TestCase):
    def set_robot_js(self, robot, next_state):
        next_state_dict = OrderedDict()
        for i, joint_symbol in enumerate(robot.joints_observables):
            next_state_dict[str(joint_symbol)] = next_state[i]
        robot.set_joint_state(next_state_dict)

    def test_qpbuilder_1(self):
        big = 1e9
        robot_weight = .9
        r = PointyBot(robot_weight)
        control_weight = 42
        c = JointSpaceControl(r, control_weight)

        qpbuilder = c.qp_problem_builder

        start = np.array([.5, 1.05, .35])
        start_dict = {'joint_x': start[0],
                      'joint_y': start[1],
                      'joint_z': start[2]}

        goal_array = np.array([.7,.8,.9])
        goal = {'joint_x_goal': goal_array[0],
                'joint_y_goal': goal_array[1],
                'joint_z_goal': goal_array[2]}

        self.set_robot_js(r, start)
        c.set_goal(goal)

        c.update_observables()
        A = qpbuilder.np_A
        expected_A = np.array([[1, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0],
                               [1, 0, 0, 1, 0, 0],
                               [0, 1, 0, 0, 1, 0],
                               [0, 0, 1, 0, 0, 1]])
        np.testing.assert_array_almost_equal(A, expected_A)

        H = qpbuilder.np_H
        expected_H = np.identity(6) * np.array([robot_weight,
                                                robot_weight,
                                                robot_weight,
                                                control_weight,
                                                control_weight,
                                                control_weight])
        np.testing.assert_array_almost_equal(H, expected_H)

        lb = qpbuilder.np_lb
        expected_lb = np.array([-0.1, -.3, -.5, -big, -big, -big])
        np.testing.assert_array_equal(lb, expected_lb)

        ub = qpbuilder.np_ub
        expected_ub = np.array([0.1, .3, .5, big, big, big])
        np.testing.assert_array_equal(ub, expected_ub)

        lbA = qpbuilder.np_lbA
        expected_lbA = np.array([-1.1, -1.3, -1.5])
        expected_lbA = np.concatenate((expected_lbA - start, goal_array - start))
        np.testing.assert_array_almost_equal(lbA, expected_lbA)

        ubA = qpbuilder.np_ubA
        expected_ubA = np.array([1.2, 1.4, 1.6])
        expected_ubA = np.concatenate((expected_ubA - start, goal_array - start))
        np.testing.assert_array_almost_equal(ubA, expected_ubA)


if __name__ == '__main__':
    import rosunit

    rosunit.unitrun(package=PKG,
                    test_name='TestQPBuilder',
                    test=TestQPBuilder)
