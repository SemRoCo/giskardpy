#!/usr/bin/env python
import unittest
from matplotlib.font_manager import FontProperties
from time import time

import numpy as np

import actionlib
import rospy
from control_msgs.msg import FollowJointTrajectoryAction, JointTrajectoryControllerState, FollowJointTrajectoryGoal
from copy import deepcopy

from giskardpy.boxy import Boxy
from giskardpy.cartesian_controller import CartesianController
from giskardpy.cartesian_line_controller import CartesianLineController
from giskardpy.donbot import DonBot
from giskardpy.giskardpy_controller import trajectory_rollout
from giskardpy.pointy_bot import PointyBot
from giskardpy.joint_space_control import JointSpaceControl
from giskardpy.pr2 import PR2
import pylab as plt

PKG = 'giskardpy'

def get_rnd_joint_state(robot):
    def f(joint):
        ll, ul = robot.get_joint_limits(joint)
        if ll is None:
            return np.random.random() * np.pi * 2
        ll += 0.02
        ul -= 0.02
        return (np.random.random() * (abs(ll) + abs(ul))) + ll

    js = {joint: f(joint) for joint in robot.get_joint_state().keys()}
    return js

class TestController(unittest.TestCase):
    _ac = None

    def sim(self, controller, goal, plot=False, time_limit=10, execute=False):
        controller.set_goal(goal)
        t = time()
        controller.get_robot().turn_off()
        goal_msg = trajectory_rollout(controller, controller.get_robot().get_joint_names(), precision=0.0025,
                                      time_limit=time_limit, frequency=25)
        print('traj rollout took {}s'.format(time() - t))
        end_joint_state = goal_msg.trajectory.points[-1]
        print('trajectory length {}'.format(len(goal_msg.trajectory.points)))
        controller.get_robot().set_joint_state(dict(zip(goal_msg.trajectory.joint_names, end_joint_state.positions)))
        if plot:
            fontP = FontProperties()
            fontP.set_size('small')
            plt.plot([p.positions for p in goal_msg.trajectory.points])
            plt.legend(goal_msg.trajectory.joint_names, prop=fontP, bbox_to_anchor=(1, 0.5), loc='center left')
            plt.tight_layout()
            plt.subplots_adjust(right=0.7)
            plt.show()
            plt.plot([p.velocities for p in goal_msg.trajectory.points])
            plt.legend(goal_msg.trajectory.joint_names, prop=fontP, bbox_to_anchor=(1, 0.5), loc='center left')
            plt.tight_layout()
            plt.subplots_adjust(right=0.7)
            plt.show()
        if execute:
            if self._ac is None:
                rospy.init_node('controller_tester')
                self._ac = actionlib.SimpleActionClient('/whole_body_controller/follow_joint_trajectory',
                                                        FollowJointTrajectoryAction)
                rospy.sleep(0.5)
            js = rospy.wait_for_message('/whole_body_controller/state', JointTrajectoryControllerState, 2)
            new_goal_msg = FollowJointTrajectoryGoal()
            new_goal_msg.trajectory.joint_names = goal_msg.trajectory.joint_names
            new_goal_msg.trajectory.joint_names.extend(
                [x for x in js.joint_names if x not in new_goal_msg.trajectory.joint_names])
            nr_missing_joints = len(js.joint_names) - len(goal_msg.trajectory.points[0].positions)
            for p in goal_msg.trajectory.points:
                new_p = deepcopy(p)
                new_p.positions.extend([0] * nr_missing_joints)
                new_p.velocities.extend([0] * nr_missing_joints)
                new_goal_msg.trajectory.points.append(new_p)
            print('sending goal')
            self._ac.send_goal_and_wait(new_goal_msg)
            print('finished')

    def check_jointspace_controller_robot(self, robot, plot=False, execute=False, time_limit=20):
        t = time()
        c = JointSpaceControl(robot)
        print('time spent on init: {}'.format(time() - t))
        for i in range(10):
            def f(joint):
                ll, ul = robot.get_joint_limits(joint)
                if ll is None:
                    return np.random.random() * np.pi * 2
                return (np.random.random() * (abs(ll) + abs(ul))) + ll

            goal_dict = {joint: f(joint) for joint in robot.get_joint_state().keys()}

            self.sim(c, goal_dict, plot=plot, execute=execute, time_limit=time_limit)

            for k, v in goal_dict.items():
                self.assertAlmostEqual(v, robot.get_state()[k], places=2,
                                       msg='joint {} failed; expected {}, got {}'.format(k, v, robot.get_state()[k]))

    def check_cartesian_controller_robot(self, controller, plot=False, execute=False, time_limit=60,
                                         x_range=(-100, 100),
                                         y_range=(-100, 100),
                                         z_range=(-100, 100),
                                         reset=False):
        robot = controller.get_robot()
        goals = []
        np.random.seed(23)
        while len(goals) < 10:
            goal_dict = get_rnd_joint_state(robot)
            robot.set_joint_state(goal_dict)
            goal = robot.get_eef_position_quaternion()
            if min([x[-1] for x in goal.values()]) > z_range[0] and \
                    max([x[-1] for x in goal.values()]) < z_range[1] and \
                    min([x[-2] for x in goal.values()]) > y_range[0] and \
                    max([x[-2] for x in goal.values()]) < y_range[1] and \
                    min([x[-3] for x in goal.values()]) > x_range[0] and \
                    max([x[-3] for x in goal.values()]) < x_range[1]:
                goals.append(goal)
                print('num of goal = {}'.format(len(goals)))
        robot.set_joint_state({joint: 0.0 for joint in robot.get_joint_state()})
        failed = False
        for goal in goals:
            if reset:
                robot.set_joint_state({joint: 0.0 for joint in robot.get_joint_state()})
            self.sim(controller, goal, time_limit=time_limit, plot=plot, execute=execute)
            for eef, pose in goal.items():
                actual_pose = controller.get_robot().get_eef_position_quaternion()[eef]
                try:
                    np.testing.assert_array_almost_equal(actual_pose, pose,
                                                         decimal=2,
                                                         err_msg='{} at \n{} instead of \n{}'.format(eef, actual_pose, pose))
                except AssertionError as e:
                    print(e)
                    failed = True
            self.assertTrue(not failed)

    def test_jointcontroller_1(self):
        np.random.seed(23)
        r = PointyBot()
        self.check_jointspace_controller_robot(r)

    def test_jointcontroller_2(self):
        np.random.seed(23)
        r = PointyBot(urdf='2d_base_bot.urdf')
        self.check_jointspace_controller_robot(r)

    def test_jointcontroller_donbot(self):
        np.random.seed(23)
        r = DonBot(default_joint_velocity=1)
        self.check_jointspace_controller_robot(r)

    def test_joint_controller_pr2(self):
        np.random.seed(23)
        r = PR2(default_joint_velocity=1)
        self.check_jointspace_controller_robot(r)

    def test_joint_controller_boxy(self):
        np.random.seed(23)
        r = Boxy(default_joint_velocity=1)
        self.check_jointspace_controller_robot(r, time_limit=120,
                                               execute=True
                                               )

    def test_cart_controller_base_bot(self):
        np.random.seed(23)
        r = PointyBot(urdf='2d_base_bot.urdf')
        c = CartesianController(r)
        self.check_cartesian_controller_robot(c)

    def test_cart_controller_pr2(self):
        np.random.seed(23)
        t = time()
        robot = PR2(default_joint_velocity=1)
        c = CartesianController(robot)
        print('init took {}'.format(time() - t))
        self.check_cartesian_controller_robot(c,
                                              x_range=(0, 0.8),
                                              # y_range=(-0.4, 0.4),
                                              z_range=(0.4, 1.4),
                                              plot=True,
                                              execute=True,
                                              reset=True
                                              )

    def test_cart_controller_boxy(self):
        np.random.seed(23)
        t = time()
        robot = Boxy(default_joint_velocity=1)
        c = CartesianController(robot)
        print('init took {}'.format(time() - t)),
        self.check_cartesian_controller_robot(c, time_limit=120, x_range=(0.3, 100), y_range=(-1, 1), z_range=(0.5, 1.8),
                                              # reset=True,
                                              execute=True,
                                              plot=True
                                              )

    def test_cart_controller_donbot(self):
        np.random.seed(23)
        t = time()
        robot = DonBot(default_joint_velocity=1)
        c = CartesianController(robot)
        print('init took {}'.format(time() - t))
        self.check_cartesian_controller_robot(c, z_range=(0.8, 100))

    def test_cart_line_controller_pr2(self):
        np.random.seed(23)
        t = time()
        r = PR2(default_joint_velocity=1)
        c = CartesianLineController(r)
        print('init took {}'.format(time() - t))
        # self.cartesian_controller_robot_test(c, x_range=(0,100), z_range=(0.5,100), plot=True, execute=True)
        # self.cartesian_controller_robot_test(c, x_range=(0.2,100), z_range=(0.5,100), plot=True)
        self.check_cartesian_controller_robot(c, x_range=(0.2, 100), z_range=(0.5, 100))


if __name__ == '__main__':
    import rosunit

    rosunit.unitrun(package=PKG,
                    test_name='TestController',
                    test=TestController)
