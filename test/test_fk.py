#!/usr/bin/env python
import unittest
from collections import OrderedDict
from simplejson import OrderedDict
from time import time

import PyKDL
import numpy as np

from tf.transformations import quaternion_from_matrix, quaternion_multiply, quaternion_about_axis
from urdf_parser_py.urdf import URDF

from giskardpy import USE_SYMENGINE
from giskardpy.donbot import DonBot
from giskardpy.eef_position_controller import EEFPositionControl
from giskardpy.pointy_bot import PointyBot
from giskardpy.joint_space_control import JointSpaceControl
from giskardpy.pr2 import PR2
from giskardpy.robot import hacky_urdf_parser_fix
from kdl_parser import kdl_tree_from_urdf_model

if USE_SYMENGINE:
    import giskardpy.symengine_wrappers as spw
else:
    import giskardpy.sympy_wrappers as spw

PKG = 'giskardpy'


class KDL(object):
    def __init__(self, urdf, start, end):
        if urdf[-5:] == '.urdf':
            with open(urdf, 'r') as file:
                urdf = file.read()
        r = URDF.from_xml_string(hacky_urdf_parser_fix(urdf))
        tree = kdl_tree_from_urdf_model(r)
        self.chain = tree.getChain(start, end)
        self.fksolver = PyKDL.ChainFkSolverPos_recursive(self.chain)
        self.jac_solver = PyKDL.ChainJntToJacSolver(self.chain)
        self.jacobian = PyKDL.Jacobian(self.chain.getNrOfJoints())
        self.joints = self.get_joints()

    def get_joints(self):
        joints = []
        for i in range(self.chain.getNrOfSegments()):
            joint = self.chain.getSegment(i).getJoint()
            if joint.getType() != 8:
                joints.append(str(joint.getName()))
        return joints

    def get_tip_nr(self, tip):
        for i in range(self.chain.getNrOfSegments()):
            link_name = str(self.chain.getSegment(i).getName())
            if link_name == tip:
                return i

    def fk(self, js_dict, tip):
        js = [js_dict[j] for j in self.joints]
        seg_nr = self.get_tip_nr(tip)
        f = PyKDL.Frame()
        joint_array = PyKDL.JntArray(len(js))
        for i in range(len(js)):
            joint_array[i] = js[i]
        self.fksolver.JntToCart(joint_array, f, seg_nr+1)
        return f

    def fk_np(self, js_dict, tip):
        f = self.fk(js_dict, tip)
        r = [[f.M[0, 0], f.M[0, 1], f.M[0, 2], f.p[0]],
             [f.M[1, 0], f.M[1, 1], f.M[1, 2], f.p[1]],
             [f.M[2, 0], f.M[2, 1], f.M[2, 2], f.p[2]],
             [0, 0, 0, 1], ]
        return np.array(r)

    def l_to_jnt_array(self, q):
        j = PyKDL.JntArray(len(q))
        for i in range(len(q)):
            j[i] = q[i]
        return j

    def get_jacobian(self, js):
        self.jac_solver.JntToJac(self.l_to_jnt_array(js), self.jacobian)
        np_jacobian = []
        for i in range(6):
            row = []
            for j in range(len(js)):
                row.append(self.jacobian[i,j])
            np_jacobian.append(row)
        return np.array(np_jacobian)


class TestFK(unittest.TestCase):
    def set_js(self, robot, joint_names, js):
        r = []
        for i, k in enumerate(joint_names):
            r.append((k, js[i]))
        r = OrderedDict(r)
        robot.set_joint_state(r)


    def compare_fk_jacobian_chain(self, robot, robot_kdl, js):
        joint_names = robot.get_joint_names()
        for i in range(robot_kdl.chain.getNrOfSegments()):
            tip = str(robot_kdl.chain.getSegment(i).getName())
            print('checking {}'.format(tip))
            js_dict = {k: js[i] for (i, k) in enumerate(joint_names)}
            self.set_js(robot,joint_names, js)
            eef_pose = robot.link_fk(tip)
            kdl_pose = robot_kdl.fk_np(js_dict, tip)
            np.testing.assert_array_almost_equal(eef_pose, kdl_pose)
            kdl_pose[0,3]=0
            kdl_pose[1,3]=0
            kdl_pose[2,3]=0
            tip_q = np.array(spw.quaternion_make_unique(robot.q_rot[tip]).subs(js_dict).T).astype(float)[0]
            self.assertTrue(tip_q[-1]>=0)
            kdl_q = robot_kdl.fk(js_dict, tip).M.GetQuaternion()
            try:
                np.testing.assert_array_almost_equal(tip_q, kdl_q)
            except AssertionError as e:
                np.testing.assert_array_almost_equal(-tip_q, kdl_q)


            kdl_aa = robot_kdl.fk(js_dict, tip).M.GetRot()
            spw_axis, spw_angle = spw.axis_angle_from_matrix(eef_pose)
            spw_aa = spw_axis*spw_angle
            if spw.nan not in spw_aa:
                np.testing.assert_array_almost_equal([x for x in kdl_aa], np.array(spw_aa).T[0].astype(float))

            if i == robot_kdl.chain.getNrOfSegments()-1:
                kdl_jacobian = robot_kdl.get_jacobian(js)
                sp_jacobian = robot.get_jacobian(tip, js_dict)
                np.testing.assert_array_almost_equal(kdl_jacobian, sp_jacobian, decimal=3)


    def test_default_pr2(self):
        l_arm = OrderedDict([('torso_lift_joint', 0),
                             ('l_shoulder_pan_joint', 0),
                             ('l_shoulder_lift_joint', 0),
                             ('l_upper_arm_roll_joint', 0),
                             ('l_elbow_flex_joint', 0),
                             ('l_forearm_roll_joint', 0),
                             ('l_wrist_flex_joint', 0),
                             ('l_wrist_roll_joint', 0)])
        r_arm = OrderedDict([('torso_lift_joint', 0),
                             ('r_shoulder_pan_joint', 0),
                             ('r_shoulder_lift_joint', 0),
                             ('r_upper_arm_roll_joint', 0),
                             ('r_elbow_flex_joint', 0),
                             ('r_forearm_roll_joint', 0),
                             ('r_wrist_flex_joint', 0),
                             ('r_wrist_roll_joint', 0)])

        left_arm = KDL('pr2.urdf', 'base_link', 'l_gripper_tool_frame')
        right_arm = KDL('pr2.urdf', 'base_link', 'r_gripper_tool_frame')
        pr2 = PR2()

        js = [0.] * 8
        self.set_js(pr2, l_arm, js)
        self.set_js(pr2, r_arm, js)
        eef_pose1 = np.array(pr2.get_eef_position()['l_gripper_tool_frame']).astype(float)
        eef_pose2 = np.array(pr2.get_eef_position()['r_gripper_tool_frame']).astype(float)
        np.testing.assert_array_almost_equal(eef_pose1, left_arm.fk_np(js))
        np.testing.assert_array_almost_equal(eef_pose2, right_arm.fk_np(js))

        js = [0.1] * 8
        self.set_js(pr2, l_arm, js)
        self.set_js(pr2, r_arm, js)
        eef_pose1 = np.array(pr2.get_eef_position()['l_gripper_tool_frame']).astype(float)
        eef_pose2 = np.array(pr2.get_eef_position()['r_gripper_tool_frame']).astype(float)
        np.testing.assert_array_almost_equal(eef_pose1, left_arm.fk_np(js))
        np.testing.assert_array_almost_equal(eef_pose2, right_arm.fk_np(js))

        js = [-0.1] * 8
        self.set_js(pr2, l_arm, js)
        self.set_js(pr2, r_arm, js)
        eef_pose1 = np.array(pr2.get_eef_position()['l_gripper_tool_frame']).astype(float)
        eef_pose2 = np.array(pr2.get_eef_position()['r_gripper_tool_frame']).astype(float)
        np.testing.assert_array_almost_equal(eef_pose1, left_arm.fk_np(js))
        np.testing.assert_array_almost_equal(eef_pose2, right_arm.fk_np(js))

        js = [(.2 if x % 2 == 0 else -.2) for x in range(8)]
        self.set_js(pr2, l_arm, js)
        self.set_js(pr2, r_arm, js)
        eef_pose1 = np.array(pr2.get_eef_position()['l_gripper_tool_frame']).astype(float)
        eef_pose2 = np.array(pr2.get_eef_position()['r_gripper_tool_frame']).astype(float)
        np.testing.assert_array_almost_equal(eef_pose1, left_arm.fk_np(js))
        np.testing.assert_array_almost_equal(eef_pose2, right_arm.fk_np(js))

    def test_pointy(self):
        eef = 'eef'
        r_kdl = KDL('pointy.urdf', 'base_link', eef)
        r = PointyBot(1)

        number_of_joints = len(r.get_joint_names())

        js = [0.] * number_of_joints
        self.compare_fk_jacobian_chain(r, r_kdl, js)

        js = [0.1] * number_of_joints
        self.compare_fk_jacobian_chain(r, r_kdl, js)

        js = [-0.2] * number_of_joints
        self.compare_fk_jacobian_chain(r, r_kdl, js)

        js = [(.3 if x % 2 == 0 else -.1) for x in range(number_of_joints)]
        self.compare_fk_jacobian_chain(r, r_kdl, js)

    def test_base(self):
        eef = 'eef'
        r_kdl = KDL('2d_base_bot.urdf', 'base_link', eef)
        r = PointyBot(1, urdf='2d_base_bot.urdf')

        number_of_joints = len(r.get_joint_names())

        js = [0.0] * number_of_joints
        self.compare_fk_jacobian_chain(r, r_kdl, js)

        js = [0.1] * number_of_joints
        self.compare_fk_jacobian_chain(r, r_kdl, js)

        js = [-0.2] * number_of_joints
        self.compare_fk_jacobian_chain(r, r_kdl, js)

        js = [(.3 if x % 2 == 0 else -.1) for x in range(number_of_joints)]
        self.compare_fk_jacobian_chain(r, r_kdl, js)

    def test_donbot(self):
        eef = 'gripper_tool_frame'
        r_kdl = KDL('iai_donbot.urdf', 'base_footprint', eef)
        r = DonBot(urdf_path='iai_donbot.urdf')

        number_of_joints = len(r.get_joint_names())

        # js = [0.01] * number_of_joints
        # self.compare_fk_jacobian_chain(r, r_kdl, js)

        js = [0.1] * number_of_joints
        self.compare_fk_jacobian_chain(r, r_kdl, js)

        js = [-0.2] * number_of_joints
        self.compare_fk_jacobian_chain(r, r_kdl, js)

        js = [(.3 if x % 2 == 0 else -.1) for x in range(number_of_joints)]
        self.compare_fk_jacobian_chain(r, r_kdl, js)

    def test_fixed(self):
        eef = 'eef'
        r_kdl = KDL('fixed.urdf', 'base_link', eef)
        r = PointyBot(1, urdf='fixed.urdf', tip=eef)

        self.compare_fk_jacobian_chain(r, r_kdl, [0.0])
        self.compare_fk_jacobian_chain(r, r_kdl, [0.1])
        self.compare_fk_jacobian_chain(r, r_kdl, [1.0])
        self.compare_fk_jacobian_chain(r, r_kdl, [-0.1])
        self.compare_fk_jacobian_chain(r, r_kdl, [-1.0])


if __name__ == '__main__':
    import rosunit

    rosunit.unitrun(package=PKG,
                    test_name='TestFK',
                    test=TestFK)
