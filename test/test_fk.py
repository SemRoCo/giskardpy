#!/usr/bin/env python
import unittest
from collections import OrderedDict
from simplejson import OrderedDict
from time import time

import PyKDL
import numpy as np
from urdf_parser_py.urdf import URDF

from giskardpy.eef_position_controller import EEFPositionControl
from giskardpy.pointy_bot import PointyBot
from giskardpy.joint_space_control import JointSpaceControl
from giskardpy.pr2 import PR2, hacky_urdf_parser_fix
from kdl_parser import kdl_tree_from_urdf_model

PKG = 'giskardpy'


class KDL(object):
    def __init__(self, urdf, start, end):
        r = URDF.from_xml_string(hacky_urdf_parser_fix(urdf))
        tree = kdl_tree_from_urdf_model(r)
        self.chain = tree.getChain(start, end)
        self.fksolver = PyKDL.ChainFkSolverPos_recursive(self.chain)

    def fk(self, js):
        f = PyKDL.Frame()
        joint_array = PyKDL.JntArray(len(js))
        for i in range(len(js)):
            joint_array[i] = js[i]
        self.fksolver.JntToCart(joint_array, f)
        r = [[f.M[0, 0], f.M[0, 1], f.M[0, 2], f.p[0]],
             [f.M[1, 0], f.M[1, 1], f.M[1, 2], f.p[1]],
             [f.M[2, 0], f.M[2, 1], f.M[2, 2], f.p[2]],
             [0, 0, 0, 1], ]
        return np.array(r)


class TestFK(unittest.TestCase):
    def set_js(self, robot, arm, js):
        r = []
        for i, k in enumerate(arm):
            r.append((k, js[i]))
        r = OrderedDict(r)
        robot.set_joint_state(r)

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
        eef_pose = pr2.get_eef_position()
        np.testing.assert_array_almost_equal(eef_pose['l_gripper_tool_frame'], left_arm.fk(js))
        np.testing.assert_array_almost_equal(eef_pose['r_gripper_tool_frame'], right_arm.fk(js))

        js = [0.1] * 8
        self.set_js(pr2, l_arm, js)
        self.set_js(pr2, r_arm, js)
        eef_pose = pr2.get_eef_position()
        np.testing.assert_array_almost_equal(eef_pose['l_gripper_tool_frame'], left_arm.fk(js))
        np.testing.assert_array_almost_equal(eef_pose['r_gripper_tool_frame'], right_arm.fk(js))

        js = [-0.1] * 8
        self.set_js(pr2, l_arm, js)
        self.set_js(pr2, r_arm, js)
        eef_pose = pr2.get_eef_position()
        np.testing.assert_array_almost_equal(eef_pose['l_gripper_tool_frame'], left_arm.fk(js))
        np.testing.assert_array_almost_equal(eef_pose['r_gripper_tool_frame'], right_arm.fk(js))

        js = [(.2 if x % 2 == 0 else -.2) for x in range(8)]
        self.set_js(pr2, l_arm, js)
        self.set_js(pr2, r_arm, js)
        eef_pose = pr2.get_eef_position()
        np.testing.assert_array_almost_equal(eef_pose['l_gripper_tool_frame'], left_arm.fk(js))
        np.testing.assert_array_almost_equal(eef_pose['r_gripper_tool_frame'], right_arm.fk(js))

    def test_pointy(self):
        r_kdl = KDL('pointy.urdf', 'base_link', 'eef')
        r = PointyBot(1)
        head = OrderedDict([('joint_x', 0),
                            ('joint_y', 0),
                            ('joint_z', 0)])

        js = [0.]*3
        self.set_js(r, head, js)
        eef_pose = r.get_eef_position()
        np.testing.assert_array_almost_equal(eef_pose['eef'], r_kdl.fk(js))

        js = [0.1]*3
        self.set_js(r, head, js)
        eef_pose = r.get_eef_position()
        np.testing.assert_array_almost_equal(eef_pose['eef'], r_kdl.fk(js))

        js = [-0.2]*3
        self.set_js(r, head, js)
        eef_pose = r.get_eef_position()
        np.testing.assert_array_almost_equal(eef_pose['eef'], r_kdl.fk(js))

        js = [(.3 if x % 2 == 0 else -.1) for x in range(3)]
        self.set_js(r, head, js)
        eef_pose = r.get_eef_position()
        np.testing.assert_array_almost_equal(eef_pose['eef'], r_kdl.fk(js))

    def test_base(self):
        r_kdl = KDL('2d_base_bot.urdf', 'base_link', 'eef')
        r = PointyBot(1, urdf='2d_base_bot.urdf')
        head = OrderedDict([('joint_x', 0),
                            ('joint_y', 0),
                            ('rot_z', 0)])

        js = [0.]*3
        self.set_js(r, head, js)
        eef_pose = r.get_eef_position()
        np.testing.assert_array_almost_equal(eef_pose['eef'], r_kdl.fk(js))

        js = [0.1]*3
        self.set_js(r, head, js)
        eef_pose = r.get_eef_position()
        np.testing.assert_array_almost_equal(eef_pose['eef'], r_kdl.fk(js))

        js = [-0.2]*3
        self.set_js(r, head, js)
        eef_pose = r.get_eef_position()
        np.testing.assert_array_almost_equal(eef_pose['eef'], r_kdl.fk(js))

        js = [(.3 if x % 2 == 0 else -.1) for x in range(3)]
        self.set_js(r, head, js)
        eef_pose = r.get_eef_position()
        np.testing.assert_array_almost_equal(eef_pose['eef'], r_kdl.fk(js))


if __name__ == '__main__':
    import rosunit

    rosunit.unitrun(package=PKG,
                    test_name='TestFK',
                    test=TestFK)
