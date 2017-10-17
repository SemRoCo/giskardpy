#!/usr/bin/env python
import unittest

import sympy
from urdf_parser_py.urdf import URDF

from giskardpy.pointy_bot import PointyBot
from giskardpy.joint_space_control import JointSpaceControl
from giskardpy.robot import Robot
from giskardpy.urdf_import import tree_joints_from_urdf_object

PKG = 'giskardpy'


class TestURDFImport(unittest.TestCase):
    def test_tree_joints_from_urdf_object(self):
        urdf = URDF.from_xml_file('boxy.urdf')
        joint_vars = tree_joints_from_urdf_object(urdf, "base_link",
                                                  ["right_gripper_tool_frame", "left_gripper_tool_frame"])
        self.assertEqual(len(joint_vars), 15)

        joint_names = ["right_arm_0_joint", "right_arm_1_joint", "right_arm_2_joint", "right_arm_3_joint",
                       "right_arm_4_joint", "right_arm_5_joint", "right_arm_6_joint", "triangle_base_joint",
                       "left_arm_0_joint", "left_arm_1_joint", "left_arm_2_joint", "left_arm_3_joint",
                       "left_arm_4_joint", "left_arm_5_joint", "left_arm_6_joint"]
        for joint_name in joint_names:
            self.assertEqual(joint_vars[joint_name], sympy.Symbol(joint_name))

    def test_robot_1(self):
        r = PointyBot()
        self.assertEqual(len(r.hard_expressions), 3)

    def test_controller_1(self):
        c = JointSpaceControl(PointyBot())
        self.assertEqual(c.get_num_controllables(), 3)
        self.assertEqual(len(c.get_soft_expressions()), 3)


if __name__ == '__main__':
    import rosunit

    rosunit.unitrun(package=PKG,
                    test_name='TestURDFImport',
                    test=TestURDFImport)
