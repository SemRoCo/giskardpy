import unittest
from collections import OrderedDict

import PyKDL
from urdf_parser_py.urdf import URDF

from giskardpy.symengine_robot import Robot, hacky_urdf_parser_fix
from giskardpy.test_utils import pr2_joint_state, rnd_joint_state
from kdl_parser import kdl_tree_from_urdf_model
import numpy as np
import giskardpy.symengine_wrappers as sw
from hypothesis import given, reproduce_failure, assume
import hypothesis.strategies as st

PKG = u'giskardpy'

np.random.seed(23)


def trajectory_rollout(controller, goal, time_limit=10, frequency=100, precision=0.0025):
    current_js = OrderedDict()
    for joint_name in controller.robot.joint_states_input.joint_map:
        current_js[joint_name] = 0.0
    state = OrderedDict()
    state.update(current_js)
    state.update(goal)
    for i in range(100):
        next_cmd = controller.get_cmd(state)
        for joint_name, vel in next_cmd.items():
            current_js[joint_name] += vel
        state.update(current_js)
    return current_js


class KDL(object):
    class KDLRobot(object):
        def __init__(self, chain):
            self.chain = chain
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

        def fk(self, js_dict):
            js = [js_dict[j] for j in self.joints]
            f = PyKDL.Frame()
            joint_array = PyKDL.JntArray(len(js))
            for i in range(len(js)):
                joint_array[i] = js[i]
            self.fksolver.JntToCart(joint_array, f)
            return f

        def fk_np(self, js_dict):
            f = self.fk(js_dict)
            r = [[f.M[0, 0], f.M[0, 1], f.M[0, 2], f.p[0]],
                 [f.M[1, 0], f.M[1, 1], f.M[1, 2], f.p[1]],
                 [f.M[2, 0], f.M[2, 1], f.M[2, 2], f.p[2]],
                 [0, 0, 0, 1], ]
            return np.array(r)

        def fk_np_inv(self, js_dict):
            f = self.fk(js_dict).Inverse()
            r = [[f.M[0, 0], f.M[0, 1], f.M[0, 2], f.p[0]],
                 [f.M[1, 0], f.M[1, 1], f.M[1, 2], f.p[1]],
                 [f.M[2, 0], f.M[2, 1], f.M[2, 2], f.p[2]],
                 [0, 0, 0, 1], ]
            return np.array(r)

    def __init__(self, urdf):
        if urdf.endswith(u'.urdf'):
            with open(urdf, u'r') as file:
                urdf = file.read()
        r = URDF.from_xml_string(hacky_urdf_parser_fix(urdf))
        self.tree = kdl_tree_from_urdf_model(r)
        self.robots = {}

    def get_robot(self, root, tip):
        root = str(root)
        tip = str(tip)
        if (root, tip) not in self.robots:
            self.chain = self.tree.getChain(root, tip)
            self.robots[root, tip] = self.KDLRobot(self.chain)
        return self.robots[root, tip]

donbot_urdf = u'../test/urdfs/iai_donbot.urdf'
pr2_urdf = u'../test/urdfs/pr2.urdf'
body_urdf = u'../test/urdfs/boxy.urdf'

class TestSymengineController(unittest.TestCase):
    pr2_joint_limits = Robot.from_urdf_file(pr2_urdf).get_joint_limits()
    donbot_joint_limits = Robot.from_urdf_file(donbot_urdf).get_joint_limits()
    boxy_joint_limits = Robot.from_urdf_file(body_urdf).get_joint_limits()

    def test_constraints_pr2(self):
        r = Robot.from_urdf_file(pr2_urdf)
        self.assertEqual(len(r.hard_constraints), 26)
        self.assertEqual(len(r.joint_constraints), 45)

    def test_constraints_donbot(self):
        r = Robot.from_urdf_file(donbot_urdf)
        self.assertEqual(len(r.hard_constraints), 9)
        self.assertEqual(len(r.joint_constraints), 10)

    def test_constraints_boxy(self):
        r = Robot.from_urdf_file(body_urdf)
        self.assertEqual(len(r.hard_constraints), 26)
        self.assertEqual(len(r.joint_constraints), 26)

    @given(rnd_joint_state(pr2_joint_limits))
    def test_pr2_fk1(self, js):
        r = Robot.from_urdf_file(pr2_urdf)
        kdl = KDL(pr2_urdf)
        root = u'base_link'
        tips = [u'l_gripper_tool_frame', u'r_gripper_tool_frame']
        for tip in tips:
            kdl_r = kdl.get_robot(root, tip)
            kdl_fk = kdl_r.fk_np(js)
            symengine_fk = r.get_fk_expression(root, tip).subs(js)
            np.testing.assert_array_almost_equal(kdl_fk, symengine_fk, decimal=3)
            np.testing.assert_array_almost_equal(kdl_r.fk_np_inv(js), sw.inverse_frame(symengine_fk), decimal=3)
        # self.assertTrue(False)

    @given(rnd_joint_state(donbot_joint_limits))
    def test_donbot_fk1(self, js):
        r = Robot.from_urdf_file(donbot_urdf)
        kdl = KDL(donbot_urdf)
        root = u'base_footprint'
        tips = [u'gripper_tool_frame']
        for tip in tips:
            kdl_r = kdl.get_robot(root, tip)
            kdl_fk = kdl_r.fk_np(js)
            symengine_fk = r.get_fk_expression(root, tip).subs(js)
            np.testing.assert_array_almost_equal(kdl_fk, symengine_fk, decimal=3)
            np.testing.assert_array_almost_equal(kdl_r.fk_np_inv(js), sw.inverse_frame(symengine_fk), decimal=3)

    @given(rnd_joint_state(boxy_joint_limits))
    def test_donbot_fk1(self, js):
        r = Robot.from_urdf_file(body_urdf)
        kdl = KDL(body_urdf)
        root = u'base_footprint'
        tips = [u'left_gripper_tool_frame', u'right_gripper_tool_frame']
        for tip in tips:
            kdl_r = kdl.get_robot(root, tip)
            kdl_fk = kdl_r.fk_np(js)
            symengine_fk = r.get_fk_expression(root, tip).subs(js)
            np.testing.assert_array_almost_equal(kdl_fk, symengine_fk, decimal=3)
            np.testing.assert_array_almost_equal(kdl_r.fk_np_inv(js), sw.inverse_frame(symengine_fk), decimal=3)

    def test_get_sub_tree_link_names_with_collision_boxy(self):
        expected = {u'left_arm_2_joint': {u'left_gripper_finger_left_link', u'left_arm_6_link',
                                          u'left_gripper_gripper_left_link', u'left_arm_5_link',
                                          u'left_gripper_base_link', u'left_arm_7_link',
                                          u'left_gripper_finger_right_link', u'left_arm_3_link', u'left_arm_4_link',
                                          u'left_gripper_gripper_right_link'},
                    u'neck_joint_end': {u'neck_look_target'},
                    u'neck_wrist_1_joint': {u'neck_look_target', u'neck_adapter_iso50_kinect2_frame_in',
                                            u'neck_wrist_3_link', u'neck_wrist_2_link', u'neck_ee_link',
                                            u'head_mount_kinect2_rgb_optical_frame', u'neck_wrist_1_link'},
                    u'right_arm_2_joint': {u'right_gripper_finger_right_link', u'right_arm_3_link', u'right_arm_5_link',
                                           u'right_gripper_gripper_right_link', u'right_gripper_gripper_left_link',
                                           u'right_arm_6_link', u'right_gripper_base_link', u'right_arm_4_link',
                                           u'right_arm_7_link', u'right_gripper_finger_left_link'},
                    u'right_arm_4_joint': {u'right_gripper_finger_right_link', u'right_arm_5_link',
                                           u'right_gripper_gripper_right_link', u'right_gripper_base_link',
                                           u'right_arm_6_link', u'right_gripper_gripper_left_link', u'right_arm_7_link',
                                           u'right_gripper_finger_left_link'},
                    u'neck_wrist_3_joint': {u'neck_look_target', u'neck_adapter_iso50_kinect2_frame_in',
                                            u'neck_ee_link', u'head_mount_kinect2_rgb_optical_frame',
                                            u'neck_wrist_3_link'},
                    u'right_arm_3_joint': {u'right_gripper_finger_right_link', u'right_arm_5_link',
                                           u'right_gripper_gripper_right_link', u'right_gripper_base_link',
                                           u'right_arm_6_link', u'right_gripper_gripper_left_link', u'right_arm_4_link',
                                           u'right_arm_7_link', u'right_gripper_finger_left_link'},
                    u'right_gripper_base_gripper_right_joint': {u'right_gripper_finger_right_link',
                                                                u'right_gripper_gripper_right_link'},
                    u'left_gripper_base_gripper_right_joint': {u'left_gripper_gripper_right_link',
                                                               u'left_gripper_finger_right_link'},
                    u'left_arm_0_joint': {u'left_gripper_finger_left_link', u'left_arm_6_link',
                                          u'left_gripper_gripper_left_link', u'left_arm_5_link',
                                          u'left_gripper_base_link', u'left_arm_1_link', u'left_arm_7_link',
                                          u'left_gripper_finger_right_link', u'left_arm_3_link', u'left_arm_4_link',
                                          u'left_arm_2_link', u'left_gripper_gripper_right_link'},
                    u'right_gripper_base_gripper_left_joint': {u'right_gripper_gripper_left_link',
                                                               u'right_gripper_finger_left_link'},
                    u'left_arm_4_joint': {u'left_gripper_finger_left_link', u'left_arm_6_link',
                                          u'left_gripper_gripper_left_link', u'left_arm_5_link',
                                          u'left_gripper_base_link', u'left_arm_7_link',
                                          u'left_gripper_finger_right_link', u'left_gripper_gripper_right_link'},
                    u'left_arm_6_joint': {u'left_gripper_finger_left_link', u'left_gripper_gripper_left_link',
                                          u'left_gripper_base_link', u'left_arm_7_link',
                                          u'left_gripper_finger_right_link', u'left_gripper_gripper_right_link'},
                    u'right_arm_1_joint': {u'right_gripper_finger_right_link', u'right_arm_3_link', u'right_arm_5_link',
                                           u'right_gripper_gripper_right_link', u'right_arm_2_link',
                                           u'right_gripper_gripper_left_link', u'right_arm_6_link',
                                           u'right_gripper_base_link', u'right_arm_4_link', u'right_arm_7_link',
                                           u'right_gripper_finger_left_link'},
                    u'left_arm_1_joint': {u'left_gripper_finger_left_link', u'left_arm_6_link',
                                          u'left_gripper_gripper_left_link', u'left_arm_5_link',
                                          u'left_gripper_base_link', u'left_arm_7_link',
                                          u'left_gripper_finger_right_link', u'left_arm_3_link', u'left_arm_4_link',
                                          u'left_arm_2_link', u'left_gripper_gripper_right_link'},
                    u'neck_wrist_2_joint': {u'neck_look_target', u'neck_adapter_iso50_kinect2_frame_in',
                                            u'neck_wrist_3_link', u'neck_wrist_2_link', u'neck_ee_link',
                                            u'head_mount_kinect2_rgb_optical_frame'},
                    u'triangle_base_joint': {u'left_arm_3_link', u'left_gripper_gripper_left_link', u'left_arm_5_link',
                                             u'left_gripper_base_link', u'left_gripper_finger_right_link',
                                             u'left_arm_2_link', u'right_gripper_finger_right_link',
                                             u'left_gripper_finger_left_link', u'right_arm_3_link',
                                             u'calib_right_arm_base_link', u'triangle_base_link', u'right_arm_4_link',
                                             u'right_gripper_finger_left_link', u'left_arm_6_link',
                                             u'calib_left_arm_base_link', u'right_gripper_base_link',
                                             u'right_gripper_gripper_right_link', u'left_arm_1_link',
                                             u'left_arm_7_link', u'right_gripper_gripper_left_link',
                                             u'right_arm_1_link', u'left_arm_4_link', u'right_arm_5_link',
                                             u'right_arm_2_link', u'right_arm_6_link', u'right_arm_7_link',
                                             u'left_gripper_gripper_right_link'},
                    u'neck_elbow_joint': {u'neck_look_target', u'neck_adapter_iso50_kinect2_frame_in',
                                          u'neck_forearm_link', u'neck_wrist_3_link', u'neck_wrist_2_link',
                                          u'neck_ee_link', u'head_mount_kinect2_rgb_optical_frame',
                                          u'neck_wrist_1_link'},
                    u'right_arm_5_joint': {u'right_gripper_finger_right_link', u'right_gripper_gripper_right_link',
                                           u'right_gripper_base_link', u'right_arm_6_link',
                                           u'right_gripper_gripper_left_link', u'right_arm_7_link',
                                           u'right_gripper_finger_left_link'},
                    u'left_arm_3_joint': {u'left_gripper_finger_left_link', u'left_arm_6_link',
                                          u'left_gripper_gripper_left_link', u'left_arm_5_link',
                                          u'left_gripper_base_link', u'left_arm_7_link',
                                          u'left_gripper_finger_right_link', u'left_arm_4_link',
                                          u'left_gripper_gripper_right_link'},
                    u'neck_shoulder_pan_joint': {u'neck_upper_arm_link', u'neck_look_target',
                                                 u'neck_adapter_iso50_kinect2_frame_in', u'neck_forearm_link',
                                                 u'neck_wrist_3_link', u'neck_wrist_2_link', u'neck_shoulder_link',
                                                 u'head_mount_kinect2_rgb_optical_frame', u'neck_wrist_1_link',
                                                 u'neck_ee_link'},
                    u'right_arm_0_joint': {u'right_gripper_finger_right_link', u'right_arm_3_link', u'right_arm_5_link',
                                           u'right_gripper_gripper_right_link', u'right_arm_2_link',
                                           u'right_gripper_gripper_left_link', u'right_arm_6_link',
                                           u'right_gripper_base_link', u'right_arm_1_link', u'right_arm_4_link',
                                           u'right_arm_7_link', u'right_gripper_finger_left_link'},
                    u'neck_shoulder_lift_joint': {u'neck_upper_arm_link', u'neck_look_target',
                                                  u'neck_adapter_iso50_kinect2_frame_in', u'neck_forearm_link',
                                                  u'neck_wrist_3_link', u'neck_wrist_2_link', u'neck_ee_link',
                                                  u'head_mount_kinect2_rgb_optical_frame', u'neck_wrist_1_link'},
                    u'left_arm_5_joint': {u'left_gripper_finger_left_link', u'left_arm_6_link',
                                          u'left_gripper_gripper_left_link', u'left_gripper_base_link',
                                          u'left_arm_7_link', u'left_gripper_finger_right_link',
                                          u'left_gripper_gripper_right_link'},
                    u'left_gripper_base_gripper_left_joint': {u'left_gripper_finger_left_link',
                                                              u'left_gripper_gripper_left_link'},
                    u'right_arm_6_joint': {u'right_gripper_finger_right_link', u'right_gripper_gripper_right_link',
                                           u'right_gripper_base_link', u'right_gripper_gripper_left_link',
                                           u'right_arm_7_link', u'right_gripper_finger_left_link'}}
        r = Robot.from_urdf_file(body_urdf)
        for joint in r.get_joint_names_controllable():
            self.assertSetEqual(set(r.get_sub_tree_link_names_with_collision(joint)), expected[joint])
            # print(u'u\'{}\': {{{}}},'.format(joint,
            #                                  u', '.join([u'u\'{}\''.format(x) for x in r.get_sub_tree_link_names_with_collision(joint)])))

    def test_get_sub_tree_link_names_with_collision_pr2(self):
        expected = {u'l_shoulder_pan_joint': {u'l_shoulder_pan_link', u'l_shoulder_lift_link', u'l_upper_arm_roll_link', u'l_upper_arm_link', u'l_elbow_flex_link', u'l_forearm_roll_link', u'l_forearm_link', u'l_wrist_flex_link', u'l_wrist_roll_link', u'l_gripper_palm_link', u'l_gripper_l_finger_link', u'l_gripper_r_finger_link', u'l_gripper_l_finger_tip_link', u'l_gripper_r_finger_tip_link'},
                    u'br_caster_l_wheel_joint': {u'br_caster_l_wheel_link'},
                    u'r_gripper_l_finger_tip_joint': {u'r_gripper_l_finger_tip_link'},
                    u'r_elbow_flex_joint': {u'r_elbow_flex_link', u'r_forearm_roll_link', u'r_forearm_link', u'r_wrist_flex_link', u'r_wrist_roll_link', u'r_gripper_palm_link', u'r_gripper_l_finger_link', u'r_gripper_r_finger_link', u'r_gripper_l_finger_tip_link', u'r_gripper_r_finger_tip_link'},
                    u'torso_lift_joint': {u'torso_lift_link', u'head_pan_link', u'laser_tilt_mount_link', u'r_shoulder_pan_link', u'l_shoulder_pan_link', u'head_tilt_link', u'r_shoulder_lift_link', u'l_shoulder_lift_link', u'head_plate_frame', u'r_upper_arm_roll_link', u'l_upper_arm_roll_link', u'r_upper_arm_link', u'l_upper_arm_link', u'r_elbow_flex_link', u'l_elbow_flex_link', u'r_forearm_roll_link', u'l_forearm_roll_link', u'r_forearm_link', u'l_forearm_link', u'r_wrist_flex_link', u'l_wrist_flex_link', u'r_wrist_roll_link', u'l_wrist_roll_link', u'r_gripper_palm_link', u'r_gripper_l_finger_link', u'r_gripper_r_finger_link', u'r_gripper_l_finger_tip_link', u'r_gripper_r_finger_tip_link', u'l_gripper_palm_link', u'l_gripper_l_finger_link', u'l_gripper_r_finger_link', u'l_gripper_l_finger_tip_link', u'l_gripper_r_finger_tip_link'},
                    u'r_gripper_l_finger_joint': {u'r_gripper_l_finger_link', u'r_gripper_l_finger_tip_link'},
                    u'r_forearm_roll_joint': {u'r_forearm_roll_link', u'r_forearm_link', u'r_wrist_flex_link', u'r_wrist_roll_link', u'r_gripper_palm_link', u'r_gripper_l_finger_link', u'r_gripper_r_finger_link', u'r_gripper_l_finger_tip_link', u'r_gripper_r_finger_tip_link'},
                    u'l_gripper_r_finger_tip_joint': {u'l_gripper_r_finger_tip_link'},
                    u'r_shoulder_lift_joint': {u'r_shoulder_lift_link', u'r_upper_arm_roll_link', u'r_upper_arm_link', u'r_elbow_flex_link', u'r_forearm_roll_link', u'r_forearm_link', u'r_wrist_flex_link', u'r_wrist_roll_link', u'r_gripper_palm_link', u'r_gripper_l_finger_link', u'r_gripper_r_finger_link', u'r_gripper_l_finger_tip_link', u'r_gripper_r_finger_tip_link'},
                    u'fl_caster_rotation_joint': {u'fl_caster_rotation_link', u'fl_caster_l_wheel_link', u'fl_caster_r_wheel_link'},
                    u'l_gripper_motor_screw_joint': set(),
                    u'r_wrist_roll_joint': {u'r_wrist_roll_link', u'r_gripper_palm_link', u'r_gripper_l_finger_link', u'r_gripper_r_finger_link', u'r_gripper_l_finger_tip_link', u'r_gripper_r_finger_tip_link'},
                    u'r_gripper_motor_slider_joint': set(),
                    u'l_forearm_roll_joint': {u'l_forearm_roll_link', u'l_forearm_link', u'l_wrist_flex_link', u'l_wrist_roll_link', u'l_gripper_palm_link', u'l_gripper_l_finger_link', u'l_gripper_r_finger_link', u'l_gripper_l_finger_tip_link', u'l_gripper_r_finger_tip_link'},
                    u'r_gripper_joint': set(),
                    u'bl_caster_rotation_joint': {u'bl_caster_rotation_link', u'bl_caster_l_wheel_link', u'bl_caster_r_wheel_link'},
                    u'fl_caster_r_wheel_joint': {u'fl_caster_r_wheel_link'},
                    u'l_shoulder_lift_joint': {u'l_shoulder_lift_link', u'l_upper_arm_roll_link', u'l_upper_arm_link', u'l_elbow_flex_link', u'l_forearm_roll_link', u'l_forearm_link', u'l_wrist_flex_link', u'l_wrist_roll_link', u'l_gripper_palm_link', u'l_gripper_l_finger_link', u'l_gripper_r_finger_link', u'l_gripper_l_finger_tip_link', u'l_gripper_r_finger_tip_link'},
                    u'head_pan_joint': {u'head_pan_link', u'head_tilt_link', u'head_plate_frame'},
                    u'head_tilt_joint': {u'head_tilt_link', u'head_plate_frame'},
                    u'fr_caster_l_wheel_joint': {u'fr_caster_l_wheel_link'},
                    u'fl_caster_l_wheel_joint': {u'fl_caster_l_wheel_link'},
                    u'l_gripper_motor_slider_joint': set(),
                    u'br_caster_r_wheel_joint': {u'br_caster_r_wheel_link'},
                    u'r_gripper_motor_screw_joint': set(),
                    u'r_upper_arm_roll_joint': {u'r_upper_arm_roll_link', u'r_upper_arm_link', u'r_elbow_flex_link', u'r_forearm_roll_link', u'r_forearm_link', u'r_wrist_flex_link', u'r_wrist_roll_link', u'r_gripper_palm_link', u'r_gripper_l_finger_link', u'r_gripper_r_finger_link', u'r_gripper_l_finger_tip_link', u'r_gripper_r_finger_tip_link'},
                    u'fr_caster_rotation_joint': {u'fr_caster_rotation_link', u'fr_caster_l_wheel_link', u'fr_caster_r_wheel_link'},
                    u'torso_lift_motor_screw_joint': set(),
                    u'bl_caster_l_wheel_joint': {u'bl_caster_l_wheel_link'},
                    u'r_wrist_flex_joint': {u'r_wrist_flex_link', u'r_wrist_roll_link', u'r_gripper_palm_link', u'r_gripper_l_finger_link', u'r_gripper_r_finger_link', u'r_gripper_l_finger_tip_link', u'r_gripper_r_finger_tip_link'},
                    u'r_gripper_r_finger_tip_joint': {u'r_gripper_r_finger_tip_link'},
                    u'l_elbow_flex_joint': {u'l_elbow_flex_link', u'l_forearm_roll_link', u'l_forearm_link', u'l_wrist_flex_link', u'l_wrist_roll_link', u'l_gripper_palm_link', u'l_gripper_l_finger_link', u'l_gripper_r_finger_link', u'l_gripper_l_finger_tip_link', u'l_gripper_r_finger_tip_link'},
                    u'laser_tilt_mount_joint': {u'laser_tilt_mount_link'},
                    u'r_shoulder_pan_joint': {u'r_shoulder_pan_link', u'r_shoulder_lift_link', u'r_upper_arm_roll_link', u'r_upper_arm_link', u'r_elbow_flex_link', u'r_forearm_roll_link', u'r_forearm_link', u'r_wrist_flex_link', u'r_wrist_roll_link', u'r_gripper_palm_link', u'r_gripper_l_finger_link', u'r_gripper_r_finger_link', u'r_gripper_l_finger_tip_link', u'r_gripper_r_finger_tip_link'},
                    u'fr_caster_r_wheel_joint': {u'fr_caster_r_wheel_link'},
                    u'l_wrist_roll_joint': {u'l_wrist_roll_link', u'l_gripper_palm_link', u'l_gripper_l_finger_link', u'l_gripper_r_finger_link', u'l_gripper_l_finger_tip_link', u'l_gripper_r_finger_tip_link'},
                    u'r_gripper_r_finger_joint': {u'r_gripper_r_finger_link', u'r_gripper_r_finger_tip_link'},
                    u'bl_caster_r_wheel_joint': {u'bl_caster_r_wheel_link'},
                    u'l_gripper_joint': set(),
                    u'l_gripper_l_finger_tip_joint': {u'l_gripper_l_finger_tip_link'},
                    u'br_caster_rotation_joint': {u'br_caster_rotation_link', u'br_caster_l_wheel_link', u'br_caster_r_wheel_link'},
                    u'l_gripper_l_finger_joint': {u'l_gripper_l_finger_link', u'l_gripper_l_finger_tip_link'},
                    u'l_wrist_flex_joint': {u'l_wrist_flex_link', u'l_wrist_roll_link', u'l_gripper_palm_link', u'l_gripper_l_finger_link', u'l_gripper_r_finger_link', u'l_gripper_l_finger_tip_link', u'l_gripper_r_finger_tip_link'},
                    u'l_upper_arm_roll_joint': {u'l_upper_arm_roll_link', u'l_upper_arm_link', u'l_elbow_flex_link', u'l_forearm_roll_link', u'l_forearm_link', u'l_wrist_flex_link', u'l_wrist_roll_link', u'l_gripper_palm_link', u'l_gripper_l_finger_link', u'l_gripper_r_finger_link', u'l_gripper_l_finger_tip_link', u'l_gripper_r_finger_tip_link'},
                    u'l_gripper_r_finger_joint': {u'l_gripper_r_finger_link', u'l_gripper_r_finger_tip_link'}}
        r = Robot.from_urdf_file(pr2_urdf)
        for joint in r.get_joint_names_controllable():
            self.assertSetEqual(set(r.get_sub_tree_link_names_with_collision(joint)), expected[joint])

    def test_get_sub_tree_link_names_with_collision_donbot(self):
        expected = {u'ur5_wrist_3_joint': {u'ur5_wrist_3_link', u'ur5_ee_link', u'gripper_base_link', u'gripper_gripper_left_link', u'gripper_finger_left_link', u'gripper_gripper_right_link', u'gripper_finger_right_link'},
                    u'ur5_elbow_joint': {u'ur5_forearm_link', u'ur5_wrist_1_link', u'ur5_wrist_2_link', u'ur5_wrist_3_link', u'ur5_ee_link', u'gripper_base_link', u'gripper_gripper_left_link', u'gripper_finger_left_link', u'gripper_gripper_right_link', u'gripper_finger_right_link'},
                    u'ur5_wrist_1_joint': {u'ur5_wrist_1_link', u'ur5_wrist_2_link', u'ur5_wrist_3_link', u'ur5_ee_link', u'gripper_base_link', u'gripper_gripper_left_link', u'gripper_finger_left_link', u'gripper_gripper_right_link', u'gripper_finger_right_link'},
                    u'odom_z_joint': {u'base_link', u'plate', u'ur5_base_link', u'ur5_shoulder_link', u'ur5_upper_arm_link', u'ur5_forearm_link', u'ur5_wrist_1_link', u'ur5_wrist_2_link', u'ur5_wrist_3_link', u'ur5_ee_link', u'gripper_base_link', u'gripper_gripper_left_link', u'gripper_finger_left_link', u'gripper_gripper_right_link', u'gripper_finger_right_link'},
                    u'ur5_shoulder_lift_joint': {u'ur5_upper_arm_link', u'ur5_forearm_link', u'ur5_wrist_1_link', u'ur5_wrist_2_link', u'ur5_wrist_3_link', u'ur5_ee_link', u'gripper_base_link', u'gripper_gripper_left_link', u'gripper_finger_left_link', u'gripper_gripper_right_link', u'gripper_finger_right_link'},
                    u'odom_y_joint': {u'base_link', u'plate', u'ur5_base_link', u'ur5_shoulder_link', u'ur5_upper_arm_link', u'ur5_forearm_link', u'ur5_wrist_1_link', u'ur5_wrist_2_link', u'ur5_wrist_3_link', u'ur5_ee_link', u'gripper_base_link', u'gripper_gripper_left_link', u'gripper_finger_left_link', u'gripper_gripper_right_link', u'gripper_finger_right_link'},
                    u'ur5_wrist_2_joint': {u'ur5_wrist_2_link', u'ur5_wrist_3_link', u'ur5_ee_link', u'gripper_base_link', u'gripper_gripper_left_link', u'gripper_finger_left_link', u'gripper_gripper_right_link', u'gripper_finger_right_link'},
                    u'odom_x_joint': {u'base_link', u'plate', u'ur5_base_link', u'ur5_shoulder_link', u'ur5_upper_arm_link', u'ur5_forearm_link', u'ur5_wrist_1_link', u'ur5_wrist_2_link', u'ur5_wrist_3_link', u'ur5_ee_link', u'gripper_base_link', u'gripper_gripper_left_link', u'gripper_finger_left_link', u'gripper_gripper_right_link', u'gripper_finger_right_link'},
                    u'ur5_shoulder_pan_joint': {u'ur5_shoulder_link', u'ur5_upper_arm_link', u'ur5_forearm_link', u'ur5_wrist_1_link', u'ur5_wrist_2_link', u'ur5_wrist_3_link', u'ur5_ee_link', u'gripper_base_link', u'gripper_gripper_left_link', u'gripper_finger_left_link', u'gripper_gripper_right_link', u'gripper_finger_right_link'},
                    u'gripper_joint': {u'gripper_gripper_right_link', u'gripper_finger_right_link'}}
        r = Robot.from_urdf_file(donbot_urdf)
        for joint in r.get_joint_names_controllable():
            self.assertSetEqual(set(r.get_sub_tree_link_names_with_collision(joint)), expected[joint])

    def test_get_joint_names_pr2(self):
        expected = {u'l_shoulder_pan_joint', u'br_caster_l_wheel_joint', u'r_gripper_l_finger_tip_joint', u'r_elbow_flex_joint', u'torso_lift_joint', u'r_gripper_l_finger_joint', u'r_forearm_roll_joint', u'l_gripper_r_finger_tip_joint', u'r_shoulder_lift_joint', u'fl_caster_rotation_joint', u'l_gripper_motor_screw_joint', u'r_wrist_roll_joint', u'r_gripper_motor_slider_joint', u'l_forearm_roll_joint', u'r_gripper_joint', u'bl_caster_rotation_joint', u'fl_caster_r_wheel_joint', u'l_shoulder_lift_joint', u'head_pan_joint', u'head_tilt_joint', u'fr_caster_l_wheel_joint', u'fl_caster_l_wheel_joint', u'l_gripper_motor_slider_joint', u'br_caster_r_wheel_joint', u'r_gripper_motor_screw_joint', u'r_upper_arm_roll_joint', u'fr_caster_rotation_joint', u'torso_lift_motor_screw_joint', u'bl_caster_l_wheel_joint', u'r_wrist_flex_joint', u'r_gripper_r_finger_tip_joint', u'l_elbow_flex_joint', u'laser_tilt_mount_joint', u'r_shoulder_pan_joint', u'fr_caster_r_wheel_joint', u'l_wrist_roll_joint', u'r_gripper_r_finger_joint', u'bl_caster_r_wheel_joint', u'l_gripper_joint', u'l_gripper_l_finger_tip_joint', u'br_caster_rotation_joint', u'l_gripper_l_finger_joint', u'l_wrist_flex_joint', u'l_upper_arm_roll_joint', u'l_gripper_r_finger_joint'}

        r = Robot.from_urdf_file(pr2_urdf)
        self.assertSetEqual(set(r.get_joint_names_controllable()), expected)
        # print(u', '.join([u'u\'{}\''.format(x) for x in r.get_joint_names()]))

    def test_get_joint_names_donbot(self):
        expected = {u'ur5_wrist_3_joint', u'ur5_elbow_joint', u'ur5_wrist_1_joint', u'odom_z_joint', u'ur5_shoulder_lift_joint', u'odom_y_joint', u'ur5_wrist_2_joint', u'odom_x_joint', u'ur5_shoulder_pan_joint', u'gripper_joint'}

        r = Robot.from_urdf_file(donbot_urdf)
        self.assertSetEqual(set(r.get_joint_names_controllable()), expected)

    def test_get_joint_names_boxy(self):
        expected = {u'left_arm_2_joint', u'neck_joint_end', u'neck_wrist_1_joint', u'right_arm_2_joint', u'right_arm_4_joint', u'neck_wrist_3_joint', u'right_arm_3_joint', u'right_gripper_base_gripper_right_joint', u'left_gripper_base_gripper_right_joint', u'left_arm_0_joint', u'right_gripper_base_gripper_left_joint', u'left_arm_4_joint', u'left_arm_6_joint', u'right_arm_1_joint', u'left_arm_1_joint', u'neck_wrist_2_joint', u'triangle_base_joint', u'neck_elbow_joint', u'right_arm_5_joint', u'left_arm_3_joint', u'neck_shoulder_pan_joint', u'right_arm_0_joint', u'neck_shoulder_lift_joint', u'left_arm_5_joint', u'left_gripper_base_gripper_left_joint', u'right_arm_6_joint'}

        r = Robot.from_urdf_file(body_urdf)
        self.assertSetEqual(set(r.get_joint_names_controllable()), expected)

if __name__ == '__main__':
    import rosunit

    rosunit.unitrun(package=PKG,
                    test_name='TestSymengineController',
                    test=TestSymengineController)
