
import giskardpy
from giskardpy.data_types import SingleJointState
from giskardpy.tfwrapper import msg_to_kdl, kdl_to_pose

giskardpy.WORLD_IMPLEMENTATION = None

import shutil
from collections import OrderedDict

import PyKDL
import pytest
from urdf_parser_py.urdf import URDF

from giskardpy.robot import Robot
from utils_for_tests import rnd_joint_state, pr2_urdf, donbot_urdf, boxy_urdf, base_bot_urdf, compare_poses
from giskardpy.urdf_object import hacky_urdf_parser_fix
from kdl_parser import kdl_tree_from_urdf_model
import numpy as np
from hypothesis import given

PKG = u'giskardpy'

np.random.seed(23)

@pytest.fixture(scope=u'module')
def module_setup(request):
    pass

@pytest.fixture()
def function_setup(request, module_setup):
    pass

@pytest.fixture()
def parsed_pr2(function_setup):
    """
    :rtype: Robot
    """
    return Robot(pr2_urdf())


@pytest.fixture()
def parsed_base_bot(function_setup):
    """
    :rtype: Robot
    """
    return Robot(base_bot_urdf())

@pytest.fixture()
def parsed_donbot(function_setup):
    """
    :rtype: Robot
    """
    return Robot(donbot_urdf())

@pytest.fixture()
def parsed_boxy(function_setup):
    """
    :rtype: Robot
    """
    return Robot(boxy_urdf())

@pytest.fixture()
def test_folder(request):
    """
    :rtype: str
    """
    folder_name = u'tmp_data/'
    def kill_pybullet():
        shutil.rmtree(folder_name)

    request.addfinalizer(kill_pybullet)
    return folder_name


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
        if urdf.endswith(u'.urdfs'):
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


class TestSymengineController(object):
    pr2_joint_limits = Robot(pr2_urdf()).get_all_joint_limits()
    donbot_joint_limits = Robot(donbot_urdf()).get_all_joint_limits()
    boxy_joint_limits = Robot(boxy_urdf()).get_all_joint_limits()

    def test_constraints_pr2(self, parsed_pr2):
        assert len(parsed_pr2.hard_constraints) == 28
        assert len(parsed_pr2.joint_constraints) == 48

    def test_constraints_donbot(self, parsed_donbot):
        assert len(parsed_donbot.hard_constraints) == 9
        assert len(parsed_donbot.joint_constraints) == 10

    def test_constraints_boxy(self, parsed_boxy):
        assert len(parsed_boxy.hard_constraints) == 26
        assert len(parsed_boxy.joint_constraints) == 26

    @given(rnd_joint_state(pr2_joint_limits))
    def test_pr2_fk1(self, parsed_pr2, js):
        """
        :type parsed_pr2: Robot
        :type js:
        :return:
        """
        kdl = KDL(pr2_urdf())
        root = u'odom_combined'
        tips = [u'l_gripper_tool_frame', u'r_gripper_tool_frame']
        for tip in tips:
            kdl_r = kdl.get_robot(root, tip)
            kdl_fk = kdl_to_pose(kdl_r.fk(js))
            mjs = {}
            for joint_name, position in js.items():
                mjs[joint_name] = SingleJointState(joint_name, position)
            parsed_pr2.joint_state = mjs
            symengine_fk = parsed_pr2.get_fk_pose(root, tip).pose
            compare_poses(kdl_fk, symengine_fk)

    @given(rnd_joint_state(donbot_joint_limits))
    def test_donbot_fk1(self, parsed_donbot, js):
        kdl = KDL(donbot_urdf())
        root = u'base_footprint'
        tips = [u'gripper_tool_frame']
        for tip in tips:
            kdl_r = kdl.get_robot(root, tip)
            kdl_fk = kdl_to_pose(kdl_r.fk(js))
            mjs = {}
            for joint_name, position in js.items():
                mjs[joint_name] = SingleJointState(joint_name, position)
            parsed_donbot.joint_state = mjs
            symengine_fk = parsed_donbot.get_fk_pose(root, tip).pose
            compare_poses(kdl_fk, symengine_fk)

    @given(rnd_joint_state(boxy_joint_limits))
    def test_boxy_fk1(self, parsed_boxy, js):
        kdl = KDL(boxy_urdf())
        root = u'base_footprint'
        tips = [u'left_gripper_tool_frame', u'right_gripper_tool_frame']
        for tip in tips:
            kdl_r = kdl.get_robot(root, tip)
            kdl_fk = kdl_to_pose(kdl_r.fk(js))
            mjs = {}
            for joint_name, position in js.items():
                mjs[joint_name] = SingleJointState(joint_name, position)
            parsed_boxy.joint_state = mjs
            symengine_fk = parsed_boxy.get_fk_pose(root, tip).pose
            compare_poses(kdl_fk, symengine_fk)


    def test_get_controllable_joint_names_pr2(self, parsed_pr2):
        expected = {u'l_shoulder_pan_joint', u'br_caster_l_wheel_joint', u'r_gripper_l_finger_tip_joint',
                    u'r_elbow_flex_joint', u'torso_lift_joint', u'r_gripper_l_finger_joint', u'r_forearm_roll_joint',
                    u'l_gripper_r_finger_tip_joint', u'r_shoulder_lift_joint', u'fl_caster_rotation_joint',
                    u'l_gripper_motor_screw_joint', u'r_wrist_roll_joint', u'r_gripper_motor_slider_joint',
                    u'l_forearm_roll_joint', u'r_gripper_joint', u'bl_caster_rotation_joint',
                    u'fl_caster_r_wheel_joint', u'l_shoulder_lift_joint', u'head_pan_joint', u'head_tilt_joint',
                    u'fr_caster_l_wheel_joint', u'fl_caster_l_wheel_joint', u'l_gripper_motor_slider_joint',
                    u'br_caster_r_wheel_joint', u'r_gripper_motor_screw_joint', u'r_upper_arm_roll_joint',
                    u'fr_caster_rotation_joint', u'torso_lift_motor_screw_joint', u'bl_caster_l_wheel_joint',
                    u'r_wrist_flex_joint', u'r_gripper_r_finger_tip_joint', u'l_elbow_flex_joint',
                    u'laser_tilt_mount_joint', u'r_shoulder_pan_joint', u'fr_caster_r_wheel_joint',
                    u'l_wrist_roll_joint', u'r_gripper_r_finger_joint', u'bl_caster_r_wheel_joint', u'l_gripper_joint',
                    u'l_gripper_l_finger_tip_joint', u'br_caster_rotation_joint', u'l_gripper_l_finger_joint',
                    u'l_wrist_flex_joint', u'l_upper_arm_roll_joint', u'l_gripper_r_finger_joint',
                    u'odom_x_joint', u'odom_y_joint', u'odom_z_joint'}

        assert set(parsed_pr2.get_joint_names_controllable()).difference(expected) == set()

    def test_get_joint_names_donbot(self, parsed_donbot):
        expected = {u'ur5_wrist_3_joint', u'ur5_elbow_joint', u'ur5_wrist_1_joint', u'odom_z_joint',
                    u'ur5_shoulder_lift_joint', u'odom_y_joint', u'ur5_wrist_2_joint', u'odom_x_joint',
                    u'ur5_shoulder_pan_joint', u'gripper_joint'}

        assert set(parsed_donbot.get_joint_names_controllable()).difference(expected) == set()

    def test_get_joint_names_boxy(self, parsed_boxy):
        expected = {u'left_arm_2_joint', u'neck_joint_end', u'neck_wrist_1_joint', u'right_arm_2_joint',
                    u'right_arm_4_joint', u'neck_wrist_3_joint', u'right_arm_3_joint',
                    u'right_gripper_base_gripper_right_joint', u'left_gripper_base_gripper_right_joint',
                    u'left_arm_0_joint', u'right_gripper_base_gripper_left_joint', u'left_arm_4_joint',
                    u'left_arm_6_joint', u'right_arm_1_joint', u'left_arm_1_joint', u'neck_wrist_2_joint',
                    u'triangle_base_joint', u'neck_elbow_joint', u'right_arm_5_joint', u'left_arm_3_joint',
                    u'neck_shoulder_pan_joint', u'right_arm_0_joint', u'neck_shoulder_lift_joint', u'left_arm_5_joint',
                    u'left_gripper_base_gripper_left_joint', u'right_arm_6_joint'}

        assert set(parsed_boxy.get_joint_names_controllable()).difference(expected) == set()


if __name__ == '__main__':
    import rosunit

    rosunit.unitrun(package=PKG,
                    test_name='TestSymengineController',
                    test=TestSymengineController)
