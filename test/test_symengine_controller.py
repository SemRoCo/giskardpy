import unittest
from collections import OrderedDict

import PyKDL
from urdf_parser_py.urdf import URDF

from giskardpy.robot_constraints import Robot, hacky_urdf_parser_fix
from giskardpy.symengine_controller import JointController
from kdl_parser import kdl_tree_from_urdf_model
import numpy as np

PKG = 'giskardpy'

np.random.seed(23)

def trajectory_rollout(controller, goal, time_limit=10, frequency=100, precision=0.0025):
    current_js = OrderedDict()
    for joint_name in controller.robot.get_joint_to_symbols():
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

    def __init__(self, urdf):
        if urdf.endswith('.urdf'):
            with open(urdf, 'r') as file:
                urdf = file.read()
        r = URDF.from_xml_string(hacky_urdf_parser_fix(urdf))
        self.tree = kdl_tree_from_urdf_model(r)
        self.robots = {}

    def get_robot(self, root, tip):
        if (root, tip) not in self.robots:
            self.chain = self.tree.getChain(root, tip)
            self.robots[root, tip] = self.KDLRobot(self.chain)
        return self.robots[root, tip]

class TestSymengineController(unittest.TestCase):
    def test_constraints1(self):
        r = Robot('pr2.urdf')
        self.assertEqual(len(r.hard_constraints), 26)
        self.assertEqual(len(r.joint_constraints), 42)

    def test_pr2_fk1(self):
        r = Robot('pr2.urdf')
        kdl = KDL('pr2.urdf')
        root = 'base_link'
        tips = ['l_gripper_tool_frame', 'r_gripper_tool_frame']
        for tip in tips:
            kdl_r = kdl.get_robot(root, tip)
            for i in range(20):
                js = r.get_rnd_joint_state()
                kdl_fk = kdl_r.fk_np(js)
                symengine_fk = r.get_fk_expression(root, tip).subs(js)
                np.testing.assert_array_almost_equal(kdl_fk, symengine_fk, decimal=3)

    def test_joint_controller_pointy1(self):
        r = Robot('pointy.urdf')
        jc = JointController('pointy.urdf')
        m = jc.default_goal_symbol_map()
        jc.init(m)
        goal = {}
        for i in range(20):
            for joint_name, joint_value in r.get_rnd_joint_state().items():
                goal[str(m[joint_name])] = joint_value
            end_state = trajectory_rollout(jc, goal)
            for joint_name in end_state:
                self.assertEqual(goal[str(m[joint_name])], end_state[joint_name])

    def test_joint_controller_donbot1(self):
        r = Robot('iai_donbot.urdf')
        jc = JointController('iai_donbot.urdf')
        m = jc.default_goal_symbol_map()
        jc.init(m)
        goal = {}
        for i in range(20):
            for joint_name, joint_value in r.get_rnd_joint_state().items():
                goal[str(m[joint_name])] = joint_value
            end_state = trajectory_rollout(jc, goal)
            for joint_name in end_state:
                self.assertEqual(goal[str(m[joint_name])], end_state[joint_name],
                                 msg='{} is wrong'.format(joint_name))

    def test_joint_controller_2d_base_bot1(self):
        r = Robot('2d_base_bot.urdf')
        jc = JointController('2d_base_bot.urdf')
        m = jc.default_goal_symbol_map()
        jc.init(m)
        goal = {}
        for i in range(20):
            for joint_name, joint_value in r.get_rnd_joint_state().items():
                goal[str(m[joint_name])] = joint_value
            end_state = trajectory_rollout(jc, goal)
            for joint_name in end_state:
                self.assertEqual(goal[str(m[joint_name])], end_state[joint_name],
                                 msg='{} is wrong'.format(joint_name))

    def test_joint_controller_boxy1(self):
        r = Robot('boxy.urdf')
        jc = JointController('boxy.urdf')
        m = jc.default_goal_symbol_map()
        jc.init(m)
        goal = {}
        for i in range(20):
            for joint_name, joint_value in r.get_rnd_joint_state().items():
                goal[str(m[joint_name])] = joint_value
            end_state = trajectory_rollout(jc, goal)
            for joint_name in end_state:
                self.assertEqual(goal[str(m[joint_name])], end_state[joint_name],
                                 msg='{} is wrong'.format(joint_name))

    def test_joint_controller_pr21(self):
        # TODO fix
        r = Robot('pr2.urdf')
        jc = JointController('pr2.urdf')
        m = jc.default_goal_symbol_map()
        jc.init(m)
        goal = {}
        for i in range(20):
            for joint_name, joint_value in r.get_rnd_joint_state().items():
                goal[str(m[joint_name])] = joint_value
            end_state = trajectory_rollout(jc, goal)
            for joint_name in end_state:
                self.assertEqual(goal[str(m[joint_name])], end_state[joint_name],
                                 msg='{} is wrong'.format(joint_name))



if __name__ == '__main__':
    import rosunit

    rosunit.unitrun(package=PKG,
                    test_name='TestSymengineController',
                    test=TestSymengineController)