from collections import namedtuple, OrderedDict

from tf.transformations import quaternion_from_matrix
from urdf_parser_py.urdf import URDF

from giskardpy.qp_problem_builder import HardConstraint, JointConstraint
from giskardpy.sympy_wrappers import *
import numpy as np

Joint = namedtuple('Joint', ['symbol', 'velocity_limit', 'lower', 'upper', 'limitless'])


class Robot(object):
    def __init__(self):
        self.urdf_robot = None
        self.joints = OrderedDict()

        self.observables = []
        self.joints_observables = []
        self.frames = {}
        self.hard_constraints = OrderedDict()
        self.joint_constraints = OrderedDict()

        self.joint_state = {}

    def update_observables(self):
        return {}

    def add_chain_joints(self, root_link, tip_link):
        """
        Returns a dict with joint names as keys and sympy symbols
        as values for all 1-dof movable robot joints in URDF between
        ROOT_LINK and TIP_LINK.

        :param root_link: str, denoting the root of the kin. chain
        :param tip_link: str, denoting the tip of the kin. chain
        :return: dict{str, sympy.Symbol}, with symbols for all joints in chain
        """

        jointsAndLinks = self.urdf_robot.get_chain(root_link, tip_link, True, True, True)
        parentFrame = self.frames[root_link]
        for i in range(1, len(jointsAndLinks), 2):
            joint_name = jointsAndLinks[i]
            link_name = jointsAndLinks[i + 1]
            joint = self.urdf_robot.joint_map[joint_name]

            if joint_name not in self.joints:
                if joint.type == 'revolute' or joint.type == 'continuous':
                    self.joints[joint_name] = Joint(Symbol(joint_name),
                                                    joint.limit.velocity,
                                                    joint.limit.lower,
                                                    joint.limit.upper,
                                                    joint.type == 'continuous')
                    self.frames[link_name] = parentFrame * frame3((joint.axis, -Symbol(joint_name)), joint.origin.xyz)
                elif joint.type == 'prismatic':
                    self.joints[joint_name] = Joint(Symbol(joint_name),
                                                    joint.limit.velocity,
                                                    joint.limit.lower,
                                                    joint.limit.upper,
                                                    False)
                    self.frames[link_name] = parentFrame * frame3(joint.origin.rpy,
                                                                  point3(joint.origin.xyz) + vec3(joint.axis) * Symbol(
                                                                      joint_name))
                elif joint.type == 'fixed':
                    self.frames[link_name] = parentFrame * frame3(joint.origin.rpy, joint.origin.xyz)
                else:
                    raise Exception('Joint type "' + joint.type + '" is not supported by urdf parser.')
            parentFrame = self.frames[link_name]

    def load_from_urdf(self, urdf_robot, root_link, tip_links, root_frame=frame3([0, 0, 0], [0, 0, 0])):
        """
        Returns a dict with joint names as keys and sympy symbols
        as values for all 1-dof movable robot joints in URDF between
        ROOT_LINK and TIP_LINKS.

        :param urdf_robot: URDF.Robot, obtained from URDF parser.
        :param root_link: str, denoting the root of the kin. tree
        :param tip_links: str, denoting the tips of the kin. tree
        :return: dict{str, sympy.Symbol}, with symbols for all joints in tree
        """
        self.urdf_robot = urdf_robot

        self.frames[root_link] = root_frame
        self.end_effectors = tip_links

        for tip_link in tip_links:
            self.add_chain_joints(root_link, tip_link)

        for i, (joint_name, joint) in enumerate(self.joints.items()):
            joint_symbol = joint.symbol
            self.joints_observables.append(joint_symbol)

            if not joint.limitless:
                self.hard_constraints[joint_name] = HardConstraint(lower=joint.lower - joint_symbol,
                                                                   upper=joint.upper - joint_symbol,
                                                                   expression=joint_symbol)

            self.joint_constraints[joint_name] = JointConstraint(lower=-joint.velocity_limit,
                                                                 upper=joint.velocity_limit,
                                                                 weight=1)

        self.observables += self.joints_observables

    def get_eef_position(self):
        eef = {}
        for end_effector in self.end_effectors:
            evaled_frame = self.frames[end_effector].subs(self.joint_state)
            eef_pos = np.array(posOf(evaled_frame).tolist(), dtype=float)[:-1].reshape(3)
            eef_rot = np.array(rotOf(evaled_frame).tolist(), dtype=float)
            eef_rot = quaternion_from_matrix(eef_rot)
            eef[end_effector] = eef_pos, eef_rot
        return eef

    def set_joint_weight(self, joint_name, weight):
        if joint_name in self.joint_constraints:
            old_constraint = self.joint_constraints[joint_name]
            self.joint_constraints[joint_name] = JointConstraint(lower=old_constraint.lower,
                                                                 upper=old_constraint.upper,
                                                                 weight=weight)
        else:
            for j, c in self.joint_constraints.iteritems():
                print(j + ': ' + str(c))

            raise Exception('Robot does not have controllable constraint for joint "' + joint_name + '"')

    def get_joint_weight(self, joint_name):
        if joint_name in self.joint_constraints:
            return self.joint_constraints[joint_name].weight
        raise Exception('Robot does not have controllable constraint for joint "' + joint_name + '"')

    def load_from_urdf_path(self, urdf_path, root_link, tip_links):
        return self.load_from_urdf(URDF.from_xml_file(urdf_path), root_link, tip_links)

    def load_from_urdf_string(self, urdf_strg, root_link, tip_links):
        return self.load_from_urdf(URDF.from_xml_string(urdf_strg), root_link, tip_links)
