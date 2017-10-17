from collections import namedtuple, OrderedDict

from urdf_parser_py.urdf import URDF
from giskardpy.sympy_wrappers import *
from sympy.vector import *
from sympy import *


Joint = namedtuple('Joint', ['symbol', 'velocity_limit', 'lower', 'upper', 'limitless'])


class Robot(object):
    def __init__(self, weights):
        self.urdf_robot = None
        self.joints = OrderedDict()
        # TODO: fk function for each end effector

        # TODO loop over joints and fill lists
        self.observables = []
        self.joints_observables = []
        self.frames = {}

        # hard stuff
        self.lbA = []  # hard lb
        self.ubA = []  # hard ub
        self.hard_expressions = []

        # controllable stuff
        self.lb = []  # joint lb
        self.ub = []  # joint ub
        self.weights = weights

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
            link_name = jointsAndLinks[i+1]
            joint = self.urdf_robot.joint_map[joint_name]

            if joint_name not in self.joints:
                if joint.type == 'revolute' or joint.type == 'continuous':
                    self.joints[joint_name] = Joint(Symbol(joint_name),
                                                  joint.limit.velocity,
                                                  joint.limit.lower,
                                                  joint.limit.upper,
                                                  joint.type == 'continuous')
                    self.frames[link_name] = frame(parentFrame, link_name, AxisOrienter(Symbol(joint_name), vec3(joint.axis)), joint.origin.xyz)
                elif joint.type == 'prismatic':
                    self.joints[joint_name] = Joint(Symbol(joint_name),
                                                    joint.limit.velocity,
                                                    joint.limit.lower,
                                                    joint.limit.upper,
                                                    False)
                    self.frames[link_name] = frame(parentFrame, link_name, joint.origin.rpy, vec3(joint.origin.xyz) + vec3(joint.axis) * Symbol(joint_name))
                elif joint.type == 'fixed':
                    self.frames[link_name] = frame(parentFrame, link_name, joint.origin.rpy, joint.origin.xyz)
                else:
                    raise Exception('Joint type "' + joint.type + '" is not supported by urdf parser.')
            parentFrame = self.frames[link_name]


    def load_from_urdf(self, urdf_robot, root_link, tip_links, odom=odom):
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

        if odom is None:
            root_frame = CoordSys3D(root_link)
        else:
            root_frame = odom.locate_new(root_link, vec3(0,0,0))

        self.frames[root_link] = root_frame

        for tip_link in tip_links:
            self.add_chain_joints(root_link, tip_link)

        for joint_name, joint in self.joints.items():
            joint_symbol = joint.symbol
            self.joints_observables.append(joint_symbol)

            # hard constraints
            if not joint.limitless:
                lbA = joint.lower - joint_symbol
                ubA = joint.upper - joint_symbol
                hard_expression = joint_symbol
                self.lbA.append(lbA)
                self.ubA.append(ubA)
                self.hard_expressions.append(hard_expression)
            # control constraints
            lb = -joint.velocity_limit
            ub = joint.velocity_limit
            self.lb.append(lb)
            self.ub.append(ub)

        self.observables += self.joints_observables

    def load_from_urdf_path(self, urdf_path, root_link, tip_links):
        return self.load_from_urdf(URDF.from_xml_file(urdf_path), root_link, tip_links)

    def load_from_urdf_string(self, urdf_strg, root_link, tip_links):
        return self.load_from_urdf(URDF.from_xml_string(urdf_strg), root_link, tip_links)
