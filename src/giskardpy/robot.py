from collections import namedtuple, OrderedDict

from urdf_parser_py.urdf import URDF
import sympy as sp

Joint = namedtuple('Joint', ['symbol', 'velocity_limit', 'lower', 'upper'])


class Robot(object):
    def __init__(self, weights):
        self.urdf_robot = None
        self.joints = OrderedDict()
        # TODO: fk function for each end effector

        # TODO loop over joints and fill lists
        self.observables = []
        self.joints_observables = []

        # hard stuff
        self.lbA = []  # hard lb
        self.ubA = []  # hard ub
        self.hard_expressions = []

        # controllable stuff
        self.lb = []  # joint lb
        self.ub = []  # joint ub
        self.weights = weights

    def get_updates(self):
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

        for joint_name in self.urdf_robot.get_chain(root_link, tip_link, True, False, False):
            joint = self.urdf_robot.joint_map[joint_name]
            self.joints[joint_name] = Joint(sp.Symbol(joint_name),
                                            joint.limit.velocity,
                                            joint.limit.lower,
                                            joint.limit.upper)

    def load_from_urdf(self, urdf_robot, root_link, tip_links):
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
        for tip_link in tip_links:
            self.add_chain_joints(root_link, tip_link)

        for joint_name, joint in self.joints.items():
            joint_symbol = joint.symbol
            self.joints_observables.append(joint_symbol)

            # hard constraints
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
