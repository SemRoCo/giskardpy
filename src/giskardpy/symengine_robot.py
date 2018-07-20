import hashlib
from collections import namedtuple, OrderedDict, defaultdict
import numpy as np
import symengine_wrappers as spw
from urdf_parser_py.urdf import URDF, Box, Sphere, Mesh, Cylinder

from giskardpy.input_system import JointStatesInput
from giskardpy.qp_problem_builder import HardConstraint, JointConstraint
from giskardpy.utils import cube_volume, cube_surface, sphere_volume, cylinder_volume, cylinder_surface, keydefaultdict, \
    suppress_stdout, suppress_stderr

Joint = namedtuple('Joint', ['symbol', 'velocity_limit', 'lower', 'upper', 'type', 'frame'])


def hacky_urdf_parser_fix(urdf_str):
    # TODO this function is inefficient but the tested urdf's aren't big enough for it to be a problem
    fixed_urdf = ''
    delete = False
    black_list = ['transmission', 'gazebo']
    black_open = ['<{}'.format(x) for x in black_list]
    black_close = ['</{}'.format(x) for x in black_list]
    for line in urdf_str.split('\n'):
        if len([x for x in black_open if x in line]) > 0:
            delete = True
        if len([x for x in black_close if x in line]) > 0:
            delete = False
            continue
        if not delete:
            fixed_urdf += line + '\n'
    return fixed_urdf


JOINT_TYPES = [u'fixed', u'revolute', u'continuous', u'prismatic']
MOVABLE_JOINT_TYPES = [u'revolute', u'continuous', u'prismatic']
ROTATIONAL_JOINT_TYPES = [u'revolute', u'continuous']
TRANSLATIONAL_JOINT_TYPES = [u'prismatic']


class Robot(object):
    # TODO split urdf part into separate file?
    # TODO remove slow shit from init?
    def __init__(self, urdf, default_joint_vel_limit=0):
        """
        :param urdf:
        :type urdf: str
        :param joints_to_symbols_map: maps urdf joint names to symbols
        :type joints_to_symbols_map: dict
        :param default_joint_vel_limit: all velocity limits which are undefined or higher than this will be set to this
        :type default_joint_vel_limit: float
        """
        self.default_joint_velocity_limit = default_joint_vel_limit
        self.default_weight = 0.0001
        self.fks = {}
        self._joint_to_frame = {}
        self.joint_to_symbol_map = keydefaultdict(lambda x: spw.Symbol(x))
        self.urdf = urdf
        with suppress_stderr():
            self._urdf_robot = URDF.from_xml_string(hacky_urdf_parser_fix(self.urdf))

    @classmethod
    def from_urdf_file(cls, urdf_file, joints_to_symbols_map=None, default_joint_vel_limit=0):
        """
        :param urdf_file: path to urdf file
        :type urdf_file: str
        :param joints_to_symbols_map: maps urdf joint names to symbols
        :type joints_to_symbols_map: dict
        :param default_joint_vel_limit: all velocity limits which are undefined or higher than this will be set to this
        :type default_joint_vel_limit: float
        :rtype: Robot
        """
        with open(urdf_file, 'r') as f:
            urdf_string = f.read()
        self = cls(urdf_string, default_joint_vel_limit)
        self.parse_urdf(joints_to_symbols_map)
        return self

    def parse_urdf(self, joints_to_symbols_map=None):
        """
        :param joints_to_symbols_map: maps urdf joint names to symbols
        :type joints_to_symbols_map: dict
        """
        if joints_to_symbols_map is not None:
            self.joint_to_symbol_map.update(joints_to_symbols_map)
        self._create_frames_expressions()
        self._create_constraints()

    def get_name(self):
        """
        :rtype: str
        """
        return self._urdf_robot.name

    def _create_frames_expressions(self):
        for joint_name, urdf_joint in self._urdf_robot.joint_map.items():
            if self.is_joint_controllable(joint_name):
                joint_symbol = self.get_joint_symbol(joint_name)
            if self.is_joint_mimic(joint_name):
                multiplier = 1 if urdf_joint.mimic.multiplier is None else urdf_joint.mimic.multiplier
                offset = 0 if urdf_joint.mimic.offset is None else urdf_joint.mimic.offset
                joint_symbol = self.get_joint_symbol(urdf_joint.mimic.joint) * multiplier + offset

            if self.is_joint_type_supported(joint_name):
                if urdf_joint.origin is not None:
                    xyz = urdf_joint.origin.xyz if urdf_joint.origin.xyz is not None else [0, 0, 0]
                    rpy = urdf_joint.origin.rpy if urdf_joint.origin.rpy is not None else [0, 0, 0]
                    joint_frame = spw.translation3(*xyz) * spw.rotation_matrix_from_rpy(*rpy)
                else:
                    joint_frame = spw.eye(4)
            else:
                # TODO more specific exception
                raise Exception('Joint type "{}" is not supported by urdf parser.'.format(urdf_joint.type))

            if urdf_joint.type in ROTATIONAL_JOINT_TYPES:
                joint_frame *= spw.rotation_matrix_from_axis_angle(spw.vector3(*urdf_joint.axis), joint_symbol)
            elif urdf_joint.type in TRANSLATIONAL_JOINT_TYPES:
                joint_frame *= spw.translation3(*(spw.point3(*urdf_joint.axis) * joint_symbol)[:3])

            self._joint_to_frame[joint_name] = joint_frame

    def _create_constraints(self):
        """
        Creates hard and joint constraints.
        """
        self.hard_constraints = OrderedDict()
        self.joint_constraints = OrderedDict()
        for i, joint_name in enumerate(self.get_joint_names_controllable()):
            lower_limit, upper_limit = self.get_joint_lower_upper_limit(joint_name)
            joint_symbol = self.get_joint_symbol(joint_name)
            velocity_limit = self.get_joint_velocity_limit(joint_name)

            if lower_limit is not None and upper_limit is not None:
                self.hard_constraints[joint_name] = HardConstraint(lower=lower_limit - joint_symbol,
                                                                   upper=upper_limit - joint_symbol,
                                                                   expression=joint_symbol)

            self.joint_constraints[joint_name] = JointConstraint(lower=-velocity_limit,
                                                                 upper=velocity_limit,
                                                                 weight=self.default_weight)

    def get_fk_expression(self, root_link, tip_link):
        """
        :type root_link: str
        :type tip_link: str
        :return: 4d matrix describing the transformation from root_link to tip_link
        :rtype: spw.Matrix
        """
        if (root_link, tip_link) not in self.fks:
            fk = spw.eye(4)
            for joint_name in self.get_joint_names_from_chain(root_link, tip_link):
                fk *= self.get_joint_frame(joint_name)
            self.fks[root_link, tip_link] = fk
        return self.fks[root_link, tip_link]

    # JOINT FUNCITONS

    def get_joint_names(self):
        """
        :rtype: list
        """
        return self._urdf_robot.joint_map.keys()

    def get_joint_names_from_chain(self, root_link, tip_link):
        """
        :rtype root: str
        :rtype tip: str
        :rtype: list
        """
        return self._urdf_robot.get_chain(root_link, tip_link, True, False, True)

    def get_joint_names_from_chain_controllable(self, root_link, tip_link):
        """
        :rtype root: str
        :rtype tip: str
        :rtype: list
        """
        return self._urdf_robot.get_chain(root_link, tip_link, True, False, False)

    def get_joint_names_controllable(self):
        """
        :return: returns the names of all movable joints which are not mimic.
        :rtype: list
        """
        return [joint_name for joint_name in self.get_joint_names() if self.is_joint_controllable(joint_name)]

    def get_joint_limits(self):
        """
        :return: dict mapping joint names to tuple containing lower and upper limits
        :rtype: dict
        """
        return {joint_name: self.get_joint_lower_upper_limit(joint_name) for joint_name in self.get_joint_names()
                if self.is_joint_controllable(joint_name)}

    def get_joint_symbols(self):
        """
        :return: dict mapping urdf joint name to symbol
        :rtype: dict
        """
        return {joint_name: self.get_joint_symbol(joint_name) for joint_name in self.get_joint_names_controllable()}

    def get_joint_lower_upper_limit(self, joint_names):
        """
        Returns joint limits specified in the safety controller entry if given, else returns the normal limits.
        :param joint_name: name of the joint in the urdf
        :type joint_names: str
        :return: lower limit, upper limit or None if not applicable
        :rtype: float, float
        """
        # TODO use min of safety and normal limits
        joint = self._urdf_robot.joint_map[joint_names]
        if joint.safety_controller is not None:
            lower_limit = joint.safety_controller.soft_lower_limit
            upper_limit = joint.safety_controller.soft_upper_limit
        else:
            if joint.limit is not None:
                lower_limit = joint.limit.lower if joint.limit.lower is not None else None
                upper_limit = joint.limit.upper if joint.limit.upper is not None else None
            else:
                lower_limit = None
                upper_limit = None
        return lower_limit, upper_limit

    def get_joint_velocity_limit(self, joint_name):
        """
        :param joint_name: name of the joint in the urdf
        :type joint_name: str
        :return: minimum of default velocity limit and limit specified in urdf
        :rtype: float
        """
        limit = self._urdf_robot.joint_map[joint_name].limit
        if limit is None or limit.velocity is None:
            return self.default_joint_velocity_limit
        else:
            return min(limit.velocity, self.default_joint_velocity_limit)

    def get_joint_frame(self, joint_name):
        """
        :param joint_name: name of the joint in the urdf
        :type joint_name: str
        :return: matrix expression describing the transformation caused by this joint
        :rtype: spw.Matrix
        """
        return self._joint_to_frame[joint_name]

    def get_joint_symbol(self, joint_name):
        """
        :param joint_name: name of the joint in the urdf
        :type joint_name: str
        :rtype: spw.Symbol
        """
        return self.joint_to_symbol_map[joint_name]

    def is_joint_controllable(self, joint_name):
        """
        :param joint_name: name of the joint in the urdf
        :type joint_name: str
        :return: True if joint type is revolute, continuous or prismatic
        :rtype: bool
        """
        joint = self._urdf_robot.joint_map[joint_name]
        return joint.type in MOVABLE_JOINT_TYPES and joint.mimic is None

    def is_joint_mimic(self, joint_name):
        """
        :param joint_name: name of the joint in the urdf
        :type joint_name: str
        :rtype: bool
        """
        joint = self._urdf_robot.joint_map[joint_name]
        return joint.type in MOVABLE_JOINT_TYPES and joint.mimic is not None

    def is_joint_continuous(self, joint_name):
        """
        :param joint_name: name of the joint in the urdf
        :type joint_name: str
        :rtype: bool
        """
        return self._urdf_robot.joint_map[joint_name].type == u'continuous'

    def is_joint_type_supported(self, joint_name):
        return self._urdf_robot.joint_map[joint_name].type in JOINT_TYPES

    # LINK FUNCTIONS

    def get_link_names_from_chain(self, root_link, tip_link):
        """
        :type root_link: str
        :type tip_link: str
        :return: list of all links in chain excluding root_link, including tip_link
        :rtype: list
        """
        return self._urdf_robot.get_chain(root_link, tip_link, False, True, False)

    def get_link_names(self):
        """
        :rtype: dict
        """
        return self._urdf_robot.link_map.keys()

    def get_sub_tree_link_names_with_collision(self, root_joint):
        """
        returns a set of links with
        :type: str
        :param volume_threshold: links with simple geometric shape and less volume than this will be ignored
        :type volume_threshold: float
        :param surface_treshold:
        :type surface_treshold: float
        :return: all links connected to root
        :rtype: list
        """
        sub_tree = self.get_sub_tree_links(root_joint)
        return [link_name for link_name in sub_tree if self.has_link_collision(link_name)]

    def get_sub_tree_links(self, root_joint):
        """
        :type root_joint: str
        :return: list containing all link names in the subtree after root_joint
        :rtype: list
        """
        links = []
        joints = [root_joint]
        for joint in joints:
            try:
                child_link = self._urdf_robot.joint_map[joint].child
                if child_link in self._urdf_robot.child_map:
                    for j, l in self._urdf_robot.child_map[child_link]:
                        joints.append(j)
                links.append(child_link)
            except KeyError as e:
                return []

        return links

    def has_link_collision(self, link_name, volume_threshold=1e-6, surface_threshold=1e-4):
        """
        :type link: str
        :param volume_threshold: m**3, ignores simple geometry shapes with a volume less than this
        :type volume_threshold: float
        :param surface_threshold: m**2, ignores simple geometry shapes with a surface area less than this
        :type surface_threshold: float
        :return: True if collision geometry is mesh or simple shape with volume/surface bigger than thresholds.
        :rtype: bool
        """
        link = self._urdf_robot.link_map[link_name]
        if link.collision is not None:
            geo = link.collision.geometry
            return isinstance(geo, Box) and (cube_volume(*geo.size) > volume_threshold or
                                             cube_surface(*geo.size) > surface_threshold) or \
                   isinstance(geo, Sphere) and sphere_volume(geo.radius) > volume_threshold or \
                   isinstance(geo, Cylinder) and (cylinder_volume(geo.radius, geo.length) > volume_threshold or
                                                  cylinder_surface(geo.radius, geo.length) > surface_threshold) or \
                   isinstance(geo, Mesh)
        return False

    def get_urdf(self):
        return self.urdf

    def get_hash(self):
        return hashlib.md5(self.urdf).hexdigest()