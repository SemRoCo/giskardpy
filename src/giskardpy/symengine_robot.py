from collections import namedtuple, OrderedDict
import numpy as np
import symengine_wrappers as spw
from urdf_parser_py.urdf import URDF, Box, Sphere, Mesh, Cylinder

from giskardpy.input_system import JointStatesInput
from giskardpy.qp_problem_builder import HardConstraint, JointConstraint
from giskardpy.utils import cube_volume, cube_surface, sphere_volume, cylinder_volume, cylinder_surface

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
    def __init__(self, urdf, default_joint_vel_limit=0):
        self.default_joint_velocity_limit = default_joint_vel_limit
        self.default_weight = 0.0001
        self.fks = {}
        self._load_from_urdf_string(urdf)

    @classmethod
    def from_urdf_file(cls, urdf_file, default_joint_vel_limit=0):
        with open(urdf_file, 'r') as f:
            urdf_string = f.read()
        self = cls(urdf_string, default_joint_vel_limit)
        return self

    def _load_from_urdf_string(self, urdf_str):
        return self._load_from_urdf(URDF.from_xml_string(hacky_urdf_parser_fix(urdf_str)))

    def _load_from_urdf(self, urdf_robot):
        self._urdf_robot = urdf_robot
        self._create_sym_frames()
        self._create_constraints()

    def get_name(self):
        return self._urdf_robot.name

    def set_joint_symbol_map(self, joint_states_input=None):
        # TODO replace self._joints with map from name to symbol and name to frame
        if joint_states_input is not None:
            self.joint_states_input = joint_states_input
            for joint_name, joint in self._joints.items():
                new_symbol = None
                if joint.symbol is not None and joint_name in self.joint_states_input.joint_map:
                    new_symbol = self.joint_states_input.joint_map[joint_name]
                self._joints[joint_name] = Joint(new_symbol,
                                                 joint.velocity_limit,
                                                 joint.lower,
                                                 joint.upper,
                                                 joint.type,
                                                 # TODO this line is relatively slow
                                                 joint.frame.subs(self.joint_states_input.joint_map))
            self._create_constraints()

    def _create_sym_frames(self):
        self._joints = {}
        joint_map = {}
        for joint_name, joint in self._urdf_robot.joint_map.items():
            joint_symbol = None
            if self.is_joint_movable(joint_name):
                joint_map[joint_name] = spw.Symbol(joint_name)
                joint_symbol = joint_map[joint_name]
            elif self.is_joint_mimic(joint_name):
                joint_map[joint.mimic.joint] = spw.Symbol(joint.mimic.joint)
                multiplier = 1 if joint.mimic.multiplier is None else joint.mimic.multiplier
                offset = 0 if joint.mimic.offset is None else joint.mimic.offset
                mimic = joint_map[joint.mimic.joint] * multiplier + offset

            if self.is_joint_type_supported(joint_name):
                if joint.origin is not None:
                    xyz = joint.origin.xyz if joint.origin.xyz is not None else [0, 0, 0]
                    rpy = joint.origin.rpy if joint.origin.rpy is not None else [0, 0, 0]
                    joint_frame = spw.translation3(*xyz) * spw.rotation_matrix_from_rpy(*rpy)
                else:
                    joint_frame = spw.eye(4)
            else:
                # TODO more specific exception
                raise Exception('Joint type "{}" is not supported by urdf parser.'.format(joint.type))

            if joint.type in ROTATIONAL_JOINT_TYPES:
                if joint.mimic is None:
                    joint_frame *= spw.rotation_matrix_from_axis_angle(spw.vector3(*joint.axis), joint_symbol)
                else:
                    joint_frame *= spw.rotation_matrix_from_axis_angle(spw.vector3(*joint.axis), mimic)

            elif joint.type in TRANSLATIONAL_JOINT_TYPES:
                if joint.mimic is None:
                    joint_frame *= spw.translation3(*(spw.point3(*joint.axis) * joint_symbol)[:3])
                else:
                    joint_frame *= spw.translation3(*(spw.point3(*joint.axis) * mimic)[:3])

            if joint.limit is not None:
                vel_limit = min(joint.limit.velocity, self.default_joint_velocity_limit)
            else:
                vel_limit = None

            lower_limit, upper_limit = self.get_joint_lower_upper_limit(joint.name)
            self._joints[joint_name] = Joint(joint_symbol,
                                             vel_limit,
                                             lower_limit,
                                             upper_limit,
                                             joint.type,
                                             joint_frame)
        self.joint_states_input = JointStatesInput(lambda x: spw.Symbol(x[0]), joint_map)

    def _create_constraints(self):
        self.hard_constraints = OrderedDict()
        self.joint_constraints = OrderedDict()
        for i, joint_name in enumerate(self.get_joint_names_movable()):
            lower_limit, upper_limit = self.get_joint_lower_upper_limit(joint_name)
            joint_symbol = self.joint_to_symbol(joint_name)
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
    def get_joint_frame(self, joint_name):
        return self._joints[joint_name].frame

    def get_joint_names_from_chain(self, root_link, tip_link):
        """
        :rtype root: str
        :rtype tip: str
        :return: list
        :rtype: list
        """
        return self._urdf_robot.get_chain(root_link, tip_link, True, False, True)

    def get_joint_names_from_chain_movable(self, root_link, tip_link):
        """
        :rtype root: str
        :rtype tip: str
        :return: list
        :rtype: list
        """
        return self._urdf_robot.get_chain(root_link, tip_link, True, False, False)

    def get_joint_names(self):
        """
        :rtype: list
        """
        return self._urdf_robot.joint_map.keys()

    def get_joint_names_movable(self):
        """
        :return: returns the names of all movable joints which are not mimic.
        :rtype: list
        """
        return [joint_name for joint_name in self.get_joint_names() if self.is_joint_movable(joint_name)]

    def joint_to_symbol(self, joint_name):
        """
        :type joint_name: str
        :rtype: spw.Symbol
        """
        return self._joints[joint_name].symbol

    def is_joint_movable(self, joint_name):
        """
        :type joint_name: str
        :return: True if joint type is revolute, continuous or prismatic
        :rtype: bool
        """
        joint = self._urdf_robot.joint_map[joint_name]
        return joint.type in MOVABLE_JOINT_TYPES and joint.mimic is None

    def is_joint_mimic(self, joint_name):
        """
        :type joint_name: str
        :rtype: bool
        """
        joint = self._urdf_robot.joint_map[joint_name]
        return joint.type in MOVABLE_JOINT_TYPES and joint.mimic is not None

    def is_joint_continuous(self, joint_name):
        """
        :type joint_name: str
        :rtype: bool
        """
        return self._urdf_robot.joint_map[joint_name].type == u'continuous'

    def is_joint_type_supported(self, joint_name):
        return self._urdf_robot.joint_map[joint_name].type in JOINT_TYPES

    def get_joint_limits(self):
        """
        :return: dict mapping joint names to tuple containing lower and upper limits
        :rtype: dict
        """
        return {joint_name: self.get_joint_lower_upper_limit(joint_name) for joint_name in self.get_joint_names()
                if self.is_joint_movable(joint_name)}

    def get_joint_lower_upper_limit(self, joint_names):
        """
        Returns joint limits specified in the safety controller entry if given, else returns the normal limits.
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
        :type joint_name: str
        :return: minimum of default velocity limit and limit specified in urdf
        :rtype: float
        """
        limit = self._urdf_robot.joint_map[joint_name].limit
        if limit is None or limit.velocity is None:
            return self.default_joint_velocity_limit
        else:
            return min(limit.velocity, self.default_joint_velocity_limit)

    # LINKFUNCTIONS

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
