from collections import namedtuple

from urdf_parser_py.urdf import Box, Sphere, Cylinder, Mesh, URDF

from giskardpy.utils import cube_volume, cube_surface, sphere_volume, cylinder_volume, cylinder_surface, suppress_stderr

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



class NewURDFObject(object):
    # TODO split urdf part into separate file?
    # TODO remove slow shit from init?
    def __init__(self, urdf):
        """
        :param urdf:
        :type urdf: str
        :param joints_to_symbols_map: maps urdf joint names to symbols
        :type joints_to_symbols_map: dict
        :param default_joint_vel_limit: all velocity limits which are undefined or higher than this will be set to this
        :type default_joint_vel_limit: Symbol
        """
        self.urdf = urdf
        with suppress_stderr():
            self._urdf_robot = URDF.from_xml_string(hacky_urdf_parser_fix(self.urdf))

    @classmethod
    def from_urdf_file(cls, urdf_file):
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
        self = cls(urdf_string,)
        return self

    @classmethod
    def from_world_body(cls, world_body):
        """
        :type world_body: giskard_msgs.msg._WorldBody.WorldBody
        :return:
        """
        pass

    def get_name(self):
        """
        :rtype: str
        """
        return self._urdf_robot.name

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
        if self.is_joint_continuous(joint_names):
            lower_limit = None
            upper_limit = None
        elif joint.safety_controller is not None:
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
        return URDF.to_xml_string(self._urdf_robot)

    def attach_urdf(self, urdf, parent_link, transform):
        pass

    def attach_urdf_object(self, urdf_object):
        pass

    def detach(self, name):
        pass
