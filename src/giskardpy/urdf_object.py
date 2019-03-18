from collections import namedtuple
from itertools import chain

from geometry_msgs.msg import Pose, Vector3
from giskard_msgs.msg import WorldBody
import urdf_parser_py.urdf as up
from std_msgs.msg import ColorRGBA
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import Marker

from giskardpy.exceptions import DuplicateNameException, UnknownBodyException, CorruptShapeException
from giskardpy.utils import cube_volume, cube_surface, sphere_volume, cylinder_volume, cylinder_surface, \
    suppress_stderr, remove_outer_tag

Joint = namedtuple('Joint', ['symbol', 'velocity_limit', 'lower', 'upper', 'type', 'frame'])


def hacky_urdf_parser_fix(urdf_str):
    # TODO this function is inefficient but the tested urdfs's aren't big enough for it to be a problem
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


class URDFObject(object):
    def __init__(self, urdf, *args, **kwargs):
        """
        :param urdf:
        :type urdf: str
        :param joints_to_symbols_map: maps urdfs joint names to symbols
        :type joints_to_symbols_map: dict
        :param default_joint_vel_limit: all velocity limits which are undefined or higher than this will be set to this
        :type default_joint_vel_limit: Symbol
        """
        self.original_urdf = hacky_urdf_parser_fix(urdf)
        with suppress_stderr():
            self._urdf_robot = up.URDF.from_xml_string(self.original_urdf)  # type: up.Robot

    @classmethod
    def from_urdf_file(cls, urdf_file, *args, **kwargs):
        """
        :param urdf_file: path to urdfs file
        :type urdf_file: str
        :param joints_to_symbols_map: maps urdfs joint names to symbols
        :type joints_to_symbols_map: dict
        :param default_joint_vel_limit: all velocity limits which are undefined or higher than this will be set to this
        :type default_joint_vel_limit: float
        :rtype: cls
        """
        with open(urdf_file, 'r') as f:
            urdf_string = f.read()
        self = cls(urdf_string, *args, **kwargs)
        return self

    @classmethod
    def from_world_body(cls, world_body, *args, **kwargs):
        """
        :type world_body: giskard_msgs.msg._WorldBody.WorldBody
        :rtype: URDFObject
        """
        links = []
        joints = []
        if world_body.type == world_body.PRIMITIVE_BODY or world_body.type == world_body.MESH_BODY:
            if world_body.shape.type == world_body.shape.BOX:
                geometry = up.Box(world_body.shape.dimensions)
            elif world_body.shape.type == world_body.shape.SPHERE:
                geometry = up.Sphere(world_body.shape.dimensions[0])
            elif world_body.shape.type == world_body.shape.CYLINDER:
                geometry = up.Cylinder(world_body.shape.dimensions[world_body.shape.CYLINDER_RADIUS],
                                       world_body.shape.dimensions[world_body.shape.CYLINDER_HEIGHT])
            elif world_body.shape.type == world_body.shape.CONE:
                raise TypeError(u'primitive shape cone not supported')
            elif world_body.type == world_body.MESH_BODY:
                geometry = up.Mesh(world_body.mesh)
            else:
                raise CorruptShapeException(u'primitive shape \'{}\' not supported'.format(world_body.shape.type))
            link = up.Link(world_body.name,
                           visual=up.Visual(geometry, material=up.Material(u'green', color=up.Color(0, 1, 0, 1))),
                           collision=up.Collision(geometry))
            links.append(link)
        elif world_body.type == world_body.URDF_BODY:
            o = cls(world_body.urdf, *args, **kwargs)
            o.set_name(world_body.name)
            return o
        else:
            raise CorruptShapeException(u'world body type \'{}\' not supported'.format(world_body.type))
        return cls.from_parts(world_body.name, links, joints, *args, **kwargs)

    @classmethod
    def from_parts(cls, robot_name, links, joints, *args, **kwargs):
        """
        :param robot_name:
        :param links:
        :param joints:
        :rtype: URDFObject
        """
        r = up.Robot(robot_name)
        for link in links:
            r.add_link(link)
        for joint in joints:
            r.add_joint(joint)
        return cls(r.to_xml_string(), *args, **kwargs)

    @classmethod
    def from_urdf_object(cls, urdf_object, *args, **kwargs):
        """
        :type urdf_object: URDFObject
        :rtype: cls
        """
        return cls(urdf_object.get_urdf(), *args, **kwargs)

    def get_name(self):
        """
        :rtype: str
        """
        return self._urdf_robot.name

    def set_name(self, name):
        self._urdf_robot.name = name
        self.reinitialize()

    def get_urdf_robot(self):
        return self._urdf_robot

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

    def get_all_joint_limits(self):
        """
        :return: dict mapping joint names to tuple containing lower and upper limits
        :rtype: dict
        """
        return {joint_name: self.get_joint_limits(joint_name) for joint_name in self.get_joint_names()
                if self.is_joint_controllable(joint_name)}

    def get_joint_limits(self, joint_names):
        """
        Returns joint limits specified in the safety controller entry if given, else returns the normal limits.
        :param joint_name: name of the joint in the urdfs
        :type joint_names: str
        :return: lower limit, upper limit or None if not applicable
        :rtype: float, float
        """
        joint = self.get_urdf_joint(joint_names)
        try:
            return max(joint.safety_controller.soft_lower_limit, joint.limit.lower), \
                   min(joint.safety_controller.soft_upper_limit, joint.limit.upper)
        except AttributeError:
            try:
                return joint.limit.lower, joint.limit.upper
            except AttributeError:
                return None, None

    def is_joint_controllable(self, name):
        """
        :param name: name of the joint in the urdfs
        :type name: str
        :return: True if joint type is revolute, continuous or prismatic
        :rtype: bool
        """
        joint = self.get_urdf_joint(name)
        return joint.type in MOVABLE_JOINT_TYPES and joint.mimic is None

    def is_joint_mimic(self, name):
        """
        :param name: name of the joint in the urdfs
        :type name: str
        :rtype: bool
        """
        joint = self.get_urdf_joint(name)
        return joint.type in MOVABLE_JOINT_TYPES and joint.mimic is not None

    def is_joint_continuous(self, name):
        """
        :param name: name of the joint in the urdfs
        :type name: str
        :rtype: bool
        """
        return self.get_joint_type(name) == u'continuous'

    def get_joint_type(self, name):
        return self.get_urdf_joint(name).type

    def is_joint_type_supported(self, name):
        return self.get_joint_type(name) in JOINT_TYPES

    def is_rotational_joint(self, name):
        return self.get_joint_type(name) in ROTATIONAL_JOINT_TYPES

    def is_translational_joint(self, name):
        return self.get_joint_type(name) in TRANSLATIONAL_JOINT_TYPES

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
        sub_tree = self.get_links_from_sub_tree(root_joint)
        return [link_name for link_name in sub_tree if self.has_link_collision(link_name)]

    def get_link_names_with_collision(self):
        return [link_name for link_name in self.get_link_names() if self.has_link_collision(link_name)]

    def get_links_from_sub_tree(self, joint_name):
        return self.get_sub_tree_at_joint(joint_name).get_link_names()

    def get_sub_tree_at_joint(self, joint_name):
        """
        :type joint_name: str
        :rtype: URDFObject
        """
        tree_links = []
        tree_joints = []
        joints = [joint_name]
        for joint in joints:
            child_link = self._urdf_robot.joint_map[joint].child
            if child_link in self._urdf_robot.child_map:
                for j, l in self._urdf_robot.child_map[child_link]:
                    joints.append(j)
                    tree_joints.append(self.get_urdf_joint(j))
            tree_links.append(self.get_urdf_link(child_link))

        return URDFObject.from_parts(joint_name, tree_links, tree_joints)

    def get_urdf_joint(self, joint_name):
        return self._urdf_robot.joint_map[joint_name]

    def get_urdf_link(self, link_name):
        return self._urdf_robot.link_map[link_name]

    def split_at_link(self, link_name):
        pass

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
            return isinstance(geo, up.Box) and (cube_volume(*geo.size) > volume_threshold or
                                                cube_surface(*geo.size) > surface_threshold) or \
                   isinstance(geo, up.Sphere) and sphere_volume(geo.radius) > volume_threshold or \
                   isinstance(geo, up.Cylinder) and (cylinder_volume(geo.radius, geo.length) > volume_threshold or
                                                     cylinder_surface(geo.radius, geo.length) > surface_threshold) or \
                   isinstance(geo, up.Mesh)
        return False

    def get_urdf(self):
        return self._urdf_robot.to_xml_string()

    def get_root(self):
        return self._urdf_robot.get_root()

    def attach_urdf_object(self, urdf_object, parent_link, pose):
        """
        Rigidly attach another object to the robot.
        :param urdf_object: Object that shall be attached to the robot.
        :type urdf_object: URDFObject
        :param parent_link_name: Name of the link to which the object shall be attached.
        :type parent_link_name: str
        :param pose: Hom. transform between the reference frames of the parent link and the object.
        :type pose: Pose
        """
        if urdf_object.get_name() in self.get_link_names():
            raise DuplicateNameException(
                u'\'{}\' already has link with name \'{}\'.'.format(self.get_name(), urdf_object.get_name()))
        if urdf_object.get_name() in self.get_joint_names():
            raise DuplicateNameException(
                u'\'{}\' already has joint with name \'{}\'.'.format(self.get_name(), urdf_object.get_name()))
        if parent_link not in self.get_link_names():
            raise UnknownBodyException(
                u'can not attach \'{}\' to non existent parent link \'{}\' of \'{}\''.format(urdf_object.get_name(),
                                                                                             parent_link,
                                                                                             self.get_name()))
        if len(set(urdf_object.get_link_names()).intersection(set(self.get_link_names()))) != 0:
            raise DuplicateNameException(u'can not merge urdfs that share link names')
        if len(set(urdf_object.get_joint_names()).intersection(set(self.get_joint_names()))) != 0:
            raise DuplicateNameException(u'can not merge urdfs that share joint names')

        joint = up.Joint(self.robot_name_to_root_joint(urdf_object.get_name()),
                         parent=parent_link,
                         child=urdf_object.get_root(),
                         joint_type=u'fixed',
                         origin=up.Pose([pose.position.x,
                                         pose.position.y,
                                         pose.position.z],
                                        euler_from_quaternion([pose.orientation.x,
                                                               pose.orientation.y,
                                                               pose.orientation.z,
                                                               pose.orientation.w])))
        self._urdf_robot.add_joint(joint)
        for j in urdf_object._urdf_robot.joints:
            self._urdf_robot.add_joint(j)
        for l in urdf_object._urdf_robot.links:
            self._urdf_robot.add_link(l)
        self.reinitialize()

    def detach_sub_tree(self, joint_name):
        """
        :rtype: URDFObject
        """
        sub_tree = self.get_sub_tree_at_joint(joint_name)
        for link in sub_tree.get_link_names():
            self._urdf_robot.remove_aggregate(self.get_urdf_link(link))
        for joint in chain([joint_name], sub_tree.get_joint_names()):
            self._urdf_robot.remove_aggregate(self.get_urdf_joint(joint))
        self.reinitialize()
        return sub_tree


    def reset(self):
        """
        Detaches all object that have been attached to the robot.
        """
        self._urdf_robot = up.URDF.from_xml_string(self.original_urdf)
        self.reinitialize()

    def __str__(self):
        return self.get_name()


    def reinitialize(self):
        self._urdf_robot = up.URDF.from_xml_string(self.get_urdf())

    def robot_name_to_root_joint(self, name):
        # TODO should this really be a class function?
        return u'{}'.format(name)

    def get_parent_link_name(self, child_link_name):
        if child_link_name in self._urdf_robot.parent_map:
            return self._urdf_robot.parent_map[child_link_name][1]

    def are_linked(self, link_a, link_b):
        return link_a == self.get_parent_link_name(link_b) or \
               (link_b == self.get_parent_link_name(link_a))

    def get_controllable_joints(self):
        return [joint_name for joint_name in self.get_joint_names() if self.is_joint_controllable(joint_name)]


    def __eq__(self, o):
        """
        :type o: URDFObject
        :rtype: bool
        """
        return o.get_urdf() == self.get_urdf()

    def has_link_visuals(self, link_name):
        link = self._urdf_robot.link_map[link_name]
        return link.visual is not None

    def as_marker_msg(self, ns=u'', id=1):
        """
        :param ns:
        :param id:
        :rtype: Marker
        """
        if len(self.get_link_names()) > 1:
            raise TypeError(u'only urdfs objects with a single link can be turned into marker')
        link = self.get_urdf_link(self.get_link_names()[0])
        m = Marker()
        m.ns = u'{}/{}'.format(ns, self.get_name())
        m.id = id
        geometry = link.visual.geometry
        if isinstance(geometry, up.Box):
            m.type = Marker.CUBE
            m.scale = Vector3(*geometry.size)
        elif isinstance(geometry, up.Sphere):
            m.type = Marker.SPHERE
            m.scale = Vector3(geometry.radius,
                              geometry.radius,
                              geometry.radius)
        elif isinstance(geometry, up.Cylinder):
            m.type = Marker.CYLINDER
            m.scale = Vector3(geometry.radius,
                              geometry.radius,
                              geometry.length)
        else:
            raise Exception(u'world body type {} can\'t be converted to marker'.format(geometry.__class__.__name__))
        m.color = ColorRGBA(0,1,0,0.5)
        return m

    def link_as_marker(self, link_name):
        marker = Marker()
        geometry = self.get_urdf_link(link_name).visual.geometry

        if isinstance(geometry, up.Mesh):
            marker.type = Marker.MESH_RESOURCE
            marker.mesh_resource = geometry.filename
            if geometry.scale is None:
                marker.scale.x = 1.0
                marker.scale.z = 1.0
                marker.scale.y = 1.0
            else:
                marker.scale.x = geometry.scale[0]
                marker.scale.z = geometry.scale[1]
                marker.scale.y = geometry.scale[2]
            marker.mesh_use_embedded_materials = True
        elif isinstance(geometry, up.Box):
            marker.type = Marker.CUBE
            marker.scale.x = geometry.size[0]
            marker.scale.y = geometry.size[1]
            marker.scale.z = geometry.size[2]
        elif isinstance(geometry, up.Cylinder):
            marker.type = Marker.CYLINDER
            marker.scale.x = geometry.radius
            marker.scale.y = geometry.radius
            marker.scale.z = geometry.length
        elif isinstance(geometry, up.Sphere):
            marker.type = Marker.SPHERE
            marker.scale.x = geometry.radius
            marker.scale.y = geometry.radius
            marker.scale.z = geometry.radius
        else:
            return None

        marker.header.frame_id = self.get_root()
        marker.action = Marker.ADD
        marker.color.a = 0.5
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        return marker

        # self.fk_dict = self.get_god_map().get_data(['fk'])
        # markers = []
        # for index, link in enumerate(self.get_link_names()):
        #     if not self.has_link_visuals(link.name):
        #         continue
        #     marker = Marker()
        #     m.ns = u'{}/{}'.format(ns, self.get_name())
        #     m.id = index
        #     link_type = type(link.visual.geometry)
        #
        #     if link_type == up.Mesh:
        #         marker.type = Marker.MESH_RESOURCE
        #         marker.mesh_resource = link.visual.geometry.filename
        #         if link.visual.geometry.scale is None:
        #             marker.scale.x = 1.0
        #             marker.scale.z = 1.0
        #             marker.scale.y = 1.0
        #         else:
        #             marker.scale.x = link.visual.geometry.scale[0]
        #             marker.scale.z = link.visual.geometry.scale[1]
        #             marker.scale.y = link.visual.geometry.scale[2]
        #         marker.mesh_use_embedded_materials = True
        #     elif link_type == up.Box:
        #         marker.type = Marker.CUBE
        #         marker.scale = Vector3(*link.visual.geometry)
        #     elif link_type == up.Cylinder:
        #         marker.type = Marker.CYLINDER
        #         marker.scale.x = link.visual.geometry.radius
        #         marker.scale.y = link.visual.geometry.radius
        #         marker.scale.z = link.visual.geometry.length
        #     elif link_type == up.Sphere:
        #         marker.type = Marker.SPHERE
        #         marker.scale.x = link.visual.geometry.radius
        #         marker.scale.y = link.visual.geometry.radius
        #         marker.scale.z = link.visual.geometry.radius
        #     else:
        #         continue
        #
        #     marker.scale.x *= 0.99
        #     marker.scale.y *= 0.99
        #     marker.scale.z *= 0.99
        #
        #     link_in_base = self.fk_dict[self.get_root(), link.name]
        #     marker.header.frame_id = self.get_root()
        #     marker.action = Marker.ADD
        #     marker.id = index
        #     marker.ns = u'planning_visualization'
        #     marker.pose = link_in_base.pose
        #     marker.color.a = 0.5
        #     marker.color.r = 1.0
        #     marker.color.g = 1.0
        #     marker.color.b = 1.0
        #
        #
        #     markers.append(marker)
        #
        # m.color = ColorRGBA(0, 1, 0, 0.8)
        # m.frame_locked = True
        # return m