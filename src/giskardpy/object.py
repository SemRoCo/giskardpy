from giskard_msgs.srv import UpdateWorldRequest
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker
from giskard_msgs.msg import WorldBody
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose as PoseMsg, Point as PointMsg, Quaternion as QuaternionMsg, Vector3

from giskardpy.exceptions import CorruptShapeException
from giskardpy.data_types import Transform, Point, Quaternion
from lxml import etree
import PyKDL as kdl # TODO: get rid of this dependency


class ColorRgba(object):
    def __init__(self, r=1.0, g=1.0, b=1.0, a=1.0):
        self.r = r
        self.g = g
        self.b = b
        self.a = a


class InertiaMatrix(object):
    def __init__(self, ixx=0.0, ixy=0.0, ixz=0.0, iyy=0.0, iyz=0.0, izz=0.0):
        self.ixx = ixx
        self.ixy = ixy
        self.ixz = ixz
        self.iyy = iyy
        self.iyz = iyz
        self.izz = izz


class GeometricShape(object):
    pass


class BoxShape(GeometricShape):
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class CylinderShape(GeometricShape):
    def __init__(self, radius=0.0, length=0.0):
        self.radius = radius
        self.length = length


class SphereShape(GeometricShape):
    def __init__(self, radius=0.0):
        self.radius = radius


class MeshShape(GeometricShape):
    def __init__(self, filename='', scale=(1.0, 1.0, 1.0)):
        self.filename = filename
        self.scale = scale


class InertialProperty(object):
    def __init__(self, origin=Transform(), mass=0.0, inertia=InertiaMatrix()):
        self.origin = origin
        self.mass = mass
        self.inertia = inertia


class MaterialProperty(object):
    def __init__(self, name=u'', color=ColorRgba(), texture_filename=''):
        self.name = name
        self.color = color
        self.texture_filename = texture_filename


class VisualProperty(object):
    def __init__(self, name=u'', origin=Transform(), geometry=None, material=None):
        self.name = name
        self.origin = origin
        self.geometry = geometry
        self.material = material


class CollisionProperty(object):
    def __init__(self, name=u'', origin=Transform(), geometry=None):
        self.name = name
        self.origin = origin
        self.geometry = geometry


class UrdfObject(object):
    def __init__(self, name=u'', inertial_props=None, visual_props=(), collision_props=()):
        self.name = name
        self.inertial_props = inertial_props
        self.visual_props = visual_props
        self.collision_props = collision_props

class Box(UrdfObject):
    def __init__(self, name, length, width, height):
        geom = BoxShape(length,
                        width,
                        height)
        col = CollisionProperty(name=name + u'_col', geometry=geom)
        vis = VisualProperty(name=name + u'_vis', geometry=geom)
        super(Box, self).__init__(name, collision_props=[col], visual_props=[vis])

class Sphere(UrdfObject):
    def __init__(self, name, radius):
        geom = SphereShape(radius)
        col = CollisionProperty(name=name + u'_col', geometry=geom)
        vis = VisualProperty(name=name + u'_vis', geometry=geom)
        super(Sphere, self).__init__(name, collision_props=[col], visual_props=[vis])

class Cylinder(UrdfObject):
    def __init__(self, name, radius, length):
        geom = CylinderShape(radius, length)
        col = CollisionProperty(name=name + u'_col', geometry=geom)
        vis = VisualProperty(name=name + u'_vis', geometry=geom)
        super(Cylinder, self).__init__(name, collision_props=[col], visual_props=[vis])

class Joint(object):
    def __init__(self, name='', origin=Transform(), parent_link_name='', child_link_name='', ):
        self.name = name
        self.origin = origin
        self.parent_link_name = parent_link_name
        self.child_link_name = child_link_name

class FixedJoint(Joint):
    pass


def to_urdf_xml(urdf_object, skip_robot_tag=False):
    """
    :param urdf_object:
    :type urdf_object: Union[WorldObject, InertialProperty]
    :return:
    :rtype: lxml.etree.Element
    """
    if isinstance(urdf_object, UrdfObject):
        link = etree.Element(u'link', name=urdf_object.name)
        if urdf_object.inertial_props:
            link.append(to_urdf_xml(urdf_object.inertial_props))
        for visual in urdf_object.visual_props:
            link.append(to_urdf_xml(visual))
        for collision in urdf_object.collision_props:
            link.append(to_urdf_xml(collision))
        if skip_robot_tag:
            root = link
        else:
            root = etree.Element(u'robot', name=urdf_object.name)
            root.append(link)
    elif isinstance(urdf_object, InertialProperty):
        root = etree.Element(u'inertial')
        root.append(to_urdf_xml(urdf_object.origin))
        root.append(to_urdf_xml(urdf_object.inertia))
        mass = etree.Element(u'mass', value=str(urdf_object.mass))
        root.append(mass)
    elif isinstance(urdf_object, VisualProperty):
        if urdf_object.name:
            root = etree.Element(u'visual', name=urdf_object.name)
        else:
            root = etree.Element(u'visual')
        root.append(to_urdf_xml(urdf_object.origin))
        root.append(to_urdf_xml(urdf_object.geometry))
        if urdf_object.material:
            root.append(to_urdf_xml(urdf_object.material))
    elif isinstance(urdf_object, CollisionProperty):
        root = etree.Element(u'collision', name=urdf_object.name)
        root.append(to_urdf_xml(urdf_object.origin))
        root.append(to_urdf_xml(urdf_object.geometry))
    elif isinstance(urdf_object, Transform):
        r = kdl.Rotation.Quaternion(urdf_object.rotation.x, urdf_object.rotation.y,
                                    urdf_object.rotation.z, urdf_object.rotation.w)
        rpy = r.GetRPY()
        rpy_string = u'{} {} {}'.format(rpy[0], rpy[1], rpy[2])
        xyz_string = u'{} {} {}'.format(urdf_object.translation.x, urdf_object.translation.y, urdf_object.translation.z)
        root = etree.Element(u'origin', xyz=xyz_string, rpy=rpy_string)
    elif isinstance(urdf_object, InertiaMatrix):
        root = etree.Element(u'inertia', ixx=str(urdf_object.ixx), ixy=str(urdf_object.ixy), ixz=str(urdf_object.ixz),
                             iyy=str(urdf_object.iyy), iyz=str(urdf_object.iyz), izz=str(urdf_object.izz))
    elif isinstance(urdf_object, BoxShape):
        root = etree.Element(u'geometry')
        size_string = u'{} {} {}'.format(urdf_object.x, urdf_object.y, urdf_object.z)
        box = etree.Element(u'box', size=size_string)
        root.append(box)
    elif isinstance(urdf_object, CylinderShape):
        root = etree.Element(u'geometry')
        cyl = etree.Element(u'cylinder', radius=str(urdf_object.radius), length=str(urdf_object.length))
        root.append(cyl)
    elif isinstance(urdf_object, SphereShape):
        root = etree.Element(u'geometry')
        sphere = etree.Element(u'sphere', radius=str(urdf_object.radius))
        root.append(sphere)
    elif isinstance(urdf_object, MeshShape):
        root = etree.Element(u'geometry')
        scale_string = u'{} {} {}'.format(urdf_object.scale[0], urdf_object.scale[1], urdf_object.scale[2])
        mesh = etree.Element(u'mesh', scale=scale_string, filename=urdf_object.filename)
        root.append(mesh)
    elif isinstance(urdf_object, MaterialProperty):
        root = etree.Element(u'material', name=urdf_object.name)
        if urdf_object.color:
            color_string = u'{} {} {} {}'.format(str(urdf_object.color.r), str(urdf_object.color.g),
                                                str(urdf_object.color.b), str(urdf_object.color.a))
            color = etree.Element(u'color', rgba=color_string)
            root.append(color)
        if urdf_object.texture_filename:
            tex =etree.Element(u'texture', filename=urdf_object.texture_filename)
            root.append(tex)
    elif isinstance(urdf_object, FixedJoint):
        root = etree.Element(u'joint', name=urdf_object.name, type=u'fixed')
        root.append(to_urdf_xml(urdf_object.origin))
        root.append(etree.Element(u'parent', link=urdf_object.parent_link_name))
        root.append(etree.Element(u'child', link=urdf_object.child_link_name))
    return root


def to_urdf_string(urdf_object, skip_robot_tag=False):
    """
    :param urdf_object:
    :type urdf_object: UrdfObject
    :return:
    :rtype: str
    """
    return etree.tostring(to_urdf_xml(urdf_object, skip_robot_tag=skip_robot_tag))

def to_marker(thing):
    """
    :type thing: Union[UpdateWorldRequest, WorldBody]
    :rtype: MarkerArray
    """
    ma = MarkerArray()
    if isinstance(thing, UrdfObject):
        pass
        # TODO
        # return urdf_object_to_marker_msg(thing)
    elif isinstance(thing, WorldBody):
        ma.markers.append(world_body_to_marker_msg(thing))
    elif isinstance(thing, UpdateWorldRequest):
        ma.markers.append(update_world_to_marker_msg(thing))
    return ma

def update_world_to_marker_msg(update_world_req, id=1, ns=u''):
    """
    :type update_world_req: UpdateWorldRequest
    :type id: int
    :type ns: str
    :rtype: Marker
    """
    m = world_body_to_marker_msg(update_world_req.body, id, ns)
    m.header = update_world_req.pose.header
    m.pose = update_world_req.pose.pose
    m.frame_locked = update_world_req.rigidly_attached
    if update_world_req.operation == UpdateWorldRequest.ADD:
        m.action = Marker.ADD
    elif update_world_req.operation == UpdateWorldRequest.REMOVE:
        m.action = Marker.DELETE
    elif update_world_req.operation == UpdateWorldRequest.REMOVE_ALL:
        m.action = Marker.DELETEALL
    return m


def world_body_to_marker_msg(world_body, id=1, ns=u''):
    """
    :type world_body: WorldBody
    :rtype: Marker
    """
    m = Marker()
    m.ns = u'{}/{}'.format(ns, world_body.name)
    m.id = id
    if world_body.type == WorldBody.URDF_BODY:
        raise Exception(u'can\'t convert urdf body world object to marker array')
    elif world_body.type == WorldBody.PRIMITIVE_BODY:
        if world_body.shape.type == SolidPrimitive.BOX:
            m.type = Marker.CUBE
            m.scale = Vector3(*world_body.shape.dimensions)
        elif world_body.shape.type == SolidPrimitive.SPHERE:
            m.type = Marker.SPHERE
            m.scale = Vector3(world_body.shape.dimensions[0],
                              world_body.shape.dimensions[0],
                              world_body.shape.dimensions[0])
        elif world_body.shape.type == SolidPrimitive.CYLINDER:
            m.type = Marker.CYLINDER
            m.scale = Vector3(world_body.shape.dimensions[SolidPrimitive.CYLINDER_RADIUS],
                              world_body.shape.dimensions[SolidPrimitive.CYLINDER_RADIUS],
                              world_body.shape.dimensions[SolidPrimitive.CYLINDER_HEIGHT])
        else:
            raise Exception(u'world body type {} can\'t be converted to marker'.format(world_body.shape.type))
    elif world_body.type == WorldBody.MESH_BODY:
        m.type = Marker.MESH_RESOURCE
        m.mesh_resource = world_body.mesh
    m.color = ColorRGBA(0,1,0,0.8)
    return m


def urdf_object_to_marker_msg(urdf_object):
    """
    :type urdf_object: UrdfObject
    :rtype: MarkerArray
    """
    ma = MarkerArray()
    for visual_property in urdf_object.visual_props:
        m = Marker()
        m.color.r = 0
        m.color.g = 1
        m.color.b = 0
        m.color.a = 0.8
        m.ns = u'bullet/{}'.format(urdf_object.name)
        m.action = Marker.ADD
        m.id = 1337
        m.header.frame_id = u'map'
        m.pose.position.x = visual_property.origin.translation.x
        m.pose.position.y = visual_property.origin.translation.y
        m.pose.position.z = visual_property.origin.translation.z
        m.pose.orientation.x = visual_property.origin.rotation.x
        m.pose.orientation.y = visual_property.origin.rotation.y
        m.pose.orientation.z = visual_property.origin.rotation.z
        m.pose.orientation.w = visual_property.origin.rotation.w
        if isinstance(visual_property.geometry, BoxShape):
            m.type = Marker.CUBE
            m.scale.x = visual_property.geometry.x
            m.scale.y = visual_property.geometry.y
            m.scale.z = visual_property.geometry.z
        if isinstance(visual_property.geometry, MeshShape):
            m.type = Marker.MESH_RESOURCE
            m.mesh_resource = visual_property.geometry.filename
            m.scale.x = visual_property.geometry.scale[0]
            m.scale.y = visual_property.geometry.scale[1]
            m.scale.z = visual_property.geometry.scale[2]
            m.mesh_use_embedded_materials = True
        ma.markers.append(m)
    return ma

def world_body_to_urdf_object(world_body_msg):
    """
    Converts a body from a ROS message to the corresponding internal representation.
    :param world_body_msg: Input message that shall be converted.
    :type world_body_msg: WorldBody
    :return: Internal representation of body, filled with data from input message.
    :rtype UrdfObject
    """
    if world_body_msg.type is WorldBody.MESH_BODY:
        geom = MeshShape(filename=world_body_msg.mesh)
    elif world_body_msg.type is WorldBody.PRIMITIVE_BODY:
        if world_body_msg.shape.type is SolidPrimitive.BOX:
            geom = BoxShape(world_body_msg.shape.dimensions[SolidPrimitive.BOX_X],
                            world_body_msg.shape.dimensions[SolidPrimitive.BOX_Y],
                            world_body_msg.shape.dimensions[SolidPrimitive.BOX_Z])
        elif world_body_msg.shape.type is SolidPrimitive.CYLINDER:
            geom = CylinderShape(world_body_msg.shape.dimensions[SolidPrimitive.CYLINDER_RADIUS],
                                 world_body_msg.shape.dimensions[SolidPrimitive.CYLINDER_HEIGHT])
        elif world_body_msg.shape.type is SolidPrimitive.SPHERE:
            geom = SphereShape(world_body_msg.shape.dimensions[SolidPrimitive.SPHERE_RADIUS])
        else:
            raise CorruptShapeException(u'Invalid primitive shape \'{}\' of world body \'{}\''.format(world_body_msg.shape.type, world_body_msg.name))
    elif world_body_msg.type is WorldBody.URDF_BODY:
        # TODO: complete me
        pass
    else:
        # TODO: replace me by a proper exception that can be reported back to the service client
        raise RuntimeError(u'Invalid shape of world body: {}'.format(world_body_msg.shape))
    col = CollisionProperty(name=world_body_msg.name + u'_col', geometry=geom)
    vis = VisualProperty(name=world_body_msg.name + u'_vis', geometry=geom)
    return UrdfObject(name=world_body_msg.name, collision_props=[col], visual_props=[vis])

def from_point_msg(point_msg):
    """

    :param point_msg:
    :type point_msg: PointMsg
    :return:
    :rtype: Point
    """
    return Point(point_msg.x, point_msg.y, point_msg.z)


def from_quaternion_msg(q_msg):
    """

    :param quaternion_msg:
    :type quaternion_msg: QuaternionMsg
    :return:
    :rtype: Quaternion
    """
    return Quaternion(q_msg.x, q_msg.y, q_msg.z, q_msg.w)


def from_pose_msg(pose_msg):
    """

    :param pose_msg:
    :type pose_msg: PoseMsg
    :return:
    :rtype: Transform
    """
    return Transform(from_point_msg(pose_msg.position), from_quaternion_msg(pose_msg.orientation))
