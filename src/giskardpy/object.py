from visualization_msgs.msg import MarkerArray, Marker
from giskard_msgs.msg import WorldBody
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose as PoseMsg, Point as PointMsg, Quaternion as QuaternionMsg

from giskardpy.exceptions import CorruptShapeException
from giskardpy.trajectory import Transform, Point, Quaternion
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

# TODO add ConeShape

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
    def __init__(self, name='', color=ColorRgba(), texture_filename=''):
        self.name = name
        self.color = color
        self.texture_filename = texture_filename


class VisualProperty(object):
    def __init__(self, name='', origin=Transform(), geometry=None, material=None):
        self.name = name
        self.origin = origin
        self.geometry = geometry
        self.material = material


class CollisionProperty(object):
    def __init__(self, name='', origin=Transform(), geometry=None):
        self.name = name
        self.origin = origin
        self.geometry = geometry


class WorldObject(object):
    def __init__(self, name='', inertial_props=None, visual_props=(), collision_props=()):
        self.name = name
        self.inertial_props = inertial_props
        self.visual_props = visual_props
        self.collision_props = collision_props

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
    if isinstance(urdf_object, WorldObject):
        link = etree.Element('link', name='{}_link'.format(urdf_object.name))
        if urdf_object.inertial_props:
            link.append(to_urdf_xml(urdf_object.inertial_props))
        for visual in urdf_object.visual_props:
            link.append(to_urdf_xml(visual))
        for collision in urdf_object.collision_props:
            link.append(to_urdf_xml(collision))
        if skip_robot_tag:
            root = link
        else:
            root = etree.Element('robot', name=urdf_object.name)
            root.append(link)
    elif isinstance(urdf_object, InertialProperty):
        root = etree.Element('inertial')
        root.append(to_urdf_xml(urdf_object.origin))
        root.append(to_urdf_xml(urdf_object.inertia))
        mass = etree.Element('mass', value=str(urdf_object.mass))
        root.append(mass)
    elif isinstance(urdf_object, VisualProperty):
        if urdf_object.name:
            root = etree.Element('visual', name=urdf_object.name)
        else:
            root = etree.Element('visual')
        root.append(to_urdf_xml(urdf_object.origin))
        root.append(to_urdf_xml(urdf_object.geometry))
        if urdf_object.material:
            root.append(to_urdf_xml(urdf_object.material))
    elif isinstance(urdf_object, CollisionProperty):
        root = etree.Element('collision', name=urdf_object.name)
        root.append(to_urdf_xml(urdf_object.origin))
        root.append(to_urdf_xml(urdf_object.geometry))
    elif isinstance(urdf_object, Transform):
        r = kdl.Rotation.Quaternion(urdf_object.rotation.x, urdf_object.rotation.y,
                                    urdf_object.rotation.z, urdf_object.rotation.w)
        rpy = r.GetRPY()
        rpy_string = '{} {} {}'.format(rpy[0], rpy[1], rpy[2])
        xyz_string = '{} {} {}'.format(urdf_object.translation.x, urdf_object.translation.y, urdf_object.translation.z)
        root = etree.Element('origin', xyz=xyz_string, rpy=rpy_string)
    elif isinstance(urdf_object, InertiaMatrix):
        root = etree.Element('inertia', ixx=str(urdf_object.ixx), ixy=str(urdf_object.ixy), ixz=str(urdf_object.ixz),
                             iyy=str(urdf_object.iyy), iyz=str(urdf_object.iyz), izz=str(urdf_object.izz))
    elif isinstance(urdf_object, BoxShape):
        root = etree.Element('geometry')
        size_string = '{} {} {}'.format(urdf_object.x, urdf_object.y, urdf_object.z)
        box = etree.Element('box', size=size_string)
        root.append(box)
    elif isinstance(urdf_object, CylinderShape):
        root = etree.Element('geometry')
        cyl = etree.Element('cylinder', radius=str(urdf_object.radius), length=str(urdf_object.length))
        root.append(cyl)
    elif isinstance(urdf_object, SphereShape):
        root = etree.Element('geometry')
        sphere = etree.Element('sphere', radius=str(urdf_object.radius))
        root.append(sphere)
    elif isinstance(urdf_object, MeshShape):
        root = etree.Element('geometry')
        scale_string = '{} {} {}'.format(urdf_object.scale[0], urdf_object.scale[1], urdf_object.scale[2])
        mesh = etree.Element('mesh', scale=scale_string, filename=urdf_object.filename)
        root.append(mesh)
    elif isinstance(urdf_object, MaterialProperty):
        root = etree.Element('material', name=urdf_object.name)
        if urdf_object.color:
            color_string = '{} {} {} {}'.format(str(urdf_object.color.r), str(urdf_object.color.g),
                                                str(urdf_object.color.b), str(urdf_object.color.a))
            color = etree.Element('color', rgba=color_string)
            root.append(color)
        if urdf_object.texture_filename:
            tex =etree.Element('texture', filename=urdf_object.texture_filename)
            root.append(tex)
    elif isinstance(urdf_object, FixedJoint):
        root = etree.Element('joint', name=urdf_object.name, type='fixed')
        root.append(to_urdf_xml(urdf_object.origin))
        root.append(etree.Element('parent', link=urdf_object.parent_link_name))
        root.append(etree.Element('child', link=urdf_object.child_link_name))
    return root


def to_urdf_string(urdf_object, skip_robot_tag=False):
    """
    :param urdf_object:
    :type urdf_object: WorldObject
    :return:
    :rtype: str
    """
    return etree.tostring(to_urdf_xml(urdf_object, skip_robot_tag=skip_robot_tag))

def to_marker(urdf_object):
    ma = MarkerArray()
    for visual_property in urdf_object.visual_props:
        m = Marker()
        m.color.r = 0
        m.color.g = 1
        m.color.b = 0
        m.color.a = 0.8
        m.ns = 'bullet/{}'.format(urdf_object.name)
        m.action = Marker.ADD
        m.id = 1337
        m.header.frame_id = 'map'
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


def from_msg(body_msg):
    """
    Converts a body from a ROS message to the corresponding internal representation.
    :param body_msg: Input message that shall be converted.
    :type body_msg: WorldBody
    :return: Internal representation of body, filled with data from input message.
    :rtype WorldObject
    """
    if body_msg.type is WorldBody.MESH_BODY:
        geom = MeshShape(filename=body_msg.mesh)
    elif body_msg.type is WorldBody.PRIMITIVE_BODY:
        if body_msg.shape.type is SolidPrimitive.BOX:
            geom = BoxShape(body_msg.shape.dimensions[SolidPrimitive.BOX_X],
                            body_msg.shape.dimensions[SolidPrimitive.BOX_Y],
                            body_msg.shape.dimensions[SolidPrimitive.BOX_Z])
        elif body_msg.shape.type is SolidPrimitive.CYLINDER:
            geom = CylinderShape(body_msg.shape.dimensions[SolidPrimitive.CYLINDER_RADIUS],
                                 body_msg.shape.dimensions[SolidPrimitive.CYLINDER_HEIGHT])
        elif body_msg.shape.type is SolidPrimitive.SPHERE:
            geom = SphereShape(body_msg.shape.dimensions[SolidPrimitive.SPHERE_RADIUS])
        else:
            raise CorruptShapeException("Invalid primitive shape '{}' of world body '{}'".format(body_msg.shape.type, body_msg.name))
    elif body_msg.type is WorldBody.URDF_BODY:
        # TODO: complete me
        pass
    else:
        # TODO: replace me by a proper exception that can be reported back to the service client
        raise RuntimeError("Invalid shape of world body: {}".format(body_msg.shape))
    col = CollisionProperty(name=body_msg.name + '_col', geometry=geom)
    vis = VisualProperty(name=body_msg.name + '_vis', geometry=geom)
    return WorldObject(name=body_msg.name, collision_props=[col], visual_props=[vis])