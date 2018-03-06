from giskardpy.trajectory import Transform
from lxml import etree
import PyKDL as kdl # TODO: get rid of this dependency

from trajectory import Point, Quaternion


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
    def __init__(self, filename='', scale=[1.0, 1.0, 1.0]):
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
    def __init__(self, name='', inertial_props=None, visual_props=[], collision_props=[]):
        self.name = name
        self.inertial_props = inertial_props
        self.visual_props = visual_props
        self.collision_props = collision_props


def to_urdf_xml(urdf_object):
    if isinstance(urdf_object, WorldObject):
        root = etree.Element('robot', name=urdf_object.name)
        if urdf_object.inertial_props:
            root.append(to_urdf_xml(urdf_object.inertial_props))
        for visual in urdf_object.visual_props:
            root.append(to_urdf_xml(visual))
        for collision in urdf_object.collision_props:
            root.append(to_urdf_xml(collision))
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
                                    urdf_object.rotation.z,urdf_object.rotation.w)
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
    return root


def to_urdf_string(urdf_object):
    return etree.tostring(to_urdf_xml(urdf_object))
