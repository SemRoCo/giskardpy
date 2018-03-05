from giskardpy.trajectory import Transform


class URDFSerializationInterface(object):
    #TODO init that sets the function parameters
    def to_xml_string(self):
        pass

    def from_xml_string(self, xml_string):
        pass


class ColorRgba(URDFSerializationInterface):
    r = 1.0
    g = 1.0
    b = 1.0
    a = 1.0


class InertiaMatrix(URDFSerializationInterface):
    ixx = 0.0  # TODO 1?
    ixy = 0.0
    ixz = 0.0
    iyy = 0.0  # TODO 1?
    iyz = 0.0
    izz = 0.0  # TODO 1?


class GeometricShape(URDFSerializationInterface):
    pass


class BoxShape(GeometricShape):
    x = 0.0
    y = 0.0
    z = 0.0


class CylinderShape(GeometricShape):
    radius = 0.0
    length = 0.0


class SphereShape(GeometricShape):
    radius = 0.0


class MeshShape(GeometricShape):
    filename = ''
    scale = [1.0, 1.0, 1.0]


class InertialProperty(URDFSerializationInterface):
    origin = Transform()
    mass = 0.0
    inertia = InertiaMatrix()


class MaterialProperty(URDFSerializationInterface):
    name = ''
    color = ColorRgba()
    texture_filename = ''


class VisualProperty(URDFSerializationInterface):
    name = ''
    origin = Transform()
    geometry = None
    material = None


class CollisionProperty(URDFSerializationInterface):
    name = ''
    origin = Transform()
    geometry = None


class WorldObject(URDFSerializationInterface):
    name = ''
    inertial_props = InertialProperty()
    visual_props = []
    collision_props = []
