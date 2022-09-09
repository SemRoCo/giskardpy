import os

import numpy as np
import urdf_parser_py.urdf as up
from geometry_msgs.msg import Pose
from std_msgs.msg import ColorRGBA
from tf.transformations import euler_matrix
from visualization_msgs.msg import Marker, MarkerArray

from giskardpy.data_types import PrefixName
from giskardpy.exceptions import CorruptShapeException
from giskardpy.model.utils import cube_volume, cube_surface, sphere_volume, cylinder_volume, cylinder_surface
from giskardpy.my_types import my_string
from giskardpy.utils.tfwrapper import np_to_pose
from giskardpy.utils.utils import resolve_ros_iris


class LinkGeometry(object):
    def __init__(self, link_T_geometry):
        self.color = ColorRGBA(20/255, 27.1/255, 41.2/255, 0.2)
        self.link_T_geometry = link_T_geometry

    @classmethod
    def from_urdf(cls, urdf_thing):
        urdf_geometry = urdf_thing.geometry
        if urdf_thing.origin is None:
            link_T_geometry = np.eye(4)
        else:
            link_T_geometry = euler_matrix(*urdf_thing.origin.rpy)
            link_T_geometry[0, 3] = urdf_thing.origin.xyz[0]
            link_T_geometry[1, 3] = urdf_thing.origin.xyz[1]
            link_T_geometry[2, 3] = urdf_thing.origin.xyz[2]
        if isinstance(urdf_geometry, up.Mesh):
            geometry = MeshGeometry(link_T_geometry, urdf_geometry.filename, urdf_geometry.scale)
        elif isinstance(urdf_geometry, up.Box):
            geometry = BoxGeometry(link_T_geometry, *urdf_geometry.size)
        elif isinstance(urdf_geometry, up.Cylinder):
            geometry = CylinderGeometry(link_T_geometry, urdf_geometry.length, urdf_geometry.radius)
        elif isinstance(urdf_geometry, up.Sphere):
            geometry = SphereGeometry(link_T_geometry, urdf_geometry.radius)
        else:
            NotImplementedError('{} geometry is not supported'.format(type(urdf_geometry)))
        return geometry

    @classmethod
    def from_world_body(cls, msg):
        """
        :type msg: giskard_msgs.msg._WorldBody.WorldBody
        """
        if msg.type == msg.URDF_BODY:
            raise NotImplementedError()
        elif msg.type == msg.PRIMITIVE_BODY:
            if msg.shape.type == msg.shape.BOX:
                geometry = BoxGeometry(np.eye(4),
                                       depth=msg.shape.dimensions[msg.shape.BOX_X],
                                       width=msg.shape.dimensions[msg.shape.BOX_Y],
                                       height=msg.shape.dimensions[msg.shape.BOX_Z])
            elif msg.shape.type == msg.shape.CYLINDER:
                geometry = CylinderGeometry(np.eye(4),
                                            height=msg.shape.dimensions[msg.shape.CYLINDER_HEIGHT],
                                            radius=msg.shape.dimensions[msg.shape.CYLINDER_RADIUS])
            elif msg.shape.type == msg.shape.SPHERE:
                geometry = SphereGeometry(np.eye(4),
                                          radius=msg.shape.dimensions[msg.shape.SPHERE_RADIUS])
            else:
                raise CorruptShapeException(f'Primitive shape of type {msg.shape.type} not supported.')
        elif msg.type == msg.MESH_BODY:
            geometry = MeshGeometry(np.eye(4), msg.mesh, scale=[msg.scale.x, msg.scale.y, msg.scale.z])
        else:
            raise CorruptShapeException(f'World body type {msg.type} not supported')
        return geometry

    def as_visualization_marker(self):
        marker = Marker()
        marker.color = self.color

        marker.pose = Pose()
        marker.pose = np_to_pose(self.link_T_geometry)
        return marker

    def is_big(self, volume_threshold=1.001e-6, surface_threshold=0.00061):
        return False


class MeshGeometry(LinkGeometry):
    def __init__(self, link_T_geometry, file_name, scale=None):
        super().__init__(link_T_geometry)
        self.file_name = file_name
        if not os.path.isfile(resolve_ros_iris(file_name)):
            raise CorruptShapeException(f'Can\'t find file {self.file_name}')
        if scale is None:
            self.scale = [1, 1, 1]
        else:
            self.scale = scale

    def as_visualization_marker(self):
        marker = super().as_visualization_marker()
        marker.type = Marker.MESH_RESOURCE
        marker.mesh_resource = self.file_name
        marker.scale.x = self.scale[0]
        marker.scale.y = self.scale[1]
        marker.scale.z = self.scale[2]
        marker.mesh_use_embedded_materials = True
        return marker

    def as_urdf(self):
        return up.Mesh(self.file_name, self.scale)

    def is_big(self, volume_threshold=1.001e-6, surface_threshold=0.00061):
        return True


class BoxGeometry(LinkGeometry):
    def __init__(self, link_T_geometry, depth, width, height):
        super(BoxGeometry, self).__init__(link_T_geometry)
        self.depth = depth
        self.width = width
        self.height = height

    def as_visualization_marker(self):
        marker = super(BoxGeometry, self).as_visualization_marker()
        marker.type = Marker.CUBE
        marker.scale.x = self.depth
        marker.scale.y = self.width
        marker.scale.z = self.height
        return marker

    def as_urdf(self):
        return up.Box([self.depth, self.width, self.height])

    def is_big(self, volume_threshold=1.001e-6, surface_threshold=0.00061):
        return (cube_volume(self.depth, self.width, self.height) > volume_threshold or
                cube_surface(self.depth, self.width, self.height) > surface_threshold)


class CylinderGeometry(LinkGeometry):
    def __init__(self, link_T_geometry, height, radius):
        super(CylinderGeometry, self).__init__(link_T_geometry)
        self.height = height
        self.radius = radius

    def as_visualization_marker(self):
        marker = super(CylinderGeometry, self).as_visualization_marker()
        marker.type = Marker.CYLINDER
        marker.scale.x = self.radius * 2
        marker.scale.y = self.radius * 2
        marker.scale.z = self.height
        return marker

    def as_urdf(self):
        return up.Cylinder(self.radius, self.height)

    def is_big(self, volume_threshold=1.001e-6, surface_threshold=0.00061):
        return (cylinder_volume(self.radius, self.height) > volume_threshold or
                cylinder_surface(self.radius, self.height) > surface_threshold)


class SphereGeometry(LinkGeometry):
    def __init__(self, link_T_geometry, radius):
        super(SphereGeometry, self).__init__(link_T_geometry)
        self.radius = radius

    def as_visualization_marker(self):
        marker = super(SphereGeometry, self).as_visualization_marker()
        marker.type = Marker.SPHERE
        marker.scale.x = self.radius * 2
        marker.scale.y = self.radius * 2
        marker.scale.z = self.radius * 2
        return marker

    def as_urdf(self):
        return up.Sphere(self.radius)

    def is_big(self, volume_threshold=1.001e-6, surface_threshold=0.00061):
        return sphere_volume(self.radius) > volume_threshold


class Link:
    def __init__(self, name: my_string):
        if isinstance(name, str):
            name = PrefixName(name, None)
        self.name = name
        self.visuals = []
        self.collisions = []
        self.parent_joint_name = None
        self.child_joint_names = []

    def name_with_collision_id(self, collision_id):
        if collision_id > len(self.collisions):
            raise AttributeError(f'Link {self.name} only has {len(self.collisions)} collisions, '
                                 f'asking for {collision_id}.')
        if collision_id == 0:
            return self.name
        return f'{self.name}/\\{collision_id}'

    @classmethod
    def from_urdf(cls, urdf_link, prefix):
        link_name = PrefixName(urdf_link.name, prefix)
        link = cls(link_name)
        for urdf_collision in urdf_link.collisions:
            link.collisions.append(LinkGeometry.from_urdf(urdf_collision))
        for urdf_visual in urdf_link.visuals:
            link.visuals.append(LinkGeometry.from_urdf(urdf_visual))
        return link

    @classmethod
    def from_world_body(cls, prefix, msg):
        """
        :type msg: giskard_msgs.msg._WorldBody.WorldBody
        :type pose: Pose
        """
        link_name = PrefixName(prefix, None)
        link = cls(link_name)
        geometry = LinkGeometry.from_world_body(msg)
        link.collisions.append(geometry)
        link.visuals.append(geometry)
        return link

    def dye_collisions(self, color: ColorRGBA):
        if self.has_collisions():
            for collision in self.collisions:
                collision.color = color

    def collision_visualization_markers(self):
        markers = MarkerArray()
        for collision in self.collisions:  # type: LinkGeometry
            marker = collision.as_visualization_marker()
            markers.markers.append(marker)
        return markers

    def as_urdf(self):
        r = up.Robot(self.name)
        r.version = '1.0'
        link = up.Link(self.name)
        # if self.visuals:
        #     link.add_aggregate('visual', up.Visual(self.visuals[0].as_urdf()))
        link.add_aggregate('collision', up.Collision(self.collisions[0].as_urdf()))
        r.add_link(link)
        return r.to_xml_string()

    def has_visuals(self):
        return len(self.visuals) > 0

    def has_collisions(self, volume_threshold=1.001e-6, surface_threshold=0.00061):
        """
        :type link: str
        :param volume_threshold: m**3, ignores simple geometry shapes with a volume less than this
        :type volume_threshold: float
        :param surface_threshold: m**2, ignores simple geometry shapes with a surface area less than this
        :type surface_threshold: float
        :return: True if collision geometry is mesh or simple shape with volume/surface bigger than thresholds.
        :rtype: bool
        """
        for collision in self.collisions:
            geo = collision
            if geo.is_big():
                return True
        return False

    def __repr__(self):
        return str(self.name)
