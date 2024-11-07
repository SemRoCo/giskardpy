from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import urdf_parser_py.urdf as up
from geometry_msgs.msg import Pose
from std_msgs.msg import ColorRGBA
from tf.transformations import euler_matrix
from visualization_msgs.msg import Marker, MarkerArray

from giskard_msgs.msg import WorldBody
from giskardpy.exceptions import CorruptShapeException, CorruptMeshException
from giskardpy.model.utils import cube_volume, cube_surface, sphere_volume, cylinder_volume, cylinder_surface
from giskardpy.data_types import PrefixName
from giskardpy.data_types import my_string
from giskardpy.utils.tfwrapper import np_to_pose, homo_matrix_to_pose
from giskardpy.utils.utils import resolve_ros_iris, get_file_hash
from giskardpy.utils.decorators import memoize, copy_memoize
import giskardpy.casadi_wrapper as w


class LinkGeometry:
    link_T_geometry: w.TransMatrix

    def __init__(self, link_T_geometry: np.ndarray, color: ColorRGBA = None):
        if color is None:
            self.color = ColorRGBA(20 / 255, 27.1 / 255, 80 / 255, 0.2)
        else:
            self.color = color
        self.link_T_geometry = w.TransMatrix(link_T_geometry)

    def to_hash(self) -> str:
        return ''

    @classmethod
    def from_urdf(cls, urdf_thing, color) -> LinkGeometry:
        urdf_geometry = urdf_thing.geometry
        if urdf_thing.origin is None:
            link_T_geometry = np.eye(4)
        else:
            link_T_geometry = euler_matrix(*urdf_thing.origin.rpy)
            link_T_geometry[0, 3] = urdf_thing.origin.xyz[0]
            link_T_geometry[1, 3] = urdf_thing.origin.xyz[1]
            link_T_geometry[2, 3] = urdf_thing.origin.xyz[2]
        if isinstance(urdf_geometry, up.Mesh):
            geometry = MeshGeometry(link_T_geometry=link_T_geometry,
                                    file_name=urdf_geometry.filename,
                                    color=color,
                                    scale=urdf_geometry.scale)
        elif isinstance(urdf_geometry, up.Box):
            geometry = BoxGeometry(link_T_geometry=link_T_geometry,
                                   depth=urdf_geometry.size[0],
                                   width=urdf_geometry.size[1],
                                   height=urdf_geometry.size[2],
                                   color=color)
        elif isinstance(urdf_geometry, up.Cylinder):
            geometry = CylinderGeometry(link_T_geometry=link_T_geometry,
                                        height=urdf_geometry.length,
                                        radius=urdf_geometry.radius,
                                        color=color)
        elif isinstance(urdf_geometry, up.Sphere):
            geometry = SphereGeometry(link_T_geometry=link_T_geometry,
                                      radius=urdf_geometry.radius,
                                      color=color)
        else:
            raise NotImplementedError(f'{type(urdf_geometry)} geometry is not supported')
        return geometry

    @classmethod
    def from_world_body(cls, msg: WorldBody, color: ColorRGBA) -> LinkGeometry:
        if msg.type == msg.URDF_BODY:
            raise NotImplementedError()
        elif msg.type == msg.PRIMITIVE_BODY:
            if msg.shape.type == msg.shape.BOX:
                geometry = BoxGeometry(link_T_geometry=np.eye(4),
                                       depth=msg.shape.dimensions[msg.shape.BOX_X],
                                       width=msg.shape.dimensions[msg.shape.BOX_Y],
                                       height=msg.shape.dimensions[msg.shape.BOX_Z],
                                       color=color)
            elif msg.shape.type == msg.shape.CYLINDER:
                geometry = CylinderGeometry(link_T_geometry=np.eye(4),
                                            height=msg.shape.dimensions[msg.shape.CYLINDER_HEIGHT],
                                            radius=msg.shape.dimensions[msg.shape.CYLINDER_RADIUS],
                                            color=color)
            elif msg.shape.type == msg.shape.SPHERE:
                geometry = SphereGeometry(link_T_geometry=np.eye(4),
                                          radius=msg.shape.dimensions[msg.shape.SPHERE_RADIUS],
                                          color=color)
            else:
                raise CorruptShapeException(f'Primitive shape of type {msg.shape.type} not supported.')
        elif msg.type == msg.MESH_BODY:
            if msg.scale.x == 0 or msg.scale.y == 0 or msg.scale.z == 0:
                raise CorruptShapeException(f'Scale of mesh contains 0: {msg.scale}')
            geometry = MeshGeometry(link_T_geometry=np.eye(4),
                                    file_name=msg.mesh,
                                    scale=[msg.scale.x, msg.scale.y, msg.scale.z],
                                    color=color)
        else:
            raise CorruptShapeException(f'World body type {msg.type} not supported')
        return geometry

    def as_visualization_marker(self, *args, **kwargs) -> Marker:
        marker = Marker()
        marker.color = self.color
        marker.pose = homo_matrix_to_pose(self.link_T_geometry.evaluate())
        return marker

    def is_big(self, volume_threshold: float = 1.001e-6, surface_threshold: float = 0.00061) -> bool:
        return False


class MeshGeometry(LinkGeometry):
    def __init__(self, link_T_geometry: np.ndarray, file_name: str, color: ColorRGBA, scale=None):
        super().__init__(link_T_geometry, color)
        self._file_name_ros_iris = file_name
        self.set_collision_file_name(self.file_name_absolute)
        if not os.path.isfile(resolve_ros_iris(file_name)):
            raise CorruptMeshException(f'Can\'t find file {file_name}')
        if scale is None:
            self.scale = [1, 1, 1]
        else:
            self.scale = scale

    def set_collision_file_name(self, new_file_name: str):
        self._collision_file_name = new_file_name

    @property
    def file_name_absolute(self) -> str:
        return resolve_ros_iris(self._file_name_ros_iris)

    @property
    def file_name_ros_iris(self) -> str:
        return self._file_name_ros_iris

    @property
    def collision_file_name_absolute(self) -> str:
        return self._collision_file_name

    def to_hash(self) -> str:
        return get_file_hash(self.file_name_absolute)

    def as_visualization_marker(self, use_decomposed_meshes, *args, **kwargs) -> Marker:
        marker = super().as_visualization_marker()
        marker.type = Marker.MESH_RESOURCE
        if use_decomposed_meshes:
            marker.mesh_resource = 'file://' + self.collision_file_name_absolute
        else:
            marker.mesh_resource = 'file://' + self.file_name_absolute
        marker.scale.x = self.scale[0]
        marker.scale.y = self.scale[1]
        marker.scale.z = self.scale[2]
        marker.mesh_use_embedded_materials = False
        return marker

    def as_urdf(self):
        return up.Mesh(self.file_name_ros_iris, self.scale)

    def is_big(self, volume_threshold: float = 1.001e-6, surface_threshold: float = 0.00061) -> bool:
        return True


class BoxGeometry(LinkGeometry):
    def __init__(self, link_T_geometry, depth, width, height, color):
        super().__init__(link_T_geometry, color)
        self.depth = depth
        self.width = width
        self.height = height

    def to_hash(self) -> str:
        return f'box{self.depth}{self.width}{self.height}'

    def as_visualization_marker(self, *args, **kwargs):
        marker = super().as_visualization_marker()
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
    def __init__(self, link_T_geometry, height, radius, color):
        super().__init__(link_T_geometry, color)
        self.height = height
        self.radius = radius

    def to_hash(self) -> str:
        return f'cylinder{self.height}{self.radius}'

    def as_visualization_marker(self, *args, **kwargs):
        marker = super().as_visualization_marker()
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
    def __init__(self, link_T_geometry, radius, color):
        super().__init__(link_T_geometry, color)
        self.radius = radius

    def to_hash(self) -> str:
        return f'sphere{self.radius}'

    def as_visualization_marker(self, *args, **kwargs):
        marker = super().as_visualization_marker()
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
    child_joint_names: List[PrefixName]
    collisions: List[LinkGeometry]
    name: PrefixName
    visuals: List[LinkGeometry]
    parent_joint_name: Optional[PrefixName]
    child_joint_names: List[PrefixName]

    def __init__(self, name: my_string):
        if isinstance(name, str):
            name = PrefixName(name, None)
        self.name = name
        self.visuals = []
        self.collisions = []
        self.parent_joint_name = None
        self.child_joint_names = []

    def _clear_memo(self, f):
        try:
            if hasattr(f, 'memo'):
                f.memo.clear()
            else:
                del f
        except:
            pass

    def reset_cache(self):
        self._clear_memo(self.collision_visualization_markers)

    def name_with_collision_id(self, collision_id):
        if collision_id > len(self.collisions):
            raise AttributeError(f'Link {self.name} only has {len(self.collisions)} collisions, '
                                 f'asking for {collision_id}.')
        if collision_id == 0:
            return self.name
        return f'{self.name}/\\{collision_id}'

    @classmethod
    def from_urdf(cls, urdf_link, prefix, color) -> Link:
        link_name = PrefixName(urdf_link.name, prefix)
        link = cls(link_name)
        for urdf_collision in urdf_link.collisions:
            link.collisions.append(LinkGeometry.from_urdf(urdf_thing=urdf_collision,
                                                          color=color))
        for urdf_visual in urdf_link.visuals:
            link.visuals.append(LinkGeometry.from_urdf(urdf_thing=urdf_visual,
                                                       color=color))
        return link

    @classmethod
    def from_world_body(cls, link_name: my_string, msg: WorldBody, color: ColorRGBA) -> Link:
        link = cls(link_name)
        geometry = LinkGeometry.from_world_body(msg=msg,
                                                color=color)
        link.collisions.append(geometry)
        link.visuals.append(geometry)
        return link

    def dye_collisions(self, color: ColorRGBA):
        self.reset_cache()
        if self.has_collisions():
            for collision in self.collisions:
                collision.color = color

    @memoize
    def collision_visualization_markers(self, *args, **kwargs) -> MarkerArray:
        markers = MarkerArray()
        for collision in self.collisions:
            marker = collision.as_visualization_marker(*args, **kwargs)
            markers.markers.append(marker)
        return markers

    @memoize
    def visuals_visualization_markers(self) -> MarkerArray:
        markers = MarkerArray()
        for visual in self.visuals:
            marker = visual.as_visualization_marker(use_decomposed_meshes=False)
            if isinstance(visual, MeshGeometry):
                marker.mesh_use_embedded_materials = True
                marker.color = ColorRGBA()
            markers.markers.append(marker)
        return markers

    def as_urdf(self):
        r = up.Robot(self.name)
        r.version = '1.0'
        link = up.Link(self.name)
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
