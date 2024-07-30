from __future__ import annotations

import os
from typing import List, Optional, Union, Tuple, Callable

import urdf_parser_py.urdf as up

from giskardpy.data_types.exceptions import CorruptMeshException
from giskardpy.middleware import middleware
from giskardpy.model.utils import cube_volume, cube_surface, sphere_volume, cylinder_volume, cylinder_surface
from giskardpy.data_types.data_types import PrefixName, ColorRGBA
from giskardpy.utils.utils import get_file_hash
import giskardpy.casadi_wrapper as cas


class LinkGeometry:
    link_T_geometry: cas.TransMatrix
    color: ColorRGBA

    def __init__(self, link_T_geometry: Optional[cas.TransMatrix] = None, color: Optional[ColorRGBA] = None):
        self.color = color or ColorRGBA(20 / 255, 27.1 / 255, 80 / 255, 0.2)
        self.link_T_geometry = link_T_geometry or cas.TransMatrix()

    def to_hash(self) -> str:
        return ''

    def as_urdf(self):
        raise NotImplementedError(f'as_urdf not implemented for {self.__class__.__name__}')

    @classmethod
    def from_urdf(cls, urdf_thing: up.Collision, color: ColorRGBA) -> LinkGeometry:
        urdf_geometry = urdf_thing.geometry
        if urdf_thing.origin is None:
            link_T_geometry = cas.TransMatrix()
        else:
            link_T_geometry = cas.TransMatrix.from_xyz_rpy(0, 0, 0, *urdf_thing.origin.rpy)
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

    def is_big(self, volume_threshold: float = 1.001e-6, surface_threshold: float = 0.00061) -> bool:
        return False


class MeshGeometry(LinkGeometry):
    def __init__(self,
                 file_name: str,
                 link_T_geometry: Optional[cas.TransMatrix] = None,
                 color: Optional[ColorRGBA] = None,
                 scale: Optional[Tuple[float, float, float]] = None):
        super().__init__(link_T_geometry, color)
        self._file_name_ros_iris = file_name
        self.set_collision_file_name(self.file_name_absolute)
        if not os.path.isfile(middleware.resolve_iri(file_name)):
            raise CorruptMeshException(f'Can\'t find file {file_name}')
        self.scale = scale or (1.0, 1.0, 1.0)

    def set_collision_file_name(self, new_file_name: str) -> None:
        self._collision_file_name = new_file_name

    @property
    def file_name_absolute(self) -> str:
        return middleware.resolve_iri(self._file_name_ros_iris)

    @property
    def file_name_ros_iris(self) -> str:
        return self._file_name_ros_iris

    @property
    def collision_file_name_absolute(self) -> str:
        return self._collision_file_name

    def to_hash(self) -> str:
        return get_file_hash(self.file_name_absolute)

    def as_urdf(self) -> up.Mesh:
        return up.Mesh(self.file_name_ros_iris, self.scale)

    def is_big(self, volume_threshold: float = 1.001e-6, surface_threshold: float = 0.00061) -> bool:
        return True


class BoxGeometry(LinkGeometry):
    def __init__(self,
                 depth: float,
                 width: float,
                 height: float,
                 link_T_geometry: Optional[cas.TransMatrix] = None,
                 color: Optional[ColorRGBA] = None):
        super().__init__(link_T_geometry, color)
        self.depth = depth
        self.width = width
        self.height = height

    def to_hash(self) -> str:
        return f'box{self.depth}{self.width}{self.height}'

    def as_urdf(self) -> up.Box:
        return up.Box([self.depth, self.width, self.height])

    def is_big(self, volume_threshold: float = 1.001e-6, surface_threshold: float = 0.00061) -> bool:
        return (cube_volume(self.depth, self.width, self.height) > volume_threshold or
                cube_surface(self.depth, self.width, self.height) > surface_threshold)


class CylinderGeometry(LinkGeometry):
    def __init__(self,
                 height: float,
                 radius: float,
                 link_T_geometry: Optional[cas.TransMatrix] = None,
                 color: Optional[ColorRGBA] = None):
        super().__init__(link_T_geometry, color)
        self.height = height
        self.radius = radius

    def to_hash(self) -> str:
        return f'cylinder{self.height}{self.radius}'

    def as_urdf(self) -> up.Cylinder:
        return up.Cylinder(self.radius, self.height)

    def is_big(self, volume_threshold: float = 1.001e-6, surface_threshold: float = 0.00061) -> bool:
        return (cylinder_volume(self.radius, self.height) > volume_threshold or
                cylinder_surface(self.radius, self.height) > surface_threshold)


class SphereGeometry(LinkGeometry):
    def __init__(self,
                 radius: float,
                 link_T_geometry: Optional[cas.TransMatrix] = None,
                 color: Optional[ColorRGBA] = None):
        super().__init__(link_T_geometry, color)
        self.radius = radius

    def to_hash(self) -> str:
        return f'sphere{self.radius}'

    def as_urdf(self) -> up.Sphere:
        return up.Sphere(self.radius)

    def is_big(self, volume_threshold: float = 1.001e-6, surface_threshold: float = 0.00061) -> bool:
        return sphere_volume(self.radius) > volume_threshold


class Link:
    child_joint_names: List[PrefixName]
    collisions: List[LinkGeometry]
    name: PrefixName
    visuals: List[LinkGeometry]
    parent_joint_name: Optional[PrefixName]

    def __init__(self, name: PrefixName):
        self.name = name
        self.visuals = []
        self.collisions = []
        self.parent_joint_name = None
        self.child_joint_names = []

    def _clear_memo(self, f: Callable) -> None:
        try:
            if hasattr(f, 'memo'):
                f.memo.clear()
            else:
                del f
        except:
            pass

    @classmethod
    def from_urdf(cls, urdf_link: up.Link, prefix: str, color: ColorRGBA) -> Link:
        link_name = PrefixName(urdf_link.name, prefix)
        link = cls(link_name)
        for urdf_collision in urdf_link.collisions:
            link.collisions.append(LinkGeometry.from_urdf(urdf_thing=urdf_collision,
                                                          color=color))
        for urdf_visual in urdf_link.visuals:
            link.visuals.append(LinkGeometry.from_urdf(urdf_thing=urdf_visual,
                                                       color=color))
        return link

    def dye_collisions(self, color: ColorRGBA) -> None:
        if self.has_collisions():
            for collision in self.collisions:
                collision.color = color

    def as_urdf(self) -> str:
        r = up.Robot(self.name)
        r.version = '1.0'
        link = up.Link(self.name)
        link.add_aggregate('collision', up.Collision(self.collisions[0].as_urdf()))
        r.add_link(link)
        return r.to_xml_string()

    def has_visuals(self) -> bool:
        return len(self.visuals) > 0

    def has_collisions(self, volume_threshold: float = 1.001e-6, surface_threshold: float = 0.00061) -> bool:
        """
        :param volume_threshold: m**3, ignores simple geometry shapes with a volume less than this
        :param surface_threshold: m**2, ignores simple geometry shapes with a surface area less than this
        :return: True if collision geometry is mesh or simple shape with volume/surface bigger than thresholds.
        """
        for collision in self.collisions:
            geo = collision
            if geo.is_big(volume_threshold=volume_threshold, surface_threshold=surface_threshold):
                return True
        return False

    def __repr__(self) -> str:
        return str(self.name)
