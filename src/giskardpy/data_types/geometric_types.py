from __future__ import annotations
from typing import Optional, Union, Tuple, Iterable

import geometry_msgs.msg as ros1_msgs
import numpy as np

from giskardpy.data_types.data_types import PrefixName
from giskardpy.utils.math import quaternion_from_rotation_matrix
import abc


class GeometricType(abc.ABC):
    reference_frame: Optional[PrefixName]

    @classmethod
    def from_ros1_msg(cls, msg) -> GeometricType: ...


class Point(GeometricType):
    @property
    def x(self): ...
    @x.setter
    def x(self, value): ...
    @property
    def y(self): ...
    @y.setter
    def y(self, value): ...
    @property
    def z(self): ...
    @z.setter
    def z(self, value): ...

    def __init__(self, data: Optional[Union[Expression, Point3, Vector3,
                                            geometry_msgs.Point, geometry_msgs.PointStamped,
                                            geometry_msgs.Vector3, geometry_msgs.Vector3Stamped,
                                            ca.SX,
                                            np.ndarray,
                                            Iterable[symbol_expr_float]]] = None): ...

    @classmethod
    def from_xyz(cls,
                 x: Optional[symbol_expr_float] = None,
                 y: Optional[symbol_expr_float] = None,
                 z: Optional[symbol_expr_float] = None): ...

    def norm(self) -> Expression: ...


class Vector3:
    frame_id: Optional[str]
    x: float
    y: float
    z: float

    def __init__(self,
                 x: float = 0,
                 y: float = 0,
                 z: float = 0,
                 frame_id: Optional[str] = None):
        self.frame_id = frame_id
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    @classmethod
    def from_ros1_vector2(cls, msg: ros1_msgs.Vector3Stamped):
        return cls(frame_id=msg.header.frame_id,
                   x=msg.vector.x,
                   y=msg.vector.y,
                   z=msg.vector.z)

    @classmethod
    def from_list(cls, data: Union[Tuple[float, float, float], list, np.ndarray], frame_id: Optional[str] = None):
        return cls(frame_id=frame_id, x=data[0], y=data[1], z=data[2])


class Quaternion:
    frame_id: Optional[str]
    x: float
    y: float
    z: float
    w: float

    def __init__(self,
                 x: float = 0,
                 y: float = 0,
                 z: float = 0,
                 w: float = 1,
                 frame_id: Optional[str] = None):
        self.frame_id = frame_id
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w)

    @classmethod
    def from_ros1_quaternion(cls, msg: ros1_msgs.QuaternionStamped):
        return cls(frame_id=msg.header.frame_id,
                   x=msg.quaternion.x,
                   y=msg.quaternion.y,
                   z=msg.quaternion.z,
                   w=msg.quaternion.w)

    @classmethod
    def from_np_matrix(cls, matrix: np.ndarray, frame_id: Optional[str] = None):
        q = quaternion_from_rotation_matrix(matrix)
        return cls(frame_id=frame_id, x=q[0], y=q[1], z=q[2], w=q[3])


class Pose:
    frame_id: Optional[str]
    position: Point
    orientation: Quaternion

    def __init__(self,
                 position: Optional[Point] = None,
                 orientation: Optional[Quaternion] = None,
                 frame_id: Optional[str] = None):
        self.frame_id = frame_id
        self.position = position or Point()
        self.orientation = orientation or Quaternion()

    @classmethod
    def from_ros1_pose(cls, msg: ros1_msgs.PoseStamped):
        position = Point(frame_id=msg.header.frame_id,
                         x=msg.pose.position.x,
                         y=msg.pose.position.y,
                         z=msg.pose.position.z)
        orientation = Quaternion(frame_id=msg.header.frame_id,
                                 x=msg.pose.orientation.x,
                                 y=msg.pose.orientation.y,
                                 z=msg.pose.orientation.z,
                                 w=msg.pose.orientation.w)
        return cls(frame_id=msg.header.frame_id,
                   position=position,
                   orientation=orientation)

    @classmethod
    def from_np_matrix(cls, matrix: np.matrix):
        quaternion_from_rotation_matrix(matrix)
