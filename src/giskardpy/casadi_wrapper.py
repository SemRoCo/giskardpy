from __future__ import annotations
import errno
import os
import pickle
from typing import List, Tuple, Union, Callable, Iterable, Optional, overload, Any

import casadi as ca
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Pose
import geometry_msgs.msg as geometry_msgs
from numpy import pi

from giskardpy.utils import logging
from giskardpy.utils.tfwrapper import msg_to_homogeneous_matrix


class CompiledFunction:
    def __init__(self, str_params: Iterable[str], fast_f: ca.Function, shape: Tuple[int, int]):
        self.str_params = str_params
        self.fast_f = fast_f
        self.shape = shape
        self.buf, self.f_eval = fast_f.buffer()
        self.out = np.zeros(self.shape, order='F')
        self.buf.set_res(0, memoryview(self.out))

    def __call__(self, **kwargs) -> np.ndarray:
        filtered_args = [kwargs[k] for k in self.str_params]
        return self.call2(filtered_args)

    def call2(self, filtered_args: Iterable[float]) -> np.ndarray:
        """
        :param filtered_args: parameter values in the same order as in self.str_params
        """

        filtered_args = np.array(filtered_args, dtype=float)
        self.buf.set_arg(0, memoryview(filtered_args))
        self.f_eval()
        return self.out


class Symbol:
    def __init__(self, name: str):
        self.s: ca.SX = ca.SX.sym(name)

    def free_symbols(self) -> List[ca.SX]:
        return free_symbols(self.s)

    def __str__(self):
        return str(self.s)

    def __repr__(self):
        return repr(self.s)

    def __add__(self, other: Union[Symbol, Expression, float]) -> Expression:
        if isinstance(other, Symbol):
            other = other.s
        return Expression(self.s.__add__(other))

    def __radd__(self, other: Union[Symbol, Expression, float]) -> Expression:
        if isinstance(other, Symbol):
            other = other.s
        return Expression(self.s.__radd__(other))

    def __sub__(self, other: Union[Symbol, Expression, float]) -> Expression:
        if isinstance(other, Symbol):
            other = other.s
        return Expression(self.s.__sub__(other))

    def __rsub__(self, other: Union[Symbol, Expression, float]) -> Expression:
        if isinstance(other, Symbol):
            other = other.s
        return Expression(self.s.__rsub__(other))

    def __mul__(self, other: Union[Symbol, Expression, float]) -> Expression:
        if isinstance(other, Symbol):
            other = other.s
        return Expression(self.s.__mul__(other))

    def __rmul__(self, other: Union[Symbol, Expression, float]) -> Expression:
        if isinstance(other, Symbol):
            other = other.s
        return Expression(self.s.__rmul__(other))

    def __truediv__(self, other: Union[Symbol, Expression, float]) -> Expression:
        if isinstance(other, Symbol):
            other = other.s
        return Expression(self.s.__truediv__(other))

    def __rtruediv__(self, other: Union[Symbol, Expression, float]) -> Expression:
        if isinstance(other, Symbol):
            other = other.s
        return Expression(self.s.__rtruediv__(other))

    def __lt__(self, other: Union[Symbol, Expression, float]) -> Expression:
        if isinstance(other, Symbol):
            other = other.s
        return Expression(self.s.__lt__(other))

    def __le__(self, other: Union[Symbol, Expression, float]) -> Expression:
        if isinstance(other, Symbol):
            other = other.s
        return Expression(self.s.__le__(other))

    def __gt__(self, other: Union[Symbol, Expression, float]) -> Expression:
        if isinstance(other, Symbol):
            other = other.s
        return Expression(self.s.__gt__(other))

    def __ge__(self, other: Union[Symbol, Expression, float]) -> Expression:
        if isinstance(other, Symbol):
            other = other.s
        return Expression(self.s.__ge__(other))

    def __eq__(self, other: Union[Symbol, Expression, float]) -> Expression:
        if isinstance(other, Symbol):
            other = other.s
        return Expression(self.s.__eq__(other))

    def __ne__(self, other: Union[Symbol, Expression, float]) -> Expression:
        if isinstance(other, Symbol):
            other = other.s
        return Expression(self.s.__ne__(other))

    def __neg__(self) -> Expression:
        return Expression(self.s.__neg__())

    def __pow__(self, other: Union[Symbol, Expression, float]) -> Expression:
        if isinstance(other, Symbol):
            other = other.s
        return Expression(self.s.__pow__(other))

    def __rpow__(self, other: Union[Symbol, Expression, float]) -> Expression:
        if isinstance(other, Symbol):
            other = other.s
        return Expression(self.s.__rpow__(other))

    def __matmul__(self, other: Union[Symbol, Expression, float]) -> Expression:
        if isinstance(other, Symbol):
            other = other.s
        return Expression(self.s.__matmul__(other))

    def __rmatmul__(self, other: Union[Symbol, Expression, float]) -> Expression:
        if isinstance(other, Symbol):
            other = other.s
        return Expression(self.s.__rmatmul__(other))


class Expression(Symbol):
    def __init__(self, data: Union[Symbol,
                                   Expression,
                                   float,
                                   Iterable[Union[Symbol, float]],
                                   Iterable[Iterable[Union[Symbol, float]]],
                                   np.ndarray]):
        if isinstance(data, (Symbol, Expression)):
            self.s = ca.SX(data.s)
        elif isinstance(data, (int, float)):
            self.s = ca.SX(data)
        else:
            if hasattr(data, 'shape'):
                self.s = ca.SX(*data.shape)
                y = data.shape[1]
            else:
                x = len(data)
                if isinstance(data[0], list) or isinstance(data[0], tuple):
                    y = len(data[0])
                else:
                    y = 1
                self.s = ca.SX(x, y)
            for i in range(self.shape[0]):
                if y > 1:
                    for j in range(self.shape[1]):
                        try:
                            self[i, j] = data[i][j]
                        except:
                            self[i, j] = data[i, j]
                else:
                    if isinstance(data[i], Symbol):
                        self[i] = data[i].s
                    else:
                        self[i] = data[i]

    def __getitem__(self, item):
        return Expression(self.s[item])

    def __setitem__(self, key, value: Union[float, Symbol, Expression]):
        if isinstance(value, Symbol):
            value = value.s
        self.s[key] = value

    @property
    def shape(self) -> Tuple[int, int]:
        return self.s.shape

    def compile(self, parameters: List[Symbol]) -> CompiledFunction:
        str_params = [str(x) for x in parameters]
        try:
            f = ca.Function('f', [Expression(parameters).s], [ca.densify(self.s)])
        except:
            f = ca.Function('f', [Expression(parameters).s], ca.densify(self.s))
        return CompiledFunction(str_params, f, self.shape)

    @property
    def T(self):
        return self.s.T


class TransMatrix(Expression):
    def __init__(self, data: Optional[Union[Iterable[Union[Symbol, float]],
                                            Iterable[Iterable[Union[Symbol, float]]],
                                            np.ndarray,
                                            Expression]] = None):
        if data is None:
            data = np.eye(4)
        super().__init__(data)
        if self.shape[0] != 4 or self.shape[1] != 4:
            raise ValueError(f'{self.__class__.__name__} can only be initialized with 4x4 shaped data.')
        self[3, 0] = 0
        self[3, 1] = 0
        self[3, 2] = 0
        self[3, 3] = 1

    @classmethod
    def from_xyz(cls, x: Union[Symbol, float], y: Union[Symbol, float], z: Union[Symbol, float]) \
            -> TransMatrix:
        r = cls()
        r[0, 3] = x
        r[1, 3] = y
        r[2, 3] = z
        return r

    @classmethod
    def from_ros_msg(cls, msg: Union[Pose, PoseStamped]) -> TransMatrix:
        if isinstance(msg, PoseStamped):
            msg = msg.pose
        p = Point3(x=msg.position.x,
                   y=msg.position.y,
                   z=msg.position.z)
        q = Quaternion(x=msg.orientation.x,
                       y=msg.orientation.y,
                       z=msg.orientation.z,
                       w=msg.orientation.w)
        return cls.from_parts(p, q)

    @classmethod
    def from_parts(cls,
                   point3: Optional[Point3] = None,
                   rotation: Optional[Union[RotationMatrix,
                                            Quaternion,
                                            Tuple[Vector3, Union[Symbol, float]],
                                            Tuple[Union[Symbol, float],
                                                  Union[Symbol, float],
                                                  Union[Symbol, float]]]] = None) -> TransMatrix:
        if point3 is not None:
            a_T_b = TransMatrix.from_xyz(point3.x, point3.y, point3.z)
        else:
            a_T_b = TransMatrix()
        if isinstance(rotation, RotationMatrix):
            a_R_b = rotation
        elif isinstance(rotation, Quaternion):
            a_R_b = RotationMatrix.from_quaternion(rotation)
        elif len(rotation) == 2:
            a_R_b = RotationMatrix.from_axis_angle(rotation[0], rotation[1])
        elif rotation is not None:
            a_R_b = RotationMatrix.from_xyz_rpy(rotation[0], rotation[1], rotation[2])
        else:
            a_R_b = TransMatrix()
        return a_T_b.dot(a_R_b)

    @overload
    def dot(self, other: Point3) -> Point3:
        ...

    @overload
    def dot(self, other: Vector3) -> Vector3:
        ...

    @overload
    def dot(self, other: TransMatrix) -> TransMatrix:
        ...

    def dot(self, other):
        result = dot(self, other)
        if isinstance(other, Point3):
            return Point3.from_matrix(result)
        if isinstance(other, Vector3):
            return Vector3.from_matrix(result)
        return self.__class__(result)

    def inverse(self) -> TransMatrix:
        inv = eye(4)
        inv[:3, :3] = self[:3, :3].T
        inv[:3, 3] = dot(-inv[:3, :3], self[:3, 3])
        return self.__class__(inv)

    @classmethod
    def from_xyz_rpy(cls, x: Union[Symbol, float], y: Union[Symbol, float], z: Union[Symbol, float],
                     roll: Union[Symbol, float], pitch: Union[Symbol, float], yaw: Union[Symbol, float]) \
            -> TransMatrix:
        a_T_b = TransMatrix.from_xyz(x, y, z)
        a_R_b = RotationMatrix.from_xyz_rpy(roll, pitch, yaw)
        return cls(dot(a_T_b, a_R_b))

    @classmethod
    def from_xy_yaw(cls, x: Union[Symbol, float], y: Union[Symbol, float], yaw: Union[Symbol, float]) \
            -> TransMatrix:
        p = Point3(x, y, 0)
        axis = Vector3(0, 0, 1)
        return cls.from_parts(p, (axis, yaw))

    def position(self):
        return Point3.from_matrix(self[:4, 3:])

    def to_translation(self) -> TransMatrix:
        """
        :return: sets the rotation part of a frame to identity
        """
        r = eye(4)
        r[0, 3] = self[0, 3]
        r[1, 3] = self[1, 3]
        r[2, 3] = self[2, 3]
        return TransMatrix(r)

    def to_rotation(self) -> RotationMatrix:
        return RotationMatrix(self)


class RotationMatrix(TransMatrix):
    def __init__(self, data: Union[Iterable[Union[Symbol, float]],
                                   Iterable[Iterable[Union[Symbol, float]]],
                                   np.ndarray,
                                   Expression]):
        super().__init__(data)
        self[0, 3] = 0
        self[1, 3] = 0
        self[2, 3] = 0

    @classmethod
    def from_axis_angle(cls, axis: Vector3, angle: Union[Symbol, float]) -> RotationMatrix:
        """
        Conversion of unit axis and angle to 4x4 rotation matrix according to:
        https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/index.htm
        """
        ct = ca.cos(angle)
        st = ca.sin(angle)
        vt = 1 - ct
        m_vt_0 = vt * axis[0]
        m_vt_1 = vt * axis[1]
        m_vt_2 = vt * axis[2]
        m_st_0 = axis[0] * st
        m_st_1 = axis[1] * st
        m_st_2 = axis[2] * st
        m_vt_0_1 = m_vt_0 * axis[1]
        m_vt_0_2 = m_vt_0 * axis[2]
        m_vt_1_2 = m_vt_1 * axis[2]
        return cls([[ct + m_vt_0 * axis[0], -m_st_2 + m_vt_0_1, m_st_1 + m_vt_0_2, 0],
                    [m_st_2 + m_vt_0_1, ct + m_vt_1 * axis[1], -m_st_0 + m_vt_1_2, 0],
                    [-m_st_1 + m_vt_0_2, m_st_0 + m_vt_1_2, ct + m_vt_2 * axis[2], 0],
                    [0, 0, 0, 1]])

    @classmethod
    def from_quaternion(cls, q: Quaternion) -> RotationMatrix:
        """
        Unit quaternion to 4x4 rotation matrix according to:
        https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/src/frames.cpp#L167
        """
        x = q[0]
        y = q[1]
        z = q[2]
        w = q[3]
        x2 = x * x
        y2 = y * y
        z2 = z * z
        w2 = w * w
        return cls([[w2 + x2 - y2 - z2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y, 0],
                    [2 * x * y + 2 * w * z, w2 - x2 + y2 - z2, 2 * y * z - 2 * w * x, 0],
                    [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, w2 - x2 - y2 + z2, 0],
                    [0, 0, 0, 1]])

    @classmethod
    def from_ros_msg(cls, msg: Union[geometry_msgs.Quaternion, geometry_msgs.QuaternionStamped]) -> TransMatrix:
        if isinstance(msg, geometry_msgs.QuaternionStamped):
            msg = msg.quaternion
        q = Quaternion(x=msg.x,
                       y=msg.y,
                       z=msg.z,
                       w=msg.w)
        return cls.from_quaternion(q)

    def to_axis_angle(self) -> Tuple[Vector3, Union[Symbol, float]]:
        return self.to_quaternion().to_axis_angle()

    def to_angle(self, hint: Callable) -> Symbol:
        """
        :param hint: A function who's sign of the result will be used to to determine if angle should be positive or
                        negative
        :return:
        """
        axis, angle = self.to_axis_angle()
        return normalize_angle(if_greater_zero(hint(axis),
                                               if_result=angle,
                                               else_result=-angle))

    @classmethod
    def from_vectors(cls,
                     x: Optional[Vector3] = None,
                     y: Optional[Vector3] = None,
                     z: Optional[Vector3] = None) -> RotationMatrix:
        if x is not None:
            x = scale(x, 1)
        if y is not None:
            y = scale(y, 1)
        if z is not None:
            z = scale(z, 1)
        if x is not None and y is not None and z is None:
            z = cross(x, y)
            z = scale(z, 1)
        elif x is not None and y is None and z is not None:
            y = cross(z, x)
            y = scale(y, 1)
        elif x is None and y is not None and z is not None:
            x = cross(y, z)
            x = scale(x, 1)
        else:
            raise AttributeError(f'only one vector can be None')
        R = cls([[x[0], y[0], z[0], 0],
                 [x[1], y[1], z[1], 0],
                 [x[2], y[2], z[2], 0],
                 [0, 0, 0, 1]])
        R.normalize()
        return R

    @classmethod
    def from_xyz_rpy(cls, roll: Union[Symbol, float], pitch: Union[Symbol, float],
                     yaw: Union[Symbol, float]) -> Expression:
        """
        Conversion of roll, pitch, yaw to 4x4 rotation matrix according to:
        https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/src/frames.cpp#L167
        """
        rx = RotationMatrix([[1, 0, 0, 0],
                             [0, ca.cos(roll), -ca.sin(roll), 0],
                             [0, ca.sin(roll), ca.cos(roll), 0],
                             [0, 0, 0, 1]])
        ry = RotationMatrix([[ca.cos(pitch), 0, ca.sin(pitch), 0],
                             [0, 1, 0, 0],
                             [-ca.sin(pitch), 0, ca.cos(pitch), 0],
                             [0, 0, 0, 1]])
        rz = RotationMatrix([[ca.cos(yaw), -ca.sin(yaw), 0, 0],
                             [ca.sin(yaw), ca.cos(yaw), 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
        return cls(dot(rz, ry, rx))

    def inverse(self) -> RotationMatrix:
        return RotationMatrix(self.T)

    def to_rpy(self) -> Tuple[Union[Symbol, float], Union[Symbol, float], Union[Symbol, float]]:
        """
        :return: roll, pitch, yaw
        """
        i = 0
        j = 1
        k = 2

        cy = sqrt(self[i, i] * self[i, i] + self[j, i] * self[j, i])
        if0 = cy - _EPS
        ax = if_greater_zero(if0,
                             atan2(self[k, j], self[k, k]),
                             atan2(-self[j, k], self[j, j]))
        ay = if_greater_zero(if0,
                             atan2(-self[k, i], cy),
                             atan2(-self[k, i], cy))
        az = if_greater_zero(if0,
                             atan2(self[j, i], self[i, i]),
                             0)
        return ax, ay, az

    def to_quaternion(self) -> Quaternion:
        return Quaternion.from_rotation_matrix(self)

    def normalize(self):
        """Scales each of the axes to the length of one."""
        scale_v = 1.0
        self[:3, 0] = scale(self[:3, 0], scale_v)
        self[:3, 1] = scale(self[:3, 1], scale_v)
        self[:3, 2] = scale(self[:3, 2], scale_v)


class Point3(Expression):
    def __init__(self, x: Union[Symbol, float], y: Union[Symbol, float], z: Union[Symbol, float]):
        super().__init__([x, y, z, 1])

    @classmethod
    def from_matrix(cls, m: Union[Iterable[Union[Symbol, float]], Expression]) -> Point3:
        try:
            return cls(m[0], m[1], m[2])
        except Exception as e:
            pass

    @property
    def x(self):
        return self[0]

    @x.setter
    def x(self, value):
        self[0] = value

    @property
    def y(self):
        return self[1]

    @y.setter
    def y(self, value):
        self[1] = value

    @property
    def z(self):
        return self[2]

    @z.setter
    def z(self, value):
        self[2] = value

    def __add__(self, other: Union[Point3, Vector3]):
        result = super().__add__(other)
        return Point3.from_matrix(result)

    def __sub__(self, other: Union[Point3, Vector3]):
        result = super().__sub__(other)
        if isinstance(other, Point3):
            return Vector3.from_matrix(result)
        return Point3.from_matrix(result)

    def __mul__(self, other: Union[Symbol, float]):
        result = super().__mul__(other)
        return Point3.from_matrix(result)

    def __truediv__(self, other: Union[Symbol, float]):
        result = super().__truediv__(other)
        return Point3.from_matrix(result)


class Vector3(Expression):
    def __init__(self, x: Union[Symbol, float], y: Union[Symbol, float], z: Union[Symbol, float]):
        super().__init__([x, y, z, 0])

    @classmethod
    def from_matrix(cls, m: Union[Iterable[Union[Symbol, float]], Expression]) -> Vector3:
        return cls(m[0], m[1], m[2])

    @property
    def x(self):
        return self[0]

    @x.setter
    def x(self, value):
        self[0] = value

    @property
    def y(self):
        return self[1]

    @y.setter
    def y(self, value):
        self[1] = value

    @property
    def z(self):
        return self[2]

    @z.setter
    def z(self, value):
        self[2] = value

    @overload
    def dot(self, other: Vector3) -> Symbol:
        ...

    @overload
    def dot(self, other: Point3) -> Symbol:
        ...

    def dot(self, other):
        if isinstance(other, Vector3) or isinstance(other, Point3):
            return Expression(dot(self[:3].T, other[:3])[0])
        return dot(self, other)

    def __add__(self, other: Union[Point3, Vector3]):
        result = super().__add__(other)
        if isinstance(other, Point3):
            return Point3.from_matrix(result)
        return Vector3.from_matrix(result)

    def __sub__(self, other: Vector3):
        result = super().__sub__(other)
        return Vector3.from_matrix(result)

    def __mul__(self, other: Union[Symbol, float]):
        result = super().__mul__(other)
        return Vector3.from_matrix(result)

    def __truediv__(self, other: Union[Symbol, float]):
        result = super().__truediv__(other)
        return Vector3.from_matrix(result)

    def cross(self, v: Vector3) -> Vector3:
        result = ca.cross(self.s[:3], v.s[:3])
        return Vector3.from_matrix(result)

    def norm(self) -> Symbol:
        return ca.norm_2(self)

    def scale(self, a: Union[Symbol, float]) -> Vector3:
        return save_division(self, self.norm()) * a


class Quaternion(Expression):
    def __init__(self, x: Union[Symbol, float], y: Union[Symbol, float],
                 z: Union[Symbol, float], w: Union[Symbol, float]):
        super().__init__([x, y, z, w])

    @property
    def x(self):
        return self[0]

    @x.setter
    def x(self, value):
        self[0] = value

    @property
    def y(self):
        return self[1]

    @y.setter
    def y(self, value):
        self[1] = value

    @property
    def z(self):
        return self[2]

    @z.setter
    def z(self, value):
        self[2] = value

    @property
    def w(self):
        return self[3]

    @w.setter
    def w(self, value):
        self[3] = value

    @classmethod
    def from_matrix(cls, m: Expression) -> Quaternion:
        return cls(m[0], m[1], m[2], m[3])

    @classmethod
    def from_axis_angle(cls, axis: Vector3, angle: Union[Symbol, float]) -> Quaternion:
        half_angle = angle / 2
        return cls(axis[0] * sin(half_angle),
                   axis[1] * sin(half_angle),
                   axis[2] * sin(half_angle),
                   cos(half_angle))

    @classmethod
    def from_rpy(cls, roll: Union[Expression, float], pitch: Union[Expression, float], yaw: Union[Expression, float]) \
            -> Quaternion:
        roll = Expression(roll).s
        pitch = Expression(pitch).s
        yaw = Expression(yaw).s
        roll_half = roll / 2.0
        pitch_half = pitch / 2.0
        yaw_half = yaw / 2.0

        c_roll = cos(roll_half)
        s_roll = sin(roll_half)
        c_pitch = cos(pitch_half)
        s_pitch = sin(pitch_half)
        c_yaw = cos(yaw_half)
        s_yaw = sin(yaw_half)

        cc = c_roll * c_yaw
        cs = c_roll * s_yaw
        sc = s_roll * c_yaw
        ss = s_roll * s_yaw

        x = c_pitch * sc - s_pitch * cs
        y = c_pitch * ss + s_pitch * cc
        z = c_pitch * cs - s_pitch * sc
        w = c_pitch * cc + s_pitch * ss

        return cls(x, y, z, w)

    @classmethod
    def from_rotation_matrix(cls, r: RotationMatrix) -> Quaternion:
        q = cls(0, 0, 0, 0)
        t = trace(r)

        if0 = t - r[3, 3]

        if1 = r[1, 1] - r[0, 0]

        m_i_i = if_greater_zero(if1, r[1, 1], r[0, 0])
        m_i_j = if_greater_zero(if1, r[1, 2], r[0, 1])
        m_i_k = if_greater_zero(if1, r[1, 0], r[0, 2])

        m_j_i = if_greater_zero(if1, r[2, 1], r[1, 0])
        m_j_j = if_greater_zero(if1, r[2, 2], r[1, 1])
        m_j_k = if_greater_zero(if1, r[2, 0], r[1, 2])

        m_k_i = if_greater_zero(if1, r[0, 1], r[2, 0])
        m_k_j = if_greater_zero(if1, r[0, 2], r[2, 1])
        m_k_k = if_greater_zero(if1, r[0, 0], r[2, 2])

        if2 = r[2, 2] - m_i_i

        m_i_i = if_greater_zero(if2, r[2, 2], m_i_i)
        m_i_j = if_greater_zero(if2, r[2, 0], m_i_j)
        m_i_k = if_greater_zero(if2, r[2, 1], m_i_k)

        m_j_i = if_greater_zero(if2, r[0, 2], m_j_i)
        m_j_j = if_greater_zero(if2, r[0, 0], m_j_j)
        m_j_k = if_greater_zero(if2, r[0, 1], m_j_k)

        m_k_i = if_greater_zero(if2, r[1, 2], m_k_i)
        m_k_j = if_greater_zero(if2, r[1, 0], m_k_j)
        m_k_k = if_greater_zero(if2, r[1, 1], m_k_k)

        t = if_greater_zero(if0, t, m_i_i - (m_j_j + m_k_k) + r[3, 3])
        q[0] = if_greater_zero(if0, r[2, 1] - r[1, 2],
                               if_greater_zero(if2, m_i_j + m_j_i,
                                               if_greater_zero(if1, m_k_i + m_i_k, t)))
        q[1] = if_greater_zero(if0, r[0, 2] - r[2, 0],
                               if_greater_zero(if2, m_k_i + m_i_k,
                                               if_greater_zero(if1, t, m_i_j + m_j_i)))
        q[2] = if_greater_zero(if0, r[1, 0] - r[0, 1],
                               if_greater_zero(if2, t, if_greater_zero(if1, m_i_j + m_j_i,
                                                                       m_k_i + m_i_k)))
        q[3] = if_greater_zero(if0, t, m_k_j - m_j_k)

        q *= 0.5 / sqrt(t * r[3, 3])
        return cls.from_matrix(q)

    def conjugate(self) -> Quaternion:
        return Quaternion(-self[0], -self[1], -self[2], self[3])

    def multiply(self, q: Quaternion) -> Quaternion:
        return Quaternion(x=self.x * q.w + self.y * q.z - self.z * q.y + self.w * q.x,
                          y=-self.x * q.z + self.y * q.w + self.z * q.x + self.w * q.y,
                          z=self.x * q.y - self.y * q.x + self.z * q.w + self.w * q.z,
                          w=-self.x * q.x - self.y * q.y - self.z * q.z + self.w * q.w)

    def diff(self, q: Quaternion) -> Quaternion:
        """
        :return: quaternion p, such that self*p=q
        """
        return self.conjugate().multiply(q)

    def norm(self) -> Expression:
        return norm(self)

    def normalize(self):
        norm = self.norm()
        self.x /= norm
        self.y /= norm
        self.z /= norm
        self.w /= norm

    def to_axis_angle(self) -> Tuple[Vector3, Expression]:
        self.normalize()
        w2 = sqrt(1 - self.w ** 2)
        m = if_eq_zero(w2, 1, w2)  # avoid /0
        angle = if_eq_zero(w2, 0, (2 * acos(limit(self.w, -1, 1))))
        x = if_eq_zero(w2, 0, self.x / m)
        y = if_eq_zero(w2, 0, self.y / m)
        z = if_eq_zero(w2, 1, self.z / m)
        return Vector3(x, y, z), angle

    def to_rotation_matrix(self):
        return RotationMatrix.from_quaternion(self)

    def to_rpy(self):
        return self.to_rotation_matrix().to_rpy()


def var(variables_names: str) -> List[Symbol]:
    """
    :param variables_names: e.g. 'a b c'
    :return: e.g. [Symbol('a'), Symbol('b'), Symbol('c')]
    """
    symbols = []
    for v in variables_names.split(' '):
        symbols.append(Symbol(v))
    return symbols


def diag(args: Union[List[Union[Symbol, float]], Expression]) -> Expression:
    return Expression(ca.diag(Expression(args).s))


# def Symbol(data: str) -> ca.SX:
#     if isinstance(data, str):
#         return ca.SX.sym(data)
#     return ca.SX(data)


def jacobian(expressions: Union[Expression, List[Expression]],
             symbols: Union[Expression, List[Symbol]],
             order: float = 1) -> Expression:
    expressions = Expression(expressions)
    if order == 1:
        return Expression(ca.jacobian(expressions.s, Expression(symbols).s))
    elif order == 2:
        j = jacobian(expressions, symbols, order=1)
        for i, symbol in enumerate(symbols):
            j[:, i] = jacobian(j[:, i], symbol)
        return j
    else:
        raise NotImplementedError('jacobian only supports order 1 and 2')


def equivalent(expression1: Union[Symbol, Expression], expression2: Union[Symbol, Expression]) -> bool:
    expression1 = Expression(expression1).s
    expression2 = Expression(expression2).s
    return ca.is_equal(ca.simplify(expression1), ca.simplify(expression2), 1)


def free_symbols(expression: Union[Symbol, Expression]) -> List[ca.SX]:
    expression = Expression(expression).s
    return ca.symvar(expression)


def is_matrix(expression: Union[Symbol, Expression]) -> bool:
    return hasattr(expression, 'shape') and expression.shape[0] * expression.shape[1] > 1


def is_symbol(expression: Union[Symbol, Expression]) -> bool:
    return expression.shape[0] * expression.shape[1] == 1


def create_symbols(names: List[str]) -> List[Symbol]:
    return [Symbol(x) for x in names]


def compile_and_execute(f: Callable[[Any], Expression], params: Union[List[Union[float, np.ndarray]], np.ndarray]) \
        -> Union[float, np.ndarray]:
    input = []
    symbol_params = []
    symbol_params2 = []

    for i, param in enumerate(params):
        if isinstance(param, list):
            param = np.array(param)
        if isinstance(param, np.ndarray):
            symbol_param = ca.SX.sym('m', *param.shape)
            if len(param.shape) == 2:
                l = param.shape[0] * param.shape[1]
            else:
                l = param.shape[0]

            input.append(param.reshape((l, 1)))
            symbol_params.append(symbol_param)
            asdf = symbol_param.T.reshape((l, 1))
            symbol_params2.extend(asdf[k] for k in range(l))
        else:
            input.append(np.array([param], ndmin=2))
            symbol_param = ca.SX.sym('s')
            symbol_params.append(symbol_param)
            symbol_params2.append(symbol_param)
    symbol_params = [Expression(x) for x in symbol_params]
    symbol_params2 = [Expression(x) for x in symbol_params2]
    expr = f(*symbol_params)
    assert isinstance(expr, Expression)
    fast_f = expr.compile(symbol_params2)
    input = np.concatenate(input).T[0]
    result = fast_f.call2(input)
    if result.shape[0] * result.shape[1] == 1:
        return result[0][0]
    elif result.shape[1] == 1:
        return result.T[0]
    elif result.shape[0] == 1:
        return result[0]
    else:
        return result


def ros_msg_to_matrix(msg: rospy.Message) -> Expression:
    return Expression(msg_to_homogeneous_matrix(msg))


def matrix_to_list(m: Union[Expression, Iterable[Symbol]]) -> List[Union[Symbol, float]]:
    try:
        len(m)
        return m
    except:
        return [m[i] for i in range(m.shape[0])]


def zeros(x: int, y: int) -> Expression:
    return Expression(ca.SX.zeros(x, y))


def ones(x: int, y: int) -> Expression:
    return Expression(ca.SX.ones(x, y))


def abs(x: Union[Symbol, float]) -> Expression:
    x = Expression(x).s
    return Expression(ca.fabs(x))


def max(x: Union[Symbol, float], y: Union[Symbol, float]) -> Expression:
    x = Expression(x).s
    y = Expression(y).s
    return Expression(ca.fmax(x, y))


def min(x: Union[Symbol, float], y: Union[Symbol, float]) -> Expression:
    x = Expression(x).s
    y = Expression(y).s
    return Expression(ca.fmin(x, y))


def limit(x: Union[Symbol, float],
          lower_limit: Union[Symbol, float],
          upper_limit: Union[Symbol, float]) -> Expression:
    if isinstance(x, Symbol):
        x = x.s
    if isinstance(lower_limit, Symbol):
        x = lower_limit.s
    if isinstance(upper_limit, Symbol):
        x = upper_limit.s
    return Expression(max(lower_limit, min(upper_limit, x)))


def if_else(condition: Union[Symbol, float], if_result: Union[Symbol, float],
            else_result: Union[Symbol, float]) -> Expression:
    condition = Expression(condition).s
    if_result = Expression(if_result).s
    else_result = Expression(else_result).s
    return Expression(ca.if_else(condition, if_result, else_result))


def logic_and(*args: List[Union[Symbol, float]]) -> Union[Symbol, float]:
    assert len(args) >= 2, 'and must be called with at least 2 arguments'
    if len(args) == 2:
        return Expression(ca.logic_and(args[0], args[1]))
    else:
        return Expression(ca.logic_and(args[0], logic_and(*args[1:])))


def logic_or(*args: List[Union[Symbol, float]]) -> Union[Symbol, float]:
    assert len(args) >= 2, 'and must be called with at least 2 arguments'
    if len(args) == 2:
        return ca.logic_or(args[0], args[1])
    else:
        return ca.logic_or(args[0], logic_and(*args[1:]))


def if_greater(a: Union[Symbol, float], b: Union[Symbol, float], if_result: Union[Symbol, float],
               else_result: Union[Symbol, float]) -> Expression:
    a = Expression(a).s
    b = Expression(b).s
    return if_else(ca.gt(a, b), if_result, else_result)


def if_less(a: Union[Symbol, float], b: Union[Symbol, float], if_result: Union[Symbol, float],
            else_result: Union[Symbol, float]) -> Expression:
    a = Expression(a).s
    b = Expression(b).s
    return if_else(ca.lt(a, b), if_result, else_result)


def if_greater_zero(condition: Union[Symbol, float],
                    if_result: Union[Symbol, float],
                    else_result: Union[Symbol, float]) -> Expression:
    """
    :return: if_result if condition > 0 else else_result
    """
    condition = Expression(condition).s
    if_result = Expression(if_result).s
    else_result = Expression(else_result).s
    _condition = sign(condition)  # 1 or -1
    _if = max(0, _condition) * if_result  # 0 or if_result
    _else = -min(0, _condition) * else_result  # 0 or else_result
    return Expression(_if + _else + (1 - abs(_condition)) * else_result)  # if_result or else_result


def if_greater_eq_zero(condition: Union[Symbol, float], if_result: Union[Symbol, float],
                       else_result: Union[Symbol, float]) -> Expression:
    """
    :return: if_result if condition >= 0 else else_result
    """
    return if_greater_eq(condition, 0, if_result, else_result)


def if_greater_eq(a: Union[Symbol, float], b: Union[Symbol, float], if_result: Union[Symbol, float],
                  else_result: Union[Symbol, float]) -> Expression:
    """
    :return: if_result if a >= b else else_result
    """
    a = Expression(a).s
    b = Expression(b).s
    return if_else(ca.ge(a, b), if_result, else_result)


def if_less_eq(a: Union[Symbol, float], b: Union[Symbol, float], if_result: Union[Symbol, float],
               else_result: Union[Symbol, float]) -> Expression:
    """
    :return: if_result if a <= b else else_result
    """
    return if_greater_eq(b, a, if_result, else_result)


def if_eq_zero(condition: Union[Symbol, float],
               if_result: Union[Symbol, float],
               else_result: Union[Symbol, float]) -> Expression:
    """
    :return: if_result if condition == 0 else else_result
    """
    return Expression(if_else(condition, else_result, if_result))


def if_eq(a: Union[Symbol, float],
          b: Union[Symbol, float],
          if_result: Union[Symbol, float],
          else_result: Union[Symbol, float]) -> Expression:
    a = Expression(a).s
    b = Expression(b).s
    return if_else(ca.eq(a, b), if_result, else_result)


def if_eq_cases(a: Union[Symbol, float],
                b_result_cases: Union[Iterable[Tuple[Union[Symbol, float], Union[Symbol, float]]], Expression],
                else_result: Union[Symbol, float]) -> Expression:
    a = Expression(a).s
    b_result_cases = Expression(b_result_cases).s
    result = Expression(else_result).s
    for i in range(b_result_cases.shape[0]):
        b = b_result_cases[i, 0]
        b_result = b_result_cases[i, 1]
        result = if_eq(a, b, b_result, result)
    return result


def if_less_eq_cases(a: Union[Symbol, float],
                     b_result_cases: Iterable[Tuple[Union[Symbol, float], Union[Symbol, float]]],
                     else_result: Union[Symbol, float]) -> Expression:
    """
    This only works if b_result_cases is sorted in ascending order.
    """
    a = Expression(a).s
    result = Expression(else_result).s
    b_result_cases = Expression(b_result_cases).s
    for i in reversed(range(b_result_cases.shape[0] - 1)):
        b = b_result_cases[i, 0]
        b_result = b_result_cases[i, 1]
        result = if_less_eq(a, b, b_result, result)
    return result


def safe_compiled_function(f, file_name):
    if not os.path.exists(os.path.dirname(file_name)):
        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(file_name, 'w') as file:
        pickle.dump(f, file)
        logging.loginfo('saved {}'.format(file_name))


def load_compiled_function(file_name):
    if os.path.isfile(file_name):
        try:
            with open(file_name, 'r') as file:
                fast_f = pickle.load(file)
                return fast_f
        except EOFError as e:
            os.remove(file_name)
            logging.logerr('{} deleted because it was corrupted'.format(file_name))


def cross(u: Union[Vector3, Expression], v: Union[Vector3, Expression]) -> Vector3:
    u = Vector3.from_matrix(u)
    v = Vector3.from_matrix(v)
    return u.cross(v)


def norm(v: Expression) -> Expression:
    v = Expression(v).s
    return Expression(ca.norm_2(v))


def scale(v: Expression, a: Union[Symbol, float]) -> Expression:
    return save_division(v, norm(v)) * a


def dot(*matrices: Expression) -> Expression:
    matrices = [Expression(x).s for x in matrices]
    result = ca.mtimes(matrices)
    return Expression(result)


def eye(size: int) -> Union[Symbol, float]:
    return ca.SX.eye(size)


def kron(m1: Expression, m2: Expression) -> Expression:
    return ca.kron(m1, m2)


def trace(matrix: Expression) -> Expression:
    matrix = Expression(matrix).s
    s = 0
    for i in range(matrix.shape[0]):
        s += matrix[i, i]
    return Expression(s)


def rotation_distance(a_R_b: Expression, a_R_c: Expression) -> Expression:
    """
    :param a_R_b: 4x4 or 3x3 Matrix
    :param a_R_c: 4x4 or 3x3 Matrix
    :return: angle of axis angle representation of b_R_c
    """
    a_R_b = Expression(a_R_b).s
    a_R_c = Expression(a_R_c).s
    difference = dot(a_R_b.T, a_R_c)
    # return axis_angle_from_matrix(difference)[1]
    angle = (trace(difference[:3, :3]) - 1) / 2
    angle = min(angle, 1)
    angle = max(angle, -1)
    return acos(angle)


def vstack(list_of_matrices: List[Expression]) -> Expression:
    return ca.vertcat(*list_of_matrices)


def hstack(list_of_matrices: List[Expression]) -> Expression:
    return ca.horzcat(*list_of_matrices)


def normalize_axis_angle(axis: Expression, angle: Union[Symbol, float]) -> Tuple[Expression, Union[Symbol, float]]:
    axis = if_less(angle, 0, -axis, axis)
    angle = abs(angle)
    return axis, angle


def axis_angle_from_rpy(roll: Union[Symbol, float], pitch: Union[Symbol, float], yaw: Union[Symbol, float]) \
        -> Tuple[Vector3, Expression]:
    return Quaternion.from_rpy(roll, pitch, yaw).to_axis_angle()


_EPS = np.finfo(float).eps * 4.0


def cosine_distance(v0: Expression, v1: Expression) -> Union[Symbol, float]:
    """
    cosine distance ranging from 0 to 2
    :param v0: nx1 Matrix
    :param v1: nx1 Matrix
    """
    return 1 - ((dot(v0.T, v1))[0] / (norm(v0) * norm(v1)))


def euclidean_distance(v1: Expression, v2: Expression) -> Union[Symbol, float]:
    """
    :param v1: nx1 Matrix
    :param v2: nx1 Matrix
    """
    return norm(v1 - v2)


def fmod(a: Union[Symbol, float], b: Union[Symbol, float]) -> Expression:
    a = Expression(a).s
    b = Expression(b).s
    return Expression(ca.fmod(a, b))


def normalize_angle_positive(angle: Union[Symbol, float]) -> Expression:
    """
    Normalizes the angle to be 0 to 2*pi
    It takes and returns radians.
    """
    return fmod(fmod(angle, 2.0 * pi) + 2.0 * pi, 2.0 * pi)


def normalize_angle(angle: Union[Symbol, float]) -> Expression:
    """
    Normalizes the angle to be -pi to +pi
    It takes and returns radians.
    """
    a = normalize_angle_positive(angle)
    return if_greater(a, pi, a - 2.0 * pi, a)


def shortest_angular_distance(from_angle: Union[Symbol, float], to_angle: Union[Symbol, float]) -> Expression:
    """
    Given 2 angles, this returns the shortest angular
    difference.  The inputs and outputs are of course radians.

    The result would always be -pi <= result <= pi. Adding the result
    to "from" will always get you an equivalent angle to "to".
    """
    return normalize_angle(to_angle - from_angle)


def quaternion_slerp(q1: Expression, q2: Expression, t: Union[Symbol, float]) -> Expression:
    """
    spherical linear interpolation that takes into account that q == -q
    :param q1: 4x1 Matrix
    :param q2: 4x1 Matrix
    :param t: float, 0-1
    :return: 4x1 Matrix; Return spherical linear interpolation between two quaternions.
    """
    q1 = Expression(q1).s
    q2 = Expression(q2).s
    t = Expression(t).s
    cos_half_theta = dot(q1.T, q2)

    if0 = -cos_half_theta
    q2 = if_greater_zero(if0, -q2, q2)
    cos_half_theta = if_greater_zero(if0, -cos_half_theta, cos_half_theta)

    if1 = abs(cos_half_theta) - 1.0

    # enforce acos(x) with -1 < x < 1
    cos_half_theta = min(1, cos_half_theta)
    cos_half_theta = max(-1, cos_half_theta)

    half_theta = acos(cos_half_theta)

    sin_half_theta = sqrt(1.0 - cos_half_theta * cos_half_theta)
    if2 = 0.001 - abs(sin_half_theta)

    ratio_a = save_division(sin((1.0 - t) * half_theta), sin_half_theta)
    ratio_b = save_division(sin(t * half_theta), sin_half_theta)
    return if_greater_eq_zero(if1,
                              Expression(q1),
                              if_greater_zero(if2,
                                              0.5 * q1 + 0.5 * q2,
                                              ratio_a * q1 + ratio_b * q2))


def slerp(v1: Expression, v2: Expression, t: float) -> Expression:
    """
    spherical linear interpolation
    :param v1: any vector
    :param v2: vector of same length as v1
    :param t: value between 0 and 1. 0 is v1 and 1 is v2
    """
    angle = save_acos(dot(v1.T, v2)[0])
    angle2 = if_eq(angle, 0, 1, angle)
    return if_eq(angle, 0,
                 v1,
                 (sin((1 - t) * angle2) / sin(angle2)) * v1 + (sin(t * angle2) / sin(angle2)) * v2)


def to_numpy(matrix: Expression) -> np.ndarray:
    return np.array(matrix.tolist()).astype(float).reshape(matrix.shape)


@overload
def save_division(nominator: Symbol, denominator: Union[Symbol, float], if_nan: Union[Symbol, float] = 0) -> Symbol: ...


@overload
def save_division(nominator: Vector3, denominator: Union[Symbol, float], if_nan: Union[Symbol, float] = 0) \
        -> Vector3: ...


@overload
def save_division(nominator: Point3, denominator: Union[Symbol, float], if_nan: Union[Symbol, float] = 0) \
        -> Point3: ...


def save_division(nominator: Union[Symbol, float],
                  denominator: Union[Symbol, float],
                  if_nan: Union[Symbol, float] = 0) -> Expression:
    save_denominator = if_eq_zero(denominator, 1, denominator)
    return nominator * if_eq_zero(denominator, if_nan, 1. / save_denominator)


def save_acos(angle: Union[Symbol, float]) -> Union[Symbol, float]:
    angle = limit(angle, -1, 1)
    return acos(angle)


def entrywise_product(matrix1: Expression, matrix2: Expression) -> Expression:
    assert matrix1.shape == matrix2.shape
    result = zeros(*matrix1.shape)
    for i in range(matrix1.shape[0]):
        for j in range(matrix1.shape[1]):
            result[i, j] = matrix1[i, j] * matrix2[i, j]
    return result


def floor(x: Union[Symbol, float]) -> Expression:
    x = Expression(x).s
    return Expression(ca.floor(x))


def ceil(x: Union[Symbol, float]) -> Expression:
    x = Expression(x).s
    return Expression(ca.ceil(x))


def round_up(x: Union[Symbol, float], decimal_places: Union[Symbol, float]) -> Union[Symbol, float]:
    f = 10 ** (decimal_places)
    return ceil(x * f) / f


def round_down(x: Union[Symbol, float], decimal_places: Union[Symbol, float]) -> Union[Symbol, float]:
    f = 10 ** (decimal_places)
    return floor(x * f) / f


def sum(matrix: Expression) -> Expression:
    """
    the equivalent to np.sum(matrix)
    """
    matrix = Expression(matrix).s
    return Expression(ca.sum1(ca.sum2(matrix)))


def sum_row(matrix: Expression) -> Expression:
    """
    the equivalent to np.sum(matrix, axis=0)
    """
    matrix = Expression(matrix).s
    return Expression(ca.sum1(matrix))


def sum_column(matrix: Expression) -> Expression:
    """
    the equivalent to np.sum(matrix, axis=1)
    """
    matrix = Expression(matrix).s
    return Expression(ca.sum2(matrix))


def distance_point_to_line_segment(point: Point3, line_start: Point3, line_end: Point3) \
        -> Tuple[Expression, Point3]:
    """
    :param point: current position of an object (i. e.) gripper tip
    :param line_start: start of the approached line
    :param line_end: end of the approached line
    :return: distance to line, the nearest point on the line
    """
    line_vec = line_end - line_start
    pnt_vec = point - line_start
    line_len = norm(line_vec)
    line_unitvec = line_vec / line_len
    pnt_vec_scaled = pnt_vec / line_len
    t = dot(line_unitvec.T, pnt_vec_scaled)[0]
    t = min(max(t, 0.0), 1.0)
    nearest = line_vec * t
    dist = norm(nearest - pnt_vec)
    nearest = nearest + line_start
    return dist, Point3.from_matrix(nearest)


def angle_between_vector(v1: Expression, v2: Expression) -> Union[Symbol, float]:
    v1 = v1[:3]
    v2 = v2[:3]
    return acos(dot(v1.T, v2) / (norm(v1) * norm(v2)))


def velocity_limit_from_position_limit(acceleration_limit: Union[Symbol, float],
                                       position_limit: Union[Symbol, float],
                                       current_position: Union[Symbol, float],
                                       step_size: Union[Symbol, float],
                                       eps: float = 1e-5) -> Expression:
    """
    Computes the velocity limit given a distance to the position limits, an acceleration limit and a step size
    :param acceleration_limit:
    :param step_size:
    :param eps: 
    :return: 
    """
    acceleration_limit = Expression(acceleration_limit).s
    position_limit = Expression(position_limit).s
    current_position = Expression(current_position).s
    step_size = Expression(step_size).s
    eps = Expression(eps).s
    distance_to_position_limit = position_limit - current_position
    acceleration_limit *= step_size
    distance_to_position_limit /= step_size
    m = 1 / acceleration_limit
    acceleration_limit *= m
    distance_to_position_limit *= m
    sign_ = sign(distance_to_position_limit)
    error = abs(distance_to_position_limit)
    # reverse gausssche summenformel to compute n from sum
    n = sqrt(2 * error + (1 / 4)) - 1 / 2
    # round up if very close to the ceiling to avoid precision errors
    n = if_less(1 - (n - floor(n)), eps, ceil(n), floor(n))
    error_rounded = (n ** 2 + n) / 2
    rest = error - error_rounded
    rest = rest / (n + 1)
    velocity_limit = n + rest
    velocity_limit *= sign_
    velocity_limit /= m
    return Expression(velocity_limit)


# def position_with_max_velocity(velocity_limit: expr_symbol, jerk_limit: expr_symbol) -> expr_symbol:
#     t = np.sqrt(np.abs(velocity_limit / jerk_limit))
#     return -t * velocity_limit


# def t_til_pos2(position_error, jerk_limit):
#     return (position_error / (2 * jerk_limit)) ** (1 / 3)


# def position_till_b(jerk_limit, t):
#     return (1 / 6) * jerk_limit * t ** 3


# def position_till_a(jerk_limit, t, t_offset, velocity_limit):
#     return (1 / 6) * jerk_limit * t ** 3 - 0.5 * jerk_limit * t_offset * t ** 2 + 0.5 * jerk_limit * t_offset ** 2 * t + velocity_limit * t


# def velocity(velocity_limit, jerk_limit, t):
#     t_b = np.sqrt(np.abs(velocity_limit / jerk_limit))
#     t_a = t_b * 2
#     if t < t_b:
#         return velocity_limit + 0.5 * jerk_limit * t ** 2
#     if t < t_a:
#         t -= t_a
#         return -0.5 * jerk_limit * t ** 2
#     return velocity_limit


# def position(jerk_limit, t, velocity_limit):
#     t_b = np.sqrt(np.abs(velocity_limit / jerk_limit))
#     t_a = t_b * 2
#     if t < t_b:
#         return (1 / 6) * jerk_limit * t ** 3 + velocity_limit * t - velocity_limit * t_b
#     if t < t_a:
#         t -= t_a
#         return -(1 / 6) * jerk_limit * t ** 3
#     return velocity_limit * t


# def compute_t_from_position(jerk_limit, position_error, velocity_limit):
#     t_b = np.sqrt(np.abs(velocity_limit / jerk_limit))
#     a = position_with_max_velocity(velocity_limit, jerk_limit)
#     b = -(1 / 6) * jerk_limit * (-t_b) ** 3
#     t_a = t_b * 2
#     if position_error < b:
#         asdf = (-(6 * position_error) / jerk_limit)
#         return np.sign(asdf) * np.abs(asdf) ** (1 / 3) + t_a
#     if position_error < a:
#         return np.real(-1.44224957030741 * (-0.5 - 0.866025403784439j) * \
#                        (((-t_b * velocity_limit - position_error) ** 2 / jerk_limit ** 2 + (
#                                8 / 9) * velocity_limit ** 3 / jerk_limit ** 3) ** (0.5 + 0j) + (1 / 6) *
#                         (-6.0 * t_b * velocity_limit - 6.0 * position_error) / jerk_limit) ** (1 / 3) \
#                        + 1.38672254870127 * velocity_limit * (-0.5 + 0.866025403784439j) / \
#                        (jerk_limit * (((-t_b * velocity_limit - position_error) ** 2 / jerk_limit ** 2 + (
#                                8 / 9) * velocity_limit ** 3 / jerk_limit ** 3) ** (0.5 + 0j)
#                                       + (1 / 6) * (
#                                               -6.0 * t_b * velocity_limit - 6.0 * position_error) / jerk_limit) ** (
#                                 1 / 3)))
#     return 0


# def jerk_limits_from_everything(position_limit, velocity_limit, jerk_limit, current_position, current_velocity,
#                                 current_acceleration, t, step_size, eps=1e-5):
#     """
#     Computes the velocity limit given a distance to the position limits, an acceleration limit and a step size
#     :param acceleration_limit:
#     :param distance_to_position_limit:
#     :param step_size:
#     :param eps:
#     :return:
#     """
#     # p(t) describes slowdown with max vel/jerk down to 0
#     # 1. get t from p(t)=position_limit - current_position
#     # 2. plug t into v(t) to get vel limit
#
#     a = position_with_max_velocity(velocity_limit, jerk_limit)
#     t_b = t_til_pos2(a, jerk_limit)
#     t_a = t_b * 2


def to_str(expression: Union[Symbol, float]) -> str:
    """
    Turns expression into a more or less readable string.
    """
    s = str(expression)
    parts = s.split(', ')
    result = parts[-1]
    for x in reversed(parts[:-1]):
        index, sub = x.split('=')
        if index not in result:
            raise Exception('fuck')
        result = result.replace(index, sub)
    return result
    pass


def total_derivative(expr: Union[Symbol, Expression],
                     symbols: Union[Expression, Iterable[Symbol]],
                     symbols_dot: Union[Expression, Iterable[Symbol]]) \
        -> Union[Symbol, Expression]:
    expr_jacobian = jacobian(expr, symbols)
    last_velocities = Expression(symbols_dot)
    velocity = dot(expr_jacobian, last_velocities)
    if velocity.shape[0] * velocity.shape[0] == 1:
        return velocity[0]
    else:
        return velocity


def quaternion_multiply(q1: Quaternion, q2: Quaternion) -> Quaternion:
    q1 = Quaternion.from_matrix(q1)
    q2 = Quaternion.from_matrix(q2)
    return q1.multiply(q2)


def quaternion_conjugate(q: Quaternion) -> Quaternion:
    q1 = Quaternion.from_matrix(q)
    return q1.conjugate()


def quaternion_diff(q1: Quaternion, q2: Quaternion) -> Quaternion:
    q1 = Quaternion.from_matrix(q1)
    q2 = Quaternion.from_matrix(q2)
    return q1.diff(q2)


def sign(x: Union[Symbol, float]) -> Expression:
    x = Expression(x).s
    return Expression(ca.sign(x))


def cos(x: Union[Symbol, float]) -> Expression:
    x = Expression(x).s
    return Expression(ca.cos(x))


def sin(x: Union[Symbol, float]) -> Expression:
    x = Expression(x).s
    return Expression(ca.sin(x))


def sqrt(x: Union[Symbol, float]) -> Expression:
    x = Expression(x).s
    return Expression(ca.sqrt(x))


def acos(x: Union[Symbol, float]) -> Expression:
    x = Expression(x).s
    return Expression(ca.acos(x))


def atan2(x: Union[Symbol, float], y: Union[Symbol, float]) -> Expression:
    x = Expression(x).s
    y = Expression(y).s
    return Expression(ca.atan2(x, y))
