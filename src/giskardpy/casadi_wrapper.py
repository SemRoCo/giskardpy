from __future__ import annotations

import builtins
from copy import copy
from typing import Union, List
import math
import casadi as ca  # type: ignore
import numpy as np
import geometry_msgs.msg as geometry_msgs
import rospy
from scipy import sparse as sp
from giskardpy.my_types import PrefixName, Derivatives
from giskardpy.utils import logging

builtin_max = builtins.max
builtin_min = builtins.min
builtin_abs = builtins.abs

_EPS = np.finfo(float).eps * 4.0
pi = ca.pi


class StackedCompiledFunction:
    def __init__(self, expressions, parameters=None, additional_views=None):
        combined_expression = vstack(expressions)
        self.compiled_f = combined_expression.compile(parameters=parameters)
        slices = []
        start = 0
        for expression in expressions[:-1]:
            end = start + expression.shape[0]
            slices.append(end)
            start = end
        self.split_out_view = np.split(self.compiled_f.out, slices)
        if additional_views is not None:
            for expression_slice in additional_views:
                self.split_out_view.append(self.compiled_f.out[expression_slice])

    @profile
    def fast_call(self, filtered_args):
        self.compiled_f.fast_call(filtered_args)
        return self.split_out_view


class CompiledFunction:
    def __init__(self, expression, parameters=None, sparse=False):
        self.sparse = sparse
        if len(expression) == 0:
            self.sparse = False
        if parameters is None:
            parameters = expression.free_symbols()

        self.str_params = [str(x) for x in parameters]
        if len(parameters) > 0:
            parameters = [Expression(parameters).s]

        if self.sparse:
            expression.s = ca.sparsify(expression.s)
            try:
                self.compiled_f = ca.Function('f', parameters, [expression.s])
            except Exception:
                self.compiled_f = ca.Function('f', parameters, expression.s)
            self.buf, self.f_eval = self.compiled_f.buffer()
            self.csc_indices, self.csc_indptr = expression.s.sparsity().get_ccs()
            self.out = sp.csc_matrix((np.zeros(expression.s.nnz()), self.csc_indptr, self.csc_indices))
            self.buf.set_res(0, memoryview(self.out.data))
        else:
            try:
                self.compiled_f = ca.Function('f', parameters, [ca.densify(expression.s)])
            except Exception:
                self.compiled_f = ca.Function('f', parameters, ca.densify(expression.s))
            self.buf, self.f_eval = self.compiled_f.buffer()
            if expression.shape[1] == 1:
                shape = expression.shape[0]
            else:
                shape = expression.shape
            self.out = np.zeros(shape, order='F')
            self.buf.set_res(0, memoryview(self.out))
        if len(self.str_params) == 0:
            self.f_eval()
            if self.sparse:
                result = self.out.toarray()
            else:
                result = self.out
            self.__call__ = lambda **kwargs: result
            self.fast_call = lambda filtered_args: result

    def __call__(self, **kwargs):
        filtered_args = [kwargs[k] for k in self.str_params]
        filtered_args = np.array(filtered_args, dtype=float)
        return self.fast_call(filtered_args)

    @profile
    def fast_call(self, filtered_args):
        """
        :param filtered_args: parameter values in the same order as in self.str_params
        """
        self.buf.set_arg(0, memoryview(filtered_args))
        self.f_eval()
        return self.out


def _operation_type_error(arg1, operation, arg2):
    return TypeError(f'unsupported operand type(s) for {operation}: \'{arg1.__class__.__name__}\' '
                     f'and \'{arg2.__class__.__name__}\'')


class Symbol_:
    s: ca.SX

    def __str__(self):
        return str(self.s)

    def pretty_str(self):
        return to_str(self)

    def __repr__(self):
        return repr(self.s)

    def __hash__(self):
        return self.s.__hash__()

    def __getitem__(self, item):
        if isinstance(item, np.ndarray) and item.dtype == bool:
            item = (np.where(item)[0], slice(None, None))
        return Expression(self.s[item])

    def __setitem__(self, key, value):
        try:
            value = value.s
        except AttributeError:
            pass
        self.s[key] = value

    @property
    def shape(self):
        return self.s.shape

    def __len__(self):
        return self.shape[0]

    def free_symbols(self):
        return free_symbols(self.s)

    def evaluate(self):
        if self.shape[0] == self.shape[1] == 0:
            return np.eye(0)
        elif self.s.shape[0] * self.s.shape[1] <= 1:
            return float(ca.evalf(self.s))
        else:
            return np.array(ca.evalf(self.s))

    def compile(self, parameters=None, sparse=False):
        return CompiledFunction(self, parameters, sparse)


class Symbol(Symbol_):
    def __init__(self, name: str):
        self.s: ca.SX = ca.SX.sym(name)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Expression(self.s.__add__(other))
        if isinstance(other, Symbol_):
            sum_ = self.s.__add__(other.s)
            if isinstance(other, (Symbol, Expression)):
                return Expression(sum_)
            elif isinstance(other, Vector3):
                return Vector3(sum_)
            elif isinstance(other, Point3):
                return Point3(sum_)
        raise _operation_type_error(self, '+', other)

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return Expression(self.s.__radd__(other))
        raise _operation_type_error(other, '+', self)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return Expression(self.s.__sub__(other))
        if isinstance(other, Symbol_):
            result = self.s.__sub__(other.s)
            if isinstance(other, (Symbol, Expression)):
                return Expression(result)
            elif isinstance(other, Vector3):
                return Vector3(result)
            elif isinstance(other, Point3):
                return Point3(result)
        raise _operation_type_error(self, '-', other)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Expression(self.s.__rsub__(other))
        raise _operation_type_error(other, '-', self)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Expression(self.s.__mul__(other))
        if isinstance(other, Symbol_):
            result = self.s.__mul__(other.s)
            if isinstance(other, (Symbol, Expression)):
                return Expression(result)
            elif isinstance(other, Vector3):
                return Vector3(result)
            elif isinstance(other, Point3):
                return Point3(result)
        raise _operation_type_error(self, '*', other)

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Expression(self.s.__rmul__(other))
        raise _operation_type_error(other, '*', self)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Expression(self.s.__truediv__(other))
        if isinstance(other, Symbol_):
            result = self.s.__truediv__(other.s)
            if isinstance(other, (Symbol, Expression)):
                return Expression(result)
            elif isinstance(other, Vector3):
                return Vector3(result)
            elif isinstance(other, Point3):
                return Point3(result)
        raise _operation_type_error(self, '/', other)

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return Expression(self.s.__rtruediv__(other))
        raise _operation_type_error(other, '/', self)

    def __floordiv__(self, other):
        return floor(self / other)

    def __mod__(self, other):
        return fmod(self, other)

    def __divmod__(self, other):
        return self // other, self % other

    def __rfloordiv__(self, other):
        return floor(other / self)

    def __rmod__(self, other):
        return fmod(other, self)

    def __rdivmod__(self, other):
        return other // self, other % self

    def __lt__(self, other):
        if isinstance(other, Symbol_):
            other = other.s
        return Expression(self.s.__lt__(other))

    def __le__(self, other):
        if isinstance(other, Symbol_):
            other = other.s
        return Expression(self.s.__le__(other))

    def __gt__(self, other):
        if isinstance(other, Symbol_):
            other = other.s
        return Expression(self.s.__gt__(other))

    def __ge__(self, other):
        if isinstance(other, Symbol_):
            other = other.s
        return Expression(self.s.__ge__(other))

    def __eq__(self, other):
        if isinstance(other, Symbol_):
            other = other.s
        return Expression(self.s.__eq__(other))

    def __ne__(self, other):
        if isinstance(other, Symbol_):
            other = other.s
        return Expression(self.s.__ne__(other))

    def __neg__(self):
        return Expression(self.s.__neg__())

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            return Expression(self.s.__pow__(other))
        if isinstance(other, Symbol_):
            result = self.s.__pow__(other.s)
            if isinstance(other, (Symbol, Expression)):
                return Expression(result)
            elif isinstance(other, Vector3):
                return Vector3(result)
            elif isinstance(other, Point3):
                return Point3(result)
        raise _operation_type_error(self, '**', other)

    def __rpow__(self, other):
        if isinstance(other, (int, float)):
            return Expression(self.s.__rpow__(other))
        raise _operation_type_error(other, '**', self)

    def __hash__(self):
        return self.s.__hash__()


class Expression(Symbol_):
    @profile
    def __init__(self, data=None):
        if data is None:
            data = []
        if isinstance(data, ca.SX):
            self.s = data
        elif isinstance(data, Symbol_):
            self.s = copy(data.s)
        elif isinstance(data, (int, float, np.ndarray)):
            self.s = ca.SX(data)
        else:
            x = len(data)
            if x == 0:
                self.s = ca.SX()
                return
            if isinstance(data[0], list) or isinstance(data[0], tuple) or isinstance(data[0], np.ndarray):
                y = len(data[0])
            else:
                y = 1
            self.s = ca.SX(x, y)
            for i in range(self.shape[0]):
                if y > 1:
                    for j in range(self.shape[1]):
                        self[i, j] = data[i][j]
                else:
                    if isinstance(data[i], Symbol):
                        self[i] = data[i].s
                    else:
                        self[i] = data[i]

    def remove(self, rows, columns):
        self.s.remove(rows, columns)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Expression(self.s.__add__(other))
        if isinstance(other, Point3):
            return Point3(self.s.__add__(other.s))
        if isinstance(other, Vector3):
            return Vector3(self.s.__add__(other.s))
        if isinstance(other, (Expression, Symbol)):
            return Expression(self.s.__add__(other.s))
        raise _operation_type_error(self, '+', other)

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return Expression(self.s.__radd__(other))
        raise _operation_type_error(other, '+', self)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return Expression(self.s.__sub__(other))
        if isinstance(other, Point3):
            return Point3(self.s.__sub__(other.s))
        if isinstance(other, Vector3):
            return Vector3(self.s.__sub__(other.s))
        if isinstance(other, (Expression, Symbol)):
            return Expression(self.s.__sub__(other.s))
        raise _operation_type_error(self, '-', other)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Expression(self.s.__rsub__(other))
        raise _operation_type_error(other, '-', self)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Expression(self.s.__truediv__(other))
        if isinstance(other, Point3):
            return Point3(self.s.__truediv__(other.s))
        if isinstance(other, Vector3):
            return Vector3(self.s.__truediv__(other.s))
        if isinstance(other, (Expression, Symbol)):
            return Expression(self.s.__truediv__(other.s))
        raise _operation_type_error(self, '/', other)

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return Expression(self.s.__rtruediv__(other))
        raise _operation_type_error(other, '/', self)

    def __floordiv__(self, other):
        return floor(self / other)

    def __mod__(self, other):
        return fmod(self, other)

    def __divmod__(self, other):
        return self // other, self % other

    def __rfloordiv__(self, other):
        return floor(other / self)

    def __rmod__(self, other):
        return fmod(other, self)

    def __rdivmod__(self, other):
        return other // self, other % self

    def __abs__(self):
        return abs(self)

    def __floor__(self):
        return floor(self)

    def __ceil__(self):
        return ceil(self)

    def __ge__(self, other):
        return greater_equal(self, other)

    def __gt__(self, other):
        return greater(self, other)

    def __le__(self, other):
        return less_equal(self, other)

    def __lt__(self, other):
        return less(self, other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Expression(self.s.__mul__(other))
        if isinstance(other, Point3):
            return Point3(self.s.__mul__(other.s))
        if isinstance(other, Vector3):
            return Vector3(self.s.__mul__(other.s))
        if isinstance(other, (Expression, Symbol)):
            return Expression(self.s.__mul__(other.s))
        raise _operation_type_error(self, '*', other)

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Expression(self.s.__rmul__(other))
        raise _operation_type_error(other, '*', self)

    def __neg__(self):
        return Expression(self.s.__neg__())

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            return Expression(self.s.__pow__(other))
        if isinstance(other, (Expression, Symbol)):
            return Expression(self.s.__pow__(other.s))
        if isinstance(other, (Vector3)):
            return Vector3(self.s.__pow__(other.s))
        if isinstance(other, (Point3)):
            return Point3(self.s.__pow__(other.s))
        raise _operation_type_error(self, '**', other)

    def __rpow__(self, other):
        if isinstance(other, (int, float)):
            return Expression(self.s.__rpow__(other))
        raise _operation_type_error(other, '**', self)

    def __eq__(self, other):
        if isinstance(other, Symbol_):
            other = other.s
        return Expression(self.s.__eq__(other))

    def __ne__(self, other):
        if isinstance(other, Symbol_):
            other = other.s
        return Expression(self.s.__ne__(other))

    def dot(self, other):
        if isinstance(other, Expression):
            if self.shape[1] == 1 and other.shape[1] == 1:
                return Expression(ca.mtimes(self.T.s, other.s))
            return Expression(ca.mtimes(self.s, other.s))
        raise _operation_type_error(self, 'dot', other)

    @property
    def T(self):
        return Expression(self.s.T)

    def reshape(self, new_shape):
        return Expression(self.s.reshape(new_shape))


class TransMatrix(Symbol_):
    @profile
    def __init__(self, data=None, sanity_check=True):
        try:
            self.reference_frame = data.reference_frame
        except AttributeError:
            self.reference_frame = None
        try:
            self.child_frame = data.child_frame
        except AttributeError:
            self.child_frame = None
        if isinstance(data, (geometry_msgs.Pose, geometry_msgs.PoseStamped)):
            if isinstance(data, geometry_msgs.PoseStamped):
                self.reference_frame = data.header.frame_id
                data = data.pose
            if isinstance(data, geometry_msgs.Pose):
                r = RotationMatrix(data.orientation)
                self.s = r.s
                self.s[0, 3] = data.position.x
                self.s[1, 3] = data.position.y
                self.s[2, 3] = data.position.z
                return
        elif data is None:
            self.s = ca.SX.eye(4)
            return
        elif isinstance(data, ca.SX):
            self.s = data
        elif isinstance(data, (Expression, RotationMatrix, TransMatrix)):
            self.s = copy(data.s)
        else:
            self.s = copy(Expression(data).s)
        if sanity_check:
            if self.shape[0] != 4 or self.shape[1] != 4:
                raise ValueError(f'{self.__class__.__name__} can only be initialized with 4x4 shaped data.')
            self[3, 0] = 0
            self[3, 1] = 0
            self[3, 2] = 0
            self[3, 3] = 1

    @classmethod
    def from_point_rotation_matrix(cls, point=None, rotation_matrix=None):
        if rotation_matrix is None:
            a_T_b = cls()
        else:
            a_T_b = cls(rotation_matrix, sanity_check=False)
        if point is not None:
            a_T_b[0, 3] = point.x
            a_T_b[1, 3] = point.y
            a_T_b[2, 3] = point.z
        return a_T_b

    @profile
    def dot(self, other):
        if isinstance(other, (Vector3, Point3, RotationMatrix, TransMatrix)):
            result = ca.mtimes(self.s, other.s)
            if isinstance(other, Vector3):
                result = Vector3(result)
                result.reference_frame = self.reference_frame
                return result
            if isinstance(other, Point3):
                result = Point3(result)
                result.reference_frame = self.reference_frame
                return result
            if isinstance(other, RotationMatrix):
                result = RotationMatrix(result, sanity_check=False)
                result.reference_frame = self.reference_frame
                return result
            if isinstance(other, TransMatrix):
                result = TransMatrix(result, sanity_check=False)
                result.reference_frame = self.reference_frame
                result.child_frame = other.child_frame
                return result
        raise _operation_type_error(self, 'dot', other)

    @profile
    def inverse(self):
        inv = TransMatrix()
        inv[:3, :3] = self[:3, :3].T
        inv[:3, 3] = dot(-inv[:3, :3], self[:3, 3])
        return inv

    @classmethod
    @profile
    def from_xyz_rpy(cls, x=None, y=None, z=None, roll=None, pitch=None, yaw=None):
        p = Point3.from_xyz(x, y, z)
        r = RotationMatrix.from_rpy(roll, pitch, yaw)
        return cls.from_point_rotation_matrix(p, r)

    def to_position(self):
        result = Point3(self[:4, 3:])
        result.reference_frame = self.reference_frame
        return result

    def to_translation(self):
        """
        :return: sets the rotation part of a frame to identity
        """
        r = TransMatrix()
        r[0, 3] = self[0, 3]
        r[1, 3] = self[1, 3]
        r[2, 3] = self[2, 3]
        r.reference_frame = self.reference_frame
        return TransMatrix(r)

    def to_rotation(self):
        return RotationMatrix(self)


class RotationMatrix(Symbol_):
    @profile
    def __init__(self, data=None, sanity_check=True):
        if hasattr(data, 'reference_frame'):
            self.reference_frame = data.reference_frame
        else:
            self.reference_frame = None
        if isinstance(data, ca.SX):
            self.s = data
        elif isinstance(data, (geometry_msgs.Quaternion, geometry_msgs.QuaternionStamped)):
            if isinstance(data, geometry_msgs.QuaternionStamped):
                self.reference_frame = data.header.frame_id
                data = data.quaternion
            if isinstance(data, geometry_msgs.Quaternion):
                self.s = self.__quaternion_to_rotation_matrix(Quaternion(data)).s
        elif isinstance(data, Quaternion):
            self.s = self.__quaternion_to_rotation_matrix(data).s
        elif data is None:
            self.s = ca.SX.eye(4)
            return
        else:
            self.s = Expression(data).s
        if sanity_check:
            if self.shape[0] != 4 or self.shape[1] != 4:
                raise ValueError(f'{self.__class__.__name__} can only be initialized with 4x4 shaped data, '
                                 f'you have{self.shape}.')
            self[0, 3] = 0
            self[1, 3] = 0
            self[2, 3] = 0
            self[3, 0] = 0
            self[3, 1] = 0
            self[3, 2] = 0
            self[3, 3] = 1

    @classmethod
    @profile
    def from_axis_angle(cls, axis, angle):
        """
        Conversion of unit axis and angle to 4x4 rotation matrix according to:
        https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/index.htm
        """
        # use casadi to prevent a bunch of Expression.__init__ calls
        axis = axis.s
        try:
            angle = angle.s
        except AttributeError:
            pass
        ct = ca.cos(angle)
        st = ca.sin(angle)
        vt = 1 - ct
        m_vt = axis * vt
        m_st = axis * st
        m_vt_0_ax = (m_vt[0] * axis)[1:]
        m_vt_1_2 = m_vt[1] * axis[2]
        s = ca.SX.eye(4)
        ct__m_vt__axis = ct + m_vt * axis
        s[0, 0] = ct__m_vt__axis[0]
        s[0, 1] = -m_st[2] + m_vt_0_ax[0]
        s[0, 2] = m_st[1] + m_vt_0_ax[1]
        s[1, 0] = m_st[2] + m_vt_0_ax[0]
        s[1, 1] = ct__m_vt__axis[1]
        s[1, 2] = -m_st[0] + m_vt_1_2
        s[2, 0] = -m_st[1] + m_vt_0_ax[1]
        s[2, 1] = m_st[0] + m_vt_1_2
        s[2, 2] = ct__m_vt__axis[2]
        return cls(s, sanity_check=False)

    @classmethod
    def __quaternion_to_rotation_matrix(cls, q):
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
    def from_quaternion(cls, q):
        return cls.__quaternion_to_rotation_matrix(q)

    def dot(self, other):
        if isinstance(other, (Vector3, Point3, RotationMatrix, TransMatrix)):
            result = ca.mtimes(self.s, other.s)
            if isinstance(other, Vector3):
                result = Vector3(result)
            elif isinstance(other, Point3):
                result = Point3(result)
            elif isinstance(other, RotationMatrix):
                result = RotationMatrix(result, sanity_check=False)
            elif isinstance(other, TransMatrix):
                result = TransMatrix(result, sanity_check=False)
            result.reference_frame = self.reference_frame
            return result
        raise _operation_type_error(self, 'dot', other)

    def to_axis_angle(self):
        return self.to_quaternion().to_axis_angle()

    def to_angle(self, hint):
        """
        :param hint: A function whose sign of the result will be used to determine if angle should be positive or
                        negative
        :return:
        """
        axis, angle = self.to_axis_angle()
        return normalize_angle(if_greater_zero(hint(axis),
                                               if_result=angle,
                                               else_result=-angle))

    @classmethod
    def from_vectors(cls, x=None, y=None, z=None):
        if x is not None:
            x.scale(1)
        if y is not None:
            y.scale(1)
        if z is not None:
            z.scale(1)
        if x is not None and y is not None and z is None:
            z = cross(x, y)
            z.scale(1)
        elif x is not None and y is None and z is not None:
            y = cross(z, x)
            y.scale(1)
        elif x is None and y is not None and z is not None:
            x = cross(y, z)
            x.scale(1)
        # else:
        #     raise AttributeError(f'only one vector can be None')
        R = cls([[x[0], y[0], z[0], 0],
                 [x[1], y[1], z[1], 0],
                 [x[2], y[2], z[2], 0],
                 [0, 0, 0, 1]])
        R.normalize()
        return R

    @classmethod
    @profile
    def from_rpy(cls, roll=None, pitch=None, yaw=None):
        """
        Conversion of roll, pitch, yaw to 4x4 rotation matrix according to:
        https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/src/frames.cpp#L167
        """
        roll = 0 if roll is None else roll
        pitch = 0 if pitch is None else pitch
        yaw = 0 if yaw is None else yaw
        try:
            roll = roll.s
        except AttributeError:
            pass
        try:
            pitch = pitch.s
        except AttributeError:
            pass
        try:
            yaw = yaw.s
        except AttributeError:
            pass
        s = ca.SX.eye(4)

        s[0, 0] = ca.cos(yaw) * ca.cos(pitch)
        s[0, 1] = (ca.cos(yaw) * ca.sin(pitch) * ca.sin(roll)) - (ca.sin(yaw) * ca.cos(roll))
        s[0, 2] = (ca.sin(yaw) * ca.sin(roll)) + (ca.cos(yaw) * ca.sin(pitch) * ca.cos(roll))
        s[1, 0] = ca.sin(yaw) * ca.cos(pitch)
        s[1, 1] = (ca.cos(yaw) * ca.cos(roll)) + (ca.sin(yaw) * ca.sin(pitch) * ca.sin(roll))
        s[1, 2] = (ca.sin(yaw) * ca.sin(pitch) * ca.cos(roll)) - (ca.cos(yaw) * ca.sin(roll))
        s[2, 0] = -ca.sin(pitch)
        s[2, 1] = ca.cos(pitch) * ca.sin(roll)
        s[2, 2] = ca.cos(pitch) * ca.cos(roll)
        return cls(s, sanity_check=False)

    def inverse(self):
        return RotationMatrix(self.T)

    def to_rpy(self):
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

    def to_quaternion(self):
        return Quaternion.from_rotation_matrix(self)

    def normalize(self):
        """Scales each of the axes to the length of one."""
        scale_v = 1.0
        self[:3, 0] = scale(self[:3, 0], scale_v)
        self[:3, 1] = scale(self[:3, 1], scale_v)
        self[:3, 2] = scale(self[:3, 2], scale_v)

    @property
    def T(self):
        return self.s.T


class Point3(Symbol_):
    @profile
    def __init__(self, data=None):
        try:
            self.reference_frame = data.reference_frame
        except AttributeError:
            self.reference_frame = None
        if data is None:
            self.s = ca.SX([0, 0, 0, 1])
            return
        if isinstance(data, rospy.Message):
            if isinstance(data, geometry_msgs.PointStamped):
                self.reference_frame = data.header.frame_id
                data = data.point
            if isinstance(data, geometry_msgs.Vector3Stamped):
                self.reference_frame = data.header.frame_id
                data = data.vector
            if isinstance(data, (Point3, Vector3, geometry_msgs.Point, geometry_msgs.Vector3)):
                self.s = ca.SX([data.x, data.y, data.z, 1])
        elif isinstance(data, Symbol_):
            self.s = ca.SX([0, 0, 0, 1])
            self[0] = data.s[0]
            self[1] = data.s[1]
            self[2] = data.s[2]
        else:
            self.s = ca.SX([0, 0, 0, 1])
            self[0] = data[0]
            self[1] = data[1]
            self[2] = data[2]

    @classmethod
    def from_xyz(cls, x=None, y=None, z=None):
        x = 0 if x is None else x
        y = 0 if y is None else y
        z = 0 if z is None else z
        return cls((x, y, z))

    def norm(self):
        return norm(self)

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

    def __add__(self, other):
        if isinstance(other, (int, float)):
            result = Point3(self.s.__add__(other))
        elif isinstance(other, (Vector3, Expression, Symbol)):
            result = Point3(self.s.__add__(other.s))
        else:
            raise _operation_type_error(self, '+', other)
        result.reference_frame = self.reference_frame
        return result

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            result = Point3(self.s.__add__(other))
        else:
            raise _operation_type_error(other, '+', self)
        result.reference_frame = self.reference_frame
        return result

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            result = Point3(self.s.__sub__(other))
        elif isinstance(other, Point3):
            result = Vector3(self.s.__sub__(other.s))
        elif isinstance(other, (Symbol, Expression, Vector3)):
            result = Point3(self.s.__sub__(other.s))
        else:
            raise _operation_type_error(self, '-', other)
        result.reference_frame = self.reference_frame
        return result

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            result = Point3(self.s.__rsub__(other))
        else:
            raise _operation_type_error(other, '-', self)
        result.reference_frame = self.reference_frame
        return result

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            result = Point3(self.s.__mul__(other))
        elif isinstance(other, (Symbol, Expression)):
            result = Point3(self.s.__mul__(other.s))
        else:
            raise _operation_type_error(self, '*', other)
        result.reference_frame = self.reference_frame
        return result

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            result = Point3(self.s.__mul__(other))
        else:
            raise _operation_type_error(other, '*', self)
        result.reference_frame = self.reference_frame
        return result

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            result = Point3(self.s.__truediv__(other))
        elif isinstance(other, (Symbol, Expression)):
            result = Point3(self.s.__truediv__(other.s))
        else:
            raise _operation_type_error(self, '/', other)
        result.reference_frame = self.reference_frame
        return result

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            result = Point3(self.s.__rtruediv__(other))
        else:
            raise _operation_type_error(other, '/', self)
        result.reference_frame = self.reference_frame
        return result

    def __neg__(self) -> Point3:
        result = Point3(self.s.__neg__())
        result.reference_frame = self.reference_frame
        return result

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            result = Point3(self.s.__pow__(other))
        elif isinstance(other, (Symbol, Expression)):
            result = Point3(self.s.__pow__(other.s))
        else:
            raise _operation_type_error(self, '**', other)
        result.reference_frame = self.reference_frame
        return result

    def __rpow__(self, other):
        if isinstance(other, (int, float)):
            result = Point3(self.s.__rpow__(other))
        else:
            raise _operation_type_error(other, '**', self)
        result.reference_frame = self.reference_frame
        return result

    def dot(self, other):
        if isinstance(other, (Point3, Vector3)):
            return Expression(ca.mtimes(self[:3].T.s, other[:3].s))
        raise _operation_type_error(self, 'dot', other)


class Vector3(Symbol_):
    @profile
    def __init__(self, data=None):
        point = Point3(data)
        self.s = point.s
        self.reference_frame = point.reference_frame
        self.vis_frame = self.reference_frame
        self[3] = 0

    @classmethod
    def from_xyz(cls, x=None, y=None, z=None):
        x = 0 if x is None else x
        y = 0 if y is None else y
        z = 0 if z is None else z
        return cls((x, y, z))

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

    def __add__(self, other):
        if isinstance(other, (int, float)):
            result = Vector3(self.s.__add__(other))
        elif isinstance(other, Point3):
            result = Point3(self.s.__add__(other.s))
        elif isinstance(other, (Vector3, Expression, Symbol)):
            result = Vector3(self.s.__add__(other.s))
        else:
            raise _operation_type_error(self, '+', other)
        result.reference_frame = self.reference_frame
        return result

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            result = Vector3(self.s.__add__(other))
        else:
            raise _operation_type_error(other, '+', self)
        result.reference_frame = self.reference_frame
        return result

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            result = Vector3(self.s.__sub__(other))
        elif isinstance(other, Point3):
            result = Point3(self.s.__sub__(other.s))
        elif isinstance(other, (Symbol, Expression, Vector3)):
            result = Vector3(self.s.__sub__(other.s))
        else:
            raise _operation_type_error(self, '-', other)
        result.reference_frame = self.reference_frame
        return result

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            result = Vector3(self.s.__rsub__(other))
        else:
            raise _operation_type_error(other, '-', self)
        result.reference_frame = self.reference_frame
        return result

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            result = Vector3(self.s.__mul__(other))
        elif isinstance(other, (Symbol, Expression)):
            result = Vector3(self.s.__mul__(other.s))
        else:
            raise _operation_type_error(self, '*', other)
        result.reference_frame = self.reference_frame
        return result

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            result = Vector3(self.s.__mul__(other))
        else:
            raise _operation_type_error(other, '*', self)
        result.reference_frame = self.reference_frame
        return result

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            result = Vector3(self.s.__pow__(other))
        elif isinstance(other, (Symbol, Expression)):
            result = Vector3(self.s.__pow__(other.s))
        else:
            raise _operation_type_error(self, '**', other)
        result.reference_frame = self.reference_frame
        return result

    def __rpow__(self, other):
        if isinstance(other, (int, float)):
            result = Vector3(self.s.__rpow__(other))
        else:
            raise _operation_type_error(other, '**', self)
        result.reference_frame = self.reference_frame
        return result

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            result = Vector3(self.s.__truediv__(other))
        elif isinstance(other, (Symbol, Expression)):
            result = Vector3(self.s.__truediv__(other.s))
        else:
            raise _operation_type_error(self, '/', other)
        result.reference_frame = self.reference_frame
        return result

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            result = Vector3(self.s.__rtruediv__(other))
        else:
            raise _operation_type_error(other, '/', self)
        result.reference_frame = self.reference_frame
        return result

    def __neg__(self):
        result = Vector3(self.s.__neg__())
        result.reference_frame = self.reference_frame
        return result

    def dot(self, other):
        if isinstance(other, (Point3, Vector3)):
            return Expression(ca.mtimes(self[:3].T.s, other[:3].s))
        raise _operation_type_error(self, 'dot', other)

    def cross(self, other):
        result = ca.cross(self.s[:3], other.s[:3])
        result = Vector3(result)
        result.reference_frame = self.reference_frame
        return result

    def norm(self):
        return norm(self)

    def scale(self, a):
        self.s = (save_division(self, self.norm()) * a).s


class Quaternion(Symbol_):
    def __init__(self, data=None):
        if data is None:
            data = (0, 0, 0, 1)
        if isinstance(data, geometry_msgs.QuaternionStamped):
            data = data.quaternion
        if isinstance(data, (Point3, Vector3, geometry_msgs.Quaternion)):
            x, y, z, w = data.x, data.y, data.z, data.w
        else:
            x, y, z, w = data[0], data[1], data[2], data[3]
        self.s = ca.SX(4, 1)
        self[0] = x
        self[1] = y
        self[2] = z
        self[3] = w

    def __neg__(self):
        return Quaternion(self.s.__neg__())

    @classmethod
    def from_xyzw(cls, x, y, z, w):
        return cls((x, y, z, w))

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
    def from_axis_angle(cls, axis, angle):
        half_angle = angle / 2
        return cls((axis[0] * sin(half_angle),
                    axis[1] * sin(half_angle),
                    axis[2] * sin(half_angle),
                    cos(half_angle)))

    @classmethod
    def from_rpy(cls, roll, pitch, yaw):
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

        return cls((x, y, z, w))

    @classmethod
    def from_rotation_matrix(cls, r):
        q = Expression((0, 0, 0, 0))
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
        return cls(q)

    def conjugate(self):
        return Quaternion((-self[0], -self[1], -self[2], self[3]))

    def multiply(self, q):
        return Quaternion((self.x * q.w + self.y * q.z - self.z * q.y + self.w * q.x,
                           -self.x * q.z + self.y * q.w + self.z * q.x + self.w * q.y,
                           self.x * q.y - self.y * q.x + self.z * q.w + self.w * q.z,
                           -self.x * q.x - self.y * q.y - self.z * q.z + self.w * q.w))

    def diff(self, q):
        """
        :return: quaternion p, such that self*p=q
        """
        return self.conjugate().multiply(q)

    def norm(self):
        return norm(self)

    def normalize(self):
        norm_ = self.norm()
        self.x /= norm_
        self.y /= norm_
        self.z /= norm_
        self.w /= norm_

    def to_axis_angle(self):
        self.normalize()
        w2 = sqrt(1 - self.w ** 2)
        m = if_eq_zero(w2, 1, w2)  # avoid /0
        angle = if_eq_zero(w2, 0, (2 * acos(limit(self.w, -1, 1))))
        x = if_eq_zero(w2, 0, self.x / m)
        y = if_eq_zero(w2, 0, self.y / m)
        z = if_eq_zero(w2, 1, self.z / m)
        return Vector3((x, y, z)), angle

    def to_rotation_matrix(self):
        return RotationMatrix.from_quaternion(self)

    def to_rpy(self):
        return self.to_rotation_matrix().to_rpy()

    def dot(self, other):
        if isinstance(other, Quaternion):
            return Expression(ca.mtimes(self.s.T, other.s))
        raise _operation_type_error(self, 'dot', other)


all_expressions = Union[Symbol_, Symbol, Expression, Point3, Vector3, RotationMatrix, TransMatrix, Quaternion]
all_expressions_float = Union[Symbol, Expression, Point3, Vector3, RotationMatrix, TransMatrix, float, Quaternion]
symbol_expr_float = Union[Symbol, Expression, float]
symbol_expr = Union[Symbol, Expression]


def var(variables_names: str):
    """
    :param variables_names: e.g. 'a b c'
    :return: e.g. [Symbol('a'), Symbol('b'), Symbol('c')]
    """
    symbols = []
    for v in variables_names.split(' '):
        symbols.append(Symbol(v))
    return symbols


def diag(args):
    try:
        return Expression(ca.diag(args.s))
    except AttributeError:
        return Expression(ca.diag(Expression(args).s))


@profile
def jacobian(expressions, symbols):
    expressions = Expression(expressions)
    return Expression(ca.jacobian(expressions.s, Expression(symbols).s))


def jacobian_dot(expressions, symbols, symbols_dot):
    Jd = jacobian(expressions, symbols)
    for i in range(Jd.shape[0]):
        for j in range(Jd.shape[1]):
            Jd[i, j] = total_derivative(Jd[i, j], symbols, symbols_dot)
    return Jd


def jacobian_ddot(expressions, symbols, symbols_dot, symbols_ddot):
    symbols_ddot = Expression(symbols_ddot)
    Jdd = jacobian(expressions, symbols)
    for i in range(Jdd.shape[0]):
        for j in range(Jdd.shape[1]):
            Jdd[i, j] = total_derivative2(Jdd[i, j], symbols, symbols_dot, symbols_ddot)
    return Jdd


def equivalent(expression1, expression2):
    expression1 = Expression(expression1).s
    expression2 = Expression(expression2).s
    return ca.is_equal(ca.simplify(expression1), ca.simplify(expression2), 5)


def free_symbols(expression):
    expression = Expression(expression).s
    return ca.symvar(expression)


def create_symbols(names):
    return [Symbol(x) for x in names]


def compile_and_execute(f, params):
    input_ = []
    symbol_params = []
    symbol_params2 = []

    for i, param in enumerate(params):
        if isinstance(param, list):
            param = np.array(param)
        if isinstance(param, np.ndarray):
            symbol_param = ca.SX.sym('m', *param.shape)
            if len(param.shape) == 2:
                number_of_params = param.shape[0] * param.shape[1]
            else:
                number_of_params = param.shape[0]

            input_.append(param.reshape((number_of_params, 1)))
            symbol_params.append(symbol_param)
            asdf = symbol_param.T.reshape((number_of_params, 1))
            symbol_params2.extend(asdf[k] for k in range(number_of_params))
        else:
            input_.append(np.array([param], ndmin=2))
            symbol_param = ca.SX.sym('s')
            symbol_params.append(symbol_param)
            symbol_params2.append(symbol_param)
    symbol_params = [Expression(x) for x in symbol_params]
    symbol_params2 = [Expression(x) for x in symbol_params2]
    expr = f(*symbol_params)
    assert isinstance(expr, Symbol_)
    fast_f = expr.compile(symbol_params2)
    input_ = np.array(np.concatenate(input_).T[0], dtype=float)
    result = fast_f.fast_call(input_)
    if len(result.shape) == 1:
        if result.shape[0] == 1:
            return result[0]
        return result
    if result.shape[0] * result.shape[1] == 1:
        return result[0][0]
    elif result.shape[1] == 1:
        return result.T[0]
    elif result.shape[0] == 1:
        return result[0]
    else:
        return result


def zeros(x, y):
    return Expression(ca.SX.zeros(x, y))


def ones(x, y):
    return Expression(ca.SX.ones(x, y))


def abs(x):
    x = Expression(x).s
    result = ca.fabs(x)
    if isinstance(x, Point3):
        return Point3(result)
    elif isinstance(x, Vector3):
        return Vector3(result)
    return Expression(result)


def max(x, y=None):
    x = Expression(x).s
    y = Expression(y).s
    return Expression(ca.fmax(x, y))


def min(x, y=None):
    x = Expression(x).s
    y = Expression(y).s
    return Expression(ca.fmin(x, y))


def limit(x, lower_limit, upper_limit):
    return Expression(max(lower_limit, min(upper_limit, x)))


def if_else(condition, if_result, else_result):
    condition = Expression(condition).s
    if isinstance(if_result, float):
        if_result = Expression(if_result)
    if isinstance(else_result, float):
        else_result = Expression(else_result)
    if isinstance(if_result, (Point3, Vector3, TransMatrix, RotationMatrix, Quaternion)):
        assert type(if_result) == type(else_result), \
            f'if_else: result types are not equal {type(if_result)} != {type(else_result)}'
    return_type = type(if_result)
    if return_type in (int, float):
        return_type = Expression
    if return_type == Symbol:
        return_type = Expression
    if_result = Expression(if_result).s
    else_result = Expression(else_result).s
    return return_type(ca.if_else(condition, if_result, else_result))


def equal(x, y):
    if isinstance(x, Symbol_):
        x = x.s
    if isinstance(y, Symbol_):
        y = y.s
    return Expression(ca.eq(x, y))


def less_equal(x, y):
    if isinstance(x, Symbol_):
        x = x.s
    if isinstance(y, Symbol_):
        y = y.s
    return Expression(ca.le(x, y))


def greater_equal(x, y):
    if isinstance(x, Symbol_):
        x = x.s
    if isinstance(y, Symbol_):
        y = y.s
    return Expression(ca.ge(x, y))


def less(x, y):
    if isinstance(x, Symbol_):
        x = x.s
    if isinstance(y, Symbol_):
        y = y.s
    return Expression(ca.lt(x, y))


def greater(x, y, decimal_places=None):
    if decimal_places is not None:
        x = round_up(x, decimal_places)
        y = round_up(y, decimal_places)
    if isinstance(x, Symbol_):
        x = x.s
    if isinstance(y, Symbol_):
        y = y.s
    return Expression(ca.gt(x, y))


def logic_and(*args):
    assert len(args) >= 2, 'and must be called with at least 2 arguments'
    if len(args) == 2:
        return Expression(ca.logic_and(args[0].s, args[1].s))
    else:
        return Expression(ca.logic_and(args[0].s, logic_and(*args[1:]).s))


def logic_any(args):
    return Expression(ca.logic_any(args.s))


def logic_all(args):
    return Expression(ca.logic_all(args.s))


def logic_or(*args):
    assert len(args) >= 2, 'and must be called with at least 2 arguments'
    if len(args) == 2:
        return Expression(ca.logic_or(args[0].s, args[1].s))
    else:
        return Expression(ca.logic_or(args[0].s, logic_or(*args[1:]).s))


def logic_not(expr):
    return Expression(ca.logic_not(expr.s))


def if_greater(a, b, if_result, else_result):
    a = Expression(a).s
    b = Expression(b).s
    return if_else(ca.gt(a, b), if_result, else_result)


def if_less(a, b, if_result, else_result):
    a = Expression(a).s
    b = Expression(b).s
    return if_else(ca.lt(a, b), if_result, else_result)


def if_greater_zero(condition, if_result, else_result):
    """
    :return: if_result if condition > 0 else else_result
    """
    condition = Expression(condition).s
    return if_else(ca.gt(condition, 0), if_result, else_result)

    # _condition = sign(condition)  # 1 or -1
    # _if = max(0, _condition) * if_result  # 0 or if_result
    # _else = -min(0, _condition) * else_result  # 0 or else_result
    # return Expression(_if + _else + (1 - abs(_condition)) * else_result)  # if_result or else_result


def if_greater_eq_zero(condition, if_result, else_result):
    """
    :return: if_result if condition >= 0 else else_result
    """
    return if_greater_eq(condition, 0, if_result, else_result)


def if_greater_eq(a, b, if_result, else_result):
    """
    :return: if_result if a >= b else else_result
    """
    a = Expression(a).s
    b = Expression(b).s
    return if_else(ca.ge(a, b), if_result, else_result)


def if_less_eq(a, b, if_result, else_result):
    """
    :return: if_result if a <= b else else_result
    """
    return if_greater_eq(b, a, if_result, else_result)


def if_eq_zero(condition, if_result, else_result):
    """
    :return: if_result if condition == 0 else else_result
    """
    return if_else(condition, else_result, if_result)


def if_eq(a, b, if_result, else_result):
    a = Expression(a).s
    b = Expression(b).s
    return if_else(ca.eq(a, b), if_result, else_result)


@profile
def if_eq_cases(a, b_result_cases, else_result):
    a = _to_sx(a)
    else_result = _to_sx(else_result)
    result = _to_sx(else_result)
    for i in range(len(b_result_cases)):
        b = _to_sx(b_result_cases[i][0])
        b_result = _to_sx(b_result_cases[i][1])
        result = ca.if_else(ca.eq(a, b), b_result, result)
    return Expression(result)


@profile
def if_less_eq_cases(a, b_result_cases, else_result):
    """
    This only works if b_result_cases is sorted in ascending order.
    """
    a = _to_sx(a)
    result = _to_sx(else_result)
    for i in reversed(range(len(b_result_cases))):
        b = _to_sx(b_result_cases[i][0])
        b_result = _to_sx(b_result_cases[i][1])
        result = ca.if_else(ca.le(a, b), b_result, result)
    return Expression(result)


def _to_sx(thing):
    try:
        return thing.s
    except AttributeError:
        return thing


def cross(u, v):
    u = Vector3(u)
    v = Vector3(v)
    return u.cross(v)


def norm(v):
    if isinstance(v, (Point3, Vector3)):
        return Expression(ca.norm_2(v[:3].s))
    v = Expression(v).s
    return Expression(ca.norm_2(v))


def scale(v, a):
    return save_division(v, norm(v)) * a


def dot(e1, e2):
    try:
        return e1.dot(e2)
    except Exception as e:
        raise _operation_type_error(e1, 'dot', e2)


def eye(size):
    return Expression(ca.SX.eye(size))


def kron(m1, m2):
    m1 = Expression(m1).s
    m2 = Expression(m2).s
    return Expression(ca.kron(m1, m2))


def trace(matrix):
    matrix = Expression(matrix).s
    s = 0
    for i in range(matrix.shape[0]):
        s += matrix[i, i]
    return Expression(s)


# def rotation_distance(a_R_b, a_R_c):
#     """
#     :param a_R_b: 4x4 or 3x3 Matrix
#     :param a_R_c: 4x4 or 3x3 Matrix
#     :return: angle of axis angle representation of b_R_c
#     """
#     a_R_b = Expression(a_R_b).s
#     a_R_c = Expression(a_R_c).s
#     difference = dot(a_R_b.T, a_R_c)
#     # return axis_angle_from_matrix(difference)[1]
#     angle = (trace(difference[:3, :3]) - 1) / 2
#     angle = min(angle, 1)
#     angle = max(angle, -1)
#     return acos(angle)


def vstack(list_of_matrices):
    if len(list_of_matrices) == 0:
        return Expression()
    return Expression(ca.vertcat(*[x.s for x in list_of_matrices]))


def hstack(list_of_matrices):
    if len(list_of_matrices) == 0:
        return Expression()
    return Expression(ca.horzcat(*[x.s for x in list_of_matrices]))


def diag_stack(list_of_matrices):
    num_rows = int(math.fsum(e.shape[0] for e in list_of_matrices))
    num_columns = int(math.fsum(e.shape[1] for e in list_of_matrices))
    combined_matrix = zeros(num_rows, num_columns)
    row_counter = 0
    column_counter = 0
    for matrix in list_of_matrices:
        combined_matrix[row_counter:row_counter + matrix.shape[0],
        column_counter:column_counter + matrix.shape[1]] = matrix
        row_counter += matrix.shape[0]
        column_counter += matrix.shape[1]
    return combined_matrix


def normalize_axis_angle(axis, angle):
    # todo add test
    axis = if_less(angle, 0, -axis, axis)
    angle = abs(angle)
    return axis, angle


def axis_angle_from_rpy(roll, pitch, yaw):
    return Quaternion.from_rpy(roll, pitch, yaw).to_axis_angle()


def cosine_distance(v0, v1):
    """
    cosine distance ranging from 0 to 2
    :param v0: nx1 Matrix
    :param v1: nx1 Matrix
    """
    return 1 - ((dot(v0.T, v1))[0] / (norm(v0) * norm(v1)))


def euclidean_distance(v1, v2):
    """
    :param v1: nx1 Matrix
    :param v2: nx1 Matrix
    """
    return norm(v1 - v2)


def fmod(a, b):
    a = Expression(a).s
    b = Expression(b).s
    return Expression(ca.fmod(a, b))


def euclidean_division(nominator, denominator):
    pass


def normalize_angle_positive(angle):
    """
    Normalizes the angle to be 0 to 2*pi
    It takes and returns radians.
    """
    return fmod(fmod(angle, 2.0 * ca.pi) + 2.0 * ca.pi, 2.0 * ca.pi)


def normalize_angle(angle):
    """
    Normalizes the angle to be -pi to +pi
    It takes and returns radians.
    """
    a = normalize_angle_positive(angle)
    return if_greater(a, ca.pi, a - 2.0 * ca.pi, a)


@profile
def shortest_angular_distance(from_angle, to_angle):
    """
    Given 2 angles, this returns the shortest angular
    difference.  The inputs and outputs are of course radians.

    The result would always be -pi <= result <= pi. Adding the result
    to "from" will always get you an equivalent angle to "to".
    """
    return normalize_angle(to_angle - from_angle)


def quaternion_slerp(q1, q2, t):
    """
    spherical linear interpolation that takes into account that q == -q
    :param q1: 4x1 Matrix
    :param q2: 4x1 Matrix
    :param t: float, 0-1
    :return: 4x1 Matrix; Return spherical linear interpolation between two quaternions.
    """
    q1 = Expression(q1)
    q2 = Expression(q2)
    cos_half_theta = q1.dot(q2)

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
    return Quaternion(if_greater_eq_zero(if1,
                                         q1,
                                         if_greater_zero(if2,
                                                         0.5 * q1 + 0.5 * q2,
                                                         ratio_a * q1 + ratio_b * q2)))


def slerp(v1, v2, t):
    """
    spherical linear interpolation
    :param v1: any vector
    :param v2: vector of same length as v1
    :param t: value between 0 and 1. 0 is v1 and 1 is v2
    """
    angle = save_acos(dot(v1, v2))
    angle2 = if_eq(angle, 0, 1, angle)
    return if_eq(angle, 0,
                 v1,
                 (sin((1 - t) * angle2) / sin(angle2)) * v1 + (sin(t * angle2) / sin(angle2)) * v2)


def save_division(nominator, denominator, if_nan=None):
    if if_nan is None:
        if isinstance(nominator, Vector3):
            if_nan = Vector3()
        elif isinstance(nominator, Point3):
            if_nan = Vector3
        else:
            if_nan = 0
    save_denominator = if_eq_zero(denominator, 1, denominator)
    return nominator * if_eq_zero(denominator, if_nan, 1. / save_denominator)


def save_acos(angle):
    angle = limit(angle, -1, 1)
    return acos(angle)


def entrywise_product(matrix1, matrix2):
    assert matrix1.shape == matrix2.shape
    result = zeros(*matrix1.shape)
    for i in range(matrix1.shape[0]):
        for j in range(matrix1.shape[1]):
            result[i, j] = matrix1[i, j] * matrix2[i, j]
    return result


def floor(x):
    x = Expression(x).s
    return Expression(ca.floor(x))


def ceil(x):
    x = Expression(x).s
    return Expression(ca.ceil(x))


def round_up(x, decimal_places):
    f = 10 ** (decimal_places)
    return ceil(x * f) / f


def round_down(x, decimal_places):
    f = 10 ** (decimal_places)
    return floor(x * f) / f


def sum(matrix):
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


def sum_column(matrix):
    """
    the equivalent to np.sum(matrix, axis=1)
    """
    matrix = Expression(matrix).s
    return Expression(ca.sum2(matrix))


def distance_point_to_line_segment(point, line_start, line_end):
    """
    :param point: current position of an object (i. e.) gripper tip
    :param line_start: start of the approached line
    :param line_end: end of the approached line
    :return: distance to line, the nearest point on the line
    """
    point = Point3(point)
    line_start = Point3(line_start)
    line_end = Point3(line_end)
    line_vec = line_end - line_start
    pnt_vec = point - line_start
    line_len = norm(line_vec)
    line_unitvec = line_vec / line_len
    pnt_vec_scaled = pnt_vec / line_len
    t = line_unitvec.dot(pnt_vec_scaled)
    t = limit(t, lower_limit=0.0, upper_limit=1.0)
    nearest = line_vec * t
    dist = norm(nearest - pnt_vec)
    nearest = nearest + line_start
    return dist, Point3(nearest)


def angle_between_vector(v1, v2):
    v1 = v1[:3]
    v2 = v2[:3]
    return acos(dot(v1.T, v2) / (norm(v1) * norm(v2)))


def velocity_limit_from_position_limit(acceleration_limit,
                                       position_limit,
                                       current_position,
                                       step_size,
                                       eps=1e-5):
    """
    Computes the velocity limit given a distance to the position limits, an acceleration limit and a step size
    :param acceleration_limit:
    :param step_size:
    :param eps: 
    :return: 
    """
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


def to_str(expression):
    """
    Turns expression into a more or less readable string.
    """
    result_list = np.zeros(expression.shape).tolist()
    for x_index in range(expression.shape[0]):
        for y_index in range(expression.shape[1]):
            s = str(expression[x_index, y_index])
            parts = s.split(', ')
            result = parts[-1]
            for x in reversed(parts[:-1]):
                equal_position = len(x.split('=')[0])
                index = x[:equal_position]
                sub = x[equal_position + 1:]
                if index not in result:
                    raise Exception('fuck')
                result = result.replace(index, sub)
            result_list[x_index][y_index] = result
    return result_list


def total_derivative(expr,
                     symbols,
                     symbols_dot):
    symbols = Expression(symbols)
    symbols_dot = Expression(symbols_dot)
    return Expression(ca.jtimes(expr.s, symbols.s, symbols_dot.s))


def total_derivative2(expr, symbols, symbols_dot, symbols_ddot):
    symbols = Expression(symbols)
    symbols_dot = Expression(symbols_dot)
    symbols_ddot = Expression(symbols_ddot)
    v = []
    for i in range(len(symbols)):
        for j in range(len(symbols)):
            if i == j:
                v.append(symbols_ddot[i].s)
            else:
                v.append(symbols_dot[i].s * symbols_dot[j].s)
    v = Expression(v)
    H = Expression(ca.hessian(expr.s, symbols.s)[0])
    H = H.reshape((1, len(H) ** 2))
    return H.dot(v)


def quaternion_multiply(q1, q2):
    q1 = Quaternion(q1)
    q2 = Quaternion(q2)
    return q1.multiply(q2)


def quaternion_conjugate(q):
    q1 = Quaternion(q)
    return q1.conjugate()


def quaternion_diff(q1, q2):
    q1 = Quaternion(q1)
    q2 = Quaternion(q2)
    return q1.diff(q2)


def sign(x):
    x = Expression(x).s
    return Expression(ca.sign(x))


def cos(x):
    x = Expression(x).s
    return Expression(ca.cos(x))


def sin(x):
    x = Expression(x).s
    return Expression(ca.sin(x))


def sqrt(x):
    x = Expression(x).s
    return Expression(ca.sqrt(x))


def acos(x):
    x = Expression(x).s
    return Expression(ca.acos(x))


def atan2(x, y):
    x = Expression(x).s
    y = Expression(y).s
    return Expression(ca.atan2(x, y))


def solve_for(expression, target_value, start_value=0.0001, max_tries=10000, eps=1e-10, max_step=50):
    f_dx = jacobian(expression, expression.free_symbols()).compile()
    f = expression.compile()
    x = start_value
    for tries in range(max_tries):
        err = f.fast_call(np.array([x]))[0] - target_value
        if builtin_abs(err) < eps:
            return x
        slope = f_dx.fast_call(np.array([x]))[0]
        if slope == 0:
            if start_value > 0:
                slope = -0.001
            else:
                slope = 0.001
        x -= builtin_max(builtin_min(err / slope, max_step), -max_step)
    raise ValueError('no solution found')


def gauss(n):
    return (n ** 2 + n) / 2


def r_gauss(integral):
    return sqrt(2 * integral + (1 / 4)) - 1 / 2


def one_step_change(current_acceleration, jerk_limit, dt):
    return current_acceleration * dt + jerk_limit * dt ** 2


def desired_velocity(current_position, goal_position, dt, ph):
    e = goal_position - current_position
    a = e / (gauss(ph) * dt)
    # a = e / ((gauss(ph-1) + ph - 1)*dt)
    return a * ph
    # return a * (ph-2)


def vel_integral(vel_limit, jerk_limit, dt, ph):
    def f(vc, ac, jl, t, dt, ph):
        return vc + (t) * ac * dt + gauss(t) * jl * dt ** 2

    half1 = math.floor(ph / 2)
    x = f(0, 0, jerk_limit, half1, dt, ph)
    return x
