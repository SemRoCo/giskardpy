import os
import pickle
from warnings import warn

import symengine as se
from symengine import Matrix, Symbol, eye, sympify, diag, zeros, lambdify, Abs, Max, Min, sin, cos, tan, acos, asin, \
    atan, atan2, nan, sqrt, log, tanh, var, floor, Piecewise, sign
import numpy as np
from numpy import pi

from symengine.lib.symengine_wrapper import Lambdify

from giskardpy.exceptions import SymengineException
from giskardpy.utils import create_path

pathSeparator = '_'

# VERY_SMALL_NUMBER = 2.22507385851e-308
VERY_SMALL_NUMBER = 1e-100
SMALL_NUMBER = 1e-10


def diffable_abs(x):
    """
    :type x: Union[float, Symbol]
    :return: abs(x)
    :rtype: Union[float, Symbol]
    """
    return se.sqrt(x ** 2)


def diffable_sign(x):
    """
    !Returns shit if x is very close to but not equal to zero!
    if x > 0:
        return 1
    if x < 0:
        return -1
    if x == 0:
        return 0

    :type x: Union[float, Symbol]
    :return: sign(x)
    :rtype: Union[float, Symbol]
    """
    # return x/(-VERY_SMALL_NUMBER + diffable_abs(x))
    return (tanh(x * 1e105))


def diffable_heaviside(x):
    return 0.5 * (diffable_sign(x + VERY_SMALL_NUMBER) + 1)


def diffable_max_fast(x, y):
    """
    Can be compiled quickly.
    !gets very imprecise if inputs outside of [-1e7,1e7]!
    :type x: Union[float, Symbol]
    :type y: Union[float, Symbol]
    :return: max(x, y)
    :rtype: Union[float, Symbol]
    """
    return ((x + y) + diffable_abs(x - y)) / 2


def diffable_max(x, y):
    """
    !takes a long time to compile!
    :type x: Union[float, Symbol]
    :type y: Union[float, Symbol]
    :return: max(x, y)
    :rtype: Union[float, Symbol]
    """
    return diffable_if_greater_zero(x - y, x, y)


def diffable_min_fast(x, y):
    """
    Can be compiled quickly.
    !gets very imprecise if inputs outside of [-1e7,1e7]!
    :type x: Union[float, Symbol]
    :type y: Union[float, Symbol]
    :return: min(x, y)
    :rtype: Union[float, Symbol]
    """
    return ((x + y) - diffable_abs(x - y)) / 2


def diffable_min(x, y):
    """
    !takes a long time to compile!
    :type x: Union[float, Symbol]
    :type y: Union[float, Symbol]
    :return: min(x, y)
    :rtype: Union[float, Symbol]
    """
    return diffable_if_greater_zero(y - x, x, y)


def diffable_if_greater_zero(condition, if_result, else_result):
    """
    !takes a long time to compile!
    !Returns shit if condition is very close to but not equal to zero!
    :type condition: Union[float, Symbol]
    :type if_result: Union[float, Symbol]
    :type else_result: Union[float, Symbol]
    :return: if_result if condition > 0 else else_result
    :rtype: Union[float, Symbol]
    """
    # condition -= VERY_SMALL_NUMBER
    # _if = diffable_heaviside(condition) * if_result
    # _else = diffable_heaviside(-condition) * else_result
    # return _if + _else
    _condition = diffable_sign(condition - VERY_SMALL_NUMBER)  # 1 or -1
    _if = diffable_max_fast(0, _condition) * if_result  # 0 or if_result
    _else = -diffable_min_fast(0, _condition) * else_result  # 0 or else_result
    return _if + _else  # if_result or else_result


def diffable_if_greater_eq_zero(condition, if_result, else_result):
    """
    !takes a long time to compile!
    !Returns shit if condition is very close to but not equal to zero!
    :type condition: Union[float, Symbol]
    :type if_result: Union[float, Symbol]
    :type else_result: Union[float, Symbol]
    :return: if_result if condition >= 0 else else_result
    :rtype: Union[float, Symbol]
    """
    return diffable_if_greater_zero(-condition, else_result, if_result)


def diffable_if_eq_zero(condition, if_result, else_result):
    """
    A short expression which can be compiled quickly.
    !Returns shit if condition is very close to but not equal to zero!
    !Returns shit if if_result is outside of [-1e8,1e8]!
    :type condition: Union[float, Symbol]
    :type if_result: Union[float, Symbol]
    :type else_result: Union[float, Symbol]
    :return: if_result if condition == 0 else else_result
    :rtype: Union[float, Symbol]
    """
    condition = diffable_abs(diffable_sign(condition))
    return (1 - condition) * if_result + condition * else_result


def if_greater_zero(condition, if_result, else_result):
    """
    :type condition: Union[float, Symbol]
    :type if_result: Union[float, Symbol]
    :type else_result: Union[float, Symbol]
    :return: if_result if condition > 0 else else_result
    :rtype: Union[float, Symbol]
    """
    return Piecewise([if_result, condition > 0], [else_result, True])


def if_greater_eq_zero(condition, if_result, else_result):
    """
    !takes a long time to compile!
    !Returns shit if condition is very close to but not equal to zero!
    :type condition: Union[float, Symbol]
    :type if_result: Union[float, Symbol]
    :type else_result: Union[float, Symbol]
    :return: if_result if condition >= 0 else else_result
    :rtype: Union[float, Symbol]
    """
    return if_greater_zero(-condition, else_result, if_result)


def if_eq_zero(condition, if_result, else_result):
    """
    A short expression which can be compiled quickly.
    :type condition: Union[float, Symbol]
    :type if_result: Union[float, Symbol]
    :type else_result: Union[float, Symbol]
    :return: if_result if condition == 0 else else_result
    :rtype: Union[float, Symbol]
    """
    condition = se.Abs(sign(condition))
    return (1 - condition) * if_result + condition * else_result


def safe_compiled_function(f, file_name):
    create_path(file_name)
    with open(file_name, 'w') as file:
        pickle.dump(f, file)
        print(u'saved {}'.format(file_name))


def load_compiled_function(file_name):
    if os.path.isfile(file_name):
        try:
            with open(file_name, u'r') as file:
                fast_f = pickle.load(file)
                return fast_f
        except EOFError as e:
            os.remove(file_name)
            print(u'{} deleted because it was corrupted'.format(file_name))


class CompiledFunction(object):
    def __init__(self, str_params, fast_f, l, shape):
        self.str_params = str_params
        self.fast_f = fast_f
        self.l = l
        self.shape = shape

    def __call__(self, **kwargs):
        try:
            filtered_args = [kwargs[k] for k in self.str_params]
            out = np.empty(self.l)
            self.fast_f.unsafe_real(np.array(filtered_args, dtype=np.double), out)
            # TODO nan to num is kinda dangerous
            return np.nan_to_num(out).reshape(self.shape)
        except KeyError as e:
            msg = u'KeyError: {}\ntry deleting the data folder to trigger recompilation'.format(e.message)
            raise SymengineException(msg)


def speed_up(function, parameters, backend=u'llvm'):
    # TODO use save/load for all options
    str_params = [str(x) for x in parameters]
    if len(parameters) == 0:
        try:
            constant_result = np.array(function).astype(float).reshape(function.shape)
        except:
            return

        def f(**kwargs):
            return constant_result
    else:
        if backend == u'llvm':
            # try:
            fast_f = Lambdify(list(parameters), function, backend=backend, cse=True, real=True)
            # except RuntimeError as e:
            #     warn(u'WARNING RuntimeError: "{}" during lambdify with LLVM backend, fallback to numpy'.format(e),
            #          RuntimeWarning)
            #     backend = u'lambda'
        if backend == u'lambda':
            try:
                fast_f = Lambdify(list(parameters), function, backend=u'lambda', cse=True, real=True)
            except RuntimeError as e:
                warn(u'WARNING RuntimeError: "{}" during lambdify with lambda backend, no speedup possible'.format(e),
                     RuntimeWarning)
                backend = None

        if backend in [u'llvm', u'lambda']:
            f = CompiledFunction(str_params, fast_f, len(function), function.shape)
        elif backend is None:
            # @profile
            def f(**kwargs):
                filtered_kwargs = {str(k): kwargs[k] for k in str_params}
                return np.array(function.subs(filtered_kwargs).tolist(), dtype=float).reshape(function.shape)
        if backend == u'python':
            f = function

    return f


def cross(u, v):
    """
    :param u: 1d matrix
    :type u: Matrix
    :param v: 1d matrix
    :type v: Matrix
    :return: 1d Matrix. If u and v have length 4, it ignores the last entry and adds a zero to the result.
    :rtype: Matrix
    """
    if len(u) != len(v):
        raise ValueError('lengths {} and {} don\' align'.format(len(u), len(v)))
    if len(u) == 3:
        return se.Matrix([u[1] * v[2] - u[2] * v[1],
                          u[2] * v[0] - u[0] * v[2],
                          u[0] * v[1] - u[1] * v[0]])
    if len(u) == 4:
        return se.Matrix([u[1] * v[2] - u[2] * v[1],
                          u[2] * v[0] - u[0] * v[2],
                          u[0] * v[1] - u[1] * v[0],
                          0])


def vector3(x, y, z):
    """
    :param x: Union[float, Symbol]
    :param y: Union[float, Symbol]
    :param z: Union[float, Symbol]
    :rtype: Matrix
    """
    return se.Matrix([x, y, z, 0])


unitX = vector3(1, 0, 0)
unitY = vector3(0, 1, 0)
unitZ = vector3(0, 0, 1)


def point3(x, y, z):
    """
    :param x: Union[float, Symbol]
    :param y: Union[float, Symbol]
    :param z: Union[float, Symbol]
    :rtype: Matrix
    """
    return se.Matrix([x, y, z, 1])


def norm(v):
    """
    :type v: Matrix
    :return: |v|_2
    :rtype: Union[float, Symbol]
    """
    r = 0
    for x in v:
        r += x ** 2
    return se.sqrt(r)


def scale(v, a):
    """
    :type v: Matrix
    :type a: Union[float, Symbol]
    :rtype: Matrix
    """
    return v / norm(v) * a


def dot(a, b):
    """
    :type a: Matrix
    :type b: Matrix
    :rtype: Union[float, Symbol]
    """
    return (a.T * b)[0]


def translation3(x, y, z):
    """
    :type x: Union[float, Symbol]
    :type y: Union[float, Symbol]
    :type z: Union[float, Symbol]
    :return: 4x4 Matrix
        [[1,0,0,x],
         [0,1,0,y],
         [0,0,1,z],
         [0,0,0,1]]
    :rtype: Matrix
    """
    r = se.eye(4)
    r[0, 3] = x
    r[1, 3] = y
    r[2, 3] = z
    return r


def rotation_matrix_from_rpy(roll, pitch, yaw):
    """
    Conversion of roll, pitch, yaw to 4x4 rotation matrix according to:
    https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/src/frames.cpp#L167
    :type roll: Union[float, Symbol]
    :type pitch: Union[float, Symbol]
    :type yaw: Union[float, Symbol]
    :return: 4x4 Matrix
    :rtype: Matrix
    """
    # TODO don't split this into 3 matrices

    rx = se.Matrix([[1, 0, 0, 0],
                    [0, se.cos(roll), -se.sin(roll), 0],
                    [0, se.sin(roll), se.cos(roll), 0],
                    [0, 0, 0, 1]])
    ry = se.Matrix([[se.cos(pitch), 0, se.sin(pitch), 0],
                    [0, 1, 0, 0],
                    [-se.sin(pitch), 0, se.cos(pitch), 0],
                    [0, 0, 0, 1]])
    rz = se.Matrix([[se.cos(yaw), -se.sin(yaw), 0, 0],
                    [se.sin(yaw), se.cos(yaw), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    return (rz * ry * rx)


def rotation_matrix_from_axis_angle(axis, angle):
    """
    Conversion of unit axis and angle to 4x4 rotation matrix according to:
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/index.htm
    :param axis: 3x1 Matrix
    :type axis: Matrix
    :type angle: Union[float, Symbol]
    :return: 4x4 Matrix
    :rtype: Matrix
    """
    ct = se.cos(angle)
    st = se.sin(angle)
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
    return se.Matrix([[ct + m_vt_0 * axis[0], -m_st_2 + m_vt_0_1, m_st_1 + m_vt_0_2, 0],
                      [m_st_2 + m_vt_0_1, ct + m_vt_1 * axis[1], -m_st_0 + m_vt_1_2, 0],
                      [-m_st_1 + m_vt_0_2, m_st_0 + m_vt_1_2, ct + m_vt_2 * axis[2], 0],
                      [0, 0, 0, 1]])


def rotation_matrix_from_quaternion(x, y, z, w):
    """
    Unit quaternion to 4x4 rotation matrix according to:
    https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/src/frames.cpp#L167
    :type x: Union[float, Symbol]
    :type y: Union[float, Symbol]
    :type z: Union[float, Symbol]
    :type w: Union[float, Symbol]
    :return: 4x4 Matrix
    :rtype: Matrix
    """
    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w
    return se.Matrix([[w2 + x2 - y2 - z2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y, 0],
                      [2 * x * y + 2 * w * z, w2 - x2 + y2 - z2, 2 * y * z - 2 * w * x, 0],
                      [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, w2 - x2 - y2 + z2, 0],
                      [0, 0, 0, 1]])


def frame_axis_angle(x, y, z, axis, angle):
    """
    :type x: Union[float, Symbol]
    :type y: Union[float, Symbol]
    :type z: Union[float, Symbol]
    :param axis: 3x1 Matrix
    :type axis: Matrix
    :type angle: Union[float, Symbol]
    :return: 4x4 Matrix
    :rtype: Matrix
    """
    return translation3(x, y, z) * rotation_matrix_from_axis_angle(axis, angle)


def frame_rpy(x, y, z, roll, pitch, yaw):
    """
    :type x: Union[float, Symbol]
    :type y: Union[float, Symbol]
    :type z: Union[float, Symbol]
    :type roll: Union[float, Symbol]
    :type pitch: Union[float, Symbol]
    :type yaw: Union[float, Symbol]
    :return: 4x4 Matrix
    :rtype: Matrix
    """
    return translation3(x, y, z) * rotation_matrix_from_rpy(roll, pitch, yaw)


def frame_quaternion(x, y, z, qx, qy, qz, qw):
    """
    :type x: Union[float, Symbol]
    :type y: Union[float, Symbol]
    :type z: Union[float, Symbol]
    :type qx: Union[float, Symbol]
    :type qy: Union[float, Symbol]
    :type qz: Union[float, Symbol]
    :type qw: Union[float, Symbol]
    :return: 4x4 Matrix
    :rtype: Matrix
    """
    return translation3(x, y, z) * rotation_matrix_from_quaternion(qx, qy, qz, qw)


def inverse_frame(frame):
    """
    :param frame: 4x4 Matrix
    :type frame: Matrix
    :return: 4x4 Matrix
    :rtype: Matrix
    """
    inv = se.eye(4)
    inv[:3, :3] = frame[:3, :3].T
    inv[:3, 3] = -inv[:3, :3] * frame[:3, 3]
    return inv


def position_of(frame):
    """
    :param frame: 4x4 Matrix
    :type frame: Matrix
    :return: 4x1 Matrix; the translation part of a frame in form of a point
    :rtype: Matrix
    """
    return frame[:4, 3:]


def translation_of(frame):
    """
    :param frame: 4x4 Matrix
    :type frame: Matrix
    :return: 4x4 Matrix; sets the rotation part of a frame to identity
    :rtype: Matrix
    """
    return se.eye(3).col_join(se.Matrix([[0] * 3])).row_join(frame[:4, 3:])


def rotation_of(frame):
    """
    :param frame: 4x4 Matrix
    :type frame: Matrix
    :return: 4x4 Matrix; sets the translation part of a frame to 0
    :rtype: Matrix
    """
    return frame[:4, :3].row_join(se.Matrix([0, 0, 0, 1]))


def trace(matrix):
    """
    :type matrix: Matrix
    :rtype: Union[float, Symbol]
    """
    return sum(matrix[i, i] for i in range(matrix.shape[0]))


def rotation_distance(rotation_matrix1, rotation_matrix2):
    """
    :param rotation_matrix1: 4x4 or 3x3 Matrix
    :type rotation_matrix1: Matrix
    :param rotation_matrix2: 4x4 or 3x3 Matrix
    :type rotation_matrix2: Matrix
    :return: angle from the axis/angle representation of rotation_matrix1 * rotation_matrix2.T
    :rtype: Union[float, Symbol]
    """
    # TODO test me
    difference = rotation_matrix1 * rotation_matrix2.T
    # return -(((trace(difference) - 1)/2)-1)
    v = (trace(difference[:3, :3]) - 1) / 2
    # v = Max(-1, v)
    # v = Min(1, v)
    return se.acos(v)


def diffable_axis_angle_from_matrix(rotation_matrix):
    """
    MAKE SURE MATRIX IS NORMALIZED
    :param rotation_matrix: 4x4 or 3x3 Matrix
    :type rotation_matrix: Matrix
    :return: 3x1 Matrix, angle
    :rtype: (Matrix, Union[float, Symbol])
    """
    # TODO nan if angle 0
    # TODO buggy when angle == pi
    # TODO use 'if' to make angle always positive?
    rm = rotation_matrix
    angle = (trace(rm[:3, :3]) - 1) / 2
    angle = se.acos(angle)
    x = (rm[2, 1] - rm[1, 2])
    y = (rm[0, 2] - rm[2, 0])
    z = (rm[1, 0] - rm[0, 1])
    n = se.sqrt(x * x + y * y + z * z)

    axis = se.Matrix([x / n, y / n, z / n])
    return axis, angle


def diffable_axis_angle_from_matrix_stable(rotation_matrix):
    """
    :param rotation_matrix: 4x4 or 3x3 Matrix
    :type rotation_matrix: Matrix
    :return: 3x1 Matrix, angle
    :rtype: (Matrix, Union[float, Symbol])
    """
    # TODO buggy when angle == pi
    rm = rotation_matrix
    angle = (trace(rm[:3, :3]) - 1) / 2
    angle = diffable_min_fast(angle, 1)
    angle = diffable_max_fast(angle, -1)
    angle = se.acos(angle)
    x = (rm[2, 1] - rm[1, 2])
    y = (rm[0, 2] - rm[2, 0])
    z = (rm[1, 0] - rm[0, 1])
    n = se.sqrt(x * x + y * y + z * z)
    m = diffable_if_eq_zero(n, 1, n)
    axis = se.Matrix([diffable_if_eq_zero(n, 0, x / m),
                      diffable_if_eq_zero(n, 0, y / m),
                      diffable_if_eq_zero(n, 1, z / m)])
    return axis, angle


def axis_angle_from_matrix(rotation_matrix):
    """
    :param rotation_matrix: 4x4 or 3x3 Matrix
    :type rotation_matrix: Matrix
    :return: 3x1 Matrix, angle
    :rtype: (Matrix, Union[float, Symbol])
    """
    rm = rotation_matrix
    angle = (trace(rm[:3, :3]) - 1) / 2
    angle = se.Min(angle, 1)
    angle = se.Max(angle, -1)
    angle = se.acos(angle)
    x = (rm[2, 1] - rm[1, 2])
    y = (rm[0, 2] - rm[2, 0])
    z = (rm[1, 0] - rm[0, 1])
    n = se.sqrt(x * x + y * y + z * z)
    m = if_eq_zero(n, 1, n)
    axis = se.Matrix([if_eq_zero(n, 0, x / m),
                      if_eq_zero(n, 0, y / m),
                      if_eq_zero(n, 1, z / m)])
    sign = diffable_sign(angle)
    axis *= sign
    angle = sign * angle
    return axis, angle


def axis_angle_from_quaternion(x, y, z, w):
    """
    :type x: Union[float, Symbol]
    :type y: Union[float, Symbol]
    :type z: Union[float, Symbol]
    :type w: Union[float, Symbol]
    :return: 4x1 Matrix
    :rtype: Matrix
    """
    # TODO buggy, angle goes from 0 - 2*pi instead of -pi - +pi
    w2 = se.sqrt(1 - w ** 2)
    angle = (2 * se.acos(w))
    x = x / w2
    y = y / w2
    z = z / w2
    return se.Matrix([x, y, z]), angle


def quaternion_from_axis_angle(axis, angle):
    """
    :param axis: 3x1 Matrix
    :type axis: Matrix
    :type angle: Union[float, Symbol]
    :return: 4x1 Matrix
    :rtype: Matrix
    """
    half_angle = angle / 2
    return se.Matrix([axis[0] * se.sin(half_angle),
                      axis[1] * se.sin(half_angle),
                      axis[2] * se.sin(half_angle),
                      se.cos(half_angle)])


def axis_angle_from_rpy(roll, pitch, yaw):
    """
    :type roll: Union[float, Symbol]
    :type pitch: Union[float, Symbol]
    :type yaw: Union[float, Symbol]
    :return: 3x1 Matrix, angle
    :rtype: (Matrix, Union[float, Symbol])
    """
    # TODO maybe go over quaternion instead of matrix
    return diffable_axis_angle_from_matrix(rotation_matrix_from_rpy(roll, pitch, yaw))


_EPS = np.finfo(float).eps * 4.0


def rpy_from_matrix(rotation_matrix):
    """
    !takes time to compile!
    :param rotation_matrix: 4x4 Matrix
    :type rotation_matrix: Matrix
    :return: roll, pitch, yaw
    :rtype: (Union[float, Symbol], Union[float, Symbol], Union[float, Symbol])
    """
    i = 0
    j = 1
    k = 2

    cy = sqrt(rotation_matrix[i, i] * rotation_matrix[i, i] + rotation_matrix[j, i] * rotation_matrix[j, i])
    if0 = cy - _EPS
    ax = diffable_if_greater_zero(if0,
                                  atan2(rotation_matrix[k, j], rotation_matrix[k, k]),
                                  atan2(-rotation_matrix[j, k], rotation_matrix[j, j]))
    ay = diffable_if_greater_zero(if0,
                                  atan2(-rotation_matrix[k, i], cy),
                                  atan2(-rotation_matrix[k, i], cy))
    az = diffable_if_greater_zero(if0,
                                  atan2(rotation_matrix[j, i], rotation_matrix[i, i]),
                                  0)
    return ax, ay, az


def quaternion_from_rpy(roll, pitch, yaw):
    """
    :type roll: Union[float, Symbol]
    :type pitch: Union[float, Symbol]
    :type yaw: Union[float, Symbol]
    :return: 4x1 Matrix
    :type: Matrix
    """
    roll_half = roll / 2.0
    pitch_half = pitch / 2.0
    yaw_half = yaw / 2.0

    c_roll = se.cos(roll_half)
    s_roll = se.sin(roll_half)
    c_pitch = se.cos(pitch_half)
    s_pitch = se.sin(pitch_half)
    c_yaw = se.cos(yaw_half)
    s_yaw = se.sin(yaw_half)

    cc = c_roll * c_yaw
    cs = c_roll * s_yaw
    sc = s_roll * c_yaw
    ss = s_roll * s_yaw

    x = c_pitch * sc - s_pitch * cs
    y = c_pitch * ss + s_pitch * cc
    z = c_pitch * cs - s_pitch * sc
    w = c_pitch * cc + s_pitch * ss

    return se.Matrix([x, y, z, w])


def quaternion_from_matrix(matrix):
    """
    !takes a loooong time to compile!
    :param matrix: 4x4 or 3x3 Matrix
    :type matrix: Matrix
    :return: 4x1 Matrix
    :rtype: Matrix
    """
    # return quaternion_from_axis_angle(*diffable_axis_angle_from_matrix_stable(matrix))
    # return quaternion_from_rpy(*rpy_from_matrix(matrix))
    q = Matrix([0, 0, 0, 0])
    if isinstance(matrix, np.ndarray):
        M = Matrix(matrix.tolist())
    else:
        M = Matrix(matrix)
    t = trace(M)

    if0 = t - M[3, 3]

    if1 = M[1, 1] - M[0, 0]

    m_i_i = diffable_if_greater_zero(if1, M[1, 1], M[0, 0])
    m_i_j = diffable_if_greater_zero(if1, M[1, 2], M[0, 1])
    m_i_k = diffable_if_greater_zero(if1, M[1, 0], M[0, 2])

    m_j_i = diffable_if_greater_zero(if1, M[2, 1], M[1, 0])
    m_j_j = diffable_if_greater_zero(if1, M[2, 2], M[1, 1])
    m_j_k = diffable_if_greater_zero(if1, M[2, 0], M[1, 2])

    m_k_i = diffable_if_greater_zero(if1, M[0, 1], M[2, 0])
    m_k_j = diffable_if_greater_zero(if1, M[0, 2], M[2, 1])
    m_k_k = diffable_if_greater_zero(if1, M[0, 0], M[2, 2])

    if2 = M[2, 2] - m_i_i

    m_i_i = diffable_if_greater_zero(if2, M[2, 2], m_i_i)
    m_i_j = diffable_if_greater_zero(if2, M[2, 0], m_i_j)
    m_i_k = diffable_if_greater_zero(if2, M[2, 1], m_i_k)

    m_j_i = diffable_if_greater_zero(if2, M[0, 2], m_j_i)
    m_j_j = diffable_if_greater_zero(if2, M[0, 0], m_j_j)
    m_j_k = diffable_if_greater_zero(if2, M[0, 1], m_j_k)

    m_k_i = diffable_if_greater_zero(if2, M[1, 2], m_k_i)
    m_k_j = diffable_if_greater_zero(if2, M[1, 0], m_k_j)
    m_k_k = diffable_if_greater_zero(if2, M[1, 1], m_k_k)

    t = diffable_if_greater_zero(if0, t, m_i_i - (m_j_j + m_k_k) + M[3, 3])
    q[0] = diffable_if_greater_zero(if0, M[2, 1] - M[1, 2],
                                    diffable_if_greater_zero(if2, m_i_j + m_j_i,
                                                             diffable_if_greater_zero(if1, m_k_i + m_i_k, t)))
    q[1] = diffable_if_greater_zero(if0, M[0, 2] - M[2, 0],
                                    diffable_if_greater_zero(if2, m_k_i + m_i_k,
                                                             diffable_if_greater_zero(if1, t, m_i_j + m_j_i)))
    q[2] = diffable_if_greater_zero(if0, M[1, 0] - M[0, 1],
                                    diffable_if_greater_zero(if2, t, diffable_if_greater_zero(if1, m_i_j + m_j_i,
                                                                                              m_k_i + m_i_k)))
    q[3] = diffable_if_greater_zero(if0, t, m_k_j - m_j_k)

    q *= 0.5 / se.sqrt(t * M[3, 3])
    return q


def quaternion_multiply(q1, q2):
    """
    :param q1: 4x1 Matrix
    :type q1: Matrix
    :param q2: 4x1 Matrix
    :type q2: Matrix
    :return: 4x1 Matrix
    :rtype: Matrix
    """
    x0, y0, z0, w0 = q2
    x1, y1, z1, w1 = q1
    return se.Matrix([x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                      -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                      x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                      -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0])


def quaternion_conjugate(quaternion):
    """
    :param quaternion: 4x1 Matrix
    :type quaternion: Matrix
    :return: 4x1 Matrix
    :rtype: Matrix
    """
    return se.Matrix([-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]])


def quaternion_diff(q0, q1):
    """
    :param q0: 4x1 Matrix
    :type q0: Matrix
    :param q1: 4x1 Matrix
    :type q1: Matrix
    :return: 4x1 Matrix p, such that q1*p=q2
    :rtype: Matrix
    """
    return quaternion_multiply(quaternion_conjugate(q0), q1)


def cosine_distance(v0, v1):
    """
    :param v0: nx1 Matrix
    :type v0: Matrix
    :param v1: nx1 Matrix
    :type v1: Matrix
    :rtype: Union[float, Symbol]
    """
    return 1 - (v0.T * v1)[0]


def euclidean_distance(v1, v2):
    """
    :param v1: nx1 Matrix
    :type v1: Matrix
    :param v2: nx1 Matrix
    :type v2: Matrix
    :rtype: Union[float, Symbol]
    """
    return norm(v1 - v2)


# def floor(a):
#     a += VERY_SMALL_NUMBER
#     return (a - 0.5) - (sp.atan(sp.tan(np.pi * (a - 0.5)))) / (pi)


def fmod(a, b):
    s = sign(a)
    a = Abs(a)
    b = Abs(b)
    f1 = a - (b * floor(a / b))
    return s * se.Piecewise([0, Abs(a - b) < SMALL_NUMBER], [f1, True])


def normalize_angle_positive(angle):
    """
    Normalizes the angle to be 0 to 2*pi
    It takes and returns radians.
    """
    return fmod(fmod(angle, 2.0 * pi) + 2.0 * pi, 2.0 * pi)


def normalize_angle(angle):
    """
    Normalizes the angle to be -pi to +pi
    It takes and returns radians.
    """
    a = normalize_angle_positive(angle)
    return Piecewise([a - 2.0 * pi, a > pi], [a, True])


def shortest_angular_distance(from_angle, to_angle):
    """
    Given 2 angles, this returns the shortest angular
    difference.  The inputs and ouputs are of course radians.

    The result would always be -pi <= result <= pi. Adding the result
    to "from" will always get you an equivelent angle to "to".
    """
    return normalize_angle(to_angle - from_angle)


def diffable_slerp(q1, q2, t):
    """
    !takes a long time to compile!
    :param q1: 4x1 Matrix
    :type q1: Matrix
    :param q2: 4x1 Matrix
    :type q2: Matrix
    :param t: float, 0-1
    :type t:  Union[float, Symbol]
    :return: 4x1 Matrix; Return spherical linear interpolation between two quaternions.
    :rtype: Matrix
    """
    cos_half_theta = dot(q1, q2)

    if0 = -cos_half_theta
    q2 = diffable_if_greater_zero(if0, -q2, q2)
    cos_half_theta = diffable_if_greater_zero(if0, -cos_half_theta, cos_half_theta)

    if1 = diffable_abs(cos_half_theta) - 1.0

    # enforce acos(x) with -1 < x < 1
    cos_half_theta = diffable_min_fast(1, cos_half_theta)
    cos_half_theta = diffable_max_fast(-1, cos_half_theta)

    half_theta = acos(cos_half_theta)

    sin_half_theta = sqrt(1.0 - cos_half_theta * cos_half_theta)
    # prevent /0
    sin_half_theta = diffable_if_eq_zero(sin_half_theta, 1, sin_half_theta)
    if2 = 0.001 - diffable_abs(sin_half_theta)

    ratio_a = sin((1.0 - t) * half_theta) / sin_half_theta
    ratio_b = sin(t * half_theta) / sin_half_theta
    return diffable_if_greater_eq_zero(if1, se.Matrix(q1),
                                       diffable_if_greater_zero(if2, 0.5 * q1 + 0.5 * q2, ratio_a * q1 + ratio_b * q2))


def to_numpy(matrix):
    return np.array(matrix.tolist()).astype(float).reshape(matrix.shape)


