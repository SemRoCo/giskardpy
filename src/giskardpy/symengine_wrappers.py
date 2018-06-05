import itertools
import os
import pickle
from warnings import warn

import errno
import symengine as sp
from symengine import Matrix, Symbol, eye, sympify, diag, zeros, lambdify, Abs, Max, Min, sin, cos, tan, acos, asin, \
    atan, atan2, nan, sqrt, log
import numpy as np

from symengine.lib.symengine_wrapper import Lambdify

pathSeparator = '_'


def fake_Abs(x):
    return sp.sqrt(x ** 2)


def fake_Max(x, y):
    return ((x + y) + fake_Abs(x - y)) / 2


def fake_Min(x, y):
    return ((x + y) - fake_Abs(x - y)) / 2

def fake_sign(a, e=-2.22507385851e-308):
    """
    if a > e:
        return 1
    if a < e:
        return -1
    if a == e:
        return nan
    """
    a -= e
    return a / fake_Abs(a)  # 1 or -1

def fake_if(a, b, c, e=-2.22507385851e-308):
    """
    if a >= 0:
        return b
    else:
        return c

    ** actual behavior **
    if a > e:
        return b
    if a < e:
        return c
    if a == e:
        return nan
    :type a: Union[float, Symbol]
    :type b: Union[float, Symbol]
    :type c: Union[float, Symbol]
    """
    a = fake_sign(a, e) # 1 or -1
    i = fake_Max(0, a) * b # 0 or b
    e = -fake_Min(0, a) * c # 0 or c
    return i + e  # i or e


def safe_compiled_function(f, file_name):
    if not os.path.exists(os.path.dirname(file_name)):
        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(file_name, 'w') as file:
        print('saved {}'.format(file_name))
        pickle.dump(f, file)


def load_compiled_function(file_name):
    if os.path.isfile(file_name):
        with open(file_name, 'r') as file:
            fast_f = pickle.load(file)
            return fast_f


class CompiledFunction(object):
    def __init__(self, str_params, fast_f, l, shape):
        self.str_params = str_params
        self.fast_f = fast_f
        self.l = l
        self.shape = shape

    def __call__(self, **kwargs):
        filtered_args = [kwargs[k] for k in self.str_params]
        out = np.empty(self.l)
        self.fast_f.unsafe_real(np.array(filtered_args, dtype=np.double), out)
        return np.nan_to_num(out).reshape(self.shape)


# @profile
def speed_up(function, parameters, backend='llvm'):
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
        if backend == 'llvm':
            try:
                fast_f = Lambdify(list(parameters), function, backend=backend, cse=True, real=True)
            except RuntimeError as e:
                warn('WARNING RuntimeError: "{}" during lambdify with LLVM backend, fallback to numpy'.format(e),
                     RuntimeWarning)
                backend = 'lambda'
        if backend == 'lambda':
            try:
                fast_f = Lambdify(list(parameters), function, backend='lambda', cse=True, real=True)
            except RuntimeError as e:
                warn('WARNING RuntimeError: "{}" during lambdify with lambda backend, no speedup possible'.format(e),
                     RuntimeWarning)
                backend = None

        if backend in ['llvm', 'lambda']:
            f = CompiledFunction(str_params, fast_f, len(function), function.shape)
        elif backend is None:
            # @profile
            def f(**kwargs):
                filtered_kwargs = {str(k): kwargs[k] for k in str_params}
                return np.array(function.subs(filtered_kwargs).tolist(), dtype=float).reshape(function.shape)
        if backend == 'python':
            f = function

    return f


def cross(u, v):
    return sp.Matrix([u[1] * v[2] - u[2] * v[1],
                      u[2] * v[0] - u[0] * v[2],
                      u[0] * v[1] - u[1] * v[0]])


def vector3(x, y, z):
    return sp.Matrix([x, y, z, 0])


unitX = vector3(1, 0, 0)
unitY = vector3(0, 1, 0)
unitZ = vector3(0, 0, 1)


def point3(x, y, z):
    return sp.Matrix([x, y, z, 1])


def norm(v):
    r = 0
    for x in v:
        r += x ** 2
    return sp.sqrt(r)


def scale(v, a):
    return v / norm(v) * a


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def translation3(x, y, z):
    r = sp.eye(4)
    r[0, 3] = x
    r[1, 3] = y
    r[2, 3] = z
    return r


def rotation3_rpy(roll, pitch, yaw):
    """ Conversion of roll, pitch, yaw to 4x4 rotation matrix according to:
        https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/src/frames.cpp#L167
    """
    # TODO don't split this into 3 matrices

    rx = sp.Matrix([[1, 0, 0, 0],
                    [0, sp.cos(roll), -sp.sin(roll), 0],
                    [0, sp.sin(roll), sp.cos(roll), 0],
                    [0, 0, 0, 1]])
    ry = sp.Matrix([[sp.cos(pitch), 0, sp.sin(pitch), 0],
                    [0, 1, 0, 0],
                    [-sp.sin(pitch), 0, sp.cos(pitch), 0],
                    [0, 0, 0, 1]])
    rz = sp.Matrix([[sp.cos(yaw), -sp.sin(yaw), 0, 0],
                    [sp.sin(yaw), sp.cos(yaw), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    return (rz * ry * rx)


def rotation3_axis_angle(axis, angle):
    """ Conversion of unit axis and angle to 4x4 rotation matrix according to:
        http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/index.htm
    """
    ct = sp.cos(angle)
    st = sp.sin(angle)
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
    return sp.Matrix([[ct + m_vt_0 * axis[0], -m_st_2 + m_vt_0_1, m_st_1 + m_vt_0_2, 0],
                      [m_st_2 + m_vt_0_1, ct + m_vt_1 * axis[1], -m_st_0 + m_vt_1_2, 0],
                      [-m_st_1 + m_vt_0_2, m_st_0 + m_vt_1_2, ct + m_vt_2 * axis[2], 0],
                      [0, 0, 0, 1]])


def rotation3_quaternion(x, y, z, w):
    """ Unit quaternion to 4x4 rotation matrix according to:
        https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/src/frames.cpp#L167
    """
    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w
    return sp.Matrix([[w2 + x2 - y2 - z2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y, 0],
                      [2 * x * y + 2 * w * z, w2 - x2 + y2 - z2, 2 * y * z - 2 * w * x, 0],
                      [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, w2 - x2 - y2 + z2, 0],
                      [0, 0, 0, 1]])


def frame3_axis_angle(axis, angle, loc):
    return translation3(*loc) * rotation3_axis_angle(axis, angle)


def frame3_rpy(r, p, y, loc):
    return translation3(*loc) * rotation3_rpy(r, p, y)


def frame3_quaternion(x, y, z, qx, qy, qz, qw):
    return translation3(x, y, z) * rotation3_quaternion(qx, qy, qz, qw)


def inverse_frame(frame):
    inv = sp.eye(4)
    inv[:3, :3] = frame[:3, :3].T
    inv[:3, 3] = -inv[:3, :3] * frame[:3, 3]
    return inv


def pos_of(frame):
    return frame[:4, 3:]


def trans_of(frame):
    return sp.eye(3).col_join(sp.Matrix([[0] * 3])).row_join(frame[:4, 3:])


def rot_of(frame):
    return frame[:4, :3].row_join(sp.Matrix([0, 0, 0, 1]))


def trace(matrix):
    return sum(matrix[i, i] for i in range(3))


def rotation_distance(rotation_matrix1, rotation_matrix2):
    difference = rotation_matrix1 * rotation_matrix2.T
    # return -(((trace(difference) - 1)/2)-1)
    v = (trace(difference) - 1) / 2
    # v = Max(-1, v)
    # v = Min(1, v)
    return sp.acos(v)


def axis_angle_from_matrix(rotation_matrix):
    rm = rotation_matrix
    angle = (trace(rm) - 1) / 2
    angle = sp.acos(angle)
    x = (rm[2, 1] - rm[1, 2])
    y = (rm[0, 2] - rm[2, 0])
    z = (rm[1, 0] - rm[0, 1])
    n = sp.sqrt(x * x + y * y + z * z)

    axis = sp.Matrix([x / n, y / n, z / n])
    return axis, angle


def axis_angle_from_quaternion(q):
    w2 = sp.sqrt(1 - q[3] ** 2)
    angle = 2 * sp.acos(q[3])
    x = q[0] / w2
    y = q[1] / w2
    z = q[2] / w2
    return sp.Matrix([x, y, z]), angle


def axis_angle_from_rpy(roll, pitch, yaw):
    return axis_angle_from_quaternion(*quaternion_from_rpy(roll, pitch, yaw))


def rpy_from_matrix(rotation_matrix):
    sy = sp.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] +
                 rotation_matrix[1, 0] * rotation_matrix[1, 0])

    x = sp.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    y = sp.atan2(-rotation_matrix[2, 0], sy)
    z = sp.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    return x, y, z


def quaternion_from_rpy(roll, pitch, yaw):
    roll_half = roll / 2.0
    pitch_half = pitch / 2.0
    yaw_half = yaw / 2.0

    c_roll = sp.cos(roll_half)
    s_roll = sp.sin(roll_half)
    c_pitch = sp.cos(pitch_half)
    s_pitch = sp.sin(pitch_half)
    c_yaw = sp.cos(yaw_half)
    s_yaw = sp.sin(yaw_half)

    cc = c_roll * c_yaw
    cs = c_roll * s_yaw
    sc = s_roll * c_yaw
    ss = s_roll * s_yaw

    x = c_pitch * sc - s_pitch * cs
    y = c_pitch * ss + s_pitch * cc
    z = c_pitch * cs - s_pitch * sc
    w = c_pitch * cc + s_pitch * ss

    return sp.Matrix([x, y, z, w])


def quaternion_multiply(q1, q2):
    x0, y0, z0, w0 = q2
    x1, y1, z1, w1 = q1
    return sp.Matrix([x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                      -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                      x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                      -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0])


def quaterntion_from_axis_angle(axis, angle):
    half_angle = angle / 2
    return sp.Matrix([axis[0] * sp.sin(half_angle),
                      axis[1] * sp.sin(half_angle),
                      axis[2] * sp.sin(half_angle),
                      sp.cos(half_angle)])


def quaternion_conjugate(quaternion):
    return sp.Matrix([-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]])


def quaternion_diff(q_current, q_goal):
    return quaternion_multiply(quaternion_conjugate(q_current), q_goal)


def quaternion_make_unique(q):
    sign = q[3] / fake_Abs(q[3])
    return q * sign


def cosine_distance(q1, q2):
    return 1 - (q1.T * q2)[0]


def euclidean_distance(v1, v2):
    return norm(v1 - v2)
