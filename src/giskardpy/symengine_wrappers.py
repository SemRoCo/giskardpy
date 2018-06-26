import itertools
import os
import pickle
from warnings import warn

import errno
import symengine as sp
from symengine import Matrix, Symbol, eye, sympify, diag, zeros, lambdify, Abs, Max, Min, sin, cos, tan, acos, asin, \
    atan, atan2, nan, sqrt, log, tanh, var
import numpy as np

from symengine.lib.symengine_wrapper import Lambdify
from tf.transformations import unit_vector

from giskardpy.exceptions import SymengineException

pathSeparator = '_'


def fake_Abs(x):
    return sp.sqrt(x ** 2)


def fake_Max(x, y):
    return ((x + y) + fake_Abs(x - y)) / 2


def fake_Min(x, y):
    return ((x + y) - fake_Abs(x - y)) / 2


def fake_sign(a, e=-2.22507385851e-308):
    """
    if a > 0:
        return 1
    if a < 0:
        return -1
    if a == 0:
        return 0
    """
    return a / (e + fake_Abs(a))


def sigmoid(a, e=9e300):
    return (tanh(a * e) + 1) / 2


def if_greater_zero(a, b, c, e=2.22507385851e-308):
    """
    if a > 0:
        return b
    else:
        return c

    ** actual behavior **
    if a > e:
        return b
    if a < e:
        return c
    if a == e:
        return 0
    :type a: Union[float, Symbol]
    :type b: Union[float, Symbol]
    :type c: Union[float, Symbol]
    """
    a = fake_sign(a - e)  # 1 or -1
    _if = fake_Max(0, a) * b  # 0 or b
    _else = -fake_Min(0, a) * c  # 0 or c
    return _if + _else  # i or e
    # return sigmoid((a-e)) * b + sigmoid(-(a-e)) * c


def if_eq_zero(a, b, c, e=2.22507385851e-308):
    """
    if a == 0:
        return b
    else:
        return c
    """
    a = fake_Abs(fake_sign(a, e))
    return (1 - a) * b + a * c


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
        try:
            filtered_args = [kwargs[k] for k in self.str_params]
            out = np.empty(self.l)
            self.fast_f.unsafe_real(np.array(filtered_args, dtype=np.double), out)
            return np.nan_to_num(out).reshape(self.shape)
        except KeyError as e:
            raise SymengineException('{}\ntry deleting the last loaded compiler to trigger recompilation'.format(e.message))


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
    return (a.T*b)[0]


def translation3(x, y, z, w=1):
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
    return sum(matrix[i, i] for i in range(matrix.shape[0]))


def rotation_distance(rotation_matrix1, rotation_matrix2):
    difference = rotation_matrix1 * rotation_matrix2.T
    # return -(((trace(difference) - 1)/2)-1)
    v = (trace(difference[:3, :3]) - 1) / 2
    # v = Max(-1, v)
    # v = Min(1, v)
    return sp.acos(v)


def axis_angle_from_matrix(rotation_matrix):
    rm = rotation_matrix
    angle = (trace(rm[:3, :3]) - 1) / 2
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


def quaternion_from_matrix(matrix):
    q = Matrix([0, 0, 0, 0])
    M = Matrix(matrix)
    t = trace(M)

    if0 = t - M[3, 3]

    if1 = M[1, 1] - M[0, 0]

    m_i_i = if_greater_zero(if1, M[1, 1], M[0, 0])
    m_j_j = if_greater_zero(if1, M[2, 2], M[1, 1])
    m_k_k = if_greater_zero(if1, M[0, 0], M[2, 2])

    m_i_j = if_greater_zero(if1, M[1, 2], M[0, 1])
    m_j_i = if_greater_zero(if1, M[2, 1], M[1, 0])
    m_k_i = if_greater_zero(if1, M[0, 1], M[2, 0])
    m_i_k = if_greater_zero(if1, M[1, 0], M[0, 2])
    m_k_j = if_greater_zero(if1, M[0, 2], M[2, 1])
    m_j_k = if_greater_zero(if1, M[2, 0], M[1, 2])

    if2 = M[2, 2] - m_i_i

    m_i_i = if_greater_zero(if2, M[2, 2], m_i_i)
    m_j_j = if_greater_zero(if2, M[0, 0], m_j_j)
    m_k_k = if_greater_zero(if2, M[1, 1], m_k_k)

    m_i_j = if_greater_zero(if2, M[2, 0], m_i_j)
    m_j_i = if_greater_zero(if2, M[0, 2], m_j_i)
    m_k_i = if_greater_zero(if2, M[1, 2], m_k_i)
    m_i_k = if_greater_zero(if2, M[2, 1], m_i_k)
    m_k_j = if_greater_zero(if2, M[1, 0], m_k_j)
    m_j_k = if_greater_zero(if2, M[0, 1], m_j_k)

    t2 = m_i_i - (m_j_j + m_k_k) + M[3, 3]
    q[0] = if_greater_zero(if0, M[2, 1] - M[1, 2],
                           if_greater_zero(if2, m_i_j + m_j_i, if_greater_zero(if1, m_k_i + m_i_k, t2)))
    q[1] = if_greater_zero(if0, M[0, 2] - M[2, 0],
                           if_greater_zero(if2, m_k_i + m_i_k, if_greater_zero(if1, t2, m_i_j + m_j_i)))
    q[2] = if_greater_zero(if0, M[1, 0] - M[0, 1],
                           if_greater_zero(if2, t2, if_greater_zero(if1, m_i_j + m_j_i, m_k_i + m_i_k)))
    q[3] = if_greater_zero(if0, t, m_k_j - m_j_k)
    q *= if_greater_zero(if0, 0.5 / sp.sqrt(t * M[3, 3]), 0.5 / sp.sqrt(t2 * M[3, 3]))
    return q


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


def slerp(r1, r2, t):
    xa, ya, za, wa = quaternion_from_matrix(r1)
    xb, yb, zb, wb = quaternion_from_matrix(r2)
    cos_half_theta = wa * wb + xa * xb + ya * yb + za * zb

    if (cos_half_theta < 0):
        wb = -wb
        xb = -xb
        yb = -yb
        zb = -zb
        cos_half_theta = -cos_half_theta

    if (abs(cos_half_theta) >= 1.0):
        return a

    half_theta = acos(cos_half_theta)
    sin_half_theta = sqrt(1.0 - cos_half_theta * cos_half_theta)

    if (abs(sin_half_theta) < 0.001):
        return Matrix([
            0.5 * xa + 0.5 * xb,
            0.5 * ya + 0.5 * yb,
            0.5 * za + 0.5 * zb,
            0.5 * wa + 0.5 * wb])

    ratio_a = sin((1.0 - t) * half_theta) / sin_half_theta
    ratio_b = sin(t * half_theta) / sin_half_theta

    return Matrix([
        ratio_a * xa + ratio_b * xb,
        ratio_a * ya + ratio_b * yb,
        ratio_a * za + ratio_b * zb,
        ratio_a * wa + ratio_b * wb])


_EPS = np.finfo(float).eps * 4.0

def slerp2(q0, q1, fraction):
    d = dot(q0, q1)
    q1 *= if_greater_zero(d, 1, -1)
    d = fake_Abs(d)
    angle = acos(d)
    isin = 1.0 / sin(angle)
    qr = q0 * sin((1.0 - fraction) * angle) * isin
    q1 *= sin(fraction * angle) * isin
    qr = qr + q1

    return if_eq_zero(fraction,
                      q0,
                      if_eq_zero(fraction - 1,
                                 q1,
                                 if_greater_zero(_EPS - fake_Abs(fake_Abs(d) - 1.0),
                                                 q0,
                                                 if_greater_zero(_EPS - fake_Abs(angle),
                                                                 q0,
                                                                 qr))))
