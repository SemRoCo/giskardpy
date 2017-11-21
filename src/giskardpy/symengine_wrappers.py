import symengine as sp
from symengine import Matrix, Symbol, eye, sympify, diag, zeros, lambdify #, Abs , Max, Min
import numpy as np
import sympy as sp2

from giskardpy import BACKEND

pathSeparator = '__'


def Abs(x):
    return sp.sqrt(x**2)

def Max(x, y):
    return ((x + y) + Abs(x - y)) / 2


def Min(x, y):
    return ((x + y) - Abs(x - y)) / 2


# @profile
def speed_up(function, parameters, backend=None):
    str_params = [str(x) for x in parameters]
    if len(parameters) == 0:
        constant_result = np.array(function).astype(float).reshape(function.shape)

        def f(**kwargs):
            return constant_result
    elif backend is None:
        # @profile
        def f(**kwargs):
            filtered_kwargs = {str(k): kwargs[k] for k in str_params}
            return np.array(function.subs(filtered_kwargs).tolist(), dtype=float).reshape(function.shape)
    elif backend == 'python':
        f = function
    else:

        try:
            fast_f = lambdify(list(parameters), function, backend=backend)
        except RuntimeError as e:
            try:
                print('WARNING RuntimeError: "{}" during lambdify with LLVM backend, fallback to numpy'.format(e))
                fast_f = lambdify(list(parameters), function, backend='lambda')
            except RuntimeError as e:
                print('WARNING RuntimeError: "{}" during lambdify with lambda backend, no speedup possible'.format(e))

                def f(**kwargs):
                    filtered_kwargs = {str(k): kwargs[k] for k in str_params}
                    return np.array(function.subs(filtered_kwargs).tolist(), dtype=float).reshape(function.shape)

                return f

        def f(**kwargs):
            filtered_args = [kwargs[k] for k in str_params]
            return np.nan_to_num(np.asarray(fast_f(*filtered_args)).reshape(function.shape))

    return f


def cross(u, v):
    return sp.Matrix([u[1] * v[2] - u[2] * v[1],
                      u[2] * v[0] - u[0] * v[2],
                      u[0] * v[1] - u[1] * v[0]])


# def cross(u,v):
#     return u.cross(v)

def vec3(x, y, z):
    return sp.Matrix([x, y, z, 0])


unitX = vec3(1, 0, 0)
unitY = vec3(0, 1, 0)
unitZ = vec3(0, 0, 1)


def point3(x, y, z):
    return sp.Matrix([x, y, z, 1])


def norm(v):
    r = 0
    for x in v:
        r += x ** 2
    return sp.sqrt(r)


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
    # return sp.Matrix(quaternion_matrix(quaternion_from_euler(roll, pitch, yaw, axes='rxyz')).T.tolist())

    rx = sp.Matrix([[1, 0, 0, 0],
                    [0, sp.cos(roll), sp.sin(roll), 0],
                    [0, -sp.sin(roll), sp.cos(roll), 0],
                    [0, 0, 0, 1]])
    ry = sp.Matrix([[sp.cos(pitch), 0, -sp.sin(pitch), 0],
                    [0, 1, 0, 0],
                    [sp.sin(pitch), 0, sp.cos(pitch), 0],
                    [0, 0, 0, 1]])
    rz = sp.Matrix([[sp.cos(yaw), sp.sin(yaw), 0, 0],
                    [-sp.sin(yaw), sp.cos(yaw), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    # return (rz * ry * rx)
    return (rx * ry * rz).T


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


def frame3_quaternion(q1, q2, q3, q4, loc):
    return translation3(*loc) * rotation3_quaternion(q1, q2, q3, q4)


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
    v = (trace(difference) - 1) / 2
    v = Max(-1, v)
    v = Min(1, v)
    return sp.acos(v)

# def quaternion_from_matrix(M):
#     # M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:4, :4]
#     m00 = M[0, 0]
#     m01 = M[0, 1]
#     m02 = M[0, 2]
#     m10 = M[1, 0]
#     m11 = M[1, 1]
#     m12 = M[1, 2]
#     m20 = M[2, 0]
#     m21 = M[2, 1]
#     m22 = M[2, 2]
#     # symmetric matrix K
#     K = Matrix([[m00 - m11 - m22, 0.0, 0.0, 0.0],
#                 [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
#                 [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
#                 [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
#     K /= 3.0
#     # quaternion is eigenvector of K that corresponds to largest eigenvalue
#     w, V = numpy.linalg.eigh(K)
#     q = V[[3, 0, 1, 2], numpy.argmax(w)]
#     if q[0] < 0.0:
#         numpy.negative(q, q)
#     return q
