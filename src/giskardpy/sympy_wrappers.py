import sympy.vector as spv
import sympy as sp
from sympy.utilities.autowrap import autowrap
from sympy.utilities.lambdify import lambdify
import numpy as np
from sympy import Matrix, Symbol, eye, sympify, diag, zeros

from giskardpy import BACKEND
import symengine as se

ODOM = spv.CoordSys3D('ODOM')


def fake_Abs(x):
    return sp.sqrt(x ** 2)


def fake_Max(x, y):
    return ((x + y) + fake_Abs(x - y)) / 2


def fake_Min(x, y):
    return ((x + y) - fake_Abs(x - y)) / 2

# @profile
def speed_up(function, parameters, backend=None):
    str_params = [str(x) for x in parameters]
    if backend is None:
        # @profile
        def f(**kwargs):
            return np.array(function.subs(kwargs).tolist(), dtype=float).reshape(function.shape)

        return f
    else:
        if backend == 'numpy':
            fast_f = lambdify(list(parameters), function, dummify=False)
        elif backend == 'cython':
            fast_f = autowrap(function, args=list(parameters), backend='Cython')

        # @profile
        def f(**kwargs):
            filtered_kwargs = {str(k): kwargs[k] for k in str_params}
            return fast_f(**filtered_kwargs).astype(float)

        return f


def cross(v1, v2):
    return v1[:3, :].cross(v2[:3, :])


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

def translation3(x, y, z):
    r = sp.eye(4)
    r[0, 3] = x
    r[1, 3] = y
    r[2, 3] = z
    return r


def rotation3_rpy(r, p, y):
    return sp.diag(spv.BodyOrienter(r, p, y, 'XYZ').rotation_matrix(), 1)


def rotation3_axis_angle(axis, angle):
    return sp.diag(spv.AxisOrienter(angle,
                                    axis[0] * ODOM.i + axis[1] * ODOM.j + axis[2] * ODOM.k).rotation_matrix(ODOM), 1)


def rotation3_quaternion(x, y, z, w):
    return sp.diag(spv.QuaternionOrienter(w, x, y, z).rotation_matrix(), 1)


def frame3_axis_angle(axis, angle, loc):
    return translation3(loc) * rotation3_axis_angle(axis, angle)


def frame3_rpy(r, p, y, loc):
    return translation3(loc) * rotation3_rpy(r, p, y)


def frame3_quaternion(q1, q2, q3, q4, loc):
    return translation3(loc) * rotation3_quaternion(q1, q2, q3, q4)


def pos_of(frame):
    return frame.col(3)


def rot_of(frame):
    return sp.diag(frame[:3, :3], 1)


def inv_frame(frame):
    f_inv = sp.Matrix(frame)
    f_inv[:3, :3] = f_inv[:3, :3].T
    f_inv[3, :3] = -f_inv[3, :3]
    return f_inv

def trace(matrix):
    return sum(matrix[i, i] for i in range(3))

def matrix_to_axis_angle(rotation_matrix):
    rm = rotation_matrix
    angle = (trace(rm) - 1) / 2
    # angle = Max(-1, angle)
    # angle = Min(1, angle)
    angle = sp.acos(angle)
    # print(angle)
    # print(angle.evalf())
    x = (rm[2, 1] - rm[1, 2])
    y = (rm[0, 2] - rm[2, 0])
    z = (rm[1, 0] - rm[0, 1])
    n = sp.sqrt(x*x+y*y+z*z)

    # x = (rm[2, 1] - rm[1, 2]) / (
    # sp.sqrt((rm[2, 1] - rm[1, 2]) ** 2 + (rm[0, 2] - rm[2, 0]) ** 2 + (rm[1, 0] - rm[0, 1]) ** 2))
    # y = (rm[0, 2] - rm[2, 0]) / (
    # sp.sqrt((rm[2, 1] - rm[1, 2]) ** 2 + (rm[0, 2] - rm[2, 0]) ** 2 + (rm[1, 0] - rm[0, 1]) ** 2))
    # z = (rm[1, 0] - rm[0, 1]) / (
    # sp.sqrt((rm[2, 1] - rm[1, 2]) ** 2 + (rm[0, 2] - rm[2, 0]) ** 2 + (rm[1, 0] - rm[0, 1]) ** 2))
    axis = sp.Matrix([x/n, y/n, z/n])
    return axis, angle


def matrix_to_euler(rotation_matrix):
    sy = sp.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] +
                 rotation_matrix[1, 0] * rotation_matrix[1, 0])

    # singular = sy < 1e-6

    # if not singular:
    x = sp.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    y = sp.atan2(-rotation_matrix[2, 0], sy)
    z = sp.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    # else:
    #     x = sp.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
    #     y = sp.atan2(-rotation_matrix[2, 0], sy)
    #     z = 0

    return x, y, z