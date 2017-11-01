import sympy.vector as spv
import sympy as sp
from sympy.utilities.autowrap import autowrap
from sympy.utilities.lambdify import lambdify
import numpy as np
from sympy import Matrix, Symbol, eye, sympify, diag, zeros

from giskardpy import BACKEND

ODOM = spv.CoordSys3D('ODOM')

pathSeparator = '__'


#@profile
def speed_up(function, parameters):
    str_params = [str(x) for x in parameters]
    if BACKEND is None:
        #@profile
        def f(**kwargs):
            return np.array(function.subs(kwargs).tolist(), dtype=float).reshape(function.shape)

        return f
    else:
        if BACKEND == 'numpy':
            fast_f = lambdify(list(parameters), function, dummify=False)
        elif BACKEND == 'cython':
            fast_f = autowrap(function, args=list(parameters), backend='Cython')
        #@profile
        def f(**kwargs):
            filtered_kwargs = {str(k): kwargs[k] for k in str_params}
            return fast_f(**filtered_kwargs).astype(float)

        return f


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


def translation3(point):
    return sp.eye(3).row_insert(3, sp.Matrix([[0] * 3])).col_insert(3, point)


def rotation3_rpy(r, p, y):
    return sp.diag(spv.BodyOrienter(r, p, y, 'XYZ').rotation_matrix(), 1)


def rotation3_axis_angle(axis, angle):
    return sp.diag(spv.AxisOrienter(angle,
                                    axis[0] * ODOM.i + axis[1] * ODOM.j + axis[2] * ODOM.k).rotation_matrix(ODOM), 1).T


def rotation3_quaternion(x, y, z, w):
    return sp.diag(spv.QuaternionOrienter(w, x, y, z).rotation_matrix(), 1).T


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
