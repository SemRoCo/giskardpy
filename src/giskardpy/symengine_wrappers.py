import symengine as sp
from symengine import Matrix, Symbol, eye, sympify, diag, zeros
import numpy as np

pathSeparator = '__'

# @profile
def speed_up(function, parameters):
    str_params = [str(x) for x in parameters]
    # @profile
    def f(**kwargs):
        filtered_kwargs = {str(k): kwargs[k] for k in str_params}
        return np.array(function.subs(filtered_kwargs).tolist(), dtype=float).reshape(function.shape)

    return f


def vec3(x, y, z):
    return sp.Matrix([x, y, z, 0])


unitX = vec3(1, 0, 0)
unitY = vec3(0, 1, 0)
unitZ = vec3(0, 0, 1)

pi = 3.14159265359

def point3(x, y, z):
    return sp.Matrix([x, y, z, 1])

def norm(v):
    r = 0
    for x in v:
        r += x ** 2
    return sp.sqrt(r)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return sp.Matrix([a[1] * b[2] - a[2] * b[1],
                      a[2] * b[0] - a[0] * b[2],
                      a[0] * b[1] - a[1] * b[0], 0])


def translation3(point):
    return sp.eye(3).col_join(sp.Matrix([[0] * 3])).row_join(point)


def rotation3_rpy(roll, pitch, yaw):
    """ Conversion of roll, pitch, yaw to 4x4 rotation matrix according to:
        https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/src/frames.cpp#L167
    """
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
    return (rz*ry*rx)

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
    return sp.Matrix([[ct + m_vt_0*axis[0],  -m_st_2 + m_vt_0_1,   m_st_1 + m_vt_0_2, 0],
                      [  m_st_2 + m_vt_0_1, ct + m_vt_1*axis[1],  -m_st_0 + m_vt_1_2, 0],
                      [ -m_st_1 + m_vt_0_2,   m_st_0 + m_vt_1_2, ct + m_vt_2*axis[2], 0],
                      [                  0,                   0,                   0, 1]])



def rotation3_quaternion(x, y, z, w):
    """ Unit quaternion to 4x4 rotation matrix according to:
        https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/src/frames.cpp#L167
    """
    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w
    return sp.Matrix([[w2+x2-y2-z2, 2*x*y-2*w*z, 2*x*z+2*w*y, 0],
                      [2*x*y+2*w*z, w2-x2+y2-z2, 2*y*z-2*w*x, 0],
                      [2*x*z-2*w*y, 2*y*z+2*w*x, w2-x2-y2+z2, 0],
                      [          0,           0,           0, 1]])


def frame3_axis_angle(axis, angle, loc):
    return translation3(loc) * rotation3_axis_angle(axis, angle)


def frame3_rpy(r, p, y, loc):
    return translation3(loc) * rotation3_rpy(r, p, y)


def frame3_quaternion(q1, q2, q3, q4, loc):
    return translation3(loc) * rotation3_quaternion(q1, q2, q3, q4)


def pos_of(frame):
    return frame[:4, 3:]


def trans_of(frame):
    return sp.eye(3).col_join(sp.Matrix([[0] * 3])).row_join(frame[:4, 3:])


def rot_of(frame):
    return frame[:4, :3].row_join(sp.Matrix([0, 0, 0, 1]))
