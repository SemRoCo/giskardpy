#import sympy.vector as spv
import symengine as sp

pathSeparator = '__'

def vec3(x, y, z):
    return sp.Matrix([x, y, z, 0])


unitX = vec3(1, 0, 0)
unitY = vec3(0, 1, 0)
unitZ = vec3(0, 0, 1)


def point3(x, y, z):
    return sp.Matrix([x, y, z, 1])

def norm(v):
    if v.rows == 2:
        return sp.sqrt(v[0] ** 2 + v[1] ** 2)
    else:
        return sp.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)

def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

def cross(a, b):
    return sp.Matrix([a[1] * b[2] - a[2] * b[1],
                      a[2] * b[0] - a[0] * b[2],
                      a[0] * b[1] - a[1] * b[0]])

def translation3(point):
    return sp.eye(3).col_join(sp.Matrix([[0] * 3])).row_join(point)


def rotation3_rpy(r, p, y):
    """ Conversion of roll, pitch, yaw to 4x4 rotation matrix according to:
        http://planning.cs.uiuc.edu/node102.html
    """
    ca = sp.cos(y)
    sa = sp.sin(y)
    cb = sp.cos(p)
    sb = sp.sin(p)
    cg = sp.cos(r)
    sg = sp.sin(r)
    return sp.Matrix([[ca*cb, ca*sb*sg - sa*cg, ca*sb*cg + sa*sg, 0],
                      [sa*cb, sa*sb*sg + ca*cg, sa*sb*cg - ca*sg, 0],
                      [  -sb,            cb*sg,            cb*cg, 0],
                      [    0,                0,                0, 1]])


def rotation3_axis_angle(axis, angle):
    """ Conversion of unit axis and angle to 4x4 rotation matrix according to:
        http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/index.htm
    """
    c = sp.cos(angle)
    s = sp.sin(angle)
    t = 1 - c
    x = axis[0]
    y = axis[1]
    z = axis[2]
    return sp.Matrix([[t*x*x +   c, t*x*y - z*s, t*x*z + y*s, 0],
                      [t*x*y + z*s, t*y*y +   c, t*y*z - x*s, 0],
                      [t*x*z - y*s, t*y*z + x*s, t*z*z +   c, 0],
                      [          0,           0,           0, 1]])


def rotation3_quaternion(qr, qi, qj, qk):
    """ Unit quaternion to 4x4 rotation matrix according to:
        https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    """
    qi_sq = qi**2
    qj_sq = qj**2
    qk_sq = qk**2
    return sp.Matrix([[1 - 2*(qj_sq + qk_sq),     2*(qi*qj - qk*qr),     2*(qi*qk + qj*qr), 0],
                      [    2*(qi*qj + qk*qr), 1 - 2*(qi_sq + qk_sq),     2*(qj*qk - qi*qr), 0],
                      [    2*(qi*qk - qj*qr),     2*(qj*qk + qi*qr), 1 - 2*(qi_sq + qj_sq), 0],
                      [                    0,                     0,                     0, 1]])


def frame3_axis_angle(axis, angle, loc):
    return translation3(loc) * rotation3_axis_angle(axis, angle)


def frame3_rpy(r, p, y, loc):
    return translation3(loc) * rotation3_rpy(r, p, y)


def frame3_quaternion(q1, q2, q3, q4, loc):
    return translation3(loc) * rotation3_quaternion(q1, q2, q3, q4)


def pos_of(frame):
    return frame[:4, 3:]

def rot_of(frame):
    return sp.diag(frame[:3, :3], 1)
