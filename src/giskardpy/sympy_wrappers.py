from sympy.vector import *
from sympy import *

ODOM = CoordSys3D('ODOM')

pathSeparator = '__'

def vec3(*args):
    if len(args) == 1 and isinstance(args[0], list):
        return Matrix([args[0][0], args[0][1], args[0][2], 0])
    elif len(args) == 3:
        return Matrix([args[0], args[1], args[2], 0])
    else:
        raise Exception('No overload for vec3 matching arguments: ' + str(args))

unitX = vec3(1,0,0)
unitY = vec3(0,1,0)
unitZ = vec3(0,0,1)

def point3(*args):
    if len(args) == 1 and isinstance(args[0], list):
        return Matrix([args[0][0], args[0][1], args[0][2], 1])
    elif len(args) == 3:
        return Matrix([args[0], args[1], args[2], 1])
    else:
        raise Exception('No overload for point3 matching arguments: ' + str(args))

def norm(v):
    if v.rows == 2:
        return sqrt(v[0] ** 2 + v[1] ** 2)
    else:
        return sqrt(v[0]**2 + v[1]**2 + v[2]**2)

def translation3(*args):
    #if len(args) == 1 and (isinstance(args[0], tuple) or isinstance(args[0], list)):
    #    args = args[0]

    if len(args) == 1:
        if (isinstance(args[0], list) or isinstance(args[0], tuple)) and len(args[0]) == 3:
            return eye(3).row_insert(3, Matrix([[0] * 3])).col_insert(3, point3(args[0]))
        elif isinstance(args[0], Matrix) and args[0].rows == 4 and args[0].cols == 1 and args[0][3] == 1:
            return eye(3).row_insert(3, Matrix([[0] * 3])).col_insert(3, args[0])
    elif len(args) == 3:
        return eye(3).row_insert(3, Matrix([[0] * 3])).col_insert(3, point3(args[0], args[1], args[2]))

    raise Exception('No overload for translation3 matching arguments: ' + str(args))

def rotation3(*args):
    if len(args) == 1 and (isinstance(args[0], tuple) or isinstance(args[0], list)):
        args = args[0]

    if len(args) == 2:
        if (isinstance(args[0], list) or isinstance(args[0], tuple)) and len(args[0]) == 3:
            return diag(AxisOrienter(args[1], args[0][0] * ODOM.i + args[0][1] * ODOM.j + args[0][2] * ODOM.k).rotation_matrix(ODOM), 1)
    elif len(args) == 3:
        return diag(BodyOrienter(args[0], args[1], args[2], 'XYZ').rotation_matrix(), 1)
    elif len(args) == 4:
        return diag(QuaternionOrienter(args[0], args[1], args[2], args[3]), 1)
    raise Exception('No overload for rotation3 matching arguments: ' + str(args))


def frame3(rot, loc):
    return translation3(loc) * rotation3(rot)

def posOf(frame):
    return frame.col(3)

def rotOf(frame):
    return diag(frame[:3, :3], 1)

def inputVec3(name, observables):
    x = Symbol(name + pathSeparator + 'x')
    y = Symbol(name + pathSeparator + 'y')
    z = Symbol(name + pathSeparator + 'z')
    observables.append(x)
    observables.append(y)
    observables.append(z)
    return vec3(x, y, z)


def inputPoint3(name, observables):
    x = Symbol(name + pathSeparator + 'x')
    y = Symbol(name + pathSeparator + 'y')
    z = Symbol(name + pathSeparator + 'z')
    observables.append(x)
    observables.append(y)
    observables.append(z)
    return point3(x, y, z)


def expandVec3Input(name, goal_dict):
    if name in goal_dict:
        if isinstance(goal_dict[name], list) or isinstance(goal_dict[name], tuple):
            vList = goal_dict[name]
            goal_dict[name + pathSeparator + 'x'] = vList[0]
            goal_dict[name + pathSeparator + 'y'] = vList[1]
            goal_dict[name + pathSeparator + 'z'] = vList[2]

    return goal_dict

