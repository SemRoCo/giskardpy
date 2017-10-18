from sympy.vector import *
from sympy import Symbol

odom = CoordSys3D('odom')

def vec3(*args, **kwds):
    frame = odom
    if 'frame' in kwds:
        if isinstance(kwds['frame'], CoordSys3D):
            frame = kwds['frame']
        else:
            raise Exception('If argument "frame" is provided, it must be a CoordSys3D. Provided argument is of type: ' + str(type(kwds['frame'])))
    if len(args) == 1 and isinstance(args[0], list):
        return frame.i * args[0][0] + frame.j * args[0][1] + frame.k * args[0][2]
    elif len(args) == 3:
        return frame.i * args[0] + frame.j * args[1] + frame.k * args[2]

def frame(parent, name, rot, loc):
    if isinstance(loc, list):
        location = vec3(loc[0], loc[1], loc[2])
    else:
        location = loc

    if isinstance(rot, list):
        if len(rot) == 3:
            rotation = BodyOrienter(rot[0], rot[1], rot[2], 'XYZ')
        else:
            raise Exception('If rotation is supplied as a list, it should contain exactly three elements.')
    else:
        rotation = rot
    return parent.orient_new(name, (rotation,), location=location)

def inputVec3(name, observables):
    x = Symbol(name + '_x')
    y = Symbol(name + '_y')
    z = Symbol(name + '_z')
    observables.append(x)
    observables.append(y)
    observables.append(z)
    return vec3(x, y, z)

def expandVec3Input(name, goal_dict):
    if name in goal_dict:
        if isinstance(goal_dict[name], list) or isinstance(goal_dict[name], tuple):
            vList = goal_dict[name]
            goal_dict[name + '_x'] = vList[0]
            goal_dict[name + '_y'] = vList[1]
            goal_dict[name + '_z'] = vList[2]

    return goal_dict

unitX = odom.i
unitY = odom.j
unitZ = odom.k
