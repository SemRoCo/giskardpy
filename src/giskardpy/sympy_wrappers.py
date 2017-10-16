from sympy.vector import *

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
            rotation = BodyOrienter(rpy[0], rpy[1], rpy[2], 'XYZ')
        else:
            raise Exception('If rotation is supplied as a list, it should contain exactly three elements.')
    else:
        rotation = rot

    return parent.orient_new(name, (rotation,), location=location)

unitX = odom.i
unitY = odom.j
unitZ = odom.k
