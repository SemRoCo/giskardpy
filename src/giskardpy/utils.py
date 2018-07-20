from __future__ import division

import codecs
import ctypes
import hashlib
import tempfile
from collections import defaultdict, OrderedDict
import numpy as np
from numpy import pi
import math
from geometry_msgs.msg import PointStamped, Point, Vector3Stamped, Vector3, Pose, PoseStamped, QuaternionStamped, \
    Quaternion
from sensor_msgs.msg import JointState
from giskardpy.data_types import SingleJointState
from giskardpy.data_types import ClosestPointInfo
from contextlib import contextmanager
import sys, os
import io

@contextmanager
def suppress_stdout(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different

@contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stdout



class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret

def urdfs_equal(urdf1, urdf2):
    return hashlib.md5(urdf1).hexdigest() == hashlib.md5(urdf2).hexdigest()

def slerp(q1, q2, t):
    cos_half_theta = np.dot(q1, q2)

    if (cos_half_theta < 0):
        q2 = -q2
        cos_half_theta = -cos_half_theta

    if (abs(cos_half_theta) >= 1.0):
        return q1

    half_theta = math.acos(cos_half_theta)
    sin_half_theta = np.sqrt(1.0 - cos_half_theta * cos_half_theta)

    if (abs(sin_half_theta) < 0.001):
        return 0.5 * q1 + 0.5 * q2

    ratio_a = np.sin((1.0 - t) * half_theta) / sin_half_theta
    ratio_b = np.sin(t * half_theta) / sin_half_theta

    return ratio_a * q1 + ratio_b * q2


def sphere_volume(r):
    return (4 / 3.) * pi * r ** 3

def sphere_surface(r):
    return 4*pi*r**2

def cube_volume(l, w, h):
    return l * w * h


def cube_surface(l, w, h):
    return 2 * (l * w + l * h + w * h)


def cylinder_volume(r, h):
    return pi * r ** 2 * h

def cylinder_surface(r, h):
    return 2*pi*r*(h+r)



def closest_point_constraint_violated(cp, multiplier=0.9):
    for link_name, cpi_info in cp.items():  # type: (str, ClosestPointInfo)
        if cpi_info.contact_distance < cpi_info.min_dist * multiplier:
            print(cpi_info.link_a, cpi_info.link_b, cpi_info.contact_distance)
            return True
    return False


#
# CONVERSION FUNCTIONS FOR ROS MESSAGES
#


def to_joint_state_dict(msg):
    """
    Converts a ROS message of type sensor_msgs/JointState into an instance of MultiJointState.
    :param msg: ROS message to convert.
    :type msg: JointState
    :return: Corresponding MultiJointState instance.
    :rtype: OrderedDict[str, SingleJointState]
    """
    mjs = OrderedDict()
    for i, joint_name in enumerate(msg.name):
        sjs = SingleJointState()
        sjs.name = joint_name
        sjs.position = msg.position[i]
        try:
            sjs.velocity = msg.velocity[i]
        except IndexError:
            sjs.velocity = 0
        try:
            sjs.effort = msg.effort[i]
        except IndexError:
            sjs.effort = 0
        mjs[joint_name] = sjs
    return mjs


def dict_to_joint_states(joint_state_dict):
    js = JointState()
    for k, v in joint_state_dict.items():
        js.name.append(k)
        js.position.append(v)
        js.velocity.append(0)
        js.effort.append(0)
    return js


def to_point_stamped(frame_id, point):
    """
    :param frame_id:
    :type frame_id: str
    :param point:
    :type point: list
    :return:
    :rtype: geometry_msgs.msg._PointStamped.PointStamped
    """
    p = PointStamped()
    p.header.frame_id = frame_id
    p.point = Point(*point)
    return p


def to_vector3_stamped(frame_id, vector):
    """
    :param frame_id:
    :type frame_id: str
    :param vector:
    :type vector: list
    :return:
    :rtype: Vector3Stamped
    """
    v = Vector3Stamped()
    v.header.frame_id = frame_id
    v.vector = Vector3(*vector)
    return v


def to_list(thing):
    if isinstance(thing, QuaternionStamped):
        thing = thing.quaternion
    if isinstance(thing, Quaternion):
        return [thing.x,
                thing.y,
                thing.z,
                thing.w]
    if isinstance(thing, PointStamped):
        thing = thing.point
    if isinstance(thing, PoseStamped):
        thing = thing.pose
    if isinstance(thing, Vector3Stamped):
        thing = thing.vector
    if isinstance(thing, Point) or isinstance(thing, Vector3):
        return [thing.x,
                thing.y,
                thing.z]
    if isinstance(thing, Pose):
        return [thing.position.x,
                thing.position.y,
                thing.position.z,
                thing.orientation.x,
                thing.orientation.y,
                thing.orientation.z,
                thing.orientation.w]
