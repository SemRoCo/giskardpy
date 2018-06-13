from collections import defaultdict, OrderedDict
import numpy as np
import math
from geometry_msgs.msg import PointStamped, Point, Vector3Stamped, Vector3, Pose, PoseStamped, QuaternionStamped, \
    Quaternion
from sensor_msgs.msg import JointState
from giskardpy.trajectory import SingleJointState


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


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


def georg_slerp(q1, q2, t):
    xa, ya, za, wa = q1
    xb, yb, zb, wb = q2
    cos_half_theta = wa * wb + xa * xb + ya * yb + za * zb

    if (cos_half_theta < 0):
        wb = -wb
        xb = -xb
        yb = -yb
        zb = -zb
        cos_half_theta = -cos_half_theta

    if (abs(cos_half_theta) >= 1.0):
        return q1

    half_theta = math.acos(cos_half_theta)
    sin_half_theta = np.sqrt(1.0 - cos_half_theta * cos_half_theta)

    if (abs(sin_half_theta) < 0.001):
        return np.array([
            0.5 * xa + 0.5 * xb,
            0.5 * ya + 0.5 * yb,
            0.5 * za + 0.5 * zb,
            0.5 * wa + 0.5 * wb])

    ratio_a = np.sin((1.0 - t) * half_theta) / sin_half_theta
    ratio_b = np.sin(t * half_theta) / sin_half_theta

    return np.array([ratio_a * xa + ratio_b * xb,
                     ratio_a * ya + ratio_b * yb,
                     ratio_a * za + ratio_b * zb,
                     ratio_a * wa + ratio_b * wb])
