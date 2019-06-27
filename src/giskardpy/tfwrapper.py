import PyKDL
import rospy
from geometry_msgs.msg import PoseStamped, Vector3Stamped, PointStamped, TransformStamped, Pose, Quaternion, Point, \
    Vector3, Twist, TwistStamped
from tf2_geometry_msgs import do_transform_pose, do_transform_vector3, do_transform_point
from tf2_kdl import transform_to_kdl
from tf2_py._tf2 import ExtrapolationException
from tf2_ros import Buffer, TransformListener

from giskardpy import logging

tfBuffer = None  # type: Buffer
tf_listener = None


def init(tf_buffer_size=15):
    """
    If you want to specify the buffer size, call this function manually, otherwise don't worry about it.
    :param tf_buffer_size: in secs
    :type tf_buffer_size: int
    """
    global tfBuffer, tf_listener
    tfBuffer = Buffer(rospy.Duration(tf_buffer_size))
    tf_listener = TransformListener(tfBuffer)
    rospy.sleep(5.0)


def wait_for_transform(target_frame, source_frame, time, timeout):
    global tfBuller
    return tfBuffer.can_transform(target_frame, source_frame, time, timeout)


def transform_pose(target_frame, pose):
    """
    Transforms a pose stamped into a different target frame.
    :type target_frame: str
    :type pose: PoseStamped
    :return: Transformed pose of None on loop failure
    :rtype: PoseStamped
    """
    global tfBuffer
    if tfBuffer is None:
        init()
    try:
        transform = tfBuffer.lookup_transform(target_frame,
                                              pose.header.frame_id,  # source frame
                                              pose.header.stamp,
                                              rospy.Duration(5.0))
        new_pose = do_transform_pose(pose, transform)
        return new_pose
    except ExtrapolationException as e:
        logging.logwarn(e)


def transform_vector(target_frame, vector):
    """
    Transforms a pose stamped into a different target frame.
    :type target_frame: str
    :type vector: Vector3Stamped
    :return: Transformed pose of None on loop failure
    :rtype: Vector3Stamped
    """
    global tfBuffer
    if tfBuffer is None:
        init()
    try:
        transform = tfBuffer.lookup_transform(target_frame,
                                              vector.header.frame_id,  # source frame
                                              vector.header.stamp,
                                              rospy.Duration(5.0))
        new_pose = do_transform_vector3(vector, transform)
        return new_pose
    except ExtrapolationException as e:
        logging.logwarn(e)


def transform_point(target_frame, point):
    """
    Transforms a pose stamped into a different target frame.
    :type target_frame: str
    :type point: PointStamped
    :return: Transformed pose of None on loop failure
    :rtype: PointStamped
    """
    global tfBuffer
    if tfBuffer is None:
        init()
    try:
        transform = tfBuffer.lookup_transform(target_frame,
                                              point.header.frame_id,  # source frame
                                              point.header.stamp,
                                              rospy.Duration(5.0))
        new_pose = do_transform_point(point, transform)
        return new_pose
    except ExtrapolationException as e:
        logging.logwarn(e)


def lookup_transform(target_frame, source_frame, time=None):
    """
    :type target_frame: str
    :type source_frame: str
    :return: Transform from target_frame to source_frame
    :rtype: TransformStamped
    """
    if not time:
        time = rospy.Time()
    global tfBuffer
    if tfBuffer is None:
        init()
    try:
        transform = tfBuffer.lookup_transform(target_frame, source_frame, time, rospy.Duration(5.0))
        return transform
    except:
        return None


def lookup_pose(target_frame, source_frame, time=None):
    """
    :type target_frame: str
    :type source_frame: str
    :return: target_frame <- source_frame
    :rtype: PoseStamped
    """
    p = PoseStamped()
    p.header.frame_id = source_frame
    if time is not None:
        p.header.stamp = time
    p.pose.orientation.w = 1.0
    return transform_pose(target_frame, p)

def lookup_point(target_frame, source_frame, time=None):
    """
    :type target_frame: str
    :type source_frame: str
    :return: target_frame <- source_frame
    :rtype: PointStamped
    """
    t = lookup_transform(target_frame, source_frame, time)
    p = PointStamped()
    p.header.frame_id = t.header.frame_id
    p.point.x = t.transform.translation.x
    p.point.y = t.transform.translation.y
    p.point.z = t.transform.translation.z
    return p


def pose_to_kdl(pose):
    """Convert a geometry_msgs Transform message to a PyKDL Frame.

    :param pose: The Transform message to convert.
    :type pose: Pose
    :return: The converted PyKDL frame.
    :rtype: PyKDL.Frame
    """
    return PyKDL.Frame(PyKDL.Rotation.Quaternion(pose.orientation.x,
                                                 pose.orientation.y,
                                                 pose.orientation.z,
                                                 pose.orientation.w),
                       PyKDL.Vector(pose.position.x,
                                    pose.position.y,
                                    pose.position.z))


def point_to_kdl(point):
    """
    :type point: Point
    :rtype: PyKDL.Vector
    """
    return PyKDL.Vector(point.x, point.y, point.z)


def twist_to_kdl(twist):
    t = PyKDL.Twist()
    t.vel[0] = twist.linear.x
    t.vel[1] = twist.linear.y
    t.vel[2] = twist.linear.z
    t.rot[0] = twist.angular.x
    t.rot[1] = twist.angular.y
    t.rot[2] = twist.angular.z
    return t


def msg_to_kdl(msg):
    if isinstance(msg, TransformStamped):
        return transform_to_kdl(msg)
    elif isinstance(msg, PoseStamped):
        return pose_to_kdl(msg.pose)
    elif isinstance(msg, Pose):
        return pose_to_kdl(msg)
    elif isinstance(msg, PointStamped):
        return point_to_kdl(msg.point)
    elif isinstance(msg, Point):
        return point_to_kdl(msg)
    elif isinstance(msg, Twist):
        return twist_to_kdl(msg)
    elif isinstance(msg, TwistStamped):
        return twist_to_kdl(msg.twist)
    else:
        raise TypeError(u'can\'t convert {} to kdl'.format(type(msg)))


def kdl_to_pose(frame):
    """
    :type frame: PyKDL.Frame
    :rtype: Pose
    """
    p = Pose()
    p.position.x = frame.p[0]
    p.position.y = frame.p[1]
    p.position.z = frame.p[2]
    p.orientation = Quaternion(*frame.M.GetQuaternion())
    return p


def kdl_to_point(vector):
    """
    :ty vector: PyKDL.Vector
    :return:
    """
    p = Point()
    p.x = vector[0]
    p.y = vector[1]
    p.z = vector[2]
    return p


def kdl_to_vector(vector):
    """
    :ty vector: PyKDL.Vector
    :return:
    """
    v = Vector3()
    v.x = vector[0]
    v.y = vector[1]
    v.z = vector[2]
    return v


def np_to_kdl(matrix):
    r = PyKDL.Rotation(matrix[0, 0], matrix[0, 1], matrix[0, 2],
                       matrix[1, 0], matrix[1, 1], matrix[1, 2],
                       matrix[2, 0], matrix[2, 1], matrix[2, 2])
    p = PyKDL.Vector(matrix[0, 3],
                     matrix[1, 3],
                     matrix[2, 3])
    return PyKDL.Frame(r, p)
