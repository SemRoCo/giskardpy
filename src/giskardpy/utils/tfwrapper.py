from copy import copy

import PyKDL
import numpy as np
import rospy
import yaml
from geometry_msgs.msg import PoseStamped, Vector3Stamped, PointStamped, TransformStamped, Pose, Quaternion, Point, \
    Vector3, Twist, TwistStamped, QuaternionStamped, Transform
from std_msgs.msg import ColorRGBA
from tf.transformations import quaternion_from_matrix, quaternion_matrix
from tf2_geometry_msgs import do_transform_pose, do_transform_vector3, do_transform_point
from tf2_kdl import transform_to_kdl as transform_stamped_to_kdl
from tf2_py import InvalidArgumentException
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import MarkerArray, Marker

from giskardpy.my_types import PrefixName
from giskardpy.utils import logging
from giskardpy.utils.decorators import memoize

tfBuffer: Buffer = None
tf_listener: TransformListener = None


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
    try:
        get_tf_root()
    except AssertionError as e:
        logging.logwarn(e)


def get_tf_buffer():
    global tfBuffer
    if tfBuffer is None:
        init()
    return tfBuffer


@memoize
def get_tf_root() -> str:
    tfBuffer = get_tf_buffer()
    frames = yaml.safe_load(tfBuffer.all_frames_as_yaml())
    frames_with_parent = set(frames.keys())
    frame_parents = set(x['parent'] for x in frames.values())
    tf_roots = frame_parents.difference(frames_with_parent)
    assert len(tf_roots) < 2, f'There are more than one tf tree: {tf_roots}.'
    assert len(tf_roots) > 0, 'There is no tf tree.'
    return tf_roots.pop()


def get_full_frame_names(frame_name):
    """
    Gets the full tf frame name if the frame with the name frame_name
    is in a separate namespace.

    :rtype: str
    """
    tfBuffer = get_tf_buffer()
    ret = list()
    tf_frames = tfBuffer._getFrameStrings()
    for tf_frame in tf_frames:
        try:
            frame = tf_frame[tf_frame.index("/") + 1:]
            if frame == frame_name or frame_name == tf_frame:
                ret.append(tf_frame)
        except ValueError:
            continue
    if len(ret) == 0:
        raise KeyError(f'Could not find frame {frame_name} in the buffer of the tf Listener.')
    return ret

def wait_for_transform(target_frame, source_frame, time, timeout):
    tfBuffer = get_tf_buffer()
    return tfBuffer.can_transform(target_frame, source_frame, time, timeout)


@profile
def transform_msg(target_frame, msg, timeout=5):
    if isinstance(msg, PoseStamped):
        return transform_pose(target_frame, msg, timeout)
    elif isinstance(msg, PointStamped):
        return transform_point(target_frame, msg, timeout)
    elif isinstance(msg, Vector3Stamped):
        return transform_vector(target_frame, msg, timeout)
    elif isinstance(msg, QuaternionStamped):
        return transform_quaternion(target_frame, msg, timeout)
    else:
        raise NotImplementedError(f'tf transform message of type \'{type(msg)}\'')


def transform_pose(target_frame, pose, timeout=5.0):
    """
    Transforms a pose stamped into a different target frame.
    :type target_frame: Union[str, unicode]
    :type pose: PoseStamped
    :return: Transformed pose of None on loop failure
    :rtype: PoseStamped
    """
    transform = lookup_transform(target_frame, pose.header.frame_id, pose.header.stamp, timeout)
    new_pose = do_transform_pose(pose, transform)
    return new_pose


def lookup_transform(target_frame, source_frame, time=None, timeout=5.0):
    if not target_frame:
        raise InvalidArgumentException('target frame can not be empty')
    if not source_frame:
        raise InvalidArgumentException('source frame can not be empty')
    if time is None:
        time = rospy.Time()
    tfBuffer = get_tf_buffer()
    return tfBuffer.lookup_transform(str(target_frame),
                                     str(source_frame),  # source frame
                                     time,
                                     rospy.Duration(timeout))


def make_transform(parent_frame: PrefixName, child_frame: PrefixName, pose: Pose, normalize_quaternion: bool = True) \
        -> TransformStamped:
    tf = TransformStamped()
    tf.header.frame_id = str(parent_frame)
    tf.header.stamp = rospy.get_rostime()
    tf.child_frame_id = str(child_frame)
    tf.transform.translation.x = pose.position.x
    tf.transform.translation.y = pose.position.y
    tf.transform.translation.z = pose.position.z
    if normalize_quaternion:
        tf.transform.rotation = normalize_quaternion_msg(pose.orientation)
    else:
        tf.transform.rotation = pose.orientation
    return tf


def transform_vector(target_frame, vector, timeout=5):
    """
    Transforms a pose stamped into a different target frame.
    :type target_frame: Union[str, unicode]
    :type vector: Vector3Stamped
    :return: Transformed pose of None on loop failure
    :rtype: Vector3Stamped
    """
    transform = lookup_transform(target_frame, vector.header.frame_id, vector.header.stamp, timeout)
    new_pose = do_transform_vector3(vector, transform)
    return new_pose


def transform_quaternion(target_frame: str, quaternion: QuaternionStamped, timeout: float = 5) -> QuaternionStamped:
    """
    Transforms a pose stamped into a different target frame.
    :return: Transformed pose of None on loop failure
    """
    p = PoseStamped()
    p.header = quaternion.header
    p.pose.orientation = quaternion.quaternion
    new_pose = transform_pose(target_frame, p, timeout)
    new_quaternion = QuaternionStamped()
    new_quaternion.header = new_pose.header
    new_quaternion.quaternion = new_pose.pose.orientation
    return new_quaternion


def transform_point(target_frame, point, timeout=5):
    """
    Transforms a pose stamped into a different target frame.
    :type target_frame: Union[str, unicode]
    :type point: PointStamped
    :return: Transformed pose of None on loop failure
    :rtype: PointStamped
    """
    transform = lookup_transform(target_frame, point.header.frame_id, point.header.stamp, timeout)
    new_pose = do_transform_point(point, transform)
    return new_pose


def lookup_pose(target_frame, source_frame, time=None):
    """
    :type target_frame: Union[str, unicode]
    :type source_frame: Union[str, unicode]
    :return: target_frame <- source_frame
    :rtype: PoseStamped
    """
    p = PoseStamped()
    p.header.frame_id = str(source_frame)
    if time is not None:
        p.header.stamp = time
    p.pose.orientation.w = 1.0
    return transform_pose(target_frame, p)


def lookup_point(target_frame, source_frame, time=None):
    """
    :type target_frame: Union[str, unicode]
    :type source_frame: Union[str, unicode]
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


def transform_to_kdl(transform):
    ts = TransformStamped()
    ts.transform = transform
    return transform_stamped_to_kdl(ts)


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


def quaternion_to_kdl(pose):
    """Convert a geometry_msgs Transform message to a PyKDL Frame.

    :param pose: The Transform message to convert.
    :type pose: Quaternion
    :return: The converted PyKDL frame.
    :rtype: PyKDL.Frame
    """
    return PyKDL.Frame(PyKDL.Rotation.Quaternion(pose.x,
                                                 pose.y,
                                                 pose.z,
                                                 pose.w))


def point_to_kdl(point):
    """
    :type point: Union[Point, Vector]
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
        return transform_stamped_to_kdl(msg)
    elif isinstance(msg, Transform):
        return transform_to_kdl(msg)
    elif isinstance(msg, PoseStamped):
        return pose_to_kdl(msg.pose)
    elif isinstance(msg, Pose):
        return pose_to_kdl(msg)
    elif isinstance(msg, PointStamped):
        return point_to_kdl(msg.point)
    elif isinstance(msg, Point):
        return point_to_kdl(msg)
    elif isinstance(msg, QuaternionStamped):
        return quaternion_to_kdl(msg.quaternion)
    elif isinstance(msg, Quaternion):
        return quaternion_to_kdl(msg)
    elif isinstance(msg, Twist):
        return twist_to_kdl(msg)
    elif isinstance(msg, TwistStamped):
        return twist_to_kdl(msg.twist)
    elif isinstance(msg, Vector3Stamped):
        return point_to_kdl(msg.vector)
    elif isinstance(msg, Vector3):
        return point_to_kdl(msg)
    else:
        raise TypeError('can\'t convert {} to kdl'.format(type(msg)))


def normalize(msg):
    if isinstance(msg, Quaternion):
        rotation = np.array([msg.x,
                             msg.y,
                             msg.z,
                             msg.w])
        normalized_rotation = rotation / np.linalg.norm(rotation)
        return Quaternion(*normalized_rotation)
    elif isinstance(msg, Vector3):
        tmp = np.array([msg.x,
                        msg.y,
                        msg.z])
        tmp = tmp / np.linalg.norm(tmp)
        return Vector3(*tmp)


def kdl_to_transform(frame):
    t = Transform()
    t.translation.x = frame.p[0]
    t.translation.y = frame.p[1]
    t.translation.z = frame.p[2]
    t.rotation = normalize(Quaternion(*frame.M.GetQuaternion()))
    return t


def kdl_to_transform_stamped(frame, frame_id, child_frame_id):
    t = TransformStamped()
    t.header.frame_id = frame_id
    t.header.stamp = rospy.get_rostime()
    t.child_frame_id = child_frame_id
    t.transform = kdl_to_transform(frame)
    return t


def kdl_to_pose(frame):
    """
    :type frame: PyKDL.Frame
    :rtype: Pose
    """
    p = Pose()
    p.position.x = frame.p[0]
    p.position.y = frame.p[1]
    p.position.z = frame.p[2]
    p.orientation = normalize(Quaternion(*frame.M.GetQuaternion()))
    return p


def kdl_to_transform(frame: PyKDL.Frame) -> Transform:
    t = Transform()
    t.translation.x = frame.p[0]
    t.translation.y = frame.p[1]
    t.translation.z = frame.p[2]
    t.rotation = normalize(Quaternion(*frame.M.GetQuaternion()))
    return t


def kdl_to_pose_stamped(frame, frame_id):
    """
    :type frame: PyKDL.Frame
    :rtype: PoseStamped
    """
    p = PoseStamped()
    p.header.frame_id = frame_id
    p.pose = kdl_to_pose(frame)
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


def kdl_to_quaternion(rotation_matrix):
    return Quaternion(*quaternion_from_matrix([[rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2], 0],
                                               [rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2], 0],
                                               [rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2], 0],
                                               [0, 0, 0, 1]]))


def np_to_kdl(matrix):
    r = PyKDL.Rotation(matrix[0, 0], matrix[0, 1], matrix[0, 2],
                       matrix[1, 0], matrix[1, 1], matrix[1, 2],
                       matrix[2, 0], matrix[2, 1], matrix[2, 2])
    p = PyKDL.Vector(matrix[0, 3],
                     matrix[1, 3],
                     matrix[2, 3])
    return PyKDL.Frame(r, p)


def kdl_to_np(kdl_thing):
    if isinstance(kdl_thing, PyKDL.Wrench):
        return np.array([kdl_thing.force[0],
                         kdl_thing.force[1],
                         kdl_thing.force[2],
                         kdl_thing.torque[0],
                         kdl_thing.torque[1],
                         kdl_thing.torque[2]])
    if isinstance(kdl_thing, PyKDL.Twist):
        return np.array([kdl_thing.vel[0],
                         kdl_thing.vel[1],
                         kdl_thing.vel[2],
                         kdl_thing.rot[0],
                         kdl_thing.rot[1],
                         kdl_thing.rot[2]])
    if isinstance(kdl_thing, PyKDL.Vector):
        return np.array([kdl_thing[0],
                         kdl_thing[1],
                         kdl_thing[2]])
    if isinstance(kdl_thing, PyKDL.Frame):
        return np.array([[kdl_thing.M[0, 0], kdl_thing.M[0, 1], kdl_thing.M[0, 2], kdl_thing.p[0]],
                         [kdl_thing.M[1, 0], kdl_thing.M[1, 1], kdl_thing.M[1, 2], kdl_thing.p[1]],
                         [kdl_thing.M[2, 0], kdl_thing.M[2, 1], kdl_thing.M[2, 2], kdl_thing.p[2]],
                         [0, 0, 0, 1]])
    if isinstance(kdl_thing, PyKDL.Rotation):
        return np.array([[kdl_thing[0, 0], kdl_thing[0, 1], kdl_thing[0, 2], 0],
                         [kdl_thing[1, 0], kdl_thing[1, 1], kdl_thing[1, 2], 0],
                         [kdl_thing[2, 0], kdl_thing[2, 1], kdl_thing[2, 2], 0],
                         [0, 0, 0, 1]])


def np_to_pose(matrix: np.ndarray) -> Pose:
    return kdl_to_pose(np_to_kdl(matrix))


def np_to_transform(matrix: np.ndarray) -> Transform:
    return kdl_to_transform(np_to_kdl(matrix))


def pose_to_np(msg):
    p = np.array([msg.position.x, msg.position.y, msg.position.z])
    q = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
    T = quaternion_matrix(q)
    T[0:3, -1] = p
    return T


def pose_stamped_to_np(msg):
    return pose_to_np(msg.pose)


def quaternion_to_np(msg: Quaternion) -> np.ndarray:
    q = np.array([msg.x, msg.y, msg.z, msg.w])
    return quaternion_matrix(q)


def quaternion_stamped_to_np(msg: QuaternionStamped) -> np.ndarray:
    return quaternion_to_np(msg.quaternion)


def transform_to_np(msg):
    p = Pose()
    p.position = msg.translation
    p.orientation = msg.rotation
    return pose_to_np(p)


def transform_stamped_to_np(msg):
    return transform_to_np(msg.transform)


def msg_to_homogeneous_matrix(msg):
    if isinstance(msg, Pose):
        return pose_to_np(msg)
    elif isinstance(msg, PoseStamped):
        return pose_stamped_to_np(msg)
    elif isinstance(msg, Transform):
        return transform_to_np(msg)
    elif isinstance(msg, TransformStamped):
        return transform_stamped_to_np(msg)
    elif isinstance(msg, Point):
        return point_to_np(msg)
    elif isinstance(msg, PointStamped):
        return point_stamped_to_np(msg)
    elif isinstance(msg, Vector3):
        return vector_to_np(msg)
    elif isinstance(msg, Vector3Stamped):
        return vector_stamped_to_np(msg)
    elif isinstance(msg, Quaternion):
        return quaternion_to_np(msg)
    elif isinstance(msg, QuaternionStamped):
        return quaternion_stamped_to_np(msg)
    else:
        raise TypeError("Invalid type for conversion to SE(3)")


def point_to_np(msg):
    """
    :type msg: Point
    :return:
    """
    return np.array([msg.x, msg.y, msg.z, 1])


def point_stamped_to_np(msg):
    """
    :type msg: PointStamped
    :return:
    """
    return point_to_np(msg.point)


def vector_to_np(msg):
    """
    :type msg: Vector3
    :return:
    """
    return np.array([msg.x, msg.y, msg.z, 0])


def vector_stamped_to_np(msg):
    return vector_to_np(msg.vector)


def publish_frame_marker(pose_stamped, id_=1, length=0.1):
    """
    :type pose_stamped: PoseStamped
    :type id_: int
    """
    kdl_pose = msg_to_kdl(pose_stamped.pose.orientation)
    ma = MarkerArray()
    x = Marker()
    x.action = x.ADD
    x.ns = 'debug'
    x.id = id_
    x.type = x.CUBE
    x.header.frame_id = pose_stamped.header.frame_id
    x.pose.position = copy(pose_stamped.pose.position)
    x.pose.orientation = pose_stamped.pose.orientation

    v = PyKDL.Vector(length / 2., 0, 0)
    v = kdl_pose * v
    x.pose.position.x += v[0]
    x.pose.position.y += v[1]
    x.pose.position.z += v[2]

    x.color = ColorRGBA(1, 0, 0, 1)
    x.scale.x = length
    x.scale.y = length / 10.
    x.scale.z = length / 10.
    ma.markers.append(x)
    y = Marker()
    y.action = y.ADD
    y.ns = 'debug'
    y.id = id_ + 1
    y.type = y.CUBE
    y.header.frame_id = pose_stamped.header.frame_id
    y.pose.position = copy(pose_stamped.pose.position)
    y.pose.orientation = pose_stamped.pose.orientation

    v = PyKDL.Vector(0, length / 2., 0)
    v = kdl_pose * v
    y.pose.position.x += v[0]
    y.pose.position.y += v[1]
    y.pose.position.z += v[2]

    y.color = ColorRGBA(0, 1, 0, 1)
    y.scale.x = length / 10.
    y.scale.y = length
    y.scale.z = length / 10.
    ma.markers.append(y)
    z = Marker()
    z.action = z.ADD
    z.ns = 'debug'
    z.id = id_ + 2
    z.type = z.CUBE
    z.header.frame_id = pose_stamped.header.frame_id
    z.pose.position = copy(pose_stamped.pose.position)
    z.pose.orientation = pose_stamped.pose.orientation

    v = PyKDL.Vector(0, 0, length / 2.)
    v = kdl_pose * v
    z.pose.position.x += v[0]
    z.pose.position.y += v[1]
    z.pose.position.z += v[2]

    z.color = ColorRGBA(0, 0, 1, 1)
    z.scale.x = length / 10.
    z.scale.y = length / 10.
    z.scale.z = length
    ma.markers.append(z)

    pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=1)
    while pub.get_num_connections() < 1:
        # wait for a connection to publisher
        # you can do whatever you like here or simply do nothing
        pass

    pub.publish(ma)


@profile
def normalize_quaternion_msg(quaternion):
    q = Quaternion()
    rotation = np.array([quaternion.x,
                         quaternion.y,
                         quaternion.z,
                         quaternion.w])
    normalized_rotation = rotation / np.linalg.norm(rotation)
    q.x = normalized_rotation[0]
    q.y = normalized_rotation[1]
    q.z = normalized_rotation[2]
    q.w = normalized_rotation[3]
    return q


def homo_matrix_to_pose(m):
    return kdl_to_pose(np_to_kdl(m))
