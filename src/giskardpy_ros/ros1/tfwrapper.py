from typing import Optional, overload, List

import genpy
import numpy as np
import rospy
import yaml
from geometry_msgs.msg import PoseStamped, Vector3Stamped, PointStamped, TransformStamped, Pose, Quaternion, Point, \
    Vector3, QuaternionStamped, Transform, TwistStamped
from line_profiler import profile
from tf2_geometry_msgs import do_transform_pose, do_transform_vector3, do_transform_point
from tf2_kdl import do_transform_twist
from tf2_py import InvalidArgumentException
from tf2_ros import Buffer, TransformListener

from giskardpy.data_types.data_types import PrefixName
from giskardpy.middleware import get_middleware
from giskardpy.utils.decorators import memoize
from giskardpy.utils.math import rotation_matrix_from_quaternion

tfBuffer: Buffer = None
tf_listener: TransformListener = None


def init(tf_buffer_size: float = 15) -> None:
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
        get_middleware().logwarn(e)


def get_tf_buffer() -> Buffer:
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


def get_full_frame_names(frame_name: str) -> List[str]:
    """
    Search for namespaced frames that include frame_name.
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


def wait_for_transform(target_frame: str, source_frame: str, time: rospy.Time, timeout: float) -> bool:
    tfBuffer = get_tf_buffer()
    return tfBuffer.can_transform(target_frame, source_frame, time, timeout)


@overload
def transform_msg(target_frame: str, msg: PoseStamped, timeout: float = 5) -> PoseStamped:
    pass


@overload
def transform_msg(target_frame: str, msg: PointStamped, timeout: float = 5) -> PointStamped:
    pass


@overload
def transform_msg(target_frame: str, msg: QuaternionStamped, timeout: float = 5) -> QuaternionStamped:
    pass


@overload
def transform_msg(target_frame: str, msg: Vector3Stamped, timeout: float = 5) -> Vector3Stamped:
    pass


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


def transform_pose(target_frame: str, pose: PoseStamped, timeout: float = 5.0) -> PoseStamped:
    """
    Transforms a pose stamped into a different target frame.
    :return: Transformed pose of None on loop failure
    """
    transform = lookup_transform(target_frame, pose.header.frame_id, pose.header.stamp, timeout)
    new_pose = do_transform_pose(pose, transform)
    return new_pose


def lookup_transform(target_frame: str, source_frame: str, time: Optional[rospy.Time] = None, timeout: float = 5.0) \
        -> TransformStamped:
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


def transform_vector(target_frame: str, vector: Vector3Stamped, timeout: float = 5) -> Vector3Stamped:
    """
    Transforms a pose stamped into a different target frame.
    :type target_frame: Union[str, unicode]
    :return: Transformed pose of None on loop failure
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


def transform_point(target_frame: str, point: PointStamped, timeout: float = 5) -> PointStamped:
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


def lookup_pose(target_frame: str, source_frame: str, time: Optional[rospy.Time] = None) -> PoseStamped:
    """
    :return: target_frame <- source_frame
    """
    p = PoseStamped()
    p.header.frame_id = str(source_frame)
    if time is not None:
        p.header.stamp = time
    p.pose.orientation.w = 1.0
    return transform_pose(target_frame, p)


def lookup_point(target_frame: str, source_frame: str, time: Optional[rospy.Time] = None) -> PointStamped:
    """
    :return: target_frame <- source_frame
    """
    t = lookup_transform(target_frame, source_frame, time)
    p = PointStamped()
    p.header.frame_id = t.header.frame_id
    p.point.x = t.transform.translation.x
    p.point.y = t.transform.translation.y
    p.point.z = t.transform.translation.z
    return p


@profile
def normalize_quaternion_msg(quaternion: Quaternion) -> Quaternion:
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
