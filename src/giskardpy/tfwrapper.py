from copy import copy

import PyKDL
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Vector3Stamped, PointStamped, TransformStamped, Pose, Quaternion, Point, \
    Vector3, Twist, TwistStamped, QuaternionStamped
from std_msgs.msg import ColorRGBA
from tf.transformations import quaternion_from_matrix, quaternion_about_axis
from tf2_geometry_msgs import do_transform_pose, do_transform_vector3, do_transform_point
from tf2_kdl import transform_to_kdl
from tf2_py._tf2 import ExtrapolationException
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import MarkerArray, Marker

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
        raise TypeError(u'can\'t convert {} to kdl'.format(type(msg)))

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

def kdl_to_pose_stamped(frame, frame_id):
    """
    :type frame: PyKDL.Frame
    :rtype: Pose
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
        return np.array([[kdl_thing.M[0,0], kdl_thing.M[0,1], kdl_thing.M[0,2], kdl_thing.p[0]],
                         [kdl_thing.M[1,0], kdl_thing.M[1,1], kdl_thing.M[1,2], kdl_thing.p[1]],
                         [kdl_thing.M[2,0], kdl_thing.M[2,1], kdl_thing.M[2,2], kdl_thing.p[2]],
                         [0, 0, 0, 1]])
    if isinstance(kdl_thing, PyKDL.Rotation):
        return np.array([[kdl_thing[0,0], kdl_thing[0,1], kdl_thing[0,2], 0],
                         [kdl_thing[1,0], kdl_thing[1,1], kdl_thing[1,2], 0],
                         [kdl_thing[2,0], kdl_thing[2,1], kdl_thing[2,2], 0],
                         [0, 0, 0, 1]])


def angle_between_vector(v1, v2):
    if isinstance(v1, PyKDL.Vector):
        v1 = kdl_to_np(v1)
    if isinstance(v2, PyKDL.Vector):
        v2 = kdl_to_np(v2)
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def np_vector(x, y, z):
    return np.array([x, y, z, 0])


def np_point(x, y, z):
    return np.array([x, y, z, 1])

### Code copied from user jarvisschultz from ROS answers
#   https://answers.ros.org/question/332407/transformstamped-to-transformation-matrix-python/
def pose_to_pq(msg):
    """Convert a C{geometry_msgs/Pose} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    p = np.array([msg.position.x, msg.position.y, msg.position.z])
    q = np.array([msg.orientation.x, msg.orientation.y,
                  msg.orientation.z, msg.orientation.w])
    return p, q


def pose_stamped_to_pq(msg):
    """Convert a C{geometry_msgs/PoseStamped} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    return pose_to_pq(msg.pose)


def transform_to_pq(msg):
    """Convert a C{geometry_msgs/Transform} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    p = np.array([msg.translation.x, msg.translation.y, msg.translation.z])
    q = np.array([msg.rotation.x, msg.rotation.y,
                  msg.rotation.z, msg.rotation.w])
    return p, q


def transform_stamped_to_pq(msg):
    """Convert a C{geometry_msgs/TransformStamped} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    return transform_to_pq(msg.transform)


def msg_to_se3(msg):
    """Conversion from geometric ROS messages into SE(3)

    @param msg: Message to transform. Acceptable types - C{geometry_msgs/Pose}, C{geometry_msgs/PoseStamped},
    C{geometry_msgs/Transform}, or C{geometry_msgs/TransformStamped}
    @return: a 4x4 SE(3) matrix as a numpy array
    @note: Throws TypeError if we receive an incorrect type.
    """
    if isinstance(msg, Pose):
        p, q = pose_to_pq(msg)
    elif isinstance(msg, PoseStamped):
        p, q = pose_stamped_to_pq(msg)
    elif isinstance(msg, Transform):
        p, q = transform_to_pq(msg)
    elif isinstance(msg, TransformStamped):
        p, q = transform_stamped_to_pq(msg)
    else:
        raise TypeError("Invalid type for conversion to SE(3)")
    norm = np.linalg.norm(q)
    if np.abs(norm - 1.0) > 1e-3:
        raise ValueError(
            "Received un-normalized quaternion (q = {0:s} ||q|| = {1:3.6f})".format(
                str(q), np.linalg.norm(q)))
    elif np.abs(norm - 1.0) > 1e-6:
        q = q / norm
    g = tr.quaternion_matrix(q)
    g[0:3, -1] = p
    return g
### end of copied code

def publish_frame_marker(pose_stamped, id_=1, length=0.1):
    """
    :type pose_stamped: PoseStamped
    :type id_: int
    """
    kdl_pose = msg_to_kdl(pose_stamped.pose.orientation)
    ma = MarkerArray()
    x = Marker()
    x.action = x.ADD
    x.ns = u'debug'
    x.id = id_
    x.type = x.CUBE
    x.header.frame_id = pose_stamped.header.frame_id
    x.pose.position = copy(pose_stamped.pose.position)
    x.pose.orientation = pose_stamped.pose.orientation

    v = PyKDL.Vector(length/2.,0,0)
    v = kdl_pose * v
    x.pose.position.x += v[0]
    x.pose.position.y += v[1]
    x.pose.position.z += v[2]

    x.color = ColorRGBA(1,0,0,1)
    x.scale.x = length
    x.scale.y = length/10.
    x.scale.z = length/10.
    ma.markers.append(x)
    y = Marker()
    y.action = y.ADD
    y.ns = u'debug'
    y.id = id_+1
    y.type = y.CUBE
    y.header.frame_id = pose_stamped.header.frame_id
    y.pose.position = copy(pose_stamped.pose.position)
    y.pose.orientation = pose_stamped.pose.orientation

    v = PyKDL.Vector(0, length / 2., 0)
    v = kdl_pose * v
    y.pose.position.x += v[0]
    y.pose.position.y += v[1]
    y.pose.position.z += v[2]

    y.color = ColorRGBA(0,1,0,1)
    y.scale.x = length/10.
    y.scale.y = length
    y.scale.z = length/10.
    ma.markers.append(y)
    z = Marker()
    z.action = z.ADD
    z.ns = u'debug'
    z.id = id_+2
    z.type = z.CUBE
    z.header.frame_id = pose_stamped.header.frame_id
    z.pose.position = copy(pose_stamped.pose.position)
    z.pose.orientation = pose_stamped.pose.orientation

    v = PyKDL.Vector(0, 0, length / 2.)
    v = kdl_pose * v
    z.pose.position.x += v[0]
    z.pose.position.y += v[1]
    z.pose.position.z += v[2]

    z.color = ColorRGBA(0,0,1,1)
    z.scale.x = length/10.
    z.scale.y = length/10.
    z.scale.z = length
    ma.markers.append(z)

    pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=1)
    while pub.get_num_connections() < 1:
        # wait for a connection to publisher
        # you can do whatever you like here or simply do nothing
        pass

    pub.publish(ma)

if __name__ == u'__main__':
    rospy.init_node(u'tf_wrapper_debug')
    p = PoseStamped()
    p.header.frame_id = u'map'
    p.pose.position.x = 1
    p.pose.orientation = Quaternion(*quaternion_about_axis(np.pi/2, [0,1,0]))
    publish_frame_marker(p)
