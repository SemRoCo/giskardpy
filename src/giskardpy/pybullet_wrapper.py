import os
import random
import string
from collections import namedtuple

import pybullet as p
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion

import giskardpy
from giskardpy import DEBUG, MAP, logging
from giskardpy.exceptions import DuplicateNameException
from giskardpy.urdf_object import URDFObject
from giskardpy.utils import write_to_tmp, NullContextManager, suppress_stdout, resolve_ros_iris_in_urdf

JointInfo = namedtuple(u'JointInfo', [u'joint_index', u'joint_name', u'joint_type', u'q_index', u'u_index', u'flags',
                                      u'joint_damping', u'joint_friction', u'joint_lower_limit', u'joint_upper_limit',
                                      u'joint_max_force', u'joint_max_velocity', u'link_name', u'joint_axis',
                                      u'parent_frame_pos', u'parent_frame_orn', u'parent_index'])

ContactInfo = namedtuple(u'ContactInfo', [u'contact_flag', u'body_unique_id_a', u'body_unique_id_b', u'link_index_a',
                                          u'link_index_b', u'position_on_a', u'position_on_b', u'contact_normal_on_b',
                                          u'contact_distance', u'normal_force', u'lateralFriction1',
                                          u'lateralFrictionDir1',
                                          u'lateralFriction2', u'lateralFrictionDir2'])


def random_string(size=6):
    """
    Creates and returns a random string.
    :param size: Number of characters that the string shall contain.
    :type size: int
    :return: Generated random sequence of chars.
    :rtype: str
    """
    return u''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(size))


def load_urdf_string_into_bullet(urdf_string, pose=None):
    """
    Loads a URDF string into the bullet world.
    :param urdf_string: XML string of the URDF to load.
    :type urdf_string: str
    :param pose: Pose at which to load the URDF into the world.
    :type pose: Pose
    :return: internal PyBullet id of the loaded urdfs
    :rtype: intload_urdf_string_into_bullet
    """
    if pose is None:
        pose = Pose()
        pose.orientation.w = 1
    if isinstance(pose, PoseStamped):
        pose = pose.pose
    object_name = URDFObject(urdf_string).get_name()
    if object_name in get_body_names():
        raise DuplicateNameException(u'an object with name \'{}\' already exists in pybullet'.format(object_name))
    resolved_urdf = resolve_ros_iris_in_urdf(urdf_string)
    filename = write_to_tmp(u'{}.urdfs'.format(random_string()), resolved_urdf)
    with NullContextManager() if giskardpy.PRINT_LEVEL == DEBUG else suppress_stdout():
        id = p.loadURDF(filename, [pose.position.x, pose.position.y, pose.position.z],
                        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
                        flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
    os.remove(filename)
    return id


def deactivate_rendering():
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)


def activate_rendering():
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)


def stop_pybullet():
    p.disconnect()


def start_pybullet(gui):
    if gui:
        # TODO expose opengl2 option for gui?
        server_id = p.connect(p.GUI, options=u'--opengl2')  # or p.DIRECT for non-graphical version
    else:
        server_id = p.connect(p.DIRECT)  # or p.DIRECT for non-graphical version
    p.setGravity(0, 0, -9.8)
    return server_id


def pybullet_pose_to_msg(pose):
    """
    :type pose: tuple
    :rtype: PoseStamped
    """
    [position, orientation] = pose
    pose = PoseStamped()
    pose.header.frame_id = MAP
    pose.pose.position = Point(*position)
    pose.pose.orientation = Quaternion(*orientation)
    return pose


def msg_to_pybullet_pose(msg):
    """
    :type msg: Pose
    :rtype: tuple
    """
    if isinstance(msg, PoseStamped):
        msg = msg.pose
    position = (msg.position.x,
                msg.position.y,
                msg.position.z)
    orientation = (msg.orientation.x,
                   msg.orientation.y,
                   msg.orientation.z,
                   msg.orientation.w)
    return position, orientation


def clear_pybullet():
    p.resetSimulation()


def get_body_names():
    return [p.getBodyInfo(p.getBodyUniqueId(i))[1] for i in range(p.getNumBodies())]


def print_body_names():
    logging.loginfo("".join(get_body_names()))
