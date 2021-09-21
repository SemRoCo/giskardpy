import os
import random
import string
from collections import namedtuple

import numpy as np
import pybullet as p
from pybullet import resetJointState, getNumJoints, resetBasePositionAndOrientation, getBasePositionAndOrientation, \
    removeBody, getClosestPoints
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion

import giskardpy
from giskardpy import DEBUG, MAP
from giskardpy.exceptions import DuplicateNameException
from giskardpy.model.urdf_object import robot_name_from_urdf_string
from giskardpy.utils import logging
from giskardpy.utils.tfwrapper import np_to_pose
from giskardpy.utils.utils import write_to_tmp, NullContextManager, suppress_stdout, resolve_ros_iris_in_urdf

JointInfo = namedtuple(u'JointInfo', [u'joint_index', u'joint_name', u'joint_type', u'q_index', u'u_index', u'flags',
                                      u'joint_damping', u'joint_friction', u'joint_lower_limit', u'joint_upper_limit',
                                      u'joint_max_force', u'joint_max_velocity', u'link_name', u'joint_axis',
                                      u'parent_frame_pos', u'parent_frame_orn', u'parent_index'])

ContactInfo = namedtuple(u'ContactInfo', [u'contact_flag', u'body_unique_id_a', u'body_unique_id_b', u'link_index_a',
                                          u'link_index_b', u'position_on_a', u'position_on_b', u'contact_normal_on_b',
                                          u'contact_distance', u'normal_force', u'lateralFriction1',
                                          u'lateralFrictionDir1',
                                          u'lateralFriction2', u'lateralFrictionDir2'])

render = True

def getJointInfo(pybullet_id, joint_index):
    result = p.getJointInfo(pybullet_id, joint_index)
    result2 = []
    for r in result:
        if isinstance(r, bytes):
            result2.append(r.decode("utf-8"))
        else:
            result2.append(r)
    return result2

def random_string(size=6):
    """
    Creates and returns a random string.
    :param size: Number of characters that the string shall contain.
    :type size: int
    :return: Generated random sequence of chars.
    :rtype: str
    """
    return u''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(size))


@profile
def load_urdf_string_into_bullet(urdf_string, pose=None, position=None, orientation=None):
    """
    Loads a URDF string into the bullet world.
    :param urdf_string: XML string of the URDF to load.
    :type urdf_string: str
    :param pose: Pose at which to load the URDF into the world.
    :type pose: Pose
    :return: internal PyBullet id of the loaded urdfs
    :rtype: intload_urdf_string_into_bullet
    """
    if isinstance(pose, PoseStamped):
        pose = pose.pose
    if isinstance(pose, Pose):
        position = [pose.position.x, pose.position.y, pose.position.z]
        orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    if position is None:
        position = [0,0,0]
    if orientation is None:
        orientation = [0,0,0,1]
    object_name = robot_name_from_urdf_string(urdf_string)
    if object_name in get_body_names():
        raise DuplicateNameException(u'an object with name \'{}\' already exists in pybullet'.format(object_name))
    resolved_urdf = resolve_ros_iris_in_urdf(urdf_string)
    filename = write_to_tmp(u'{}.urdf'.format(random_string()), resolved_urdf)
    with NullContextManager() if giskardpy.PRINT_LEVEL == DEBUG else suppress_stdout():
        id = p.loadURDF(filename, position, orientation,
                        flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
    os.remove(filename)
    return id


def deactivate_rendering():
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)


def activate_rendering():
    if render:
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)


def stop_pybullet():
    p.disconnect()


def start_pybullet(gui, gravity=0):
    if not p.isConnected():
        if gui:
            # TODO expose opengl2 option for gui?
            server_id = p.connect(p.GUI, options=u'--opengl2')  # or p.DIRECT for non-graphical version
        else:
            server_id = p.connect(p.DIRECT)  # or p.DIRECT for non-graphical version
        p.setGravity(0, 0, gravity)
        return server_id


def create_box(depth, width, height, position, orientation):
    return createCollisionShape(shapeType=p.GEOM_BOX,
                                halfExtents=[depth, width, height],
                                collisionFramePosition=position,
                                collisionFrameOrientation=orientation)


def createCollisionShape(shapeType, radius=None, halfExtents=None, height=None, fileName=None, meshScale=None,
                         planeNormal=None, flags=None, collisionFramePosition=None, collisionFrameOrientation=None,
                         vertices=None, indices=None, heightfieldTextureScaling=None, numHeightfieldRows=None,
                         numHeightfieldColumns=None, replaceHeightfieldIdIndex=None, physicsClientId=None):
    kwargs = {'shapeType': shapeType}
    if radius is not None:
        kwargs['radius'] = radius
    if height is not None:
        kwargs['height'] = height
    if halfExtents is not None:
        kwargs['halfExtents'] = halfExtents
    if fileName is not None:
        kwargs['fileName'] = fileName
    if meshScale is not None:
        kwargs['meshScale'] = meshScale
    if collisionFramePosition is not None:
        kwargs['collisionFramePosition'] = collisionFramePosition
    if collisionFrameOrientation is not None:
        kwargs['collisionFrameOrientation'] = collisionFrameOrientation
    return p.createCollisionShape(**kwargs)

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
    # return [p.getBodyInfo(p.getBodyUniqueId(i))[1] for i in range(p.getNumBodies())]
    return [p.getBodyInfo(p.getBodyUniqueId(i))[1].decode("utf-8")  for i in range(p.getNumBodies())]


def print_body_names():
    logging.loginfo("".join(get_body_names()))
