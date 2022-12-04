import os
import random
import string
from collections import namedtuple

import pybullet as p
from geometry_msgs.msg import Pose, PoseStamped

import giskardpy
from giskardpy import DEBUG
from giskardpy.exceptions import DuplicateNameException
from giskardpy.model.utils import robot_name_from_urdf_string
from giskardpy.utils import logging
from giskardpy.utils.utils import write_to_tmp, NullContextManager, suppress_stdout, resolve_ros_iris_in_urdf

JointInfo = namedtuple('JointInfo', ['joint_index', 'joint_name', 'joint_type', 'q_index', 'u_index', 'flags',
                                      'joint_damping', 'joint_friction', 'joint_lower_limit', 'joint_upper_limit',
                                      'joint_max_force', 'joint_max_velocity', 'link_name', 'joint_axis',
                                      'parent_frame_pos', 'parent_frame_orn', 'parent_index'])

ContactInfo = namedtuple('ContactInfo', ['contact_flag', 'body_unique_id_a', 'body_unique_id_b', 'link_index_a',
                                          'link_index_b', 'position_on_a', 'position_on_b', 'contact_normal_on_b',
                                          'contact_distance', 'normal_force', 'lateralFriction1',
                                          'lateralFrictionDir1',
                                          'lateralFriction2', 'lateralFrictionDir2'])

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
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(size))


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
        raise DuplicateNameException('an object with name \'{}\' already exists in pybullet'.format(object_name))
    resolved_urdf = resolve_ros_iris_in_urdf(urdf_string)
    filename = write_to_tmp('{}.urdf'.format(random_string()), resolved_urdf)
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
            server_id = p.connect(p.GUI, options='--opengl2')  # or p.DIRECT for non-graphical version
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
