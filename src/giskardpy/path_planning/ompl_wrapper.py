import threading
from copy import deepcopy

from geometry_msgs.msg import Pose
from ompl import base as ob

import numpy as np
import rospy
from nav_msgs.srv import GetMap
from tf.transformations import quaternion_from_euler


def is_3D(space):
    return type(space) == type(ob.SE3StateSpace())


def ompl_states_matrix_to_np(str: str, line_sep='\n', float_sep=' '):
    states_strings = str.split(line_sep)
    while '' in states_strings:
        states_strings.remove('')
    return np.array(list(map(lambda x: np.fromstring(x, dtype=float, sep=float_sep), states_strings)))


def ompl_state_to_pose(state, is_3D):
    if is_3D:
        pose = ompl_se3_state_to_pose(state)
    else:
        pose = ompl_se2_state_to_pose(state)
    return pose


def ompl_se3_state_to_pose(state):
    pose = Pose()
    pose.position.x = state.getX()
    pose.position.y = state.getY()
    pose.position.z = state.getZ()
    pose.orientation.x = state.rotation().x
    pose.orientation.y = state.rotation().y
    pose.orientation.z = state.rotation().z
    pose.orientation.w = state.rotation().w
    return pose


def pose_to_ompl_state(space, pose, is_3D):
    if is_3D:
        pose = pose_to_ompl_se3(space, pose)
    else:
        pose = pose_to_ompl_se2(space, pose)
    return pose


def pose_to_ompl_se3(space, pose):
    state = ob.State(space)
    state().setX(pose.position.x)
    state().setY(pose.position.y)
    state().setZ(pose.position.z)
    state().rotation().x = pose.orientation.x
    state().rotation().y = pose.orientation.y
    state().rotation().z = pose.orientation.z
    state().rotation().w = pose.orientation.w
    return state


def pose_to_ompl_se2(space, pose):
    state = ob.State(space)
    state().setX(pose.position.x)
    state().setY(pose.position.y)
    state().setYaw(pose.orientation.z)
    return state


def copy_pose_to_ompl_se3(state, pose):
    state().setX(pose.position.x)
    state().setY(pose.position.y)
    state().setZ(pose.position.z)
    state().rotation().x = pose.orientation.x
    state().rotation().y = pose.orientation.y
    state().rotation().z = pose.orientation.z
    state().rotation().w = pose.orientation.w
    return state


def ompl_se2_state_to_pose(state):
    pose = Pose()
    pose.position.x = state.getX()
    pose.position.y = state.getY()
    pose.position.z = 0
    yaw = state.getYaw()
    rot = quaternion_from_euler(0, 0, yaw)
    pose.orientation.x = rot[0]
    pose.orientation.y = rot[1]
    pose.orientation.z = rot[2]
    pose.orientation.w = rot[3]
    return pose


class OMPLStateValidator(ob.StateValidityChecker):

    def __init__(self, si, is_3D, collision_checker):
        ob.StateValidityChecker.__init__(self, si)
        self.lock = threading.Lock()
        self.is_3D = is_3D
        self.collision_checker = collision_checker

    def isValid(self, state):
        with self.lock:
            return self.collision_checker.isValid(ompl_state_to_pose(state, self.is_3D))


class OMPLMotionValidator(ob.MotionValidator):
    """
    This class ensures that every Planner in OMPL makes the same assumption for
    planning edges in the resulting path. The state validator can be if needed
    deactivated by passing ignore_state_validator as True. The real motion checking must be implemented
    by overwriting the function ompl_check_motion.
    """

    def __init__(self, si, is_3D, motion_validator):
        ob.MotionValidator.__init__(self, si)
        self.si = si
        self.lock = threading.Lock()
        self.is_3D = is_3D
        self.motion_validator = motion_validator

    def checkMotion(self, *args):
        with self.lock:
            if len(args) == 2:
                s1, s2 = args
            elif len(args) == 3:
                s1, s2, last_valid = args
            else:
                raise Exception('Invalid input arguments.')
            if self.is_3D:
                s1_pose = ompl_se3_state_to_pose(s1)
                s2_pose = ompl_se3_state_to_pose(s2)
            else:
                s1_pose = ompl_se2_state_to_pose(s1)
                s2_pose = ompl_se2_state_to_pose(s2)
            if len(args) == 2:
                return self.motion_validator.checkMotion(s1_pose, s2_pose)
            elif len(args) == 3:
                ret, last_valid_pose, time = self.motion_validator.checkMotionTimed(s1_pose, s2_pose)
                if not ret:
                    valid_state = pose_to_ompl_state(self.si.getStateSpace(), last_valid_pose, self.is_3D)
                    last_valid = (valid_state, time)
                return ret
            else:
                raise Exception('Invalid input arguments.')