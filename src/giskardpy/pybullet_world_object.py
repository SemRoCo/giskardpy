import hashlib
import os
import pickle
from collections import OrderedDict

import errno
from geometry_msgs.msg import PoseStamped, Point, Quaternion

from giskardpy import MAP
from giskardpy.data_types import SingleJointState
from giskardpy.pybullet_wrapper import load_urdf_string_into_bullet, JointInfo, pybullet_pose_to_msg, \
    deactivate_rendering, activate_rendering, msg_to_pybullet_pose
from giskardpy.utils import resolve_ros_iris_in_urdf
from giskardpy.world_object import WorldObject
import pybullet as p


class PyBulletWorldObject(WorldObject):
    """
    Keeps track of and offers convenience functions for an urdf object in bullet.
    """
    base_link_name = u'base'

    def __init__(self, urdf, controlled_joints=None, base_pose=None, calc_self_collision_matrix=False,
                 path_to_data_folder=u''):
        """
        :type name: str
        :param urdf: Path to URDF file, or content of already loaded URDF file.
        :type urdf: str
        :type base_pose: Transform
        :type calc_self_collision_matrix: bool
        :param path_to_data_folder: where the self collision matrix is stored
        :type path_to_data_folder: str
        """
        self.path_to_data_folder = path_to_data_folder + u'collision_matrix/'
        self.id = None
        super(WorldObject, self).__init__(urdf)
        self.reinitialize()
        self.__sync_with_bullet()
        if base_pose is not None:
            self.set_base_pose(base_pose)
        self.controlled_joints = controlled_joints
        if calc_self_collision_matrix:
            if not self.load_self_collision_matrix():
                self.init_self_collision_matrix()
                self.safe_self_collision_matrix()
        else:
            self.self_collision_matrix = set()

    def __sync_with_bullet(self):
        """
        Syncs joint and link infos with bullet
        """

        self.joint_id_map = {}
        self.link_name_to_id = {}
        self.link_id_to_name = {}
        self.joint_name_to_info = OrderedDict()
        self.joint_id_to_info = OrderedDict()
        self.joint_name_to_info[self.base_link_name] = JointInfo(*([-1, self.base_link_name] + [None] * 10 +
                                                                   [self.base_link_name] + [None] * 4))
        self.joint_id_to_info[-1] = JointInfo(*([-1, self.base_link_name] + [None] * 10 +
                                                [self.base_link_name] + [None] * 4))
        self.link_id_to_name[-1] = self.base_link_name
        self.link_name_to_id[self.base_link_name] = -1
        for joint_index in range(p.getNumJoints(self.id)):
            joint_info = JointInfo(*p.getJointInfo(self.id, joint_index))
            self.joint_name_to_info[joint_info.joint_name] = joint_info
            self.joint_id_to_info[joint_info.joint_index] = joint_info
            self.joint_id_map[joint_index] = joint_info.joint_name
            self.joint_id_map[joint_info.joint_name] = joint_index
            self.link_name_to_id[joint_info.link_name] = joint_index
            self.link_id_to_name[joint_index] = joint_info.link_name
        self.link_name_to_id[self.get_root()] = -1
        self.link_id_to_name[-1] = self.get_root()

    def reinitialize(self):
        super(PyBulletWorldObject, self).reinitialize()
        deactivate_rendering()
        joint_state = None
        base_pose = None
        if self.id is not None:
            joint_state = self.get_joint_state()
            base_pose = self.get_base_pose()
            self.suicide()
        self.id = load_urdf_string_into_bullet(self.get_urdf(), base_pose)
        self.__sync_with_bullet()
        if joint_state is not None:
            joint_state = {k: v for k, v in joint_state.items() if k in self.get_joint_names()}
            self.set_joint_state(joint_state)
        activate_rendering()

    def suicide(self):
        p.removeBody(self.id)

    def __del__(self):
        self.suicide()

    def set_joint_state(self, joint_state):
        """

        :param joint_state:
        :type joint_state: dict
        :return:
        """
        for joint_name, singe_joint_state in joint_state.items():
            p.resetJointState(self.id, self.joint_name_to_info[joint_name].joint_index, singe_joint_state.position)

    def set_base_pose(self, pose):
        """
        Set base pose in bullet world frame.
        :param position:
        :type position: list
        :param orientation:
        :type orientation: list
        """
        position, orientation = msg_to_pybullet_pose(pose)
        p.resetBasePositionAndOrientation(self.id, position, orientation)

    def get_base_pose(self):
        """
        Retrieves the current base pose of the robot in the PyBullet world.
        :return: Base pose of the robot in the world.
        :rtype: Transform
        """
        return pybullet_pose_to_msg(p.getBasePositionAndOrientation(self.id))

    def get_joint_state(self):
        mjs = {}
        for joint_info in self.joint_name_to_info.values():
            # FIXME? why no continuous?
            if joint_info.joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                sjs = SingleJointState()
                sjs.name = joint_info.joint_name
                sjs.position = p.getJointState(self.id, joint_info.joint_index)[0]
                mjs[sjs.name] = sjs
        return mjs

    def __get_pybullet_link_id(self, link_name):
        """
        :type link_name: str
        :rtype: int
        """
        return self.link_name_to_id[link_name]

    def in_collision(self, link_a, link_b, distance):
        link_id_a = self.__get_pybullet_link_id(link_a)
        link_id_b = self.__get_pybullet_link_id(link_b)
        return len(p.getClosestPoints(self.id, self.id, distance, link_id_a, link_id_b)) > 0
