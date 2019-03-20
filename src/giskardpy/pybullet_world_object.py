from collections import OrderedDict
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Pose
from giskardpy.pybullet_wrapper import load_urdf_string_into_bullet, JointInfo, pybullet_pose_to_msg, \
    deactivate_rendering, activate_rendering, msg_to_pybullet_pose
from giskardpy.world_object import WorldObject
import pybullet as p


class PyBulletWorldObject(WorldObject):
    """
    Keeps track of and offers convenience functions for an urdfs object in bullet.
    """
    base_link_name = u'base'

    def __init__(self, urdf, base_pose=None, controlled_joints=None, path_to_data_folder=u'', *args, **kwargs):
        """
        :type name: str
        :param urdf: Path to URDF file, or content of already loaded URDF file.
        :type urdf: str
        :type base_pose: Transform
        :type calc_self_collision_matrix: bool
        :param path_to_data_folder: where the self collision matrix is stored
        :type path_to_data_folder: str
        """
        self._pybullet_id = None
        super(PyBulletWorldObject, self).__init__(urdf,
                                                  base_pose=base_pose,
                                                  controlled_joints=controlled_joints,
                                                  path_to_data_folder=path_to_data_folder,
                                                  *args, **kwargs)
        self.reinitialize()
        if base_pose is None:
            p = Pose()
            p.orientation.w = 1
            self.base_pose = p
        self.self_collision_matrix = set()

    @WorldObject.joint_state.setter
    def joint_state(self, value):
        """
        :param joint_state:
        :type joint_state: dict
        :return:
        """
        self._js = value
        for joint_name, singe_joint_state in value.items():
            p.resetJointState(self._pybullet_id, self.joint_name_to_info[joint_name].joint_index, singe_joint_state.position)

    @WorldObject.base_pose.setter
    def base_pose(self, value):
        if self._pybullet_id is not None:
            self._base_pose = value
            WorldObject.base_pose.fset(self, value)
            position, orientation = msg_to_pybullet_pose(value)
            p.resetBasePositionAndOrientation(self._pybullet_id, position, orientation)

    def get_pybullet_id(self):
        return self._pybullet_id

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
        for joint_index in range(p.getNumJoints(self._pybullet_id)):
            joint_info = JointInfo(*p.getJointInfo(self._pybullet_id, joint_index))
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
        if self._pybullet_id is not None:
            joint_state = self.joint_state
            base_pose = self.base_pose
            self.suicide()
        self._pybullet_id = load_urdf_string_into_bullet(self.get_urdf(), base_pose)
        self.__sync_with_bullet()
        if joint_state is not None:
            joint_state = {k: v for k, v in joint_state.items() if k in self.get_joint_names()}
            self.joint_state = joint_state
        activate_rendering()

    def suicide(self):
        p.removeBody(self._pybullet_id)
        print(u'removed {} from pybullet'.format(self.get_name()))

    def __del__(self):
        self.suicide()

    def get_base_pose(self):
        """
        Retrieves the current base pose of the robot in the PyBullet world.
        :return: Base pose of the robot in the world.
        :rtype: Transform
        """
        return pybullet_pose_to_msg(p.getBasePositionAndOrientation(self._pybullet_id))

    def get_pybullet_link_id(self, link_name):
        """
        :type link_name: str
        :rtype: int
        """
        return self.link_name_to_id[link_name]

    def in_collision(self, link_a, link_b, distance):
        link_id_a = self.get_pybullet_link_id(link_a)
        link_id_b = self.get_pybullet_link_id(link_b)
        return len(p.getClosestPoints(self._pybullet_id, self._pybullet_id, distance, link_id_a, link_id_b)) > 0
