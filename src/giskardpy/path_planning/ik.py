from copy import deepcopy

import rospy

from giskardpy.data_types import PrefixName
from giskardpy.utils.kdl_parser import KDL
from giskardpy.utils.tfwrapper import pose_to_kdl, pose_to_list, lookup_pose
import giskardpy.model.pybullet_wrapper as pbw


class IK(object):

    def __init__(self, root_link, tip_link):
        self.root_link = root_link
        self.tip_link = tip_link

    def get_ik(self, old_js, pose):
        raise Exception('Implement me.')

    def clear(self):
        pass


class KDLIK(IK):

    def __init__(self, root_link, tip_link, static_joints=None, robot_description='robot_description'):
        IK.__init__(self, root_link, tip_link)
        self.robot_description = rospy.get_param(robot_description)
        self._kdl_robot = None
        self.robot_kdl_tree = None
        self.static_joints = static_joints
        self.setup()

    def setup(self):
        self.robot_kdl_tree = KDL(self.robot_description)
        self._kdl_robot = self.robot_kdl_tree.get_robot(self.root_link, self.tip_link, static_joints=self.static_joints)

    def get_ik(self, js, pose):
        new_js = deepcopy(js)
        js_dict_position = {}
        for k, v in js.items():
            js_dict_position[str(k)] = v.position
        joint_array = self._kdl_robot.ik(js_dict_position, pose_to_kdl(pose))
        for i, joint_name in enumerate(self._kdl_robot.joints):
            new_js[PrefixName(joint_name, None)].position = joint_array[i]
        return new_js


class PyBulletIK(IK):

    def __init__(self, root_link, tip_link, static_joints=None, robot_description='robot_description'):
        IK.__init__(self, root_link, tip_link)
        self.robot_description = rospy.get_param(robot_description)
        self.pybullet_joints = list()
        self.robot_id = None
        self.pybullet_tip_link_id = None
        self.once = False
        self.static_joints = static_joints
        self.joint_lowers = list()
        self.joint_uppers = list()
        self.setup()

    def setup(self):
        for i in range(0, 100):
            if not pbw.p.isConnected(physicsClientId=i):
                self.client_id = i
                break
        pbw.start_pybullet(False, client_id=self.client_id)
        pos, q = pose_to_list(lookup_pose('map', self.root_link).pose)
        self.robot_id = pbw.load_urdf_string_into_bullet(self.robot_description, position=pos,
                                                         orientation=q, client_id=self.client_id)
        for i in range(0, pbw.p.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            j = pbw.p.getJointInfo(self.robot_id, i, physicsClientId=self.client_id)
            if j[2] != pbw.p.JOINT_FIXED:
                joint_name = j[1].decode('UTF-8')
                if self.static_joints and joint_name not in self.static_joints:
                    self.joint_lowers.append(j[8])
                    self.joint_uppers.append(j[9])
                else:
                    self.joint_lowers.append(0)
                    self.joint_uppers.append(0)
                self.pybullet_joints.append(joint_name)
            if j[12].decode('UTF-8') == self.tip_link:
                self.pybullet_tip_link_id = j[0]

    def get_ik(self, js, pose):
        if not self.once:
            self.update_pybullet(js)
            self.once = True
        new_js = deepcopy(js)
        pose = pose_to_list(pose)
        # rospy.logerr(pose[1])
        state_ik = pbw.p.calculateInverseKinematics(self.robot_id, self.pybullet_tip_link_id,
                                                    pose[0], pose[1], self.joint_lowers, self.joint_uppers,
                                                    physicsClientId=self.client_id)
        for joint_name, joint_state in zip(self.pybullet_joints, state_ik):
            new_js[PrefixName(joint_name, None)].position = joint_state
        return new_js

    def update_pybullet(self, js):
        for joint_id in range(0, pbw.p.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            joint_name = pbw.p.getJointInfo(self.robot_id, joint_id, physicsClientId=self.client_id)[1].decode()
            joint_state = js[PrefixName(joint_name, None)].position
            pbw.p.resetJointState(self.robot_id, joint_id, joint_state, physicsClientId=self.client_id)

    def clear(self):
        pbw.stop_pybullet(client_id=self.client_id)