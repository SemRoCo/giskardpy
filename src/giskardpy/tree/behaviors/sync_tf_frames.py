from typing import List, Tuple, Dict, Optional
import giskardpy.casadi_wrapper as w
from py_trees import Status

from giskardpy.model.joints import Joint6DOF
from giskardpy.my_types import PrefixName
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.math import compare_poses
from giskardpy.utils.tfwrapper import lookup_pose, msg_to_homogeneous_matrix
from giskardpy.utils.decorators import catch_and_raise_to_blackboard, record_time


class SyncTfFrames(GiskardBehavior):
    joint_map: Dict[PrefixName, Tuple[str, str]]

    @profile
    def __init__(self, name, joint_map: Optional[Dict[PrefixName, Tuple[str, str]]] = None):
        super().__init__(name)
        if joint_map is None:
            self.joint_map = {}
        else:
            self.joint_map = joint_map

    def sync_6dof_joint_with_tf_frame(self, joint_name: PrefixName, tf_parent_frame: str, tf_child_frame: str):
        if joint_name in self.joint_map:
            raise AttributeError(f'Joint \'{joint_name}\' is already being tracking with a tf frame: '
                                 f'\'{self.joint_map[joint_name][0]}\'<-\'{self.joint_map[joint_name][1]}\'')
        joint = self.world.joints[joint_name]
        if not isinstance(joint, Joint6DOF):
            raise AttributeError(f'Can only sync Joint6DOF with tf but \'{joint_name}\' is of type \'{type(joint)}\'.')
        self.joint_map[joint_name] = (tf_parent_frame, tf_child_frame)

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        with self.god_map:
            for joint_name in self.joint_map:
                joint: Joint6DOF = self.world.joints[joint_name]
                parent_T_child = lookup_pose(joint.parent_link_name, joint.child_link_name)
                # parent_T_child_old = self.world.compute_fk_pose(joint.parent_link_name, joint.child_link_name)
                # try:
                #     compare_poses(parent_T_child_old.pose, parent_T_child.pose, decimal=3)
                #     raise Exception()
                # except AssertionError as e:
                joint.update_transform(parent_T_child.pose)

        return Status.SUCCESS
