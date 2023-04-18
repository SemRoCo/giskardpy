from typing import List, Tuple
import giskardpy.casadi_wrapper as w
from py_trees import Status

from giskardpy.model.joints import TFJoint
from giskardpy.my_types import PrefixName
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.math import compare_poses
from giskardpy.utils.tfwrapper import lookup_pose, msg_to_homogeneous_matrix
from giskardpy.utils.decorators import catch_and_raise_to_blackboard, record_time


class SyncTfFrames(GiskardBehavior):
    @profile
    def __init__(self, name, joint_names: List[PrefixName]):
        super().__init__(name)
        self.joint_names = joint_names

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        with self.god_map:
            for joint_name in self.joint_names:
                joint: TFJoint = self.world.joints[joint_name]
                parent_T_child = lookup_pose(joint.parent_link_name, joint.child_link_name)
                parent_T_child_old = self.world.compute_fk_pose(joint.parent_link_name, joint.child_link_name)
                try:
                    compare_poses(parent_T_child_old.pose, parent_T_child.pose, decimal=3)
                    # raise Exception()
                except AssertionError as e:
                    joint.update_transform(parent_T_child.pose)

        return Status.SUCCESS
