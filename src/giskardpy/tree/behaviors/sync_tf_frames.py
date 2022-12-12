from typing import List, Tuple
import giskardpy.casadi_wrapper as w
from py_trees import Status

from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.math import compare_poses
from giskardpy.utils.tfwrapper import lookup_pose, msg_to_homogeneous_matrix
from giskardpy.utils.utils import catch_and_raise_to_blackboard


class SyncTfFrames(GiskardBehavior):
    @profile
    def __init__(self, name, frames: List[Tuple[str, str]]):
        super().__init__(name)
        self.frames = frames

    @catch_and_raise_to_blackboard
    @profile
    def update(self):
        with self.god_map:
            for parent_link, child_link in self.frames:
                parent_T_child = lookup_pose(parent_link, child_link)
                parent_T_child_old = self.world.compute_fk_pose(parent_link, child_link)
                try:
                    compare_poses(parent_T_child_old.pose, parent_T_child.pose, decimal=3)
                    # raise Exception()
                except:
                    parent_T_child = msg_to_homogeneous_matrix(parent_T_child)
                    chain = self.world.compute_chain(parent_link, child_link, add_joints=True, add_links=False, add_fixed_joints=True,
                                                     add_non_controlled_joints=True)
                    if len(chain) > 1:
                        raise Exception('todo')
                    joint_name = chain[0]
                    self.world.update_joint_parent_T_child(joint_name, w.TransMatrix(parent_T_child))

        return Status.SUCCESS
