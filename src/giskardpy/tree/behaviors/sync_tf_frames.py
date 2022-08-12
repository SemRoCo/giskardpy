from typing import List, Tuple

import numpy as np
import rospy
from py_trees import Status
from tf.transformations import rotation_from_matrix
from tf2_py import LookupException

from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.tfwrapper import lookup_pose, msg_to_homogeneous_matrix
from giskardpy.utils.utils import catch_and_raise_to_blackboard


class SyncTfFrames(GiskardBehavior):
    def __init__(self, name, frames: List[Tuple[str, str]]):
        super().__init__(name)
        self.frames = frames


    @profile
    @catch_and_raise_to_blackboard
    def update(self):
        with self.god_map:
            for parent_link, child_link in self.frames:
                parent_T_child = lookup_pose(parent_link, child_link)
                parent_T_child = msg_to_homogeneous_matrix(parent_T_child)
                chain = self.world.compute_chain(parent_link, child_link, joints=True, links=False, fixed=True,
                                                 non_controlled=True)
                if len(chain) > 1:
                    raise Exception('todo')
                joint_name = chain[0]
                # if self.should_update(map_T_base):
                self.world.update_joint_parent_T_child(joint_name, parent_T_child)

        return Status.SUCCESS
