from typing import List, Tuple

import numpy as np
import rospy
from py_trees import Status
from tf.transformations import rotation_from_matrix
from tf2_py import LookupException

from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.tfwrapper import lookup_pose, msg_to_homogeneous_matrix


class SyncTfFrames(GiskardBehavior):
    def __init__(self, name, frames: List[Tuple[str, str]]):
        super(SyncTfFrames, self).__init__(name)
        self.map_frame = self.world.root_link_name
        self.last_position = np.zeros(3)
        self.last_rotation = np.eye(4)
        self.frames = frames

    def should_update(self, new_map_T_base):
        rotation = new_map_T_base.copy()
        rotation[3, :3] = 0
        position_diff = self.last_position - new_map_T_base[3, :3]
        angle, _, _ = rotation_from_matrix(np.dot(rotation, self.last_rotation))
        result = np.linalg.norm(position_diff) > 0.01 or angle > 0.02
        if result:
            self.last_position = new_map_T_base[3, :3]
            self.last_rotation = rotation.T
        return result

    @profile
    def update(self):
        try:
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
        except LookupException as e:
            rospy.logwarn(e)
            return Status.FAILURE

        return Status.SUCCESS
