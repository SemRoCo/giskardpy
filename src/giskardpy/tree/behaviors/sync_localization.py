import numpy as np
import rospy
from py_trees import Status
from tf.transformations import rotation_from_matrix
from tf2_py import LookupException

from giskardpy.data_types import PrefixName
from giskardpy.model.world import SubWorldTree
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.tfwrapper import lookup_pose, msg_to_homogeneous_matrix


class SyncLocalization(GiskardBehavior):
    def __init__(self, name, group_name, namespace, tf_root_link_name=None):
        super(SyncLocalization, self).__init__(name)
        self.map_frame = self.world.root_link_name
        self.group_name = group_name
        self.tf_prefix = namespace
        self.group = self.world.groups[self.group_name]  # type: SubWorldTree
        if tf_root_link_name is None:
            root_link_name = self.group.root_link_name
            self.tf_root_link_name = PrefixName(root_link_name.short_name, self.tf_prefix)
        else:
            self.tf_root_link_name = tf_root_link_name
        self.last_position = np.zeros(3)
        self.last_rotation = np.eye(4)

    def should_update(self, new_map_T_base):
        rotation = new_map_T_base.copy()
        rotation[3,:3] = 0
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
            map_T_base = lookup_pose(self.map_frame, self.tf_root_link_name)
        except LookupException as e:
            rospy.logwarn(e)
            return Status.FAILURE
        map_T_base = msg_to_homogeneous_matrix(map_T_base)
        if self.should_update(map_T_base):
            self.world.update_joint_parent_T_child(self.group.attachment_joint_name, map_T_base)

        return Status.SUCCESS
