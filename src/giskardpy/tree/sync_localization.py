import rospy
from py_trees import Status
from tf2_py import LookupException

from giskardpy.model.world import SubWorldTree
from giskardpy.tree.plugin import GiskardBehavior
from giskardpy.utils.tfwrapper import lookup_pose, msg_to_homogeneous_matrix


class SyncLocalization(GiskardBehavior):
    def __init__(self, name, group_name, tf_root_link_name=None):
        super(SyncLocalization, self).__init__(name)
        self.map_frame = self.world.root_link_name
        self.group_name = group_name
        self.group = self.world.groups[self.group_name]  # type: SubWorldTree
        if tf_root_link_name is None:
            self.tf_root_link_name = self.group.root_link_name
        else:
            self.tf_root_link_name = tf_root_link_name

    def update(self):
        try:
            map_T_base = lookup_pose(self.map_frame, self.tf_root_link_name)
        except LookupException as e:
            rospy.logwarn(e)
            return Status.FAILURE
        map_T_base = msg_to_homogeneous_matrix(map_T_base)
        self.world.update_joint_parent_T_child(self.group.attachment_joint_name, map_T_base)

        return Status.SUCCESS
